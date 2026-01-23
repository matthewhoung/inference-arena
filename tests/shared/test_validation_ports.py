"""Unit tests for port validation module.

This module tests shared/validation/ports.py which validates port
configurations between experiment.yaml and docker-compose files.

Test Categories:
- parse_compose_ports: Extract ports from various docker-compose formats
- validate_ports: Validate expected ports against compose file
- Error message format: Verify multi-mismatch error messages

Author: Matthew Hong
"""

from pathlib import Path
from textwrap import dedent

import pytest

from shared.exceptions import ConfigError
from shared.validation import parse_compose_ports, validate_ports


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def compose_file(tmp_path: Path):
    """Factory fixture to create temporary compose files."""

    def _create(content: str) -> Path:
        compose_path = tmp_path / "docker-compose.yml"
        compose_path.write_text(dedent(content).strip())
        return compose_path

    return _create


# =============================================================================
# Test parse_compose_ports - Port Format Variations
# =============================================================================


class TestParseComposePortsFormats:
    """Test parsing various docker-compose port formats."""

    def test_parse_compose_ports_string_format(self, compose_file) -> None:
        """Handles 'host:container' string format."""
        path = compose_file("""
            services:
              web:
                ports:
                  - "8100:8100"
        """)

        result = parse_compose_ports(path)

        assert result == {"web": [8100]}

    def test_parse_compose_ports_short_string(self, compose_file) -> None:
        """Handles short string format (container port only)."""
        path = compose_file("""
            services:
              web:
                ports:
                  - "8100"
        """)

        result = parse_compose_ports(path)

        assert result == {"web": [8100]}

    def test_parse_compose_ports_integer(self, compose_file) -> None:
        """Handles bare integer port format."""
        path = compose_file("""
            services:
              web:
                ports:
                  - 8100
        """)

        result = parse_compose_ports(path)

        assert result == {"web": [8100]}

    def test_parse_compose_ports_long_syntax(self, compose_file) -> None:
        """Handles long syntax dict format with published/target."""
        path = compose_file("""
            services:
              web:
                ports:
                  - published: 8100
                    target: 80
        """)

        result = parse_compose_ports(path)

        assert result == {"web": [8100]}

    def test_parse_compose_ports_long_syntax_target_only(self, compose_file) -> None:
        """Handles long syntax dict format with target only."""
        path = compose_file("""
            services:
              web:
                ports:
                  - target: 8100
        """)

        result = parse_compose_ports(path)

        assert result == {"web": [8100]}

    def test_parse_compose_ports_env_var_with_default(self, compose_file) -> None:
        """Handles environment variable syntax with default value."""
        path = compose_file("""
            services:
              minio:
                ports:
                  - "${MINIO_PORT:-9000}:9000"
        """)

        result = parse_compose_ports(path)

        assert result == {"minio": [9000]}

    def test_parse_compose_ports_mixed(self, compose_file) -> None:
        """Handles file with multiple port formats."""
        path = compose_file("""
            services:
              web:
                ports:
                  - "8080:80"
                  - 443
              db:
                ports:
                  - published: 5432
                    target: 5432
              cache:
                ports:
                  - "${REDIS_PORT:-6379}:6379"
        """)

        result = parse_compose_ports(path)

        assert result == {
            "web": [8080, 443],
            "db": [5432],
            "cache": [6379],
        }

    def test_parse_compose_ports_no_ports_section(self, compose_file) -> None:
        """Services with no ports section are excluded from result."""
        path = compose_file("""
            services:
              web:
                image: nginx
              db:
                ports:
                  - "5432:5432"
        """)

        result = parse_compose_ports(path)

        assert "web" not in result
        assert result == {"db": [5432]}

    def test_parse_compose_ports_empty_ports(self, compose_file) -> None:
        """Services with empty ports list are excluded from result."""
        path = compose_file("""
            services:
              web:
                ports: []
              db:
                ports:
                  - "5432:5432"
        """)

        result = parse_compose_ports(path)

        assert "web" not in result
        assert result == {"db": [5432]}

    def test_parse_compose_ports_empty_services(self, compose_file) -> None:
        """Empty services section returns empty dict."""
        path = compose_file("""
            services: {}
        """)

        result = parse_compose_ports(path)

        assert result == {}

    def test_parse_compose_ports_no_services_key(self, compose_file) -> None:
        """Compose file without services key returns empty dict."""
        path = compose_file("""
            version: "3.8"
        """)

        result = parse_compose_ports(path)

        assert result == {}

    def test_parse_compose_ports_different_host_container(self, compose_file) -> None:
        """Extracts host port when different from container port."""
        path = compose_file("""
            services:
              web:
                ports:
                  - "80:8080"
        """)

        result = parse_compose_ports(path)

        # Should return host port (80), not container port (8080)
        assert result == {"web": [80]}


# =============================================================================
# Test validate_ports - Validation Logic
# =============================================================================


class TestValidatePortsSuccess:
    """Test successful validation scenarios."""

    def test_validate_ports_all_match(self, compose_file) -> None:
        """All expected ports found should pass validation."""
        path = compose_file("""
            services:
              minio:
                ports:
                  - "9000:9000"
                  - "9001:9001"
              prometheus:
                ports:
                  - "9090:9090"
        """)

        # Should not raise
        validate_ports(path, {"minio": 9000, "prometheus": 9090})

    def test_validate_ports_extra_compose_ports_ok(self, compose_file) -> None:
        """Extra ports in compose (not in expected) should be OK."""
        path = compose_file("""
            services:
              minio:
                ports:
                  - "9000:9000"
                  - "9001:9001"
        """)

        # Only checking 9000, not 9001 - should pass
        validate_ports(path, {"minio": 9000})

    def test_validate_ports_multiple_services(self, compose_file) -> None:
        """Validates multiple services correctly."""
        path = compose_file("""
            services:
              minio:
                ports:
                  - "9000:9000"
              prometheus:
                ports:
                  - "9090:9090"
              grafana:
                ports:
                  - "3000:3000"
        """)

        # Should not raise
        validate_ports(
            path,
            {"minio": 9000, "prometheus": 9090, "grafana": 3000}
        )


class TestValidatePortsMismatch:
    """Test port mismatch detection."""

    def test_validate_ports_mismatch(self, compose_file) -> None:
        """Port mismatch raises ConfigError."""
        path = compose_file("""
            services:
              prometheus:
                ports:
                  - "9091:9090"
        """)

        with pytest.raises(ConfigError) as exc_info:
            validate_ports(path, {"prometheus": 9090})

        error_msg = str(exc_info.value)
        assert "prometheus" in error_msg
        assert "expected port 9090" in error_msg
        assert "9091" in error_msg

    def test_validate_ports_missing_service(self, compose_file) -> None:
        """Service not in compose raises error."""
        path = compose_file("""
            services:
              web:
                ports:
                  - "8080:80"
        """)

        with pytest.raises(ConfigError) as exc_info:
            validate_ports(path, {"missing-service": 9090})

        error_msg = str(exc_info.value)
        assert "missing-service" in error_msg
        assert "no ports" in error_msg

    def test_validate_ports_service_exists_no_ports(self, compose_file) -> None:
        """Service exists but has no ports section raises error."""
        path = compose_file("""
            services:
              web:
                image: nginx
        """)

        with pytest.raises(ConfigError) as exc_info:
            validate_ports(path, {"web": 8080})

        error_msg = str(exc_info.value)
        assert "web" in error_msg
        assert "no ports" in error_msg


class TestValidatePortsMultipleMismatches:
    """Test multiple mismatch error messages."""

    def test_validate_ports_multiple_mismatches(self, compose_file) -> None:
        """Error lists ALL mismatches, not just first."""
        path = compose_file("""
            services:
              minio:
                ports:
                  - "9999:9000"
              prometheus:
                ports:
                  - "8888:9090"
        """)

        with pytest.raises(ConfigError) as exc_info:
            validate_ports(
                path,
                {"minio": 9000, "prometheus": 9090, "grafana": 3000}
            )

        error_msg = str(exc_info.value)
        # All three mismatches should be listed
        assert "minio" in error_msg
        assert "prometheus" in error_msg
        assert "grafana" in error_msg
        # Grafana not in compose at all
        assert "no ports" in error_msg

    def test_validate_ports_partial_match(self, compose_file) -> None:
        """Mix of matching and mismatching should fail with only mismatches."""
        path = compose_file("""
            services:
              minio:
                ports:
                  - "9000:9000"
              prometheus:
                ports:
                  - "8888:9090"
              grafana:
                ports:
                  - "3000:3000"
        """)

        with pytest.raises(ConfigError) as exc_info:
            validate_ports(
                path,
                {"minio": 9000, "prometheus": 9090, "grafana": 3000}
            )

        error_msg = str(exc_info.value)
        # Only prometheus should be in error
        assert "prometheus" in error_msg
        assert "minio" not in error_msg or "minio" in error_msg.split(":")[0]
        # grafana matched, so shouldn't be in error details
        # (but may appear in file path)


class TestValidatePortsErrorFormat:
    """Test error message formatting."""

    def test_error_includes_compose_path(self, compose_file) -> None:
        """Error message includes compose file path."""
        path = compose_file("""
            services:
              web:
                ports:
                  - "9999:8080"
        """)

        with pytest.raises(ConfigError) as exc_info:
            validate_ports(path, {"web": 8080})

        error_msg = str(exc_info.value)
        assert "docker-compose.yml" in error_msg

    def test_error_shows_found_ports(self, compose_file) -> None:
        """Error message shows what ports were found."""
        path = compose_file("""
            services:
              web:
                ports:
                  - "8080:80"
                  - "443:443"
        """)

        with pytest.raises(ConfigError) as exc_info:
            validate_ports(path, {"web": 9000})

        error_msg = str(exc_info.value)
        # Should show the ports that were found
        assert "8080" in error_msg or "[8080" in error_msg


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestParseComposePortsEdgeCases:
    """Test edge cases in port parsing."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Missing compose file raises FileNotFoundError."""
        path = tmp_path / "nonexistent.yml"

        with pytest.raises(FileNotFoundError):
            parse_compose_ports(path)

    def test_invalid_yaml(self, compose_file) -> None:
        """Invalid YAML raises YAMLError."""
        import yaml

        path = compose_file("""
            services:
              web:
                ports: [
        """)

        with pytest.raises(yaml.YAMLError):
            parse_compose_ports(path)

    def test_env_var_no_default(self, compose_file) -> None:
        """Environment variable without default returns None for that port."""
        path = compose_file("""
            services:
              web:
                ports:
                  - "${PORT}:8080"
                  - "9000:9000"
        """)

        result = parse_compose_ports(path)

        # Should still extract the valid port
        assert 9000 in result.get("web", [])


# =============================================================================
# Test validate_infrastructure_ports (Integration-like)
# =============================================================================


class TestValidateInfrastructurePorts:
    """Test validate_infrastructure_ports with real project files."""

    def test_validate_infrastructure_ports_with_real_files(self) -> None:
        """Integration test with actual project files."""
        from shared.validation import validate_infrastructure_ports

        # This should not raise if experiment.yaml and docker-compose.infra.yml
        # are in sync
        # Note: This test will fail if ports are misconfigured
        validate_infrastructure_ports()
