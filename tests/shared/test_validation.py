"""Unit tests for container validation module.

This module tests shared/validation/containers.py which validates
container existence, running status, and health before operations
that depend on them (e.g., Prometheus queries).

Test Categories:
- Container existence: validate_containers finds missing containers
- Running status: validate_containers detects stopped containers
- Health status: validate_containers checks container health
- Multi-failure: Error messages list ALL failures, not just first
- Edge cases: Empty list, no healthcheck, Docker connection issues

Author: Matthew Hong
"""

from unittest.mock import MagicMock, patch

import pytest

from shared.exceptions import ConfigError
from shared.validation import validate_containers

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_docker_client():
    """Create a mock Docker client."""
    with patch("shared.validation.containers.docker") as mock_docker:
        client = MagicMock()
        mock_docker.from_env.return_value = client
        mock_docker.errors.DockerException = Exception
        yield client


@pytest.fixture
def healthy_container():
    """Create a mock healthy container."""
    container = MagicMock()
    container.status = "running"
    container.attrs = {
        "State": {
            "Health": {
                "Status": "healthy"
            }
        }
    }
    return container


@pytest.fixture
def running_no_health_container():
    """Create a mock running container with no healthcheck."""
    container = MagicMock()
    container.status = "running"
    container.attrs = {
        "State": {}  # No Health key
    }
    return container


@pytest.fixture
def stopped_container():
    """Create a mock stopped container."""
    container = MagicMock()
    container.status = "exited"
    container.attrs = {
        "State": {}
    }
    return container


@pytest.fixture
def unhealthy_container():
    """Create a mock unhealthy container."""
    container = MagicMock()
    container.status = "running"
    container.attrs = {
        "State": {
            "Health": {
                "Status": "unhealthy"
            }
        }
    }
    return container


@pytest.fixture
def starting_container():
    """Create a mock container with health status 'starting'."""
    container = MagicMock()
    container.status = "running"
    container.attrs = {
        "State": {
            "Health": {
                "Status": "starting"
            }
        }
    }
    return container


# =============================================================================
# Success Cases
# =============================================================================


class TestValidateContainersSuccess:
    """Test successful container validation scenarios."""

    def test_validate_containers_all_healthy(
        self, mock_docker_client, healthy_container
    ) -> None:
        """All containers running and healthy should pass validation."""
        mock_docker_client.containers.get.return_value = healthy_container

        # Should not raise
        validate_containers(["prometheus", "cadvisor"])

        # Verify both containers were checked
        assert mock_docker_client.containers.get.call_count == 2

    def test_validate_containers_no_health_check(
        self, mock_docker_client, running_no_health_container
    ) -> None:
        """Container running without healthcheck should pass validation."""
        mock_docker_client.containers.get.return_value = running_no_health_container

        # Should not raise - no healthcheck means assume healthy if running
        validate_containers(["my-container"])

    def test_validate_containers_empty_list(self, mock_docker_client) -> None:
        """Empty container list should pass validation (nothing to validate)."""
        # Should not raise
        validate_containers([])

        # Docker client should not be called
        mock_docker_client.containers.get.assert_not_called()

    def test_validate_containers_mixed_health_configs(
        self, mock_docker_client, healthy_container, running_no_health_container
    ) -> None:
        """Mix of containers with/without healthcheck should pass if all running."""
        mock_docker_client.containers.get.side_effect = [
            healthy_container,
            running_no_health_container,
        ]

        # Should not raise
        validate_containers(["with-health", "without-health"])


# =============================================================================
# Container Not Found
# =============================================================================


class TestValidateContainersNotFound:
    """Test container not found scenarios."""

    def test_validate_containers_not_found(self, mock_docker_client) -> None:
        """Missing container should raise ConfigError with 'not found' message."""
        # Import here to get the real NotFound exception type
        from docker.errors import NotFound

        mock_docker_client.containers.get.side_effect = NotFound("Container not found")

        with pytest.raises(ConfigError) as exc_info:
            validate_containers(["missing-container"])

        assert "missing-container" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_validate_containers_not_found_multiple(self, mock_docker_client) -> None:
        """Multiple missing containers should all be listed in error."""
        from docker.errors import NotFound

        mock_docker_client.containers.get.side_effect = NotFound("Container not found")

        with pytest.raises(ConfigError) as exc_info:
            validate_containers(["missing-1", "missing-2", "missing-3"])

        error_msg = str(exc_info.value)
        assert "missing-1" in error_msg
        assert "missing-2" in error_msg
        assert "missing-3" in error_msg


# =============================================================================
# Container Not Running
# =============================================================================


class TestValidateContainersNotRunning:
    """Test container not running scenarios."""

    def test_validate_containers_not_running(
        self, mock_docker_client, stopped_container
    ) -> None:
        """Stopped container should raise ConfigError with status message."""
        mock_docker_client.containers.get.return_value = stopped_container

        with pytest.raises(ConfigError) as exc_info:
            validate_containers(["stopped-container"])

        error_msg = str(exc_info.value)
        assert "stopped-container" in error_msg
        assert "not running" in error_msg
        assert "exited" in error_msg

    def test_validate_containers_created_status(self, mock_docker_client) -> None:
        """Container with 'created' status should fail validation."""
        container = MagicMock()
        container.status = "created"
        container.attrs = {"State": {}}
        mock_docker_client.containers.get.return_value = container

        with pytest.raises(ConfigError) as exc_info:
            validate_containers(["created-container"])

        assert "not running" in str(exc_info.value)
        assert "created" in str(exc_info.value)


# =============================================================================
# Container Unhealthy
# =============================================================================


class TestValidateContainersUnhealthy:
    """Test container unhealthy scenarios."""

    def test_validate_containers_unhealthy(
        self, mock_docker_client, unhealthy_container
    ) -> None:
        """Unhealthy container should raise ConfigError with health status."""
        mock_docker_client.containers.get.return_value = unhealthy_container

        with pytest.raises(ConfigError) as exc_info:
            validate_containers(["unhealthy-container"])

        error_msg = str(exc_info.value)
        assert "unhealthy-container" in error_msg
        assert "unhealthy" in error_msg

    def test_validate_containers_starting(
        self, mock_docker_client, starting_container
    ) -> None:
        """Container with 'starting' health should fail validation."""
        mock_docker_client.containers.get.return_value = starting_container

        with pytest.raises(ConfigError) as exc_info:
            validate_containers(["starting-container"])

        error_msg = str(exc_info.value)
        assert "starting-container" in error_msg
        # Health status other than 'healthy' should fail
        assert "starting" in error_msg


# =============================================================================
# Multiple Failures
# =============================================================================


class TestValidateContainersMultipleFailures:
    """Test multiple failure scenarios (error lists ALL failures)."""

    def test_validate_containers_multiple_failures_listed(
        self, mock_docker_client, stopped_container, unhealthy_container
    ) -> None:
        """Multiple failures should all be listed in error message."""
        from docker.errors import NotFound

        mock_docker_client.containers.get.side_effect = [
            NotFound("Container not found"),  # container-1: not found
            stopped_container,                 # container-2: not running
            unhealthy_container,              # container-3: unhealthy
        ]

        with pytest.raises(ConfigError) as exc_info:
            validate_containers(["container-1", "container-2", "container-3"])

        error_msg = str(exc_info.value)
        # All three failures should be listed
        assert "container-1" in error_msg
        assert "not found" in error_msg
        assert "container-2" in error_msg
        assert "not running" in error_msg
        assert "container-3" in error_msg
        assert "unhealthy" in error_msg

    def test_validate_containers_partial_success(
        self, mock_docker_client, healthy_container, stopped_container
    ) -> None:
        """Mix of valid and invalid containers should fail listing only invalids."""
        mock_docker_client.containers.get.side_effect = [
            healthy_container,    # valid
            stopped_container,    # invalid
            healthy_container,    # valid
        ]

        with pytest.raises(ConfigError) as exc_info:
            validate_containers(["good-1", "bad-1", "good-2"])

        error_msg = str(exc_info.value)
        # Only the failed container should be in the error
        assert "bad-1" in error_msg
        assert "good-1" not in error_msg
        assert "good-2" not in error_msg


# =============================================================================
# Docker Connection Errors
# =============================================================================


class TestValidateContainersDockerErrors:
    """Test Docker connection and API error handling."""

    def test_validate_containers_docker_connection_failed(self) -> None:
        """Failed Docker connection should raise ConfigError."""
        with patch("shared.validation.containers.docker") as mock_docker:
            mock_docker.from_env.side_effect = Exception("Cannot connect to Docker")
            mock_docker.errors.DockerException = Exception

            with pytest.raises(ConfigError) as exc_info:
                validate_containers(["any-container"])

            assert "Docker daemon" in str(exc_info.value)

    def test_validate_containers_api_error(self, mock_docker_client) -> None:
        """Docker API error should be included in error message."""
        from docker.errors import APIError

        mock_docker_client.containers.get.side_effect = APIError("API failed")

        with pytest.raises(ConfigError) as exc_info:
            validate_containers(["api-error-container"])

        error_msg = str(exc_info.value)
        assert "api-error-container" in error_msg
        assert "Docker API error" in error_msg


# =============================================================================
# Error Message Format
# =============================================================================


class TestValidateContainersErrorFormat:
    """Test error message formatting."""

    def test_error_message_has_header(
        self, mock_docker_client, stopped_container
    ) -> None:
        """Error message should have 'Container validation failed' header."""
        mock_docker_client.containers.get.return_value = stopped_container

        with pytest.raises(ConfigError) as exc_info:
            validate_containers(["container"])

        assert "Container validation failed" in str(exc_info.value)

    def test_error_message_bullets_each_failure(self, mock_docker_client) -> None:
        """Each failure should be on its own bulleted line."""
        from docker.errors import NotFound

        mock_docker_client.containers.get.side_effect = NotFound("Not found")

        with pytest.raises(ConfigError) as exc_info:
            validate_containers(["a", "b"])

        error_msg = str(exc_info.value)
        # Should have bullet points for each error
        assert error_msg.count("  - ") >= 2


# =============================================================================
# max_wait Parameter
# =============================================================================


class TestValidateContainersMaxWait:
    """Test max_wait parameter behavior."""

    def test_max_wait_parameter_accepted(
        self, mock_docker_client, healthy_container
    ) -> None:
        """max_wait parameter should be accepted (reserved for future use)."""
        mock_docker_client.containers.get.return_value = healthy_container

        # Should not raise with custom max_wait
        validate_containers(["container"], max_wait=60.0)
