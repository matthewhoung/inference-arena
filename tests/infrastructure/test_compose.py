"""
Unit Tests for Docker Compose Infrastructure Configuration

This module validates the docker-compose.infra.yml file without requiring Docker.
Tests cover YAML structure, required services, port mappings, and configuration.

Test Categories:
- YAML parsing: File loads without errors
- Service validation: Required services exist with correct configuration
- Network validation: Networks properly defined
- Volume validation: Persistent storage configured

Author: Matthew Hong
Specification Reference: Ch3 Methodology §3.4.5
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def project_root() -> Path:
    """Get project root directory."""
    # Navigate from tests/infrastructure/ to project root
    return Path(__file__).parent.parent.parent


@pytest.fixture
def compose_file(project_root: Path) -> Path:
    """Get path to docker-compose.infra.yml."""
    return project_root / "infrastructure" / "docker-compose.infra.yml"


@pytest.fixture
def compose_config(compose_file: Path) -> Dict[str, Any]:
    """Load and parse docker-compose.infra.yml."""
    if not compose_file.exists():
        pytest.skip(f"Compose file not found: {compose_file}")
    
    with open(compose_file, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def prometheus_config(project_root: Path) -> Dict[str, Any]:
    """Load and parse prometheus.yml."""
    config_path = project_root / "infrastructure" / "prometheus" / "prometheus.yml"
    if not config_path.exists():
        pytest.skip(f"Prometheus config not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def grafana_datasources(project_root: Path) -> Dict[str, Any]:
    """Load and parse Grafana datasources.yml."""
    config_path = project_root / "infrastructure" / "grafana" / "provisioning" / "datasources" / "datasources.yml"
    if not config_path.exists():
        pytest.skip(f"Grafana datasources not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def grafana_dashboard(project_root: Path) -> Dict[str, Any]:
    """Load and parse Grafana dashboard JSON."""
    dashboard_path = project_root / "infrastructure" / "grafana" / "provisioning" / "dashboards" / "infrastructure.json"
    if not dashboard_path.exists():
        pytest.skip(f"Grafana dashboard not found: {dashboard_path}")
    
    with open(dashboard_path, "r") as f:
        return json.load(f)


# =============================================================================
# Docker Compose Structure Tests
# =============================================================================

class TestComposeStructure:
    """Test docker-compose.infra.yml structure and syntax."""

    def test_compose_file_exists(self, compose_file: Path) -> None:
        """Compose file should exist."""
        assert compose_file.exists(), f"Missing: {compose_file}"

    def test_compose_is_valid_yaml(self, compose_config: Dict[str, Any]) -> None:
        """Compose file should be valid YAML."""
        assert compose_config is not None
        assert isinstance(compose_config, dict)

    def test_has_services_section(self, compose_config: Dict[str, Any]) -> None:
        """Compose file should have services section."""
        assert "services" in compose_config
        assert isinstance(compose_config["services"], dict)

    def test_has_networks_section(self, compose_config: Dict[str, Any]) -> None:
        """Compose file should have networks section."""
        assert "networks" in compose_config
        assert isinstance(compose_config["networks"], dict)

    def test_has_volumes_section(self, compose_config: Dict[str, Any]) -> None:
        """Compose file should have volumes section."""
        assert "volumes" in compose_config
        assert isinstance(compose_config["volumes"], dict)


# =============================================================================
# Required Services Tests
# =============================================================================

class TestRequiredServices:
    """Test that all required infrastructure services are defined."""

    REQUIRED_SERVICES = ["minio", "cadvisor", "prometheus", "grafana"]

    def test_all_services_exist(self, compose_config: Dict[str, Any]) -> None:
        """All required services should be defined."""
        services = compose_config.get("services", {})
        for service in self.REQUIRED_SERVICES:
            assert service in services, f"Missing service: {service}"

    @pytest.mark.parametrize("service", REQUIRED_SERVICES)
    def test_service_has_image(self, compose_config: Dict[str, Any], service: str) -> None:
        """Each service should specify an image."""
        services = compose_config.get("services", {})
        assert "image" in services[service], f"{service} missing image"

    @pytest.mark.parametrize("service", REQUIRED_SERVICES)
    def test_service_has_container_name(self, compose_config: Dict[str, Any], service: str) -> None:
        """Each service should have a container name for consistent identification."""
        services = compose_config.get("services", {})
        assert "container_name" in services[service], f"{service} missing container_name"
        # Container names should follow naming convention
        container_name = services[service]["container_name"]
        assert container_name.startswith("inference-arena-"), f"{service} container name should start with 'inference-arena-'"


# =============================================================================
# MinIO Service Tests
# =============================================================================

class TestMinIOService:
    """Test MinIO service configuration."""

    def test_minio_image_version(self, compose_config: Dict[str, Any]) -> None:
        """MinIO should use a pinned version."""
        minio = compose_config["services"]["minio"]
        assert "minio/minio:" in minio["image"]
        # Should have specific release tag, not 'latest'
        assert "latest" not in minio["image"].lower()

    def test_minio_ports(self, compose_config: Dict[str, Any]) -> None:
        """MinIO should expose API (9000) and console (9001) ports."""
        minio = compose_config["services"]["minio"]
        ports = minio.get("ports", [])
        port_mappings = [str(p) for p in ports]
        assert any("9000" in p for p in port_mappings), "MinIO API port 9000 not exposed"
        assert any("9001" in p for p in port_mappings), "MinIO console port 9001 not exposed"

    def test_minio_healthcheck(self, compose_config: Dict[str, Any]) -> None:
        """MinIO should have a healthcheck for startup ordering."""
        minio = compose_config["services"]["minio"]
        assert "healthcheck" in minio, "MinIO should have healthcheck"
        healthcheck = minio["healthcheck"]
        assert "test" in healthcheck
        assert "interval" in healthcheck

    def test_minio_persistent_volume(self, compose_config: Dict[str, Any]) -> None:
        """MinIO should have persistent volume for data."""
        minio = compose_config["services"]["minio"]
        volumes = minio.get("volumes", [])
        assert any("minio" in str(v).lower() and "/data" in str(v) for v in volumes), \
            "MinIO should mount data volume"

    def test_minio_credentials(self, compose_config: Dict[str, Any]) -> None:
        """MinIO should have environment variables for credentials."""
        minio = compose_config["services"]["minio"]
        env = minio.get("environment", {})
        # Can be dict or list
        if isinstance(env, list):
            env_str = " ".join(env)
            assert "MINIO_ROOT_USER" in env_str
            assert "MINIO_ROOT_PASSWORD" in env_str
        else:
            assert "MINIO_ROOT_USER" in env
            assert "MINIO_ROOT_PASSWORD" in env


# =============================================================================
# cAdvisor Service Tests
# =============================================================================

class TestCAdvisorService:
    """Test cAdvisor service configuration."""

    def test_cadvisor_image_version(self, compose_config: Dict[str, Any]) -> None:
        """cAdvisor should use a pinned version."""
        cadvisor = compose_config["services"]["cadvisor"]
        assert "cadvisor" in cadvisor["image"].lower()
        assert "latest" not in cadvisor["image"].lower()

    def test_cadvisor_port(self, compose_config: Dict[str, Any]) -> None:
        """cAdvisor should expose port 8080."""
        cadvisor = compose_config["services"]["cadvisor"]
        ports = cadvisor.get("ports", [])
        port_mappings = [str(p) for p in ports]
        assert any("8080" in p for p in port_mappings), "cAdvisor port 8080 not exposed"

    def test_cadvisor_required_volumes(self, compose_config: Dict[str, Any]) -> None:
        """cAdvisor should have required host volume mounts."""
        cadvisor = compose_config["services"]["cadvisor"]
        volumes = cadvisor.get("volumes", [])
        volume_str = " ".join(str(v) for v in volumes)
        
        # cAdvisor needs these mounts to read container stats
        assert "/var/run" in volume_str, "cAdvisor needs /var/run mount"
        assert "/sys" in volume_str, "cAdvisor needs /sys mount"
        assert "/var/lib/docker" in volume_str, "cAdvisor needs /var/lib/docker mount"

    def test_cadvisor_privileged(self, compose_config: Dict[str, Any]) -> None:
        """cAdvisor should run privileged for full access."""
        cadvisor = compose_config["services"]["cadvisor"]
        assert cadvisor.get("privileged") is True, "cAdvisor should be privileged"


# =============================================================================
# Prometheus Service Tests
# =============================================================================

class TestPrometheusService:
    """Test Prometheus service configuration."""

    def test_prometheus_image_version(self, compose_config: Dict[str, Any]) -> None:
        """Prometheus should use a pinned version."""
        prometheus = compose_config["services"]["prometheus"]
        assert "prometheus" in prometheus["image"].lower()
        assert "latest" not in prometheus["image"].lower()

    def test_prometheus_port(self, compose_config: Dict[str, Any]) -> None:
        """Prometheus should expose port 9090."""
        prometheus = compose_config["services"]["prometheus"]
        ports = prometheus.get("ports", [])
        port_mappings = [str(p) for p in ports]
        assert any("9090" in p for p in port_mappings), "Prometheus port 9090 not exposed"

    def test_prometheus_config_mount(self, compose_config: Dict[str, Any]) -> None:
        """Prometheus should mount configuration file."""
        prometheus = compose_config["services"]["prometheus"]
        volumes = prometheus.get("volumes", [])
        volume_str = " ".join(str(v) for v in volumes)
        assert "prometheus.yml" in volume_str, "Prometheus config not mounted"

    def test_prometheus_depends_on_cadvisor(self, compose_config: Dict[str, Any]) -> None:
        """Prometheus should depend on cAdvisor."""
        prometheus = compose_config["services"]["prometheus"]
        depends_on = prometheus.get("depends_on", {})
        # Can be list or dict
        if isinstance(depends_on, list):
            assert "cadvisor" in depends_on
        else:
            assert "cadvisor" in depends_on


# =============================================================================
# Grafana Service Tests
# =============================================================================

class TestGrafanaService:
    """Test Grafana service configuration."""

    def test_grafana_image_version(self, compose_config: Dict[str, Any]) -> None:
        """Grafana should use a pinned version."""
        grafana = compose_config["services"]["grafana"]
        assert "grafana" in grafana["image"].lower()
        assert "latest" not in grafana["image"].lower()

    def test_grafana_port(self, compose_config: Dict[str, Any]) -> None:
        """Grafana should expose port 3000."""
        grafana = compose_config["services"]["grafana"]
        ports = grafana.get("ports", [])
        port_mappings = [str(p) for p in ports]
        assert any("3000" in p for p in port_mappings), "Grafana port 3000 not exposed"

    def test_grafana_provisioning_mounts(self, compose_config: Dict[str, Any]) -> None:
        """Grafana should mount provisioning directories."""
        grafana = compose_config["services"]["grafana"]
        volumes = grafana.get("volumes", [])
        volume_str = " ".join(str(v) for v in volumes)
        assert "datasources" in volume_str, "Grafana datasources not mounted"
        assert "dashboards" in volume_str, "Grafana dashboards not mounted"

    def test_grafana_depends_on_prometheus(self, compose_config: Dict[str, Any]) -> None:
        """Grafana should depend on Prometheus."""
        grafana = compose_config["services"]["grafana"]
        depends_on = grafana.get("depends_on", {})
        if isinstance(depends_on, list):
            assert "prometheus" in depends_on
        else:
            assert "prometheus" in depends_on


# =============================================================================
# Network Tests
# =============================================================================

class TestNetworks:
    """Test network configuration."""

    def test_infra_network_exists(self, compose_config: Dict[str, Any]) -> None:
        """Infrastructure network should be defined."""
        networks = compose_config.get("networks", {})
        assert "infra-network" in networks

    def test_backend_network_exists(self, compose_config: Dict[str, Any]) -> None:
        """Backend network should be defined."""
        networks = compose_config.get("networks", {})
        assert "backend-network" in networks

    def test_networks_have_names(self, compose_config: Dict[str, Any]) -> None:
        """Networks should have explicit names for external reference."""
        networks = compose_config.get("networks", {})
        
        for network_key, network_config in networks.items():
            assert "name" in network_config, f"Network {network_key} should have explicit name"
            assert "inference-arena" in network_config["name"], \
                f"Network name should include 'inference-arena' prefix"

    def test_minio_on_backend_network(self, compose_config: Dict[str, Any]) -> None:
        """MinIO should be on backend network for architecture access."""
        minio = compose_config["services"]["minio"]
        networks = minio.get("networks", [])
        assert "backend-network" in networks, "MinIO should be on backend-network"

    def test_prometheus_on_both_networks(self, compose_config: Dict[str, Any]) -> None:
        """Prometheus should be on both networks to scrape cAdvisor and serve Grafana."""
        prometheus = compose_config["services"]["prometheus"]
        networks = prometheus.get("networks", [])
        assert "infra-network" in networks, "Prometheus should be on infra-network"
        assert "backend-network" in networks, "Prometheus should be on backend-network"


# =============================================================================
# Prometheus Configuration Tests
# =============================================================================

class TestPrometheusConfig:
    """Test prometheus.yml configuration."""

    def test_scrape_interval_is_one_second(self, prometheus_config: Dict[str, Any]) -> None:
        """Scrape interval should be 1 second per methodology specification."""
        global_config = prometheus_config.get("global", {})
        scrape_interval = global_config.get("scrape_interval", "")
        assert scrape_interval == "1s", f"Scrape interval should be 1s, got {scrape_interval}"

    def test_cadvisor_job_exists(self, prometheus_config: Dict[str, Any]) -> None:
        """cAdvisor scrape job should be configured."""
        scrape_configs = prometheus_config.get("scrape_configs", [])
        job_names = [c.get("job_name") for c in scrape_configs]
        assert "cadvisor" in job_names, "cAdvisor job not configured"

    def test_cadvisor_target(self, prometheus_config: Dict[str, Any]) -> None:
        """cAdvisor target should point to correct host:port."""
        scrape_configs = prometheus_config.get("scrape_configs", [])
        cadvisor_config = next((c for c in scrape_configs if c.get("job_name") == "cadvisor"), None)
        assert cadvisor_config is not None
        
        static_configs = cadvisor_config.get("static_configs", [])
        targets = []
        for sc in static_configs:
            targets.extend(sc.get("targets", []))
        
        assert any("cadvisor:8080" in t for t in targets), "cAdvisor target should be cadvisor:8080"


# =============================================================================
# Grafana Configuration Tests
# =============================================================================

class TestGrafanaConfig:
    """Test Grafana provisioning configuration."""

    def test_prometheus_datasource(self, grafana_datasources: Dict[str, Any]) -> None:
        """Prometheus should be configured as datasource."""
        datasources = grafana_datasources.get("datasources", [])
        prometheus_ds = next((d for d in datasources if d.get("type") == "prometheus"), None)
        assert prometheus_ds is not None, "Prometheus datasource not configured"
        assert prometheus_ds.get("isDefault") is True, "Prometheus should be default datasource"

    def test_dashboard_exists(self, grafana_dashboard: Dict[str, Any]) -> None:
        """Dashboard should be valid JSON with panels."""
        assert "panels" in grafana_dashboard
        assert len(grafana_dashboard["panels"]) > 0

    def test_dashboard_has_cpu_panel(self, grafana_dashboard: Dict[str, Any]) -> None:
        """Dashboard should have CPU utilization panel."""
        panels = grafana_dashboard.get("panels", [])
        panel_titles = [p.get("title", "") for p in panels]
        assert any("cpu" in t.lower() for t in panel_titles), "Dashboard should have CPU panel"

    def test_dashboard_has_memory_panel(self, grafana_dashboard: Dict[str, Any]) -> None:
        """Dashboard should have memory usage panel."""
        panels = grafana_dashboard.get("panels", [])
        panel_titles = [p.get("title", "") for p in panels]
        assert any("memory" in t.lower() for t in panel_titles), "Dashboard should have Memory panel"
