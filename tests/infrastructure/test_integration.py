"""
Integration Tests for Infrastructure Services

This module tests the infrastructure services with Docker.
Tests verify that services start correctly and respond on expected ports.

Requirements:
- Docker and Docker Compose installed
- Sufficient permissions to run Docker

Test Categories:
- Service startup: Services come up healthy
- Connectivity: Services respond on expected ports
- MinIO: Bucket operations work
- Prometheus: Scraping cAdvisor

Run with: pytest tests/infrastructure/test_integration.py -v -m integration

Author: Matthew Hong
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Generator

import pytest
import requests

from shared.config import get_infrastructure_config, get_controlled_variables


# =============================================================================
# Constants (Loaded from experiment.yaml where available)
# =============================================================================

COMPOSE_FILE = "infrastructure/docker-compose.infra.yml"
STARTUP_TIMEOUT = 60  # seconds
HEALTH_CHECK_INTERVAL = 2  # seconds

# Service URLs derived from experiment.yaml infrastructure config
_infra_config = get_infrastructure_config()
_minio_endpoint = _infra_config["minio"]["external_endpoint"]
_monitoring_config = get_controlled_variables("monitoring")
_cadvisor_port = _monitoring_config["cadvisor"]["port"]

MINIO_API_URL = f"http://{_minio_endpoint}"
MINIO_CONSOLE_URL = "http://localhost:9001"  # MinIO console port (not in config yet)
CADVISOR_URL = f"http://localhost:{_cadvisor_port}"
PROMETHEUS_URL = "http://localhost:9090"  # Prometheus port (not in config yet)
GRAFANA_URL = "http://localhost:3000"  # Grafana port (not in config yet)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="module")
def compose_file_path(project_root: Path) -> Path:
    """Get absolute path to compose file."""
    return project_root / COMPOSE_FILE


def is_docker_available() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def is_compose_available() -> bool:
    """Check if Docker Compose is available."""
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.fixture(scope="module")
def infrastructure_services(
    project_root: Path,
    compose_file_path: Path
) -> Generator[None, None, None]:
    """
    Start infrastructure services for testing.
    
    This fixture:
    1. Starts all infrastructure services
    2. Waits for health checks
    3. Yields for tests to run
    4. Tears down services after tests
    """
    if not is_docker_available():
        pytest.skip("Docker not available")
    
    if not is_compose_available():
        pytest.skip("Docker Compose not available")
    
    if not compose_file_path.exists():
        pytest.skip(f"Compose file not found: {compose_file_path}")
    
    # Start services
    start_cmd = [
        "docker", "compose",
        "-f", str(compose_file_path),
        "up", "-d", "--wait"
    ]
    
    try:
        subprocess.run(
            start_cmd,
            cwd=project_root,
            check=True,
            capture_output=True,
            timeout=STARTUP_TIMEOUT
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Failed to start services: {e.stderr.decode()}")
    except subprocess.TimeoutExpired:
        pytest.fail("Timeout waiting for services to start")
    
    # Wait for services to be ready
    _wait_for_services()
    
    yield
    
    # Teardown
    stop_cmd = [
        "docker", "compose",
        "-f", str(compose_file_path),
        "down", "-v"
    ]
    
    subprocess.run(
        stop_cmd,
        cwd=project_root,
        capture_output=True,
        timeout=60
    )


def _wait_for_services() -> None:
    """Wait for all services to respond."""
    services = [
        ("MinIO", f"{MINIO_API_URL}/minio/health/live"),
        ("cAdvisor", f"{CADVISOR_URL}/healthz"),
        ("Prometheus", f"{PROMETHEUS_URL}/-/healthy"),
        ("Grafana", f"{GRAFANA_URL}/api/health"),
    ]
    
    deadline = time.time() + STARTUP_TIMEOUT
    
    for name, url in services:
        while time.time() < deadline:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(HEALTH_CHECK_INTERVAL)
        else:
            pytest.fail(f"{name} did not become healthy within {STARTUP_TIMEOUT}s")


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.integration
class TestServiceConnectivity:
    """Test that services are reachable."""

    def test_minio_api_responds(self, infrastructure_services: None) -> None:
        """MinIO API should respond on port 9000."""
        response = requests.get(f"{MINIO_API_URL}/minio/health/live", timeout=5)
        assert response.status_code == 200

    def test_minio_console_responds(self, infrastructure_services: None) -> None:
        """MinIO console should respond on port 9001."""
        response = requests.get(MINIO_CONSOLE_URL, timeout=5)
        # Console returns redirect or login page
        assert response.status_code in [200, 302, 307]

    def test_cadvisor_responds(self, infrastructure_services: None) -> None:
        """cAdvisor should respond on port 8080."""
        response = requests.get(f"{CADVISOR_URL}/healthz", timeout=5)
        assert response.status_code == 200

    def test_prometheus_responds(self, infrastructure_services: None) -> None:
        """Prometheus should respond on port 9090."""
        response = requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=5)
        assert response.status_code == 200

    def test_grafana_responds(self, infrastructure_services: None) -> None:
        """Grafana should respond on port 3000."""
        response = requests.get(f"{GRAFANA_URL}/api/health", timeout=5)
        assert response.status_code == 200


@pytest.mark.integration
class TestMinIOOperations:
    """Test MinIO bucket and object operations."""

    def test_minio_list_buckets(self, infrastructure_services: None) -> None:
        """Should be able to list buckets (empty initially)."""
        try:
            from minio import Minio
        except ImportError:
            pytest.skip("minio package not installed")

        # Use config values for MinIO connection
        client = Minio(
            _minio_endpoint,
            access_key=_infra_config["minio"]["access_key"],
            secret_key=_infra_config["minio"]["secret_key"],
            secure=_infra_config["minio"]["secure"]
        )

        buckets = client.list_buckets()
        assert isinstance(buckets, list)

    def test_minio_create_bucket(self, infrastructure_services: None) -> None:
        """Should be able to create a bucket."""
        try:
            from minio import Minio
        except ImportError:
            pytest.skip("minio package not installed")

        # Use config values for MinIO connection
        client = Minio(
            _minio_endpoint,
            access_key=_infra_config["minio"]["access_key"],
            secret_key=_infra_config["minio"]["secret_key"],
            secure=_infra_config["minio"]["secure"]
        )

        bucket_name = "test-bucket"

        # Create bucket
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)

        assert client.bucket_exists(bucket_name)

        # Cleanup
        client.remove_bucket(bucket_name)


@pytest.mark.integration
class TestPrometheusMetrics:
    """Test Prometheus metrics collection."""

    def test_prometheus_targets_healthy(self, infrastructure_services: None) -> None:
        """Prometheus should have healthy targets."""
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/targets", timeout=5)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        
        active_targets = data["data"]["activeTargets"]
        assert len(active_targets) > 0, "No active scrape targets"

    def test_prometheus_scrapes_cadvisor(self, infrastructure_services: None) -> None:
        """Prometheus should successfully scrape cAdvisor."""
        # Wait a bit for scraping to occur
        time.sleep(3)
        
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/targets", timeout=5)
        data = response.json()
        
        cadvisor_target = None
        for target in data["data"]["activeTargets"]:
            if target.get("labels", {}).get("job") == "cadvisor":
                cadvisor_target = target
                break
        
        assert cadvisor_target is not None, "cAdvisor target not found"
        assert cadvisor_target["health"] == "up", "cAdvisor target not healthy"

    def test_prometheus_has_container_metrics(self, infrastructure_services: None) -> None:
        """Prometheus should have container CPU metrics."""
        # Wait for metrics to be scraped
        time.sleep(5)
        
        query = "container_cpu_usage_seconds_total"
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=5
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        
        # Should have some results (at least for infrastructure containers)
        results = data["data"]["result"]
        assert len(results) > 0, f"No results for {query}"


@pytest.mark.integration
class TestGrafanaProvisioning:
    """Test Grafana datasource and dashboard provisioning."""

    def test_grafana_prometheus_datasource(self, infrastructure_services: None) -> None:
        """Grafana should have Prometheus datasource configured."""
        response = requests.get(
            f"{GRAFANA_URL}/api/datasources",
            auth=("admin", "admin"),
            timeout=5
        )
        
        assert response.status_code == 200
        datasources = response.json()
        
        prometheus_ds = next(
            (ds for ds in datasources if ds["type"] == "prometheus"),
            None
        )
        assert prometheus_ds is not None, "Prometheus datasource not found"

    def test_grafana_dashboard_provisioned(self, infrastructure_services: None) -> None:
        """Grafana should have infrastructure dashboard."""
        response = requests.get(
            f"{GRAFANA_URL}/api/search",
            params={"query": "Infrastructure"},
            auth=("admin", "admin"),
            timeout=5
        )
        
        assert response.status_code == 200
        dashboards = response.json()
        
        # Should find at least one dashboard
        assert len(dashboards) > 0, "No dashboards found"


@pytest.mark.integration
class TestNetworkConnectivity:
    """Test network isolation and connectivity."""

    def test_containers_on_backend_network(self, infrastructure_services: None) -> None:
        """MinIO and cAdvisor should be on backend network."""
        result = subprocess.run(
            ["docker", "network", "inspect", "inference-arena-backend"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        network_info = json.loads(result.stdout)
        
        containers = network_info[0].get("Containers", {})
        container_names = [c["Name"] for c in containers.values()]
        
        assert any("minio" in name for name in container_names), \
            "MinIO not on backend network"
        assert any("cadvisor" in name for name in container_names), \
            "cAdvisor not on backend network"

    def test_grafana_on_infra_network(self, infrastructure_services: None) -> None:
        """Grafana should be on infra network."""
        result = subprocess.run(
            ["docker", "network", "inspect", "inference-arena-infra"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        network_info = json.loads(result.stdout)
        
        containers = network_info[0].get("Containers", {})
        container_names = [c["Name"] for c in containers.values()]
        
        assert any("grafana" in name for name in container_names), \
            "Grafana not on infra network"
