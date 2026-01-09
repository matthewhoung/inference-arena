"""Prometheus client for querying container resource metrics.

This module provides the PrometheusClient class for querying CPU and
memory metrics from Prometheus during load tests.

Metrics are collected from cAdvisor and filtered by container ID
to get architecture-specific resource usage.

Usage:
    from experiments.results import PrometheusClient

    client = PrometheusClient()
    cpu = client.query_cpu_usage("monolithic", start_time, end_time)
    memory = client.query_memory_usage("monolithic", start_time, end_time)

Author: Matthew Hong
Specification Reference: infrastructure/prometheus/prometheus.yml
"""

import json
import logging
import os
import subprocess
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# Container name patterns for docker ps lookup
CONTAINER_NAME_PATTERNS: dict[str, str] = {
    "monolithic": "inference-arena-monolithic",
    "detection": "inference-arena-detection",
    "classification": "inference-arena-classification",
    "triton-server": "inference-arena-triton-server",
    "triton-gateway": "inference-arena-triton-gateway",
}

# Default Prometheus URL
DEFAULT_PROMETHEUS_URL = "http://localhost:9090"


class PrometheusClient:
    """Client for querying Prometheus metrics.

    This class queries Prometheus for container resource metrics
    collected by cAdvisor, providing CPU and memory usage data
    for experiment analysis.

    Attributes:
        url: Prometheus server URL

    Example:
        >>> client = PrometheusClient()
        >>> cpu = client.query_cpu_usage("monolithic", start, end)
        >>> print(f"CPU avg: {cpu['avg_percent']}%")
    """

    def __init__(self, url: str | None = None):
        """Initialize the Prometheus client.

        Args:
            url: Prometheus server URL. Defaults to PROMETHEUS_URL env var
                 or http://localhost:9090
        """
        self.url = url or os.environ.get("PROMETHEUS_URL", DEFAULT_PROMETHEUS_URL)
        self._timeout = 30  # Query timeout in seconds
        self._container_id_cache: dict[str, str] = {}

        logger.info(f"PrometheusClient initialized with URL: {self.url}")

    def _get_container_id(self, container_name: str) -> str | None:
        """Get container ID for a service name.

        Uses docker ps to look up the current container ID for a service.
        Results are cached for the lifetime of this client instance.

        Args:
            container_name: Service name (e.g., "monolithic", "detection")

        Returns:
            12-character container ID or None if not found
        """
        # Check cache first
        if container_name in self._container_id_cache:
            return self._container_id_cache[container_name]

        # Get docker container name pattern
        docker_name = CONTAINER_NAME_PATTERNS.get(container_name)
        if not docker_name:
            logger.warning(f"Unknown container name: {container_name}")
            return None

        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={docker_name}", "--format", "{{.ID}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            container_id = result.stdout.strip()
            if container_id:
                self._container_id_cache[container_name] = container_id
                logger.debug(f"Found container ID for {container_name}: {container_id}")
                return container_id
            else:
                logger.warning(f"Container not running: {docker_name}")
                return None
        except Exception as e:
            logger.error(f"Failed to get container ID for {container_name}: {e}")
            return None

    def _query(self, query: str) -> dict[str, Any]:
        """Execute an instant query against Prometheus.

        Args:
            query: PromQL query string

        Returns:
            Query result dictionary

        Raises:
            RuntimeError: If query fails
        """
        endpoint = f"{self.url}/api/v1/query"
        params = {"query": query}
        url = f"{endpoint}?{urlencode(params)}"

        try:
            req = Request(url)
            with urlopen(req, timeout=self._timeout) as response:
                data = json.loads(response.read().decode())

            if data.get("status") != "success":
                raise RuntimeError(f"Prometheus query failed: {data}")

            return data.get("data", {})

        except URLError as e:
            logger.error(f"Prometheus query failed: {e}")
            raise RuntimeError(f"Failed to query Prometheus: {e}")

    def _query_range(
        self,
        query: str,
        start: float,
        end: float,
        step: str = "1s",
    ) -> dict[str, Any]:
        """Execute a range query against Prometheus.

        Args:
            query: PromQL query string
            start: Start time as Unix timestamp
            end: End time as Unix timestamp
            step: Query resolution step (default: 1s)

        Returns:
            Query result dictionary
        """
        endpoint = f"{self.url}/api/v1/query_range"
        params = {
            "query": query,
            "start": start,
            "end": end,
            "step": step,
        }
        url = f"{endpoint}?{urlencode(params)}"

        try:
            req = Request(url)
            with urlopen(req, timeout=self._timeout) as response:
                data = json.loads(response.read().decode())

            if data.get("status") != "success":
                raise RuntimeError(f"Prometheus range query failed: {data}")

            return data.get("data", {})

        except URLError as e:
            logger.error(f"Prometheus range query failed: {e}")
            raise RuntimeError(f"Failed to query Prometheus: {e}")

    def query_cpu_usage(
        self,
        container_name: str,
        start: float,
        end: float,
    ) -> dict[str, float]:
        """Query CPU usage for a container during time range.

        Uses the rate of container_cpu_usage_seconds_total to calculate
        CPU utilization percentage.

        Args:
            container_name: Docker compose service name
            start: Start time as Unix timestamp
            end: End time as Unix timestamp

        Returns:
            Dictionary with avg_percent, max_percent, min_percent
        """
        # Get container ID for this service
        container_id = self._get_container_id(container_name)
        if not container_id:
            logger.warning(f"Cannot query CPU - container not found: {container_name}")
            return {"avg_percent": 0.0, "max_percent": 0.0, "min_percent": 0.0}

        # PromQL: CPU usage rate over 1 minute, filtered by container ID
        # Raw CPU usage: 100% = 1 vCPU core, 200% = 2 vCPU cores
        query = (
            f"sum(rate(container_cpu_usage_seconds_total{{"
            f'container_id="{container_id}"'
            f"}}[1m])) * 100"
        )

        try:
            result = self._query_range(query, start, end, step="5s")
            values = self._extract_values(result)

            if not values:
                logger.warning(f"No CPU data for container: {container_name} (ID: {container_id})")
                return {"avg_percent": 0.0, "max_percent": 0.0, "min_percent": 0.0}

            return {
                "avg_percent": round(sum(values) / len(values), 2),
                "max_percent": round(max(values), 2),
                "min_percent": round(min(values), 2),
            }

        except Exception as e:
            logger.error(f"Failed to query CPU usage: {e}")
            return {"avg_percent": 0.0, "max_percent": 0.0, "min_percent": 0.0}

    def query_memory_usage(
        self,
        container_name: str,
        start: float,
        end: float,
    ) -> dict[str, float]:
        """Query memory usage for a container during time range.

        Uses container_memory_usage_bytes converted to megabytes.

        Args:
            container_name: Docker compose service name
            start: Start time as Unix timestamp
            end: End time as Unix timestamp

        Returns:
            Dictionary with avg_mb, max_mb, min_mb
        """
        # Get container ID for this service
        container_id = self._get_container_id(container_name)
        if not container_id:
            logger.warning(f"Cannot query memory - container not found: {container_name}")
            return {"avg_mb": 0.0, "max_mb": 0.0, "min_mb": 0.0}

        # PromQL: Memory usage in MB, filtered by container ID
        query = (
            f"sum(container_memory_usage_bytes{{"
            f'container_id="{container_id}"'
            f"}}) / 1024 / 1024"
        )

        try:
            result = self._query_range(query, start, end, step="5s")
            values = self._extract_values(result)

            if not values:
                logger.warning(
                    f"No memory data for container: {container_name} (ID: {container_id})"
                )
                return {"avg_mb": 0.0, "max_mb": 0.0, "min_mb": 0.0}

            return {
                "avg_mb": round(sum(values) / len(values), 2),
                "max_mb": round(max(values), 2),
                "min_mb": round(min(values), 2),
            }

        except Exception as e:
            logger.error(f"Failed to query memory usage: {e}")
            return {"avg_mb": 0.0, "max_mb": 0.0, "min_mb": 0.0}

    def query_container_metrics(
        self,
        container_names: list[str],
        start: float,
        end: float,
    ) -> dict[str, dict[str, Any]]:
        """Query CPU and memory for multiple containers.

        Args:
            container_names: List of docker-compose service names
            start: Start time as Unix timestamp
            end: End time as Unix timestamp

        Returns:
            Dictionary mapping container name to metrics
        """
        results = {}
        for name in container_names:
            results[name] = {
                "cpu": self.query_cpu_usage(name, start, end),
                "memory": self.query_memory_usage(name, start, end),
            }
        return results

    def _extract_values(self, result: dict[str, Any]) -> list[float]:
        """Extract numeric values from Prometheus result.

        Args:
            result: Prometheus query result

        Returns:
            List of float values
        """
        values = []
        result_type = result.get("resultType")

        if result_type == "matrix":
            # Range query result
            for series in result.get("result", []):
                for timestamp, value in series.get("values", []):
                    try:
                        values.append(float(value))
                    except (ValueError, TypeError):
                        pass

        elif result_type == "vector":
            # Instant query result
            for item in result.get("result", []):
                try:
                    values.append(float(item.get("value", [None, 0])[1]))
                except (ValueError, TypeError, IndexError):
                    pass

        return values

    def is_available(self) -> bool:
        """Check if Prometheus is available.

        Returns:
            True if Prometheus responds, False otherwise
        """
        try:
            self._query("up")
            return True
        except Exception:
            return False

    def get_container_names(self) -> list[str]:
        """Get list of available container names.

        Returns known container names that can be queried.

        Returns:
            List of container names
        """
        return sorted(CONTAINER_NAME_PATTERNS.keys())
