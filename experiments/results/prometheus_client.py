"""Prometheus client for querying container resource metrics.

This module provides the PrometheusClient class for querying CPU, memory,
and network metrics from Prometheus during load tests.

Metrics are collected from OpenTelemetry Collector's docker_stats receiver
and filtered by container_name label for architecture-specific resource usage.

Usage:
    from experiments.results import PrometheusClient

    client = PrometheusClient()
    cpu = client.query_cpu_usage("inference-arena-monolithic", start_time, end_time)
    memory = client.query_memory_usage("inference-arena-monolithic", start_time, end_time)

Author: Matthew Hong
Specification Reference: infrastructure/otel/otel-collector-config.yaml
"""

import json
import logging
import os
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# Default Prometheus URL
DEFAULT_PROMETHEUS_URL = "http://localhost:9090"


class PrometheusClient:
    """Client for querying Prometheus metrics.

    This class queries Prometheus for container resource metrics
    collected by OpenTelemetry Collector, providing CPU, memory,
    and network usage data for experiment analysis.

    Attributes:
        url: Prometheus server URL

    Example:
        >>> client = PrometheusClient()
        >>> cpu = client.query_cpu_usage("inference-arena-monolithic", start, end)
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

        logger.info(f"PrometheusClient initialized with URL: {self.url}")

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
            raise RuntimeError(f"Failed to query Prometheus: {e}") from e

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
            raise RuntimeError(f"Failed to query Prometheus: {e}") from e

    def query_cpu_usage(
        self,
        container_name: str,
        start: float,
        end: float,
    ) -> dict[str, float]:
        """Query CPU usage for a container during time range.

        Uses container_cpu_utilization_ratio gauge from OTel docker_stats receiver.
        This metric is already a percentage (0-100+ per core, 0-200+ for 2 cores).

        Args:
            container_name: Full container name (e.g., "inference-arena-monolithic")
            start: Start time as Unix timestamp
            end: End time as Unix timestamp

        Returns:
            Dictionary with avg_percent, max_percent, min_percent
        """
        # PromQL: CPU utilization from OTel docker_stats
        # container_cpu_utilization_ratio is already a percentage (0-100+ per core)
        query = f'container_cpu_utilization_ratio{{container_name="{container_name}"}}'

        try:
            result = self._query_range(query, start, end, step="5s")
            values = self._extract_values(result)

            if not values:
                logger.warning(f"No CPU data for container: {container_name}")
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

        Uses container_memory_usage_total from OTel docker_stats receiver.

        Args:
            container_name: Full container name (e.g., "inference-arena-monolithic")
            start: Start time as Unix timestamp
            end: End time as Unix timestamp

        Returns:
            Dictionary with avg_mb, max_mb, min_mb
        """
        # PromQL: Memory usage in MB from OTel docker_stats
        # container_memory_usage_total_bytes is the actual metric name
        query = (
            f'container_memory_usage_total_bytes{{container_name="{container_name}"}} / 1024 / 1024'
        )

        try:
            result = self._query_range(query, start, end, step="5s")
            values = self._extract_values(result)

            if not values:
                logger.warning(f"No memory data for container: {container_name}")
                return {"avg_mb": 0.0, "max_mb": 0.0, "min_mb": 0.0}

            return {
                "avg_mb": round(sum(values) / len(values), 2),
                "max_mb": round(max(values), 2),
                "min_mb": round(min(values), 2),
            }

        except Exception as e:
            logger.error(f"Failed to query memory usage: {e}")
            return {"avg_mb": 0.0, "max_mb": 0.0, "min_mb": 0.0}

    def query_network_io(
        self,
        container_name: str,
        start: float,
        end: float,
    ) -> dict[str, float]:
        """Query network I/O for a container during time range.

        Uses container_network_io_usage_rx_bytes and tx_bytes from OTel docker_stats.
        NOTE: This now works in WSL2, unlike cAdvisor.

        Args:
            container_name: Full container name (e.g., "inference-arena-monolithic")
            start: Start time as Unix timestamp
            end: End time as Unix timestamp

        Returns:
            Dictionary with rx_bytes_per_sec, tx_bytes_per_sec (average rates)
        """
        rx_query = f'rate(container_network_io_usage_rx_bytes_total{{container_name="{container_name}"}}[1m])'
        tx_query = f'rate(container_network_io_usage_tx_bytes_total{{container_name="{container_name}"}}[1m])'

        try:
            rx_result = self._query_range(rx_query, start, end, step="5s")
            tx_result = self._query_range(tx_query, start, end, step="5s")

            rx_values = self._extract_values(rx_result)
            tx_values = self._extract_values(tx_result)

            return {
                "rx_bytes_per_sec": round(sum(rx_values) / len(rx_values), 2) if rx_values else 0.0,
                "tx_bytes_per_sec": round(sum(tx_values) / len(tx_values), 2) if tx_values else 0.0,
            }

        except Exception as e:
            logger.error(f"Failed to query network I/O: {e}")
            return {"rx_bytes_per_sec": 0.0, "tx_bytes_per_sec": 0.0}

    def query_container_metrics(
        self,
        container_names: list[str],
        start: float,
        end: float,
    ) -> dict[str, dict[str, Any]]:
        """Query CPU, memory, and network for multiple containers.

        Args:
            container_names: List of full container names
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
                "network": self.query_network_io(name, start, end),
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
                for _timestamp, value in series.get("values", []):
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
