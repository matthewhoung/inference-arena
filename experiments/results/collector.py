"""Results aggregation for load testing experiments.

This module provides the ResultsCollector class that aggregates
metrics from Locust and Prometheus into a unified result structure.

Usage:
    from experiments.results import ResultsCollector, PrometheusClient
    from experiments.metrics import get_collector

    collector = ResultsCollector()
    result = collector.collect(
        metrics=get_collector(),
        prometheus=PrometheusClient(),
        architecture="monolithic",
        user_count=10,
        run_number=1,
        measurement_start=start_time,
        measurement_end=end_time,
    )

Author: Matthew Hong
"""

import logging
from datetime import datetime
from typing import Any

from shared.validation import validate_containers

from ..config import CONTAINER_NAMES
from ..metrics import MetricsCollector
from .prometheus_client import PrometheusClient

logger = logging.getLogger(__name__)


class ResultsCollector:
    """Aggregate metrics from Locust and Prometheus.

    This class combines performance metrics from the load testing
    framework with resource metrics from Prometheus to create
    a comprehensive result structure for each experiment run.

    Example:
        >>> collector = ResultsCollector()
        >>> result = collector.collect(...)
        >>> result["architecture"]
        "monolithic"
        >>> result["throughput_rps"]
        45.2
    """

    def collect(
        self,
        metrics: MetricsCollector,
        prometheus: PrometheusClient | None,
        architecture: str,
        user_count: int,
        run_number: int,
        measurement_start: float,
        measurement_end: float,
    ) -> dict[str, Any]:
        """Collect all metrics for a single experiment run.

        Args:
            metrics: MetricsCollector with recorded request metrics
            prometheus: PrometheusClient for resource queries (optional)
            architecture: Architecture name (monolithic, microservices, triton)
            user_count: Number of concurrent users
            run_number: Run number (1, 2, 3, ...)
            measurement_start: Measurement phase start time (Unix timestamp)
            measurement_end: Measurement phase end time (Unix timestamp)

        Returns:
            Dictionary with all collected metrics
        """
        # Get Locust statistics
        stats = metrics.calculate_statistics()

        # Build result structure
        result = {
            # Experiment identification
            "architecture": architecture,
            "concurrent_users": user_count,
            "run_number": run_number,
            "timestamp": datetime.now().isoformat(),
            # Request metrics
            "total_requests": stats["total_requests"],
            "successful_requests": stats["successful_requests"],
            "failed_requests": stats["failed_requests"],
            "error_rate_percent": stats["error_rate_percent"],
            "throughput_rps": stats["throughput_rps"],
            "duration_seconds": stats["duration_seconds"],
            # Latency metrics
            "client_latency": stats["client_latency"],
            "server_latency": stats["server_latency"],
            # Phase summary
            "phase_summary": metrics.get_phase_summary(),
        }

        # Query Prometheus for resource metrics
        if prometheus:
            resource_metrics = self._collect_resource_metrics(
                prometheus=prometheus,
                architecture=architecture,
                start=measurement_start,
                end=measurement_end,
            )
            result["resources"] = resource_metrics
        else:
            result["resources"] = None

        return result

    def collect_from_stats(
        self,
        stats: dict[str, Any] | None,
        phase_summary: dict[str, int] | None,
        prometheus: PrometheusClient | None,
        architecture: str,
        user_count: int,
        run_number: int,
        measurement_start: float,
        measurement_end: float,
    ) -> dict[str, Any]:
        """Collect metrics from pre-computed statistics.

        This method is used when stats are computed in a subprocess (Locust)
        and passed via file to the runner process.

        Args:
            stats: Pre-computed statistics dictionary
            phase_summary: Phase summary dictionary
            prometheus: PrometheusClient for resource queries (optional)
            architecture: Architecture name (monolithic, microservices, triton)
            user_count: Number of concurrent users
            run_number: Run number (1, 2, 3, ...)
            measurement_start: Measurement phase start time (Unix timestamp)
            measurement_end: Measurement phase end time (Unix timestamp)

        Returns:
            Dictionary with all collected metrics
        """
        # Use provided stats or create empty defaults
        if stats is None:
            stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "error_rate_percent": 0.0,
                "throughput_rps": 0.0,
                "duration_seconds": 0.0,
                "client_latency": None,
                "server_latency": None,
            }

        if phase_summary is None:
            phase_summary = {"warmup": 0, "measurement": 0, "cooldown": 0}

        # Build result structure
        result = {
            # Experiment identification
            "architecture": architecture,
            "concurrent_users": user_count,
            "run_number": run_number,
            "timestamp": datetime.now().isoformat(),
            # Request metrics
            "total_requests": stats.get("total_requests", 0),
            "successful_requests": stats.get("successful_requests", 0),
            "failed_requests": stats.get("failed_requests", 0),
            "error_rate_percent": stats.get("error_rate_percent", 0.0),
            "throughput_rps": stats.get("throughput_rps", 0.0),
            "duration_seconds": stats.get("duration_seconds", 0.0),
            # Latency metrics
            "client_latency": stats.get("client_latency"),
            "server_latency": stats.get("server_latency"),
            # Phase summary
            "phase_summary": phase_summary,
        }

        # Query Prometheus for resource metrics
        if prometheus:
            resource_metrics = self._collect_resource_metrics(
                prometheus=prometheus,
                architecture=architecture,
                start=measurement_start,
                end=measurement_end,
            )
            result["resources"] = resource_metrics
        else:
            result["resources"] = None

        return result

    def _collect_resource_metrics(
        self,
        prometheus: PrometheusClient,
        architecture: str,
        start: float,
        end: float,
    ) -> dict[str, Any]:
        """Collect resource metrics from Prometheus.

        Args:
            prometheus: PrometheusClient instance
            architecture: Architecture name
            start: Start time (Unix timestamp)
            end: End time (Unix timestamp)

        Returns:
            Dictionary with CPU and memory metrics per container
        """
        container_names = CONTAINER_NAMES.get(architecture, [])

        if not container_names:
            logger.warning(f"No container names configured for: {architecture}")
            return {}

        # Validate containers exist and are healthy before querying Prometheus
        # ConfigError will propagate up if any container is invalid
        validate_containers(container_names)

        try:
            # Query metrics for all containers
            container_metrics = prometheus.query_container_metrics(
                container_names=container_names,
                start=start,
                end=end,
            )

            # Aggregate across containers
            cpu_values = []
            memory_values = []
            network_rx_values = []
            network_tx_values = []

            for name, metrics in container_metrics.items():
                if metrics["cpu"]["avg_percent"] > 0:
                    cpu_values.append(metrics["cpu"]["avg_percent"])
                if metrics["memory"]["avg_mb"] > 0:
                    memory_values.append(metrics["memory"]["avg_mb"])
                # Network I/O (RX = receive, TX = transmit)
                if metrics["network"]["rx_bytes_per_sec"] > 0:
                    network_rx_values.append(metrics["network"]["rx_bytes_per_sec"])
                if metrics["network"]["tx_bytes_per_sec"] > 0:
                    network_tx_values.append(metrics["network"]["tx_bytes_per_sec"])

            # Calculate totals (sum across all containers)
            result = {
                "containers": container_metrics,
                "totals": {
                    "cpu_avg_percent": (round(sum(cpu_values), 2) if cpu_values else 0.0),
                    "cpu_max_percent": (
                        round(
                            sum(m["cpu"]["max_percent"] for m in container_metrics.values()),
                            2,
                        )
                        if container_metrics
                        else 0.0
                    ),
                    "memory_avg_mb": (round(sum(memory_values), 2) if memory_values else 0.0),
                    "memory_max_mb": (
                        round(
                            sum(m["memory"]["max_mb"] for m in container_metrics.values()),
                            2,
                        )
                        if container_metrics
                        else 0.0
                    ),
                    "network_rx_bytes_per_sec": (
                        round(sum(network_rx_values), 2) if network_rx_values else 0.0
                    ),
                    "network_tx_bytes_per_sec": (
                        round(sum(network_tx_values), 2) if network_tx_values else 0.0
                    ),
                },
            }

            return result

        except Exception as e:
            logger.error(f"Failed to collect resource metrics: {e}")
            return {
                "containers": {},
                "totals": {
                    "cpu_avg_percent": 0.0,
                    "cpu_max_percent": 0.0,
                    "memory_avg_mb": 0.0,
                    "memory_max_mb": 0.0,
                    "network_rx_bytes_per_sec": 0.0,
                    "network_tx_bytes_per_sec": 0.0,
                },
                "error": str(e),
            }

    def to_csv_row(self, result: dict[str, Any]) -> dict[str, Any]:
        """Convert result to flat dictionary for CSV export.

        Args:
            result: Result dictionary from collect()

        Returns:
            Flat dictionary suitable for CSV row
        """
        row = {
            "architecture": result["architecture"],
            "concurrent_users": result["concurrent_users"],
            "run_number": result["run_number"],
            "timestamp": result["timestamp"],
            "total_requests": result["total_requests"],
            "successful_requests": result["successful_requests"],
            "error_rate_percent": result["error_rate_percent"],
            "throughput_rps": result["throughput_rps"],
        }

        # Client latency
        cl = result.get("client_latency") or {}
        row["client_p50_ms"] = cl.get("p50_ms", 0.0)
        row["client_p95_ms"] = cl.get("p95_ms", 0.0)
        row["client_p99_ms"] = cl.get("p99_ms", 0.0)

        # Server latency
        sl = result.get("server_latency") or {}
        row["server_p50_ms"] = sl.get("p50_ms", 0.0)
        row["server_p95_ms"] = sl.get("p95_ms", 0.0)
        row["server_p99_ms"] = sl.get("p99_ms", 0.0)

        # Resource metrics
        resources = result.get("resources") or {}
        totals = resources.get("totals") or {}
        row["cpu_avg_percent"] = totals.get("cpu_avg_percent", 0.0)
        row["cpu_max_percent"] = totals.get("cpu_max_percent", 0.0)
        row["memory_avg_mb"] = totals.get("memory_avg_mb", 0.0)
        row["memory_max_mb"] = totals.get("memory_max_mb", 0.0)
        # Network I/O (RX = receive, TX = transmit)
        row["network_rx_bytes_per_sec"] = totals.get("network_rx_bytes_per_sec", 0.0)
        row["network_tx_bytes_per_sec"] = totals.get("network_tx_bytes_per_sec", 0.0)

        return row
