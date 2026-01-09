"""Thread-safe metrics collection for load testing.

This module provides the MetricsCollector class that captures per-request
metrics during load tests, filtering to measurement phase only for statistics.

Key Features:
    - Thread-safe recording from concurrent Locust users
    - Phase-aware filtering (warmup/measurement/cooldown)
    - Server-side timing capture from API responses
    - Percentile calculations (P50, P95, P99)

Usage:
    from experiments.metrics import MetricsCollector, RequestMetric

    collector = MetricsCollector()
    collector.record(RequestMetric(
        timestamp=time.time(),
        phase="measurement",
        client_latency_ms=150.5,
        server_total_ms=145.2,
        success=True
    ))
    stats = collector.calculate_statistics()

Author: Matthew Hong
Specification Reference: experiment.yaml, METHODOLOGY.md
"""

import statistics
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class RequestMetric:
    """Container for per-request metrics.

    Attributes:
        timestamp: Unix timestamp when request completed
        phase: Test phase ("warmup", "measurement", "cooldown")
        client_latency_ms: Round-trip time measured by Locust client
        server_total_ms: Server processing time from response JSON
        success: Whether request succeeded (HTTP 200)
        error: Error message if request failed
        detections: Number of detections in response (optional)
    """

    timestamp: float
    phase: str
    client_latency_ms: float
    server_total_ms: float | None
    success: bool
    error: str | None = None
    detections: int | None = None


@dataclass
class LatencyStats:
    """Latency statistics for a set of requests.

    Attributes:
        count: Number of requests
        min_ms: Minimum latency
        max_ms: Maximum latency
        mean_ms: Mean latency
        p50_ms: 50th percentile (median)
        p95_ms: 95th percentile
        p99_ms: 99th percentile
    """

    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


class MetricsCollector:
    """Thread-safe metrics collector for load testing.

    This class collects per-request metrics from concurrent Locust users
    and provides statistics calculation filtered to measurement phase only.

    Thread Safety:
        All public methods are thread-safe via internal locking.

    Example:
        >>> collector = MetricsCollector()
        >>> collector.record(RequestMetric(...))
        >>> stats = collector.calculate_statistics()
        >>> stats["throughput_rps"]
        45.2
    """

    def __init__(self):
        """Initialize the metrics collector."""
        self._metrics: list[RequestMetric] = []
        self._lock = threading.Lock()
        self._start_time: float | None = None
        self._end_time: float | None = None

    def record(self, metric: RequestMetric) -> None:
        """Record a request metric.

        Thread-safe method to record metrics from concurrent users.

        Args:
            metric: RequestMetric instance to record
        """
        with self._lock:
            self._metrics.append(metric)

            # Track time bounds
            if self._start_time is None:
                self._start_time = metric.timestamp
            self._end_time = metric.timestamp

    def get_all_metrics(self) -> list[RequestMetric]:
        """Get all recorded metrics.

        Returns:
            Copy of all metrics list
        """
        with self._lock:
            return list(self._metrics)

    def get_measurement_metrics(self) -> list[RequestMetric]:
        """Get metrics from measurement phase only.

        Returns:
            List of metrics where phase == "measurement"
        """
        with self._lock:
            return [m for m in self._metrics if m.phase == "measurement"]

    def get_metrics_by_phase(self, phase: str) -> list[RequestMetric]:
        """Get metrics for a specific phase.

        Args:
            phase: Phase name ("warmup", "measurement", "cooldown")

        Returns:
            List of metrics for the specified phase
        """
        with self._lock:
            return [m for m in self._metrics if m.phase == phase]

    def calculate_latency_stats(self, latencies: list[float]) -> LatencyStats | None:
        """Calculate latency statistics from a list of values.

        Args:
            latencies: List of latency values in milliseconds

        Returns:
            LatencyStats or None if list is empty
        """
        if not latencies:
            return None

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return LatencyStats(
            count=n,
            min_ms=sorted_latencies[0],
            max_ms=sorted_latencies[-1],
            mean_ms=statistics.mean(sorted_latencies),
            p50_ms=self._percentile(sorted_latencies, 50),
            p95_ms=self._percentile(sorted_latencies, 95),
            p99_ms=self._percentile(sorted_latencies, 99),
        )

    @staticmethod
    def _percentile(sorted_data: list[float], percentile: float) -> float:
        """Calculate percentile from sorted data.

        Args:
            sorted_data: Sorted list of values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        if not sorted_data:
            return 0.0

        n = len(sorted_data)
        k = (n - 1) * percentile / 100
        f = int(k)
        c = f + 1 if f + 1 < n else f

        if f == c:
            return sorted_data[f]

        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)

    def calculate_statistics(self) -> dict[str, Any]:
        """Calculate comprehensive statistics for measurement phase.

        This method filters to measurement phase metrics and calculates:
        - Throughput (requests per second)
        - Error rate (percentage of failed requests)
        - Client latency percentiles (P50, P95, P99)
        - Server latency percentiles (P50, P95, P99)

        Returns:
            Dictionary with all calculated statistics
        """
        metrics = self.get_measurement_metrics()

        if not metrics:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "error_rate_percent": 0.0,
                "throughput_rps": 0.0,
                "client_latency": None,
                "server_latency": None,
                "duration_seconds": 0.0,
            }

        # Count requests
        total = len(metrics)
        successful = sum(1 for m in metrics if m.success)
        failed = total - successful

        # Calculate duration from first to last request
        timestamps = [m.timestamp for m in metrics]
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 1.0

        # Calculate throughput
        throughput = successful / duration if duration > 0 else 0.0

        # Error rate
        error_rate = (failed / total) * 100 if total > 0 else 0.0

        # Client latency stats
        client_latencies = [m.client_latency_ms for m in metrics if m.success]
        client_stats = self.calculate_latency_stats(client_latencies)

        # Server latency stats (only for successful requests with server timing)
        server_latencies = [
            m.server_total_ms for m in metrics if m.success and m.server_total_ms is not None
        ]
        server_stats = self.calculate_latency_stats(server_latencies)

        return {
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": failed,
            "error_rate_percent": round(error_rate, 2),
            "throughput_rps": round(throughput, 2),
            "duration_seconds": round(duration, 2),
            "client_latency": self._stats_to_dict(client_stats),
            "server_latency": self._stats_to_dict(server_stats),
        }

    @staticmethod
    def _stats_to_dict(stats: LatencyStats | None) -> dict[str, float] | None:
        """Convert LatencyStats to dictionary.

        Args:
            stats: LatencyStats instance or None

        Returns:
            Dictionary representation or None
        """
        if stats is None:
            return None

        return {
            "count": stats.count,
            "min_ms": round(stats.min_ms, 2),
            "max_ms": round(stats.max_ms, 2),
            "mean_ms": round(stats.mean_ms, 2),
            "p50_ms": round(stats.p50_ms, 2),
            "p95_ms": round(stats.p95_ms, 2),
            "p99_ms": round(stats.p99_ms, 2),
        }

    def get_phase_summary(self) -> dict[str, int]:
        """Get count of requests per phase.

        Returns:
            Dictionary mapping phase name to request count
        """
        with self._lock:
            summary = {"warmup": 0, "measurement": 0, "cooldown": 0}
            for m in self._metrics:
                if m.phase in summary:
                    summary[m.phase] += 1
            return summary

    def clear(self) -> None:
        """Clear all recorded metrics.

        Thread-safe method to reset the collector.
        """
        with self._lock:
            self._metrics.clear()
            self._start_time = None
            self._end_time = None

    def __len__(self) -> int:
        """Return total number of recorded metrics."""
        with self._lock:
            return len(self._metrics)


# Module-level singleton for shared access
_collector_instance: MetricsCollector | None = None


def get_collector() -> MetricsCollector:
    """Get or create the shared metrics collector.

    Returns:
        Shared MetricsCollector instance
    """
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = MetricsCollector()
    return _collector_instance


def reset_collector() -> None:
    """Reset the shared metrics collector.

    Creates a new instance, discarding all previously recorded metrics.
    """
    global _collector_instance
    _collector_instance = MetricsCollector()
