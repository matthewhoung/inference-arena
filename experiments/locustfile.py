"""Locust load testing for Inference Arena.

This module provides the InferenceUser class for load testing the three
ML serving architectures (Monolithic, Microservices, Triton).

Features:
    - Preloaded test images (100 COCO images, ~16MB in memory)
    - Three-phase protocol via ThreePhaseShape
    - Server-side timing capture from response JSON
    - Thread-safe metrics collection
    - Zero wait time for maximum throughput

Test Images:
    Loaded from the curated thesis test set (100 COCO images with 3-5
    detections each, μ=4, σ≈0.8).

Usage:
    # Via runner (recommended):
    python -m experiments.runner -a monolithic -u 10 -r 1 --no-docker

    # Direct Locust (manual testing):
    locust -f experiments/locustfile.py --host=http://localhost:8100 \
           --headless -u 10 -t 270s

    # With Web UI (debugging):
    locust -f experiments/locustfile.py --host=http://localhost:8100

Architecture Ports:
    - Monolithic:    http://localhost:8100
    - Microservices: http://localhost:8200
    - Triton:        http://localhost:8300

Author: Matthew Hong
Specification Reference: experiment.yaml, METHODOLOGY.md
"""

import logging
import time
from typing import TYPE_CHECKING

from locust import HttpUser, constant, events, task

from experiments.dataset import TestDataset, get_dataset, reset_dataset
from experiments.metrics import MetricsCollector, RequestMetric, get_collector, reset_collector
from experiments.shapes import ThreePhaseShape, set_shape

if TYPE_CHECKING:
    from locust.env import Environment

logger = logging.getLogger(__name__)

# =============================================================================
# Global State
# =============================================================================
# These are initialized once at test start and shared across all users.

# Test dataset (preloaded images)
dataset: TestDataset | None = None

# Metrics collector (thread-safe)
metrics: MetricsCollector | None = None

# Load test shape (for phase awareness)
shape: ThreePhaseShape | None = None


# =============================================================================
# Locust User Definition
# =============================================================================


class InferenceUser(HttpUser):
    """Simulates a user sending inference requests.

    Each user continuously sends random test images to the /predict endpoint
    and records metrics for both client-side and server-side latency.

    The user automatically integrates with ThreePhaseShape to tag metrics
    with the current phase (warmup, measurement, cooldown).

    Attributes:
        wait_time: Time between requests (0 for maximum throughput)
    """

    # Maximum throughput - no wait between requests
    wait_time = constant(0)

    def on_start(self) -> None:
        """Called when a simulated user starts.

        Verifies that global state (dataset, metrics, shape) is initialized.
        """
        global dataset, metrics, shape

        if dataset is None:
            raise RuntimeError(
                "Dataset not initialized. Run via experiments.runner or "
                "ensure @events.test_start initializes global state."
            )

        if metrics is None:
            raise RuntimeError("Metrics collector not initialized.")

        logger.debug(f"User started, dataset has {len(dataset)} images")

    @task
    def predict(self) -> None:
        """Send prediction request with random test image.

        This task:
        1. Gets a random image from the preloaded dataset
        2. Sends POST /predict with the image
        3. Captures server-side timing from response JSON
        4. Records metrics with current phase tag
        """
        global dataset, metrics, shape

        # Get random image (preloaded, no disk I/O)
        filename, image_bytes = dataset.get_random_image()

        # Record start time for client latency (use perf_counter for monotonic timing)
        start_time = time.perf_counter()

        # Send request with catch_response for custom success/failure handling
        with self.client.post(
            "/predict",
            files={"file": (filename, image_bytes, "image/jpeg")},
            catch_response=True,
        ) as response:
            # Calculate client-side latency (monotonic, cannot go negative)
            client_latency_ms = (time.perf_counter() - start_time) * 1000

            # Get current phase from shape
            phase = shape.get_current_phase() if shape else "measurement"

            # Process response
            if response.status_code == 200:
                try:
                    data = response.json()

                    # Extract server-side timing
                    timing = data.get("timing", {})
                    server_total_ms = timing.get("total_ms")

                    # Extract detection count
                    detections = len(data.get("detections", []))

                    # Mark as success
                    response.success()

                    # Record metric
                    metrics.record(
                        RequestMetric(
                            timestamp=time.time(),
                            phase=phase,
                            client_latency_ms=client_latency_ms,
                            server_total_ms=server_total_ms,
                            success=True,
                            detections=detections,
                        )
                    )

                except Exception as e:
                    # JSON parsing failed
                    response.failure(f"Failed to parse response: {e}")
                    metrics.record(
                        RequestMetric(
                            timestamp=time.time(),
                            phase=phase,
                            client_latency_ms=client_latency_ms,
                            server_total_ms=None,
                            success=False,
                            error=f"JSON parse error: {e}",
                        )
                    )
            else:
                # HTTP error
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_detail = response.text[:200]
                    error_msg = f"{error_msg}: {error_detail}"
                except Exception:
                    pass

                response.failure(error_msg)
                metrics.record(
                    RequestMetric(
                        timestamp=time.time(),
                        phase=phase,
                        client_latency_ms=client_latency_ms,
                        server_total_ms=None,
                        success=False,
                        error=error_msg,
                    )
                )


# =============================================================================
# Event Handlers
# =============================================================================


@events.test_start.add_listener
def on_test_start(environment: "Environment", **kwargs) -> None:
    """Initialize global state when test starts.

    This handler:
    1. Initializes the test dataset (preloads images)
    2. Creates a fresh metrics collector
    3. Sets up the shape reference for phase tracking
    """
    global dataset, metrics, shape

    logger.info("=" * 60)
    logger.info("Inference Arena - Load Test Starting")
    logger.info("=" * 60)

    # Initialize dataset
    logger.info("Loading test dataset...")
    reset_dataset()
    dataset = get_dataset()
    logger.info(
        f"Dataset loaded: {len(dataset)} images, " f"{dataset.get_memory_usage_mb():.2f} MB"
    )

    # Initialize metrics collector
    reset_collector()
    metrics = get_collector()
    logger.info("Metrics collector initialized")

    # Get shape reference (if using ThreePhaseShape)
    if hasattr(environment, "shape_class") and environment.shape_class:
        shape = environment.shape_class
        set_shape(shape)
        logger.info(
            f"Shape: ThreePhaseShape "
            f"(warmup={shape.warmup}s, measurement={shape.measurement}s, "
            f"cooldown={shape.cooldown}s)"
        )
    else:
        # Create a default shape for phase tracking
        shape = ThreePhaseShape()
        set_shape(shape)
        logger.info("Using default ThreePhaseShape for phase tracking")

    # Log test configuration
    logger.info(f"Host: {environment.host}")
    if hasattr(environment, "parsed_options") and environment.parsed_options:
        opts = environment.parsed_options
        logger.info(f"Users: {getattr(opts, 'num_users', 'N/A')}")
        logger.info(f"Duration: {getattr(opts, 'run_time', 'N/A')}")

    logger.info("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment: "Environment", **kwargs) -> None:
    """Log summary when test stops and export metrics to file.

    Prints phase summary and measurement statistics.
    Exports metrics to JSON file for runner to collect.
    """
    import json
    import os

    global metrics, shape

    logger.info("\n" + "=" * 60)
    logger.info("Load Test Complete")
    logger.info("=" * 60)

    if metrics:
        # Log phase summary
        phase_summary = metrics.get_phase_summary()
        logger.info(
            f"Requests by phase: "
            f"warmup={phase_summary['warmup']}, "
            f"measurement={phase_summary['measurement']}, "
            f"cooldown={phase_summary['cooldown']}"
        )

        # Log measurement statistics
        stats = metrics.calculate_statistics()
        if stats["total_requests"] > 0:
            logger.info("Measurement Phase Statistics:")
            logger.info(f"  Total requests: {stats['total_requests']}")
            logger.info(f"  Successful: {stats['successful_requests']}")
            logger.info(f"  Failed: {stats['failed_requests']}")
            logger.info(f"  Error rate: {stats['error_rate_percent']}%")
            logger.info(f"  Throughput: {stats['throughput_rps']} RPS")

            if stats["client_latency"]:
                cl = stats["client_latency"]
                logger.info(
                    f"  Client latency: "
                    f"P50={cl['p50_ms']}ms, P95={cl['p95_ms']}ms, P99={cl['p99_ms']}ms"
                )

            if stats["server_latency"]:
                sl = stats["server_latency"]
                logger.info(
                    f"  Server latency: "
                    f"P50={sl['p50_ms']}ms, P95={sl['p95_ms']}ms, P99={sl['p99_ms']}ms"
                )
        else:
            logger.warning("No measurement phase requests recorded")

        # Export metrics to file for runner to collect
        metrics_file = os.environ.get("LOCUST_METRICS_FILE")
        if metrics_file:
            export_data = {
                "stats": stats,
                "phase_summary": phase_summary,
            }
            with open(metrics_file, "w") as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"Metrics exported to: {metrics_file}")

    logger.info("=" * 60 + "\n")


@events.request.add_listener
def on_request(
    request_type: str,
    name: str,
    response_time: float,
    response_length: int,
    exception: Exception | None,
    **kwargs,
) -> None:
    """Log individual requests for debugging (optional).

    This handler is disabled by default to avoid log spam.
    Enable by setting LOG_REQUESTS=1 environment variable.
    """
    import os

    if os.environ.get("LOG_REQUESTS") == "1":
        status = "FAIL" if exception else "OK"
        logger.debug(
            f"[{status}] {request_type} {name} - " f"{response_time:.1f}ms, {response_length} bytes"
        )
