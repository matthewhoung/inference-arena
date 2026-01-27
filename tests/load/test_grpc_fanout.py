"""Load tests for gRPC parallel classification fan-out.

This module tests the parallel gRPC fan-out behavior implemented in
ClassificationClient.classify_parallel() using asyncio.gather().

These tests verify:
1. Correctness: All parallel classification results are valid
2. Timing: Parallel execution is faster than sequential
3. Sustained load: Multiple iterations complete without degradation

Tests require a running Classification gRPC service (part of the
microservices architecture). They are skipped by default; run with
--load flag to execute.

Author: Matthew Hong
"""

import logging
import os
import time
from typing import Any

import numpy as np
import psutil
import pytest

# Import the ClassificationClient for testing
# Note: This import requires the architectures path in pytest pythonpath
from microservices.detection.app.grpc_client import ClassificationClient

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default gRPC endpoint for Classification service
DEFAULT_GRPC_ENDPOINT = "localhost:8201"

# Test configuration
NUM_CROPS = 5  # Number of crops for parallel fan-out (typical detection count)
CROP_SIZE = 224  # MobileNet input size
NUM_ITERATIONS = 10  # Iterations for sustained load test


# =============================================================================
# Helper Functions
# =============================================================================


def create_test_crops(n: int) -> list[np.ndarray]:
    """Generate N random RGB crops for testing.

    Args:
        n: Number of crops to generate

    Returns:
        List of RGB uint8 numpy arrays with shape [224, 224, 3]
    """
    rng = np.random.default_rng(42)
    return [rng.integers(0, 256, (CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8) for _ in range(n)]


def create_test_boxes(n: int) -> list[dict[str, Any]]:
    """Generate N mock detection boxes for testing.

    Args:
        n: Number of boxes to generate

    Returns:
        List of detection box dictionaries
    """
    return [
        {
            "x1": float(i * 100),
            "y1": float(i * 100),
            "x2": float(i * 100 + 50),
            "y2": float(i * 100 + 50),
            "confidence": 0.9,
            "class_id": 0,
        }
        for i in range(n)
    ]


def capture_resources() -> dict[str, float]:
    """Capture current system resource metrics.

    Returns:
        Dictionary with cpu_percent and memory_mb
    """
    process = psutil.Process()
    return {
        "cpu_percent": process.cpu_percent(),
        "memory_mb": process.memory_info().rss / (1024 * 1024),
    }


# =============================================================================
# Skip Condition
# =============================================================================


def grpc_service_available() -> bool:
    """Check if gRPC Classification service is available.

    Returns:
        True if service responds, False otherwise
    """
    import grpc

    endpoint = os.environ.get("CLASSIFICATION_GRPC_URL", DEFAULT_GRPC_ENDPOINT)
    try:
        channel = grpc.insecure_channel(endpoint)
        grpc.channel_ready_future(channel).result(timeout=2)
        channel.close()
        return True
    except grpc.FutureTimeoutError:
        return False
    except Exception:
        return False


# =============================================================================
# Test Class
# =============================================================================


@pytest.mark.load
@pytest.mark.integration
@pytest.mark.skipif(
    not grpc_service_available(),
    reason="Classification gRPC service not available",
)
class TestGRPCFanout:
    """Load tests for gRPC parallel classification fan-out.

    These tests verify the behavior of ClassificationClient.classify_parallel()
    which uses asyncio.gather() to execute multiple classification requests
    concurrently.

    The parallel fan-out is critical for H1b hypothesis:
    - Microservices P99 should be competitive with monolithic despite network overhead
    - asyncio.gather enables parallel calls that mask gRPC latency
    """

    @pytest.fixture
    async def client(self) -> ClassificationClient:
        """Create and connect a ClassificationClient."""
        endpoint = os.environ.get("CLASSIFICATION_GRPC_URL", DEFAULT_GRPC_ENDPOINT)
        client = ClassificationClient(endpoint)
        await client.connect()
        yield client
        await client.close()

    @pytest.fixture
    def test_crops(self) -> list[np.ndarray]:
        """Generate test crops for fan-out."""
        return create_test_crops(NUM_CROPS)

    @pytest.fixture
    def test_boxes(self) -> list[dict[str, Any]]:
        """Generate test boxes for fan-out."""
        return create_test_boxes(NUM_CROPS)

    async def test_parallel_fanout_correctness(
        self,
        client: ClassificationClient,
        test_crops: list[np.ndarray],
        test_boxes: list[dict[str, Any]],
    ) -> None:
        """Test that parallel fan-out returns correct results for all crops.

        This test verifies:
        1. All N crops return classification results
        2. Each result has a valid class_id (0-999 for ImageNet)
        3. No exceptions during parallel execution
        """
        # Execute parallel classification
        request_id = "test_correctness"
        responses = await client.classify_parallel(request_id, test_crops, test_boxes)

        # Verify all results returned
        assert len(responses) == NUM_CROPS, f"Expected {NUM_CROPS} responses, got {len(responses)}"

        # Verify each result has valid class_id (ImageNet has 1000 classes)
        for i, response in enumerate(responses):
            assert hasattr(response, "class_id"), f"Response {i} missing class_id"
            assert (
                0 <= response.class_id < 1000
            ), f"Response {i} has invalid class_id: {response.class_id}"
            logger.info(
                f"Crop {i}: class_id={response.class_id}, "
                f"confidence={getattr(response, 'confidence', 'N/A')}"
            )

    async def test_parallel_fanout_timing(
        self,
        client: ClassificationClient,
        test_crops: list[np.ndarray],
        test_boxes: list[dict[str, Any]],
    ) -> None:
        """Test that parallel execution is faster than sequential.

        This test measures:
        1. Sequential execution time (one by one)
        2. Parallel execution time (asyncio.gather)
        3. Resource metrics (CPU, memory)

        Note: Timing is logged for analysis, not assertion. Parallel should
        generally be faster, but the exact speedup depends on network latency
        and server load.
        """
        # Capture baseline resources
        resources_before = capture_resources()

        # Sequential execution
        t_seq_start = time.perf_counter()
        for i, (crop, box) in enumerate(zip(test_crops, test_boxes, strict=True)):
            await client.classify(f"seq_{i}", crop, box)
        t_seq_end = time.perf_counter()
        sequential_ms = (t_seq_end - t_seq_start) * 1000

        # Parallel execution
        t_par_start = time.perf_counter()
        await client.classify_parallel("parallel", test_crops, test_boxes)
        t_par_end = time.perf_counter()
        parallel_ms = (t_par_end - t_par_start) * 1000

        # Capture resources after
        resources_after = capture_resources()

        # Log timing comparison (not assertion - per CONTEXT.md)
        speedup = sequential_ms / parallel_ms if parallel_ms > 0 else 0
        logger.info(
            f"Timing comparison for {NUM_CROPS} crops:\n"
            f"  Sequential: {sequential_ms:.2f}ms\n"
            f"  Parallel:   {parallel_ms:.2f}ms\n"
            f"  Speedup:    {speedup:.2f}x"
        )

        # Log resource metrics (for analysis, not assertion)
        logger.info(
            f"Resource metrics:\n"
            f"  CPU before: {resources_before['cpu_percent']:.1f}%\n"
            f"  CPU after:  {resources_after['cpu_percent']:.1f}%\n"
            f"  Memory:     {resources_after['memory_mb']:.1f}MB"
        )

        # Verify timing was captured (basic sanity check)
        assert sequential_ms > 0, "Sequential timing not captured"
        assert parallel_ms > 0, "Parallel timing not captured"

    async def test_sustained_fanout_load(
        self,
        client: ClassificationClient,
        test_crops: list[np.ndarray],
        test_boxes: list[dict[str, Any]],
    ) -> None:
        """Test sustained parallel fan-out load over multiple iterations.

        This test verifies:
        1. Multiple iterations of parallel classification complete successfully
        2. No degradation or failures over time
        3. Resource metrics captured for analysis

        Note: Resource metrics are logged for analysis, not pass/fail criteria.
        """
        # Capture baseline resources
        resources_before = capture_resources()

        iteration_times_ms: list[float] = []

        for iteration in range(NUM_ITERATIONS):
            t_start = time.perf_counter()
            responses = await client.classify_parallel(
                f"sustained_{iteration}", test_crops, test_boxes
            )
            t_end = time.perf_counter()

            # Verify all results returned
            assert len(responses) == NUM_CROPS, (
                f"Iteration {iteration}: expected {NUM_CROPS} responses, " f"got {len(responses)}"
            )

            iteration_time_ms = (t_end - t_start) * 1000
            iteration_times_ms.append(iteration_time_ms)
            logger.info(f"Iteration {iteration + 1}/{NUM_ITERATIONS}: {iteration_time_ms:.2f}ms")

        # Capture resources after sustained load
        resources_after = capture_resources()

        # Calculate timing statistics
        avg_time_ms = sum(iteration_times_ms) / len(iteration_times_ms)
        min_time_ms = min(iteration_times_ms)
        max_time_ms = max(iteration_times_ms)

        # Log summary (for analysis, not assertion)
        logger.info(
            f"Sustained load summary ({NUM_ITERATIONS} iterations, {NUM_CROPS} crops each):\n"
            f"  Avg: {avg_time_ms:.2f}ms\n"
            f"  Min: {min_time_ms:.2f}ms\n"
            f"  Max: {max_time_ms:.2f}ms\n"
            f"  Memory delta: {resources_after['memory_mb'] - resources_before['memory_mb']:.1f}MB"
        )

        # Verify all iterations completed (implicit from loop completing)
        assert len(iteration_times_ms) == NUM_ITERATIONS
