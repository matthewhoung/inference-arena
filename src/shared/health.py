"""Health check utilities with exponential backoff.

This module provides utilities for waiting on service health checks
with configurable exponential backoff.

Example:
    from shared.health import wait_for_healthy
    import requests

    def check_api():
        try:
            r = requests.get("http://localhost:8000/health", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    wait_for_healthy("api-service", check_api)

Author: Matthew Hong
"""

import logging
import time
from collections.abc import Callable

from shared.exceptions import InferenceArenaError

logger = logging.getLogger(__name__)


class HealthCheckTimeoutError(InferenceArenaError):
    """Raised when a service does not become healthy within the timeout."""

    pass


def wait_for_healthy(
    name: str,
    check_fn: Callable[[], bool],
    *,
    initial_delay: float = 1.0,
    max_wait: float = 30.0,
    backoff_multiplier: float = 2.0,
    max_interval: float = 5.0,
) -> None:
    """Wait for a service to become healthy with exponential backoff.

    Args:
        name: Service name for logging and error messages
        check_fn: Callable that returns True if service is healthy, False otherwise.
                  Should handle its own exceptions and return False on error.
        initial_delay: Seconds to wait before first health check (default: 1.0)
        max_wait: Maximum total seconds to wait (default: 30.0)
        backoff_multiplier: Multiplier for backoff interval (default: 2.0)
        max_interval: Maximum seconds between checks (default: 5.0)

    Raises:
        HealthCheckTimeoutError: If service not healthy within max_wait

    Example:
        wait_for_healthy(
            "minio",
            lambda: requests.get("http://localhost:9000/minio/health/ready").ok,
            max_wait=60.0
        )
    """
    logger.info(f"Waiting for {name} to become healthy (max {max_wait}s)")

    # Initial delay before first check
    time.sleep(initial_delay)

    start_time = time.monotonic()
    interval = initial_delay
    last_error: str | None = None

    while (elapsed := time.monotonic() - start_time) < max_wait:
        try:
            if check_fn():
                logger.info(f"{name} is healthy after {elapsed:.1f}s")
                return
            last_error = "health check returned False"
        except Exception as e:
            last_error = str(e)

        # Calculate next interval with backoff, capped at max_interval
        interval = min(interval * backoff_multiplier, max_interval)

        # Don't sleep past max_wait
        remaining = max_wait - elapsed
        sleep_time = min(interval, remaining)

        if sleep_time > 0:
            time.sleep(sleep_time)

    # Timeout reached
    error_detail = f": {last_error}" if last_error else ""
    raise HealthCheckTimeoutError(
        f"{name} not healthy after {max_wait}s{error_detail}"
    )
