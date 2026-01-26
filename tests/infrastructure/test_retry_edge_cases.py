"""
Integration tests for Triton client retry edge cases.

These tests verify the retry logic in wait_for_healthy() handles edge cases correctly:
- Timeout failure mode
- Connection refused failure mode
- Max retries exhausted
- Exponential backoff timing

Tests use a real configurable HTTP server (not mocks) to verify behavior.
"""

import time

import pytest
import requests

from shared.health import HealthCheckTimeoutError, wait_for_healthy

from .conftest import run_health_server


@pytest.mark.integration
def test_retry_timeout_raises_health_check_timeout_error() -> None:
    """Test that timeout in health check raises HealthCheckTimeoutError.

    Configures server with delay longer than max_wait to trigger timeout.
    """
    # Server delays 5 seconds, but we only wait 1 second total
    with run_health_server(delay=5.0) as (server, url):

        def check_fn() -> bool:
            try:
                r = requests.get(f"{url}/health", timeout=0.1)
                return r.status_code == 200
            except requests.RequestException:
                return False

        with pytest.raises(HealthCheckTimeoutError) as exc_info:
            wait_for_healthy(
                "test-service",
                check_fn,
                initial_delay=0.1,
                max_wait=0.5,
                backoff_multiplier=2.0,
                max_interval=0.5,
            )

        # Verify error message mentions timeout context
        assert "test-service" in str(exc_info.value)
        assert "0.5s" in str(exc_info.value)


@pytest.mark.integration
def test_retry_connection_refused(unavailable_port: int) -> None:
    """Test that connection refused is handled gracefully.

    Uses unavailable_port fixture (no server listening) to test connection errors.
    """

    def check_fn() -> bool:
        try:
            r = requests.get(f"http://localhost:{unavailable_port}/health", timeout=0.5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    with pytest.raises(HealthCheckTimeoutError) as exc_info:
        wait_for_healthy(
            "connection-test",
            check_fn,
            initial_delay=0.05,
            max_wait=0.3,
            backoff_multiplier=2.0,
            max_interval=0.2,
        )

    # Verify error indicates we waited and failed
    assert "connection-test" in str(exc_info.value)
    assert "not healthy after" in str(exc_info.value)


@pytest.mark.integration
def test_retry_max_attempts_exhausted() -> None:
    """Test that max retries exhausted includes last error in message.

    Configures server to fail more times than retries allow.
    """
    # Server fails 100 times (effectively forever)
    with run_health_server(fail_count=100, fail_with=503) as (server, url):

        def check_fn() -> bool:
            try:
                r = requests.get(f"{url}/health", timeout=1.0)
                return r.status_code == 200
            except requests.RequestException:
                return False

        with pytest.raises(HealthCheckTimeoutError) as exc_info:
            wait_for_healthy(
                "max-retry-test",
                check_fn,
                initial_delay=0.05,
                max_wait=0.3,
                backoff_multiplier=2.0,
                max_interval=0.2,
            )

        # Verify error message includes the last error indication
        error_msg = str(exc_info.value)
        assert "max-retry-test" in error_msg
        assert "not healthy after" in error_msg
        # last_error should be captured (check returned False)
        assert "health check returned False" in error_msg


@pytest.mark.integration
def test_retry_exponential_backoff_timing() -> None:
    """Test that retry intervals follow exponential backoff pattern.

    Tracks time between retries and verifies they increase exponentially.
    """
    request_times: list[float] = []

    # Server fails first 3 requests, then succeeds
    with run_health_server(fail_count=3, fail_with=503) as (server, url):

        def check_fn() -> bool:
            request_times.append(time.monotonic())
            try:
                r = requests.get(f"{url}/health", timeout=1.0)
                return r.status_code == 200
            except requests.RequestException:
                return False

        # Use fast backoff for testing: 0.1s initial, 2x multiplier
        wait_for_healthy(
            "backoff-test",
            check_fn,
            initial_delay=0.1,
            max_wait=5.0,
            backoff_multiplier=2.0,
            max_interval=2.0,
        )

        # Should have 4 requests: 3 failures + 1 success
        assert len(request_times) >= 4, f"Expected at least 4 requests, got {len(request_times)}"

        # Calculate intervals between requests
        intervals = [
            request_times[i + 1] - request_times[i] for i in range(len(request_times) - 1)
        ]

        # Verify exponential backoff: each interval should be roughly 2x the previous
        # Allow 50% tolerance for timing variations
        for i in range(1, len(intervals) - 1):
            ratio = intervals[i] / intervals[i - 1]
            # Ratio should be approximately 2.0 (backoff_multiplier)
            # Allow generous tolerance (0.8 to 3.0) due to timing jitter
            assert 0.8 <= ratio <= 3.0, (
                f"Backoff ratio {ratio:.2f} at interval {i} outside tolerance. "
                f"Intervals: {intervals}"
            )


@pytest.mark.integration
def test_retry_succeeds_after_transient_failures() -> None:
    """Test that retry succeeds after transient failures.

    Configures server to fail first 2 requests, then succeed.
    """
    with run_health_server(fail_count=2, fail_with=503) as (server, url):

        def check_fn() -> bool:
            try:
                r = requests.get(f"{url}/health", timeout=1.0)
                return r.status_code == 200
            except requests.RequestException:
                return False

        # Should succeed without raising exception
        wait_for_healthy(
            "transient-test",
            check_fn,
            initial_delay=0.05,
            max_wait=5.0,
            backoff_multiplier=2.0,
            max_interval=0.5,
        )

        # Verify server received expected number of requests (2 failures + 1 success)
        assert server.request_count >= 3, (
            f"Expected at least 3 requests, got {server.request_count}"
        )


@pytest.mark.integration
def test_retry_with_500_error_code() -> None:
    """Test retry handles HTTP 500 errors correctly."""
    with run_health_server(fail_count=1, fail_with=500) as (server, url):

        def check_fn() -> bool:
            try:
                r = requests.get(f"{url}/health", timeout=1.0)
                return r.status_code == 200
            except requests.RequestException:
                return False

        # Should succeed after one 500 error
        wait_for_healthy(
            "500-error-test",
            check_fn,
            initial_delay=0.05,
            max_wait=2.0,
            backoff_multiplier=2.0,
            max_interval=0.5,
        )

        # Server should have received at least 2 requests
        assert server.request_count >= 2


@pytest.mark.integration
def test_retry_immediate_success() -> None:
    """Test that immediate success returns without retry loops."""
    with run_health_server(fail_count=0) as (server, url):
        start_time = time.monotonic()

        def check_fn() -> bool:
            try:
                r = requests.get(f"{url}/health", timeout=1.0)
                return r.status_code == 200
            except requests.RequestException:
                return False

        wait_for_healthy(
            "immediate-success",
            check_fn,
            initial_delay=0.05,
            max_wait=5.0,
            backoff_multiplier=2.0,
            max_interval=0.5,
        )

        elapsed = time.monotonic() - start_time

        # Should complete quickly (just initial delay + one check)
        assert elapsed < 1.0, f"Immediate success took too long: {elapsed:.2f}s"
        # Server should receive exactly 1 request
        assert server.request_count == 1


@pytest.mark.integration
def test_retry_check_function_exception_captured() -> None:
    """Test that exceptions from check function are captured in error message."""
    call_count = 0

    def failing_check_fn() -> bool:
        nonlocal call_count
        call_count += 1
        raise ValueError("Custom test exception message")

    with pytest.raises(HealthCheckTimeoutError) as exc_info:
        wait_for_healthy(
            "exception-test",
            failing_check_fn,
            initial_delay=0.05,
            max_wait=0.2,
            backoff_multiplier=2.0,
            max_interval=0.2,
        )

    # Verify the exception message includes the last error
    error_msg = str(exc_info.value)
    assert "exception-test" in error_msg
    assert "Custom test exception message" in error_msg
    # Verify multiple retries happened
    assert call_count >= 2
