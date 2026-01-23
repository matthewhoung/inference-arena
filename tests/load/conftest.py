"""Load test configuration and fixtures.

Load tests are integration tests that require external services (Triton server)
and verify behavior under concurrent load. They are skipped by default to avoid
slow CI runs.

Usage:
    # Run all tests except load tests (default)
    pytest tests/

    # Run only load tests
    pytest tests/load/ --load

    # Run load tests with verbose output
    pytest tests/load/ --load -v

Requirements:
    - Triton server running at localhost:8001 (or TRITON_URL env var)
    - psutil package for resource metric capture

Notes:
    - Resource metrics (CPU, memory) are captured for analysis, not assertions
    - Tests fail immediately on any failure (no flaky tolerance)
    - Timing comparisons are logged for analysis, not pass/fail criteria
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --load command line option."""
    parser.addoption(
        "--load",
        action="store_true",
        default=False,
        help="run load tests (skipped by default)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register load marker."""
    config.addinivalue_line(
        "markers",
        "load: marks tests as load tests (run with --load flag)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip load tests unless --load flag provided.

    Only skips tests that have the explicit @pytest.mark.load marker.
    Tests without this marker (like unit tests using mocks) run normally.
    """
    if not config.getoption("--load"):
        skip_load = pytest.mark.skip(reason="need --load option to run load tests")
        for item in items:
            # Check for explicit @pytest.mark.load marker, not just "load" in keywords
            # (keywords includes directory names which would skip ALL tests in tests/load/)
            if list(item.iter_markers(name="load")):
                item.add_marker(skip_load)
