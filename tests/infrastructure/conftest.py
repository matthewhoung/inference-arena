"""
Pytest Configuration for Infrastructure Tests

This module provides shared fixtures and configuration for infrastructure tests.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require Docker)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests by default unless explicitly requested."""
    if config.getoption("-m") and "integration" in config.getoption("-m"):
        # Integration tests explicitly requested
        return
    
    skip_integration = pytest.mark.skip(
        reason="Integration tests skipped by default. Run with: pytest -m integration"
    )
    
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
