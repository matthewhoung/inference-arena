"""Load testing framework for Inference Arena.

This package provides a professional, rigorous load testing framework for
the thesis experiment matrix (3 architectures × 7 load levels × 3 runs).

Features:
    - Three-phase protocol (warmup 60s → measurement 180s → cooldown 30s)
    - Server-side timing capture from API responses
    - Thread-safe metrics collection (measurement phase only)
    - Prometheus integration for CPU/memory resource metrics
    - Click-based CLI for experiment orchestration

Usage:
    # Run full experiment matrix
    python -m experiments.runner

    # Run specific configuration
    python -m experiments.runner -a monolithic -u 10 -r 1 --no-docker

    # Dry run (show plan without executing)
    python -m experiments.runner --dry-run

Author: Matthew Hong
Specification Reference: experiment.yaml, .claude/LOADTESTING.md
"""

__version__ = "1.0.0"
__author__ = "Matthew Hong"

# Lazy imports to avoid circular dependencies and heavy imports at package level
# Use: from experiments import TestDataset, MetricsCollector, etc.


def __getattr__(name: str):
    """Lazy import for package-level exports."""
    if name == "ARCHITECTURE_ENDPOINTS":
        from .config import ARCHITECTURE_ENDPOINTS

        return ARCHITECTURE_ENDPOINTS
    elif name == "get_spawn_rate":
        from .config import get_spawn_rate

        return get_spawn_rate
    elif name == "TestDataset":
        from .dataset import TestDataset

        return TestDataset
    elif name == "MetricsCollector":
        from .metrics import MetricsCollector

        return MetricsCollector
    elif name == "RequestMetric":
        from .metrics import RequestMetric

        return RequestMetric
    elif name == "ThreePhaseShape":
        from .shapes import ThreePhaseShape

        return ThreePhaseShape
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ARCHITECTURE_ENDPOINTS",
    "get_spawn_rate",
    "TestDataset",
    "MetricsCollector",
    "RequestMetric",
    "ThreePhaseShape",
]
