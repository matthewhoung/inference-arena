"""Experiment configuration module.

This module bridges the load testing framework to experiment.yaml,
the single source of truth for all experimental parameters.

Usage:
    from experiments.config import (
        ARCHITECTURE_ENDPOINTS,
        COMPOSE_FILES,
        get_spawn_rate,
        get_phase_durations,
    )

Author: Matthew Hong
Specification Reference: experiment.yaml
"""

import os
from pathlib import Path

# Import from shared config (single source of truth)
from shared.config import get_concurrent_user_levels, get_container_names, get_load_testing_config

# =============================================================================
# Architecture Configuration
# =============================================================================

ARCHITECTURE_ENDPOINTS: dict[str, str] = {
    "monolithic": "http://localhost:8100",
    "microservices": "http://localhost:8200",
    "triton": "http://localhost:8300",
}

# Project root for relative path resolution
PROJECT_ROOT = Path(__file__).parent.parent

COMPOSE_FILES: dict[str, Path] = {
    "monolithic": PROJECT_ROOT / "architectures" / "monolithic" / "docker-compose.yml",
    "microservices": PROJECT_ROOT / "architectures" / "microservices" / "docker-compose.yml",
    "triton": PROJECT_ROOT / "architectures" / "triton" / "docker-compose.yml",
}

# =============================================================================
# Container Names (from experiment.yaml single source of truth)
# =============================================================================


def get_architecture_container_names(architecture: str) -> list[str]:
    """Get full container names for an architecture from experiment.yaml.

    These are the actual Docker container names used for Prometheus queries
    with the container_name label.

    Args:
        architecture: Architecture name ("monolithic", "microservices", "triton")

    Returns:
        List of full container names (e.g., ["inference-arena-monolithic"])
    """
    try:
        return get_container_names(architecture)
    except Exception:
        # Fallback to hardcoded values if config unavailable
        _fallback = {
            "monolithic": ["inference-arena-monolithic"],
            "microservices": ["inference-arena-detection", "inference-arena-classification"],
            "triton": ["inference-arena-triton-server", "inference-arena-triton-gateway"],
        }
        return _fallback.get(architecture, [])


# Legacy compatibility dict - use get_architecture_container_names() instead
# Maps architecture to list of container names for Prometheus queries
CONTAINER_NAMES: dict[str, list[str]] = {
    arch: get_architecture_container_names(arch)
    for arch in ["monolithic", "microservices", "triton"]
}

# =============================================================================
# Load Testing Configuration
# =============================================================================

# Spawn rates for each user level (users/second)
# From experiment.yaml / METHODOLOGY.md
SPAWN_RATES: dict[int, int] = {
    1: 1,
    5: 2,
    10: 3,
    25: 5,
    50: 10,
    75: 15,
    100: 20,
}

# Default durations (fallback if experiment.yaml not available)
DEFAULT_WARMUP_SECONDS = 60
DEFAULT_MEASUREMENT_SECONDS = 180
DEFAULT_COOLDOWN_SECONDS = 30


def get_spawn_rate(user_count: int) -> int:
    """Get spawn rate for a given user count.

    Args:
        user_count: Number of concurrent users

    Returns:
        Spawn rate (users per second)

    Raises:
        ValueError: If user count not in predefined levels
    """
    if user_count not in SPAWN_RATES:
        valid_levels = list(SPAWN_RATES.keys())
        raise ValueError(f"Invalid user count: {user_count}. Valid levels: {valid_levels}")
    return SPAWN_RATES[user_count]


def get_phase_durations() -> dict[str, int]:
    """Get phase durations from experiment.yaml.

    Returns:
        Dictionary with warmup, measurement, cooldown durations in seconds
    """
    try:
        config = get_load_testing_config()
        phases = config.get("phases", {})
        return {
            "warmup": phases.get("warmup", {}).get("duration_seconds", DEFAULT_WARMUP_SECONDS),
            "measurement": phases.get("measurement", {}).get(
                "duration_seconds", DEFAULT_MEASUREMENT_SECONDS
            ),
            "cooldown": phases.get("cooldown", {}).get(
                "duration_seconds", DEFAULT_COOLDOWN_SECONDS
            ),
        }
    except Exception:
        # Fallback to defaults if config not available
        return {
            "warmup": DEFAULT_WARMUP_SECONDS,
            "measurement": DEFAULT_MEASUREMENT_SECONDS,
            "cooldown": DEFAULT_COOLDOWN_SECONDS,
        }


def get_total_duration() -> int:
    """Get total test duration (warmup + measurement + cooldown).

    Returns:
        Total duration in seconds
    """
    durations = get_phase_durations()
    return durations["warmup"] + durations["measurement"] + durations["cooldown"]


def get_runs_per_configuration() -> int:
    """Get number of runs per configuration from experiment.yaml.

    Returns:
        Number of runs (default: 3)
    """
    try:
        config = get_load_testing_config()
        return config.get("runs_per_configuration", 3)
    except Exception:
        return 3


def get_user_levels() -> list[int]:
    """Get list of concurrent user levels.

    Returns:
        List of user counts [1, 5, 10, 25, 50, 75, 100]
    """
    try:
        return get_concurrent_user_levels()
    except Exception:
        return list(SPAWN_RATES.keys())


def get_architectures() -> list[str]:
    """Get list of architecture names.

    Returns:
        List of architecture identifiers
    """
    return list(ARCHITECTURE_ENDPOINTS.keys())


# =============================================================================
# Environment Configuration
# =============================================================================

# Prometheus URL (default with env var override)
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")

# Triton batching toggle
# When True, Triton uses batched model variants with dynamic batching enabled
# Set via environment: TRITON_BATCHING=true
TRITON_BATCHING_ENABLED = os.environ.get("TRITON_BATCHING", "false").lower() == "true"

# Results output directory
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "experiment"


def get_output_dir() -> Path:
    """Get results output directory.

    Returns:
        Path to output directory
    """
    env_dir = os.environ.get("EXPERIMENT_OUTPUT_DIR")
    if env_dir:
        return Path(env_dir)
    return DEFAULT_OUTPUT_DIR


def is_triton_batching_enabled() -> bool:
    """Check if Triton batching mode is enabled.

    When enabled, Triton uses batched model variants (yolov5n_batched,
    mobilenetv2_batched) with dynamic batching configuration.

    Returns:
        True if TRITON_BATCHING=true in environment
    """
    return TRITON_BATCHING_ENABLED


def get_triton_model_suffix() -> str:
    """Get model name suffix based on batching configuration.

    Returns:
        "_batched" if batching enabled, empty string otherwise
    """
    return "_batched" if TRITON_BATCHING_ENABLED else ""
