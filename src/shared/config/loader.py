"""Configuration Loader Module.

This module provides functions for loading and accessing experiment configuration
from experiment.yaml - the single source of truth for all experimental parameters.

Functions:
    get_config: Load and cache the experiment configuration
    reload_config: Force reload of configuration (clears cache)
    get_controlled_variable: Get a specific controlled variable value
    get_controlled_variables: Get all controlled variables for a section
    get_model_config: Get configuration for a specific model
    get_model_names: Get list of all model names
    get_hypothesis: Get a specific hypothesis by ID
    get_hypotheses_by_category: Get all hypotheses for a category
    get_hypotheses: Get all hypotheses
    get_infrastructure_config: Get infrastructure configuration
    get_minio_config: Get MinIO configuration
    get_triton_config: Get Triton Inference Server configuration
    get_triton_batching_config: Get Triton dynamic batching configuration
    get_load_testing_config: Get load testing protocol configuration
    get_concurrent_user_levels: Get concurrent user levels for experiments
    get_container_names: Get container names for architectures
    get_container_name: Get container name for a specific architecture
    get_monitoring_config: Get monitoring configuration
    get_metadata: Get experiment metadata
    get_spec_version: Get specification version

Author: Matthew Hong
Specification Reference: experiment.yaml
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# =============================================================================
# Constants
# =============================================================================

_POSSIBLE_PATHS = [
    Path(__file__).parent.parent.parent / "experiment.yaml",
    Path.cwd() / "experiment.yaml",
]


def _find_config_path() -> Path:
    """Find the configuration file path, defaulting to first option if not found."""
    for path in _POSSIBLE_PATHS:
        if path.exists():
            return path
    # If not found, default to first option for better error messages
    return _POSSIBLE_PATHS[0]


_CONFIG_PATH: Path = _find_config_path()


# =============================================================================
# Configuration Loading
# =============================================================================


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Load and cache the experiment configuration.

    Returns:
        Complete experiment configuration dictionary

    Raises:
        FileNotFoundError: If experiment.yaml not found
        yaml.YAMLError: If YAML parsing fails

    Example:
        >>> config = get_config()
        >>> config["metadata"]["title"]
        'Characterizing ML Serving Architectures in CPU-Constrained Environments'
    """
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Experiment configuration not found: {_CONFIG_PATH}\n"
            f"Expected location: {_CONFIG_PATH.absolute()}"
        )

    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def reload_config() -> dict[str, Any]:
    """Force reload of configuration (clears cache).

    Useful for testing or when config file is modified at runtime.

    Returns:
        Freshly loaded configuration dictionary
    """
    get_config.cache_clear()
    return get_config()


# =============================================================================
# Controlled Variables Access
# =============================================================================


def get_controlled_variable(section: str, key: str) -> Any:
    """Get a controlled variable value by section and key.

    Args:
        section: Top-level section name (e.g., "onnx_runtime", "resources")
        key: Key within the section (e.g., "intra_op_num_threads")

    Returns:
        The controlled variable value

    Raises:
        KeyError: If section or key not found

    Example:
        >>> get_controlled_variable("onnx_runtime", "intra_op_num_threads")
        2
        >>> get_controlled_variable("resources", "vcpu_per_container")
        2
    """
    config = get_config()
    controlled = config.get("controlled_variables", {})

    if section not in controlled:
        available = list(controlled.keys())
        raise KeyError(
            f"Section '{section}' not found in controlled_variables. "
            f"Available sections: {available}"
        )

    section_data = controlled[section]

    if key not in section_data:
        available = list(section_data.keys())
        raise KeyError(
            f"Key '{key}' not found in controlled_variables.{section}. "
            f"Available keys: {available}"
        )

    return section_data[key]


def get_controlled_variables(section: str) -> dict[str, Any]:
    """Get all controlled variables for a section.

    Args:
        section: Section name (e.g., "onnx_runtime", "models")

    Returns:
        Dictionary of all variables in the section

    Example:
        >>> onnx_config = get_controlled_variables("onnx_runtime")
        >>> onnx_config["intra_op_num_threads"]
        2
    """
    config = get_config()
    controlled = config.get("controlled_variables", {})

    if section not in controlled:
        available = list(controlled.keys())
        raise KeyError(f"Section '{section}' not found. Available: {available}")

    return controlled[section]


# =============================================================================
# Model Configuration
# =============================================================================


def get_model_config(model_name: str) -> dict[str, Any]:
    """Get configuration for a specific model.

    Args:
        model_name: Model identifier ("yolov5n" or "mobilenetv2")

    Returns:
        Model configuration dictionary

    Example:
        >>> yolo = get_model_config("yolov5n")
        >>> yolo["input"]["shape"]
        [1, 3, 640, 640]
        >>> yolo["opset_version"]
        17
    """
    models = get_controlled_variable("models", model_name)
    return models


def get_model_names() -> list[str]:
    """Get list of all model names.

    Returns:
        List of model identifiers

    Example:
        >>> get_model_names()
        ['yolov5n', 'mobilenetv2']
    """
    models = get_controlled_variables("models")
    return list(models.keys())


# =============================================================================
# Hypothesis Access
# =============================================================================


def get_hypothesis(hypothesis_id: str) -> dict[str, Any]:
    """Get a specific hypothesis by ID.

    Args:
        hypothesis_id: Hypothesis identifier (e.g., "H1a", "H2b")

    Returns:
        Hypothesis configuration dictionary

    Example:
        >>> h1a = get_hypothesis("H1a")
        >>> h1a["statement"]
        'Monolithic architecture exhibits lowest P50 and P99 latency...'
    """
    config = get_config()
    hypotheses = config.get("hypotheses", {})

    if hypothesis_id not in hypotheses:
        available = list(hypotheses.keys())
        raise KeyError(f"Hypothesis '{hypothesis_id}' not found. Available: {available}")

    return hypotheses[hypothesis_id]


def get_hypotheses_by_category(category: str) -> dict[str, dict[str, Any]]:
    """Get all hypotheses for a category.

    Args:
        category: Category name ("performance", "resource_efficiency", "operational_complexity")

    Returns:
        Dictionary of hypothesis_id -> hypothesis_config

    Example:
        >>> perf = get_hypotheses_by_category("performance")
        >>> list(perf.keys())
        ['H1a', 'H1b', 'H1c', 'H1d']
    """
    config = get_config()
    hypotheses = config.get("hypotheses", {})

    return {
        h_id: h_config
        for h_id, h_config in hypotheses.items()
        if h_config.get("category") == category
    }


def get_hypotheses() -> dict[str, dict[str, Any]]:
    """Get all hypotheses.

    Returns:
        Dictionary of hypothesis_id -> hypothesis_config

    Example:
        >>> hypotheses = get_hypotheses()
        >>> list(hypotheses.keys())
        ['H1a', 'H1b', 'H1c', 'H1d', 'H2a', 'H2b', 'H3a', 'H3b']
    """
    config = get_config()
    return config.get("hypotheses", {})


# =============================================================================
# Infrastructure Configuration
# =============================================================================


def get_infrastructure_config(service: str | None = None) -> dict[str, Any]:
    """Get infrastructure configuration.

    Args:
        service: Optional service name (e.g., "minio", "networks")
                 If None, returns full infrastructure config

    Returns:
        Infrastructure configuration dictionary

    Example:
        >>> minio = get_infrastructure_config("minio")
        >>> minio["bucket"]
        'models'
    """
    config = get_config()
    infra = config.get("infrastructure", {})

    if service is None:
        return infra

    if service not in infra:
        available = list(infra.keys())
        raise KeyError(f"Service '{service}' not found. Available: {available}")

    return infra[service]


def get_minio_config() -> dict[str, Any]:
    """Get MinIO configuration.

    Returns:
        MinIO configuration dictionary

    Example:
        >>> minio = get_minio_config()
        >>> minio["bucket"]
        'models'
        >>> minio["endpoint"]
        'minio:9000'
    """
    return get_infrastructure_config("minio")


# =============================================================================
# Triton Configuration
# =============================================================================


def get_triton_config() -> dict[str, Any]:
    """Get Triton Inference Server configuration.

    Returns:
        Triton configuration dictionary

    Example:
        >>> triton = get_triton_config()
        >>> triton["model_repository"]
        's3://minio:9000/models'
    """
    config = get_config()
    return config.get("triton", {})


def get_triton_batching_config() -> dict[str, Any]:
    """Get Triton dynamic batching configuration.

    Returns:
        Batching configuration dictionary with keys:
        - enabled: bool (default False)
        - max_batch_size: int (default 8)
        - preferred_batch_size: list[int]
        - max_queue_delay_microseconds: int

    Example:
        >>> batching = get_triton_batching_config()
        >>> batching["enabled"]
        False
        >>> batching["max_batch_size"]
        8
    """
    triton = get_triton_config()
    return triton.get(
        "batching",
        {
            "enabled": False,
            "max_batch_size": 8,
            "preferred_batch_size": [4, 8],
            "max_queue_delay_microseconds": 5000,
        },
    )


# =============================================================================
# Load Testing Configuration
# =============================================================================


def get_load_testing_config() -> dict[str, Any]:
    """Get load testing protocol configuration.

    Returns:
        Load testing configuration dictionary

    Example:
        >>> lt = get_load_testing_config()
        >>> lt["phases"]["warmup"]["duration_seconds"]
        60
    """
    return get_controlled_variables("load_testing")


def get_concurrent_user_levels() -> list[int]:
    """Get the list of concurrent user levels for experiments.

    Returns:
        List of concurrent user counts

    Example:
        >>> get_concurrent_user_levels()
        [1, 5, 10, 25, 50, 75, 100]
    """
    config = get_config()
    iv = config.get("independent_variables", {})
    return iv.get("concurrent_users", {}).get("levels", [])


# =============================================================================
# Container Names (Single Source of Truth)
# =============================================================================


def get_container_names(architecture: str | None = None) -> dict[str, list[str]] | list[str]:
    """Get container names for architectures from experiment.yaml.

    Container names are the single source of truth for:
    - Prometheus metric queries (using container.name label)
    - Grafana dashboard queries
    - prometheus_client.py resource metrics

    Args:
        architecture: Optional architecture name ("monolithic", "microservices", "triton").
                     If None, returns all architectures.

    Returns:
        If architecture specified: List of container names for that architecture
        If None: Dictionary mapping architecture -> list of container names

    Raises:
        KeyError: If architecture not found

    Example:
        >>> get_container_names("monolithic")
        ['inference-arena-monolithic']
        >>> get_container_names("microservices")
        ['inference-arena-detection', 'inference-arena-classification']
        >>> get_container_names()
        {'monolithic': [...], 'microservices': [...], 'triton': [...]}
    """
    container_names = get_controlled_variables("container_names")

    if architecture is None:
        return container_names

    if architecture not in container_names:
        available = list(container_names.keys())
        raise KeyError(f"Architecture '{architecture}' not found. Available: {available}")

    return container_names[architecture]


def get_monitoring_config() -> dict[str, Any]:
    """Get monitoring configuration including OTel Collector settings.

    Returns:
        Monitoring configuration dictionary

    Example:
        >>> monitoring = get_monitoring_config()
        >>> monitoring["otel_collector"]["port"]
        8889
    """
    return get_controlled_variables("monitoring")


# =============================================================================
# Metadata
# =============================================================================


def get_metadata() -> dict[str, Any]:
    """Get experiment metadata.

    Returns:
        Metadata dictionary

    Example:
        >>> meta = get_metadata()
        >>> meta["author"]
        'Matthew Hong'
    """
    config = get_config()
    return config.get("metadata", {})


def get_spec_version() -> str:
    """Get specification version.

    Returns:
        Version string

    Example:
        >>> get_spec_version()
        '1.0.0'
    """
    return get_metadata().get("spec_version", "0.0.0")


# =============================================================================
# Download Configuration
# =============================================================================


def get_downloads_config() -> dict[str, Any]:
    """Get download configuration.

    Returns:
        Downloads configuration dictionary with keys:
        - max_concurrent: int (default 3)
        - timeout: int (default 300)

    Example:
        >>> downloads = get_downloads_config()
        >>> downloads["max_concurrent"]
        3
    """
    config = get_config()
    return config.get(
        "downloads",
        {
            "max_concurrent": 3,
            "timeout": 300,
        },
    )


def get_download_max_concurrent() -> int:
    """Get maximum concurrent downloads.

    Returns:
        Maximum number of parallel downloads (default: 3)

    Example:
        >>> get_download_max_concurrent()
        3
    """
    return get_downloads_config().get("max_concurrent", 3)


def get_download_timeout() -> int:
    """Get download timeout in seconds.

    Returns:
        Timeout per download in seconds (default: 300)

    Example:
        >>> get_download_timeout()
        300
    """
    return get_downloads_config().get("timeout", 300)


# =============================================================================
# Module Initialization Check
# =============================================================================


def _check_config_exists() -> None:
    """Warn if config file is missing (for development)."""
    if not _CONFIG_PATH.exists():
        import warnings

        warnings.warn(
            f"experiment.yaml not found at {_CONFIG_PATH}. " f"Some functionality may not work.",
            UserWarning,
            stacklevel=2,
        )


# Run check on import (non-blocking)
# Only catch expected exceptions; let unexpected exceptions propagate
try:
    _check_config_exists()
except (FileNotFoundError, PermissionError):
    # Expected: config file missing or unreadable during import
    # _check_config_exists already emits a warning, no action needed
    pass
