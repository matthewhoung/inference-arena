"""Experiment Configuration Module.

This module provides a Python interface to experiment.yaml,
the single source of truth for all experimental parameters.

Usage:
    from shared.config import get_config, get_controlled_variable, get_hypothesis

    # Get full config
    config = get_config()

    # Get specific controlled variable
    threads = get_controlled_variable("onnx_runtime", "intra_op_num_threads")

    # Get model config
    yolo_config = get_model_config("yolov5n")

    # Get hypothesis
    h1a = get_hypothesis("H1a")

Author: Matthew Hong
Specification Reference: experiment.yaml
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator
import yaml

from shared.exceptions import ConfigKeyError
from shared.warnings import warn_once

# =============================================================================
# Constants
# =============================================================================

_POSSIBLE_PATHS = [
    Path(__file__).parent.parent.parent / "experiment.yaml",
    Path.cwd() / "experiment.yaml",
]

# Find first existing path
_CONFIG_PATH = None
for path in _POSSIBLE_PATHS:
    if path.exists():
        _CONFIG_PATH = path
        break

# If not found, default to first option for better error messages
if _CONFIG_PATH is None:
    _CONFIG_PATH = _POSSIBLE_PATHS[0]


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
# Service Ports Configuration
# =============================================================================


class ServicePorts(BaseModel):
    """Service port configuration with validation.

    Provides typed access to service ports from experiment.yaml.
    Validates port ranges and detects duplicates.

    Attributes:
        minio_api: MinIO API port (default 9000)
        minio_console: MinIO console port (default 9001)
        prometheus: Prometheus port (default 9090)
        grafana: Grafana port (default 3000)
        otel_collector: OTel Collector Prometheus metrics port (default 8889)
        otel_health: OTel Collector health check port (default 13133)
    """

    minio_api: int
    minio_console: int
    prometheus: int
    grafana: int
    otel_collector: int
    otel_health: int

    @field_validator("*")
    @classmethod
    def port_in_valid_range(cls, v: int) -> int:
        """Validate port is in valid range 1-65535."""
        if not 1 <= v <= 65535:
            raise ValueError(f"Port {v} not in valid range 1-65535")
        return v


_service_ports_cache: ServicePorts | None = None


def get_service_ports() -> ServicePorts:
    """Get service ports from experiment.yaml with validation.

    Ports are loaded from the services section in experiment.yaml.
    The function validates port ranges and detects duplicate port
    assignments, logging a warning (W002) if duplicates are found.

    Raises:
        ConfigKeyError: If required port config missing
        ValueError: If port validation fails (invalid port range)

    Returns:
        ServicePorts model with all configured ports

    Example:
        >>> ports = get_service_ports()
        >>> ports.minio_api
        9000
        >>> ports.prometheus
        9090
    """
    global _service_ports_cache
    if _service_ports_cache is not None:
        return _service_ports_cache

    config = get_config()
    try:
        services = config["services"]
        ports = ServicePorts(
            minio_api=services["minio"]["api_port"],
            minio_console=services["minio"]["console_port"],
            prometheus=services["prometheus"]["port"],
            grafana=services["grafana"]["port"],
            otel_collector=services["otel_collector"]["port"],
            otel_health=services["otel_collector"]["health_port"],
        )
    except KeyError as e:
        raise ConfigKeyError(
            f"Missing service port configuration in experiment.yaml: {e}. "
            "Ensure services section is defined with all required ports."
        ) from e

    # Detect duplicate ports
    seen: dict[int, str] = {}
    for name, port in [
        ("minio_api", ports.minio_api),
        ("minio_console", ports.minio_console),
        ("prometheus", ports.prometheus),
        ("grafana", ports.grafana),
        ("otel_collector", ports.otel_collector),
        ("otel_health", ports.otel_health),
    ]:
        if port in seen:
            warn_once(
                "W002",
                f"Duplicate port {port} configured for {name} and {seen[port]}",
                "Check services section in experiment.yaml",
            )
        seen[port] = name

    _service_ports_cache = ports
    return ports


def clear_service_ports_cache() -> None:
    """Clear the service ports cache. Primarily for testing.

    Clears the cached ServicePorts instance, causing the next call
    to get_service_ports() to reload from configuration.
    """
    global _service_ports_cache
    _service_ports_cache = None


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
# Validation
# =============================================================================


def validate_config() -> list[str]:
    """Validate the experiment configuration.

    Returns:
        List of validation error messages (empty if valid)

    Example:
        >>> errors = validate_config()
        >>> if errors:
        ...     print("Validation failed:", errors)
    """
    errors = []

    try:
        config = get_config()
    except FileNotFoundError as e:
        return [f"Configuration file not found: {e}"]
    except yaml.YAMLError as e:
        return [f"YAML parsing error: {e}"]

    # Check required sections
    required_sections = [
        "metadata",
        "research_questions",
        "hypotheses",
        "independent_variables",
        "controlled_variables",
        "infrastructure",
    ]

    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")

    # Check controlled variables
    cv = config.get("controlled_variables", {})
    required_cv = [
        "models",
        "preprocessing",
        "resources",
        "onnx_runtime",
        "dataset",
        "load_testing",
    ]

    for section in required_cv:
        if section not in cv:
            errors.append(f"Missing controlled_variables section: {section}")

    # Check models
    models = cv.get("models", {})
    for model_name in ["yolov5n", "mobilenetv2"]:
        if model_name not in models:
            errors.append(f"Missing model configuration: {model_name}")
        else:
            model = models[model_name]
            for field in ["opset_version", "input", "output"]:
                if field not in model:
                    errors.append(f"Model {model_name} missing field: {field}")

    # Check ONNX runtime config
    onnx = cv.get("onnx_runtime", {})
    for field in ["intra_op_num_threads", "inter_op_num_threads"]:
        if field not in onnx:
            errors.append(f"Missing onnx_runtime field: {field}")

    # Check hypotheses
    hypotheses = config.get("hypotheses", {})
    for h_id, h_config in hypotheses.items():
        # Required fields for all hypotheses
        required_fields = ["category", "statement", "rationale"]
        for field in required_fields:
            if field not in h_config:
                errors.append(f"Hypothesis {h_id} missing required field: {field}")

        # Must have either 'testable_prediction' or 'prediction'
        if "testable_prediction" not in h_config and "prediction" not in h_config:
            errors.append(f"Hypothesis {h_id} missing testable_prediction or prediction")

    return errors


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
