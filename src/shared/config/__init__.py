"""Experiment Configuration Package.

This package provides a Python interface to experiment.yaml,
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

    # Get service ports with validation
    ports = get_service_ports()

Modules:
    loader: Configuration loading and access functions
    models: Pydantic models for typed configuration
    validators: Configuration validation functions

Author: Matthew Hong
Specification Reference: experiment.yaml
"""

from .loader import (
    get_concurrent_user_levels,
    get_config,
    get_container_names,
    get_controlled_variable,
    get_controlled_variables,
    get_hypotheses,
    get_hypotheses_by_category,
    get_hypothesis,
    get_infrastructure_config,
    get_load_testing_config,
    get_metadata,
    get_minio_config,
    get_model_config,
    get_model_names,
    get_monitoring_config,
    get_spec_version,
    get_triton_batching_config,
    get_triton_config,
    reload_config,
)
from .models import (
    ServicePorts,
    clear_service_ports_cache,
    get_service_ports,
)
from .validators import (
    validate_config,
)

__all__ = [
    # loader.py exports
    "get_config",
    "reload_config",
    "get_controlled_variable",
    "get_controlled_variables",
    "get_model_config",
    "get_model_names",
    "get_hypothesis",
    "get_hypotheses_by_category",
    "get_hypotheses",
    "get_infrastructure_config",
    "get_minio_config",
    "get_triton_config",
    "get_triton_batching_config",
    "get_load_testing_config",
    "get_concurrent_user_levels",
    "get_container_names",
    "get_monitoring_config",
    "get_metadata",
    "get_spec_version",
    # models.py exports
    "ServicePorts",
    "get_service_ports",
    "clear_service_ports_cache",
    # validators.py exports
    "validate_config",
]
