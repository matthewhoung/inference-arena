"""Triton Configuration Generator.

This module generates config.pbtxt files for NVIDIA Triton Inference Server.
Configuration values are sourced from experiment.yaml to ensure consistency.

Usage:
    from shared.triton.config import generate_config_pbtxt

    config = generate_config_pbtxt("yolov5n")
    print(config)

Author: Matthew Hong
Specification Reference: experiment.yaml, Ch3 Methodology §3.4.3
"""

from pathlib import Path

from shared.config import (
    get_model_config,
    get_model_names,
    get_triton_batching_config,
    get_triton_config,
)

# =============================================================================
# Constants
# =============================================================================

# Map Python/numpy dtypes to Triton data types
DTYPE_MAP = {
    "float32": "TYPE_FP32",
    "float16": "TYPE_FP16",
    "int32": "TYPE_INT32",
    "int64": "TYPE_INT64",
    "int8": "TYPE_INT8",
    "uint8": "TYPE_UINT8",
    "bool": "TYPE_BOOL",
    "string": "TYPE_STRING",
}


# =============================================================================
# Config Generation
# =============================================================================


def generate_config_pbtxt(model_name: str, batching_enabled: bool = False) -> str:
    """Generate config.pbtxt content for a model.

    Args:
        model_name: Model identifier ("yolov5n" or "mobilenetv2")
        batching_enabled: If True, enables dynamic batching with config from experiment.yaml.
                         When enabled, dims exclude batch dimension (Triton adds it).
                         When disabled, max_batch_size=0 and dims include batch.

    Returns:
        config.pbtxt content as string

    Example:
        >>> config = generate_config_pbtxt("yolov5n")
        >>> 'platform: "onnxruntime_onnx"' in config
        True
        >>> 'max_batch_size: 0' in config
        True
        >>> config_batched = generate_config_pbtxt("yolov5n", batching_enabled=True)
        >>> 'dynamic_batching' in config_batched
        True
    """
    # Strip "_batched" suffix to get base model name for config lookup
    base_model_name = model_name.replace("_batched", "")
    model_config = get_model_config(base_model_name)
    triton_config = get_triton_config()
    batching_config = get_triton_batching_config()

    # Get input/output specs
    input_spec = model_config["input"]
    output_spec = model_config["output"]

    # Get instance group config
    instance_group = triton_config.get("instance_group", {})
    instance_count = instance_group.get("count", 1)
    instance_kind = instance_group.get("kind", "KIND_CPU")

    # Get threading parameters
    params = triton_config.get("parameters", {})
    intra_threads = params.get("intra_op_thread_count", "2")
    inter_threads = params.get("inter_op_thread_count", "1")

    # Handle batching configuration
    if batching_enabled:
        # With batching: dims EXCLUDE batch dimension (Triton adds it automatically)
        # Shape [1, 3, 640, 640] becomes [3, 640, 640]
        max_batch_size = batching_config.get("max_batch_size", 8)
        input_dims = _format_dims(input_spec["shape"][1:])  # Remove batch dim

        # For YOLOv5 dynamic export, the output has dynamic anchors dimension
        # The model output is [-1, 84, -1] so we need [-1] for the last dim
        output_shape = output_spec["shape"][1:]  # Remove batch dim
        if base_model_name == "yolov5n":
            # YOLOv5 dynamic model has variable anchors dimension (depends on input size)
            # Use -1 for the last dimension to indicate dynamic
            output_shape = list(output_shape)
            output_shape[-1] = -1
        output_dims = _format_dims(output_shape)
    else:
        # Without batching: dims INCLUDE batch dimension
        # max_batch_size=0 means batching disabled
        max_batch_size = 0
        input_dims = _format_dims(input_spec["shape"])
        output_dims = _format_dims(output_spec["shape"])

    # Get data types
    input_dtype = DTYPE_MAP.get(input_spec.get("dtype", "float32"), "TYPE_FP32")
    output_dtype = DTYPE_MAP.get(output_spec.get("dtype", "float32"), "TYPE_FP32")

    # Build batching mode indicator for header
    batching_mode = "ENABLED" if batching_enabled else "DISABLED"

    config = f"""# =============================================================================
# Triton Model Configuration: {model_name}
# =============================================================================
# Auto-generated from experiment.yaml
# DO NOT EDIT MANUALLY - regenerate using: python scripts/models/generate-pbtxt.py
#
# Source: experiment.yaml controlled_variables.models.{base_model_name}
# Batching: {batching_mode}
# =============================================================================

name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}

input [{{
  name: "{input_spec["name"]}"
  data_type: {input_dtype}
  dims: {input_dims}
}}]

output [{{
  name: "{output_spec["name"]}"
  data_type: {output_dtype}
  dims: {output_dims}
}}]

instance_group [{{
  count: {instance_count}
  kind: {instance_kind}
}}]

parameters [
  {{
    key: "intra_op_thread_count"
    value: {{ string_value: "{intra_threads}" }}
  }},
  {{
    key: "inter_op_thread_count"
    value: {{ string_value: "{inter_threads}" }}
  }}
]
"""

    # Add dynamic batching section if enabled
    if batching_enabled:
        preferred_batch_sizes = batching_config.get("preferred_batch_size", [4, 8])
        max_queue_delay = batching_config.get("max_queue_delay_microseconds", 5000)
        preferred_str = ", ".join(str(s) for s in preferred_batch_sizes)

        config += f"""
dynamic_batching {{
  preferred_batch_size: [ {preferred_str} ]
  max_queue_delay_microseconds: {max_queue_delay}
}}
"""

    return config


def _format_dims(shape: list[int]) -> str:
    """Format shape list as Triton dims string.

    Args:
        shape: List of dimensions [1, 3, 640, 640]

    Returns:
        Formatted string "[ 1, 3, 640, 640 ]"
    """
    return "[ " + ", ".join(str(d) for d in shape) + " ]"


def generate_all_configs(batching_enabled: bool = False) -> dict[str, str]:
    """Generate config.pbtxt for all models.

    Args:
        batching_enabled: If True, generates batching-enabled configs with "_batched" suffix.

    Returns:
        Dictionary mapping model_name -> config_content

    Example:
        >>> configs = generate_all_configs()
        >>> list(configs.keys())
        ['yolov5n', 'mobilenetv2']
        >>> batched_configs = generate_all_configs(batching_enabled=True)
        >>> list(batched_configs.keys())
        ['yolov5n_batched', 'mobilenetv2_batched']
    """
    result = {}
    for base_model_name in get_model_names():
        if batching_enabled:
            model_name = f"{base_model_name}_batched"
        else:
            model_name = base_model_name
        result[model_name] = generate_config_pbtxt(model_name, batching_enabled)
    return result


def save_config_pbtxt(model_name: str, output_dir: Path, batching_enabled: bool = False) -> Path:
    """Generate and save config.pbtxt to disk.

    Args:
        model_name: Model identifier (e.g., "yolov5n" or "yolov5n_batched")
        output_dir: Directory to save config (model_name/config.pbtxt)
        batching_enabled: If True, generates batching-enabled config

    Returns:
        Path to saved config file
    """
    config_content = generate_config_pbtxt(model_name, batching_enabled)

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    config_path = model_dir / "config.pbtxt"
    config_path.write_text(config_content)

    return config_path


# =============================================================================
# Validation
# =============================================================================


def validate_config_pbtxt(config_content: str) -> list[str]:
    """Validate config.pbtxt content.

    Args:
        config_content: Config file content

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    required_fields = [
        "name:",
        "platform:",
        "input [",
        "output [",
        "instance_group [",
    ]

    for field in required_fields:
        if field not in config_content:
            errors.append(f"Missing required field: {field}")

    # Check platform is ONNX Runtime
    if "onnxruntime_onnx" not in config_content:
        errors.append("Platform should be 'onnxruntime_onnx' for ONNX models")

    return errors
