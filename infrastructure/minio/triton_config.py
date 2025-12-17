"""
Triton Configuration Generator

This module generates config.pbtxt files for NVIDIA Triton Inference Server.
Configuration values are sourced from experiment.yaml to ensure consistency.

Usage:
    from infrastructure.minio.triton_config import generate_config_pbtxt
    
    config = generate_config_pbtxt("yolov5n")
    print(config)

Author: Matthew Hong
Specification Reference: experiment.yaml, Ch3 Methodology §3.4.3
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shared.config import get_model_config, get_triton_config, get_controlled_variable


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

def generate_config_pbtxt(model_name: str) -> str:
    """
    Generate config.pbtxt content for a model.
    
    Args:
        model_name: Model identifier ("yolov5n" or "mobilenetv2")
        
    Returns:
        config.pbtxt content as string
        
    Example:
        >>> config = generate_config_pbtxt("yolov5n")
        >>> "platform: \"onnxruntime_onnx\"" in config
        True
    """
    model_config = get_model_config(model_name)
    triton_config = get_triton_config()
    
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
    
    # Format dimensions
    input_dims = _format_dims(input_spec["shape"])
    output_dims = _format_dims(output_spec["shape"])
    
    # Get data types
    input_dtype = DTYPE_MAP.get(input_spec.get("dtype", "float32"), "TYPE_FP32")
    output_dtype = DTYPE_MAP.get(output_spec.get("dtype", "float32"), "TYPE_FP32")
    
    config = f'''# =============================================================================
# Triton Model Configuration: {model_name}
# =============================================================================
# Auto-generated from experiment.yaml
# DO NOT EDIT MANUALLY - regenerate using init_models.py
#
# Source: experiment.yaml controlled_variables.models.{model_name}
# =============================================================================

name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 0

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
'''
    
    return config


def _format_dims(shape: List[int]) -> str:
    """
    Format shape list as Triton dims string.
    
    Args:
        shape: List of dimensions [1, 3, 640, 640]
        
    Returns:
        Formatted string "[ 1, 3, 640, 640 ]"
    """
    return "[ " + ", ".join(str(d) for d in shape) + " ]"


def generate_all_configs() -> Dict[str, str]:
    """
    Generate config.pbtxt for all models.
    
    Returns:
        Dictionary mapping model_name -> config_content
        
    Example:
        >>> configs = generate_all_configs()
        >>> list(configs.keys())
        ['yolov5n', 'mobilenetv2']
    """
    from shared.config import get_model_names
    
    return {
        model_name: generate_config_pbtxt(model_name)
        for model_name in get_model_names()
    }


def save_config_pbtxt(model_name: str, output_dir: Path) -> Path:
    """
    Generate and save config.pbtxt to disk.
    
    Args:
        model_name: Model identifier
        output_dir: Directory to save config (model_name/config.pbtxt)
        
    Returns:
        Path to saved config file
    """
    config_content = generate_config_pbtxt(model_name)
    
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = model_dir / "config.pbtxt"
    config_path.write_text(config_content)
    
    return config_path


# =============================================================================
# Validation
# =============================================================================

def validate_config_pbtxt(config_content: str) -> List[str]:
    """
    Validate config.pbtxt content.
    
    Args:
        config_content: Config file content
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    required_fields = [
        'name:',
        'platform:',
        'input [',
        'output [',
        'instance_group [',
    ]
    
    for field in required_fields:
        if field not in config_content:
            errors.append(f"Missing required field: {field}")
    
    # Check platform is ONNX Runtime
    if 'onnxruntime_onnx' not in config_content:
        errors.append("Platform should be 'onnxruntime_onnx' for ONNX models")
    
    return errors


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Triton config.pbtxt files")
    parser.add_argument(
        "--model",
        choices=["yolov5n", "mobilenetv2", "all"],
        default="all",
        help="Model to generate config for"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "triton",
        help="Output directory"
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_config",
        help="Print config to stdout instead of saving"
    )
    
    args = parser.parse_args()
    
    if args.model == "all":
        models = ["yolov5n", "mobilenetv2"]
    else:
        models = [args.model]
    
    for model_name in models:
        config = generate_config_pbtxt(model_name)
        
        if args.print_config:
            print(f"\n{'='*60}")
            print(f"# {model_name}/config.pbtxt")
            print('='*60)
            print(config)
        else:
            output_path = save_config_pbtxt(model_name, args.output_dir)
            print(f"Generated: {output_path}")
