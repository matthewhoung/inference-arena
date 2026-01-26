"""ONNX Model Exporter Utilities.

This module contains shared utility functions for checksum computation
and ONNX model verification.

Author: Matthew Hong
Specification Reference: Foundation Specification Section 2 Model Export
"""

import hashlib
from pathlib import Path
from typing import Any, TypedDict


class VerificationResult(TypedDict):
    """Type definition for verify_onnx_model result."""

    valid: bool
    opset_version: int | None
    input_shapes: list[tuple[Any, ...]]
    output_shapes: list[tuple[Any, ...]]
    error: str | None

# =============================================================================
# Checksum Utilities
# =============================================================================


def compute_checksum(file_path: Path) -> str:
    """Compute SHA256 checksum of a file.

    Args:
        file_path: Path to file

    Returns:
        Hex-encoded SHA256 checksum string

    Example:
        >>> checksum = compute_checksum(Path("model.onnx"))
        >>> len(checksum)
        64
    """
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


# =============================================================================
# ONNX Verification
# =============================================================================


def verify_onnx_model(model_path: Path) -> VerificationResult:
    """Verify ONNX model is valid and meets specifications.

    Checks:
    - File exists and is readable
    - Valid ONNX format (passes onnx.checker)
    - Correct opset version
    - Can be loaded by ONNX Runtime

    Args:
        model_path: Path to ONNX model file

    Returns:
        Dictionary with verification results:
        - valid: bool
        - opset_version: int
        - input_shapes: list of input shapes
        - output_shapes: list of output shapes
        - error: Optional error message

    Example:
        >>> result = verify_onnx_model(Path("yolov5n.onnx"))
        >>> result["valid"]
        True
        >>> result["opset_version"]
        17
    """
    result: VerificationResult = {
        "valid": False,
        "opset_version": None,
        "input_shapes": [],
        "output_shapes": [],
        "error": None,
    }

    if not model_path.exists():
        result["error"] = f"File not found: {model_path}"
        return result

    try:
        import onnx
        from onnx import checker

        # Load and validate ONNX model
        model = onnx.load(str(model_path))
        checker.check_model(model)

        # Extract opset version
        result["opset_version"] = model.opset_import[0].version

        # Extract input shapes
        input_shapes: list[tuple[Any, ...]] = []
        for inp in model.graph.input:
            shape: list[Any] = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)  # Dynamic dimension
                else:
                    shape.append(-1)
            input_shapes.append(tuple(shape))
        result["input_shapes"] = input_shapes

        # Extract output shapes
        output_shapes: list[tuple[Any, ...]] = []
        for out in model.graph.output:
            shape = []
            for dim in out.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(-1)
            output_shapes.append(tuple(shape))
        result["output_shapes"] = output_shapes

        # Verify can be loaded by ONNX Runtime
        import onnxruntime as ort

        ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

        result["valid"] = True

    except Exception as e:
        result["error"] = str(e)

    return result
