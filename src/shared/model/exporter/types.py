"""ONNX Model Exporter Types.

This module contains data classes and constants used across the exporter package.

Author: Matthew Hong
Specification Reference: Foundation Specification Section 2 Model Export
"""

from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# Constants
# =============================================================================

ONNX_OPSET_VERSION: int = 17
"""ONNX opset version for Triton 24.08 compatibility."""

YOLO_INPUT_SIZE: int = 640
"""YOLOv5 input dimension (square)."""

MOBILENET_INPUT_SIZE: int = 224
"""MobileNetV2 input dimension (square)."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ExportResult:
    """Result container for model export operation.

    Attributes:
        model_path: Path to exported ONNX file
        checksum: SHA256 checksum of exported file
        opset_version: ONNX opset version used
        input_shape: Model input tensor shape
        output_shape: Model output tensor shape (may contain dynamic dims)
        file_size_mb: File size in megabytes
    """

    model_path: Path
    checksum: str
    opset_version: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    file_size_mb: float
