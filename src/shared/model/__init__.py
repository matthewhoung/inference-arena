"""
Model Module - ONNX Model Export and Registry

This module provides:
- exporter: Export PyTorch models (YOLOv5n, MobileNetV2) to ONNX format
- registry: Load and cache ONNX Runtime inference sessions

All architectures use this module to ensure identical model artifacts
and consistent inference session configuration.

Specification Reference:
    Foundation Specification §2 Model Export
"""

from shared.model.exporter import (
    export_yolov5n,
    export_mobilenetv2,
    verify_onnx_model,
    compute_checksum,
    ONNX_OPSET_VERSION,
    YOLO_INPUT_SIZE,
    MOBILENET_INPUT_SIZE,
)

from shared.model.registry import (
    ModelRegistry,
    get_default_registry,
)

__all__ = [
    # Exporter
    "export_yolov5n",
    "export_mobilenetv2",
    "verify_onnx_model",
    "compute_checksum",
    "ONNX_OPSET_VERSION",
    "YOLO_INPUT_SIZE",
    "MOBILENET_INPUT_SIZE",
    # Registry
    "ModelRegistry",
    "get_default_registry",
]
