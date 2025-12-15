"""
Shared Module - Controlled Variables for ML Serving Architecture Comparison

This module contains shared components used identically across all architectures
to ensure fair experimental comparison:

- processing: Image preprocessing (YOLO letterbox, MobileNet ImageNet normalization)
- model: ONNX model registry and export utilities
- data: COCO dataset download and curation
- proto: gRPC service definitions

All architectures import from this module to eliminate implementation
variance as a confounding variable.
"""

from shared.processing import (
    YOLOPreprocessor,
    MobileNetPreprocessor,
    letterbox,
    imagenet_normalize,
    scale_boxes,
)

__all__ = [
    "YOLOPreprocessor",
    "MobileNetPreprocessor",
    "letterbox",
    "imagenet_normalize",
    "scale_boxes",
]

__version__ = "0.1.0"
