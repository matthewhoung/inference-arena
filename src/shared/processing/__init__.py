"""
Processing Module - Shared Preprocessing for All Architectures

This module provides identical preprocessing logic for:
- YOLOv5n object detection (letterbox resize, [0,1] normalization)
- MobileNetV2 classification (resize, ImageNet normalization)

Using shared preprocessing ensures that input tensor preparation
is not a confounding variable in the architectural comparison.

Specification Reference:
    Foundation Specification §3 Preprocessing
"""

from shared.processing.transforms import (
    letterbox,
    imagenet_normalize,
    scale_boxes,
    load_image,
    load_image_from_bytes,
)

from shared.processing.yolo_preprocess import YOLOPreprocessor
from shared.processing.mobilenet_preprocess import MobileNetPreprocessor

__all__ = [
    # Low-level transforms
    "letterbox",
    "imagenet_normalize",
    "scale_boxes",
    "load_image",
    "load_image_from_bytes",
    # High-level preprocessors
    "YOLOPreprocessor",
    "MobileNetPreprocessor",
]
