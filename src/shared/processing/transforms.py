"""
Low-Level Image Transforms

This module contains atomic transformation functions used by
YOLOPreprocessor and MobileNetPreprocessor.

Functions:
    load_image: Load image file as RGB numpy array
    load_image_from_bytes: Decode image bytes as RGB numpy array
    letterbox: Resize with aspect ratio preservation and padding
    imagenet_normalize: Apply ImageNet mean/std normalization
    scale_boxes: Convert coordinates from letterboxed to original space

Constants:
    IMAGENET_MEAN: ImageNet dataset channel means [R, G, B]
    IMAGENET_STD: ImageNet dataset channel standard deviations [R, G, B]

Author: Matthew Hong
Specification Reference: Foundation Specification §3.2, §3.3, §3.4
"""

from typing import Tuple

import cv2
import numpy as np


# =============================================================================
# Constants
# =============================================================================

# ImageNet normalization constants
# Reference: https://pytorch.org/vision/stable/models.html
IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default letterbox padding color (gray, matches Ultralytics default)
LETTERBOX_COLOR: Tuple[int, int, int] = (114, 114, 114)


# =============================================================================
# Image Loading
# =============================================================================

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image file as an RGB numpy array.

    Uses OpenCV for decoding (libjpeg-turbo optimized) with explicit
    BGR to RGB conversion for consistency with model training pipelines.

    Args:
        image_path: Path to image file (JPEG, PNG, etc.)

    Returns:
        RGB uint8 array with shape [H, W, 3]

    Raises:
        ValueError: If image cannot be loaded (file not found or corrupted)

    Example:
        >>> image = load_image("path/to/image.jpg")
        >>> image.shape
        (1080, 1920, 3)
        >>> image.dtype
        dtype('uint8')
    """
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes as an RGB numpy array.

    Useful for processing images received via HTTP/gRPC without
    writing to disk.

    Args:
        image_bytes: Raw image bytes (JPEG, PNG, etc.)

    Returns:
        RGB uint8 array with shape [H, W, 3]

    Raises:
        ValueError: If image cannot be decoded

    Example:
        >>> with open("image.jpg", "rb") as f:
        ...     image_bytes = f.read()
        >>> image = load_image_from_bytes(image_bytes)
        >>> image.shape
        (1080, 1920, 3)
    """
    if not image_bytes:
        raise ValueError("Failed to decode image from bytes: empty input")

    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError("Failed to decode image from bytes")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb


# =============================================================================
# Geometric Transforms
# =============================================================================

def letterbox(
    image: np.ndarray,
    target_size: int,
    color: Tuple[int, int, int] = LETTERBOX_COLOR,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with letterboxing to maintain aspect ratio.

    Letterboxing scales the image to fit within a square target size
    while preserving aspect ratio, then pads the remaining space with
    a solid color. This prevents distortion for object detection models.

    Args:
        image: RGB uint8 array with shape [H, W, 3]
        target_size: Target dimension for square output (e.g., 640)
        color: RGB tuple for padding color (default: gray 114)

    Returns:
        Tuple of:
            - letterboxed: Resized and padded image [target_size, target_size, 3]
            - scale: Scale factor applied (for coordinate conversion)
            - padding: (pad_w, pad_h) pixels added to top-left

    Example:
        >>> image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        >>> letterboxed, scale, padding = letterbox(image, 640)
        >>> letterboxed.shape
        (640, 640, 3)
        >>> scale  # 640/1920 = 0.333...
        0.3333333333333333
        >>> padding  # No horizontal padding, vertical padding centered
        (0, 140)
    """
    height, width = image.shape[:2]

    # Calculate scale to fit within target_size
    scale = min(target_size / height, target_size / width)

    # New dimensions after scaling
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize image with bilinear interpolation
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    # Create padded canvas
    letterboxed = np.full(
        (target_size, target_size, 3),
        color,
        dtype=np.uint8,
    )

    # Calculate padding offsets (center the image)
    pad_w = (target_size - new_width) // 2
    pad_h = (target_size - new_height) // 2

    # Place resized image on canvas
    letterboxed[pad_h : pad_h + new_height, pad_w : pad_w + new_width] = resized

    return letterboxed, scale, (pad_w, pad_h)


def scale_boxes(
    boxes: np.ndarray,
    scale: float,
    padding: Tuple[int, int],
    original_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Convert bounding boxes from letterboxed coordinates to original image coordinates.

    YOLO outputs coordinates in the letterboxed space (e.g., 640x640).
    This function reverses the letterbox transformation to get coordinates
    in the original image space for accurate cropping.

    Args:
        boxes: Array with shape [N, 4+] where first 4 columns are [x1, y1, x2, y2]
        scale: Scale factor from letterbox()
        padding: (pad_w, pad_h) from letterbox()
        original_shape: (height, width) of original image

    Returns:
        Array with same shape, coordinates converted to original space

    Example:
        >>> boxes = np.array([[320, 320, 400, 400, 0.9, 0]], dtype=np.float32)
        >>> scaled = scale_boxes(boxes, 0.333, (0, 140), (1080, 1920))
        >>> scaled[0, :4]  # Coordinates expanded back to original size
        array([960., 540., 1200., 780.], dtype=float32)
    """
    boxes = boxes.copy()  # Don't modify input

    pad_w, pad_h = padding
    orig_h, orig_w = original_shape

    # Remove padding offset
    boxes[:, 0] -= pad_w  # x1
    boxes[:, 1] -= pad_h  # y1
    boxes[:, 2] -= pad_w  # x2
    boxes[:, 3] -= pad_h  # y2

    # Remove scale (divide by scale to get original coords)
    boxes[:, :4] /= scale

    # Clip to image bounds
    boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)  # y2

    return boxes


# =============================================================================
# Intensity Transforms
# =============================================================================

def imagenet_normalize(image: np.ndarray) -> np.ndarray:
    """
    Apply ImageNet normalization to an image.

    Converts image to float32, scales to [0, 1], then normalizes using
    ImageNet channel means and standard deviations.

    Formula: normalized = (pixel / 255.0 - mean) / std

    Args:
        image: RGB uint8 or float32 array with shape [H, W, 3]

    Returns:
        Normalized float32 array with shape [H, W, 3]
        Value range approximately [-2.1, 2.6]

    Example:
        >>> image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        >>> normalized = imagenet_normalize(image)
        >>> normalized.dtype
        dtype('float32')
        >>> -3.0 < normalized.min() < normalized.max() < 3.0
        True
    """
    # Convert to float32 and scale to [0, 1]
    if image.dtype == np.uint8:
        normalized = image.astype(np.float32) / 255.0
    else:
        normalized = image.astype(np.float32)
        if normalized.max() > 1.0:
            normalized /= 255.0

    # Apply ImageNet normalization: (x - mean) / std
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

    return normalized
