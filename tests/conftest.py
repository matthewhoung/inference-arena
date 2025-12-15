"""
Pytest Fixtures - Shared Test Fixtures for Inference Arena

This module provides reusable fixtures for all test modules.

Fixtures:
    sample_image: Sample RGB image (1080x1920) for testing
    sample_image_square: Sample RGB image (640x640) for testing
    sample_crop: Sample RGB crop (100x150) for testing
    sample_crop_small: Small RGB crop (10x10) for edge case testing
    sample_boxes: Sample YOLO-format detection boxes

Author: Matthew Hong
"""

import numpy as np
import pytest


# =============================================================================
# Image Fixtures
# =============================================================================

@pytest.fixture
def sample_image() -> np.ndarray:
    """
    Sample 1080p RGB image for testing.

    Returns:
        RGB uint8 array with shape [1080, 1920, 3]
    """
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_square() -> np.ndarray:
    """
    Sample square RGB image (640x640) for testing.

    Returns:
        RGB uint8 array with shape [640, 640, 3]
    """
    rng = np.random.default_rng(43)
    return rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_portrait() -> np.ndarray:
    """
    Sample portrait RGB image (1920x1080) for testing letterbox.

    Returns:
        RGB uint8 array with shape [1920, 1080, 3]
    """
    rng = np.random.default_rng(44)
    return rng.integers(0, 256, (1920, 1080, 3), dtype=np.uint8)


# =============================================================================
# Crop Fixtures
# =============================================================================

@pytest.fixture
def sample_crop() -> np.ndarray:
    """
    Sample RGB crop for MobileNet testing.

    Returns:
        RGB uint8 array with shape [100, 150, 3]
    """
    rng = np.random.default_rng(45)
    return rng.integers(0, 256, (100, 150, 3), dtype=np.uint8)


@pytest.fixture
def sample_crop_small() -> np.ndarray:
    """
    Small RGB crop for edge case testing.

    Returns:
        RGB uint8 array with shape [10, 10, 3]
    """
    rng = np.random.default_rng(46)
    return rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)


@pytest.fixture
def sample_crop_large() -> np.ndarray:
    """
    Large RGB crop for testing.

    Returns:
        RGB uint8 array with shape [500, 800, 3]
    """
    rng = np.random.default_rng(47)
    return rng.integers(0, 256, (500, 800, 3), dtype=np.uint8)


# =============================================================================
# Bounding Box Fixtures
# =============================================================================

@pytest.fixture
def sample_boxes() -> np.ndarray:
    """
    Sample YOLO-format detection boxes.

    Format: [x1, y1, x2, y2, confidence, class_id]

    Returns:
        Array with shape [3, 6] containing 3 sample detections
    """
    return np.array(
        [
            [100, 100, 200, 200, 0.95, 0],  # Box 1
            [300, 150, 450, 350, 0.87, 1],  # Box 2
            [50, 400, 150, 550, 0.72, 2],   # Box 3
        ],
        dtype=np.float32,
    )


@pytest.fixture
def sample_boxes_letterbox_space() -> np.ndarray:
    """
    Sample detection boxes in 640x640 letterbox space.

    These boxes simulate YOLO output before coordinate conversion.

    Returns:
        Array with shape [2, 6] in letterbox coordinates
    """
    return np.array(
        [
            [320, 320, 400, 400, 0.92, 0],  # Center box
            [100, 200, 250, 350, 0.85, 1],  # Off-center box
        ],
        dtype=np.float32,
    )


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def yolo_input_shape() -> tuple[int, int, int, int]:
    """Expected YOLO input tensor shape."""
    return (1, 3, 640, 640)


@pytest.fixture
def mobilenet_input_shape() -> tuple[int, int, int, int]:
    """Expected MobileNet input tensor shape."""
    return (1, 3, 224, 224)


@pytest.fixture
def imagenet_mean() -> np.ndarray:
    """ImageNet channel means."""
    return np.array([0.485, 0.456, 0.406], dtype=np.float32)


@pytest.fixture
def imagenet_std() -> np.ndarray:
    """ImageNet channel standard deviations."""
    return np.array([0.229, 0.224, 0.225], dtype=np.float32)
