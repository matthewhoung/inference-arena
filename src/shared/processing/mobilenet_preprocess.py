"""
MobileNetV2 Preprocessing Pipeline

This module provides the MobileNetPreprocessor class for preparing
image crops for MobileNetV2 classification inference.

Pipeline:
    1. Resize crop to 224x224 (bilinear interpolation)
    2. Convert to float32 and scale to [0, 1]
    3. Apply ImageNet normalization (subtract mean, divide by std)
    4. Transpose HWC → CHW (channels first for ONNX)
    5. Add batch dimension → [1, 3, 224, 224]

Author: Matthew Hong
Specification Reference: Foundation Specification §3.4
"""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from shared.processing.transforms import imagenet_normalize, IMAGENET_MEAN, IMAGENET_STD


# =============================================================================
# Constants
# =============================================================================

MOBILENET_INPUT_SIZE: int = 224
"""Standard MobileNetV2 input dimension (square)."""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MobileNetPreprocessResult:
    """
    Result container for MobileNet preprocessing.

    Attributes:
        tensor: Preprocessed image tensor [1, 3, 224, 224], float32, ImageNet normalized
        original_shape: (height, width) of input crop
    """

    tensor: np.ndarray
    original_shape: Tuple[int, int]


# =============================================================================
# Preprocessor Class
# =============================================================================

class MobileNetPreprocessor:
    """
    Preprocessor for MobileNetV2 classification model.

    Transforms cropped image regions (from YOLO detections) into tensors
    suitable for ONNX Runtime inference. Uses ImageNet normalization
    to match the model's training distribution.

    Attributes:
        input_size: Target input dimension (default: 224)
        mean: ImageNet channel means for normalization
        std: ImageNet channel standard deviations for normalization

    Example:
        >>> preprocessor = MobileNetPreprocessor()
        >>> crop = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
        >>> result = preprocessor(crop)
        >>> result.tensor.shape
        (1, 3, 224, 224)
        >>> result.tensor.dtype
        dtype('float32')
        >>> # ImageNet normalized range is approximately [-2.1, 2.6]
        >>> -3.0 < result.tensor.min() < result.tensor.max() < 3.0
        True
    """

    def __init__(
        self,
        input_size: int = MOBILENET_INPUT_SIZE,
        mean: np.ndarray = IMAGENET_MEAN,
        std: np.ndarray = IMAGENET_STD,
    ) -> None:
        """
        Initialize MobileNetPreprocessor.

        Args:
            input_size: Target square dimension for model input (default: 224)
            mean: Channel means for normalization (default: ImageNet)
            std: Channel standard deviations for normalization (default: ImageNet)
        """
        self.input_size = input_size
        self.mean = mean
        self.std = std

    def __call__(self, crop: np.ndarray) -> MobileNetPreprocessResult:
        """
        Preprocess crop for MobileNetV2 inference.

        Args:
            crop: RGB uint8 array with shape [H, W, 3]

        Returns:
            MobileNetPreprocessResult containing tensor and metadata

        Raises:
            ValueError: If crop has invalid shape or dtype
        """
        return self.preprocess(crop)

    def preprocess(self, crop: np.ndarray) -> MobileNetPreprocessResult:
        """
        Preprocess crop for MobileNetV2 inference.

        Pipeline:
            1. Resize to 224x224 (bilinear interpolation)
            2. Scale to [0, 1] and apply ImageNet normalization
            3. Transpose HWC → CHW
            4. Add batch dimension

        Args:
            crop: RGB uint8 array with shape [H, W, 3]

        Returns:
            MobileNetPreprocessResult containing:
                - tensor: [1, 3, 224, 224] float32, ImageNet normalized
                - original_shape: Input crop dimensions

        Raises:
            ValueError: If crop has invalid shape or dtype
        """
        self._validate_input(crop)

        original_shape = (crop.shape[0], crop.shape[1])

        # Step 1: Resize to 224x224
        resized = cv2.resize(
            crop,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )

        # Step 2: Apply ImageNet normalization
        normalized = imagenet_normalize(resized)

        # Step 3: Transpose HWC → CHW
        transposed = normalized.transpose(2, 0, 1)

        # Step 4: Add batch dimension
        batched = np.expand_dims(transposed, axis=0)

        # Ensure contiguous memory layout for ONNX Runtime
        tensor = np.ascontiguousarray(batched)

        return MobileNetPreprocessResult(
            tensor=tensor,
            original_shape=original_shape,
        )

    def preprocess_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        """
        Preprocess multiple crops for batched inference.

        Note: This is provided for future batching support but is not used
        in the current single-request comparison study.

        Args:
            crops: List of RGB uint8 arrays with shape [H, W, 3]

        Returns:
            Batched tensor with shape [N, 3, 224, 224], ImageNet normalized

        Raises:
            ValueError: If any crop has invalid shape or dtype
        """
        if not crops:
            return np.zeros(
                (0, 3, self.input_size, self.input_size),
                dtype=np.float32,
            )

        tensors = [self.preprocess(crop).tensor for crop in crops]
        return np.concatenate(tensors, axis=0)

    def _validate_input(self, crop: np.ndarray) -> None:
        """
        Validate input crop.

        Args:
            crop: Crop to validate

        Raises:
            ValueError: If crop has invalid shape or dtype
        """
        if not isinstance(crop, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(crop)}")

        if crop.ndim != 3:
            raise ValueError(f"Expected 3D array [H, W, C], got {crop.ndim}D")

        if crop.shape[2] != 3:
            raise ValueError(f"Expected 3 channels, got {crop.shape[2]}")

        # Allow both uint8 and float32 inputs
        if crop.dtype not in (np.uint8, np.float32):
            raise ValueError(f"Expected uint8 or float32 dtype, got {crop.dtype}")

        # Check for valid dimensions (at least 1x1 pixel)
        if crop.shape[0] < 1 or crop.shape[1] < 1:
            raise ValueError(f"Invalid crop dimensions: {crop.shape[:2]}")

    @staticmethod
    def get_input_shape() -> Tuple[int, int, int, int]:
        """
        Get expected ONNX model input shape.

        Returns:
            Tuple of (batch, channels, height, width)
        """
        return (1, 3, MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE)

    @staticmethod
    def get_input_dtype() -> np.dtype:
        """
        Get expected ONNX model input dtype.

        Returns:
            numpy dtype (float32)
        """
        return np.dtype(np.float32)


# =============================================================================
# Utility Functions
# =============================================================================

def extract_crop(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Extract a crop from an image using a bounding box.

    Crops are extracted from the original-resolution image (not the
    letterboxed YOLO input) to preserve maximum detail for classification.

    Args:
        image: RGB uint8 array with shape [H, W, 3] (original image)
        box: Array with [x1, y1, x2, y2, ...] coordinates in original image space

    Returns:
        Cropped region as RGB uint8 array

    Example:
        >>> image = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> box = np.array([100, 100, 300, 400, 0.9, 0])
        >>> crop = extract_crop(image, box)
        >>> crop.shape
        (300, 200, 3)
    """
    x1, y1, x2, y2 = map(int, box[:4])

    # Ensure valid bounds
    height, width = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    # Handle edge case of zero-size crop
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    return image[y1:y2, x1:x2].copy()
