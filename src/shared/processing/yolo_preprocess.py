"""YOLOv5 Preprocessing Pipeline.

This module provides the YOLOPreprocessor class for preparing images
for YOLOv5n object detection inference.

Pipeline:
    1. Letterbox resize to 640x640 (preserve aspect ratio)
    2. Normalize to [0, 1] by dividing by 255.0
    3. Transpose HWC → CHW (channels first for ONNX)
    4. Add batch dimension → [1, 3, 640, 640]

The preprocessor also provides coordinate conversion utilities
to map detection outputs back to original image coordinates.

Author: Matthew Hong
Specification Reference: Foundation Specification §3.2
"""

from dataclasses import dataclass

import numpy as np

from shared.config import get_controlled_variable
from shared.processing.transforms import letterbox, scale_boxes

# =============================================================================
# Constants (Loaded from experiment.yaml)
# =============================================================================

# Load preprocessing parameters from centralized config (experiment.yaml)
_yolo_config = get_controlled_variable("preprocessing", "yolo")
YOLO_INPUT_SIZE: int = _yolo_config["target_size"]
"""Standard YOLOv5 input dimension (square) from experiment.yaml."""

YOLO_NORMALIZATION_SCALE: float = _yolo_config["normalization_scale"]
"""YOLOv5 expects pixels normalized to [0, 1] from experiment.yaml."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class YOLOPreprocessResult:
    """Result container for YOLO preprocessing.

    Attributes:
        tensor: Preprocessed image tensor [1, 3, 640, 640], float32, range [0, 1]
        scale: Scale factor applied during letterbox (for coordinate conversion)
        padding: (pad_w, pad_h) pixels added during letterbox
        original_shape: (height, width) of input image
    """

    tensor: np.ndarray
    scale: float
    padding: tuple[int, int]
    original_shape: tuple[int, int]

    def scale_boxes_to_original(self, boxes: np.ndarray) -> np.ndarray:
        """Convert detection boxes from YOLO output space to original image coordinates.

        Args:
            boxes: Detection boxes [N, 4+] with [x1, y1, x2, y2, ...] in 640x640 space

        Returns:
            Boxes with coordinates in original image space
        """
        return scale_boxes(boxes, self.scale, self.padding, self.original_shape)


# =============================================================================
# Preprocessor Class
# =============================================================================


class YOLOPreprocessor:
    """Preprocessor for YOLOv5n object detection model.

    Transforms input images into tensors suitable for ONNX Runtime inference.
    Provides consistent preprocessing across all architectures (monolithic,
    microservices, triton) to ensure fair experimental comparison.

    Attributes:
        input_size: Target input dimension (default: 640)
        normalization_scale: Divisor for pixel normalization (default: 255.0)

    Example:
        >>> preprocessor = YOLOPreprocessor()
        >>> image = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        >>> result = preprocessor(image)
        >>> result.tensor.shape
        (1, 3, 640, 640)
        >>> result.tensor.dtype
        dtype('float32')
        >>> 0.0 <= result.tensor.min() <= result.tensor.max() <= 1.0
        True
    """

    def __init__(
        self,
        input_size: int = YOLO_INPUT_SIZE,
        normalization_scale: float = YOLO_NORMALIZATION_SCALE,
    ) -> None:
        """Initialize YOLOPreprocessor.

        Args:
            input_size: Target square dimension for model input (default: 640)
            normalization_scale: Divisor for normalization (default: 255.0)
        """
        self.input_size = input_size
        self.normalization_scale = normalization_scale

    def __call__(self, image: np.ndarray) -> YOLOPreprocessResult:
        """Preprocess image for YOLOv5n inference.

        Args:
            image: RGB uint8 array with shape [H, W, 3]

        Returns:
            YOLOPreprocessResult containing tensor and metadata

        Raises:
            ValueError: If image has invalid shape or dtype
        """
        return self.preprocess(image)

    def preprocess(self, image: np.ndarray) -> YOLOPreprocessResult:
        """Preprocess image for YOLOv5n inference.

        Pipeline:
            1. Letterbox resize to 640x640
            2. Normalize to [0, 1]
            3. Transpose HWC → CHW
            4. Add batch dimension

        Args:
            image: RGB uint8 array with shape [H, W, 3]

        Returns:
            YOLOPreprocessResult containing:
                - tensor: [1, 3, 640, 640] float32 in [0, 1]
                - scale: Scale factor for coordinate conversion
                - padding: Padding offset for coordinate conversion
                - original_shape: Input image dimensions

        Raises:
            ValueError: If image has invalid shape or dtype
        """
        self._validate_input(image)

        original_shape = (image.shape[0], image.shape[1])

        # Step 1: Letterbox resize
        letterboxed, scale, padding = letterbox(image, self.input_size)

        # Step 2: Normalize to [0, 1]
        normalized = letterboxed.astype(np.float32) / self.normalization_scale

        # Step 3: Transpose HWC → CHW
        transposed = normalized.transpose(2, 0, 1)

        # Step 4: Add batch dimension
        batched = np.expand_dims(transposed, axis=0)

        # Ensure contiguous memory layout for ONNX Runtime
        tensor = np.ascontiguousarray(batched)

        return YOLOPreprocessResult(
            tensor=tensor,
            scale=scale,
            padding=padding,
            original_shape=original_shape,
        )

    def _validate_input(self, image: np.ndarray) -> None:
        """Validate input image.

        Args:
            image: Image to validate

        Raises:
            ValueError: If image has invalid shape or dtype
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(image)}")

        if image.ndim != 3:
            raise ValueError(f"Expected 3D array [H, W, C], got {image.ndim}D")

        if image.shape[2] != 3:
            raise ValueError(f"Expected 3 channels, got {image.shape[2]}")

        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 dtype, got {image.dtype}")

    @staticmethod
    def get_input_shape() -> tuple[int, int, int, int]:
        """Get expected ONNX model input shape.

        Returns:
            Tuple of (batch, channels, height, width)
        """
        return (1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE)

    @staticmethod
    def get_input_dtype() -> np.dtype:
        """Get expected ONNX model input dtype.

        Returns:
            numpy dtype (float32)
        """
        return np.dtype(np.float32)
