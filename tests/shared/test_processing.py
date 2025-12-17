"""
Unit Tests for Processing Module

This module tests:
- transforms.py: letterbox, imagenet_normalize, scale_boxes, load functions
- yolo_preprocess.py: YOLOPreprocessor class
- mobilenet_preprocess.py: MobileNetPreprocessor class, extract_crop

Test Categories:
- Shape validation: Output tensor dimensions match specification
- Dtype validation: Output tensors are float32
- Range validation: Output values within expected bounds
- Edge cases: Small inputs, boundary conditions

Author: Matthew Hong
Specification Reference: Foundation Specification §3
"""

import numpy as np
import pytest

from shared.processing.mobilenet_preprocess import (
    MobileNetPreprocessor,
    MobileNetPreprocessResult,
    extract_crop,
)
from shared.processing.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    LETTERBOX_COLOR,
    imagenet_normalize,
    letterbox,
    load_image,
    load_image_from_bytes,
    scale_boxes,
)
from shared.processing.yolo_preprocess import (
    YOLOPreprocessor,
    YOLOPreprocessResult,
)

# =============================================================================
# Tests for transforms.py
# =============================================================================


class TestLetterbox:
    """Tests for letterbox transform."""

    def test_letterbox_output_shape(self, sample_image: np.ndarray) -> None:
        """Letterbox should produce square output of target size."""
        letterboxed, scale, padding = letterbox(sample_image, 640)

        assert letterboxed.shape == (640, 640, 3)

    def test_letterbox_preserves_dtype(self, sample_image: np.ndarray) -> None:
        """Letterbox should preserve uint8 dtype."""
        letterboxed, _, _ = letterbox(sample_image, 640)

        assert letterboxed.dtype == np.uint8

    @pytest.mark.parametrize(
        "shape,expected_scale",
        [
            ((1080, 1920, 3), 640 / 1920),  # landscape: width is limiting
            ((1920, 1080, 3), 640 / 1920),  # portrait: height is limiting
            ((640, 640, 3), 1.0),  # square: exact fit
            ((320, 320, 3), 2.0),  # small square: upscale
            ((480, 640, 3), 1.0),  # 4:3 landscape
            ((640, 480, 3), 640 / 640),  # 4:3 portrait
        ],
    )
    def test_letterbox_scale(self, shape: tuple, expected_scale: float) -> None:
        """Letterbox scale should match expected for various aspect ratios."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, shape, dtype=np.uint8)

        _, scale, _ = letterbox(image, 640)

        assert np.isclose(scale, expected_scale, rtol=1e-5)

    @pytest.mark.parametrize(
        "shape,expected_pad_w_zero,expected_pad_h_zero",
        [
            ((1080, 1920, 3), True, False),  # landscape: pad height
            ((1920, 1080, 3), False, True),  # portrait: pad width
            ((640, 640, 3), True, True),  # square: no padding
        ],
    )
    def test_letterbox_padding_direction(
        self,
        shape: tuple,
        expected_pad_w_zero: bool,
        expected_pad_h_zero: bool,
    ) -> None:
        """Letterbox padding should be applied to correct dimension."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, shape, dtype=np.uint8)

        _, _, (pad_w, pad_h) = letterbox(image, 640)

        assert (pad_w == 0) == expected_pad_w_zero
        assert (pad_h == 0) == expected_pad_h_zero

    def test_letterbox_padding_color(self, sample_image: np.ndarray) -> None:
        """Padding should use default letterbox color (114, 114, 114)."""
        letterboxed, _, (pad_w, pad_h) = letterbox(sample_image, 640)

        # Check top padding row
        if pad_h > 0:
            padding_row = letterboxed[0, :, :]
            expected = np.array(LETTERBOX_COLOR, dtype=np.uint8)
            assert np.all(padding_row == expected)

    @pytest.mark.parametrize("target_size", [320, 416, 512, 640, 1280])
    def test_letterbox_various_target_sizes(self, target_size: int) -> None:
        """Letterbox should support various target sizes."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)

        letterboxed, _, _ = letterbox(image, target_size)

        assert letterboxed.shape == (target_size, target_size, 3)


class TestImageNetNormalize:
    """Tests for ImageNet normalization."""

    def test_imagenet_normalize_output_dtype(self, sample_crop: np.ndarray) -> None:
        """Output should be float32."""
        normalized = imagenet_normalize(sample_crop)

        assert normalized.dtype == np.float32

    def test_imagenet_normalize_output_shape(self, sample_crop: np.ndarray) -> None:
        """Output shape should match input shape."""
        normalized = imagenet_normalize(sample_crop)

        assert normalized.shape == sample_crop.shape

    def test_imagenet_normalize_range(self, sample_crop: np.ndarray) -> None:
        """Output should be in approximately [-2.1, 2.6] for ImageNet."""
        normalized = imagenet_normalize(sample_crop)

        # ImageNet normalized range is approximately [-2.1, 2.6]
        assert normalized.min() >= -3.0
        assert normalized.max() <= 3.0

    def test_imagenet_normalize_zero_image(self) -> None:
        """Zero image should normalize to -mean/std."""
        zero_image = np.zeros((10, 10, 3), dtype=np.uint8)
        normalized = imagenet_normalize(zero_image)

        expected = -IMAGENET_MEAN / IMAGENET_STD
        assert np.allclose(normalized[0, 0, :], expected)

    def test_imagenet_normalize_white_image(self) -> None:
        """White image (255) should normalize to (1-mean)/std."""
        white_image = np.full((10, 10, 3), 255, dtype=np.uint8)
        normalized = imagenet_normalize(white_image)

        expected = (1.0 - IMAGENET_MEAN) / IMAGENET_STD
        assert np.allclose(normalized[0, 0, :], expected)

    def test_imagenet_normalize_accepts_float32(self) -> None:
        """Should accept float32 input already in [0, 1]."""
        float_image = np.random.rand(10, 10, 3).astype(np.float32)
        normalized = imagenet_normalize(float_image)

        assert normalized.dtype == np.float32


class TestScaleBoxes:
    """Tests for bounding box coordinate conversion."""

    def test_scale_boxes_preserves_shape(self, sample_boxes: np.ndarray) -> None:
        """Output shape should match input shape."""
        scaled = scale_boxes(sample_boxes, 0.5, (0, 0), (1080, 1920))

        assert scaled.shape == sample_boxes.shape

    def test_scale_boxes_removes_padding(self) -> None:
        """Coordinates should be adjusted for padding offset."""
        boxes = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)
        padding = (50, 50)

        scaled = scale_boxes(boxes, 1.0, padding, (1000, 1000))

        # x1, y1 should decrease by padding
        assert scaled[0, 0] == 50  # x1: 100 - 50 = 50
        assert scaled[0, 1] == 50  # y1: 100 - 50 = 50

    def test_scale_boxes_removes_scale(self) -> None:
        """Coordinates should be divided by scale factor."""
        boxes = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)
        scale = 0.5

        scaled = scale_boxes(boxes, scale, (0, 0), (2000, 2000))

        # Coordinates should double (divide by 0.5)
        assert scaled[0, 0] == 200  # x1: 100 / 0.5 = 200
        assert scaled[0, 2] == 400  # x2: 200 / 0.5 = 400

    def test_scale_boxes_clips_to_bounds(self) -> None:
        """Coordinates should be clipped to image bounds."""
        boxes = np.array([[0, 0, 1000, 1000, 0.9, 0]], dtype=np.float32)

        scaled = scale_boxes(boxes, 1.0, (0, 0), (480, 640))

        assert scaled[0, 2] == 640  # x2 clipped to width
        assert scaled[0, 3] == 480  # y2 clipped to height

    def test_scale_boxes_does_not_modify_input(self, sample_boxes: np.ndarray) -> None:
        """Original array should not be modified."""
        original = sample_boxes.copy()

        scale_boxes(sample_boxes, 0.5, (10, 10), (1080, 1920))

        assert np.array_equal(sample_boxes, original)


class TestLoadImage:
    """Tests for image loading functions."""

    def test_load_image_nonexistent_file(self) -> None:
        """Should raise ValueError for missing file."""
        with pytest.raises(ValueError, match="Failed to load"):
            load_image("/nonexistent/path/image.jpg")

    def test_load_image_invalid_path(self) -> None:
        """Should raise ValueError for invalid path."""
        with pytest.raises(ValueError, match="Failed to load"):
            load_image("")

    def test_load_image_from_bytes_invalid(self) -> None:
        """Should raise ValueError for invalid bytes."""
        with pytest.raises(ValueError, match="Failed to decode"):
            load_image_from_bytes(b"not an image")

    def test_load_image_from_bytes_empty(self) -> None:
        """Should raise ValueError for empty bytes."""
        with pytest.raises(ValueError, match="Failed to decode"):
            load_image_from_bytes(b"")

    # TODO: Add tests with real images when COCO val2017 is integrated (Issue #2/#3)
    # def test_load_image_jpg(self, coco_image_path: str) -> None:
    # def test_load_image_from_bytes_jpg(self, coco_image_bytes: bytes) -> None:


# =============================================================================
# Tests for YOLOPreprocessor
# =============================================================================


class TestYOLOPreprocessor:
    """Tests for YOLOPreprocessor class."""

    @pytest.fixture
    def preprocessor(self) -> YOLOPreprocessor:
        """Create YOLOPreprocessor instance."""
        return YOLOPreprocessor()

    def test_output_shape(
        self,
        preprocessor: YOLOPreprocessor,
        sample_image: np.ndarray,
        yolo_input_shape: tuple,
    ) -> None:
        """Output tensor should have shape [1, 3, 640, 640]."""
        result = preprocessor(sample_image)

        assert result.tensor.shape == yolo_input_shape

    def test_output_dtype(
        self,
        preprocessor: YOLOPreprocessor,
        sample_image: np.ndarray,
    ) -> None:
        """Output tensor should be float32."""
        result = preprocessor(sample_image)

        assert result.tensor.dtype == np.float32

    def test_output_range(
        self,
        preprocessor: YOLOPreprocessor,
        sample_image: np.ndarray,
    ) -> None:
        """Output tensor should be in range [0, 1]."""
        result = preprocessor(sample_image)

        assert result.tensor.min() >= 0.0
        assert result.tensor.max() <= 1.0

    def test_output_contiguous(
        self,
        preprocessor: YOLOPreprocessor,
        sample_image: np.ndarray,
    ) -> None:
        """Output tensor should be contiguous for ONNX Runtime."""
        result = preprocessor(sample_image)

        assert result.tensor.flags["C_CONTIGUOUS"]

    def test_result_contains_scale(
        self,
        preprocessor: YOLOPreprocessor,
        sample_image: np.ndarray,
    ) -> None:
        """Result should contain scale factor."""
        result = preprocessor(sample_image)

        assert isinstance(result.scale, float)
        assert result.scale > 0

    def test_result_contains_padding(
        self,
        preprocessor: YOLOPreprocessor,
        sample_image: np.ndarray,
    ) -> None:
        """Result should contain padding tuple."""
        result = preprocessor(sample_image)

        assert isinstance(result.padding, tuple)
        assert len(result.padding) == 2

    def test_result_contains_original_shape(
        self,
        preprocessor: YOLOPreprocessor,
        sample_image: np.ndarray,
    ) -> None:
        """Result should contain original image shape."""
        result = preprocessor(sample_image)

        assert result.original_shape == (1080, 1920)

    def test_scale_boxes_to_original(
        self,
        preprocessor: YOLOPreprocessor,
        sample_image: np.ndarray,
        sample_boxes_letterbox_space: np.ndarray,
    ) -> None:
        """Result should provide method to scale boxes back."""
        result = preprocessor(sample_image)
        scaled = result.scale_boxes_to_original(sample_boxes_letterbox_space)

        # Scaled boxes should be larger than letterbox space boxes
        assert scaled[0, 2] > sample_boxes_letterbox_space[0, 2]

    def test_callable_interface(
        self,
        preprocessor: YOLOPreprocessor,
        sample_image: np.ndarray,
    ) -> None:
        """Preprocessor should be callable."""
        result = preprocessor(sample_image)

        assert isinstance(result, YOLOPreprocessResult)

    def test_invalid_input_wrong_ndim(
        self,
        preprocessor: YOLOPreprocessor,
    ) -> None:
        """Should raise ValueError for non-3D input."""
        invalid_image = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises(ValueError, match="3D array"):
            preprocessor(invalid_image)

    def test_invalid_input_wrong_channels(
        self,
        preprocessor: YOLOPreprocessor,
    ) -> None:
        """Should raise ValueError for non-RGB input."""
        invalid_image = np.zeros((100, 100, 4), dtype=np.uint8)

        with pytest.raises(ValueError, match="3 channels"):
            preprocessor(invalid_image)

    def test_invalid_input_wrong_dtype(
        self,
        preprocessor: YOLOPreprocessor,
    ) -> None:
        """Should raise ValueError for non-uint8 input."""
        invalid_image = np.zeros((100, 100, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="uint8"):
            preprocessor(invalid_image)

    def test_get_input_shape(self) -> None:
        """Static method should return expected shape."""
        shape = YOLOPreprocessor.get_input_shape()

        assert shape == (1, 3, 640, 640)

    def test_get_input_dtype(self) -> None:
        """Static method should return float32 dtype."""
        dtype = YOLOPreprocessor.get_input_dtype()

        assert dtype == np.float32


# =============================================================================
# Tests for MobileNetPreprocessor
# =============================================================================


class TestMobileNetPreprocessor:
    """Tests for MobileNetPreprocessor class."""

    @pytest.fixture
    def preprocessor(self) -> MobileNetPreprocessor:
        """Create MobileNetPreprocessor instance."""
        return MobileNetPreprocessor()

    def test_output_shape(
        self,
        preprocessor: MobileNetPreprocessor,
        sample_crop: np.ndarray,
        mobilenet_input_shape: tuple,
    ) -> None:
        """Output tensor should have shape [1, 3, 224, 224]."""
        result = preprocessor(sample_crop)

        assert result.tensor.shape == mobilenet_input_shape

    def test_output_dtype(
        self,
        preprocessor: MobileNetPreprocessor,
        sample_crop: np.ndarray,
    ) -> None:
        """Output tensor should be float32."""
        result = preprocessor(sample_crop)

        assert result.tensor.dtype == np.float32

    def test_output_range_imagenet(
        self,
        preprocessor: MobileNetPreprocessor,
        sample_crop: np.ndarray,
    ) -> None:
        """Output tensor should be in ImageNet normalized range."""
        result = preprocessor(sample_crop)

        # ImageNet normalized range is approximately [-2.1, 2.6]
        assert result.tensor.min() >= -3.0
        assert result.tensor.max() <= 3.0

    def test_output_contiguous(
        self,
        preprocessor: MobileNetPreprocessor,
        sample_crop: np.ndarray,
    ) -> None:
        """Output tensor should be contiguous for ONNX Runtime."""
        result = preprocessor(sample_crop)

        assert result.tensor.flags["C_CONTIGUOUS"]

    def test_result_contains_original_shape(
        self,
        preprocessor: MobileNetPreprocessor,
        sample_crop: np.ndarray,
    ) -> None:
        """Result should contain original crop shape."""
        result = preprocessor(sample_crop)

        assert result.original_shape == (100, 150)

    def test_small_crop(
        self,
        preprocessor: MobileNetPreprocessor,
        sample_crop_small: np.ndarray,
        mobilenet_input_shape: tuple,
    ) -> None:
        """Should handle small crops (10x10)."""
        result = preprocessor(sample_crop_small)

        assert result.tensor.shape == mobilenet_input_shape

    def test_large_crop(
        self,
        preprocessor: MobileNetPreprocessor,
        sample_crop_large: np.ndarray,
        mobilenet_input_shape: tuple,
    ) -> None:
        """Should handle large crops (500x800)."""
        result = preprocessor(sample_crop_large)

        assert result.tensor.shape == mobilenet_input_shape

    def test_callable_interface(
        self,
        preprocessor: MobileNetPreprocessor,
        sample_crop: np.ndarray,
    ) -> None:
        """Preprocessor should be callable."""
        result = preprocessor(sample_crop)

        assert isinstance(result, MobileNetPreprocessResult)

    def test_preprocess_batch_empty(
        self,
        preprocessor: MobileNetPreprocessor,
    ) -> None:
        """Batch preprocessing should handle empty list."""
        result = preprocessor.preprocess_batch([])

        assert result.shape == (0, 3, 224, 224)

    def test_preprocess_batch_multiple(
        self,
        preprocessor: MobileNetPreprocessor,
        sample_crop: np.ndarray,
        sample_crop_small: np.ndarray,
    ) -> None:
        """Batch preprocessing should stack multiple crops."""
        result = preprocessor.preprocess_batch([sample_crop, sample_crop_small])

        assert result.shape == (2, 3, 224, 224)

    def test_invalid_input_wrong_ndim(
        self,
        preprocessor: MobileNetPreprocessor,
    ) -> None:
        """Should raise ValueError for non-3D input."""
        invalid_crop = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises(ValueError, match="3D array"):
            preprocessor(invalid_crop)

    def test_invalid_input_wrong_channels(
        self,
        preprocessor: MobileNetPreprocessor,
    ) -> None:
        """Should raise ValueError for non-RGB input."""
        invalid_crop = np.zeros((100, 100, 1), dtype=np.uint8)

        with pytest.raises(ValueError, match="3 channels"):
            preprocessor(invalid_crop)

    def test_invalid_input_wrong_dtype_int32(
        self,
        preprocessor: MobileNetPreprocessor,
    ) -> None:
        """Should raise ValueError for int32 input."""
        invalid_crop = np.zeros((100, 100, 3), dtype=np.int32)

        with pytest.raises(ValueError, match="uint8 or float32"):
            preprocessor(invalid_crop)

    def test_invalid_input_wrong_dtype_float64(
        self,
        preprocessor: MobileNetPreprocessor,
    ) -> None:
        """Should raise ValueError for float64 input."""
        invalid_crop = np.zeros((100, 100, 3), dtype=np.float64)

        with pytest.raises(ValueError, match="uint8 or float32"):
            preprocessor(invalid_crop)

    def test_valid_input_float32(
        self,
        preprocessor: MobileNetPreprocessor,
        mobilenet_input_shape: tuple,
    ) -> None:
        """Should accept float32 input."""
        rng = np.random.default_rng(42)
        float_crop = rng.random((100, 100, 3)).astype(np.float32)

        result = preprocessor(float_crop)

        assert result.tensor.shape == mobilenet_input_shape

    def test_get_input_shape(self) -> None:
        """Static method should return expected shape."""
        shape = MobileNetPreprocessor.get_input_shape()

        assert shape == (1, 3, 224, 224)

    def test_get_input_dtype(self) -> None:
        """Static method should return float32 dtype."""
        dtype = MobileNetPreprocessor.get_input_dtype()

        assert dtype == np.float32


class TestExtractCrop:
    """Tests for extract_crop utility function."""

    def test_extract_crop_shape(self, sample_image: np.ndarray) -> None:
        """Crop should have expected dimensions."""
        box = np.array([100, 100, 300, 400, 0.9, 0])

        crop = extract_crop(sample_image, box)

        # height = 400 - 100 = 300, width = 300 - 100 = 200
        assert crop.shape == (300, 200, 3)

    def test_extract_crop_dtype(self, sample_image: np.ndarray) -> None:
        """Crop should preserve uint8 dtype."""
        box = np.array([100, 100, 300, 400, 0.9, 0])

        crop = extract_crop(sample_image, box)

        assert crop.dtype == np.uint8

    def test_extract_crop_clips_to_bounds(self, sample_image: np.ndarray) -> None:
        """Crop should clip to image bounds."""
        # Box extends beyond image (1920 width, 1080 height)
        box = np.array([1800, 1000, 2000, 1200, 0.9, 0])

        crop = extract_crop(sample_image, box)

        # Should be clipped to (1920-1800) x (1080-1000) = 120 x 80
        assert crop.shape == (80, 120, 3)

    def test_extract_crop_zero_size_box(self, sample_image: np.ndarray) -> None:
        """Should handle zero-size boxes gracefully."""
        box = np.array([100, 100, 100, 100, 0.9, 0])  # x1==x2, y1==y2

        crop = extract_crop(sample_image, box)

        # Should return minimal valid crop
        assert crop.shape == (1, 1, 3)

    def test_extract_crop_negative_coords(self, sample_image: np.ndarray) -> None:
        """Should handle negative coordinates by clipping to 0."""
        box = np.array([-50, -50, 100, 100, 0.9, 0])

        crop = extract_crop(sample_image, box)

        # Negative coords clipped to 0, so crop is 100x100
        assert crop.shape == (100, 100, 3)

    def test_extract_crop_returns_copy(self, sample_image: np.ndarray) -> None:
        """Crop should be a copy, not a view."""
        box = np.array([100, 100, 200, 200, 0.9, 0])

        crop = extract_crop(sample_image, box)
        crop[:] = 0  # Modify crop

        # Original should be unchanged
        assert sample_image[100:200, 100:200].sum() > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestPreprocessingPipeline:
    """Integration tests for complete preprocessing pipeline."""

    def test_yolo_to_mobilenet_pipeline(
        self,
        sample_image: np.ndarray,
    ) -> None:
        """Test complete YOLO → crop → MobileNet pipeline."""
        yolo_preprocessor = YOLOPreprocessor()
        mobilenet_preprocessor = MobileNetPreprocessor()

        # Step 1: YOLO preprocessing
        yolo_result = yolo_preprocessor(sample_image)
        assert yolo_result.tensor.shape == (1, 3, 640, 640)

        # Step 2: Simulate YOLO detection output (in letterbox space)
        mock_detections = np.array(
            [[200, 200, 400, 400, 0.95, 0]],
            dtype=np.float32,
        )

        # Step 3: Scale boxes to original coordinates
        scaled_boxes = yolo_result.scale_boxes_to_original(mock_detections)

        # Step 4: Extract crop from original image
        crop = extract_crop(sample_image, scaled_boxes[0])

        # Step 5: MobileNet preprocessing
        mobilenet_result = mobilenet_preprocessor(crop)
        assert mobilenet_result.tensor.shape == (1, 3, 224, 224)

    def test_reproducibility(self, sample_image: np.ndarray) -> None:
        """Same input should produce identical output."""
        preprocessor = YOLOPreprocessor()

        result1 = preprocessor(sample_image)
        result2 = preprocessor(sample_image)

        assert np.array_equal(result1.tensor, result2.tensor)
        assert result1.scale == result2.scale
        assert result1.padding == result2.padding
