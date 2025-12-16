"""
Unit Tests for Data Module

This module tests:
- coco_dataset.py: Download utilities and image loading
- curator.py: Dataset curation and manifest generation

Test Categories:
- Download state detection (no network required)
- Image loading
- Curation configuration
- Manifest serialization
- Integration tests (marked slow, require COCO + model)

Author: Matthew Hong
Specification Reference: Foundation Specification §5
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from shared.data.coco_dataset import (
    is_coco_downloaded,
    load_coco_image,
    get_coco_image_paths,
    COCO_VAL2017_URL,
    COCO_VAL2017_COUNT,
)
from shared.data.curator import (
    CurationConfig,
    CurationResult,
    ImageRecord,
    DatasetManifest,
    DatasetCurator,
    DEFAULT_TARGET_COUNT,
    DEFAULT_MIN_DETECTIONS,
    DEFAULT_MAX_DETECTIONS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    TARGET_MEAN_DETECTIONS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir() -> Path:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_coco_dir(temp_dir: Path) -> Path:
    """Create mock COCO directory structure with sample images."""
    coco_dir = temp_dir / "coco" / "val2017"
    coco_dir.mkdir(parents=True)

    # Create sample images (just valid JPEG files)
    for i in range(10):
        # Create a minimal valid image using numpy and cv2
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        image_path = coco_dir / f"00000000{i:04d}.jpg"

        import cv2
        cv2.imwrite(str(image_path), image)

    return temp_dir


@pytest.fixture
def sample_manifest_data() -> dict:
    """Sample manifest data for testing."""
    return {
        "version": "1.0",
        "created": "2024-01-01T00:00:00+00:00",
        "source": "COCO val2017",
        "config": {
            "target_count": 100,
            "min_detections": 3,
            "max_detections": 5,
            "confidence_threshold": 0.5,
            "random_seed": 42,
        },
        "statistics": {
            "total_images": 100,
            "mean_detections": 4.02,
            "std_detections": 0.78,
            "min_detections": 3,
            "max_detections": 5,
        },
        "distribution": {"3": 25, "4": 50, "5": 25},
        "images": [
            {"filename": f"00000000{i:04d}.jpg", "detections": 4}
            for i in range(100)
        ],
    }


# =============================================================================
# Tests for COCO Dataset Constants
# =============================================================================

class TestCocoConstants:
    """Tests for COCO dataset constants."""

    def test_coco_url(self) -> None:
        """COCO URL should be official images URL."""
        assert "cocodataset.org" in COCO_VAL2017_URL
        assert "val2017" in COCO_VAL2017_URL

    def test_coco_count(self) -> None:
        """Val2017 should have 5000 images."""
        assert COCO_VAL2017_COUNT == 5000


# =============================================================================
# Tests for COCO State Detection
# =============================================================================

class TestIsCOCODownloaded:
    """Tests for is_coco_downloaded function."""

    def test_not_downloaded_missing_dir(self, temp_dir: Path) -> None:
        """Should return False if directory doesn't exist."""
        ready, msg = is_coco_downloaded(temp_dir)

        assert ready is False
        assert "not found" in msg.lower()

    def test_not_downloaded_empty_dir(self, temp_dir: Path) -> None:
        """Should return False if directory is empty."""
        coco_dir = temp_dir / "coco" / "val2017"
        coco_dir.mkdir(parents=True)

        ready, msg = is_coco_downloaded(temp_dir)

        assert ready is False
        assert "incomplete" in msg.lower()

    def test_not_downloaded_partial(self, temp_dir: Path) -> None:
        """Should return False if not all images present."""
        coco_dir = temp_dir / "coco" / "val2017"
        coco_dir.mkdir(parents=True)

        # Create only 100 images
        for i in range(100):
            (coco_dir / f"{i:012d}.jpg").touch()

        ready, msg = is_coco_downloaded(temp_dir)

        assert ready is False
        assert "100" in msg

    def test_downloaded_complete(self, temp_dir: Path) -> None:
        """Should return True if all images present."""
        coco_dir = temp_dir / "coco" / "val2017"
        coco_dir.mkdir(parents=True)

        # Create 5000 images
        for i in range(5000):
            (coco_dir / f"{i:012d}.jpg").touch()

        ready, msg = is_coco_downloaded(temp_dir)

        assert ready is True
        assert "5000" in msg


# =============================================================================
# Tests for Image Loading
# =============================================================================

class TestLoadCocoImage:
    """Tests for load_coco_image function."""

    def test_load_valid_image(self, mock_coco_dir: Path) -> None:
        """Should load image as RGB numpy array."""
        image_path = mock_coco_dir / "coco" / "val2017" / "000000000000.jpg"
        image = load_coco_image(image_path)

        assert isinstance(image, np.ndarray)
        assert image.ndim == 3
        assert image.shape[2] == 3
        assert image.dtype == np.uint8

    def test_load_nonexistent_raises(self, temp_dir: Path) -> None:
        """Should raise ValueError for missing file."""
        with pytest.raises(ValueError, match="Failed to load"):
            load_coco_image(temp_dir / "nonexistent.jpg")


class TestGetCocoImagePaths:
    """Tests for get_coco_image_paths function."""

    def test_returns_sorted_paths(self, mock_coco_dir: Path) -> None:
        """Should return sorted list of paths."""
        paths = get_coco_image_paths(mock_coco_dir)

        assert len(paths) == 10
        assert all(isinstance(p, Path) for p in paths)
        # Check sorted
        filenames = [p.name for p in paths]
        assert filenames == sorted(filenames)

    def test_missing_directory_raises(self, temp_dir: Path) -> None:
        """Should raise FileNotFoundError if directory missing."""
        with pytest.raises(FileNotFoundError):
            get_coco_image_paths(temp_dir)


# =============================================================================
# Tests for CurationConfig
# =============================================================================

class TestCurationConfig:
    """Tests for CurationConfig dataclass."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        config = CurationConfig()

        assert config.target_count == DEFAULT_TARGET_COUNT
        assert config.min_detections == DEFAULT_MIN_DETECTIONS
        assert config.max_detections == DEFAULT_MAX_DETECTIONS
        assert config.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
        assert config.random_seed == 42

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        config = CurationConfig(
            target_count=50,
            min_detections=2,
            max_detections=6,
            confidence_threshold=0.7,
            random_seed=123,
        )

        assert config.target_count == 50
        assert config.min_detections == 2

    def test_default_iou_threshold(self) -> None:
        """Default IoU threshold should be 0.45."""
        config = CurationConfig()
        assert config.iou_threshold == 0.45


# =============================================================================
# Tests for ImageRecord
# =============================================================================

class TestImageRecord:
    """Tests for ImageRecord dataclass."""

    def test_create_record(self) -> None:
        """Should create record with required fields."""
        record = ImageRecord(
            filename="000000001234.jpg",
            detection_count=4,
        )

        assert record.filename == "000000001234.jpg"
        assert record.detection_count == 4
        assert record.original_path is None

    def test_create_with_path(self) -> None:
        """Should accept optional original_path."""
        record = ImageRecord(
            filename="000000001234.jpg",
            detection_count=4,
            original_path="/path/to/image.jpg",
        )

        assert record.original_path == "/path/to/image.jpg"


# =============================================================================
# Tests for CurationResult
# =============================================================================

class TestCurationResult:
    """Tests for CurationResult dataclass."""

    def test_default_values(self) -> None:
        """Should initialize with empty/zero values."""
        result = CurationResult()

        assert result.images == []
        assert result.total_scanned == 0
        assert result.total_selected == 0
        assert result.skipped_low == 0
        assert result.skipped_high == 0
        assert result.errors == 0

    def test_add_images(self) -> None:
        """Should allow adding images."""
        result = CurationResult()
        record = ImageRecord(filename="test.jpg", detection_count=4)

        result.images.append(record)
        result.total_selected += 1

        assert len(result.images) == 1
        assert result.total_selected == 1


# =============================================================================
# Tests for DatasetManifest
# =============================================================================

class TestDatasetManifest:
    """Tests for DatasetManifest dataclass."""

    def test_default_values(self) -> None:
        """Should initialize with defaults."""
        manifest = DatasetManifest()

        assert manifest.version == "1.0"
        assert manifest.source == "COCO val2017"

    def test_to_dict(self, sample_manifest_data: dict) -> None:
        """Should convert to dictionary."""
        manifest = DatasetManifest(**sample_manifest_data)
        result = manifest.to_dict()

        assert result["version"] == "1.0"
        assert result["statistics"]["mean_detections"] == 4.02

    def test_save_and_load(self, temp_dir: Path, sample_manifest_data: dict) -> None:
        """Should save and load correctly."""
        manifest = DatasetManifest(**sample_manifest_data)
        path = temp_dir / "manifest.json"

        manifest.save(path)
        loaded = DatasetManifest.load(path)

        assert loaded.version == manifest.version
        assert loaded.statistics == manifest.statistics
        assert len(loaded.images) == len(manifest.images)

    def test_save_creates_valid_json(
        self,
        temp_dir: Path,
        sample_manifest_data: dict,
    ) -> None:
        """Saved file should be valid JSON."""
        manifest = DatasetManifest(**sample_manifest_data)
        path = temp_dir / "manifest.json"

        manifest.save(path)

        # Load as raw JSON
        with open(path) as f:
            data = json.load(f)

        assert data["version"] == "1.0"


# =============================================================================
# Tests for DatasetCurator
# =============================================================================

class TestDatasetCurator:
    """Tests for DatasetCurator class."""

    def test_init(self, temp_dir: Path) -> None:
        """Should initialize with paths."""
        curator = DatasetCurator(
            data_dir=temp_dir / "data",
            models_dir=temp_dir / "models",
            output_dir=temp_dir / "output",
        )

        assert curator.data_dir == temp_dir / "data"
        assert curator.config.target_count == 100

    def test_init_with_config(self, temp_dir: Path) -> None:
        """Should accept custom config."""
        config = CurationConfig(target_count=50)
        curator = DatasetCurator(
            data_dir=temp_dir / "data",
            models_dir=temp_dir / "models",
            output_dir=temp_dir / "output",
            config=config,
        )

        assert curator.config.target_count == 50

    def test_is_curated_false_no_manifest(self, temp_dir: Path) -> None:
        """Should return False if no manifest."""
        curator = DatasetCurator(
            data_dir=temp_dir,
            models_dir=temp_dir,
            output_dir=temp_dir / "output",
        )

        ready, msg = curator.is_curated()

        assert ready is False
        assert "manifest" in msg.lower()

    def test_is_curated_true_with_manifest(
        self,
        temp_dir: Path,
        sample_manifest_data: dict,
    ) -> None:
        """Should return True if valid manifest exists."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Create manifest
        manifest = DatasetManifest(**sample_manifest_data)
        manifest.save(output_dir / "manifest.json")

        # Create image files
        for i in range(100):
            (output_dir / f"00000000{i:04d}.jpg").touch()

        curator = DatasetCurator(
            data_dir=temp_dir,
            models_dir=temp_dir,
            output_dir=output_dir,
        )

        ready, msg = curator.is_curated()

        assert ready is True
        assert "100" in msg


# =============================================================================
# Tests for Manifest Statistics
# =============================================================================

class TestManifestStatistics:
    """Tests for manifest statistics calculation."""

    def test_statistics_structure(self, sample_manifest_data: dict) -> None:
        """Statistics should have required fields."""
        manifest = DatasetManifest(**sample_manifest_data)

        assert "total_images" in manifest.statistics
        assert "mean_detections" in manifest.statistics
        assert "std_detections" in manifest.statistics
        assert "min_detections" in manifest.statistics
        assert "max_detections" in manifest.statistics

    def test_target_mean(self) -> None:
        """Target mean should be 4.0."""
        assert TARGET_MEAN_DETECTIONS == 4.0

    def test_distribution_keys(self, sample_manifest_data: dict) -> None:
        """Distribution keys should be string detection counts."""
        manifest = DatasetManifest(**sample_manifest_data)

        assert "3" in manifest.distribution
        assert "4" in manifest.distribution
        assert "5" in manifest.distribution


# =============================================================================
# Integration Tests (Slow)
# =============================================================================

class TestCurationIntegration:
    """Integration tests for full curation process."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_curate_creates_manifest(
        self,
        mock_coco_dir: Path,
        temp_dir: Path,
    ) -> None:
        """Full curation should create manifest."""
        # This test requires:
        # 1. Mock COCO images (created by fixture)
        # 2. YOLOv5n model
        # Skip if model not available
        models_dir = temp_dir / "models"
        if not (models_dir / "yolov5n.onnx").exists():
            pytest.skip("YOLOv5n model not available")

        output_dir = temp_dir / "thesis_test_set"

        curator = DatasetCurator(
            data_dir=mock_coco_dir,
            models_dir=models_dir,
            output_dir=output_dir,
            config=CurationConfig(target_count=5),  # Small for testing
        )

        result = curator.curate()

        assert (output_dir / "manifest.json").exists()
        assert result.total_selected <= 5

    @pytest.mark.slow
    @pytest.mark.integration
    def test_curate_idempotent(
        self,
        temp_dir: Path,
        sample_manifest_data: dict,
    ) -> None:
        """Curation should be idempotent."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Create existing manifest
        sample_manifest_data["statistics"]["total_images"] = 100
        manifest = DatasetManifest(**sample_manifest_data)
        manifest.save(output_dir / "manifest.json")

        # Create image files
        for i in range(100):
            (output_dir / f"00000000{i:04d}.jpg").touch()

        curator = DatasetCurator(
            data_dir=temp_dir,
            models_dir=temp_dir,
            output_dir=output_dir,
        )

        # Should not re-curate
        result = curator.curate(force=False)

        assert result.total_selected == 100


# =============================================================================
# Tests for Detection Range Validation
# =============================================================================

class TestDetectionRange:
    """Tests for detection range constants."""

    def test_min_less_than_max(self) -> None:
        """Min detections should be less than max."""
        assert DEFAULT_MIN_DETECTIONS < DEFAULT_MAX_DETECTIONS

    def test_range_includes_target_mean(self) -> None:
        """Detection range should include target mean."""
        assert DEFAULT_MIN_DETECTIONS <= TARGET_MEAN_DETECTIONS <= DEFAULT_MAX_DETECTIONS

    def test_config_allows_custom_range(self) -> None:
        """Config should allow custom detection range."""
        config = CurationConfig(
            min_detections=2,
            max_detections=8,
        )

        assert config.min_detections == 2
        assert config.max_detections == 8
