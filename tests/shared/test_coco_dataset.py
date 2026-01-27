"""
Unit Tests for COCO Dataset Module

This module tests coco_dataset.py functions:
- DownloadProgressBar class
- is_coco_downloaded state detection edge cases
- COCO annotation structure validation

Test categories:
- DownloadProgressBar state and calculations
- State detection edge cases (non-jpg files, extra images, message format)
- Annotation structure validation using synthetic fixture

Author: Matthew Hong
Specification Reference: Foundation Specification §5
"""

import json
import tempfile
from pathlib import Path

import pytest

from shared.data.coco_dataset import (
    COCO_VAL2017_COUNT,
    DownloadProgressBar,
    is_coco_downloaded,
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
def synthetic_coco_annotations() -> dict:
    """Load synthetic COCO annotations fixture.

    Returns:
        Parsed COCO annotations dictionary with images, annotations, categories.
    """
    fixture_path = Path(__file__).parent.parent / "fixtures" / "coco" / "synthetic_annotations.json"
    with open(fixture_path) as f:
        return json.load(f)


# =============================================================================
# Tests for DownloadProgressBar
# =============================================================================


class TestDownloadProgressBar:
    """Tests for DownloadProgressBar class."""

    def test_init_stores_total_size(self) -> None:
        """Should store total_size_mb on initialization."""
        progress = DownloadProgressBar(total_size_mb=778.0)

        assert progress.total_size_mb == 778.0
        assert progress.downloaded == 0
        assert progress.last_percent == -1

    def test_call_updates_downloaded_count(self) -> None:
        """Should update downloaded count when called."""
        progress = DownloadProgressBar(total_size_mb=100.0)

        # Simulate downloading first block
        progress(block_num=0, block_size=1024, total_size=104857600)  # 100 MB

        assert progress.downloaded == 1024

    def test_call_accumulates_downloads(self) -> None:
        """Should accumulate downloaded bytes across multiple calls."""
        progress = DownloadProgressBar(total_size_mb=100.0)

        # Simulate downloading multiple blocks
        progress(block_num=0, block_size=1024, total_size=104857600)
        progress(block_num=1, block_size=1024, total_size=104857600)
        progress(block_num=2, block_size=1024, total_size=104857600)

        assert progress.downloaded == 3072  # 3 * 1024

    def test_percentage_calculation_with_known_total(self) -> None:
        """Should calculate percentage from actual total_size when known."""
        progress = DownloadProgressBar(total_size_mb=100.0)

        # Download 50% of 100 MB file
        total_size = 104857600  # 100 MB in bytes
        progress.downloaded = total_size // 2

        # Simulate callback
        progress(block_num=50, block_size=0, total_size=total_size)

        # Last percent should be updated to 50
        assert progress.last_percent == 50

    def test_percentage_calculation_with_unknown_total(self) -> None:
        """Should estimate percentage from total_size_mb when total_size=-1."""
        progress = DownloadProgressBar(total_size_mb=100.0)

        # Simulate downloading 50 MB worth
        progress.downloaded = 50 * 1024 * 1024  # 50 MB

        # Callback with unknown total size
        progress(block_num=50, block_size=0, total_size=-1)

        # Should use fallback calculation: 50MB/100MB = 50%
        assert progress.last_percent == 50

    def test_percentage_capped_at_99_for_unknown_total(self) -> None:
        """Should cap percentage at 99% when using estimated total."""
        progress = DownloadProgressBar(total_size_mb=100.0)

        # Simulate downloading more than expected
        progress.downloaded = 150 * 1024 * 1024  # 150 MB (more than 100 MB estimate)

        progress(block_num=150, block_size=0, total_size=-1)

        # Should cap at 99% since we don't know actual completion
        assert progress.last_percent == 99


# =============================================================================
# Tests for is_coco_downloaded Edge Cases
# =============================================================================


class TestIsCOCODownloadedExtended:
    """Extended tests for is_coco_downloaded function edge cases."""

    def test_handles_non_jpg_files(self, temp_dir: Path) -> None:
        """Should return False if directory only contains non-jpg files."""
        coco_dir = temp_dir / "coco" / "val2017"
        coco_dir.mkdir(parents=True)

        # Create only .txt files (not .jpg)
        for i in range(100):
            (coco_dir / f"{i:012d}.txt").touch()

        ready, msg = is_coco_downloaded(temp_dir)

        assert ready is False
        assert "incomplete" in msg.lower()

    def test_handles_mixed_file_types(self, temp_dir: Path) -> None:
        """Should only count .jpg files, ignoring other file types."""
        coco_dir = temp_dir / "coco" / "val2017"
        coco_dir.mkdir(parents=True)

        # Create mix of files: 50 jpg, 50 txt
        for i in range(50):
            (coco_dir / f"{i:012d}.jpg").touch()
            (coco_dir / f"{i:012d}.txt").touch()

        ready, msg = is_coco_downloaded(temp_dir)

        assert ready is False
        # Should report 50 jpg images, not 100 total files
        assert "50" in msg

    def test_more_than_expected_images(self, temp_dir: Path) -> None:
        """Should return True if more than expected images present."""
        coco_dir = temp_dir / "coco" / "val2017"
        coco_dir.mkdir(parents=True)

        # Create more than COCO_VAL2017_COUNT images
        for i in range(COCO_VAL2017_COUNT + 100):
            (coco_dir / f"{i:012d}.jpg").touch()

        ready, msg = is_coco_downloaded(temp_dir)

        assert ready is True
        assert str(COCO_VAL2017_COUNT + 100) in msg

    def test_returns_accurate_count_message(self, temp_dir: Path) -> None:
        """Message should include actual image count."""
        coco_dir = temp_dir / "coco" / "val2017"
        coco_dir.mkdir(parents=True)

        # Create specific count
        specific_count = 1234
        for i in range(specific_count):
            (coco_dir / f"{i:012d}.jpg").touch()

        ready, msg = is_coco_downloaded(temp_dir)

        assert ready is False
        assert "1234" in msg
        assert str(COCO_VAL2017_COUNT) in msg

    def test_empty_directory_returns_zero_count(self, temp_dir: Path) -> None:
        """Should handle empty directory with zero count message."""
        coco_dir = temp_dir / "coco" / "val2017"
        coco_dir.mkdir(parents=True)

        ready, msg = is_coco_downloaded(temp_dir)

        assert ready is False
        assert "0" in msg or "incomplete" in msg.lower()


# =============================================================================
# Tests for COCO Annotation Structure
# =============================================================================


class TestCOCOAnnotationStructure:
    """Tests for COCO annotation structure validation using synthetic fixture."""

    def test_annotation_has_required_fields(self, synthetic_coco_annotations: dict) -> None:
        """Each annotation should have required COCO fields."""
        required_fields = {"id", "image_id", "category_id", "bbox", "area", "iscrowd"}

        for annotation in synthetic_coco_annotations["annotations"]:
            assert required_fields.issubset(
                annotation.keys()
            ), f"Annotation missing fields: {required_fields - annotation.keys()}"

    def test_bbox_format_is_xywh(self, synthetic_coco_annotations: dict) -> None:
        """Bbox should have exactly 4 elements (x, y, width, height)."""
        for annotation in synthetic_coco_annotations["annotations"]:
            bbox = annotation["bbox"]

            assert isinstance(bbox, list), "bbox should be a list"
            assert len(bbox) == 4, f"bbox should have 4 elements, got {len(bbox)}"

            # All values should be non-negative numbers
            for i, val in enumerate(bbox):
                assert isinstance(
                    val, (int, float)
                ), f"bbox[{i}] should be numeric, got {type(val)}"
                assert val >= 0, f"bbox[{i}] should be non-negative, got {val}"

    def test_bbox_width_height_positive(self, synthetic_coco_annotations: dict) -> None:
        """Bbox width and height (elements 2, 3) should be positive."""
        for annotation in synthetic_coco_annotations["annotations"]:
            bbox = annotation["bbox"]
            width, height = bbox[2], bbox[3]

            assert width > 0, f"bbox width should be positive, got {width}"
            assert height > 0, f"bbox height should be positive, got {height}"

    def test_image_dimensions_are_positive(self, synthetic_coco_annotations: dict) -> None:
        """All image dimensions should be positive integers."""
        for image in synthetic_coco_annotations["images"]:
            assert "width" in image, "image missing width"
            assert "height" in image, "image missing height"
            assert image["width"] > 0, f"width should be positive: {image['width']}"
            assert image["height"] > 0, f"height should be positive: {image['height']}"

    def test_image_has_required_fields(self, synthetic_coco_annotations: dict) -> None:
        """Each image entry should have required fields."""
        required_fields = {"id", "file_name", "width", "height"}

        for image in synthetic_coco_annotations["images"]:
            assert required_fields.issubset(
                image.keys()
            ), f"Image missing fields: {required_fields - image.keys()}"

    def test_category_ids_are_valid(self, synthetic_coco_annotations: dict) -> None:
        """Annotation category_ids should reference existing categories."""
        valid_category_ids = {cat["id"] for cat in synthetic_coco_annotations["categories"]}

        for annotation in synthetic_coco_annotations["annotations"]:
            category_id = annotation["category_id"]
            assert (
                category_id in valid_category_ids
            ), f"Invalid category_id {category_id}, valid: {valid_category_ids}"

    def test_image_ids_are_valid(self, synthetic_coco_annotations: dict) -> None:
        """Annotation image_ids should reference existing images."""
        valid_image_ids = {img["id"] for img in synthetic_coco_annotations["images"]}

        for annotation in synthetic_coco_annotations["annotations"]:
            image_id = annotation["image_id"]
            assert (
                image_id in valid_image_ids
            ), f"Invalid image_id {image_id}, valid: {valid_image_ids}"

    def test_area_matches_bbox_dimensions(self, synthetic_coco_annotations: dict) -> None:
        """Area should equal bbox width * height."""
        for annotation in synthetic_coco_annotations["annotations"]:
            bbox = annotation["bbox"]
            expected_area = bbox[2] * bbox[3]  # width * height
            actual_area = annotation["area"]

            assert actual_area == expected_area, f"Area {actual_area} != bbox area {expected_area}"

    def test_annotation_ids_unique(self, synthetic_coco_annotations: dict) -> None:
        """All annotation IDs should be unique."""
        ids = [ann["id"] for ann in synthetic_coco_annotations["annotations"]]
        assert len(ids) == len(set(ids)), "Duplicate annotation IDs found"

    def test_image_ids_unique(self, synthetic_coco_annotations: dict) -> None:
        """All image IDs should be unique."""
        ids = [img["id"] for img in synthetic_coco_annotations["images"]]
        assert len(ids) == len(set(ids)), "Duplicate image IDs found"

    def test_categories_have_required_fields(self, synthetic_coco_annotations: dict) -> None:
        """Each category should have id, name, supercategory."""
        required_fields = {"id", "name", "supercategory"}

        for category in synthetic_coco_annotations["categories"]:
            assert required_fields.issubset(
                category.keys()
            ), f"Category missing fields: {required_fields - category.keys()}"

    def test_fixture_has_expected_counts(self, synthetic_coco_annotations: dict) -> None:
        """Synthetic fixture should have expected number of entries."""
        assert len(synthetic_coco_annotations["images"]) == 5
        assert len(synthetic_coco_annotations["annotations"]) == 15
        assert len(synthetic_coco_annotations["categories"]) == 4
