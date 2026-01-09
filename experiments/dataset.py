"""Test dataset loader for load testing.

This module provides the TestDataset class that preloads curated COCO images
from the thesis test set into memory for efficient load testing.

The dataset contains 100 images with 3-5 detections each (μ=4, σ≈0.8),
providing consistent workload across all test runs.

Usage:
    from experiments.dataset import TestDataset

    dataset = TestDataset()
    filename, image_bytes = dataset.get_random_image()

Author: Matthew Hong
Specification Reference: experiment.yaml controlled_variables.dataset
"""

import json
import logging
import random
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Default data directory
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "thesis_test_set"


class ImageData(NamedTuple):
    """Container for image data."""

    filename: str
    data: bytes
    detections: int


class TestDataset:
    """Preloaded test images for load testing.

    This class loads all curated test images into memory at initialization,
    eliminating disk I/O during load tests for consistent performance.

    Attributes:
        data_dir: Path to the thesis_test_set directory
        images: List of (filename, bytes, detections) tuples

    Example:
        >>> dataset = TestDataset()
        >>> len(dataset)
        100
        >>> filename, data = dataset.get_random_image()
        >>> isinstance(data, bytes)
        True
    """

    def __init__(self, data_dir: Path | str | None = None, preload: bool = True):
        """Initialize the test dataset.

        Args:
            data_dir: Path to thesis_test_set directory. Defaults to
                      data/thesis_test_set relative to project root.
            preload: Whether to load all images into memory immediately.
                     Set to False for lazy loading (not recommended for load tests).

        Raises:
            FileNotFoundError: If data directory or manifest.json not found
            ValueError: If manifest.json is invalid
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self._images: list[ImageData] = []
        self._manifest: dict | None = None

        # Load manifest
        self._load_manifest()

        # Preload images if requested
        if preload:
            self._preload_images()

    def _load_manifest(self) -> None:
        """Load manifest.json containing image metadata."""
        manifest_path = self.data_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                "Run 'python scripts/setup/download-data.py' to create the test dataset."
            )

        with open(manifest_path) as f:
            self._manifest = json.load(f)

        # Validate manifest structure
        if "images" not in self._manifest:
            raise ValueError(f"Invalid manifest: missing 'images' key in {manifest_path}")

        logger.info(
            f"Loaded manifest: {len(self._manifest['images'])} images, "
            f"mean={self._manifest.get('statistics', {}).get('mean_detections', 'N/A')} detections"
        )

    def _preload_images(self) -> None:
        """Load all images into memory."""
        if self._manifest is None:
            raise RuntimeError("Manifest not loaded. Call _load_manifest() first.")

        total_bytes = 0
        missing_files = []

        for img_info in self._manifest["images"]:
            filename = img_info["filename"]
            detections = img_info.get("detections", 0)
            image_path = self.data_dir / filename

            if not image_path.exists():
                missing_files.append(filename)
                continue

            # Read image bytes
            with open(image_path, "rb") as f:
                data = f.read()

            self._images.append(ImageData(filename=filename, data=data, detections=detections))
            total_bytes += len(data)

        if missing_files:
            logger.warning(f"Missing {len(missing_files)} image files: {missing_files[:5]}...")

        # Log memory usage
        memory_mb = total_bytes / (1024 * 1024)
        logger.info(f"Preloaded {len(self._images)} images ({memory_mb:.2f} MB) into memory")

    def get_random_image(self) -> tuple[str, bytes]:
        """Get a random image for load testing.

        Returns:
            Tuple of (filename, image_bytes)

        Raises:
            RuntimeError: If no images are loaded
        """
        if not self._images:
            raise RuntimeError("No images loaded. Ensure data directory exists and preload=True.")

        img = random.choice(self._images)
        return img.filename, img.data

    def get_image_by_index(self, index: int) -> tuple[str, bytes]:
        """Get a specific image by index.

        Args:
            index: Image index (0-based)

        Returns:
            Tuple of (filename, image_bytes)

        Raises:
            IndexError: If index out of range
        """
        if index < 0 or index >= len(self._images):
            raise IndexError(f"Image index {index} out of range [0, {len(self._images)})")

        img = self._images[index]
        return img.filename, img.data

    def __len__(self) -> int:
        """Return the number of loaded images."""
        return len(self._images)

    def __iter__(self):
        """Iterate over all images."""
        for img in self._images:
            yield img.filename, img.data

    @property
    def manifest(self) -> dict | None:
        """Return the loaded manifest."""
        return self._manifest

    @property
    def statistics(self) -> dict:
        """Return dataset statistics from manifest."""
        if self._manifest is None:
            return {}
        return self._manifest.get("statistics", {})

    def get_memory_usage_mb(self) -> float:
        """Calculate total memory usage of loaded images.

        Returns:
            Memory usage in megabytes
        """
        total_bytes = sum(len(img.data) for img in self._images)
        return total_bytes / (1024 * 1024)


# Module-level singleton for shared access
_dataset_instance: TestDataset | None = None


def get_dataset() -> TestDataset:
    """Get or create the shared dataset instance.

    This function provides a module-level singleton for the test dataset,
    useful for sharing across multiple Locust users.

    Returns:
        Shared TestDataset instance
    """
    global _dataset_instance
    if _dataset_instance is None:
        _dataset_instance = TestDataset()
    return _dataset_instance


def reset_dataset() -> None:
    """Reset the shared dataset instance.

    Useful for testing or when data directory changes.
    """
    global _dataset_instance
    _dataset_instance = None
