"""COCO Dataset Utilities.

This module provides utilities for downloading and loading COCO val2017 images.

The COCO (Common Objects in Context) val2017 dataset contains 5,000 images
with Creative Commons Attribution 4.0 license, suitable for academic research.

Functions:
    download_coco_val2017: Download and extract COCO val2017 dataset
    load_coco_image: Load a single image as RGB numpy array
    get_coco_image_paths: List all image paths in dataset
    is_coco_downloaded: Check if dataset is already downloaded

Author: Matthew Hong
Specification Reference: Foundation Specification §5.1
"""

import logging
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

COCO_VAL2017_URL: str = "http://images.cocodataset.org/zips/val2017.zip"
"""URL for COCO val2017 images zip file."""

COCO_VAL2017_COUNT: int = 5000
"""Expected number of images in val2017 dataset."""

COCO_ZIP_SIZE_MB: float = 778.0
"""Approximate size of val2017.zip in megabytes."""


# =============================================================================
# Download Progress
# =============================================================================

class DownloadProgressBar:
    """Progress bar callback for urllib downloads.

    Displays download progress to console with percentage and MB transferred.
    """

    def __init__(self, total_size_mb: float) -> None:
        """Initialize progress bar.

        Args:
            total_size_mb: Expected total size in megabytes
        """
        self.total_size_mb = total_size_mb
        self.downloaded = 0
        self.last_percent = -1

    def __call__(
        self,
        block_num: int,
        block_size: int,
        total_size: int,
    ) -> None:
        """Update progress bar.

        Args:
            block_num: Current block number
            block_size: Size of each block in bytes
            total_size: Total file size in bytes (-1 if unknown)
        """
        self.downloaded += block_size
        downloaded_mb = self.downloaded / (1024 * 1024)

        if total_size > 0:
            percent = int(100 * self.downloaded / total_size)
            total_mb = total_size / (1024 * 1024)
        else:
            percent = min(int(100 * downloaded_mb / self.total_size_mb), 99)
            total_mb = self.total_size_mb

        if percent != self.last_percent:
            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(
                f"\r  [{bar}] {percent}% - {downloaded_mb:.1f} MB / {total_mb:.1f} MB",
                end="",
                flush=True,
            )
            self.last_percent = percent

# =============================================================================
# State Detection
# =============================================================================

def is_coco_downloaded(data_dir: Path) -> tuple[bool, str]:
    """Check if COCO val2017 is already downloaded.

    Args:
        data_dir: Base data directory (parent of coco/val2017/)

    Returns:
        Tuple of (is_ready, message)
        - is_ready: True if dataset is complete
        - message: Description of current state

    Example:
        >>> ready, msg = is_coco_downloaded(Path("data/"))
        >>> ready
        True
        >>> msg
        'Found (5000 images)'
    """
    coco_dir = data_dir / "coco" / "val2017"

    if not coco_dir.exists():
        return False, "Directory not found"

    jpg_count = len(list(coco_dir.glob("*.jpg")))

    if jpg_count < COCO_VAL2017_COUNT:
        return False, f"Incomplete ({jpg_count}/{COCO_VAL2017_COUNT} images)"

    return True, f"Found ({jpg_count} images)"


# =============================================================================
# Download Functions
# =============================================================================

def download_coco_val2017(
    data_dir: Path,
    force: bool = False,
    cleanup_zip: bool = True,
) -> Path:
    """Download and extract COCO val2017 dataset.

    Downloads the dataset from the official COCO website and extracts
    to data_dir/coco/val2017/. Idempotent: skips if already downloaded.

    Args:
        data_dir: Base data directory
        force: Re-download even if exists
        cleanup_zip: Delete zip file after extraction

    Returns:
        Path to extracted images directory

    Raises:
        RuntimeError: If download or extraction fails

    Example:
        >>> images_dir = download_coco_val2017(Path("data/"))
        >>> images_dir
        PosixPath('data/coco/val2017')
        >>> len(list(images_dir.glob("*.jpg")))
        5000
    """
    coco_dir = data_dir / "coco"
    images_dir = coco_dir / "val2017"
    zip_path = coco_dir / "val2017.zip"

    # Check if already downloaded
    if not force:
        ready, msg = is_coco_downloaded(data_dir)
        if ready:
            logger.info(f"COCO val2017 already downloaded: {msg}")
            return images_dir

    # Create directory
    coco_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading COCO val2017 (~{COCO_ZIP_SIZE_MB:.0f} MB)...")
    logger.info(f"  Source: {COCO_VAL2017_URL}")
    logger.info(f"  Destination: {zip_path}")

    try:
        # Download with progress bar
        progress = DownloadProgressBar(COCO_ZIP_SIZE_MB)
        urllib.request.urlretrieve(COCO_VAL2017_URL, zip_path, progress)
        print()  # New line after progress bar

    except Exception as e:
        raise RuntimeError(f"Download failed: {e}") from e

    # Extract
    logger.info("Extracting...")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(coco_dir)

    except Exception as e:
        raise RuntimeError(f"Extraction failed: {e}") from e

    # Verify extraction
    jpg_count = len(list(images_dir.glob("*.jpg")))
    logger.info(f"  Extracted {jpg_count} images to {images_dir}")

    if jpg_count < COCO_VAL2017_COUNT:
        raise RuntimeError(
            f"Extraction incomplete: got {jpg_count}, expected {COCO_VAL2017_COUNT}"
        )

    # Cleanup zip file
    if cleanup_zip and zip_path.exists():
        logger.info("  Removing zip file to save space...")
        zip_path.unlink()

    return images_dir


# =============================================================================
# Image Loading
# =============================================================================

def load_coco_image(image_path: Path) -> np.ndarray:
    """Load a COCO image as RGB numpy array.

    Uses OpenCV for efficient JPEG decoding with BGR to RGB conversion.

    Args:
        image_path: Path to image file

    Returns:
        RGB uint8 array with shape [H, W, 3]

    Raises:
        ValueError: If image cannot be loaded

    Example:
        >>> image = load_coco_image(Path("data/coco/val2017/000000001234.jpg"))
        >>> image.shape
        (480, 640, 3)
    """
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb


def get_coco_image_paths(data_dir: Path) -> list[Path]:
    """Get sorted list of all COCO val2017 image paths.

    Args:
        data_dir: Base data directory

    Returns:
        List of Path objects for each image, sorted by filename

    Raises:
        FileNotFoundError: If COCO directory not found

    Example:
        >>> paths = get_coco_image_paths(Path("data/"))
        >>> len(paths)
        5000
        >>> paths[0].name
        '000000000139.jpg'
    """
    images_dir = data_dir / "coco" / "val2017"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"COCO images not found at {images_dir}. "
            "Run 'make setup-data' or 'python scripts/setup_data.py' first."
        )

    return sorted(images_dir.glob("*.jpg"))


def iter_coco_images(
    data_dir: Path,
    limit: int | None = None,
) -> Iterator[tuple[Path, np.ndarray]]:
    """Iterate over COCO images, yielding path and loaded image.

    Useful for processing all images without loading entire dataset into memory.

    Args:
        data_dir: Base data directory
        limit: Maximum number of images to yield (None for all)

    Yields:
        Tuple of (image_path, image_array)

    Example:
        >>> for path, image in iter_coco_images(Path("data/"), limit=10):
        ...     print(f"{path.name}: {image.shape}")
    """
    paths = get_coco_image_paths(data_dir)

    if limit is not None:
        paths = paths[:limit]

    for path in paths:
        try:
            image = load_coco_image(path)
            yield path, image
        except ValueError as e:
            logger.warning(f"Skipping {path}: {e}")
            continue
