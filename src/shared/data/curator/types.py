"""Curator Types and Configuration.

This module contains:
- Configuration constants loaded from experiment.yaml
- Data classes for curation (CurationConfig, ImageRecord, CurationResult, DatasetManifest)

All parameters are loaded from experiment.yaml (single source of truth).

Author: Matthew Hong
Specification Reference: experiment.yaml controlled_variables.dataset
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from shared.config import get_controlled_variable, get_model_config

# =============================================================================
# Configuration from experiment.yaml
# =============================================================================


def _get_dataset_config() -> tuple:
    """Load dataset configuration from experiment.yaml."""
    return (
        get_controlled_variable("dataset", "sample_size"),
        get_controlled_variable("dataset", "detection_range"),
        get_controlled_variable("dataset", "target_distribution"),
        get_controlled_variable("dataset", "random_seed"),
    )


def _get_yolo_thresholds() -> tuple[float, float]:
    """Load YOLO thresholds from experiment.yaml."""
    yolo_config = get_model_config("yolov5n")
    return yolo_config.get("confidence_threshold", 0.5), yolo_config.get("iou_threshold", 0.45)


# Load defaults from experiment.yaml
_sample_size, _detection_range, _target_dist, _random_seed = _get_dataset_config()
_conf_threshold, _iou_threshold = _get_yolo_thresholds()

DEFAULT_TARGET_COUNT: int = _sample_size
"""Default number of images to curate (from experiment.yaml)."""

DEFAULT_MIN_DETECTIONS: int = _detection_range["min"]
"""Minimum detections per image (from experiment.yaml)."""

DEFAULT_MAX_DETECTIONS: int = _detection_range["max"]
"""Maximum detections per image (from experiment.yaml)."""

DEFAULT_CONFIDENCE_THRESHOLD: float = _conf_threshold
"""Minimum confidence score for valid detection (from experiment.yaml)."""

DEFAULT_IOU_THRESHOLD: float = _iou_threshold
"""IoU threshold for NMS (from experiment.yaml)."""

TARGET_MEAN_DETECTIONS: float = _target_dist["mean"]
"""Target mean detections per image (from experiment.yaml)."""

TARGET_STD_DETECTIONS: float = _target_dist["std"]
"""Target standard deviation of detections (from experiment.yaml)."""

DEFAULT_RANDOM_SEED: int = _random_seed
"""Random seed for reproducible sampling (from experiment.yaml)."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CurationConfig:
    """Configuration for dataset curation.

    All defaults are loaded from experiment.yaml (single source of truth).

    Attributes:
        target_count: Number of images to curate
        min_detections: Minimum detections per image (inclusive)
        max_detections: Maximum detections per image (inclusive)
        confidence_threshold: Minimum confidence for valid detection
        iou_threshold: IoU threshold for NMS
        random_seed: Random seed for reproducible sampling
    """

    target_count: int = DEFAULT_TARGET_COUNT
    min_detections: int = DEFAULT_MIN_DETECTIONS
    max_detections: int = DEFAULT_MAX_DETECTIONS
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    random_seed: int = DEFAULT_RANDOM_SEED


@dataclass
class ImageRecord:
    """Record for a curated image.

    Attributes:
        filename: Image filename (e.g., "000000001234.jpg")
        detection_count: Number of detections in image
        original_path: Original path in COCO dataset
    """

    filename: str
    detection_count: int
    original_path: str | None = None


@dataclass
class CurationResult:
    """Result of curation process.

    Attributes:
        images: List of curated image records
        total_scanned: Number of images scanned
        total_selected: Number of images selected
        skipped_low: Count of images with too few detections
        skipped_high: Count of images with too many detections
        errors: Count of images that failed to process
    """

    images: list[ImageRecord] = field(default_factory=list)
    total_scanned: int = 0
    total_selected: int = 0
    skipped_low: int = 0
    skipped_high: int = 0
    errors: int = 0


@dataclass
class DatasetManifest:
    """Manifest for curated dataset.

    Contains all metadata needed for reproducibility.

    Attributes:
        version: Manifest format version
        created: ISO timestamp of creation
        source: Source dataset name
        config: Curation configuration used
        statistics: Dataset statistics (mean, std, etc.)
        distribution: Count of images per detection count
        images: List of image records
    """

    version: str = "1.0"
    created: str = ""
    source: str = "COCO val2017"
    config: dict = field(default_factory=dict)
    statistics: dict = field(default_factory=dict)
    distribution: dict = field(default_factory=dict)
    images: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        """Load manifest from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
