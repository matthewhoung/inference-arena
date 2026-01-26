"""Dataset Curator Package.

This package provides dataset curation utilities for creating thesis test
datasets from COCO val2017 with controlled detection counts (fan-out).

The package is split into three modules:
- types.py: Configuration constants and data classes
- sampling.py: Detection counting using YOLOv5n ONNX model
- manifest.py: Dataset curation orchestration

The curation process:
1. Run YOLOv5n inference on each COCO image
2. Count detections above confidence threshold
3. Select images with exactly 3-5 detections
4. Sample 100 images to achieve target distribution (mu=4, sigma~0.8)
5. Generate manifest with statistics for reproducibility

Controlling fan-out ensures that workload variance is not a
confounding variable in the architectural comparison.

All parameters are loaded from experiment.yaml (single source of truth).

Author: Matthew Hong
Specification Reference: experiment.yaml controlled_variables.dataset
"""

from .manifest import DatasetCurator
from .sampling import DetectionCounter
from .types import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
    DEFAULT_MIN_DETECTIONS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TARGET_COUNT,
    TARGET_MEAN_DETECTIONS,
    TARGET_STD_DETECTIONS,
    CurationConfig,
    CurationResult,
    DatasetManifest,
    ImageRecord,
)

__all__ = [
    # Constants
    "DEFAULT_TARGET_COUNT",
    "DEFAULT_MIN_DETECTIONS",
    "DEFAULT_MAX_DETECTIONS",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_IOU_THRESHOLD",
    "TARGET_MEAN_DETECTIONS",
    "TARGET_STD_DETECTIONS",
    "DEFAULT_RANDOM_SEED",
    # Data classes
    "CurationConfig",
    "ImageRecord",
    "CurationResult",
    "DatasetManifest",
    # Classes
    "DetectionCounter",
    "DatasetCurator",
]
