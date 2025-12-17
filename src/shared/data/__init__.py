"""Data Module - COCO Dataset Download and Curation.

This module provides:
- coco_dataset: Download and load COCO val2017 images
- curator: Curate thesis dataset with controlled detection counts

The curated dataset ensures controlled fan-out (μ=4, σ≈0.8 detections per image)
to eliminate workload variance as a confounding variable.

Specification Reference:
    Foundation Specification §5 Dataset Specification
"""

from shared.data.coco_dataset import (
    COCO_VAL2017_COUNT,
    COCO_VAL2017_URL,
    download_coco_val2017,
    get_coco_image_paths,
    is_coco_downloaded,
    load_coco_image,
)
from shared.data.curator import (
    CurationConfig,
    CurationResult,
    DatasetCurator,
    DatasetManifest,
    ImageRecord,
)

__all__ = [
    # COCO dataset
    "download_coco_val2017",
    "load_coco_image",
    "get_coco_image_paths",
    "is_coco_downloaded",
    "COCO_VAL2017_URL",
    "COCO_VAL2017_COUNT",
    # Curator
    "DatasetCurator",
    "CurationConfig",
    "CurationResult",
    "ImageRecord",
    "DatasetManifest",
]
