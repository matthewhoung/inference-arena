"""Dataset Curation Manifest Module.

This module contains the DatasetCurator class for curating
thesis test datasets from COCO val2017 with controlled fan-out
(detection count per image).

The curation process:
1. Run YOLOv5n inference on each COCO image
2. Count detections above confidence threshold
3. Select images with exactly 3-5 detections
4. Sample 100 images to achieve target distribution
5. Generate manifest with statistics for reproducibility

Controlling fan-out ensures that workload variance is not a
confounding variable in the architectural comparison.

Author: Matthew Hong
Specification Reference: experiment.yaml controlled_variables.dataset
"""

import logging
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from shared.data.coco_dataset import (
    get_coco_image_paths,
    load_coco_image,
)

from .sampling import DetectionCounter
from .types import (
    TARGET_MEAN_DETECTIONS,
    TARGET_STD_DETECTIONS,
    CurationConfig,
    CurationResult,
    DatasetManifest,
    ImageRecord,
)

logger = logging.getLogger(__name__)


class DatasetCurator:
    """Curates thesis test dataset from COCO val2017.

    Selects images with controlled detection counts (fan-out)
    to ensure consistent workload across experimental runs.

    Example:
        >>> curator = DatasetCurator(
        ...     data_dir=Path("data/"),
        ...     models_dir=Path("models/"),
        ...     output_dir=Path("data/thesis_test_set/"),
        ... )
        >>> result = curator.curate()
        >>> result.total_selected
        100
    """

    def __init__(
        self,
        data_dir: Path,
        models_dir: Path,
        output_dir: Path,
        config: CurationConfig | None = None,
    ) -> None:
        """Initialize curator.

        Args:
            data_dir: Base data directory (contains coco/val2017/)
            models_dir: Directory containing ONNX models
            output_dir: Output directory for curated dataset
            config: Curation configuration
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.config = config or CurationConfig()

        self._counter = DetectionCounter(
            models_dir=self.models_dir,
            confidence_threshold=self.config.confidence_threshold,
            iou_threshold=self.config.iou_threshold,
        )

    def is_curated(self) -> tuple[bool, str]:
        """Check if dataset is already curated.

        Returns:
            Tuple of (is_ready, message)
        """
        manifest_path = self.output_dir / "manifest.json"

        if not manifest_path.exists():
            return False, "Manifest not found"

        try:
            manifest = DatasetManifest.load(manifest_path)
            image_count = manifest.statistics.get("total_images", 0)
            mean_det = manifest.statistics.get("mean_detections", 0)

            if image_count < self.config.target_count:
                return False, f"Incomplete ({image_count}/{self.config.target_count})"

            # Verify images exist
            jpg_count = len(list(self.output_dir.glob("*.jpg")))
            if jpg_count < image_count:
                return False, f"Missing images ({jpg_count}/{image_count})"

            return True, f"Found ({image_count} images, mu={mean_det:.2f})"

        except Exception as e:
            return False, f"Invalid manifest: {e}"

    def curate(
        self,
        force: bool = False,
        progress_callback: Callable | None = None,
    ) -> CurationResult:
        """Curate thesis test dataset.

        Scans COCO images, counts detections, and selects images
        matching the configured detection range.

        Args:
            force: Re-curate even if dataset exists
            progress_callback: Called with (current, total) for progress updates

        Returns:
            CurationResult with selected images and statistics

        Raises:
            FileNotFoundError: If COCO images or model not found
        """
        # Check if already curated
        if not force:
            ready, msg = self.is_curated()
            if ready:
                logger.info(f"Dataset already curated: {msg}")
                return self._load_existing_result()

        logger.info("Curating thesis dataset...")
        logger.info(f"  Target: {self.config.target_count} images")
        logger.info(f"  Detection range: {self.config.min_detections}-{self.config.max_detections}")
        logger.info(f"  Confidence threshold: {self.config.confidence_threshold}")
        logger.info(f"  IoU threshold: {self.config.iou_threshold}")

        # Get all COCO images
        image_paths = get_coco_image_paths(self.data_dir)
        total_images = len(image_paths)

        logger.info(f"  Scanning {total_images} COCO images...")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track results by detection count for balanced sampling
        candidates: dict[int, list[ImageRecord]] = {
            i: []
            for i in range(
                self.config.min_detections,
                self.config.max_detections + 1,
            )
        }

        result = CurationResult()

        # Scan images
        for i, image_path in enumerate(image_paths):
            # Progress update
            if progress_callback:
                progress_callback(i + 1, total_images)
            elif (i + 1) % 500 == 0 or (i + 1) == total_images:
                candidate_count = sum(len(v) for v in candidates.values())
                print(
                    f"\r  Scanning: {i + 1}/{total_images} images, "
                    f"candidates: {candidate_count} "
                    f"(3:{len(candidates.get(3, []))} 4:{len(candidates.get(4, []))} 5:{len(candidates.get(5, []))})",
                    end="",
                    flush=True,
                )

            result.total_scanned += 1

            try:
                # Load image
                image = load_coco_image(image_path)

                # Count detections
                count = self._counter.count_detections(image)

                # Filter by detection range
                if count < self.config.min_detections:
                    result.skipped_low += 1
                    continue
                elif count > self.config.max_detections:
                    result.skipped_high += 1
                    continue

                # Add to candidates
                record = ImageRecord(
                    filename=image_path.name,
                    detection_count=count,
                    original_path=str(image_path),
                )
                candidates[count].append(record)

            except Exception as e:
                logger.debug(f"Error processing {image_path}: {e}")
                result.errors += 1
                continue

        print()  # New line after progress

        # Sample to achieve target distribution
        selected = self._sample_balanced(candidates)

        # Copy selected images to output directory
        for record in selected:
            assert record.original_path is not None  # Set during curation
            src_path = Path(record.original_path)
            dst_path = self.output_dir / record.filename
            shutil.copy(src_path, dst_path)
            result.images.append(record)

        result.total_selected = len(selected)

        # Generate and save manifest
        manifest = self._generate_manifest(result)
        manifest.save(self.output_dir / "manifest.json")

        # Log summary
        self._log_summary(result, manifest)

        return result

    def _sample_balanced(
        self,
        candidates: dict[int, list[ImageRecord]],
    ) -> list[ImageRecord]:
        """Sample images to achieve target distribution.

        Calculates distribution based on target_std from experiment.yaml.
        For detection range [3, 4, 5] with mean=4.0 and target_std:
        - variance = std^2 = 2a/n where a = count at extremes (3 and 5)
        - a = n * std^2 / 2, b = n - 2a (count at middle value 4)
        """
        np.random.seed(self.config.random_seed)

        selected = []
        n = self.config.target_count

        detection_range = list(
            range(
                self.config.min_detections,
                self.config.max_detections + 1,
            )
        )

        # Calculate distribution based on target std from experiment.yaml
        # For symmetric distribution {a, b, a} around mean:
        # variance = 2a/n, so a = n * variance / 2 = n * std^2 / 2
        target_std = TARGET_STD_DETECTIONS
        target_variance = target_std**2
        extreme_count = int(round(n * target_variance / 2))

        # Ensure we don't exceed total count
        extreme_count = min(extreme_count, n // 2)
        middle_count = n - 2 * extreme_count

        # Build targets dict: {3: extreme, 4: middle, 5: extreme}
        targets = {}
        mid = (self.config.min_detections + self.config.max_detections) / 2
        for d in detection_range:
            if d == mid:
                targets[d] = middle_count
            else:
                targets[d] = extreme_count

        logger.info(f"  Sampling targets: {targets}")

        # Sample from each bucket
        for det_count, target in targets.items():
            available = candidates.get(det_count, [])
            if len(available) == 0:
                logger.warning(f"  No candidates with {det_count} detections")
                continue

            sample_count = min(target, len(available))
            indices = np.random.choice(
                len(available),
                size=sample_count,
                replace=False,
            )
            selected.extend([available[i] for i in indices])
            logger.info(f"  Selected {sample_count}/{target} images with {det_count} detections")

        # If we don't have enough, sample more from available buckets
        while len(selected) < self.config.target_count:
            for det_count in detection_range:
                available = candidates.get(det_count, [])
                already_selected = {r.filename for r in selected}
                remaining_available = [r for r in available if r.filename not in already_selected]
                if remaining_available:
                    selected.append(remaining_available[0])
                    if len(selected) >= self.config.target_count:
                        break

            # Break if no more candidates
            all_available = sum(len(v) for v in candidates.values())
            if len(selected) >= all_available:
                break

        return selected[: self.config.target_count]

    def _generate_manifest(self, result: CurationResult) -> DatasetManifest:
        """Generate manifest from curation result."""
        counts = [img.detection_count for img in result.images]

        if counts:
            mean_det = sum(counts) / len(counts)
            variance = sum((x - mean_det) ** 2 for x in counts) / len(counts)
            std_det = variance**0.5
            min_det = min(counts)
            max_det = max(counts)
        else:
            mean_det = std_det = min_det = max_det = 0

        # Count distribution
        distribution: dict[int, int] = {}
        for count in counts:
            distribution[count] = distribution.get(count, 0) + 1

        manifest = DatasetManifest(
            version="1.0",
            created=datetime.now(UTC).isoformat(),
            source="COCO val2017",
            config={
                "target_count": self.config.target_count,
                "min_detections": self.config.min_detections,
                "max_detections": self.config.max_detections,
                "confidence_threshold": self.config.confidence_threshold,
                "iou_threshold": self.config.iou_threshold,
                "random_seed": self.config.random_seed,
            },
            statistics={
                "total_images": len(result.images),
                "mean_detections": round(mean_det, 2),
                "std_detections": round(std_det, 2),
                "min_detections": min_det,
                "max_detections": max_det,
            },
            distribution={str(k): v for k, v in sorted(distribution.items())},
            images=[
                {"filename": img.filename, "detections": img.detection_count}
                for img in result.images
            ],
        )

        return manifest

    def _load_existing_result(self) -> CurationResult:
        """Load result from existing manifest."""
        manifest = DatasetManifest.load(self.output_dir / "manifest.json")

        result = CurationResult()
        result.total_selected = manifest.statistics.get("total_images", 0)

        for img_data in manifest.images:
            result.images.append(
                ImageRecord(
                    filename=img_data["filename"],
                    detection_count=img_data["detections"],
                )
            )

        return result

    def _log_summary(self, result: CurationResult, manifest: DatasetManifest) -> None:
        """Log curation summary."""
        stats = manifest.statistics

        logger.info("")
        logger.info("  Curation complete!")
        logger.info("  " + "-" * 40)
        logger.info(f"  Total scanned:     {result.total_scanned}")
        logger.info(f"  Skipped (low):     {result.skipped_low}")
        logger.info(f"  Skipped (high):    {result.skipped_high}")
        logger.info(f"  Errors:            {result.errors}")
        logger.info(f"  Selected:          {result.total_selected}")
        logger.info("")
        logger.info(
            f"  Mean detections:   {stats['mean_detections']:.2f} (target: {TARGET_MEAN_DETECTIONS})"
        )
        logger.info(
            f"  Std detections:    {stats['std_detections']:.2f} (target: ~{TARGET_STD_DETECTIONS})"
        )
        logger.info(f"  Distribution:      {manifest.distribution}")
        logger.info(f"  Output:            {self.output_dir}")
