"""Dataset Curator.

This module curates a thesis test dataset from COCO val2017 with
controlled fan-out (detection count per image).

The curation process:
1. Run YOLOv5n inference on each COCO image
2. Count detections above confidence threshold
3. Select images with exactly 3-5 detections
4. Sample 100 images to achieve target distribution (μ=4, σ≈0.8)
5. Generate manifest with statistics for reproducibility

Controlling fan-out ensures that workload variance is not a
confounding variable in the architectural comparison.

Author: Matthew Hong
Specification Reference: Foundation Specification §5.2
"""

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from shared.data.coco_dataset import (
    get_coco_image_paths,
    load_coco_image,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_TARGET_COUNT: int = 100
"""Default number of images to curate."""

DEFAULT_MIN_DETECTIONS: int = 3
"""Minimum detections per image (inclusive)."""

DEFAULT_MAX_DETECTIONS: int = 5
"""Maximum detections per image (inclusive)."""

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.25
"""Minimum confidence score for valid detection (lowered for YOLOv8 format)."""

DEFAULT_IOU_THRESHOLD: float = 0.45
"""IoU threshold for NMS."""

TARGET_MEAN_DETECTIONS: float = 4.0
"""Target mean detections per image."""

TARGET_STD_DETECTIONS: float = 0.8
"""Target standard deviation of detections."""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CurationConfig:
    """Configuration for dataset curation.

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
    random_seed: int = 42


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


# =============================================================================
# Detection Counter
# =============================================================================

class DetectionCounter:
    """Counts detections in images using YOLOv5n ONNX model.

    Handles YOLOv8-style output format [batch, 84, num_predictions]:
    - 84 = 4 (bbox) + 80 (class scores)
    - No separate objectness score
    - Requires transpose and NMS
    """

    def __init__(
        self,
        models_dir: Path,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:
        """Initialize detection counter.

        Args:
            models_dir: Directory containing ONNX models
            confidence_threshold: Minimum confidence for valid detection
            iou_threshold: IoU threshold for NMS
        """
        self.models_dir = models_dir
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self._session = None

    def _load_model(self) -> None:
        """Load YOLOv5n ONNX model."""
        if self._session is not None:
            return

        import onnxruntime as ort

        model_path = self.models_dir / "yolov5n.onnx"

        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLOv5n model not found at {model_path}. "
                "Run 'make setup-models' first."
            )

        # Configure session for CPU with controlled threading
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=["CPUExecutionProvider"],
        )

        logger.info(f"Loaded YOLOv5n model from {model_path}")

    def count_detections(self, image: np.ndarray) -> int:
        """Count detections in an image.

        Args:
            image: RGB uint8 array with shape [H, W, 3]

        Returns:
            Number of detections above confidence threshold after NMS
        """
        self._load_model()

        # Import preprocessing here to avoid circular imports
        from shared.processing import YOLOPreprocessor

        # Preprocess image
        preprocessor = YOLOPreprocessor()
        result = preprocessor(image)

        # Run inference
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: result.tensor})

        # Parse output based on shape
        detections = outputs[0]

        # Handle YOLOv8-style output: [batch, 84, num_predictions]
        if len(detections.shape) == 3 and detections.shape[1] == 84:
            return self._parse_yolov8_output(detections)

        # Handle YOLOv5 raw output: [batch, num_predictions, 85]
        elif len(detections.shape) == 3 and detections.shape[2] == 85:
            return self._parse_yolov5_raw_output(detections)

        # Handle post-NMS output: [batch, num_detections, 6]
        elif len(detections.shape) == 3 and detections.shape[2] == 6:
            return self._parse_post_nms_output(detections)

        else:
            logger.warning(f"Unknown output shape: {detections.shape}")
            return 0

    def _parse_yolov8_output(self, detections: np.ndarray) -> int:
        """Parse YOLOv8-style output format.

        Input shape: [batch, 84, num_predictions]
        - 84 = 4 (x, y, w, h) + 80 (class scores)
        - No separate objectness score
        """
        # Remove batch dimension and transpose to [num_predictions, 84]
        detections = detections[0].T  # [8400, 84]

        # Split into boxes and class scores
        boxes = detections[:, :4]  # [x_center, y_center, width, height]
        class_scores = detections[:, 4:]  # [80 class scores]

        # Get confidence (max class score) and class id
        confidences = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        # Apply NMS
        count = self._apply_nms(boxes, confidences, class_ids)
        return count

    def _parse_yolov5_raw_output(self, detections: np.ndarray) -> int:
        """Parse YOLOv5 raw output format.

        Input shape: [batch, num_predictions, 85]
        - 85 = 4 (x, y, w, h) + 1 (objectness) + 80 (class scores)
        """
        detections = detections[0]  # Remove batch: [num_predictions, 85]

        boxes = detections[:, :4]
        obj_conf = detections[:, 4]
        class_scores = detections[:, 5:]

        # Combined confidence = objectness * class_score
        max_class_scores = class_scores.max(axis=1)
        confidences = obj_conf * max_class_scores
        class_ids = class_scores.argmax(axis=1)

        count = self._apply_nms(boxes, confidences, class_ids)
        return count

    def _parse_post_nms_output(self, detections: np.ndarray) -> int:
        """Parse post-NMS output format.

        Input shape: [batch, num_detections, 6]
        - 6 = x1, y1, x2, y2, confidence, class_id
        """
        detections = detections[0]  # Remove batch

        if len(detections) == 0:
            return 0

        confidences = detections[:, 4]
        return int(np.sum(confidences >= self.confidence_threshold))

    def _apply_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> int:
        """Apply Non-Maximum Suppression and return detection count.

        Args:
            boxes: [N, 4] array of [x_center, y_center, width, height]
            scores: [N] array of confidence scores
            class_ids: [N] array of class IDs

        Returns:
            Number of detections after NMS
        """
        # Filter by confidence first
        mask = scores >= self.confidence_threshold
        if not mask.any():
            return 0

        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Convert from center format to corner format
        # [x_center, y_center, w, h] -> [x1, y1, x2, y2]
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        # Class-aware NMS (NMS per class)
        unique_classes = np.unique(class_ids)
        keep_indices = []

        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_x1 = x1[cls_mask]
            cls_y1 = y1[cls_mask]
            cls_x2 = x2[cls_mask]
            cls_y2 = y2[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices = np.where(cls_mask)[0]

            # Sort by score
            order = cls_scores.argsort()[::-1]

            while len(order) > 0:
                i = order[0]
                keep_indices.append(cls_indices[i])

                if len(order) == 1:
                    break

                # Compute IoU with remaining boxes
                xx1 = np.maximum(cls_x1[i], cls_x1[order[1:]])
                yy1 = np.maximum(cls_y1[i], cls_y1[order[1:]])
                xx2 = np.minimum(cls_x2[i], cls_x2[order[1:]])
                yy2 = np.minimum(cls_y2[i], cls_y2[order[1:]])

                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                intersection = w * h

                area_i = (cls_x2[i] - cls_x1[i]) * (cls_y2[i] - cls_y1[i])
                area_others = (cls_x2[order[1:]] - cls_x1[order[1:]]) * (cls_y2[order[1:]] - cls_y1[order[1:]])
                union = area_i + area_others - intersection

                iou = intersection / (union + 1e-6)

                # Keep boxes with IoU below threshold
                keep = np.where(iou <= self.iou_threshold)[0]
                order = order[keep + 1]

        return len(keep_indices)


# =============================================================================
# Dataset Curator
# =============================================================================

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

            return True, f"Found ({image_count} images, μ={mean_det:.2f})"

        except Exception as e:
            return False, f"Invalid manifest: {e}"

    def curate(
        self,
        force: bool = False,
        progress_callback: callable | None = None,
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
            i: [] for i in range(
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
        """Sample images to achieve balanced distribution.

        Aims for approximately equal representation of each detection count
        to achieve target mean of 4.0.
        """
        np.random.seed(self.config.random_seed)

        selected = []
        remaining = self.config.target_count

        # Calculate target per bucket for mean=4
        # For range [3, 4, 5], we want more 4s to center the mean
        detection_range = list(range(
            self.config.min_detections,
            self.config.max_detections + 1,
        ))
        len(detection_range)

        # Weight middle values more heavily for tighter std
        weights = []
        mid = (self.config.min_detections + self.config.max_detections) / 2
        for d in detection_range:
            # Higher weight for values closer to mean
            weight = 1.0 / (1.0 + abs(d - mid))
            weights.append(weight)

        total_weight = sum(weights)
        targets = {
            d: int(remaining * w / total_weight)
            for d, w in zip(detection_range, weights, strict=False)
        }

        # Adjust to hit exact target count
        allocated = sum(targets.values())
        if allocated < remaining:
            targets[4] += remaining - allocated  # Add extra to middle

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
                remaining_available = [
                    r for r in available if r.filename not in already_selected
                ]
                if remaining_available:
                    selected.append(remaining_available[0])
                    if len(selected) >= self.config.target_count:
                        break

            # Break if no more candidates
            all_available = sum(len(v) for v in candidates.values())
            if len(selected) >= all_available:
                break

        return selected[:self.config.target_count]

    def _generate_manifest(self, result: CurationResult) -> DatasetManifest:
        """Generate manifest from curation result."""
        counts = [img.detection_count for img in result.images]

        if counts:
            mean_det = sum(counts) / len(counts)
            variance = sum((x - mean_det) ** 2 for x in counts) / len(counts)
            std_det = variance ** 0.5
            min_det = min(counts)
            max_det = max(counts)
        else:
            mean_det = std_det = min_det = max_det = 0

        # Count distribution
        distribution = {}
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
            result.images.append(ImageRecord(
                filename=img_data["filename"],
                detection_count=img_data["detections"],
            ))

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
        logger.info(f"  Mean detections:   {stats['mean_detections']:.2f} (target: {TARGET_MEAN_DETECTIONS})")
        logger.info(f"  Std detections:    {stats['std_detections']:.2f} (target: ~{TARGET_STD_DETECTIONS})")
        logger.info(f"  Distribution:      {manifest.distribution}")
        logger.info(f"  Output:            {self.output_dir}")
