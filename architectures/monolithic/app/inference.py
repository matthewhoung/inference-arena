"""Core inference pipeline for monolithic architecture.

This module orchestrates the complete detection + classification pipeline:
1. YOLOv5n object detection
2. Crop extraction for each detection
3. MobileNetV2 classification for each crop
4. Result aggregation with timing breakdown

All processing happens in-process (true monolithic architecture).

Author: Matthew Hong
"""

import logging
import time
from pathlib import Path

import numpy as np

from shared.config import get_controlled_variables, get_model_config
from shared.model.registry import ModelRegistry, SessionConfig
from shared.processing import MobileNetPreprocessor, YOLOPreprocessor
from shared.processing.mobilenet_preprocess import extract_crop
from shared.processing.transforms import load_image_from_bytes

from .postprocess import parse_yolo_output

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Inference pipeline for monolithic architecture.

    Runs both detection and classification in-process using ONNX Runtime.
    Threading configuration loaded from experiment.yaml for consistency
    across all architectures.

    Attributes:
        registry: ModelRegistry for ONNX session management
        yolo_session: YOLOv5n inference session
        mobilenet_session: MobileNetV2 inference session
        labels: ImageNet class labels [0-999]
        yolo_preprocessor: YOLO preprocessing pipeline
        mobilenet_preprocessor: MobileNet preprocessing pipeline
        conf_threshold: Detection confidence threshold
        iou_threshold: NMS IoU threshold
    """

    def __init__(self, models_dir: Path, labels_file: Path) -> None:
        """Initialize the inference pipeline.

        Args:
            models_dir: Directory containing ONNX model files
            labels_file: Path to ImageNet labels file (1000 lines)
        """
        # Load threading config from experiment.yaml
        onnx_config = get_controlled_variables("onnx_runtime")
        session_config = SessionConfig(
            intra_op_threads=onnx_config["intra_op_num_threads"],
            inter_op_threads=onnx_config["inter_op_num_threads"],
        )

        logger.info(
            f"Initializing pipeline with {session_config.intra_op_threads} "
            f"intra-op, {session_config.inter_op_threads} inter-op threads"
        )

        # Initialize model registry
        self.registry = ModelRegistry(models_dir, session_config)

        # Preload models
        logger.info("Loading YOLOv5n model")
        self.yolo_session = self.registry.get_session("yolov5n")

        logger.info("Loading MobileNetV2 model")
        self.mobilenet_session = self.registry.get_session("mobilenetv2")

        # Load ImageNet labels
        logger.info(f"Loading ImageNet labels from {labels_file}")
        self.labels = self._load_labels(labels_file)

        # Initialize preprocessors
        self.yolo_preprocessor = YOLOPreprocessor()
        self.mobilenet_preprocessor = MobileNetPreprocessor()

        # Load detection thresholds from experiment.yaml
        yolo_config = get_model_config("yolov5n")
        self.conf_threshold = yolo_config["confidence_threshold"]
        self.iou_threshold = yolo_config["iou_threshold"]

        logger.info(
            f"Detection thresholds: confidence={self.conf_threshold}, "
            f"iou={self.iou_threshold}"
        )

    def _load_labels(self, labels_file: Path) -> list[str]:
        """Load ImageNet class labels from file.

        Args:
            labels_file: Path to labels file (one per line)

        Returns:
            List of 1000 class names indexed 0-999

        Raises:
            FileNotFoundError: If labels file not found
            ValueError: If labels file doesn't contain exactly 1000 lines
        """
        if not labels_file.exists():
            raise FileNotFoundError(
                f"ImageNet labels file not found: {labels_file}. "
                "Ensure src/shared/data/imagenet_labels.txt exists."
            )

        with open(labels_file) as f:
            labels = [line.strip() for line in f]

        if len(labels) != 1000:
            raise ValueError(
                f"Expected 1000 ImageNet labels, got {len(labels)}. "
                f"Check {labels_file} format."
            )

        logger.info(f"Loaded {len(labels)} ImageNet class labels")
        return labels

    def predict(self, image_bytes: bytes) -> tuple[list[dict], dict[str, float]]:
        """Run full detection + classification pipeline.

        Pipeline:
        1. Decode image from bytes
        2. YOLOv5n detection (preprocess → inference → NMS)
        3. For each detection:
           - Extract crop
           - MobileNetV2 classification
        4. Aggregate results with timing

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.)

        Returns:
            Tuple of (results, timing):
            - results: List of dicts with 'detection' and 'classification'
            - timing: Dict with 'detection_ms', 'classification_ms', 'total_ms'

        Raises:
            ValueError: If image cannot be decoded
        """
        timing: dict[str, float] = {}
        t0 = time.perf_counter()

        # Decode image
        image = load_image_from_bytes(image_bytes)

        # ====================================================================
        # YOLO Detection
        # ====================================================================
        t_det_start = time.perf_counter()

        # Preprocess
        yolo_result = self.yolo_preprocessor(image)

        # Run YOLO inference
        yolo_output = self.yolo_session.run(None, {"images": yolo_result.tensor})[0]

        # Parse detections (NMS)
        detections_letterbox = parse_yolo_output(
            yolo_output,
            self.conf_threshold,
            self.iou_threshold,
        )

        # Scale boxes to original coordinates
        if len(detections_letterbox) > 0:
            detections_orig = yolo_result.scale_boxes_to_original(detections_letterbox)
        else:
            detections_orig = detections_letterbox

        t_det_end = time.perf_counter()
        timing["detection_ms"] = (t_det_end - t_det_start) * 1000

        # ====================================================================
        # Classification
        # ====================================================================
        t_cls_start = time.perf_counter()
        results = []

        for det in detections_orig:
            # Extract crop from detection bounding box
            crop = extract_crop(image, det)

            # Preprocess for MobileNet
            mobilenet_result = self.mobilenet_preprocessor(crop)

            # Run classification inference
            cls_output = self.mobilenet_session.run(
                None, {"input": mobilenet_result.tensor}
            )[0]

            # Get top-1 class
            class_id = int(np.argmax(cls_output[0]))
            confidence = float(cls_output[0, class_id])
            class_name = self.labels[class_id]

            results.append(
                {
                    "detection": {
                        "x1": float(det[0]),
                        "y1": float(det[1]),
                        "x2": float(det[2]),
                        "y2": float(det[3]),
                        "confidence": float(det[4]),
                        "class_id": int(det[5]),
                    },
                    "classification": {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                    },
                }
            )

        t_cls_end = time.perf_counter()
        timing["classification_ms"] = (t_cls_end - t_cls_start) * 1000
        timing["total_ms"] = (t_cls_end - t0) * 1000

        return results, timing
