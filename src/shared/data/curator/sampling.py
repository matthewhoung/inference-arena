"""Detection Sampling Module.

This module contains the DetectionCounter class for counting
detections in images using YOLOv5n ONNX model.

Handles YOLOv8-style output format [batch, 84, num_predictions]:
- 84 = 4 (bbox) + 80 (class scores)
- No separate objectness score
- Requires transpose and NMS

Author: Matthew Hong
Specification Reference: experiment.yaml controlled_variables.dataset
"""

import logging
from pathlib import Path

import numpy as np

from .types import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_IOU_THRESHOLD

logger = logging.getLogger(__name__)


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
                f"YOLOv5n model not found at {model_path}. " "Run 'make setup-models' first."
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
                area_others = (cls_x2[order[1:]] - cls_x1[order[1:]]) * (
                    cls_y2[order[1:]] - cls_y1[order[1:]]
                )
                union = area_i + area_others - intersection

                iou = intersection / (union + 1e-6)

                # Keep boxes with IoU below threshold
                keep = np.where(iou <= self.iou_threshold)[0]
                order = order[keep + 1]

        return len(keep_indices)
