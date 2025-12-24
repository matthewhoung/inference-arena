"""YOLO output postprocessing with NMS.

This module provides Non-Maximum Suppression (NMS) and output parsing for
YOLOv5n/YOLOv8 detection models. Adapted from the proven curator.py implementation.

Author: Matthew Hong
"""

import numpy as np


def parse_yolo_output(
    raw_output: np.ndarray,
    confidence_threshold: float,
    iou_threshold: float,
) -> np.ndarray:
    """Parse YOLO output and apply NMS.

    Handles YOLOv8-style output format: [batch, 84, num_predictions]
    - 84 = 4 (x, y, w, h) + 80 (class scores)
    - No separate objectness score

    Args:
        raw_output: ONNX output with shape [1, 84, 8400]
        confidence_threshold: Minimum confidence for valid detection
        iou_threshold: IoU threshold for NMS

    Returns:
        Detections array [N, 6] with [x1, y1, x2, y2, confidence, class_id]
        in corner format, ready for use. Returns empty array if no detections.
    """
    # Remove batch dimension and transpose to [num_predictions, 84]
    detections = raw_output[0].T  # [8400, 84]

    # Split into boxes and class scores
    boxes = detections[:, :4]  # [x_center, y_center, width, height]
    class_scores = detections[:, 4:]  # [80 class scores]

    # Get confidence (max class score) and class id
    confidences = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    # Apply class-aware NMS
    keep_indices = apply_nms(
        boxes,
        confidences,
        class_ids,
        confidence_threshold,
        iou_threshold,
    )

    if len(keep_indices) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # Convert to corner format and construct output
    boxes_keep = boxes[keep_indices]
    x1 = boxes_keep[:, 0] - boxes_keep[:, 2] / 2
    y1 = boxes_keep[:, 1] - boxes_keep[:, 3] / 2
    x2 = boxes_keep[:, 0] + boxes_keep[:, 2] / 2
    y2 = boxes_keep[:, 1] + boxes_keep[:, 3] / 2

    result = np.column_stack(
        [
            x1,
            y1,
            x2,
            y2,
            confidences[keep_indices],
            class_ids[keep_indices],
        ]
    )

    return result.astype(np.float32)


def apply_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
) -> list[int]:
    """Apply class-aware Non-Maximum Suppression.

    Performs NMS separately for each class to avoid suppressing detections
    of different object types that overlap spatially.

    Args:
        boxes: [N, 4] array of [x_center, y_center, width, height]
        scores: [N] array of confidence scores
        class_ids: [N] array of class IDs
        conf_threshold: Minimum confidence threshold
        iou_threshold: IoU threshold for suppression

    Returns:
        List of indices to keep after NMS
    """
    # Filter by confidence first
    mask = scores >= conf_threshold
    if not mask.any():
        return []

    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    original_indices = np.where(mask)[0]

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

        # Sort by score descending
        order = cls_scores.argsort()[::-1]

        while len(order) > 0:
            i = order[0]
            keep_indices.append(original_indices[cls_indices[i]])

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
            keep = np.where(iou <= iou_threshold)[0]
            order = order[keep + 1]

    return keep_indices
