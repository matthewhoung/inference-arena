"""Pydantic models for API request/response schemas.

This module defines the data models for the detection service API.
Schemas are compatible with the proto definitions for consistency across
architectures.

Author: Matthew Hong
"""

from typing import Any

from pydantic import BaseModel, Field


class DetectionBox(BaseModel):
    """Bounding box for object detection.

    Attributes:
        x1: Left coordinate
        y1: Top coordinate
        x2: Right coordinate
        y2: Bottom coordinate
        confidence: Detection confidence score [0, 1]
        class_id: COCO class ID
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class Classification(BaseModel):
    """Classification result for detected object.

    Attributes:
        class_id: ImageNet class ID [0, 999]
        class_name: Human-readable class name
        confidence: Classification confidence score [0, 1]
    """

    class_id: int
    class_name: str
    confidence: float


class DetectionWithClassification(BaseModel):
    """Combined detection and classification result.

    Attributes:
        detection: Bounding box and detection info
        classification: Classification result for the detected crop
    """

    detection: DetectionBox
    classification: Classification


class PredictResponse(BaseModel):
    """Response model for /predict endpoint.

    Attributes:
        request_id: Unique request identifier for tracing
        detections: List of detected objects with classifications
        timing: Performance breakdown (detection_ms, classification_ms, total_ms)
    """

    request_id: str
    detections: list[DetectionWithClassification]
    timing: dict[str, float] = Field(
        description="Performance timing breakdown in milliseconds"
    )


class HealthResponse(BaseModel):
    """Response model for /health endpoint.

    Attributes:
        status: Service health status
        models_loaded: Whether models are loaded and ready
    """

    status: str = "healthy"
    models_loaded: bool
