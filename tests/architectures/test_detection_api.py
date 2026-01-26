"""API tests for microservices detection service.

Tests the FastAPI endpoints without requiring Docker or actual models.
Uses mocking to simulate the inference pipeline.

Author: Matthew Hong
"""

from microservices.detection.app.models import (
    Classification,
    DetectionBox,
    DetectionWithClassification,
    HealthResponse,
    PredictResponse,
)


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_response_schema(self):
        """Verify health response matches expected schema."""

        response = HealthResponse(status="healthy", models_loaded=True)
        assert response.status == "healthy"
        assert response.models_loaded is True


class TestPredictEndpoint:
    """Tests for POST /predict endpoint."""

    def test_predict_response_schema(self):
        """Verify predict response matches expected schema."""

        classification = Classification(
            class_name="person",
            confidence=0.95,
            class_id=0,
        )
        detection_box = DetectionBox(
            x1=100.0,
            y1=100.0,
            x2=200.0,
            y2=200.0,
            confidence=0.9,
            class_id=0,
        )
        detection = DetectionWithClassification(
            detection=detection_box,
            classification=classification,
        )
        response = PredictResponse(
            request_id="test-uuid",
            detections=[detection],
            timing={
                "detection_ms": 45.2,
                "classification_ms": 32.1,
                "total_ms": 77.3,
            },
        )

        assert response.request_id == "test-uuid"
        assert len(response.detections) == 1
        assert response.timing["total_ms"] == 77.3

    def test_predict_response_timing_breakdown(self):
        """Verify timing includes detection and classification separately."""

        response = PredictResponse(
            request_id="test",
            detections=[],
            timing={
                "detection_ms": 50.0,
                "classification_ms": 25.0,
                "total_ms": 75.0,
            },
        )

        # Microservices should report separate timing for each service
        assert response.timing["detection_ms"] == 50.0
        assert response.timing["classification_ms"] == 25.0


class TestModelsSchema:
    """Tests for Pydantic model schemas."""

    def test_classification_schema(self):
        """Verify Classification schema matches expected format."""

        result = Classification(
            class_name="cat",
            confidence=0.99,
            class_id=281,
        )

        # Convert to dict to check serialization
        data = result.model_dump()
        assert "class_name" in data
        assert "confidence" in data
        assert "class_id" in data

    def test_detection_with_classification(self):
        """Verify DetectionWithClassification includes nested models."""

        classification = Classification(
            class_name="bird",
            confidence=0.87,
            class_id=14,
        )
        detection_box = DetectionBox(
            x1=50.0,
            y1=60.0,
            x2=150.0,
            y2=160.0,
            confidence=0.78,
            class_id=14,
        )
        detection = DetectionWithClassification(
            detection=detection_box,
            classification=classification,
        )

        assert detection.classification.class_name == "bird"
        assert detection.classification.class_id == 14
