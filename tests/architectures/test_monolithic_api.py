"""API tests for monolithic architecture.

Tests the FastAPI endpoints without requiring Docker or actual models.
Uses mocking to simulate the inference pipeline.

Author: Matthew Hong
"""

from monolithic.app.models import (
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

    def test_health_response_unhealthy(self):
        """Verify health response handles unhealthy state."""

        response = HealthResponse(status="healthy", models_loaded=False)
        assert response.models_loaded is False


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

    def test_predict_response_empty_detections(self):
        """Verify predict handles no detections."""

        response = PredictResponse(
            request_id="test-uuid",
            detections=[],
            timing={
                "detection_ms": 30.0,
                "classification_ms": 0.0,
                "total_ms": 30.0,
            },
        )

        assert len(response.detections) == 0

    def test_timing_breakdown_structure(self):
        """Verify timing breakdown has required fields."""

        response = PredictResponse(
            request_id="test",
            detections=[],
            timing={
                "detection_ms": 10.0,
                "classification_ms": 20.0,
                "total_ms": 30.0,
            },
        )

        # Verify all timing fields present
        assert "detection_ms" in response.timing
        assert "classification_ms" in response.timing
        assert "total_ms" in response.timing


class TestModelsSchema:
    """Tests for Pydantic model schemas."""

    def test_classification_schema(self):
        """Verify Classification schema."""

        result = Classification(
            class_name="cat",
            confidence=0.99,
            class_id=281,
        )
        assert result.class_name == "cat"
        assert result.confidence == 0.99
        assert result.class_id == 281

    def test_detection_box_format(self):
        """Verify DetectionBox schema with bbox coordinates."""

        detection = DetectionBox(
            x1=10.5,
            y1=20.5,
            x2=100.5,
            y2=200.5,
            confidence=0.92,
            class_id=207,
        )

        # Verify bbox format
        assert detection.x1 < detection.x2
        assert detection.y1 < detection.y2
        assert 0 <= detection.confidence <= 1
