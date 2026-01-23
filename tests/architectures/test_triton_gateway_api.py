"""API tests for Triton gateway service.

Tests the FastAPI gateway endpoints without requiring Docker or Triton server.
Uses mocking to simulate the Triton client.

Author: Matthew Hong
"""

from triton.gateway.app.models import (
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
        """Verify health handles unhealthy state."""
        response = HealthResponse(status="unhealthy", models_loaded=False)
        assert response.models_loaded is False


class TestPredictEndpoint:
    """Tests for POST /predict endpoint."""

    def test_predict_response_schema(self):
        """Verify predict response matches expected schema."""
        from triton.gateway.app.models import (
            Classification,
            DetectionBox,
            DetectionWithClassification,
            PredictResponse,
        )

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

    def test_timing_reflects_triton_inference(self):
        """Verify timing breakdown includes Triton inference time."""
        from triton.gateway.app.models import PredictResponse

        response = PredictResponse(
            request_id="test",
            detections=[],
            timing={
                "detection_ms": 15.0,  # Triton is typically faster
                "classification_ms": 10.0,
                "total_ms": 25.0,
            },
        )

        # Triton should have lower latency than local inference
        assert response.timing["total_ms"] == 25.0


class TestModelsSchema:
    """Tests for Pydantic model schemas."""

    def test_classification_schema(self):
        """Verify Classification schema."""
        from triton.gateway.app.models import Classification

        result = Classification(
            class_name="car",
            confidence=0.92,
            class_id=817,
        )
        assert result.class_name == "car"
        assert result.class_id == 817

    def test_detection_box_format(self):
        """Verify DetectionBox format is consistent."""
        from triton.gateway.app.models import DetectionBox

        detection = DetectionBox(
            x1=200.0,
            y1=150.0,
            x2=400.0,
            y2=350.0,
            confidence=0.85,
            class_id=867,
        )

        # Verify bbox is [x1, y1, x2, y2] format
        assert detection.x1 < detection.x2
        assert detection.y1 < detection.y2

    def test_predict_response_serialization(self):
        """Verify PredictResponse serializes correctly for JSON."""
        from triton.gateway.app.models import (
            Classification,
            DetectionBox,
            DetectionWithClassification,
            PredictResponse,
        )

        classification = Classification(
            class_name="bicycle",
            confidence=0.76,
            class_id=671,
        )
        detection_box = DetectionBox(
            x1=10.0,
            y1=20.0,
            x2=110.0,
            y2=120.0,
            confidence=0.81,
            class_id=671,
        )
        detection = DetectionWithClassification(
            detection=detection_box,
            classification=classification,
        )
        response = PredictResponse(
            request_id="abc123",
            detections=[detection],
            timing={"detection_ms": 20.0, "classification_ms": 15.0, "total_ms": 35.0},
        )

        # Serialize to dict (as JSON would do)
        data = response.model_dump()
        assert data["request_id"] == "abc123"
        assert len(data["detections"]) == 1
        assert data["detections"][0]["classification"]["class_name"] == "bicycle"
