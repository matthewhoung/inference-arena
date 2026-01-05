"""Fixtures for architecture API tests.

These tests validate the FastAPI endpoints without requiring Docker or models.
They use mocking to simulate the inference pipelines.

Author: Matthew Hong
"""

import pytest


@pytest.fixture
def mock_models_dir(tmp_path):
    """Create a temporary models directory with dummy model files."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Create dummy model files (just empty files for existence checks)
    (models_dir / "yolov5n.onnx").touch()
    (models_dir / "mobilenetv2.onnx").touch()
    (models_dir / "mobilenetv2.onnx.data").touch()

    return models_dir


@pytest.fixture
def mock_detection_result():
    """Sample detection result for mocking."""
    return {
        "bbox": [100.0, 100.0, 200.0, 200.0],
        "confidence": 0.95,
        "class_name": "person",
        "classification": {
            "class_name": "person_standing",
            "confidence": 0.88,
            "class_id": 42,
        },
    }


@pytest.fixture
def mock_timing():
    """Sample timing result for mocking."""
    return {
        "detection_ms": 45.2,
        "classification_ms": 32.1,
        "total_ms": 77.3,
    }


@pytest.fixture
def sample_image_bytes():
    """Create a minimal valid JPEG image for testing."""
    # Minimal valid JPEG (1x1 red pixel)
    import io

    from PIL import Image

    img = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()
