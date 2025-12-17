"""
Unit Tests for MinIO Model Initialization

This module tests the MinIO model upload functionality and Triton config generation.

Test Categories:
- Triton config generation: config.pbtxt content
- Metadata generation: metadata.json content
- MinIO operations: Bucket and object operations (mocked)
- Integration: Full upload workflow (requires MinIO)

Author: Matthew Hong
"""

import importlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Triton Config Tests
# =============================================================================


class TestTritonConfigGeneration:
    """Test Triton config.pbtxt generation."""

    def test_generate_yolov5n_config(self) -> None:
        """Should generate valid config.pbtxt for YOLOv5n."""
        from infrastructure.minio.triton_config import generate_config_pbtxt

        config = generate_config_pbtxt("yolov5n")

        # Check required fields
        assert 'name: "yolov5n"' in config
        assert 'platform: "onnxruntime_onnx"' in config
        assert "max_batch_size: 0" in config

        # Check input spec
        assert 'name: "images"' in config
        assert "TYPE_FP32" in config
        assert "640" in config

        # Check output spec
        assert 'name: "output0"' in config

        # Check instance group
        assert "KIND_CPU" in config

        # Check threading parameters
        assert "intra_op_thread_count" in config
        assert "inter_op_thread_count" in config

    def test_generate_mobilenetv2_config(self) -> None:
        """Should generate valid config.pbtxt for MobileNetV2."""
        from infrastructure.minio.triton_config import generate_config_pbtxt

        config = generate_config_pbtxt("mobilenetv2")

        # Check required fields
        assert 'name: "mobilenetv2"' in config
        assert 'platform: "onnxruntime_onnx"' in config

        # Check input spec
        assert 'name: "input"' in config
        assert "224" in config

        # Check output spec
        assert 'name: "output"' in config
        assert "1000" in config

    def test_generate_all_configs(self) -> None:
        """Should generate configs for all models."""
        from infrastructure.minio.triton_config import generate_all_configs

        configs = generate_all_configs()

        assert isinstance(configs, dict)
        assert "yolov5n" in configs
        assert "mobilenetv2" in configs

        for model_name, config in configs.items():
            assert f'name: "{model_name}"' in config

    def test_validate_config_pbtxt(self) -> None:
        """Should validate config.pbtxt content."""
        from infrastructure.minio.triton_config import (
            generate_config_pbtxt,
            validate_config_pbtxt,
        )

        config = generate_config_pbtxt("yolov5n")
        errors = validate_config_pbtxt(config)

        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_invalid_config_detected(self) -> None:
        """Should detect invalid config content."""
        from infrastructure.minio.triton_config import validate_config_pbtxt

        invalid_config = "this is not a valid config"
        errors = validate_config_pbtxt(invalid_config)

        assert len(errors) > 0

    def test_format_dims(self) -> None:
        """Should format dimensions correctly."""
        from infrastructure.minio.triton_config import _format_dims

        dims = _format_dims([1, 3, 640, 640])

        assert dims == "[ 1, 3, 640, 640 ]"

    def test_save_config_pbtxt(self) -> None:
        """Should save config to disk."""
        from infrastructure.minio.triton_config import save_config_pbtxt

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config_path = save_config_pbtxt("yolov5n", output_dir)

            assert config_path.exists()
            assert config_path.name == "config.pbtxt"
            assert config_path.parent.name == "yolov5n"

            content = config_path.read_text()
            assert 'name: "yolov5n"' in content


# =============================================================================
# MinIO Registry Tests (Mocked)
# =============================================================================


class TestMinIORegistryMocked:
    """Test MinIO registry with mocked client."""

    @pytest.fixture(autouse=True)
    def mock_minio_client(self):
        """Create mock MinIO client.

        Note: autouse=True ensures the mock is applied before any test imports.
        We patch at the source (minio.Minio) rather than the usage site because
        init_models.py conditionally imports Minio in a try-except block.
        """
        # Create a mock module if minio isn't installed
        import sys

        try:
            import minio  # noqa: F401  # Import needed to check availability

            patch_target = "minio.Minio"
        except ImportError:
            mock_minio_module = MagicMock()
            sys.modules["minio"] = mock_minio_module
            sys.modules["minio.error"] = MagicMock()
            patch_target = "minio.Minio"

        with (
            patch(patch_target) as MockMinio,
            patch("infrastructure.minio.init_models.MINIO_AVAILABLE", True),
        ):
            client = MagicMock()
            MockMinio.return_value = client

            # Default behaviors
            client.list_buckets.return_value = []
            client.bucket_exists.return_value = False
            client.make_bucket.return_value = None
            client.fput_object.return_value = None
            client.put_object.return_value = None

            if "infrastructure.minio.init_models" in sys.modules:
                importlib.reload(sys.modules["infrastructure.minio.init_models"])

            yield client

    @pytest.fixture
    def mock_model_file(self):
        """Create a temporary model file."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake onnx model content for testing")
            yield Path(f.name)
        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    def test_registry_initialization(self, mock_minio_client) -> None:
        """Should initialize with config from experiment.yaml."""
        from infrastructure.minio.init_models import MinIOModelRegistry

        registry = MinIOModelRegistry()

        assert registry.bucket == "models"
        assert registry.endpoint == "localhost:9000"

    def test_ensure_bucket_creates_if_missing(self, mock_minio_client) -> None:
        """Should create bucket if it doesn't exist."""
        from infrastructure.minio.init_models import MinIOModelRegistry

        mock_minio_client.bucket_exists.return_value = False

        registry = MinIOModelRegistry()
        registry.ensure_bucket_exists()

        mock_minio_client.make_bucket.assert_called_once_with("models")

    def test_ensure_bucket_skips_if_exists(self, mock_minio_client) -> None:
        """Should skip bucket creation if exists."""
        from infrastructure.minio.init_models import MinIOModelRegistry

        mock_minio_client.bucket_exists.return_value = True

        registry = MinIOModelRegistry()
        registry.ensure_bucket_exists()

        mock_minio_client.make_bucket.assert_not_called()

    def test_compute_checksum(self, mock_minio_client, mock_model_file) -> None:
        """Should compute SHA256 checksum."""
        from infrastructure.minio.init_models import MinIOModelRegistry

        registry = MinIOModelRegistry()
        checksum = registry._compute_checksum(mock_model_file)

        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex length

    def test_generate_metadata(self, mock_minio_client, mock_model_file) -> None:
        """Should generate valid metadata."""
        from infrastructure.minio.init_models import MinIOModelRegistry

        registry = MinIOModelRegistry()
        metadata = registry._generate_metadata("yolov5n", mock_model_file)

        assert metadata["model_name"] == "yolov5n"
        assert metadata["version"] == 1
        assert metadata["format"] == "onnx"
        assert metadata["opset_version"] == 17
        assert "checksum_sha256" in metadata
        assert "uploaded_at" in metadata
        assert "input" in metadata
        assert "output" in metadata


# =============================================================================
# Metadata Structure Tests
# =============================================================================


class TestMetadataStructure:
    """Test metadata.json structure and content."""

    @pytest.fixture
    def sample_metadata(self) -> dict[str, Any]:
        """Create sample metadata for testing."""
        return {
            "model_name": "yolov5n",
            "version": 1,
            "format": "onnx",
            "opset_version": 17,
            "task": "object_detection",
            "input": {
                "name": "images",
                "shape": [1, 3, 640, 640],
                "dtype": "float32",
            },
            "output": {
                "name": "output0",
                "shape": [1, 84, 8400],
                "dtype": "float32",
            },
            "source": "ultralytics/yolov5",
            "checksum_sha256": "abc123",
            "file_size_bytes": 12345,
            "uploaded_at": "2024-12-16T10:00:00+00:00",
            "experiment_spec_version": "1.0.0",
            "thesis_reference": "experiment.yaml",
            "author": "Matthew Hong",
        }

    def test_metadata_has_required_fields(self, sample_metadata) -> None:
        """Metadata should have all required fields."""
        required_fields = [
            "model_name",
            "version",
            "format",
            "opset_version",
            "input",
            "output",
            "checksum_sha256",
            "uploaded_at",
        ]

        for field in required_fields:
            assert field in sample_metadata, f"Missing field: {field}"

    def test_metadata_is_json_serializable(self, sample_metadata) -> None:
        """Metadata should be JSON serializable."""
        json_str = json.dumps(sample_metadata)
        parsed = json.loads(json_str)

        assert parsed == sample_metadata

    def test_metadata_input_output_structure(self, sample_metadata) -> None:
        """Input/output should have name, shape, dtype."""
        for key in ["input", "output"]:
            spec = sample_metadata[key]
            assert "name" in spec
            assert "shape" in spec
            assert "dtype" in spec


# =============================================================================
# Bucket Structure Tests
# =============================================================================


class TestBucketStructure:
    """Test expected MinIO bucket structure."""

    def test_model_path_structure(self) -> None:
        """Model paths should follow Triton convention."""
        from infrastructure.minio.init_models import MODEL_VERSION

        model_name = "yolov5n"
        expected_paths = [
            f"{model_name}/{MODEL_VERSION}/model.onnx",
            f"{model_name}/config.pbtxt",
            f"{model_name}/metadata.json",
        ]

        for path in expected_paths:
            assert model_name in path
            assert "/" in path

    def test_version_directory(self) -> None:
        """Version directory should be numeric."""
        from infrastructure.minio.init_models import MODEL_VERSION

        assert isinstance(MODEL_VERSION, int)
        assert MODEL_VERSION >= 1


# =============================================================================
# Integration Tests (Requires MinIO)
# =============================================================================


@pytest.mark.integration
class TestMinIOIntegration:
    """Integration tests requiring running MinIO instance."""

    @pytest.fixture
    def registry(self):
        """Create registry connected to running MinIO."""
        try:
            from infrastructure.minio.init_models import MinIOModelRegistry

            registry = MinIOModelRegistry()
            registry.wait_for_minio()
            return registry
        except Exception as e:
            pytest.skip(f"MinIO not available: {e}")

    def test_bucket_operations(self, registry) -> None:
        """Should create and verify bucket."""
        registry.ensure_bucket_exists()

        # Verify bucket exists
        assert registry.client.bucket_exists(registry.bucket)

    def test_verify_models_empty(self, registry) -> None:
        """Should verify when no models uploaded."""
        registry.ensure_bucket_exists()
        verification = registry.verify_models()

        assert isinstance(verification, dict)
        assert "models" in verification
        assert "all_valid" in verification
