"""
Unit Tests for Model Module

This module tests:
- exporter.py: ONNX export functions and verification
- registry.py: Model loading and caching

Test Categories:
- Export tests: Verify ONNX file generation (requires torch, marked slow)
- Verification tests: Check ONNX model validity
- Registry tests: Session loading, caching, configuration
- Checksum tests: SHA256 computation

Author: Matthew Hong
Specification Reference: Foundation Specification §2 Model Export
"""

import hashlib
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pytest

from shared.model.exporter import (
    MOBILENET_INPUT_SIZE,
    ONNX_OPSET_VERSION,
    YOLO_INPUT_SIZE,
    ExportResult,
    compute_checksum,
    export_all_models,
    export_mobilenetv2,
    export_yolov5n,
    verify_onnx_model,
)
from shared.model.registry import (
    DEFAULT_INTER_OP_THREADS,
    DEFAULT_INTRA_OP_THREADS,
    ModelInfo,
    ModelRegistry,
    SessionConfig,
    get_default_registry,
    reset_default_registry,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Path:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_file(temp_dir: Path) -> Path:
    """Create a sample file for checksum testing."""
    file_path = temp_dir / "sample.bin"
    file_path.write_bytes(b"Hello, World!")
    return file_path


@pytest.fixture
def mock_onnx_model(temp_dir: Path) -> Path:
    """
    Create a minimal valid ONNX model for testing.

    This creates a simple identity model without requiring torch.
    Uses IR version 9 for onnxruntime compatibility (1.19.x supports up to IR 10).
    """
    try:
        import onnx
        from onnx import TensorProto, helper

        # Create a simple identity model
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

        identity_node = helper.make_node("Identity", ["input"], ["output"])

        graph = helper.make_graph([identity_node], "test_model", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        model.ir_version = 9

        model_path = temp_dir / "test_model.onnx"
        onnx.save(model, str(model_path))

        return model_path

    except ImportError:
        pytest.skip("onnx package not installed")


@pytest.fixture
def registry_with_mock_model(temp_dir: Path, mock_onnx_model: Path) -> ModelRegistry:
    """Create registry with mock model available."""
    # Copy mock model as yolov5n.onnx
    import shutil

    yolo_path = temp_dir / "yolov5n.onnx"
    shutil.copy(mock_onnx_model, yolo_path)

    return ModelRegistry(models_dir=temp_dir)


# =============================================================================
# Tests for Checksum
# =============================================================================


class TestComputeChecksum:
    """Tests for compute_checksum function."""

    def test_checksum_returns_string(self, sample_file: Path) -> None:
        """Checksum should return hex string."""
        checksum = compute_checksum(sample_file)

        assert isinstance(checksum, str)

    def test_checksum_length(self, sample_file: Path) -> None:
        """SHA256 checksum should be 64 characters."""
        checksum = compute_checksum(sample_file)

        assert len(checksum) == 64

    def test_checksum_is_hex(self, sample_file: Path) -> None:
        """Checksum should be valid hex string."""
        checksum = compute_checksum(sample_file)

        # Should not raise ValueError
        int(checksum, 16)

    def test_checksum_deterministic(self, sample_file: Path) -> None:
        """Same file should produce same checksum."""
        checksum1 = compute_checksum(sample_file)
        checksum2 = compute_checksum(sample_file)

        assert checksum1 == checksum2

    def test_checksum_matches_hashlib(self, sample_file: Path) -> None:
        """Checksum should match hashlib computation."""
        expected = hashlib.sha256(sample_file.read_bytes()).hexdigest()
        actual = compute_checksum(sample_file)

        assert actual == expected

    def test_checksum_different_content(self, temp_dir: Path) -> None:
        """Different content should produce different checksum."""
        file1 = temp_dir / "file1.bin"
        file2 = temp_dir / "file2.bin"

        file1.write_bytes(b"content1")
        file2.write_bytes(b"content2")

        assert compute_checksum(file1) != compute_checksum(file2)


# =============================================================================
# Tests for ONNX Verification
# =============================================================================


class TestVerifyOnnxModel:
    """Tests for verify_onnx_model function."""

    def test_verify_nonexistent_file(self, temp_dir: Path) -> None:
        """Should return error for nonexistent file."""
        result = verify_onnx_model(temp_dir / "nonexistent.onnx")

        assert result["valid"] is False
        assert "not found" in result["error"].lower()

    def test_verify_valid_model(self, mock_onnx_model: Path) -> None:
        """Should validate correct ONNX model."""
        result = verify_onnx_model(mock_onnx_model)

        assert result["valid"] is True
        assert result["error"] is None

    def test_verify_returns_opset(self, mock_onnx_model: Path) -> None:
        """Should return opset version."""
        result = verify_onnx_model(mock_onnx_model)

        assert result["opset_version"] == 17

    def test_verify_returns_input_shapes(self, mock_onnx_model: Path) -> None:
        """Should return input shapes."""
        result = verify_onnx_model(mock_onnx_model)

        assert len(result["input_shapes"]) > 0
        assert result["input_shapes"][0] == (1, 3, 224, 224)

    def test_verify_returns_output_shapes(self, mock_onnx_model: Path) -> None:
        """Should return output shapes."""
        result = verify_onnx_model(mock_onnx_model)

        assert len(result["output_shapes"]) > 0

    def test_verify_invalid_file(self, temp_dir: Path) -> None:
        """Should return error for invalid ONNX file."""
        invalid_path = temp_dir / "invalid.onnx"
        invalid_path.write_bytes(b"not a valid onnx file")

        result = verify_onnx_model(invalid_path)

        assert result["valid"] is False
        assert result["error"] is not None


# =============================================================================
# Tests for Export Constants
# =============================================================================


class TestExportConstants:
    """Tests for export module constants."""

    def test_opset_version(self) -> None:
        """ONNX opset should be 17 for Triton compatibility."""
        assert ONNX_OPSET_VERSION == 17

    def test_yolo_input_size(self) -> None:
        """YOLO input should be 640x640."""
        assert YOLO_INPUT_SIZE == 640

    def test_mobilenet_input_size(self) -> None:
        """MobileNet input should be 224x224."""
        assert MOBILENET_INPUT_SIZE == 224


# =============================================================================
# Tests for Export Functions (Slow - require torch)
# =============================================================================


class TestExportYOLOv5n:
    """Tests for YOLOv5n export function."""

    @pytest.mark.slow
    def test_export_creates_file(self, temp_dir: Path) -> None:
        """Export should create ONNX file."""
        pytest.importorskip("torch")
        pytest.importorskip("ultralytics")

        output_path = temp_dir / "yolov5n.onnx"
        export_yolov5n(output_path)

        assert output_path.exists()

    @pytest.mark.slow
    def test_export_returns_result(self, temp_dir: Path) -> None:
        """Export should return ExportResult."""
        pytest.importorskip("torch")
        pytest.importorskip("ultralytics")

        output_path = temp_dir / "yolov5n.onnx"
        result = export_yolov5n(output_path)

        assert isinstance(result, ExportResult)
        assert result.model_path == output_path
        assert len(result.checksum) == 64

    @pytest.mark.slow
    def test_export_correct_opset(self, temp_dir: Path) -> None:
        """Exported model should have correct opset version."""
        pytest.importorskip("torch")
        pytest.importorskip("ultralytics")

        output_path = temp_dir / "yolov5n.onnx"
        result = export_yolov5n(output_path)

        assert result.opset_version == ONNX_OPSET_VERSION

    def test_export_raises_if_exists(self, mock_onnx_model: Path) -> None:
        """Should raise FileExistsError if file exists and force=False."""
        with pytest.raises(FileExistsError):
            export_yolov5n(mock_onnx_model, force=False)

    @pytest.mark.slow
    def test_export_force_overwrites(self, temp_dir: Path) -> None:
        """Should overwrite existing file if force=True."""
        pytest.importorskip("torch")
        pytest.importorskip("ultralytics")

        output_path = temp_dir / "yolov5n.onnx"
        output_path.write_bytes(b"dummy")

        export_yolov5n(output_path, force=True)

        assert output_path.stat().st_size > 100  # Real model is much larger


class TestExportMobileNetV2:
    """Tests for MobileNetV2 export function."""

    @pytest.mark.slow
    def test_export_creates_file(self, temp_dir: Path) -> None:
        """Export should create ONNX file."""
        pytest.importorskip("torch")
        pytest.importorskip("torchvision")

        output_path = temp_dir / "mobilenetv2.onnx"
        export_mobilenetv2(output_path)

        assert output_path.exists()

    @pytest.mark.slow
    def test_export_returns_result(self, temp_dir: Path) -> None:
        """Export should return ExportResult."""
        pytest.importorskip("torch")
        pytest.importorskip("torchvision")

        output_path = temp_dir / "mobilenetv2.onnx"
        result = export_mobilenetv2(output_path)

        assert isinstance(result, ExportResult)
        assert result.output_shape == (1, 1000)

    @pytest.mark.slow
    def test_export_correct_input_shape(self, temp_dir: Path) -> None:
        """Exported model should have correct input shape."""
        pytest.importorskip("torch")
        pytest.importorskip("torchvision")

        output_path = temp_dir / "mobilenetv2.onnx"
        result = export_mobilenetv2(output_path)

        assert result.input_shape == (1, 3, 224, 224)


# =============================================================================
# Tests for SessionConfig
# =============================================================================


class TestSessionConfig:
    """Tests for SessionConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config should use 2 intra-op, 1 inter-op threads."""
        config = SessionConfig()

        assert config.intra_op_threads == DEFAULT_INTRA_OP_THREADS
        assert config.inter_op_threads == DEFAULT_INTER_OP_THREADS

    def test_default_providers(self) -> None:
        """Default providers should be CPUExecutionProvider."""
        config = SessionConfig()

        assert config.providers == ["CPUExecutionProvider"]

    def test_custom_values(self) -> None:
        """Should accept custom configuration."""
        config = SessionConfig(
            intra_op_threads=4,
            inter_op_threads=2,
            providers=["CUDAExecutionProvider"],
        )

        assert config.intra_op_threads == 4
        assert config.inter_op_threads == 2


# =============================================================================
# Tests for ModelRegistry
# =============================================================================


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_init_creates_registry(self, temp_dir: Path) -> None:
        """Should create registry instance."""
        registry = ModelRegistry(models_dir=temp_dir)

        assert registry.models_dir == temp_dir

    def test_init_with_custom_config(self, temp_dir: Path) -> None:
        """Should accept custom session config."""
        config = SessionConfig(intra_op_threads=4)
        registry = ModelRegistry(models_dir=temp_dir, config=config)

        assert registry.config.intra_op_threads == 4

    def test_get_session_file_not_found(self, temp_dir: Path) -> None:
        """Should raise FileNotFoundError for missing model."""
        registry = ModelRegistry(models_dir=temp_dir)

        with pytest.raises(FileNotFoundError):
            registry.get_session("yolov5n")

    def test_get_session_loads_model(
        self,
        registry_with_mock_model: ModelRegistry,
    ) -> None:
        """Should load model and return session."""
        session = registry_with_mock_model.get_session("yolov5n")

        assert session is not None

    def test_get_session_caches_session(
        self,
        registry_with_mock_model: ModelRegistry,
    ) -> None:
        """Should cache session for subsequent calls."""
        session1 = registry_with_mock_model.get_session("yolov5n")
        session2 = registry_with_mock_model.get_session("yolov5n")

        assert session1 is session2

    def test_is_loaded(
        self,
        registry_with_mock_model: ModelRegistry,
    ) -> None:
        """Should track loaded models."""
        assert not registry_with_mock_model.is_loaded("yolov5n")

        registry_with_mock_model.get_session("yolov5n")

        assert registry_with_mock_model.is_loaded("yolov5n")

    def test_get_model_info(
        self,
        registry_with_mock_model: ModelRegistry,
    ) -> None:
        """Should return model info after loading."""
        registry_with_mock_model.get_session("yolov5n")
        info = registry_with_mock_model.get_model_info("yolov5n")

        assert isinstance(info, ModelInfo)
        assert info.name == "yolov5n"

    def test_clear_cache(
        self,
        registry_with_mock_model: ModelRegistry,
    ) -> None:
        """Should clear cached sessions."""
        registry_with_mock_model.get_session("yolov5n")
        assert registry_with_mock_model.is_loaded("yolov5n")

        registry_with_mock_model.clear_cache()

        assert not registry_with_mock_model.is_loaded("yolov5n")

    def test_list_available(
        self,
        registry_with_mock_model: ModelRegistry,
    ) -> None:
        """Should list available models."""
        available = registry_with_mock_model.list_available()

        assert "yolov5n" in available


class TestGetDefaultRegistry:
    """Tests for default registry singleton."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_default_registry()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_default_registry()

    def test_returns_registry(self, temp_dir: Path) -> None:
        """Should return ModelRegistry instance."""
        registry = get_default_registry(models_dir=temp_dir)

        assert isinstance(registry, ModelRegistry)

    def test_returns_same_instance(self, temp_dir: Path) -> None:
        """Should return same instance on subsequent calls."""
        registry1 = get_default_registry(models_dir=temp_dir)
        registry2 = get_default_registry()

        assert registry1 is registry2

    def test_reset_clears_singleton(self, temp_dir: Path) -> None:
        """Reset should clear the singleton."""
        registry1 = get_default_registry(models_dir=temp_dir)
        reset_default_registry()
        registry2 = get_default_registry(models_dir=temp_dir)

        assert registry1 is not registry2

    def test_singleton_thread_safe(self, temp_dir: Path) -> None:
        """Concurrent calls to get_default_registry should return same instance."""
        NUM_THREADS = 10
        barrier = threading.Barrier(NUM_THREADS)
        registry_ids: list[int] = []

        def get_registry(thread_id: int) -> int:
            barrier.wait()  # Synchronize for maximum contention
            registry = get_default_registry(models_dir=temp_dir)
            return id(registry)

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = [executor.submit(get_registry, i) for i in range(NUM_THREADS)]
            for future in as_completed(futures):
                registry_ids.append(future.result())

        # All threads should get the same singleton instance
        unique_registries = set(registry_ids)
        assert len(unique_registries) == 1, (
            f"Singleton violated: got {len(unique_registries)} different instances"
        )


# =============================================================================
# Integration Tests
# =============================================================================


class TestModelIntegration:
    """Integration tests for model export and registry."""

    @pytest.mark.slow
    def test_export_and_load(self, temp_dir: Path) -> None:
        """Export model and load with registry."""
        pytest.importorskip("torch")
        pytest.importorskip("torchvision")

        # Export
        export_mobilenetv2(temp_dir / "mobilenetv2.onnx")

        # Load with registry
        registry = ModelRegistry(models_dir=temp_dir)
        session = registry.get_session("mobilenetv2")

        # Run inference
        input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
        output = session.run(None, {"input": input_tensor})

        assert output[0].shape == (1, 1000)

    @pytest.mark.slow
    def test_export_all_models(self, temp_dir: Path) -> None:
        """Export all models function."""
        pytest.importorskip("torch")
        pytest.importorskip("torchvision")
        pytest.importorskip("ultralytics")

        results = export_all_models(temp_dir)

        assert "yolov5n" in results
        assert "mobilenetv2" in results
        assert (temp_dir / "yolov5n.onnx").exists()
        assert (temp_dir / "mobilenetv2.onnx").exists()


# =============================================================================
# Tests for ModelRegistry Thread Safety
# =============================================================================


class TestModelRegistryThreadSafety:
    """Thread safety tests for ModelRegistry under concurrent load.

    Per CONTEXT.md:
    - 5-10 threads (low concurrency)
    - Test read/write races
    - Success criteria: no exceptions AND data consistency
    - Single run per test (no iterations)
    """

    NUM_THREADS = 10

    def test_concurrent_get_session_no_exceptions(
        self,
        registry_with_mock_model: ModelRegistry,
    ) -> None:
        """Multiple threads can get_session() without exceptions."""
        errors: list[tuple[int, Exception]] = []
        results: list[tuple[int, int]] = []  # (thread_id, session_id)

        def get_session_task(thread_id: int) -> None:
            try:
                session = registry_with_mock_model.get_session("yolov5n")
                results.append((thread_id, id(session)))
            except Exception as e:
                errors.append((thread_id, e))

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(get_session_task, i) for i in range(self.NUM_THREADS)]
            for future in as_completed(futures):
                future.result()  # Propagate any unexpected exceptions

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == self.NUM_THREADS

    def test_concurrent_readers_get_same_instance(
        self,
        registry_with_mock_model: ModelRegistry,
    ) -> None:
        """All concurrent readers should receive same cached session instance."""
        session_ids: list[int] = []
        barrier = threading.Barrier(self.NUM_THREADS)

        def synchronized_read(thread_id: int) -> int:
            barrier.wait()  # Synchronize start for maximum contention
            session = registry_with_mock_model.get_session("yolov5n")
            return id(session)

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(synchronized_read, i) for i in range(self.NUM_THREADS)]
            for future in as_completed(futures):
                session_ids.append(future.result())

        # All threads should get the exact same session instance
        unique_sessions = set(session_ids)
        assert len(unique_sessions) == 1, (
            f"Expected 1 unique session, got {len(unique_sessions)}: {unique_sessions}"
        )

    def test_concurrent_read_during_clear_no_corruption(
        self,
        temp_dir: Path,
        mock_onnx_model: Path,
    ) -> None:
        """Concurrent reads during cache clear should not corrupt state."""
        import shutil

        # Setup fresh registry
        yolo_path = temp_dir / "yolov5n.onnx"
        shutil.copy(mock_onnx_model, yolo_path)
        registry = ModelRegistry(models_dir=temp_dir)

        # Pre-load to warm cache
        registry.get_session("yolov5n")

        errors: list[tuple[int, Exception]] = []
        NUM_READERS = 8
        NUM_WRITERS = 2

        def reader_task(thread_id: int) -> None:
            try:
                # May get session or see cleared cache (both OK)
                registry.get_session("yolov5n")
            except FileNotFoundError:
                # This is acceptable - model file still exists
                pass
            except Exception as e:
                errors.append((thread_id, e))

        def writer_task(thread_id: int) -> None:
            try:
                registry.clear_cache()
            except Exception as e:
                errors.append((thread_id, e))

        with ThreadPoolExecutor(max_workers=NUM_READERS + NUM_WRITERS) as executor:
            # Mix of readers and writers
            futures = []
            for i in range(NUM_READERS):
                futures.append(executor.submit(reader_task, i))
            for i in range(NUM_WRITERS):
                futures.append(executor.submit(writer_task, NUM_READERS + i))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass  # Errors captured in errors list

        # No corruption or unexpected exceptions
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_is_loaded_thread_safe(
        self,
        registry_with_mock_model: ModelRegistry,
    ) -> None:
        """is_loaded() should be consistent under concurrent access."""
        # First load the model
        registry_with_mock_model.get_session("yolov5n")

        results: list[bool] = []

        def check_loaded(thread_id: int) -> bool:
            return registry_with_mock_model.is_loaded("yolov5n")

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(check_loaded, i) for i in range(self.NUM_THREADS)]
            for future in as_completed(futures):
                results.append(future.result())

        # All should report True (model was loaded)
        assert all(results), f"Inconsistent is_loaded results: {results}"
