"""
Unit Tests for Experiment Configuration Module

This module tests shared/config.py which provides the Python interface
to experiment.yaml - the single source of truth for experimental parameters.

Test Categories:
- Config loading: File parsing and caching
- Controlled variables: Access and validation
- Model configuration: Model-specific settings
- Hypotheses: Pre-registered hypothesis access
- Infrastructure: MinIO and Triton config
- Validation: Config integrity checks

Author: Matthew Hong
"""

import pytest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch, mock_open

# Import module under test
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shared.config import (
    get_config,
    reload_config,
    get_controlled_variable,
    get_controlled_variables,
    get_model_config,
    get_model_names,
    get_hypothesis,
    get_hypotheses_by_category,
    get_infrastructure_config,
    get_minio_config,
    get_triton_config,
    get_load_testing_config,
    get_concurrent_user_levels,
    get_metadata,
    get_spec_version,
    validate_config,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def clear_config_cache():
    """Clear config cache before each test."""
    reload_config.cache_clear() if hasattr(reload_config, 'cache_clear') else None
    get_config.cache_clear()
    yield
    get_config.cache_clear()


@pytest.fixture
def config() -> Dict[str, Any]:
    """Load the actual config for testing."""
    return get_config()


# =============================================================================
# Config Loading Tests
# =============================================================================

class TestConfigLoading:
    """Test configuration file loading."""

    def test_config_loads_successfully(self) -> None:
        """Config should load without errors."""
        config = get_config()
        assert config is not None
        assert isinstance(config, dict)

    def test_config_is_cached(self) -> None:
        """Subsequent calls should return cached config."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2  # Same object (cached)

    def test_reload_clears_cache(self) -> None:
        """reload_config should clear the cache."""
        config1 = get_config()
        config2 = reload_config()
        # After reload, should be new dict (though equal content)
        assert config1 == config2

    def test_config_has_required_sections(self, config: Dict[str, Any]) -> None:
        """Config should have all required top-level sections."""
        required = [
            "metadata",
            "research_questions",
            "hypotheses",
            "independent_variables",
            "controlled_variables",
            "infrastructure",
        ]
        for section in required:
            assert section in config, f"Missing section: {section}"


# =============================================================================
# Metadata Tests
# =============================================================================

class TestMetadata:
    """Test metadata access."""

    def test_get_metadata(self) -> None:
        """Should return metadata dictionary."""
        meta = get_metadata()
        assert isinstance(meta, dict)
        assert "title" in meta
        assert "author" in meta

    def test_get_spec_version(self) -> None:
        """Should return version string."""
        version = get_spec_version()
        assert isinstance(version, str)
        assert version  # Not empty
        # Should be semver-like
        parts = version.split(".")
        assert len(parts) >= 2


# =============================================================================
# Controlled Variables Tests
# =============================================================================

class TestControlledVariables:
    """Test controlled variable access."""

    def test_get_onnx_runtime_config(self) -> None:
        """Should return ONNX runtime settings."""
        intra = get_controlled_variable("onnx_runtime", "intra_op_num_threads")
        inter = get_controlled_variable("onnx_runtime", "inter_op_num_threads")
        
        assert intra == 2
        assert inter == 1

    def test_get_resources_config(self) -> None:
        """Should return resource allocation settings."""
        vcpu = get_controlled_variable("resources", "vcpu_per_container")
        memory = get_controlled_variable("resources", "memory_gb_per_container")
        
        assert vcpu == 2
        assert memory == 4

    def test_get_dataset_config(self) -> None:
        """Should return dataset configuration."""
        sample_size = get_controlled_variable("dataset", "sample_size")
        target_mean = get_controlled_variable("dataset", "target_distribution")["mean"]
        
        assert sample_size == 100
        assert target_mean == 4.0

    def test_get_controlled_variables_section(self) -> None:
        """Should return entire section."""
        onnx = get_controlled_variables("onnx_runtime")
        
        assert isinstance(onnx, dict)
        assert "intra_op_num_threads" in onnx
        assert "inter_op_num_threads" in onnx

    def test_invalid_section_raises_keyerror(self) -> None:
        """Should raise KeyError for invalid section."""
        with pytest.raises(KeyError) as exc_info:
            get_controlled_variable("nonexistent_section", "key")
        
        assert "nonexistent_section" in str(exc_info.value)

    def test_invalid_key_raises_keyerror(self) -> None:
        """Should raise KeyError for invalid key."""
        with pytest.raises(KeyError) as exc_info:
            get_controlled_variable("onnx_runtime", "nonexistent_key")
        
        assert "nonexistent_key" in str(exc_info.value)


# =============================================================================
# Model Configuration Tests
# =============================================================================

class TestModelConfiguration:
    """Test model configuration access."""

    def test_get_model_names(self) -> None:
        """Should return list of model names."""
        names = get_model_names()
        
        assert isinstance(names, list)
        assert "yolov5n" in names
        assert "mobilenetv2" in names

    def test_get_yolov5n_config(self) -> None:
        """Should return YOLOv5n configuration."""
        yolo = get_model_config("yolov5n")
        
        assert yolo["name"] == "yolov5n"
        assert yolo["format"] == "onnx"
        assert yolo["opset_version"] == 17
        assert yolo["input"]["shape"] == [1, 3, 640, 640]
        assert yolo["output"]["shape"] == [1, 84, 8400]

    def test_get_mobilenetv2_config(self) -> None:
        """Should return MobileNetV2 configuration."""
        mobilenet = get_model_config("mobilenetv2")
        
        assert mobilenet["name"] == "mobilenetv2"
        assert mobilenet["format"] == "onnx"
        assert mobilenet["opset_version"] == 17
        assert mobilenet["input"]["shape"] == [1, 3, 224, 224]
        assert mobilenet["output"]["shape"] == [1, 1000]

    def test_invalid_model_raises_keyerror(self) -> None:
        """Should raise KeyError for invalid model name."""
        with pytest.raises(KeyError):
            get_model_config("nonexistent_model")


# =============================================================================
# Hypothesis Tests
# =============================================================================

class TestHypotheses:
    """Test hypothesis access."""

    def test_get_hypothesis_h1a(self) -> None:
        """Should return H1a hypothesis."""
        h1a = get_hypothesis("H1a")
        
        assert isinstance(h1a, dict)
        assert "statement" in h1a
        assert "rationale" in h1a
        assert "testable_prediction" in h1a
        assert h1a["category"] == "performance"

    def test_get_hypotheses_by_category(self) -> None:
        """Should filter hypotheses by category."""
        performance = get_hypotheses_by_category("performance")
        
        assert isinstance(performance, dict)
        assert "H1a" in performance
        assert "H1b" in performance
        
        # All should be performance category
        for h_id, h_config in performance.items():
            assert h_config["category"] == "performance"

    def test_hypothesis_categories(self) -> None:
        """Should have hypotheses in all expected categories."""
        categories = ["performance", "resource_efficiency", "operational_complexity"]
        
        for category in categories:
            hypotheses = get_hypotheses_by_category(category)
            assert len(hypotheses) > 0, f"No hypotheses in category: {category}"

    def test_invalid_hypothesis_raises_keyerror(self) -> None:
        """Should raise KeyError for invalid hypothesis ID."""
        with pytest.raises(KeyError):
            get_hypothesis("H99z")


# =============================================================================
# Infrastructure Tests
# =============================================================================

class TestInfrastructureConfig:
    """Test infrastructure configuration access."""

    def test_get_minio_config(self) -> None:
        """Should return MinIO configuration."""
        minio = get_minio_config()
        
        assert isinstance(minio, dict)
        assert "bucket" in minio
        assert "endpoint" in minio
        assert minio["bucket"] == "models"

    def test_get_triton_config(self) -> None:
        """Should return Triton configuration."""
        triton = get_triton_config()
        
        assert isinstance(triton, dict)
        assert "model_repository" in triton
        assert "instance_group" in triton
        assert "parameters" in triton

    def test_triton_threading_matches_onnx(self) -> None:
        """Triton threading should match ONNX runtime settings."""
        triton = get_triton_config()
        onnx_intra = get_controlled_variable("onnx_runtime", "intra_op_num_threads")
        onnx_inter = get_controlled_variable("onnx_runtime", "inter_op_num_threads")
        
        triton_params = triton["parameters"]
        assert triton_params["intra_op_thread_count"] == str(onnx_intra)
        assert triton_params["inter_op_thread_count"] == str(onnx_inter)

    def test_get_infrastructure_config(self) -> None:
        """Should return full infrastructure config."""
        infra = get_infrastructure_config()
        
        assert "orchestration" in infra
        assert "minio" in infra
        assert "networks" in infra


# =============================================================================
# Load Testing Tests
# =============================================================================

class TestLoadTestingConfig:
    """Test load testing configuration access."""

    def test_get_load_testing_config(self) -> None:
        """Should return load testing configuration."""
        lt = get_load_testing_config()
        
        assert "phases" in lt
        assert "runs_per_configuration" in lt
        assert lt["tool"] == "locust"

    def test_phase_durations(self) -> None:
        """Should have correct phase durations."""
        lt = get_load_testing_config()
        phases = lt["phases"]
        
        assert phases["warmup"]["duration_seconds"] == 60
        assert phases["measurement"]["duration_seconds"] == 180
        assert phases["cooldown"]["duration_seconds"] == 30

    def test_get_concurrent_user_levels(self) -> None:
        """Should return user concurrency levels."""
        levels = get_concurrent_user_levels()
        
        assert isinstance(levels, list)
        assert levels == [1, 5, 10, 25, 50, 75, 100]


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidation:
    """Test configuration validation."""

    def test_validate_config_passes(self) -> None:
        """Validation should pass for valid config."""
        errors = validate_config()
        
        assert isinstance(errors, list)
        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_validation_checks_models(self) -> None:
        """Validation should check for required models."""
        # This tests the validation logic indirectly
        errors = validate_config()
        
        # If we got here without errors, models are valid
        assert len(errors) == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestConfigIntegration:
    """Integration tests for config consistency."""

    def test_model_input_shapes_are_valid(self) -> None:
        """Model input shapes should be valid 4D tensors."""
        for model_name in get_model_names():
            config = get_model_config(model_name)
            input_shape = config["input"]["shape"]
            
            assert len(input_shape) == 4, f"{model_name} input should be 4D"
            assert input_shape[0] == 1, f"{model_name} batch size should be 1"
            assert input_shape[1] == 3, f"{model_name} should have 3 channels"

    def test_hypothesis_predictions_reference_architectures(self) -> None:
        """Hypothesis predictions should reference valid architectures."""
        config = get_config()
        architectures = config["independent_variables"]["architecture"]["levels"]
        
        for h_id, h_config in config["hypotheses"].items():
            prediction = h_config.get("testable_prediction", "")
            # At least one architecture should be mentioned
            mentioned = any(arch in prediction.lower() for arch in architectures)
            assert mentioned or "all" in prediction.lower(), \
                f"{h_id} prediction doesn't reference architectures"

    def test_concurrent_users_are_ordered(self) -> None:
        """Concurrent user levels should be in ascending order."""
        levels = get_concurrent_user_levels()
        
        assert levels == sorted(levels)
        assert levels[0] == 1  # Start at 1
        assert levels[-1] == 100  # End at 100
