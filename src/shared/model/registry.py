"""ONNX Model Registry.

This module provides a registry for loading and caching ONNX Runtime
inference sessions with consistent configuration across all architectures.

Features:
- Lazy loading: Models loaded on first access
- Session caching: Avoid redundant model loading
- Thread configuration: Consistent intra_op/inter_op thread settings
- Validation: Verify model compatibility before inference

All architectures use this registry to ensure identical inference
session configuration, eliminating runtime variance as a confounding variable.

Author: Matthew Hong
Specification Reference: Foundation Specification §4 Resource Constraints
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_INTRA_OP_THREADS: int = 2
"""ONNX Runtime intra-op parallelism (within single operator)."""

DEFAULT_INTER_OP_THREADS: int = 1
"""ONNX Runtime inter-op parallelism (across operators)."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ModelInfo:
    """Information about a loaded model.

    Attributes:
        name: Model identifier
        path: Path to ONNX file
        input_name: Name of input tensor
        input_shape: Expected input shape
        input_dtype: Expected input dtype
        output_name: Name of output tensor
        output_shape: Expected output shape
    """

    name: str
    path: Path
    input_name: str
    input_shape: tuple[int, ...]
    input_dtype: np.dtype
    output_name: str
    output_shape: tuple[int, ...]


@dataclass
class SessionConfig:
    """Configuration for ONNX Runtime inference session.

    Attributes:
        intra_op_threads: Number of threads for intra-op parallelism
        inter_op_threads: Number of threads for inter-op parallelism
        providers: Execution providers (default: CPUExecutionProvider)
    """

    intra_op_threads: int = DEFAULT_INTRA_OP_THREADS
    inter_op_threads: int = DEFAULT_INTER_OP_THREADS
    providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])


# =============================================================================
# Model Registry
# =============================================================================


class ModelRegistry:
    """Registry for loading and caching ONNX Runtime inference sessions.

    Provides consistent model loading with:
    - Configurable thread settings for fair CPU allocation
    - Session caching to avoid redundant loading
    - Thread-safe access for concurrent inference

    Example:
        >>> registry = ModelRegistry(models_dir=Path("models/"))
        >>> session = registry.get_session("yolov5n")
        >>> output = session.run(None, {"images": input_tensor})

    Attributes:
        models_dir: Base directory for ONNX model files
        config: Session configuration (thread settings, providers)
    """

    # Map of known model names to filenames
    MODEL_FILES: dict[str, str] = {
        "yolov5n": "yolov5n.onnx",
        "mobilenetv2": "mobilenetv2.onnx",
    }

    def __init__(
        self,
        models_dir: Path,
        config: SessionConfig | None = None,
    ) -> None:
        """Initialize ModelRegistry.

        Args:
            models_dir: Directory containing ONNX model files
            config: Session configuration (default: 2 intra-op, 1 inter-op threads)
        """
        self.models_dir = Path(models_dir)
        self.config = config or SessionConfig()

        self._sessions: dict[str, ort.InferenceSession] = {}
        self._model_info: dict[str, ModelInfo] = {}
        self._lock = Lock()

        logger.info("ModelRegistry initialized")
        logger.info(f"  Models dir: {self.models_dir}")
        logger.info(f"  Intra-op threads: {self.config.intra_op_threads}")
        logger.info(f"  Inter-op threads: {self.config.inter_op_threads}")

    def get_session(self, model_name: str) -> "ort.InferenceSession":
        """Get ONNX Runtime inference session for a model.

        Sessions are cached after first load. Thread-safe for concurrent access.

        Args:
            model_name: Name of model ("yolov5n" or "mobilenetv2")

        Returns:
            ONNX Runtime InferenceSession ready for inference

        Raises:
            ValueError: If model_name is not recognized
            FileNotFoundError: If model file not found
            RuntimeError: If model fails to load

        Example:
            >>> session = registry.get_session("yolov5n")
            >>> input_name = session.get_inputs()[0].name
        """
        with self._lock:
            if model_name not in self._sessions:
                self._load_model(model_name)

            return self._sessions[model_name]

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get information about a loaded model.

        Args:
            model_name: Name of model

        Returns:
            ModelInfo with input/output specifications

        Raises:
            ValueError: If model not loaded

        Example:
            >>> info = registry.get_model_info("yolov5n")
            >>> info.input_shape
            (1, 3, 640, 640)
        """
        if model_name not in self._model_info:
            # Trigger load if not loaded
            self.get_session(model_name)

        return self._model_info[model_name]

    def _load_model(self, model_name: str) -> None:
        """Load a model into the registry.

        Args:
            model_name: Name of model to load

        Raises:
            ValueError: If model_name is not recognized
            FileNotFoundError: If model file not found
        """
        import onnxruntime as ort

        # Resolve model path
        if model_name in self.MODEL_FILES:
            model_path = self.models_dir / self.MODEL_FILES[model_name]
        else:
            # Allow direct filename
            model_path = self.models_dir / f"{model_name}.onnx"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                f"Run 'make setup-models' or 'python scripts/export_models.py' first."
            )

        logger.info(f"Loading model: {model_name} from {model_path}")

        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.config.intra_op_threads
        sess_options.inter_op_num_threads = self.config.inter_op_threads

        # Disable memory pattern optimization for consistent behavior
        sess_options.enable_mem_pattern = False

        # Create session
        session = ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=self.config.providers,
        )

        # Extract model info
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]

        # Map ONNX dtype to numpy dtype
        onnx_to_numpy = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
        }

        model_info = ModelInfo(
            name=model_name,
            path=model_path,
            input_name=input_meta.name,
            input_shape=tuple(input_meta.shape),
            input_dtype=onnx_to_numpy.get(input_meta.type, np.float32),
            output_name=output_meta.name,
            output_shape=tuple(output_meta.shape),
        )

        # Cache session and info
        self._sessions[model_name] = session
        self._model_info[model_name] = model_info

        logger.info(f"  ✓ Loaded {model_name}")
        logger.info(f"    Input: {model_info.input_name} {model_info.input_shape}")
        logger.info(f"    Output: {model_info.output_name} {model_info.output_shape}")

    def is_loaded(self, model_name: str) -> bool:
        """Check if a model is already loaded.

        Args:
            model_name: Name of model

        Returns:
            True if model session is cached
        """
        return model_name in self._sessions

    def preload_all(self) -> None:
        """Preload all known models into cache.

        Useful for warming up before experiments to avoid
        first-request loading latency.
        """
        for model_name in self.MODEL_FILES:
            if not self.is_loaded(model_name):
                try:
                    self.get_session(model_name)
                except FileNotFoundError as e:
                    logger.warning(f"Could not preload {model_name}: {e}")

    def clear_cache(self) -> None:
        """Clear all cached sessions.

        Useful for testing or when models have been updated.
        """
        with self._lock:
            self._sessions.clear()
            self._model_info.clear()
            logger.info("Model cache cleared")

    def list_available(self) -> list[str]:
        """List models available in the models directory.

        Returns:
            List of model names that can be loaded
        """
        available = []

        for name, filename in self.MODEL_FILES.items():
            if (self.models_dir / filename).exists():
                available.append(name)

        return available


# =============================================================================
# Default Registry Singleton
# =============================================================================

_default_registry: ModelRegistry | None = None
_default_registry_lock = Lock()


def get_default_registry(
    models_dir: Path | None = None,
    config: SessionConfig | None = None,
) -> ModelRegistry:
    """Get or create the default ModelRegistry singleton.

    Args:
        models_dir: Directory containing models (default: ./models)
        config: Session configuration

    Returns:
        Shared ModelRegistry instance

    Example:
        >>> registry = get_default_registry()
        >>> session = registry.get_session("yolov5n")
    """
    global _default_registry

    with _default_registry_lock:
        if _default_registry is None:
            if models_dir is None:
                # Default to models/ in project root
                models_dir = Path(__file__).parent.parent.parent.parent / "models"

            _default_registry = ModelRegistry(models_dir, config)

        return _default_registry


def reset_default_registry() -> None:
    """Reset the default registry singleton.

    Useful for testing or reconfiguration.
    """
    global _default_registry

    with _default_registry_lock:
        if _default_registry is not None:
            _default_registry.clear_cache()
        _default_registry = None
