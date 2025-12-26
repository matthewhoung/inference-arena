"""MobileNetV2 classification inference engine.

This module provides the core classification functionality using ONNX Runtime.
Handles preprocessing, inference, and result formatting.

Author: Matthew Hong
"""

import logging
from pathlib import Path

import numpy as np

from shared.model.registry import ModelRegistry
from shared.processing import MobileNetPreprocessor

logger = logging.getLogger(__name__)


class ClassificationInference:
    """MobileNetV2 classification inference engine.

    Loads and runs MobileNetV2 for image classification using ONNX Runtime.
    Threading configuration is loaded via the ModelRegistry.

    Attributes:
        registry: ModelRegistry for ONNX session management
        mobilenet_session: MobileNetV2 inference session
        labels: ImageNet class labels [0-999]
        preprocessor: MobileNet preprocessing pipeline
    """

    def __init__(self, registry: ModelRegistry, labels_file: Path) -> None:
        """Initialize the classification inference engine.

        Args:
            registry: Pre-configured ModelRegistry instance
            labels_file: Path to ImageNet labels file (1000 lines)
        """
        self.registry = registry

        # Load MobileNetV2 model
        logger.info("Loading MobileNetV2 model")
        self.mobilenet_session = self.registry.get_session("mobilenetv2")

        # Get model info
        model_info = self.registry.get_model_info("mobilenetv2")
        logger.info(f"MobileNetV2 input shape: {model_info.input_shape}")

        # Load ImageNet labels
        logger.info(f"Loading ImageNet labels from {labels_file}")
        self.labels = self._load_labels(labels_file)

        # Initialize preprocessor
        self.preprocessor = MobileNetPreprocessor()

        logger.info("Classification inference engine ready")

    def _load_labels(self, labels_file: Path) -> list[str]:
        """Load ImageNet class labels from file.

        Args:
            labels_file: Path to labels file (one per line)

        Returns:
            List of 1000 class names indexed 0-999

        Raises:
            FileNotFoundError: If labels file not found
            ValueError: If labels file doesn't contain exactly 1000 lines
        """
        labels_file = Path(labels_file)
        if not labels_file.exists():
            raise FileNotFoundError(
                f"ImageNet labels file not found: {labels_file}. "
                "Ensure src/shared/data/imagenet_labels.txt exists."
            )

        with open(labels_file) as f:
            labels = [line.strip() for line in f]

        if len(labels) != 1000:
            raise ValueError(
                f"Expected 1000 ImageNet labels, got {len(labels)}. "
                f"Check {labels_file} format."
            )

        logger.info(f"Loaded {len(labels)} ImageNet class labels")
        return labels

    def classify(
        self,
        crop: np.ndarray,
        top_k: int = 5,
    ) -> tuple[int, str, float, list[tuple[int, str, float]]]:
        """Classify an image crop.

        Args:
            crop: RGB uint8 numpy array [H, W, 3]
            top_k: Number of top predictions to return

        Returns:
            Tuple of (class_id, class_name, confidence, top_k_results)
            where top_k_results is a list of (class_id, class_name, confidence) tuples
        """
        # Preprocess
        result = self.preprocessor(crop)

        # Run inference
        output = self.mobilenet_session.run(None, {"input": result.tensor})[0]

        # Apply softmax to get probabilities
        scores = output[0]
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        probs = exp_scores / exp_scores.sum()

        # Get top-k indices
        top_k_indices = np.argsort(probs)[::-1][:top_k]

        # Build top-k results
        top_k_results = []
        for idx in top_k_indices:
            top_k_results.append((
                int(idx),
                self.labels[idx],
                float(probs[idx]),
            ))

        # Top-1 result
        top_class_id = int(top_k_indices[0])
        top_class_name = self.labels[top_class_id]
        top_confidence = float(probs[top_class_id])

        return top_class_id, top_class_name, top_confidence, top_k_results
