"""ONNX Model Exporter Package.

This package exports PyTorch models to ONNX format with controlled parameters
to ensure reproducibility and compatibility across all architectures.

Exports:
- YOLOv5n: Object detection model (640x640 input, opset 17)
- MobileNetV2: Classification model (224x224 input, opset 17)

Both models are exported with:
- Static input shapes (batch_size=1)
- ONNX opset version 17 (Triton 24.08 compatibility)
- SHA256 checksums for verification

Package Structure:
- types.py: Data classes (ExportResult) and constants
- utils.py: Shared utilities (compute_checksum, verify_onnx_model)
- detection.py: YOLO export (export_yolov5n)
- classification.py: MobileNet export (export_mobilenetv2)

Author: Matthew Hong
Specification Reference: Foundation Specification Section 2 Model Export
"""

import logging
from pathlib import Path

from .classification import export_mobilenetv2
from .detection import export_yolov5n
from .types import (
    MOBILENET_INPUT_SIZE,
    ONNX_OPSET_VERSION,
    YOLO_INPUT_SIZE,
    ExportResult,
)
from .utils import compute_checksum, verify_onnx_model

logger = logging.getLogger(__name__)


# =============================================================================
# Batch Export
# =============================================================================


def export_all_models(
    output_dir: Path,
    force: bool = False,
) -> dict[str, ExportResult]:
    """Export all models required for the experiment.

    Args:
        output_dir: Directory to save ONNX files
        force: Overwrite existing files if True

    Returns:
        Dictionary mapping model name to ExportResult

    Example:
        >>> results = export_all_models(Path("models/"))
        >>> results["yolov5n"].checksum
        'a1b2c3...'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Export YOLOv5n
    try:
        results["yolov5n"] = export_yolov5n(
            output_dir / "yolov5n.onnx",
            force=force,
        )
    except FileExistsError:
        logger.info("YOLOv5n already exists, skipping (use force=True to overwrite)")
        # Load existing model info
        model_path = output_dir / "yolov5n.onnx"
        verification = verify_onnx_model(model_path)
        if verification["valid"]:
            results["yolov5n"] = ExportResult(
                model_path=model_path,
                checksum=compute_checksum(model_path),
                opset_version=verification["opset_version"],
                input_shape=verification["input_shapes"][0],
                output_shape=verification["output_shapes"][0],
                file_size_mb=model_path.stat().st_size / (1024 * 1024),
            )

    # Export MobileNetV2
    try:
        results["mobilenetv2"] = export_mobilenetv2(
            output_dir / "mobilenetv2.onnx",
            force=force,
        )
    except FileExistsError:
        logger.info("MobileNetV2 already exists, skipping (use force=True to overwrite)")
        # Load existing model info
        model_path = output_dir / "mobilenetv2.onnx"
        verification = verify_onnx_model(model_path)
        if verification["valid"]:
            results["mobilenetv2"] = ExportResult(
                model_path=model_path,
                checksum=compute_checksum(model_path),
                opset_version=verification["opset_version"],
                input_shape=verification["input_shapes"][0],
                output_shape=verification["output_shapes"][0],
                file_size_mb=model_path.stat().st_size / (1024 * 1024),
            )

    return results


__all__ = [
    # Types and constants
    "ExportResult",
    "ONNX_OPSET_VERSION",
    "YOLO_INPUT_SIZE",
    "MOBILENET_INPUT_SIZE",
    # Utilities
    "compute_checksum",
    "verify_onnx_model",
    # Detection exports
    "export_yolov5n",
    # Classification exports
    "export_mobilenetv2",
    # Batch export
    "export_all_models",
]
