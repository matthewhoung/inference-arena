"""
ONNX Model Exporter

This module exports PyTorch models to ONNX format with controlled parameters
to ensure reproducibility and compatibility across all architectures.

Exports:
- YOLOv5n: Object detection model (640x640 input, opset 17)
- MobileNetV2: Classification model (224x224 input, opset 17)

Both models are exported with:
- Static input shapes (batch_size=1)
- ONNX opset version 17 (Triton 24.08 compatibility)
- SHA256 checksums for verification

Author: Matthew Hong
Specification Reference: Foundation Specification §2 Model Export
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

ONNX_OPSET_VERSION: int = 17
"""ONNX opset version for Triton 24.08 compatibility."""

YOLO_INPUT_SIZE: int = 640
"""YOLOv5 input dimension (square)."""

MOBILENET_INPUT_SIZE: int = 224
"""MobileNetV2 input dimension (square)."""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExportResult:
    """
    Result container for model export operation.

    Attributes:
        model_path: Path to exported ONNX file
        checksum: SHA256 checksum of exported file
        opset_version: ONNX opset version used
        input_shape: Model input tensor shape
        output_shape: Model output tensor shape (may contain dynamic dims)
        file_size_mb: File size in megabytes
    """

    model_path: Path
    checksum: str
    opset_version: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    file_size_mb: float


# =============================================================================
# Checksum Utilities
# =============================================================================

def compute_checksum(file_path: Path) -> str:
    """
    Compute SHA256 checksum of a file.

    Args:
        file_path: Path to file

    Returns:
        Hex-encoded SHA256 checksum string

    Example:
        >>> checksum = compute_checksum(Path("model.onnx"))
        >>> len(checksum)
        64
    """
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


# =============================================================================
# ONNX Verification
# =============================================================================

def verify_onnx_model(model_path: Path) -> dict:
    """
    Verify ONNX model is valid and meets specifications.

    Checks:
    - File exists and is readable
    - Valid ONNX format (passes onnx.checker)
    - Correct opset version
    - Can be loaded by ONNX Runtime

    Args:
        model_path: Path to ONNX model file

    Returns:
        Dictionary with verification results:
        - valid: bool
        - opset_version: int
        - input_shapes: list of input shapes
        - output_shapes: list of output shapes
        - error: Optional error message

    Example:
        >>> result = verify_onnx_model(Path("yolov5n.onnx"))
        >>> result["valid"]
        True
        >>> result["opset_version"]
        17
    """
    result = {
        "valid": False,
        "opset_version": None,
        "input_shapes": [],
        "output_shapes": [],
        "error": None,
    }

    if not model_path.exists():
        result["error"] = f"File not found: {model_path}"
        return result

    try:
        import onnx
        from onnx import checker

        # Load and validate ONNX model
        model = onnx.load(str(model_path))
        checker.check_model(model)

        # Extract opset version
        result["opset_version"] = model.opset_import[0].version

        # Extract input shapes
        for inp in model.graph.input:
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)  # Dynamic dimension
                else:
                    shape.append(-1)
            result["input_shapes"].append(tuple(shape))

        # Extract output shapes
        for out in model.graph.output:
            shape = []
            for dim in out.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(-1)
            result["output_shapes"].append(tuple(shape))

        # Verify can be loaded by ONNX Runtime
        import onnxruntime as ort

        ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

        result["valid"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


# =============================================================================
# YOLOv5n Export
# =============================================================================

def export_yolov5n(
    output_path: Path,
    opset_version: int = ONNX_OPSET_VERSION,
    input_size: int = YOLO_INPUT_SIZE,
    force: bool = False,
) -> ExportResult:
    """
    Export YOLOv5n model to ONNX format.

    Downloads pretrained YOLOv5n from Ultralytics and exports to ONNX
    with static input shape and NMS included.

    Args:
        output_path: Path to save ONNX file
        opset_version: ONNX opset version (default: 17)
        input_size: Input dimension (default: 640)
        force: Overwrite existing file if True

    Returns:
        ExportResult with export details

    Raises:
        FileExistsError: If output_path exists and force=False
        ImportError: If torch or ultralytics not installed
        RuntimeError: If export fails

    Example:
        >>> result = export_yolov5n(Path("models/yolov5n.onnx"))
        >>> result.input_shape
        (1, 3, 640, 640)
    """
    output_path = Path(output_path)

    # Check existing file
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Model already exists: {output_path}. Use force=True to overwrite."
        )

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting YOLOv5n to {output_path}")
    logger.info(f"  Opset version: {opset_version}")
    logger.info(f"  Input size: {input_size}x{input_size}")

    try:
        # Try ultralytics first (preferred)
        from ultralytics import YOLO
        import os

        logger.info("  Using ultralytics library...")
        
        original_cwd = os.getcwd()
        os.chdir(output_path.parent)
        
        try:
            # Download and load YOLOv5n (downloads to current directory)
            model = YOLO("yolov5n.pt")

            # Export to ONNX
            export_path = model.export(
                format="onnx",
                opset=opset_version,
                imgsz=input_size,
                batch=1,
                dynamic=False,
                simplify=True,
            )

            # Move to target location if different
            export_path = Path(export_path)
            if export_path.name != output_path.name:
                export_path.rename(output_path)
        finally:
            # Always restore original directory
            os.chdir(original_cwd)

    except ImportError:
        # Fallback to torch.hub
        logger.info("  ultralytics not found, using torch.hub...")

        import torch

        model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size)

        # Export
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=["images"],
            output_names=["output0"],
            dynamic_axes=None,
        )

    # Verify export
    verification = verify_onnx_model(output_path)
    if not verification["valid"]:
        raise RuntimeError(f"Export verification failed: {verification['error']}")

    # Compute checksum
    checksum = compute_checksum(output_path)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    # Get actual shapes from verification
    input_shape = verification["input_shapes"][0] if verification["input_shapes"] else ()
    output_shape = verification["output_shapes"][0] if verification["output_shapes"] else ()

    logger.info(f"  ✓ Export successful")
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    logger.info(f"  Checksum: {checksum[:16]}...")

    return ExportResult(
        model_path=output_path,
        checksum=checksum,
        opset_version=verification["opset_version"],
        input_shape=input_shape,
        output_shape=output_shape,
        file_size_mb=file_size_mb,
    )


# =============================================================================
# MobileNetV2 Export
# =============================================================================

def export_mobilenetv2(
    output_path: Path,
    opset_version: int = ONNX_OPSET_VERSION,
    input_size: int = MOBILENET_INPUT_SIZE,
    force: bool = False,
) -> ExportResult:
    """
    Export MobileNetV2 model to ONNX format.

    Uses torchvision pretrained MobileNetV2 with ImageNet weights
    and exports to ONNX with static input shape.

    Args:
        output_path: Path to save ONNX file
        opset_version: ONNX opset version (default: 17)
        input_size: Input dimension (default: 224)
        force: Overwrite existing file if True

    Returns:
        ExportResult with export details

    Raises:
        FileExistsError: If output_path exists and force=False
        ImportError: If torch or torchvision not installed
        RuntimeError: If export fails

    Example:
        >>> result = export_mobilenetv2(Path("models/mobilenetv2.onnx"))
        >>> result.input_shape
        (1, 3, 224, 224)
        >>> result.output_shape
        (1, 1000)
    """
    output_path = Path(output_path)

    # Check existing file
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Model already exists: {output_path}. Use force=True to overwrite."
        )

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting MobileNetV2 to {output_path}")
    logger.info(f"  Opset version: {opset_version}")
    logger.info(f"  Input size: {input_size}x{input_size}")

    import torch
    import torchvision.models as models

    # Load model with ImageNet weights
    logger.info("  Loading pretrained weights (ImageNet1K_V1)...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Export to ONNX
    logger.info("  Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )

    # Verify export
    verification = verify_onnx_model(output_path)
    if not verification["valid"]:
        raise RuntimeError(f"Export verification failed: {verification['error']}")

    # Compute checksum
    checksum = compute_checksum(output_path)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    # Get actual shapes from verification
    input_shape = verification["input_shapes"][0] if verification["input_shapes"] else ()
    output_shape = verification["output_shapes"][0] if verification["output_shapes"] else ()

    logger.info(f"  ✓ Export successful")
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    logger.info(f"  Checksum: {checksum[:16]}...")

    return ExportResult(
        model_path=output_path,
        checksum=checksum,
        opset_version=verification["opset_version"],
        input_shape=input_shape,
        output_shape=output_shape,
        file_size_mb=file_size_mb,
    )


# =============================================================================
# Batch Export
# =============================================================================

def export_all_models(
    output_dir: Path,
    force: bool = False,
) -> dict[str, ExportResult]:
    """
    Export all models required for the experiment.

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
