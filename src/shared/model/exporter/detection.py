"""ONNX Model Exporter - Detection Models.

This module handles export of object detection models (YOLOv5n) to ONNX format.

Author: Matthew Hong
Specification Reference: Foundation Specification Section 2 Model Export
"""

import logging
from pathlib import Path

from .types import ONNX_OPSET_VERSION, YOLO_INPUT_SIZE, ExportResult
from .utils import compute_checksum, verify_onnx_model

logger = logging.getLogger(__name__)


# =============================================================================
# YOLOv5n Export
# =============================================================================


def export_yolov5n(
    output_path: Path,
    opset_version: int = ONNX_OPSET_VERSION,
    input_size: int = YOLO_INPUT_SIZE,
    force: bool = False,
    dynamic_batch: bool = False,
) -> ExportResult:
    """Export YOLOv5n model to ONNX format.

    Downloads pretrained YOLOv5n from Ultralytics and exports to ONNX
    with static input shape and NMS included.

    Args:
        output_path: Path to save ONNX file
        opset_version: ONNX opset version (default: 17)
        input_size: Input dimension (default: 640)
        force: Overwrite existing file if True
        dynamic_batch: If True, export with dynamic batch dimension for Triton batching

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
        raise FileExistsError(f"Model already exists: {output_path}. Use force=True to overwrite.")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_mode = "dynamic" if dynamic_batch else "static"
    logger.info(f"Exporting YOLOv5n to {output_path}")
    logger.info(f"  Opset version: {opset_version}")
    logger.info(f"  Input size: {input_size}x{input_size}")
    logger.info(f"  Batch mode: {batch_mode}")

    try:
        # Try ultralytics first (preferred)
        import os

        from ultralytics import YOLO

        logger.info("  Using ultralytics library...")

        original_cwd = os.getcwd()
        os.chdir(output_path.parent)

        try:
            # Download and load YOLOv5n (downloads to current directory)
            model = YOLO("yolov5n.pt")

            # Export to ONNX
            # dynamic=True enables dynamic batch dimension for Triton batching
            export_result = model.export(
                format="onnx",
                opset=opset_version,
                imgsz=input_size,
                batch=1,
                dynamic=dynamic_batch,
                simplify=True,
            )

            # Move to target location if different
            exported_path = Path(str(export_result))
            if exported_path.name != output_path.name:
                exported_path.rename(output_path)
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

        # Set dynamic axes for batch dimension if requested
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                "images": {0: "batch"},
                "output0": {0: "batch"},
            }

        # Export
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=["images"],
            output_names=["output0"],
            dynamic_axes=dynamic_axes,
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

    logger.info("  Export successful")
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    logger.info(f"  Checksum: {checksum[:16]}...")

    # opset_version is guaranteed to be set when valid=True
    assert verification["opset_version"] is not None

    return ExportResult(
        model_path=output_path,
        checksum=checksum,
        opset_version=verification["opset_version"],
        input_shape=input_shape,
        output_shape=output_shape,
        file_size_mb=file_size_mb,
    )
