"""ONNX Model Exporter - Classification Models.

This module handles export of classification models (MobileNetV2) to ONNX format.

Author: Matthew Hong
Specification Reference: Foundation Specification Section 2 Model Export
"""

import logging
from pathlib import Path

from .types import MOBILENET_INPUT_SIZE, ONNX_OPSET_VERSION, ExportResult
from .utils import compute_checksum, verify_onnx_model

logger = logging.getLogger(__name__)


# =============================================================================
# MobileNetV2 Export
# =============================================================================


def export_mobilenetv2(
    output_path: Path,
    opset_version: int = ONNX_OPSET_VERSION,
    input_size: int = MOBILENET_INPUT_SIZE,
    force: bool = False,
    dynamic_batch: bool = False,
) -> ExportResult:
    """Export MobileNetV2 model to ONNX format.

    Uses torchvision pretrained MobileNetV2 with ImageNet weights
    and exports to ONNX with static input shape.

    Args:
        output_path: Path to save ONNX file
        opset_version: ONNX opset version (default: 17)
        input_size: Input dimension (default: 224)
        force: Overwrite existing file if True
        dynamic_batch: If True, export with dynamic batch dimension for Triton batching

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
        raise FileExistsError(f"Model already exists: {output_path}. Use force=True to overwrite.")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_mode = "dynamic" if dynamic_batch else "static"
    logger.info(f"Exporting MobileNetV2 to {output_path}")
    logger.info(f"  Opset version: {opset_version}")
    logger.info(f"  Input size: {input_size}x{input_size}")
    logger.info(f"  Batch mode: {batch_mode}")

    import torch
    import torchvision.models as models

    # Load model with ImageNet weights
    logger.info("  Loading pretrained weights (ImageNet1K_V1)...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Set dynamic axes for batch dimension if requested
    dynamic_axes = None
    export_params = True
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch"},
            "output": {0: "batch"},
        }

    # Export to ONNX
    # Use dynamo=False to ensure the legacy exporter is used, which properly
    # supports dynamic_axes. The newer torch.export pathway may not honor it.
    logger.info("  Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        export_params=export_params,
        do_constant_folding=True,
        dynamo=False,  # Force legacy exporter for proper dynamic_axes support
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

    return ExportResult(
        model_path=output_path,
        checksum=checksum,
        opset_version=verification["opset_version"],
        input_shape=input_shape,
        output_shape=output_shape,
        file_size_mb=file_size_mb,
    )
