#!/usr/bin/env python3
"""
Export Models Script - Export YOLOv5n and MobileNetV2 to ONNX format.

This script is a thin CLI wrapper around shared.model.exporter.
It is idempotent: existing models are skipped unless --force is used.

Usage:
    python scripts/export_models.py                 # Export all models
    python scripts/export_models.py --force         # Re-export even if exists
    python scripts/export_models.py --yolo-only     # Export only YOLOv5n
    python scripts/export_models.py --mobilenet-only # Export only MobileNetV2
    python scripts/export_models.py --verify        # Verify existing models

Author: Matthew Hong
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shared.model.exporter import (
    export_yolov5n,
    export_mobilenetv2,
    export_all_models,
    verify_onnx_model,
    compute_checksum,
    ExportResult,
    ONNX_OPSET_VERSION,
)


# =============================================================================
# Configuration
# =============================================================================

MODELS_DIR = PROJECT_ROOT / "models"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CLI Functions
# =============================================================================

def print_header() -> None:
    """Print script header."""
    print()
    print("=" * 60)
    print("Inference Arena - Model Export")
    print("=" * 60)
    print(f"  Output directory: {MODELS_DIR}")
    print(f"  ONNX opset: {ONNX_OPSET_VERSION}")
    print()


def print_result(name: str, result: ExportResult) -> None:
    """Print export result summary."""
    print(f"\n  {name}:")
    print(f"    Path: {result.model_path}")
    print(f"    Size: {result.file_size_mb:.2f} MB")
    print(f"    Input: {result.input_shape}")
    print(f"    Output: {result.output_shape}")
    print(f"    Checksum: {result.checksum[:16]}...")


def verify_existing_models() -> bool:
    """Verify existing models in models directory."""
    print("\nVerifying existing models...")
    
    all_valid = True
    
    for model_name, filename in [("YOLOv5n", "yolov5n.onnx"), ("MobileNetV2", "mobilenetv2.onnx")]:
        model_path = MODELS_DIR / filename
        
        if not model_path.exists():
            print(f"  ✗ {model_name}: Not found")
            all_valid = False
            continue
        
        result = verify_onnx_model(model_path)
        
        if result["valid"]:
            checksum = compute_checksum(model_path)
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  {model_name}: Valid (opset {result['opset_version']}, {size_mb:.2f} MB)")
            print(f"      Input: {result['input_shapes']}")
            print(f"      Checksum: {checksum[:16]}...")
        else:
            print(f"  {model_name}: Invalid - {result['error']}")
            all_valid = False
    
    return all_valid


def export_models(
    force: bool = False,
    yolo_only: bool = False,
    mobilenet_only: bool = False,
) -> bool:
    """
    Export models to ONNX format.
    
    Args:
        force: Overwrite existing files
        yolo_only: Export only YOLOv5n
        mobilenet_only: Export only MobileNetV2
    
    Returns:
        True if all exports successful
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    success = True
    results = {}
    
    # Export YOLOv5n
    if not mobilenet_only:
        print("\n→ Exporting YOLOv5n...")
        try:
            yolo_path = MODELS_DIR / "yolov5n.onnx"
            
            if yolo_path.exists() and not force:
                print("  ⊘ Already exists (use --force to overwrite)")
                # Still verify it's valid
                verification = verify_onnx_model(yolo_path)
                if verification["valid"]:
                    results["yolov5n"] = ExportResult(
                        model_path=yolo_path,
                        checksum=compute_checksum(yolo_path),
                        opset_version=verification["opset_version"],
                        input_shape=verification["input_shapes"][0],
                        output_shape=verification["output_shapes"][0],
                        file_size_mb=yolo_path.stat().st_size / (1024 * 1024),
                    )
            else:
                results["yolov5n"] = export_yolov5n(yolo_path, force=force)
                print("Export successful")
                
        except ImportError as e:
            print(f"Missing dependency: {e}")
            print("    Install with: pip install torch ultralytics")
            success = False
        except Exception as e:
            print(f"Export failed: {e}")
            success = False
    
    # Export MobileNetV2
    if not yolo_only:
        print("\n→ Exporting MobileNetV2...")
        try:
            mobilenet_path = MODELS_DIR / "mobilenetv2.onnx"
            
            if mobilenet_path.exists() and not force:
                print("  ⊘ Already exists (use --force to overwrite)")
                # Still verify it's valid
                verification = verify_onnx_model(mobilenet_path)
                if verification["valid"]:
                    results["mobilenetv2"] = ExportResult(
                        model_path=mobilenet_path,
                        checksum=compute_checksum(mobilenet_path),
                        opset_version=verification["opset_version"],
                        input_shape=verification["input_shapes"][0],
                        output_shape=verification["output_shapes"][0],
                        file_size_mb=mobilenet_path.stat().st_size / (1024 * 1024),
                    )
            else:
                results["mobilenetv2"] = export_mobilenetv2(mobilenet_path, force=force)
                print("Export successful")
                
        except ImportError as e:
            print(f"Missing dependency: {e}")
            print("    Install with: pip install torch torchvision")
            success = False
        except Exception as e:
            print(f"Export failed: {e}")
            success = False
    
    # Print summary
    if results:
        print("\n" + "-" * 60)
        print("Export Summary:")
        for name, result in results.items():
            print_result(name, result)
    
    # Write checksums file
    if results:
        checksums_path = MODELS_DIR / "checksums.txt"
        with open(checksums_path, "w") as f:
            f.write("# Model checksums for reproducibility\n")
            f.write(f"# ONNX opset version: {ONNX_OPSET_VERSION}\n\n")
            for name, result in results.items():
                f.write(f"{result.model_path.name}: sha256:{result.checksum}\n")
        print(f"\n  Checksums saved to: {checksums_path}")
    
    return success


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    """Main entry point."""
    global MODELS_DIR

    parser = argparse.ArgumentParser(
        description="Export YOLOv5n and MobileNetV2 to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/export_models.py               # Export all models
  python scripts/export_models.py --force       # Re-export even if exists
  python scripts/export_models.py --verify      # Verify existing models only
  python scripts/export_models.py --yolo-only   # Export only YOLOv5n
        """,
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing model files",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing models without exporting",
    )
    parser.add_argument(
        "--yolo-only",
        action="store_true",
        help="Export only YOLOv5n",
    )
    parser.add_argument(
        "--mobilenet-only",
        action="store_true",
        help="Export only MobileNetV2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {MODELS_DIR})",
    )
    
    args = parser.parse_args()
    
    # Update output dir if specified
    if args.output_dir is not None:
        MODELS_DIR = args.output_dir
    
    print_header()
    
    if args.verify:
        success = verify_existing_models()
    else:
        success = export_models(
            force=args.force,
            yolo_only=args.yolo_only,
            mobilenet_only=args.mobilenet_only,
        )
    
    print()
    print("=" * 60)
    if success:
        print("Complete")
    else:
        print("Operations failed")
    print("=" * 60)
    print()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
