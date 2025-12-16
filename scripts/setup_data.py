"""
Setup Data Script - Download COCO and curate thesis dataset.

This script is a thin CLI wrapper around shared.data module.
It is idempotent: existing data is skipped unless --force is used.

Usage:
    python scripts/setup_data.py                  # Full setup (download + curate)
    python scripts/setup_data.py --download-only  # Download COCO only
    python scripts/setup_data.py --curate-only    # Curate only (requires COCO)
    python scripts/setup_data.py --force          # Re-run even if exists
    python scripts/setup_data.py --verify         # Verify existing data

Author: Matthew Hong
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shared.data.coco_dataset import (
    download_coco_val2017,
    is_coco_downloaded,
    get_coco_image_paths,
    COCO_VAL2017_COUNT,
)
from shared.data.curator import (
    DatasetCurator,
    CurationConfig,
    DatasetManifest,
    TARGET_MEAN_DETECTIONS,
    TARGET_STD_DETECTIONS,
)


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = DATA_DIR / "thesis_test_set"

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
    print("Inference Arena - Data Setup")
    print("=" * 60)
    print(f"  Data directory:   {DATA_DIR}")
    print(f"  Models directory: {MODELS_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print()


def verify_data() -> bool:
    """Verify existing data setup."""
    print("Verifying data setup...")
    all_valid = True

    # Check COCO
    ready, msg = is_coco_downloaded(DATA_DIR)
    if ready:
        print(f"  ✓ COCO val2017: {msg}")
    else:
        print(f"  ✗ COCO val2017: {msg}")
        all_valid = False

    # Check thesis dataset
    manifest_path = OUTPUT_DIR / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = DatasetManifest.load(manifest_path)
            stats = manifest.statistics
            total = stats.get("total_images", 0)
            mean = stats.get("mean_detections", 0)
            std = stats.get("std_detections", 0)

            # Verify images exist
            jpg_count = len(list(OUTPUT_DIR.glob("*.jpg")))

            if jpg_count >= total:
                print(f"  ✓ Thesis dataset: {total} images (μ={mean:.2f}, σ={std:.2f})")
                print(f"      Distribution: {manifest.distribution}")
            else:
                print(f"  ✗ Thesis dataset: Missing images ({jpg_count}/{total})")
                all_valid = False

        except Exception as e:
            print(f"  ✗ Thesis dataset: Invalid manifest ({e})")
            all_valid = False
    else:
        print(f"  ✗ Thesis dataset: Not found")
        all_valid = False

    # Check models (needed for curation)
    yolo_path = MODELS_DIR / "yolov5n.onnx"
    if yolo_path.exists():
        size_mb = yolo_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ YOLOv5n model: {size_mb:.2f} MB")
    else:
        print(f"  ⚠ YOLOv5n model: Not found (needed for curation)")

    return all_valid


def download_coco(force: bool = False) -> bool:
    """Download COCO val2017 dataset."""
    print("→ COCO val2017 Dataset")

    # Check if already downloaded
    ready, msg = is_coco_downloaded(DATA_DIR)
    if ready and not force:
        print(f"  ⊘ Already downloaded: {msg}")
        return True

    try:
        images_dir = download_coco_val2017(DATA_DIR, force=force)
        jpg_count = len(list(images_dir.glob("*.jpg")))
        print(f"  ✓ Downloaded {jpg_count} images to {images_dir}")
        return True

    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False


def curate_dataset(force: bool = False) -> bool:
    """Curate thesis test dataset."""
    print("\n→ Thesis Test Dataset")

    # Check if model exists
    yolo_path = MODELS_DIR / "yolov5n.onnx"
    if not yolo_path.exists():
        print(f"  ✗ YOLOv5n model not found at {yolo_path}")
        print(f"    Run 'make setup-models' or 'python scripts/export_models.py' first")
        return False

    # Check if COCO exists
    ready, msg = is_coco_downloaded(DATA_DIR)
    if not ready:
        print(f"  ✗ COCO not downloaded: {msg}")
        print(f"    Run with --download-only first or without --curate-only")
        return False

    config = CurationConfig(
        target_count=100,
        min_detections=3,
        max_detections=5,
        confidence_threshold=0.5,
        random_seed=42,
    )

    curator = DatasetCurator(
        data_dir=DATA_DIR,
        models_dir=MODELS_DIR,
        output_dir=OUTPUT_DIR,
        config=config,
    )

    # Check if already curated
    curated, msg = curator.is_curated()
    if curated and not force:
        print(f"  ⊘ Already curated: {msg}")
        return True

    try:
        print(f"  Curating {config.target_count} images with {config.min_detections}-{config.max_detections} detections...")
        print(f"  Target: μ={TARGET_MEAN_DETECTIONS}, σ≈{TARGET_STD_DETECTIONS}")
        print()

        result = curator.curate(force=force)

        # Load manifest for stats
        manifest = DatasetManifest.load(OUTPUT_DIR / "manifest.json")
        stats = manifest.statistics

        print()
        print(f"  ✓ Curated {result.total_selected} images")
        print(f"    Mean detections: {stats['mean_detections']:.2f}")
        print(f"    Std detections:  {stats['std_detections']:.2f}")
        print(f"    Distribution:    {manifest.distribution}")
        print(f"    Output:          {OUTPUT_DIR}")

        return True

    except Exception as e:
        print(f"  ✗ Curation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download COCO and curate thesis dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup_data.py                  # Full setup
  python scripts/setup_data.py --download-only  # Download COCO only
  python scripts/setup_data.py --curate-only    # Curate only
  python scripts/setup_data.py --verify         # Verify existing data
  python scripts/setup_data.py --force          # Re-run everything
        """,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download/re-curate even if exists",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing data without downloading",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download COCO only, skip curation",
    )
    parser.add_argument(
        "--curate-only",
        action="store_true",
        help="Curate only, skip download (requires existing COCO)",
    )
    global DATA_DIR, MODELS_DIR, OUTPUT_DIR

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Data directory (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help=f"Models directory (default: {MODELS_DIR})",
    )

    args = parser.parse_args()

    # Update paths if specified
    if args.data_dir is not None:
        DATA_DIR = args.data_dir
    if args.models_dir is not None:
        MODELS_DIR = args.models_dir
    OUTPUT_DIR = DATA_DIR / "thesis_test_set"

    print_header()

    if args.verify:
        success = verify_data()
    elif args.download_only:
        success = download_coco(force=args.force)
    elif args.curate_only:
        success = curate_dataset(force=args.force)
    else:
        # Full setup
        success = download_coco(force=args.force)
        if success:
            success = curate_dataset(force=args.force)

    print()
    print("=" * 60)
    if success:
        print("✓ Complete")
    else:
        print("✗ Some operations failed")
    print("=" * 60)
    print()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
