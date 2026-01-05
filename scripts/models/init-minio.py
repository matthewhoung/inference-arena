#!/usr/bin/env python3
"""Initialize MinIO with ONNX models for Triton Inference Server.

This script uploads ONNX models to MinIO with Triton-compatible structure:
    models/
    ├── yolov5n/
    │   ├── 1/
    │   │   └── model.onnx
    │   ├── config.pbtxt
    │   └── metadata.json
    └── mobilenetv2/
        ├── 1/
        │   └── model.onnx
        ├── config.pbtxt
        └── metadata.json

Prerequisites:
    1. MinIO infrastructure must be running:
       docker compose -f infrastructure/docker-compose.infra.yml up -d

    2. Models must be exported to ./models/ directory:
       python scripts/models/export.py

Usage:
    # Full setup (requires MinIO running)
    python scripts/models/init-minio.py

    # Verify existing setup
    python scripts/models/init-minio.py --verify

    # Force re-upload
    python scripts/models/init-minio.py --force

Author: Matthew Hong
Specification Reference: experiment.yaml, Ch3 Methodology §3.4.4
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shared.config import get_spec_version

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_header() -> None:
    """Print script header."""
    print()
    print("=" * 60)
    print("Inference Arena - MinIO Model Initialization")
    print("=" * 60)
    print(f"  Models directory: {PROJECT_ROOT / 'models'}")
    print(f"  Spec version:     {get_spec_version()}")
    print()


def print_verification(verification: Dict[str, Any]) -> None:
    """Print verification results."""
    print()
    print("Verification Results:")
    print("-" * 40)

    for model_name, status in verification["models"].items():
        status_icon = "[OK]" if status["valid"] else "[FAIL]"
        print(f"  {status_icon} {model_name}")

        for file_type, exists in status.items():
            if file_type == "valid":
                continue
            file_icon = "[OK]" if exists else "[FAIL]"
            print(f"      {file_icon} {file_type.replace('_', '.')}")

    print()
    if verification["all_valid"]:
        print("[OK] All models verified successfully")
    else:
        print("[FAIL] Verification failed - some models missing")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload ONNX models to MinIO with Triton-compatible structure"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing models without uploading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-upload even if files exist",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="MinIO endpoint (default: from experiment.yaml)",
    )

    args = parser.parse_args()

    print_header()

    # Import MinIO registry (may fail if minio not installed)
    try:
        from shared.triton.minio import MinIOModelRegistry, MINIO_AVAILABLE
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Install with: pip install minio")
        return 1

    if not MINIO_AVAILABLE:
        logger.error("minio package not installed")
        logger.error("Install with: pip install minio")
        return 1

    try:
        # Initialize registry
        registry = MinIOModelRegistry(endpoint=args.endpoint)

        # Wait for MinIO
        logger.info("Connecting to MinIO...")
        registry.wait_for_minio()

        if args.verify:
            # Verify only
            verification = registry.verify_models()
            print_verification(verification)
            return 0 if verification["all_valid"] else 1

        # Ensure bucket exists
        registry.ensure_bucket_exists()

        # Upload all models
        logger.info("Uploading models...")
        results = registry.upload_all_models(force=args.force)

        # Summary
        print()
        print("=" * 60)
        print("Upload Summary")
        print("=" * 60)

        total_uploaded = 0
        total_skipped = 0
        errors = []

        for result in results:
            if "error" in result:
                errors.append(result)
            else:
                total_uploaded += len(result.get("uploads", []))
                total_skipped += len(result.get("skipped", []))

        print(f"  Uploaded: {total_uploaded} files")
        print(f"  Skipped:  {total_skipped} files (already exist)")
        print(f"  Errors:   {len(errors)}")

        if errors:
            print()
            print("Errors:")
            for err in errors:
                print(f"  [FAIL] {err['model_name']}: {err['error']}")
            return 1

        # Verify
        print()
        logger.info("Verifying uploads...")
        verification = registry.verify_models()
        print_verification(verification)

        return 0 if verification["all_valid"] else 1

    except ConnectionError as e:
        logger.error(f"Connection failed: {e}")
        logger.error("Is MinIO running? Start with:")
        logger.error("  docker compose -f infrastructure/docker-compose.infra.yml up -d")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
