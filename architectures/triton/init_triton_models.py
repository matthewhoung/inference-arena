#!/usr/bin/env python3
"""
Download models and config.pbtxt from MinIO to Triton model repository.

This script runs as an init container before Triton server starts.
Downloads all required model files from MinIO to the shared volume.

Author: Matthew Hong
"""

import os
import sys
from pathlib import Path
from minio import Minio
from minio.error import S3Error

# Configuration from environment variables
MINIO_ENDPOINT = os.getenv("MINIO_INTERNAL_ENDPOINT", "minio:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = os.getenv("MINIO_BUCKET", "models")
MODEL_REPO = Path("/models")


def download_model(client: Minio, model_name: str) -> None:
    """Download model files and config.pbtxt from MinIO.

    Args:
        client: MinIO client instance
        model_name: Name of the model (e.g., "yolov5n", "mobilenetv2")
    """
    print(f"\n{'='*60}")
    print(f"Downloading {model_name}")
    print('='*60)

    # Create model directory structure
    model_dir = MODEL_REPO / model_name / "1"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Download model.onnx
    model_onnx = f"{model_name}/1/model.onnx"
    local_path = model_dir / "model.onnx"
    print(f"  ⬇ Downloading {model_onnx}...")
    try:
        client.fget_object(BUCKET, model_onnx, str(local_path))
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Downloaded {model_onnx} ({size_mb:.2f} MB)")
    except S3Error as e:
        print(f"  ✗ Error downloading {model_onnx}: {e}")
        raise

    # Download model.onnx.data if exists (for MobileNetV2)
    model_data = f"{model_name}/1/model.onnx.data"
    local_data_path = model_dir / "model.onnx.data"
    try:
        print(f"  ⬇ Downloading {model_data}...")
        client.fget_object(BUCKET, model_data, str(local_data_path))
        size_mb = local_data_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Downloaded {model_data} ({size_mb:.2f} MB)")

        # Create symlink with model-specific name for ONNX internal references
        # E.g., mobilenetv2.onnx.data -> model.onnx.data
        symlink_path = model_dir / f"{model_name}.onnx.data"
        if not symlink_path.exists():
            symlink_path.symlink_to("model.onnx.data")
            print(f"  ✓ Created symlink: {model_name}.onnx.data -> model.onnx.data")
    except S3Error:
        # Not all models have external data - this is OK
        print(f"  ℹ No external data file for {model_name}")

    # Download config.pbtxt
    config_pbtxt = f"{model_name}/config.pbtxt"
    local_config_path = MODEL_REPO / model_name / "config.pbtxt"
    print(f"  ⬇ Downloading {config_pbtxt}...")
    try:
        client.fget_object(BUCKET, config_pbtxt, str(local_config_path))
        print(f"  ✓ Downloaded {config_pbtxt}")
    except S3Error as e:
        print(f"  ✗ Error downloading {config_pbtxt}: {e}")
        raise

    print(f"  ✓ {model_name} complete")


def main() -> int:
    """Main entry point."""
    print("\n" + "="*60)
    print("Triton Model Repository Initialization")
    print("="*60)
    print(f"MinIO Endpoint: {MINIO_ENDPOINT}")
    print(f"Bucket: {BUCKET}")
    print(f"Target Directory: {MODEL_REPO}")
    print("="*60)

    # Connect to MinIO
    print(f"\n⬇ Connecting to MinIO at {MINIO_ENDPOINT}...")
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            secure=False,
        )
        # Test connection
        if not client.bucket_exists(BUCKET):
            print(f"✗ Bucket '{BUCKET}' does not exist!")
            return 1
        print(f"✓ Connected to MinIO")
    except Exception as e:
        print(f"✗ Failed to connect to MinIO: {e}")
        return 1

    # Download models
    models = ["yolov5n", "mobilenetv2"]
    for model_name in models:
        try:
            download_model(client, model_name)
        except Exception as e:
            print(f"\n✗ Failed to download {model_name}: {e}")
            return 1

    # Success summary
    print("\n" + "="*60)
    print("✓ All models downloaded successfully")
    print("="*60)
    print("\nModel Repository Structure:")
    for model_name in models:
        print(f"\n{model_name}/")
        model_path = MODEL_REPO / model_name
        if model_path.exists():
            for item in sorted(model_path.rglob("*")):
                if item.is_file():
                    rel_path = item.relative_to(MODEL_REPO)
                    size_mb = item.stat().st_size / (1024 * 1024)
                    print(f"  {rel_path} ({size_mb:.2f} MB)")

    print("\n" + "="*60)
    print("Init container completed successfully")
    print("Triton server can now start")
    print("="*60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
