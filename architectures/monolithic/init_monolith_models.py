#!/usr/bin/env python3
"""
Download models from MinIO to monolithic model repository.

This script runs as an init container before the monolithic service starts.
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
MODEL_DIR = Path("/app/models")


def download_model(client: Minio, model_name: str, remote_name: str = None) -> None:
    """Download model files from MinIO.

    Args:
        client: MinIO client instance
        model_name: Name of the model (e.g., "yolov5n", "mobilenetv2")
        remote_name: Remote file name override (defaults to model_name)
    """
    if remote_name is None:
        remote_name = model_name

    print(f"\n{'='*60}")
    print(f"Downloading {model_name}")
    print('='*60)

    # Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Download model.onnx
    remote_onnx = f"{remote_name}/1/model.onnx"
    local_onnx = MODEL_DIR / f"{model_name}.onnx"
    print(f"  ⬇ Downloading {remote_onnx}...")
    try:
        client.fget_object(BUCKET, remote_onnx, str(local_onnx))
        size_mb = local_onnx.stat().st_size / (1024 * 1024)
        print(f"  ✓ Downloaded {model_name}.onnx ({size_mb:.2f} MB)")
    except S3Error as e:
        print(f"  ✗ Error downloading {remote_onnx}: {e}")
        raise

    # Download model.onnx.data if exists (for MobileNetV2)
    remote_data = f"{remote_name}/1/model.onnx.data"
    local_data = MODEL_DIR / f"{model_name}.onnx.data"
    try:
        print(f"  ⬇ Downloading {remote_data}...")
        client.fget_object(BUCKET, remote_data, str(local_data))
        size_mb = local_data.stat().st_size / (1024 * 1024)
        print(f"  ✓ Downloaded {model_name}.onnx.data ({size_mb:.2f} MB)")
    except S3Error:
        # Not all models have external data - this is OK
        print(f"  ℹ No external data file for {model_name}")

    print(f"  ✓ {model_name} complete")


def main() -> int:
    """Main entry point."""
    print("\n" + "="*60)
    print("Monolithic Model Repository Initialization")
    print("="*60)
    print(f"MinIO Endpoint: {MINIO_ENDPOINT}")
    print(f"Bucket: {BUCKET}")
    print(f"Target Directory: {MODEL_DIR}")
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
    print("\nModel Directory Structure:")
    if MODEL_DIR.exists():
        for item in sorted(MODEL_DIR.glob("*")):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item.name} ({size_mb:.2f} MB)")

    print("\n" + "="*60)
    print("Init container completed successfully")
    print("Monolithic service can now start")
    print("="*60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
