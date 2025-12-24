#!/usr/bin/env python3
"""Upload ONNX models to MinIO for inference services.

This script uploads local ONNX models to MinIO with the directory structure
expected by the inference services:
- yolov5n/1/model.onnx
- mobilenetv2/1/model.onnx
- mobilenetv2/1/model.onnx.data

Prerequisites:
1. MinIO infrastructure must be running:
   docker compose -f infrastructure/docker-compose.infra.yml up -d

2. Models must exist in ./models/ directory

Usage:
    python scripts/upload_models_to_minio.py

Environment Variables:
    MINIO_ENDPOINT: MinIO endpoint (default: localhost:9000)
    MINIO_ACCESS_KEY: Access key (default: minioadmin)
    MINIO_SECRET_KEY: Secret key (default: minioadmin)
    MINIO_BUCKET: Bucket name (default: models)

Author: Matthew Hong
"""

import os
import sys
from pathlib import Path

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    print("Error: minio package not installed")
    print("Install with: pip install minio")
    sys.exit(1)


def get_env(key: str, default: str) -> str:
    """Get environment variable with default."""
    return os.getenv(key, default)


def upload_models():
    """Upload models to MinIO."""
    # Configuration
    endpoint = get_env("MINIO_ENDPOINT", "localhost:9000")
    access_key = get_env("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = get_env("MINIO_SECRET_KEY", "minioadmin")
    bucket_name = get_env("MINIO_BUCKET", "models")

    print("=" * 80)
    print("MinIO Model Upload")
    print("=" * 80)
    print(f"Endpoint:     {endpoint}")
    print(f"Bucket:       {bucket_name}")
    print(f"Access Key:   {access_key}")
    print("=" * 80)

    # Initialize MinIO client
    try:
        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,  # Use HTTP for local development
        )
        print(f"\n✓ Connected to MinIO at {endpoint}")
    except Exception as e:
        print(f"\n✗ Failed to connect to MinIO: {e}")
        print("\nMake sure infrastructure is running:")
        print("  docker compose -f infrastructure/docker-compose.infra.yml up -d")
        sys.exit(1)

    # Create bucket if it doesn't exist
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"✓ Created bucket: {bucket_name}")
        else:
            print(f"✓ Bucket exists: {bucket_name}")
    except S3Error as e:
        print(f"\n✗ Failed to create/check bucket: {e}")
        sys.exit(1)

    # Models to upload
    models_dir = Path("models")
    if not models_dir.exists():
        print(f"\n✗ Models directory not found: {models_dir}")
        sys.exit(1)

    uploads = [
        {
            "local": models_dir / "yolov5n.onnx",
            "remote": "yolov5n/1/model.onnx",
            "description": "YOLOv5n object detection model",
        },
        {
            "local": models_dir / "mobilenetv2.onnx",
            "remote": "mobilenetv2/1/model.onnx",
            "description": "MobileNetV2 classification model",
        },
        {
            "local": models_dir / "mobilenetv2.onnx.data",
            "remote": "mobilenetv2/1/model.onnx.data",
            "description": "MobileNetV2 external data file",
        },
    ]

    print("\n" + "=" * 80)
    print("Uploading Models")
    print("=" * 80)

    success_count = 0
    for upload in uploads:
        local_path = upload["local"]
        remote_path = upload["remote"]
        description = upload["description"]

        if not local_path.exists():
            print(f"\n✗ {description}")
            print(f"  Local file not found: {local_path}")
            continue

        # Get file size
        size_mb = local_path.stat().st_size / (1024 * 1024)

        print(f"\n→ {description}")
        print(f"  Local:  {local_path}")
        print(f"  Remote: {bucket_name}/{remote_path}")
        print(f"  Size:   {size_mb:.2f} MB")

        try:
            # Upload file
            client.fput_object(
                bucket_name,
                remote_path,
                str(local_path),
            )
            print(f"  ✓ Upload successful")
            success_count += 1
        except S3Error as e:
            print(f"  ✗ Upload failed: {e}")

    print("\n" + "=" * 80)
    print(f"Upload Summary: {success_count}/{len(uploads)} successful")
    print("=" * 80)

    if success_count == len(uploads):
        print("\n✓ All models uploaded successfully!")
        print("\nYou can now start the monolithic service:")
        print("  docker compose -f architectures/monolithic/docker-compose.yml up -d")
        return 0
    else:
        print("\n✗ Some uploads failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(upload_models())
