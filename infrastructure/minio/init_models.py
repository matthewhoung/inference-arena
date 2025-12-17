#!/usr/bin/env python3
"""
MinIO Model Initialization Script

This script uploads ONNX models to MinIO with Triton-compatible structure.
Configuration is sourced from experiment.yaml for consistency.

Bucket Structure (Option C - Single Bucket with Metadata):
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

Usage:
    # Full setup (requires MinIO running)
    python infrastructure/minio/init_models.py
    
    # Verify existing setup
    python infrastructure/minio/init_models.py --verify
    
    # Force re-upload
    python infrastructure/minio/init_models.py --force

Author: Matthew Hong
Specification Reference: experiment.yaml, Ch3 Methodology §3.4.4
"""

import argparse
import hashlib
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import (
    get_config,
    get_model_config,
    get_model_names,
    get_minio_config,
    get_spec_version,
    get_metadata,
)

# Local imports
from infrastructure.minio.triton_config import generate_config_pbtxt

# Third-party imports (with graceful fallback)
try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_VERSION = 1  # Triton model version directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Retry Decorator
# =============================================================================

if TENACITY_AVAILABLE:
    retry_on_connection = retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, OSError)),
        before_sleep=lambda retry_state: logger.info(
            f"Retrying MinIO connection (attempt {retry_state.attempt_number}/10)..."
        ),
    )
else:
    # Fallback: no retry
    def retry_on_connection(func):
        return func


# =============================================================================
# MinIO Client
# =============================================================================

class MinIOModelRegistry:
    """
    MinIO client for model registry operations.
    
    Handles:
    - Bucket creation
    - Model upload with Triton structure
    - Metadata generation
    - Verification
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        secure: bool = False,
        bucket: Optional[str] = None,
    ):
        """
        Initialize MinIO client.
        
        Args:
            endpoint: MinIO endpoint (default: from experiment.yaml)
            access_key: Access key (default: from experiment.yaml)
            secret_key: Secret key (default: from experiment.yaml)
            secure: Use HTTPS (default: False)
            bucket: Bucket name (default: from experiment.yaml)
        """
        if not MINIO_AVAILABLE:
            raise ImportError(
                "minio package not installed. "
                "Install with: pip install minio"
            )
        
        # Load from experiment.yaml if not provided
        minio_config = get_minio_config()
        
        self.endpoint = endpoint or minio_config.get("external_endpoint", "localhost:9000")
        self.access_key = access_key or minio_config.get("access_key", "minioadmin")
        self.secret_key = secret_key or minio_config.get("secret_key", "minioadmin")
        self.secure = secure if secure is not None else minio_config.get("secure", False)
        self.bucket = bucket or minio_config.get("bucket", "models")
        
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )
    
    @retry_on_connection
    def wait_for_minio(self) -> bool:
        """
        Wait for MinIO to be ready.
        
        Returns:
            True if MinIO is ready
            
        Raises:
            ConnectionError: If MinIO not reachable after retries
        """
        try:
            self.client.list_buckets()
            logger.info(f"✓ MinIO connected at {self.endpoint}")
            return True
        except Exception as e:
            raise ConnectionError(f"Cannot connect to MinIO: {e}")
    
    def ensure_bucket_exists(self) -> bool:
        """
        Create bucket if it doesn't exist.
        
        Returns:
            True if bucket exists or was created
        """
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)
            logger.info(f"✓ Created bucket: {self.bucket}")
            return True
        
        logger.info(f"✓ Bucket exists: {self.bucket}")
        return True
    
    def upload_model(
        self,
        model_name: str,
        model_path: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Upload a model with Triton-compatible structure.
        
        Creates:
            {bucket}/{model_name}/1/model.onnx
            {bucket}/{model_name}/config.pbtxt
            {bucket}/{model_name}/metadata.json
        
        Args:
            model_name: Model identifier (e.g., "yolov5n")
            model_path: Path to local ONNX file
            force: Overwrite existing files
            
        Returns:
            Upload result with checksums and paths
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        result = {
            "model_name": model_name,
            "local_path": str(model_path),
            "uploads": [],
            "skipped": [],
        }
        
        # 1. Upload model.onnx to version directory
        model_object_name = f"{model_name}/{MODEL_VERSION}/model.onnx"
        if force or not self._object_exists(model_object_name):
            checksum = self._compute_checksum(model_path)
            self.client.fput_object(
                self.bucket,
                model_object_name,
                str(model_path),
                content_type="application/octet-stream",
            )
            result["uploads"].append(model_object_name)
            result["checksum"] = checksum
            logger.info(f"  ✓ Uploaded: {model_object_name}")
        else:
            result["skipped"].append(model_object_name)
            logger.info(f"  ○ Skipped (exists): {model_object_name}")
        
        # 2. Upload config.pbtxt
        config_object_name = f"{model_name}/config.pbtxt"
        if force or not self._object_exists(config_object_name):
            config_content = generate_config_pbtxt(model_name)
            self._upload_string(config_object_name, config_content, "text/plain")
            result["uploads"].append(config_object_name)
            logger.info(f"  ✓ Uploaded: {config_object_name}")
        else:
            result["skipped"].append(config_object_name)
            logger.info(f"  ○ Skipped (exists): {config_object_name}")
        
        # 3. Upload metadata.json
        metadata_object_name = f"{model_name}/metadata.json"
        if force or not self._object_exists(metadata_object_name):
            metadata = self._generate_metadata(model_name, model_path)
            metadata_json = json.dumps(metadata, indent=2)
            self._upload_string(metadata_object_name, metadata_json, "application/json")
            result["uploads"].append(metadata_object_name)
            result["metadata"] = metadata
            logger.info(f"  ✓ Uploaded: {metadata_object_name}")
        else:
            result["skipped"].append(metadata_object_name)
            logger.info(f"  ○ Skipped (exists): {metadata_object_name}")
        
        return result
    
    def upload_all_models(self, force: bool = False) -> List[Dict[str, Any]]:
        """
        Upload all models from experiment.yaml.
        
        Args:
            force: Overwrite existing files
            
        Returns:
            List of upload results
        """
        results = []
        
        for model_name in get_model_names():
            # Construct expected local path
            model_path = MODELS_DIR / f"{model_name}.onnx"
            
            if not model_path.exists():
                logger.warning(f"⚠ Model not found: {model_path}")
                logger.warning(f"  Run: python scripts/export_models.py")
                results.append({
                    "model_name": model_name,
                    "error": f"File not found: {model_path}",
                })
                continue
            
            logger.info(f"Uploading {model_name}...")
            result = self.upload_model(model_name, model_path, force=force)
            results.append(result)
        
        return results
    
    def verify_models(self) -> Dict[str, Any]:
        """
        Verify that all models are correctly uploaded.
        
        Returns:
            Verification result with status per model
        """
        verification = {
            "bucket": self.bucket,
            "models": {},
            "all_valid": True,
        }
        
        for model_name in get_model_names():
            model_status = {
                "model_onnx": False,
                "config_pbtxt": False,
                "metadata_json": False,
            }
            
            # Check each required file
            model_status["model_onnx"] = self._object_exists(
                f"{model_name}/{MODEL_VERSION}/model.onnx"
            )
            model_status["config_pbtxt"] = self._object_exists(
                f"{model_name}/config.pbtxt"
            )
            model_status["metadata_json"] = self._object_exists(
                f"{model_name}/metadata.json"
            )
            
            model_status["valid"] = all(model_status.values())
            verification["models"][model_name] = model_status
            
            if not model_status["valid"]:
                verification["all_valid"] = False
        
        return verification
    
    def _object_exists(self, object_name: str) -> bool:
        """Check if object exists in bucket."""
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error:
            return False
    
    def _upload_string(
        self,
        object_name: str,
        content: str,
        content_type: str = "text/plain",
    ) -> None:
        """Upload string content to MinIO."""
        data = content.encode("utf-8")
        self.client.put_object(
            self.bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
    
    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _generate_metadata(self, model_name: str, model_path: Path) -> Dict[str, Any]:
        """Generate metadata.json content for a model."""
        model_config = get_model_config(model_name)
        experiment_meta = get_metadata()
        
        return {
            "model_name": model_name,
            "version": MODEL_VERSION,
            "format": model_config.get("format", "onnx"),
            "opset_version": model_config.get("opset_version"),
            "task": model_config.get("task"),
            "input": {
                "name": model_config["input"]["name"],
                "shape": model_config["input"]["shape"],
                "dtype": model_config["input"].get("dtype", "float32"),
            },
            "output": {
                "name": model_config["output"]["name"],
                "shape": model_config["output"]["shape"],
                "dtype": model_config["output"].get("dtype", "float32"),
            },
            "source": model_config.get("source"),
            "checksum_sha256": self._compute_checksum(model_path),
            "file_size_bytes": model_path.stat().st_size,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "experiment_spec_version": get_spec_version(),
            "thesis_reference": f"experiment.yaml controlled_variables.models.{model_name}",
            "author": experiment_meta.get("author", "Unknown"),
        }


# =============================================================================
# CLI Functions
# =============================================================================

def print_header() -> None:
    """Print script header."""
    print()
    print("=" * 60)
    print("Inference Arena - MinIO Model Initialization")
    print("=" * 60)
    print(f"  Models directory: {MODELS_DIR}")
    print(f"  Spec version:     {get_spec_version()}")
    print()


def print_verification(verification: Dict[str, Any]) -> None:
    """Print verification results."""
    print()
    print("Verification Results:")
    print("-" * 40)
    
    for model_name, status in verification["models"].items():
        status_icon = "✓" if status["valid"] else "✗"
        print(f"  {status_icon} {model_name}")
        
        for file_type, exists in status.items():
            if file_type == "valid":
                continue
            file_icon = "✓" if exists else "✗"
            print(f"      {file_icon} {file_type.replace('_', '.')}")
    
    print()
    if verification["all_valid"]:
        print("✓ All models verified successfully")
    else:
        print("✗ Verification failed - some models missing")


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
    
    # Check dependencies
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
                print(f"  ✗ {err['model_name']}: {err['error']}")
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
