"""MinIO Model Registry for Triton Inference Server.

This module provides the MinIOModelRegistry class for uploading ONNX models
to MinIO with the directory structure expected by Triton Inference Server.

Bucket Structure:
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
    from shared.triton.minio import MinIOModelRegistry

    registry = MinIOModelRegistry()
    registry.wait_for_minio()
    registry.ensure_bucket_exists()
    registry.upload_all_models()

Author: Matthew Hong
Specification Reference: experiment.yaml, Ch3 Methodology §3.4.4
"""

import hashlib
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.config import (
    get_metadata,
    get_minio_config,
    get_model_config,
    get_model_names,
    get_spec_version,
)
from shared.triton.config import generate_config_pbtxt

# Third-party imports (with graceful fallback)
try:
    from minio import Minio
    from minio.error import S3Error

    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    Minio = None
    S3Error = Exception

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Constants
MODEL_VERSION = 1  # Triton model version directory


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
# MinIO Model Registry
# =============================================================================


class MinIOModelRegistry:
    """MinIO client for model registry operations.

    Handles:
    - Bucket creation
    - Model upload with Triton structure
    - Metadata generation
    - Verification

    Attributes:
        endpoint: MinIO endpoint (host:port)
        bucket: Bucket name for models
        client: MinIO client instance
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        secure: bool = False,
        bucket: Optional[str] = None,
        models_dir: Optional[Path] = None,
    ):
        """Initialize MinIO client.

        Args:
            endpoint: MinIO endpoint (default: from experiment.yaml)
            access_key: Access key (default: from experiment.yaml)
            secret_key: Secret key (default: from experiment.yaml)
            secure: Use HTTPS (default: False)
            bucket: Bucket name (default: from experiment.yaml)
            models_dir: Local models directory (default: PROJECT_ROOT/models)
        """
        if not MINIO_AVAILABLE:
            raise ImportError(
                "minio package not installed. " "Install with: pip install minio"
            )

        # Load from experiment.yaml if not provided
        minio_config = get_minio_config()

        self.endpoint = endpoint or minio_config.get(
            "external_endpoint", "localhost:9000"
        )
        self.access_key = access_key or minio_config.get("access_key", "minioadmin")
        self.secret_key = secret_key or minio_config.get("secret_key", "minioadmin")
        self.secure = secure if secure is not None else minio_config.get("secure", False)
        self.bucket = bucket or minio_config.get("bucket", "models")
        self.models_dir = models_dir or Path(__file__).parent.parent.parent.parent / "models"

        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )

    @retry_on_connection
    def wait_for_minio(self) -> bool:
        """Wait for MinIO to be ready.

        Returns:
            True if MinIO is ready

        Raises:
            ConnectionError: If MinIO not reachable after retries
        """
        try:
            self.client.list_buckets()
            logger.info(f"Connected to MinIO at {self.endpoint}")
            return True
        except Exception as e:
            raise ConnectionError(f"Cannot connect to MinIO: {e}")

    def ensure_bucket_exists(self) -> bool:
        """Create bucket if it doesn't exist.

        Returns:
            True if bucket exists or was created
        """
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)
            logger.info(f"Created bucket: {self.bucket}")
            return True

        logger.info(f"Bucket exists: {self.bucket}")
        return True

    def upload_model(
        self,
        model_name: str,
        model_path: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Upload a model with Triton-compatible structure.

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
            logger.info(f"  Uploaded: {model_object_name}")
        else:
            result["skipped"].append(model_object_name)
            logger.info(f"  Skipped (exists): {model_object_name}")

        # 1b. Upload model.onnx.data if it exists (external data format)
        data_path = model_path.with_suffix(".onnx.data")
        if data_path.exists():
            data_object_name = f"{model_name}/{MODEL_VERSION}/model.onnx.data"
            if force or not self._object_exists(data_object_name):
                self.client.fput_object(
                    self.bucket,
                    data_object_name,
                    str(data_path),
                    content_type="application/octet-stream",
                )
                result["uploads"].append(data_object_name)
                logger.info(f"  Uploaded: {data_object_name}")
            else:
                result["skipped"].append(data_object_name)
                logger.info(f"  Skipped (exists): {data_object_name}")

        # 2. Upload config.pbtxt
        config_object_name = f"{model_name}/config.pbtxt"
        if force or not self._object_exists(config_object_name):
            config_content = generate_config_pbtxt(model_name)
            self._upload_string(config_object_name, config_content, "text/plain")
            result["uploads"].append(config_object_name)
            logger.info(f"  Uploaded: {config_object_name}")
        else:
            result["skipped"].append(config_object_name)
            logger.info(f"  Skipped (exists): {config_object_name}")

        # 3. Upload metadata.json
        metadata_object_name = f"{model_name}/metadata.json"
        if force or not self._object_exists(metadata_object_name):
            metadata = self._generate_metadata(model_name, model_path)
            metadata_json = json.dumps(metadata, indent=2)
            self._upload_string(metadata_object_name, metadata_json, "application/json")
            result["uploads"].append(metadata_object_name)
            result["metadata"] = metadata
            logger.info(f"  Uploaded: {metadata_object_name}")
        else:
            result["skipped"].append(metadata_object_name)
            logger.info(f"  Skipped (exists): {metadata_object_name}")

        return result

    def upload_all_models(self, force: bool = False) -> List[Dict[str, Any]]:
        """Upload all models from experiment.yaml.

        Args:
            force: Overwrite existing files

        Returns:
            List of upload results
        """
        results = []

        for model_name in get_model_names():
            # Construct expected local path
            model_path = self.models_dir / f"{model_name}.onnx"

            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                logger.warning("  Run: python scripts/models/export.py")
                results.append(
                    {
                        "model_name": model_name,
                        "error": f"File not found: {model_path}",
                    }
                )
                continue

            logger.info(f"Uploading {model_name}...")
            result = self.upload_model(model_name, model_path, force=force)
            results.append(result)

        return results

    def verify_models(self) -> Dict[str, Any]:
        """Verify that all models are correctly uploaded.

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

            model_status["valid"] = all(
                v for k, v in model_status.items() if k != "valid"
            )
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
