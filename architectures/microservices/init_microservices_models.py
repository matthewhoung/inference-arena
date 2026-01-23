#!/usr/bin/env python3
"""
Download models from MinIO to microservices model repository.

This script runs as an init container before the microservices start.
Downloads all required model files from MinIO to the shared volume.

Features:
- Parallel downloads using ThreadPoolExecutor
- Configurable concurrency via experiment.yaml (downloads.max_concurrent)
- Progress bars showing per-file download progress
- Fail-fast: first error cancels remaining downloads
- Cached file skip: existing files with matching size are skipped
- Atomic writes: downloads to .tmp file, renames on success

Author: Matthew Hong
"""

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

from minio import Minio
from minio.error import S3Error
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
MINIO_ENDPOINT = os.getenv("MINIO_INTERNAL_ENDPOINT", "minio:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = os.getenv("MINIO_BUCKET", "models")
MODEL_DIR = Path("/app/models")

# Default download settings (used when experiment.yaml not available)
DEFAULT_MAX_CONCURRENT = 3
DEFAULT_TIMEOUT = 300


class DownloadTask(NamedTuple):
    """Represents a single file download task."""

    remote_path: str
    local_path: Path
    model_name: str


def _get_download_settings() -> tuple[int, int]:
    """Get download settings from experiment.yaml or use defaults.

    Returns:
        Tuple of (max_concurrent, timeout)
    """
    try:
        # Import here to avoid dependency issues in init container
        # If shared.config is not available, use defaults
        from shared.config import get_download_max_concurrent, get_download_timeout

        return get_download_max_concurrent(), get_download_timeout()
    except ImportError:
        logger.debug("shared.config not available, using default download settings")
        return DEFAULT_MAX_CONCURRENT, DEFAULT_TIMEOUT
    except Exception as e:
        logger.debug(f"Failed to load config: {e}, using defaults")
        return DEFAULT_MAX_CONCURRENT, DEFAULT_TIMEOUT


def _is_cached(minio_client: Minio, bucket: str, object_name: str, local_path: Path) -> bool:
    """Check if file already exists locally with correct size.

    Args:
        minio_client: MinIO client instance
        bucket: Bucket name
        object_name: Remote object path
        local_path: Local file path

    Returns:
        True if local file exists and matches remote size
    """
    if not local_path.exists():
        return False

    try:
        stat = minio_client.stat_object(bucket, object_name)
        local_size = local_path.stat().st_size
        if local_size == stat.size:
            logger.info(f"Cached: {local_path.name} (size matches: {local_size} bytes)")
            return True
        logger.info(f"Re-downloading: {local_path.name} (size mismatch: local={local_size}, remote={stat.size})")
        return False
    except S3Error:
        # Remote object doesn't exist
        return False


def _download_with_progress(
    minio_client: Minio,
    bucket: str,
    object_name: str,
    local_path: Path,
) -> None:
    """Download file with progress bar and atomic write.

    Downloads to a temporary file first, then renames to final path on success.
    Cleans up partial file on failure.

    Args:
        minio_client: MinIO client instance
        bucket: Bucket name
        object_name: Remote object path
        local_path: Local file path

    Raises:
        S3Error: If download fails
    """
    # Get remote file size for progress bar
    stat = minio_client.stat_object(bucket, object_name)

    # Download to temp file for atomic write
    temp_path = local_path.with_suffix(local_path.suffix + ".tmp")

    try:
        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress bar
        with tqdm(
            total=stat.size,
            unit="B",
            unit_scale=True,
            desc=local_path.name,
            leave=True,
        ) as pbar:
            # MinIO's fget_object doesn't support progress callback directly,
            # so we use get_object with streaming and write chunks manually
            response = minio_client.get_object(bucket, object_name)
            try:
                with open(temp_path, "wb") as f:
                    for chunk in response.stream(8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            finally:
                response.close()
                response.release_conn()

        # Atomic rename on success
        temp_path.rename(local_path)
        logger.info(f"Downloaded: {object_name} -> {local_path}")

    except Exception:
        # Clean up partial file on failure
        if temp_path.exists():
            temp_path.unlink()
            logger.debug(f"Cleaned up partial file: {temp_path}")
        raise


def _build_download_tasks(
    minio_client: Minio,
    models: list[str],
) -> list[DownloadTask]:
    """Build list of download tasks, skipping cached files.

    Args:
        minio_client: MinIO client instance
        models: List of model names to download

    Returns:
        List of DownloadTask for files that need downloading
    """
    tasks = []

    for model_name in models:
        # Check model.onnx
        onnx_remote = f"{model_name}/1/model.onnx"
        onnx_local = MODEL_DIR / f"{model_name}.onnx"

        if not _is_cached(minio_client, BUCKET, onnx_remote, onnx_local):
            tasks.append(DownloadTask(onnx_remote, onnx_local, model_name))

        # Check model.onnx.data (optional - may not exist)
        data_remote = f"{model_name}/1/model.onnx.data"
        data_local = MODEL_DIR / f"{model_name}.onnx.data"

        try:
            # Only add task if remote file exists
            minio_client.stat_object(BUCKET, data_remote)
            if not _is_cached(minio_client, BUCKET, data_remote, data_local):
                tasks.append(DownloadTask(data_remote, data_local, model_name))
        except S3Error:
            # No .data file for this model - that's OK
            logger.debug(f"No external data file for {model_name}")

    return tasks


def download_models_parallel(minio_client: Minio, models: list[str]) -> None:
    """Download models in parallel with fail-fast behavior.

    Args:
        minio_client: MinIO client instance
        models: List of model names to download

    Raises:
        Exception: If any download fails (remaining downloads are cancelled)
    """
    max_concurrent, _ = _get_download_settings()

    # Build download tasks
    tasks = _build_download_tasks(minio_client, models)

    if not tasks:
        logger.info("All models already cached, skipping download")
        return

    logger.info(f"Downloading {len(tasks)} files with max {max_concurrent} concurrent connections")

    # Execute downloads in parallel
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(
                _download_with_progress,
                minio_client,
                BUCKET,
                task.remote_path,
                task.local_path,
            ): task
            for task in tasks
        }

        try:
            for future in as_completed(futures):
                task = futures[future]
                # This will raise if the download failed
                future.result()
        except Exception as e:
            # Fail fast: cancel remaining downloads
            logger.warning(f"Download failed, cancelling remaining: {e}")
            executor.shutdown(wait=False, cancel_futures=True)
            raise


def main() -> int:
    """Main entry point."""
    print("\n" + "=" * 60)
    print("Microservices Model Repository Initialization")
    print("=" * 60)
    print(f"MinIO Endpoint: {MINIO_ENDPOINT}")
    print(f"Bucket: {BUCKET}")
    print(f"Target Directory: {MODEL_DIR}")

    max_concurrent, timeout = _get_download_settings()
    print(f"Max Concurrent Downloads: {max_concurrent}")
    print(f"Download Timeout: {timeout}s")
    print("=" * 60)

    # Connect to MinIO
    logger.info(f"Connecting to MinIO at {MINIO_ENDPOINT}...")
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            secure=False,
        )
        # Test connection
        if not client.bucket_exists(BUCKET):
            logger.error(f"Bucket '{BUCKET}' does not exist!")
            return 1
        logger.info("Connected to MinIO")
    except Exception as e:
        logger.error(f"Failed to connect to MinIO: {e}")
        logger.debug("Stack trace:", exc_info=True)
        return 1

    # Ensure model directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Download models in parallel
    models = ["yolov5n", "mobilenetv2"]
    try:
        download_models_parallel(client, models)
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        logger.debug("Stack trace:", exc_info=True)
        return 1

    # Success summary
    print("\n" + "=" * 60)
    print("All models downloaded successfully")
    print("=" * 60)
    print("\nModel Directory Structure:")
    if MODEL_DIR.exists():
        for item in sorted(MODEL_DIR.glob("*")):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item.name} ({size_mb:.2f} MB)")

    print("\n" + "=" * 60)
    print("Init container completed successfully")
    print("Microservices can now start")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
