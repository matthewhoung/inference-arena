"""FastAPI application for monolithic inference service.

This module provides the HTTP API for the monolithic architecture:
- POST /predict: Run detection + classification on an image
- GET /health: Service health check

Models are downloaded from MinIO on startup and loaded into memory.

Author: Matthew Hong
"""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from minio import Minio

from .config import get_settings
from .inference import InferencePipeline
from .logger import request_id_var, setup_logging
from .models import HealthResponse, PredictResponse

# Global pipeline (initialized during lifespan)
pipeline: InferencePipeline | None = None
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles startup and shutdown:
    - Startup: Setup logging, download models from MinIO, initialize pipeline
    - Shutdown: Cleanup resources

    Args:
        app: FastAPI application instance
    """
    global pipeline
    settings = get_settings()

    # Setup JSON structured logging
    setup_logging(settings.LOG_LEVEL)
    logger.info("Starting monolithic service", extra={"port": settings.PORT})

    # Download models from MinIO
    models_dir = Path(settings.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    minio_client = Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )

    # Download YOLOv5n model
    yolo_path = models_dir / "yolov5n.onnx"
    if not yolo_path.exists():
        logger.info("Downloading yolov5n from MinIO")
        minio_client.fget_object(
            settings.MINIO_BUCKET,
            "yolov5n/1/model.onnx",
            str(yolo_path),
        )
        logger.info(f"Downloaded yolov5n to {yolo_path}")
    else:
        logger.info(f"YOLOv5n already exists at {yolo_path}")

    # Download MobileNetV2 model
    mobilenet_path = models_dir / "mobilenetv2.onnx"
    mobilenet_data_path = models_dir / "mobilenetv2.onnx.data"

    if not mobilenet_path.exists():
        logger.info("Downloading mobilenetv2 from MinIO")
        minio_client.fget_object(
            settings.MINIO_BUCKET,
            "mobilenetv2/1/model.onnx",
            str(mobilenet_path),
        )
        logger.info(f"Downloaded mobilenetv2 to {mobilenet_path}")
    else:
        logger.info(f"MobileNetV2 already exists at {mobilenet_path}")

    # Download MobileNetV2 external data file
    if not mobilenet_data_path.exists():
        logger.info("Downloading mobilenetv2 external data from MinIO")
        minio_client.fget_object(
            settings.MINIO_BUCKET,
            "mobilenetv2/1/model.onnx.data",
            str(mobilenet_data_path),
        )
        logger.info(f"Downloaded mobilenetv2.onnx.data to {mobilenet_data_path}")
    else:
        logger.info(f"MobileNetV2 external data already exists at {mobilenet_data_path}")

    # ImageNet labels path
    labels_file = (
        Path(__file__).parent.parent.parent.parent / "src/shared/data/imagenet_labels.txt"
    )

    # Initialize inference pipeline
    logger.info("Initializing inference pipeline")
    pipeline = InferencePipeline(models_dir, labels_file)
    logger.info("Service ready for requests")

    yield

    # Cleanup
    logger.info("Shutting down monolithic service")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Monolithic Inference Service",
    description="True monolithic architecture with in-process detection and classification",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """Run detection and classification on uploaded image.

    Pipeline:
    1. Decode uploaded image
    2. YOLOv5n object detection
    3. MobileNetV2 classification for each detection
    4. Return results with timing breakdown

    Args:
        file: Uploaded image file (JPEG, PNG, etc.)

    Returns:
        PredictResponse with detections, classifications, and timing

    Raises:
        HTTPException: If pipeline not initialized or inference fails
    """
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)

    logger.info("Received predict request", extra={"endpoint": "/predict"})

    if pipeline is None:
        logger.error("Pipeline not initialized", extra={"endpoint": "/predict"})
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        # Read image bytes
        image_bytes = await file.read()

        # Run inference
        results, timing = pipeline.predict(image_bytes)

        logger.info(
            "Predict complete",
            extra={
                "endpoint": "/predict",
                "latency_ms": timing["total_ms"],
                "detections": len(results),
                "status_code": 200,
            },
        )

        return PredictResponse(
            request_id=request_id,
            detections=results,
            timing=timing,
        )

    except Exception as e:
        logger.error(
            f"Predict failed: {e}",
            extra={"endpoint": "/predict", "status_code": 500},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint.

    Returns:
        HealthResponse indicating service health and model status
    """
    request_id_var.set(str(uuid.uuid4()))

    return HealthResponse(
        status="healthy",
        models_loaded=pipeline is not None,
    )
