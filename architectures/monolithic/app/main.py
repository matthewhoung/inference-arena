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
    - Startup: Setup logging, verify models, initialize pipeline
    - Shutdown: Cleanup resources

    Models are pre-downloaded by init container to /app/models volume.

    Args:
        app: FastAPI application instance
    """
    global pipeline
    settings = get_settings()

    # Setup JSON structured logging
    setup_logging(settings.LOG_LEVEL)
    logger.info("Starting monolithic service", extra={"port": settings.PORT})

    # Verify models exist (downloaded by init container)
    models_dir = Path(settings.MODELS_DIR)
    yolo_path = models_dir / "yolov5n.onnx"
    mobilenet_path = models_dir / "mobilenetv2.onnx"
    mobilenet_data_path = models_dir / "mobilenetv2.onnx.data"

    # Check all required model files
    missing_files = []
    if not yolo_path.exists():
        missing_files.append(str(yolo_path))
    if not mobilenet_path.exists():
        missing_files.append(str(mobilenet_path))
    if not mobilenet_data_path.exists():
        missing_files.append(str(mobilenet_data_path))

    if missing_files:
        error_msg = f"Missing model files (should be downloaded by init container): {', '.join(missing_files)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f"Models verified at {models_dir}")
    logger.info(f"  - YOLOv5n: {yolo_path.stat().st_size / (1024*1024):.2f} MB")
    logger.info(f"  - MobileNetV2: {mobilenet_path.stat().st_size / (1024*1024):.2f} MB")
    logger.info(f"  - MobileNetV2 data: {mobilenet_data_path.stat().st_size / (1024*1024):.2f} MB")

    # Load ImageNet labels
    labels_file = Path("/app/shared/data/imagenet_labels.txt")
    if not labels_file.exists():
        # Fallback for local development
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
        raise HTTPException(status_code=500, detail=str(e)) from e


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
