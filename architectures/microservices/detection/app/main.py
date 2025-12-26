"""FastAPI application for Detection service.

HTTP API for the microservices architecture Detection service:
- POST /predict: Run detection + classification (via gRPC to Classification service)
- GET /health: Service health check

Author: Matthew Hong
"""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import get_settings
from .grpc_client import ClassificationClient
from .inference import DetectionPipeline
from .logger import request_id_var, setup_logging
from .models import HealthResponse, PredictResponse

# Global instances
pipeline: DetectionPipeline | None = None
classification_client: ClassificationClient | None = None
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles startup and shutdown:
    - Startup: Initialize gRPC client, connect to Classification service, load YOLO model
    - Shutdown: Close gRPC connection
    """
    global pipeline, classification_client
    settings = get_settings()

    # Setup logging
    setup_logging(settings.LOG_LEVEL)
    logger.info("Starting Detection service", extra={"port": settings.PORT})

    # Verify YOLO model exists
    models_dir = Path(settings.MODELS_DIR)
    yolo_path = models_dir / "yolov5n.onnx"
    if not yolo_path.exists():
        raise RuntimeError(f"YOLO model not found: {yolo_path}")

    # Initialize gRPC client for classification service
    logger.info(
        f"Connecting to Classification service at {settings.CLASSIFICATION_GRPC_ENDPOINT}"
    )
    classification_client = ClassificationClient(settings.CLASSIFICATION_GRPC_ENDPOINT)
    await classification_client.connect()

    # Initialize detection pipeline
    logger.info("Initializing detection pipeline")
    pipeline = DetectionPipeline(models_dir, classification_client)
    logger.info("Detection service ready for requests", extra={"port": settings.PORT})

    yield

    # Cleanup
    logger.info("Shutting down Detection service")
    if classification_client:
        await classification_client.close()


app = FastAPI(
    title="Detection Service",
    description="Microservices architecture - Detection with gRPC classification",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """Run detection and classification on uploaded image.

    Pipeline:
    1. YOLOv5n detection (in-process)
    2. Parallel classification via gRPC to Classification service
    3. Return combined results with timing breakdown

    Args:
        file: Uploaded image file (JPEG, PNG, etc.)

    Returns:
        PredictResponse with request_id, detections, and timing
    """
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)

    logger.info("Received predict request", extra={"endpoint": "/predict"})

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        image_bytes = await file.read()
        results, timing = await pipeline.predict(image_bytes, request_id)

        logger.info(
            "Predict complete",
            extra={
                "endpoint": "/predict",
                "latency_ms": timing["total_ms"],
                "detections": len(results),
            },
        )

        return PredictResponse(
            request_id=request_id,
            detections=results,
            timing=timing,
        )
    except Exception as e:
        logger.error(f"Predict failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint.

    Returns:
        HealthResponse with status and models_loaded flag
    """
    return HealthResponse(
        status="healthy",
        models_loaded=pipeline is not None,
    )
