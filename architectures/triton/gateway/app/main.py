"""FastAPI application for Triton gateway service.

This module provides the HTTP API for the Triton architecture:
- POST /predict: Run detection + classification via Triton
- GET /health: Service health check

Pipeline is orchestrated by the gateway, with inference delegated to Triton.

Author: Matthew Hong
"""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import get_settings
from .logger import request_id_var, setup_logging
from .models import HealthResponse, PredictResponse
from .pipeline import TritonInferencePipeline
from .triton_client import TritonInferenceClient

# Global pipeline (initialized during lifespan)
pipeline: TritonInferencePipeline | None = None
triton_client: TritonInferenceClient | None = None
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles startup and shutdown:
    - Startup: Setup logging, initialize Triton client, wait for ready, init pipeline
    - Shutdown: Close Triton connection, cleanup resources

    Args:
        app: FastAPI application instance
    """
    global pipeline, triton_client
    settings = get_settings()

    # Setup JSON structured logging
    setup_logging(settings.LOG_LEVEL)
    logger.info("Starting Triton gateway", extra={"port": settings.PORT})

    # Initialize Triton client
    logger.info(f"Connecting to Triton at {settings.TRITON_GRPC_ENDPOINT}")
    triton_client = TritonInferenceClient(settings.TRITON_GRPC_ENDPOINT)

    # Wait for Triton server to be ready
    logger.info("Waiting for Triton server...")
    triton_client.wait_for_server_ready(timeout=settings.TRITON_TIMEOUT_SECONDS)

    # Verify models are loaded
    try:
        yolo_metadata = triton_client.get_model_metadata("yolov5n")
        mobilenet_metadata = triton_client.get_model_metadata("mobilenetv2")
        logger.info(f"Triton models loaded: yolov5n (v{yolo_metadata['versions']}), "
                   f"mobilenetv2 (v{mobilenet_metadata['versions']})")
    except Exception as e:
        logger.error(f"Failed to verify Triton models: {e}")
        raise

    # Load ImageNet labels
    labels_file = Path(settings.LABELS_FILE)
    if not labels_file.exists():
        # Fallback for local development
        labels_file = (
            Path(__file__).parent.parent.parent.parent / "src/shared/data/imagenet_labels.txt"
        )
        logger.warning(f"Labels file not found at {settings.LABELS_FILE}, using fallback: {labels_file}")

    # Initialize inference pipeline
    logger.info("Initializing inference pipeline")
    pipeline = TritonInferencePipeline(triton_client, labels_file)
    logger.info("Gateway ready for requests")

    yield

    # Cleanup
    logger.info("Shutting down Triton gateway")
    if triton_client:
        triton_client.close()


# Create FastAPI app with lifespan
app = FastAPI(
    title="Triton Gateway Service",
    description="Gateway for Triton Inference Server orchestration (detection + classification)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """Run detection and classification via Triton.

    Pipeline:
    1. Decode uploaded image
    2. YOLOv5n object detection (via Triton gRPC)
    3. MobileNetV2 classification for each detection (via Triton gRPC)
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

        # Run inference pipeline
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
