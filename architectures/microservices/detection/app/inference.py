"""Detection pipeline with async classification calls.

This module handles YOLO detection in-process and delegates
classification to the Classification gRPC service using parallel
asyncio.gather() calls for optimal performance.

Author: Matthew Hong
"""

import logging
import time
from pathlib import Path

import numpy as np

from shared.config import get_controlled_variables, get_model_config
from shared.model.registry import ModelRegistry, SessionConfig
from shared.processing import YOLOPreprocessor
from shared.processing.mobilenet_preprocess import extract_crop
from shared.processing.transforms import load_image_from_bytes

from .grpc_client import ClassificationClient
from .postprocess import parse_yolo_output

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Detection pipeline with async classification via gRPC.

    Runs YOLO detection in-process, then delegates classification
    to the Classification service using parallel gRPC calls.

    This implements the microservices architecture pattern:
    - Detection: In-process (YOLOv5n via ONNX Runtime)
    - Classification: Remote via gRPC (with asyncio.gather for parallelism)

    Attributes:
        registry: ModelRegistry for ONNX session management
        yolo_session: YOLOv5n inference session
        yolo_preprocessor: YOLO preprocessing pipeline
        classification_client: Async gRPC client for classification
        conf_threshold: Detection confidence threshold
        iou_threshold: NMS IoU threshold
    """

    def __init__(
        self,
        models_dir: Path,
        classification_client: ClassificationClient,
    ) -> None:
        """Initialize detection pipeline.

        Args:
            models_dir: Directory containing YOLO ONNX model
            classification_client: Pre-connected async gRPC client
        """
        # Load threading config from experiment.yaml
        onnx_config = get_controlled_variables("onnx_runtime")
        session_config = SessionConfig(
            intra_op_threads=onnx_config["intra_op_num_threads"],
            inter_op_threads=onnx_config["inter_op_num_threads"],
        )

        logger.info(
            f"Initializing pipeline with {session_config.intra_op_threads} "
            f"intra-op, {session_config.inter_op_threads} inter-op threads"
        )

        # Initialize model registry
        self.registry = ModelRegistry(models_dir, session_config)

        # Load YOLO model only (classification via gRPC)
        logger.info("Loading YOLOv5n model")
        self.yolo_session = self.registry.get_session("yolov5n")

        # Get model info
        model_info = self.registry.get_model_info("yolov5n")
        logger.info(f"YOLOv5n input shape: {model_info.input_shape}")

        # Initialize preprocessor
        self.yolo_preprocessor = YOLOPreprocessor()

        # Store classification client
        self.classification_client = classification_client

        # Load detection thresholds from experiment.yaml
        yolo_config = get_model_config("yolov5n")
        self.conf_threshold = yolo_config["confidence_threshold"]
        self.iou_threshold = yolo_config["iou_threshold"]

        logger.info(
            f"Detection thresholds: confidence={self.conf_threshold}, "
            f"iou={self.iou_threshold}"
        )

        logger.info("Detection pipeline initialized")

    async def predict(
        self,
        image_bytes: bytes,
        request_id: str,
    ) -> tuple[list[dict], dict[str, float]]:
        """Run detection + classification pipeline.

        Pipeline:
        1. Decode image from bytes
        2. YOLOv5n detection (in-process)
        3. Extract crops for each detection
        4. Parallel classification via gRPC (asyncio.gather) - H1b KEY
        5. Aggregate results with timing

        Args:
            image_bytes: Raw image bytes (JPEG, PNG)
            request_id: Unique request ID for tracing

        Returns:
            Tuple of (results, timing)
        """
        timing: dict[str, float] = {}
        t0 = time.perf_counter()

        # Decode image
        image = load_image_from_bytes(image_bytes)

        # ====================================================================
        # YOLO Detection (in-process)
        # ====================================================================
        t_det_start = time.perf_counter()

        # Preprocess
        yolo_result = self.yolo_preprocessor(image)

        # Run YOLO inference
        yolo_output = self.yolo_session.run(None, {"images": yolo_result.tensor})[0]

        # Parse detections (NMS)
        detections_letterbox = parse_yolo_output(
            yolo_output,
            self.conf_threshold,
            self.iou_threshold,
        )

        # Scale boxes to original coordinates
        if len(detections_letterbox) > 0:
            detections_orig = yolo_result.scale_boxes_to_original(detections_letterbox)
        else:
            detections_orig = detections_letterbox

        t_det_end = time.perf_counter()
        timing["detection_ms"] = (t_det_end - t_det_start) * 1000

        # ====================================================================
        # Classification via gRPC (parallel with asyncio.gather)
        # ====================================================================
        t_cls_start = time.perf_counter()

        results = []

        if len(detections_orig) > 0:
            # Extract all crops and build box dicts
            crops = []
            boxes = []
            for det in detections_orig:
                crop = extract_crop(image, det)
                crops.append(crop)
                boxes.append({
                    "x1": float(det[0]),
                    "y1": float(det[1]),
                    "x2": float(det[2]),
                    "y2": float(det[3]),
                    "confidence": float(det[4]),
                    "class_id": int(det[5]),
                })

            # ==========================================================
            # CRITICAL: Parallel classification using asyncio.gather()
            # This is the KEY H1b hypothesis requirement
            # ==========================================================
            classification_responses = await self.classification_client.classify_parallel(
                request_id, crops, boxes
            )

            # Aggregate results
            for box, cls_response in zip(boxes, classification_responses):
                if cls_response.error:
                    logger.warning(
                        f"Classification error for {cls_response.request_id}: "
                        f"{cls_response.error}"
                    )
                    continue

                results.append({
                    "detection": box,
                    "classification": {
                        "class_id": cls_response.result.class_id,
                        "class_name": cls_response.result.class_name,
                        "confidence": cls_response.result.confidence,
                    },
                })

        t_cls_end = time.perf_counter()
        timing["classification_ms"] = (t_cls_end - t_cls_start) * 1000
        timing["total_ms"] = (t_cls_end - t0) * 1000

        return results, timing
