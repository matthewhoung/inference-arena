"""ClassificationService gRPC servicer implementation.

Implements the ClassificationService defined in inference.proto.
Runs MobileNetV2 inference in-process using ONNX Runtime.

Author: Matthew Hong
"""

import io
import logging
import time

import grpc
import numpy as np
from PIL import Image

from shared.proto import inference_pb2, inference_pb2_grpc

from .inference import ClassificationInference
from .logger import request_id_var

logger = logging.getLogger(__name__)


class ClassificationServiceServicer(inference_pb2_grpc.ClassificationServiceServicer):
    """gRPC servicer for MobileNetV2 classification.

    Implements:
    - Classify: Single image crop classification
    - ClassifyBatch: Batch classification (for future optimization)

    Attributes:
        inference: ClassificationInference instance with loaded model
    """

    def __init__(self, inference: ClassificationInference) -> None:
        """Initialize servicer with inference engine.

        Args:
            inference: Pre-initialized ClassificationInference instance
        """
        self.inference = inference
        logger.info("ClassificationServiceServicer initialized")

    async def Classify(
        self,
        request: inference_pb2.ClassificationRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.ClassificationResponse:
        """Classify a single image crop.

        Args:
            request: ClassificationRequest with image_crop bytes
            context: gRPC servicer context

        Returns:
            ClassificationResponse with result and timing
        """
        t_start = time.perf_counter()

        # Set request context for logging
        request_id_var.set(request.request_id)

        try:
            # Decode image from bytes
            t_preprocess_start = time.perf_counter()
            image = Image.open(io.BytesIO(request.image_crop))
            crop = np.array(image)

            # Ensure RGB format
            if crop.ndim == 2:
                # Grayscale to RGB
                crop = np.stack([crop, crop, crop], axis=-1)
            elif crop.shape[2] == 4:
                # RGBA to RGB
                crop = crop[:, :, :3]

            t_preprocess_end = time.perf_counter()

            # Run inference
            t_inference_start = time.perf_counter()
            class_id, class_name, confidence, top_k = self.inference.classify(crop)
            t_inference_end = time.perf_counter()

            t_total = time.perf_counter()

            # Build response
            response = inference_pb2.ClassificationResponse(
                request_id=request.request_id,
                result=inference_pb2.ClassificationResult(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                ),
                timing=inference_pb2.TimingInfo(
                    preprocessing_ms=(t_preprocess_end - t_preprocess_start) * 1000,
                    inference_ms=(t_inference_end - t_inference_start) * 1000,
                    total_ms=(t_total - t_start) * 1000,
                ),
            )

            # Add top-k results
            for cls_id, cls_name, conf in top_k:
                response.top_k.append(inference_pb2.ClassificationResult(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                ))

            logger.debug(
                f"Classify complete: {class_name} ({confidence:.3f})",
                extra={
                    "class_name": class_name,
                    "confidence": confidence,
                    "latency_ms": (t_total - t_start) * 1000,
                }
            )

            return response

        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)
            return inference_pb2.ClassificationResponse(
                request_id=request.request_id,
                error=str(e),
            )

    async def ClassifyBatch(
        self,
        request: inference_pb2.BatchClassificationRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.BatchClassificationResponse:
        """Classify multiple crops in a batch.

        Currently processes sequentially. Could be optimized for
        batched ONNX inference in future.

        Args:
            request: BatchClassificationRequest with list of requests
            context: gRPC servicer context

        Returns:
            BatchClassificationResponse with all results
        """
        t_start = time.perf_counter()

        responses = []
        for single_request in request.requests:
            response = await self.Classify(single_request, context)
            responses.append(response)

        t_total = time.perf_counter()

        return inference_pb2.BatchClassificationResponse(
            responses=responses,
            batch_timing=inference_pb2.TimingInfo(
                total_ms=(t_total - t_start) * 1000,
            ),
        )
