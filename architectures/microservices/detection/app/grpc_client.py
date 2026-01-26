"""Async gRPC client for Classification service.

This module provides an async interface for calling the Classification
gRPC service. Uses grpc.aio for native asyncio support required for
parallel classification calls with asyncio.gather().

Author: Matthew Hong
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import grpc
import grpc.aio
import numpy as np

from shared.proto import inference_pb2, inference_pb2_grpc

if TYPE_CHECKING:
    # These types are only used for type checking, not at runtime
    from shared.proto.inference_pb2 import ClassificationResponse

logger = logging.getLogger(__name__)


def _check_proto_available() -> None:
    """Verify proto modules are available."""
    if inference_pb2 is None or inference_pb2_grpc is None:
        raise ImportError(
            "Proto files not generated. Run 'python scripts/generate_proto.py' first."
        )


class ClassificationClient:
    """Async gRPC client for ClassificationService.

    Uses grpc.aio for native asyncio integration, enabling parallel
    classification calls via asyncio.gather() to minimize network overhead.

    This is the KEY implementation for H1b hypothesis testing:
    - Parallel gRPC calls mask network overhead
    - Target: <20% overhead vs monolithic

    Attributes:
        endpoint: gRPC server endpoint (e.g., "classification:8201")
        channel: Async gRPC channel
        stub: ClassificationService stub
    """

    def __init__(self, endpoint: str) -> None:
        """Initialize async gRPC client.

        Args:
            endpoint: Classification service gRPC endpoint
        """
        self.endpoint = endpoint
        self.channel: grpc.aio.Channel | None = None
        self.stub: Any = None  # inference_pb2_grpc.ClassificationServiceStub | None
        logger.info(f"ClassificationClient configured for {endpoint}")

    async def connect(self) -> None:
        """Establish async gRPC channel connection."""
        _check_proto_available()

        # Create channel with options
        options = [
            ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
        ]
        self.channel = grpc.aio.insecure_channel(self.endpoint, options=options)
        self.stub = inference_pb2_grpc.ClassificationServiceStub(self.channel)

        # Wait for channel ready with timeout
        try:
            await asyncio.wait_for(
                self.channel.channel_ready(),
                timeout=30.0,
            )
            logger.info(f"Connected to Classification service at {self.endpoint}")
        except TimeoutError as err:
            raise RuntimeError(
                f"Timeout connecting to Classification service at {self.endpoint}"
            ) from err

    async def close(self) -> None:
        """Close gRPC channel."""
        if self.channel:
            await self.channel.close()
            logger.info("Classification client closed")

    async def classify(
        self,
        request_id: str,
        crop: np.ndarray,
        source_box: dict[str, Any] | None = None,
    ) -> ClassificationResponse:
        """Classify a single image crop.

        Args:
            request_id: Unique request ID for tracing
            crop: RGB uint8 numpy array [H, W, 3]
            source_box: Optional detection box context

        Returns:
            ClassificationResponse proto with result and timing
        """
        if self.stub is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        # Send raw RGB bytes (lossless transport, no JPEG compression)
        # Ensure array is contiguous for proper serialization
        crop_contiguous = np.ascontiguousarray(crop)
        height, width = crop_contiguous.shape[:2]

        # Build request with raw bytes and dimensions
        request = inference_pb2.ClassificationRequest(
            request_id=request_id,
            image_crop=crop_contiguous.tobytes(),
            crop_height=height,
            crop_width=width,
        )

        # Add source box if provided
        if source_box:
            request.source_box.CopyFrom(inference_pb2.BoundingBox(
                x1=source_box["x1"],
                y1=source_box["y1"],
                x2=source_box["x2"],
                y2=source_box["y2"],
                confidence=source_box.get("confidence", 0.0),
                class_id=source_box.get("class_id", 0),
            ))

        # Make async RPC call
        response = await self.stub.Classify(request)
        return response

    async def classify_parallel(
        self,
        request_id: str,
        crops: list[np.ndarray],
        boxes: list[dict[str, Any]],
    ) -> list[ClassificationResponse]:
        """Classify multiple crops in parallel using asyncio.gather().

        ============================================================
        CRITICAL: This is the KEY implementation for H1b hypothesis
        ============================================================

        H1b tests whether parallel gRPC calls can mask network overhead
        and keep microservices competitive with monolithic at low concurrency.

        Testable prediction: (microservices.p99 - monolithic.p99) / monolithic.p99 < 0.20

        By using asyncio.gather(), all classification requests are sent
        concurrently, overlapping network latency with inference time.

        Args:
            request_id: Base request ID (suffixed with index)
            crops: List of RGB uint8 numpy arrays
            boxes: List of detection box dicts for context

        Returns:
            List of ClassificationResponse protos
        """
        if not crops:
            return []

        # Create tasks for parallel execution
        tasks = [
            self.classify(f"{request_id}_{i}", crop, box)
            for i, (crop, box) in enumerate(zip(crops, boxes, strict=True))
        ]

        # Execute all classification calls in parallel
        # This is the CRITICAL H1b requirement - asyncio.gather enables
        # concurrent RPC calls that overlap network latency
        responses = await asyncio.gather(*tasks)

        return list(responses)
