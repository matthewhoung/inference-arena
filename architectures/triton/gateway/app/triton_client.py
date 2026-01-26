"""Triton gRPC client wrapper for inference.

This module provides a high-level interface to communicate with Triton
Inference Server via gRPC. Handles connection management, inference requests,
and error handling.

When TRITON_BATCHING=true, uses batched model variants (yolov5n_batched,
mobilenetv2_batched) which have dynamic batching enabled.

Author: Matthew Hong
"""

import asyncio
import logging
import os
import time

import numpy as np
import tritonclient.grpc.aio as grpcclient

logger = logging.getLogger(__name__)

# Batching configuration from environment
TRITON_BATCHING = os.getenv("TRITON_BATCHING", "false").lower() == "true"


class TritonInferenceClient:
    """Wrapper for Triton gRPC client.

    Provides simplified interface for running inference on YOLOv5n and
    MobileNetV2 models via Triton Inference Server.

    Attributes:
        server_url: Triton server gRPC endpoint (e.g., "triton-server:8001")
        client: grpcclient.InferenceServerClient instance
    """

    def __init__(self, server_url: str):
        """Initialize Triton client.

        Args:
            server_url: Triton server gRPC endpoint (e.g., "localhost:8001")
        """
        self.server_url = server_url
        self.client = grpcclient.InferenceServerClient(url=server_url)
        logger.info(f"Triton client initialized: {server_url}")

    async def wait_for_server_ready(self, timeout: int = 60) -> bool:
        """Async wait for Triton server."""
        start = time.time()
        attempt = 0
        while time.time() - start < timeout:
            try:
                if await self.client.is_server_ready():
                    logger.info("Triton server is ready")
                    return True
            except Exception:
                attempt += 1
                wait_time = min(2**attempt, 5)
                logger.debug(f"Waiting for Triton... (attempt {attempt})")
                await asyncio.sleep(wait_time)

        raise ConnectionError(f"Triton server not ready after {timeout}s")

    async def infer_yolo(self, image_tensor: np.ndarray) -> np.ndarray:
        """Run YOLOv5n inference asynchronously."""
        inputs = [grpcclient.InferInput("images", image_tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(image_tensor)
        outputs = [grpcclient.InferRequestedOutput("output0")]

        model_name = "yolov5n_batched" if TRITON_BATCHING else "yolov5n"

        # Await the inference result
        response = await self.client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
        )

        return response.as_numpy("output0")

    async def infer_mobilenet(self, crop_tensor: np.ndarray) -> np.ndarray:
        """Run MobileNetV2 classification asynchronously."""
        inputs = [grpcclient.InferInput("input", crop_tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(crop_tensor)
        outputs = [grpcclient.InferRequestedOutput("output")]

        model_name = "mobilenetv2_batched" if TRITON_BATCHING else "mobilenetv2"

        response = await self.client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
        )

        return response.as_numpy("output")

    async def close(self):
        """Close gRPC connection."""
        self.client.close()
        logger.info("Triton client closed")
