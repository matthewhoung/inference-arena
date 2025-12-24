"""Triton gRPC client wrapper for inference.

This module provides a high-level interface to communicate with Triton
Inference Server via gRPC. Handles connection management, inference requests,
and error handling.

Author: Matthew Hong
"""

import logging
import time
import numpy as np
import tritonclient.grpc as grpcclient

logger = logging.getLogger(__name__)


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

    def wait_for_server_ready(self, timeout: int = 60) -> bool:
        """Wait for Triton server to be ready.

        Polls server health with exponential backoff until ready or timeout.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if server is ready

        Raises:
            ConnectionError: If server not ready within timeout
        """
        start = time.time()
        attempt = 0
        while time.time() - start < timeout:
            try:
                if self.client.is_server_ready():
                    logger.info("Triton server is ready")
                    return True
            except Exception as e:
                attempt += 1
                wait_time = min(2**attempt, 10)  # Exponential backoff, max 10s
                logger.debug(
                    f"Waiting for Triton... (attempt {attempt}, {e})"
                )
                time.sleep(wait_time)

        raise ConnectionError(f"Triton server not ready after {timeout}s")

    def infer_yolo(self, image_tensor: np.ndarray) -> np.ndarray:
        """Run YOLOv5n inference.

        Args:
            image_tensor: Preprocessed image [1, 3, 640, 640] float32

        Returns:
            YOLO output [1, 84, 8400] float32

        Raises:
            Exception: If inference fails
        """
        # Validate input shape
        expected_shape = (1, 3, 640, 640)
        if image_tensor.shape != expected_shape:
            raise ValueError(
                f"Invalid YOLO input shape: {image_tensor.shape}, "
                f"expected {expected_shape}"
            )

        # Prepare input
        inputs = [
            grpcclient.InferInput("images", image_tensor.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(image_tensor)

        # Prepare output
        outputs = [grpcclient.InferRequestedOutput("output0")]

        # Run inference
        response = self.client.infer(
            model_name="yolov5n",
            inputs=inputs,
            outputs=outputs,
        )

        return response.as_numpy("output0")

    def infer_mobilenet(self, crop_tensor: np.ndarray) -> np.ndarray:
        """Run MobileNetV2 classification.

        Args:
            crop_tensor: Preprocessed crop [1, 3, 224, 224] float32

        Returns:
            Classification logits [1, 1000] float32

        Raises:
            Exception: If inference fails
        """
        # Validate input shape
        expected_shape = (1, 3, 224, 224)
        if crop_tensor.shape != expected_shape:
            raise ValueError(
                f"Invalid MobileNet input shape: {crop_tensor.shape}, "
                f"expected {expected_shape}"
            )

        # Prepare input
        inputs = [
            grpcclient.InferInput("input", crop_tensor.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(crop_tensor)

        # Prepare output
        outputs = [grpcclient.InferRequestedOutput("output")]

        # Run inference
        response = self.client.infer(
            model_name="mobilenetv2",
            inputs=inputs,
            outputs=outputs,
        )

        return response.as_numpy("output")

    def get_model_metadata(self, model_name: str) -> dict:
        """Query Triton model metadata.

        Args:
            model_name: Name of the model (e.g., "yolov5n", "mobilenetv2")

        Returns:
            Dictionary with model metadata (name, versions, platform, inputs, outputs)

        Raises:
            Exception: If metadata query fails
        """
        metadata = self.client.get_model_metadata(model_name)
        return {
            "name": metadata.name,
            "versions": metadata.versions,
            "platform": metadata.platform,
            "inputs": [
                {
                    "name": inp.name,
                    "datatype": inp.datatype,
                    "shape": list(inp.shape),
                }
                for inp in metadata.inputs
            ],
            "outputs": [
                {
                    "name": out.name,
                    "datatype": out.datatype,
                    "shape": list(out.shape),
                }
                for out in metadata.outputs
            ],
        }

    def close(self):
        """Close gRPC connection."""
        self.client.close()
        logger.info("Triton client closed")
