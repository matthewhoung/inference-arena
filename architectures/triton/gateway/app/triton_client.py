import logging
import os
import time
import numpy as np
import tritonclient.grpc as grpcclient

logger = logging.getLogger(__name__)

TRITON_BATCHING = os.getenv("TRITON_BATCHING", "false").lower() == "true"

class TritonInferenceClient:
    def __init__(self, server_url: str):
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
                wait_time = min(2**attempt, 5)
                logger.debug(f"Waiting for Triton... (attempt {attempt})")
                time.sleep(wait_time)  # <--- Blocking sleep
        raise ConnectionError(f"Triton server not ready after {timeout}s")

    def infer_yolo(self, image_tensor: np.ndarray) -> np.ndarray:
        inputs = [grpcclient.InferInput("images", image_tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(image_tensor)
        outputs = [grpcclient.InferRequestedOutput("output0")]
        
        model_name = "yolov5n_batched" if TRITON_BATCHING else "yolov5n"
        
        response = self.client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        return response.as_numpy("output0")

    def infer_mobilenet(self, crop_tensor: np.ndarray) -> np.ndarray:
        inputs = [grpcclient.InferInput("input", crop_tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(crop_tensor)
        outputs = [grpcclient.InferRequestedOutput("output")]
        
        model_name = "mobilenetv2_batched" if TRITON_BATCHING else "mobilenetv2"
        
        # Blocking call (No await)
        response = self.client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
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
        self.client.close()