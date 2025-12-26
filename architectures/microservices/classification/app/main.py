"""gRPC server for Classification service.

Standalone gRPC server running MobileNetV2 classification.
No HTTP endpoint - pure gRPC service.

Author: Matthew Hong
"""

import asyncio
import logging
import signal
from pathlib import Path

import grpc
from grpc import aio

from shared.config import get_controlled_variables
from shared.model.registry import ModelRegistry, SessionConfig
from shared.proto import inference_pb2_grpc

from .config import get_settings
from .inference import ClassificationInference
from .logger import setup_logging
from .servicer import ClassificationServiceServicer

logger = logging.getLogger(__name__)


async def serve() -> None:
    """Start and run the gRPC server."""
    settings = get_settings()

    # Setup logging
    setup_logging(settings.LOG_LEVEL)
    logger.info(f"Starting Classification gRPC service on port {settings.PORT}")

    # Load threading config from experiment.yaml
    onnx_config = get_controlled_variables("onnx_runtime")
    session_config = SessionConfig(
        intra_op_threads=onnx_config["intra_op_num_threads"],
        inter_op_threads=onnx_config["inter_op_num_threads"],
    )
    logger.info(
        f"ONNX Runtime threads: {session_config.intra_op_threads} intra-op, "
        f"{session_config.inter_op_threads} inter-op"
    )

    # Verify model exists
    models_dir = Path(settings.MODELS_DIR)
    mobilenet_path = models_dir / "mobilenetv2.onnx"
    if not mobilenet_path.exists():
        raise RuntimeError(f"Model not found: {mobilenet_path}")

    logger.info(f"Loading MobileNetV2 model from {models_dir}")

    # Initialize model registry
    registry = ModelRegistry(models_dir, session_config)

    # Load labels
    labels_file = Path(settings.LABELS_FILE)

    # Initialize inference engine
    inference = ClassificationInference(registry, labels_file)
    logger.info("Classification inference engine initialized")

    # Create gRPC server with options
    server_options = [
        ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
    ]
    server = aio.server(options=server_options)

    # Add servicer
    servicer = ClassificationServiceServicer(inference)
    inference_pb2_grpc.add_ClassificationServiceServicer_to_server(servicer, server)

    # Bind to port
    listen_addr = f"[::]:{settings.PORT}"
    server.add_insecure_port(listen_addr)

    logger.info(f"Classification service listening on {listen_addr}")

    # Start server
    await server.start()

    # Setup graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(sig):
        logger.info(f"Received {sig.name}, initiating shutdown...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))

    logger.info("Classification service ready for requests", extra={"port": settings.PORT})

    # Wait for shutdown signal
    await shutdown_event.wait()

    # Graceful shutdown with timeout
    logger.info("Shutting down gRPC server...")
    await server.stop(grace=5)
    logger.info("Classification service stopped")


def main() -> None:
    """Entry point for classification service."""
    asyncio.run(serve())


if __name__ == "__main__":
    main()
