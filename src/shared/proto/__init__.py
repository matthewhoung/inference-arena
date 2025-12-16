"""
Proto Module - gRPC Service Definitions

This module provides gRPC service definitions for inter-service
communication in the microservices architecture.

Services:
    ClassificationService: Classify image crops from detection results

The proto definitions ensure consistent API contracts between:
- Detection service (producer)
- Classification service (consumer)

Generated files (inference_pb2.py, inference_pb2_grpc.py) are created
by running: python scripts/generate_proto.py

Specification Reference:
    Foundation Specification §6 gRPC Interface
"""

from pathlib import Path

# Proto file location
PROTO_DIR = Path(__file__).parent
PROTO_FILE = PROTO_DIR / "inference.proto"


def get_proto_path() -> Path:
    """Get path to inference.proto file."""
    return PROTO_FILE


def is_generated() -> bool:
    """Check if proto files have been generated."""
    pb2_file = PROTO_DIR / "inference_pb2.py"
    grpc_file = PROTO_DIR / "inference_pb2_grpc.py"
    return pb2_file.exists() and grpc_file.exists()


# Lazy imports for generated modules
def get_messages():
    """
    Get generated protobuf message classes.
    
    Returns:
        Module containing message classes
    
    Raises:
        ImportError: If proto files not generated
    """
    if not is_generated():
        raise ImportError(
            "Proto files not generated. Run 'python scripts/generate_proto.py' first."
        )
    from shared.proto import inference_pb2
    return inference_pb2


def get_services():
    """
    Get generated gRPC service classes.
    
    Returns:
        Module containing service stubs and servicers
    
    Raises:
        ImportError: If proto files not generated
    """
    if not is_generated():
        raise ImportError(
            "Proto files not generated. Run 'python scripts/generate_proto.py' first."
        )
    from shared.proto import inference_pb2_grpc
    return inference_pb2_grpc
