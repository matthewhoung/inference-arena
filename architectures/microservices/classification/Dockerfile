# =============================================================================
# Classification gRPC Service - Dockerfile
# =============================================================================
# MobileNetV2 classification service with gRPC interface
#
# Architecture:
#   - gRPC server (no HTTP endpoint)
#   - In-process ONNX Runtime inference (CPU-only)
#   - Models pre-loaded from shared volume
#
# Build:
#   docker build -t classification-service -f architectures/microservices/classification/Dockerfile .
#
# Author: Matthew Hong
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files needed for installation
COPY pyproject.toml README.md ./

# Install Python dependencies
# Classification service needs: grpcio, onnxruntime, opencv, PIL
# No FastAPI needed - pure gRPC service
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
    "opencv-python>=4.8.0" \
    "numpy>=1.24.0" \
    "pyyaml>=6.0" \
    "onnxruntime>=1.18.0,<1.23.0" \
    "grpcio>=1.62.0" \
    "grpcio-tools>=1.62.0" \
    "protobuf>=4.25.0" \
    "pillow>=10.0.0" \
    "pydantic-settings>=2.0.0"

# Copy shared library code
COPY src/shared /app/shared

# Copy experiment configuration (required for threading config)
COPY experiment.yaml /app/experiment.yaml

# Copy classification service code
COPY architectures/microservices/classification/app /app/app

# Create models directory (will be populated from init container)
RUN mkdir -p /app/models

# Set Python path to find shared module
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose gRPC port
EXPOSE 8201

# Health check using gRPC channel ready
# Note: grpc_health_probe could be added for production
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import grpc; c=grpc.insecure_channel('localhost:8201'); grpc.channel_ready_future(c).result(timeout=5)" || exit 1

# Run gRPC server
CMD ["python", "-m", "app.main"]
