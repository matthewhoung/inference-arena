# =============================================================================
# Detection HTTP + gRPC Client Service - Dockerfile
# =============================================================================
# YOLOv5n detection service with FastAPI HTTP and gRPC client for classification
#
# Architecture:
#   - FastAPI HTTP server (external interface)
#   - gRPC client to Classification service
#   - In-process ONNX Runtime inference for YOLO (CPU-only)
#
# Build:
#   docker build -t detection-service -f architectures/microservices/detection/Dockerfile .
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
# Detection service needs: fastapi, grpcio, onnxruntime, opencv, PIL
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
    "opencv-python>=4.8.0" \
    "numpy>=1.24.0" \
    "pyyaml>=6.0" \
    "onnxruntime>=1.18.0,<1.23.0" \
    "fastapi>=0.109.0" \
    "uvicorn[standard]>=0.27.0" \
    "python-multipart>=0.0.6" \
    "httpx>=0.26.0" \
    "pydantic-settings>=2.0.0" \
    "grpcio>=1.62.0" \
    "grpcio-tools>=1.62.0" \
    "protobuf>=4.25.0" \
    "pillow>=10.0.0"

# Copy shared library code
COPY src/shared /app/shared

# Copy experiment configuration (required for threading config)
COPY experiment.yaml /app/experiment.yaml

# Copy detection service code
COPY architectures/microservices/detection/app /app/app

# Create models directory (will be populated from init container)
RUN mkdir -p /app/models

# Set Python path to find shared module
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose HTTP port
EXPOSE 8200

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8200/health')" || exit 1

# Run FastAPI with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8200", "--log-level", "info"]
