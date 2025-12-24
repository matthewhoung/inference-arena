# =============================================================================
# Monolithic Inference Service - Dockerfile
# =============================================================================
# CPU-constrained inference service with YOLOv5n + MobileNetV2
#
# Architecture:
#   - FastAPI application server
#   - In-process ONNX Runtime inference (CPU-only)
#   - Models downloaded from MinIO on startup
#
# Build:
#   docker build -t monolithic-service -f architectures/monolithic/Dockerfile .
#
# Run:
#   docker compose -f architectures/monolithic/docker-compose.yml up
#
# Author: Matthew Hong
# =============================================================================

# Use official Python 3.11 slim image for minimal size
FROM python:3.11-slim

# Set working directory
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
# We install dependencies manually to avoid pulling in CUDA packages from torch
# The monolithic service only needs ONNX Runtime (CPU), not PyTorch for inference
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
    "minio>=7.2.0" \
    "prometheus-client>=0.19.0"

# Copy shared library code
# Note: Code imports as 'shared', so copy to /app/shared (not /app/src/shared)
COPY src/shared /app/shared

# Copy experiment configuration (required for threading config)
COPY experiment.yaml /app/experiment.yaml

# Copy monolithic application code
COPY architectures/monolithic/app /app/app

# Create models directory (will be populated from MinIO on startup)
RUN mkdir -p /app/models

# Set Python path to find shared module
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose FastAPI port
EXPOSE 8100

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/health')" || exit 1

# Run FastAPI with uvicorn
# - Host 0.0.0.0 to accept connections from Docker network
# - Port 8100 (matches config.py default)
# - Log level info for structured JSON logs
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100", "--log-level", "info"]
