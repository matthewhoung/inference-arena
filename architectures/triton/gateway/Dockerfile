# =============================================================================
# Triton Gateway - Dockerfile
# =============================================================================
#
# FastAPI service orchestrating detection → classification via Triton gRPC
#
# Build Context: Project root (../../)
# Build Command:
#   docker build -t inference-arena-triton-gateway:latest \
#     -f architectures/triton/gateway/Dockerfile .
#
# Author: Matthew Hong
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# =============================================================================
# System Dependencies
# =============================================================================
# Install OpenCV runtime dependencies
# =============================================================================
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 && \
    rm -rf /var/lib/apt/lists/*

# =============================================================================
# Python Dependencies
# =============================================================================
# Install via uv for faster installs
# =============================================================================
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
    "fastapi>=0.109.0" \
    "uvicorn[standard]>=0.27.0" \
    "python-multipart>=0.0.6" \
    "pydantic-settings>=2.0.0" \
    "tritonclient[grpc]>=2.45.0" \
    "opencv-python>=4.8.0" \
    "numpy>=1.24.0" \
    "pyyaml>=6.0"

# =============================================================================
# Application Files
# =============================================================================
# Copy shared modules and experiment configuration
# =============================================================================
COPY src/shared /app/shared
COPY experiment.yaml /app/experiment.yaml

# Copy gateway application code
COPY architectures/triton/gateway/app /app/app

# =============================================================================
# Environment Configuration
# =============================================================================
ENV PYTHONPATH=/app:$PYTHONPATH

# =============================================================================
# Expose Port
# =============================================================================
EXPOSE 8300

# =============================================================================
# Health Check
# =============================================================================
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8300/health')" || exit 1

# =============================================================================
# Start Server
# =============================================================================
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8300", "--log-level", "info"]
