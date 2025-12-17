"""
MinIO Infrastructure Module

This module provides utilities for MinIO model registry operations:
- init_models.py: Upload ONNX models with Triton-compatible structure
- triton_config.py: Generate config.pbtxt files

Usage:
    python infrastructure/minio/init_models.py
"""

from pathlib import Path

MODULE_DIR = Path(__file__).parent
