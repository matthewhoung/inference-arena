"""Configuration for Classification gRPC service.

This module provides environment-based settings for the classification service.
Uses pydantic-settings for validation and type coercion.

Author: Matthew Hong
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Classification service settings.

    Attributes:
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
        PORT: gRPC server port (default: 8201)
        MODELS_DIR: Directory containing ONNX model files
        LABELS_FILE: Path to ImageNet labels file
    """

    LOG_LEVEL: str = "INFO"
    PORT: int = 8201
    MODELS_DIR: str = "/app/models"
    LABELS_FILE: str = "/app/shared/data/imagenet_labels.txt"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached Settings instance (singleton pattern).

    Returns:
        Validated Settings instance
    """
    return Settings()
