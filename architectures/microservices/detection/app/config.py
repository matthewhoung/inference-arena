"""Configuration for Detection service.

This module provides environment-based settings for the detection service.
Uses pydantic-settings for validation and type coercion.

Author: Matthew Hong
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Detection service settings.

    Attributes:
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
        PORT: HTTP server port (default: 8200)
        MODELS_DIR: Directory containing ONNX model files
        CLASSIFICATION_GRPC_ENDPOINT: gRPC endpoint for classification service
    """

    LOG_LEVEL: str = "INFO"
    PORT: int = 8200
    MODELS_DIR: str = "/app/models"
    CLASSIFICATION_GRPC_ENDPOINT: str = "classification:8201"

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
