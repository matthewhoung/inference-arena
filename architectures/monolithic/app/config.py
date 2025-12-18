"""Configuration module using pydantic-settings.

This module provides environment variable management for the monolithic
inference service. Uses pydantic-settings for automatic validation and
.env file support.

Author: Matthew Hong
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        MINIO_ENDPOINT: MinIO server endpoint (host:port)
        MINIO_ACCESS_KEY: MinIO access key
        MINIO_SECRET_KEY: MinIO secret key
        MINIO_BUCKET: MinIO bucket name for models
        MINIO_SECURE: Use HTTPS for MinIO connection
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
        PORT: FastAPI server port
        MODELS_DIR: Local directory for downloaded models
    """

    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "models"
    MINIO_SECURE: bool = False
    LOG_LEVEL: str = "INFO"
    PORT: int = 8100
    MODELS_DIR: str = "./models"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached Settings instance (singleton pattern).

    Returns:
        Settings instance loaded from environment
    """
    return Settings()
