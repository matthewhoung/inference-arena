"""Configuration module using pydantic-settings.

This module provides environment variable management for the Triton gateway
service. Uses pydantic-settings for automatic validation and .env file support.

Author: Matthew Hong
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        TRITON_GRPC_ENDPOINT: Triton server gRPC endpoint (host:port)
        TRITON_TIMEOUT_SECONDS: Maximum wait time for Triton server ready
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
        PORT: FastAPI server port
        LABELS_FILE: Path to ImageNet labels file
    """

    TRITON_GRPC_ENDPOINT: str = "localhost:8001"
    TRITON_TIMEOUT_SECONDS: int = 60
    LOG_LEVEL: str = "INFO"
    PORT: int = 8300
    LABELS_FILE: str = "/app/shared/data/imagenet_labels.txt"

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
