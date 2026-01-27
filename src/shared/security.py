"""Security utilities for credential validation.

This module provides security utilities for checking credentials and
detecting insecure configurations. It integrates with the warning system
to provide appropriate feedback based on the environment.

Usage:
    from shared.security import check_credentials, is_production

    # Check credentials - warns in dev, raises in prod
    check_credentials("minioadmin", "minioadmin", "MinIO")

    # Check environment
    if is_production():
        # Production-specific behavior
        pass

Warning Codes:
    W003: MinIO using default credentials

Author: Matthew Hong
"""

import logging
import os

from shared.exceptions import InsecureCredentialsError
from shared.warnings import warn_once

logger = logging.getLogger(__name__)

# Known default/insecure credentials (frozen for immutability)
DEFAULT_CREDENTIALS = frozenset({"minioadmin"})

# Warning code for insecure credentials
SUPPRESS_WARNING_CODE = "W003"


def is_production() -> bool:
    """Check if running in production environment.

    Returns True only if ENVIRONMENT is exactly "production" (case-sensitive).

    Returns:
        bool: True if ENVIRONMENT == "production", False otherwise

    Examples:
        >>> # ENVIRONMENT not set
        >>> is_production()
        False

        >>> # ENVIRONMENT="production"
        >>> is_production()
        True

        >>> # ENVIRONMENT="Production" (wrong case)
        >>> is_production()
        False
    """
    return os.environ.get("ENVIRONMENT") == "production"


def _is_warning_suppressed(code: str) -> bool:
    """Check if a warning code is suppressed.

    Checks the INFERENCE_ARENA_SUPPRESS_WARNINGS environment variable
    for comma-separated warning codes.

    Args:
        code: Warning code to check (e.g., "W003")

    Returns:
        bool: True if the warning code is in the suppression list
    """
    suppress_env = os.environ.get("INFERENCE_ARENA_SUPPRESS_WARNINGS", "")
    suppressed_codes = {c.strip() for c in suppress_env.split(",") if c.strip()}
    return code in suppressed_codes


def check_credentials(access_key: str, secret_key: str, service_name: str = "MinIO") -> None:
    """Check for insecure default credentials.

    Validates that credentials are not using known default values. Behavior
    depends on the environment:

    - Development: Logs a warning (W003) via warn_once
    - Production: Raises InsecureCredentialsError (unless suppressed)
    - Production + Suppressed: Logs warning but does not raise

    Args:
        access_key: The access key to check
        secret_key: The secret key to check
        service_name: Service name for error messages (default: "MinIO")

    Raises:
        InsecureCredentialsError: In production with default credentials
            (unless suppressed via INFERENCE_ARENA_SUPPRESS_WARNINGS)

    Examples:
        >>> # Custom credentials - no warning
        >>> check_credentials("myuser", "mysecret", "MinIO")

        >>> # Default credentials in development - warning logged
        >>> check_credentials("minioadmin", "minioadmin", "MinIO")
        # Logs: WARNING [W003] MinIO using default credentials (minioadmin)...

        >>> # Default credentials in production - raises error
        >>> # ENVIRONMENT=production
        >>> check_credentials("minioadmin", "minioadmin", "MinIO")
        # Raises: InsecureCredentialsError
    """
    # Only check if BOTH credentials are default
    if access_key not in DEFAULT_CREDENTIALS or secret_key not in DEFAULT_CREDENTIALS:
        return

    message = f"{service_name} using default credentials ({access_key}). " "See docs/ENVIRONMENT.md"

    if is_production():
        if _is_warning_suppressed(SUPPRESS_WARNING_CODE):
            # Suppressed in production: log for audit trail but don't raise
            logger.warning(f"[{SUPPRESS_WARNING_CODE}] {message} (suppressed)")
            return
        # Not suppressed in production: raise error
        raise InsecureCredentialsError(f"[{SUPPRESS_WARNING_CODE}] {message}")

    # Development: use warn_once for deduplication
    warn_once(SUPPRESS_WARNING_CODE, message)
