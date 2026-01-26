r"""Container validation using Docker SDK.

This module validates that containers exist, are running, and are healthy
before performing operations that depend on them (e.g., Prometheus queries).

The validation collects ALL errors rather than failing fast on the first,
providing a comprehensive error message listing all invalid containers.

Usage:
    from shared.validation import validate_containers
    from shared.exceptions import ConfigError

    try:
        validate_containers(["prometheus", "cadvisor"])
    except ConfigError as e:
        print(f"Container validation failed:\n{e}")

Author: Matthew Hong
"""

import logging
from typing import Any

import docker
from docker.errors import APIError, NotFound

from shared.exceptions import ConfigError

logger = logging.getLogger(__name__)


def _get_container_status(container: Any) -> tuple[str, str | None]:
    """Get container running status and health status.

    Args:
        container: Docker container object

    Returns:
        Tuple of (running_status, health_status or None if no healthcheck)
    """
    running_status = container.status

    # Health status is only available if container has HEALTHCHECK directive
    health = container.attrs.get("State", {}).get("Health", {})
    health_status = health.get("Status") if health else None

    return running_status, health_status


def validate_containers(
    container_names: list[str],
    max_wait: float = 30.0,  # noqa: ARG001 - Reserved for future health polling
) -> None:
    """Validate containers exist and are healthy.

    Checks each container for:
    1. Existence (container exists in Docker)
    2. Running status (container.status == "running")
    3. Health status (if container has HEALTHCHECK, must be "healthy")

    If a container has no HEALTHCHECK directive, it is considered healthy
    if it is running.

    Args:
        container_names: List of container names to validate
        max_wait: Maximum seconds to wait for health (reserved for future use)

    Raises:
        ConfigError: If any container is invalid, with details for ALL failures

    Example:
        >>> validate_containers(["prometheus", "cadvisor"])
        # No error if all containers valid

        >>> validate_containers(["missing-container"])
        # Raises ConfigError with "missing-container: container not found"
    """
    if not container_names:
        logger.debug("No containers to validate (empty list)")
        return

    try:
        client = docker.from_env()
    except docker.errors.DockerException as e:
        raise ConfigError(f"Failed to connect to Docker daemon: {e}") from e

    errors: list[str] = []

    for name in container_names:
        logger.debug(f"Validating container: {name}")

        try:
            container = client.containers.get(name)
            running_status, health_status = _get_container_status(container)

            # Check running status
            if running_status != "running":
                error_msg = f"{name}: not running (status: {running_status})"
                logger.debug(f"Container validation failed: {error_msg}")
                errors.append(error_msg)
                continue

            # Check health status (if healthcheck exists)
            if health_status is not None:
                if health_status != "healthy":
                    error_msg = f"{name}: unhealthy (health: {health_status})"
                    logger.debug(f"Container validation failed: {error_msg}")
                    errors.append(error_msg)
                else:
                    logger.debug(f"Container {name}: running and healthy")
            else:
                # No healthcheck defined - treat as healthy if running
                logger.debug(f"Container {name}: running (no healthcheck)")

        except NotFound:
            error_msg = f"{name}: container not found"
            logger.debug(f"Container validation failed: {error_msg}")
            errors.append(error_msg)

        except APIError as e:
            error_msg = f"{name}: Docker API error - {e}"
            logger.debug(f"Container validation failed: {error_msg}")
            errors.append(error_msg)

    if errors:
        error_detail = "\n".join(f"  - {e}" for e in errors)
        raise ConfigError(f"Container validation failed:\n{error_detail}")

    logger.debug(f"All {len(container_names)} containers validated successfully")
