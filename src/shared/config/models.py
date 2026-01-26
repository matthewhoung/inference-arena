"""Configuration Models Module.

This module provides Pydantic models for typed configuration access.

Classes:
    ServicePorts: Service port configuration with validation

Functions:
    get_service_ports: Get service ports from experiment.yaml with validation
    clear_service_ports_cache: Clear the service ports cache

Author: Matthew Hong
"""

from pydantic import BaseModel, field_validator

from shared.exceptions import ConfigKeyError
from shared.warnings import warn_once

from .loader import get_config

# =============================================================================
# Service Ports Configuration
# =============================================================================


class ServicePorts(BaseModel):
    """Service port configuration with validation.

    Provides typed access to service ports from experiment.yaml.
    Validates port ranges and detects duplicates.

    Attributes:
        minio_api: MinIO API port (default 9000)
        minio_console: MinIO console port (default 9001)
        prometheus: Prometheus port (default 9090)
        grafana: Grafana port (default 3000)
        otel_collector: OTel Collector Prometheus metrics port (default 8889)
        otel_health: OTel Collector health check port (default 13133)
    """

    minio_api: int
    minio_console: int
    prometheus: int
    grafana: int
    otel_collector: int
    otel_health: int

    @field_validator("*")
    @classmethod
    def port_in_valid_range(cls, v: int) -> int:
        """Validate port is in valid range 1-65535."""
        if not 1 <= v <= 65535:
            raise ValueError(f"Port {v} not in valid range 1-65535")
        return v


_service_ports_cache: ServicePorts | None = None


def get_service_ports() -> ServicePorts:
    """Get service ports from experiment.yaml with validation.

    Ports are loaded from the services section in experiment.yaml.
    The function validates port ranges and detects duplicate port
    assignments, logging a warning (W002) if duplicates are found.

    Raises:
        ConfigKeyError: If required port config missing
        ValueError: If port validation fails (invalid port range)

    Returns:
        ServicePorts model with all configured ports

    Example:
        >>> ports = get_service_ports()
        >>> ports.minio_api
        9000
        >>> ports.prometheus
        9090
    """
    global _service_ports_cache
    if _service_ports_cache is not None:
        return _service_ports_cache

    config = get_config()
    try:
        services = config["services"]
        ports = ServicePorts(
            minio_api=services["minio"]["api_port"],
            minio_console=services["minio"]["console_port"],
            prometheus=services["prometheus"]["port"],
            grafana=services["grafana"]["port"],
            otel_collector=services["otel_collector"]["port"],
            otel_health=services["otel_collector"]["health_port"],
        )
    except KeyError as e:
        raise ConfigKeyError(
            f"Missing service port configuration in experiment.yaml: {e}. "
            "Ensure services section is defined with all required ports."
        ) from e

    # Detect duplicate ports
    seen: dict[int, str] = {}
    for name, port in [
        ("minio_api", ports.minio_api),
        ("minio_console", ports.minio_console),
        ("prometheus", ports.prometheus),
        ("grafana", ports.grafana),
        ("otel_collector", ports.otel_collector),
        ("otel_health", ports.otel_health),
    ]:
        if port in seen:
            warn_once(
                "W002",
                f"Duplicate port {port} configured for {name} and {seen[port]}",
                "Check services section in experiment.yaml",
            )
        seen[port] = name

    _service_ports_cache = ports
    return ports


def clear_service_ports_cache() -> None:
    """Clear the service ports cache. Primarily for testing.

    Clears the cached ServicePorts instance, causing the next call
    to get_service_ports() to reload from configuration.
    """
    global _service_ports_cache
    _service_ports_cache = None
