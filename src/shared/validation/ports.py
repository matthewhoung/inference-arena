"""Port Validation Module.

This module provides validation utilities for comparing port configurations
between experiment.yaml and docker-compose files.

Functions:
    parse_compose_ports: Extract host ports from docker-compose file
    validate_ports: Compare expected ports against compose file ports
    validate_infrastructure_ports: Validate infrastructure compose against ServicePorts

The validation follows the "fail fast" pattern - any port mismatch raises
a ConfigError immediately at startup rather than allowing silent misconfiguration.

Author: Matthew Hong
"""

from pathlib import Path
from typing import Any

import yaml

from shared.exceptions import ConfigError


def parse_compose_ports(compose_path: Path) -> dict[str, list[int]]:
    """Extract host ports from docker-compose.yml.

    Parses a docker-compose file and extracts all host port mappings
    for each service. Handles all docker-compose port format variations:
    - "8100:8100" (string with host:container)
    - "8100" (string, container port only - used as host port)
    - 8100 (integer)
    - {"published": 8100, "target": 8100} (long syntax dict)

    Args:
        compose_path: Path to docker-compose.yml file

    Returns:
        Dict mapping service name to list of host ports.
        Services with no ports section are excluded from the result.

    Raises:
        FileNotFoundError: If compose file does not exist
        yaml.YAMLError: If compose file contains invalid YAML

    Example:
        >>> ports = parse_compose_ports(Path("docker-compose.yml"))
        >>> ports["minio"]
        [9000, 9001]
    """
    with open(compose_path) as f:
        config = yaml.safe_load(f)

    service_ports: dict[str, list[int]] = {}

    services = config.get("services", {})
    if not services:
        return service_ports

    for service_name, service_config in services.items():
        ports: list[int] = []
        port_mappings = service_config.get("ports", [])

        for port_mapping in port_mappings:
            host_port = _extract_host_port(port_mapping)
            if host_port is not None:
                ports.append(host_port)

        if ports:
            service_ports[service_name] = ports

    return service_ports


def _extract_host_port(port_mapping: Any) -> int | None:
    """Extract host port from a docker-compose port mapping.

    Handles all port mapping formats:
    - "8100:8100" -> 8100
    - "${MINIO_API_PORT:-9000}:9000" -> 9000 (extracts default from env var)
    - "8100" -> 8100
    - 8100 -> 8100
    - {"published": 8100, "target": 8100} -> 8100

    Args:
        port_mapping: Port mapping in any docker-compose format

    Returns:
        Host port as integer, or None if parsing fails
    """
    if isinstance(port_mapping, str):
        # Handle "host:container" or "port" format
        # Need to be careful with env var syntax like ${VAR:-default}:container
        # which contains colons inside the env var reference
        host_part = _extract_host_part(port_mapping)
        return _parse_port_value(host_part)

    elif isinstance(port_mapping, int):
        return port_mapping

    elif isinstance(port_mapping, dict):
        # Long syntax: {"published": host, "target": container}
        if "published" in port_mapping:
            published: int = port_mapping["published"]
            return published
        elif "target" in port_mapping:
            target: int = port_mapping["target"]
            return target

    return None


def _extract_host_part(port_mapping: str) -> str:
    """Extract the host port part from a port mapping string.

    Handles env var syntax like ${VAR:-default}:container where the
    env var reference contains colons that shouldn't be used as separators.

    Args:
        port_mapping: Port mapping string (e.g., "8100:8100" or "${VAR:-9000}:9000")

    Returns:
        Host part of the port mapping
    """
    # If it starts with ${, find the closing } and then look for : after it
    if port_mapping.startswith("${"):
        # Find the closing brace
        close_brace = port_mapping.find("}")
        if close_brace != -1:
            # Look for : after the closing brace
            colon_pos = port_mapping.find(":", close_brace)
            if colon_pos != -1:
                return port_mapping[:colon_pos]
            # No colon after brace - entire string is host part
            return port_mapping

    # Standard case: split on first colon
    colon_pos = port_mapping.find(":")
    if colon_pos != -1:
        return port_mapping[:colon_pos]

    # No colon - entire string is port
    return port_mapping


def _parse_port_value(value: str) -> int | None:
    """Parse a port value string, handling environment variable syntax.

    Args:
        value: Port value string, possibly with env var syntax

    Returns:
        Port as integer, or None if parsing fails

    Examples:
        >>> _parse_port_value("9000")
        9000
        >>> _parse_port_value("${MINIO_API_PORT:-9000}")
        9000
    """
    value = value.strip()

    # Handle ${VAR:-default} syntax - extract default value
    if value.startswith("${") and ":-" in value:
        # Extract default value after :-
        default_start = value.index(":-") + 2
        default_end = value.rindex("}")
        value = value[default_start:default_end]

    try:
        return int(value)
    except ValueError:
        return None


def validate_ports(
    compose_path: Path, expected_ports: dict[str, int]
) -> None:
    """Validate docker-compose ports match expected configuration.

    Compares expected service ports against those defined in a docker-compose
    file. For each service/port pair in expected_ports, verifies that:
    1. The service exists in the compose file
    2. The expected port is in the service's port list

    All mismatches are collected and reported together, rather than failing
    on the first error.

    Args:
        compose_path: Path to docker-compose.yml file
        expected_ports: Dict mapping service name to expected host port

    Raises:
        ConfigError: If any port mismatch detected. Error message lists
            all mismatches in format:
            "service: expected port X, found [Y, Z] or 'no ports'"

    Example:
        >>> validate_ports(
        ...     Path("docker-compose.yml"),
        ...     {"minio": 9000, "prometheus": 9090}
        ... )  # Raises ConfigError if ports don't match
    """
    compose_ports = parse_compose_ports(compose_path)
    errors: list[str] = []

    for service, expected_port in expected_ports.items():
        actual_ports = compose_ports.get(service)

        if actual_ports is None:
            errors.append(
                f"{service}: expected port {expected_port}, found no ports"
            )
        elif expected_port not in actual_ports:
            errors.append(
                f"{service}: expected port {expected_port}, found {actual_ports}"
            )

    if errors:
        raise ConfigError(
            f"Port mismatch between experiment.yaml and {compose_path}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def validate_infrastructure_ports() -> None:
    """Validate infrastructure docker-compose against ServicePorts.

    Loads service ports from experiment.yaml via ServicePorts model and
    validates them against the infrastructure docker-compose file.

    This is a convenience function that combines:
    1. Loading ServicePorts from shared.config
    2. Building expected_ports dict for infrastructure services
    3. Calling validate_ports() with infrastructure compose path

    Raises:
        ConfigError: If any port mismatch between experiment.yaml and
            docker-compose.infra.yml
        ConfigKeyError: If required port config missing in experiment.yaml

    Example:
        >>> validate_infrastructure_ports()  # Raises if ports mismatch
    """
    from shared.config import get_service_ports

    ports = get_service_ports()

    # Map ServicePorts attributes to docker-compose service names
    # Service names in docker-compose.infra.yml
    expected_ports = {
        "minio": ports.minio_api,
        "prometheus": ports.prometheus,
        "grafana": ports.grafana,
        "otel-collector": ports.otel_collector,
    }

    # Find infrastructure compose file path
    # The compose file is at infrastructure/docker-compose.infra.yml
    # relative to project root
    compose_path = _find_infrastructure_compose()

    validate_ports(compose_path, expected_ports)


def _find_infrastructure_compose() -> Path:
    """Find the infrastructure docker-compose file.

    Searches for docker-compose.infra.yml in expected locations
    relative to project structure.

    Returns:
        Path to docker-compose.infra.yml

    Raises:
        ConfigError: If compose file cannot be found
    """
    # Try common locations relative to where code runs
    candidates = [
        Path("infrastructure/docker-compose.infra.yml"),
        Path(__file__).parent.parent.parent.parent
        / "infrastructure"
        / "docker-compose.infra.yml",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise ConfigError(
        "Cannot find infrastructure/docker-compose.infra.yml. "
        "Ensure you are running from project root or compose file exists."
    )
