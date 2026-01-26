"""Validation utilities for Inference Arena.

This package provides utilities for validating container existence,
status, and health, as well as port configuration validation between
experiment.yaml and docker-compose files.

Usage:
    from shared.validation import validate_containers, validate_ports

    # Validate containers before Prometheus queries
    validate_containers(["container1", "container2"])

    # Validate port configuration matches
    from pathlib import Path
    validate_ports(Path("docker-compose.yml"), {"minio": 9000})

    # Validate infrastructure ports against experiment.yaml
    from shared.validation import validate_infrastructure_ports
    validate_infrastructure_ports()

Author: Matthew Hong
"""

from .containers import validate_containers
from .ports import parse_compose_ports, validate_infrastructure_ports, validate_ports

__all__ = [
    "validate_containers",
    "validate_ports",
    "validate_infrastructure_ports",
    "parse_compose_ports",
]
