"""Warning utilities with deduplication and codes for documentation lookup.

This module provides warning utilities that:
- Log warnings only once per session (deduplication)
- Use warning codes (W001, W002, etc.) for documentation lookup
- Support optional fix suggestions
- Allow suppression via environment variable

Usage:
    from shared.warnings import warn_once, reset_warnings

    # Log a warning (only shown once per session)
    warn_once("W001", "Configuration missing optional field", "Add 'timeout' to config")

    # Suppress specific warnings via environment variable:
    # export INFERENCE_ARENA_SUPPRESS_WARNINGS="W001,W002"

Author: Matthew Hong
"""

import logging
import os
from typing import Set

logger = logging.getLogger(__name__)
_warned: Set[str] = set()

# Environment variable to suppress specific warnings (comma-separated codes)
_SUPPRESS_WARNINGS = [
    code.strip()
    for code in os.environ.get("INFERENCE_ARENA_SUPPRESS_WARNINGS", "").split(",")
    if code.strip()
]


def warn_once(code: str, message: str, fix_suggestion: str | None = None) -> None:
    """Log warning once per session with code for lookup.

    Warnings are deduplicated by code, so the same warning code will only
    be logged once per session. Warnings can be suppressed by adding their
    codes to the INFERENCE_ARENA_SUPPRESS_WARNINGS environment variable.

    Args:
        code: Warning code (W001, W002, etc.) for documentation lookup
        message: Warning message describing the issue
        fix_suggestion: Optional actionable fix suggestion

    Example:
        >>> warn_once("W001", "Config missing timeout", "Add 'timeout: 30' to config")
        # Logs: WARNING [W001] Config missing timeout Fix: Add 'timeout: 30' to config
        >>> warn_once("W001", "Config missing timeout")  # Not logged again
    """
    if code in _warned or code in _SUPPRESS_WARNINGS:
        return
    _warned.add(code)
    full_message = f"[{code}] {message}"
    if fix_suggestion:
        full_message += f" Fix: {fix_suggestion}"
    logger.warning(full_message)


def reset_warnings() -> None:
    """Reset warned codes. Primarily for testing.

    Clears the set of already-warned codes, allowing warnings
    to be logged again. This is useful in test scenarios where
    you want to verify warning behavior multiple times.

    Example:
        >>> warn_once("W001", "Test warning")  # Logged
        >>> warn_once("W001", "Test warning")  # Not logged (duplicate)
        >>> reset_warnings()
        >>> warn_once("W001", "Test warning")  # Logged again
    """
    _warned.clear()
