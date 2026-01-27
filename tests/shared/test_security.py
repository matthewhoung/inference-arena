"""Tests for shared.security module.

Tests credential checking behavior in different environments:
- Development mode: warnings logged via warn_once
- Production mode: InsecureCredentialsError raised
- Production + Suppressed: warnings logged but no error raised

Author: Matthew Hong
"""

import logging
import os
from unittest.mock import patch

import pytest

from shared.exceptions import InsecureCredentialsError
from shared.security import (
    DEFAULT_CREDENTIALS,
    _is_warning_suppressed,
    check_credentials,
    is_production,
)
from shared.warnings import reset_warnings


@pytest.fixture(autouse=True)
def reset_warning_state():
    """Reset warning state before and after each test."""
    reset_warnings()
    yield
    reset_warnings()


class TestIsProduction:
    """Tests for is_production() function."""

    def test_returns_false_when_not_set(self):
        """Returns False when ENVIRONMENT is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure ENVIRONMENT is not set
            os.environ.pop("ENVIRONMENT", None)
            assert is_production() is False

    def test_returns_false_when_empty(self):
        """Returns False when ENVIRONMENT is empty string."""
        with patch.dict(os.environ, {"ENVIRONMENT": ""}):
            assert is_production() is False

    def test_returns_false_when_development(self):
        """Returns False when ENVIRONMENT is 'development'."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            assert is_production() is False

    def test_returns_true_when_production(self):
        """Returns True when ENVIRONMENT is 'production'."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            assert is_production() is True

    def test_case_sensitive(self):
        """Returns False when ENVIRONMENT is 'Production' (wrong case)."""
        with patch.dict(os.environ, {"ENVIRONMENT": "Production"}):
            assert is_production() is False


class TestIsWarningSuppressed:
    """Tests for _is_warning_suppressed() function."""

    def test_not_suppressed_when_empty(self):
        """Returns False when INFERENCE_ARENA_SUPPRESS_WARNINGS is not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("INFERENCE_ARENA_SUPPRESS_WARNINGS", None)
            assert _is_warning_suppressed("W003") is False

    def test_suppressed_when_in_list(self):
        """Returns True when code is in suppression list."""
        with patch.dict(os.environ, {"INFERENCE_ARENA_SUPPRESS_WARNINGS": "W003"}):
            assert _is_warning_suppressed("W003") is True

    def test_suppressed_with_multiple_codes(self):
        """Returns True when code is among multiple suppressed codes."""
        with patch.dict(os.environ, {"INFERENCE_ARENA_SUPPRESS_WARNINGS": "W001,W003"}):
            assert _is_warning_suppressed("W003") is True

    def test_handles_whitespace(self):
        """Handles whitespace in suppression list."""
        with patch.dict(os.environ, {"INFERENCE_ARENA_SUPPRESS_WARNINGS": "W001, W003 "}):
            assert _is_warning_suppressed("W003") is True

    def test_not_suppressed_when_different_code(self):
        """Returns False when code is not in suppression list."""
        with patch.dict(os.environ, {"INFERENCE_ARENA_SUPPRESS_WARNINGS": "W001,W002"}):
            assert _is_warning_suppressed("W003") is False


class TestCheckCredentials:
    """Tests for check_credentials() function."""

    def test_no_warning_with_custom_access_key(self, caplog):
        """No warning when access_key is custom."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ENVIRONMENT", None)
            with caplog.at_level(logging.WARNING):
                check_credentials("custom", "minioadmin", "MinIO")
            assert "W003" not in caplog.text

    def test_no_warning_with_custom_secret_key(self, caplog):
        """No warning when secret_key is custom."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ENVIRONMENT", None)
            with caplog.at_level(logging.WARNING):
                check_credentials("minioadmin", "custom", "MinIO")
            assert "W003" not in caplog.text

    def test_no_warning_with_both_custom(self, caplog):
        """No warning when both credentials are custom."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ENVIRONMENT", None)
            with caplog.at_level(logging.WARNING):
                check_credentials("custom", "secret", "MinIO")
            assert "W003" not in caplog.text

    def test_warns_in_development_with_defaults(self, caplog):
        """Logs W003 warning in development with default credentials."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ENVIRONMENT", None)
            with caplog.at_level(logging.WARNING):
                check_credentials("minioadmin", "minioadmin", "MinIO")
            assert "W003" in caplog.text
            assert "minioadmin" in caplog.text
            assert "MinIO" in caplog.text

    def test_raises_in_production_with_defaults(self):
        """Raises InsecureCredentialsError in production with default credentials."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            with pytest.raises(InsecureCredentialsError):
                check_credentials("minioadmin", "minioadmin", "MinIO")

    def test_error_message_includes_code(self):
        """Exception message includes [W003] code."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            with pytest.raises(InsecureCredentialsError) as exc_info:
                check_credentials("minioadmin", "minioadmin", "MinIO")
            assert "[W003]" in str(exc_info.value)

    def test_error_message_includes_credentials(self):
        """Exception message includes the credential value for debugging."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            with pytest.raises(InsecureCredentialsError) as exc_info:
                check_credentials("minioadmin", "minioadmin", "MinIO")
            assert "minioadmin" in str(exc_info.value)

    def test_suppression_in_production_still_logs(self, caplog):
        """Suppressed warning in production logs but doesn't raise."""
        with patch.dict(
            os.environ,
            {"ENVIRONMENT": "production", "INFERENCE_ARENA_SUPPRESS_WARNINGS": "W003"},
        ):
            with caplog.at_level(logging.WARNING):
                # Should NOT raise
                check_credentials("minioadmin", "minioadmin", "MinIO")
            # But should still log for audit
            assert "W003" in caplog.text
            assert "(suppressed)" in caplog.text

    def test_warn_once_deduplication(self, caplog):
        """Only one warning logged when called multiple times in development."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ENVIRONMENT", None)
            with caplog.at_level(logging.WARNING):
                check_credentials("minioadmin", "minioadmin", "MinIO")
                check_credentials("minioadmin", "minioadmin", "MinIO")
            # Count occurrences of W003 in log
            w003_count = caplog.text.count("[W003]")
            assert w003_count == 1, f"Expected 1 warning, got {w003_count}"


class TestDefaultCredentials:
    """Tests for DEFAULT_CREDENTIALS constant."""

    def test_is_frozen_set(self):
        """DEFAULT_CREDENTIALS is a frozenset (immutable)."""
        assert isinstance(DEFAULT_CREDENTIALS, frozenset)

    def test_contains_minioadmin(self):
        """DEFAULT_CREDENTIALS contains 'minioadmin'."""
        assert "minioadmin" in DEFAULT_CREDENTIALS
