"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def models_dir(project_root: Path) -> Path:
    """Return the models directory."""
    return project_root / "models"


@pytest.fixture
def test_data_dir(project_root: Path) -> Path:
    """Return the test dataset directory."""
    return project_root / "data" / "thesis_test_set"
