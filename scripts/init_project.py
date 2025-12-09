#!/usr/bin/env python3
"""
Initialize the inference-arena project directory structure.

This script creates all necessary directories and placeholder files
for the thesis experimental repository. Run from project root:

    python scripts/init_project.py

No external dependencies required - uses only Python standard library.
"""

from pathlib import Path


def create_gitkeep(directory: Path) -> None:
    """Create a .gitkeep file in an empty directory."""
    gitkeep = directory / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()


def init_project() -> None:
    """Create the complete project directory structure."""
    
    # Get project root (parent of scripts/)
    project_root = Path(__file__).parent.parent
    
    print("🏟️  Initializing Inference Arena project structure...")
    print(f"   Project root: {project_root.resolve()}")
    print()
    
    # =========================================================================
    # Directory Structure Definition
    # =========================================================================
    
    directories = [
        # Common (shared controlled variables)
        "common",
        "common/proto",
        
        # Models (downloaded, gitignored)
        "models",
        
        # Data (test dataset)
        "data",
        "data/thesis_test_set",
        
        # Architecture A: Monolithic
        "architectures/monolithic",
        "architectures/monolithic/app",
        
        # Architecture B: Microservices
        "architectures/microservices",
        "architectures/microservices/detection",
        "architectures/microservices/detection/app",
        "architectures/microservices/classification",
        "architectures/microservices/classification/app",
        
        # Architecture C: Triton
        "architectures/triton",
        "architectures/triton/gateway",
        "architectures/triton/gateway/app",
        "architectures/triton/model_repository",
        "architectures/triton/model_repository/yolov5n",
        "architectures/triton/model_repository/mobilenetv2",
        
        # Infrastructure
        "infrastructure",
        "infrastructure/minio",
        "infrastructure/prometheus",
        
        # Experiments
        "experiments",
        "experiments/configs",
        
        # Analysis
        "analysis",
        
        # Results (gitignored)
        "results",
        "results/raw",
        
        # Tests
        "tests",
        
        # Scripts
        "scripts",
        
        # GitHub templates
        ".github",
        ".github/ISSUE_TEMPLATE",
        ".github/workflows",
    ]
    
    # =========================================================================
    # Create Directories
    # =========================================================================
    
    print("📁 Creating directories...")
    
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {dir_path}/")
    
    print()
    
    # =========================================================================
    # Create .gitkeep files for empty directories
    # =========================================================================
    
    gitkeep_dirs = [
        "data/thesis_test_set",
        "models",
        "results/raw",
        "experiments/configs",
        ".github/workflows",
    ]
    
    print("📄 Creating .gitkeep files...")
    
    for dir_path in gitkeep_dirs:
        full_path = project_root / dir_path
        create_gitkeep(full_path)
        print(f"   ✓ {dir_path}/.gitkeep")
    
    print()
    
    # =========================================================================
    # Create __init__.py files for Python packages
    # =========================================================================
    
    init_files = [
        "common/__init__.py",
        "common/proto/__init__.py",
        "tests/__init__.py",
    ]
    
    print("🐍 Creating __init__.py files...")
    
    for file_path in init_files:
        full_path = project_root / file_path
        if not full_path.exists():
            full_path.write_text('"""Package initialization."""\n')
            print(f"   ✓ {file_path}")
        else:
            print(f"   - {file_path} (already exists)")
    
    print()
    
    # =========================================================================
    # Create placeholder conftest.py for pytest
    # =========================================================================
    
    conftest_path = project_root / "tests" / "conftest.py"
    if not conftest_path.exists():
        conftest_content = '''"""Pytest configuration and fixtures."""

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
'''
        conftest_path.write_text(conftest_content)
        print("🧪 Created tests/conftest.py")
    
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("=" * 60)
    print("✅ Project structure initialized successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Copy generated files from Claude to project root:")
    print("     - .gitignore")
    print("     - pyproject.toml")
    print("     - README.md")
    print("     - MILESTONES.md")
    print("     - .github/ISSUE_TEMPLATE/*.yml")
    print()
    print("  2. Install dependencies:")
    print("     uv sync")
    print()
    print("  3. Commit initial structure:")
    print("     git add .")
    print('     git commit -m "🎉 Initialize project structure"')
    print("     git push -u origin main")
    print()
    print("  4. Create milestones on GitHub (see MILESTONES.md)")
    print()


if __name__ == "__main__":
    init_project()