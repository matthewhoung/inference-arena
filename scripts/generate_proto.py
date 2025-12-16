#!/usr/bin/env python3
"""
Generate Proto Script - Compile .proto files to Python.

This script compiles inference.proto to:
- inference_pb2.py (message classes)
- inference_pb2_grpc.py (service stubs and servicers)

Usage:
    python scripts/generate_proto.py           # Generate proto files
    python scripts/generate_proto.py --verify  # Verify generation
    python scripts/generate_proto.py --clean   # Remove generated files

Author: Matthew Hong
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

PROTO_DIR = PROJECT_ROOT / "src" / "shared" / "proto"
PROTO_FILE = PROTO_DIR / "inference.proto"

GENERATED_FILES = [
    PROTO_DIR / "inference_pb2.py",
    PROTO_DIR / "inference_pb2_grpc.py",
]


# =============================================================================
# Functions
# =============================================================================

def check_dependencies() -> bool:
    """Check if grpcio-tools is installed."""
    try:
        import grpc_tools.protoc
        return True
    except ImportError:
        return False


def generate_proto() -> bool:
    """
    Generate Python files from proto definition.
    
    Returns:
        True if generation successful
    """
    if not PROTO_FILE.exists():
        logger.error(f"Proto file not found: {PROTO_FILE}")
        return False

    if not check_dependencies():
        logger.error("grpcio-tools not installed.")
        logger.error("Install with: pip install grpcio-tools")
        return False

    logger.info(f"Generating proto files...")
    logger.info(f"  Source: {PROTO_FILE}")
    logger.info(f"  Output: {PROTO_DIR}")

    try:
        from grpc_tools import protoc

        # Run protoc with grpc plugin
        result = protoc.main([
            "grpc_tools.protoc",
            f"--proto_path={PROTO_DIR}",
            f"--python_out={PROTO_DIR}",
            f"--grpc_python_out={PROTO_DIR}",
            str(PROTO_FILE),
        ])

        if result != 0:
            logger.error(f"protoc failed with code {result}")
            return False

        # Fix imports in generated files (use relative imports)
        fix_imports()

        # Verify files were created
        for f in GENERATED_FILES:
            if f.exists():
                logger.info(f"  ✓ Generated: {f.name}")
            else:
                logger.error(f"  ✗ Missing: {f.name}")
                return False

        return True

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return False


def fix_imports() -> None:
    """
    Fix imports in generated files to use relative imports.
    
    The generated files use absolute imports which don't work
    when the package is installed. This fixes them to use
    relative imports.
    """
    grpc_file = PROTO_DIR / "inference_pb2_grpc.py"
    
    if not grpc_file.exists():
        return

    content = grpc_file.read_text()
    
    old_import = "import inference_pb2 as inference__pb2"
    new_import = "from . import inference_pb2 as inference__pb2"
    
    if old_import in content:
        content = content.replace(old_import, new_import)
        grpc_file.write_text(content)
        logger.info("  ✓ Fixed imports in inference_pb2_grpc.py")


def verify_proto() -> bool:
    """
    Verify proto files are correctly generated.
    
    Returns:
        True if all files valid
    """
    logger.info("Verifying proto files...")

    # Check files exist
    for f in GENERATED_FILES:
        if not f.exists():
            logger.error(f"  ✗ Missing: {f}")
            return False
        logger.info(f"  ✓ Exists: {f.name}")

    # Try importing
    try:
        from shared.proto import inference_pb2
        from shared.proto import inference_pb2_grpc

        # List available message types
        messages = [
            name for name in dir(inference_pb2)
            if not name.startswith("_") and name[0].isupper()
        ]
        logger.info(f"  ✓ Messages: {', '.join(messages[:5])}...")

        # List available services
        services = [
            name for name in dir(inference_pb2_grpc)
            if name.endswith("Stub") or name.endswith("Servicer")
        ]
        logger.info(f"  ✓ Services: {', '.join(services)}")

        # Test creating a message
        request = inference_pb2.ClassificationRequest(
            request_id="test-123",
            image_crop=b"fake-image-data",
        )
        logger.info(f"  ✓ Created test message: ClassificationRequest")

        # Test serialization
        serialized = request.SerializeToString()
        deserialized = inference_pb2.ClassificationRequest()
        deserialized.ParseFromString(serialized)
        assert deserialized.request_id == "test-123"
        logger.info(f"  ✓ Serialization/deserialization works")

        return True

    except ImportError as e:
        logger.error(f"  ✗ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"  ✗ Verification failed: {e}")
        return False


def clean_proto() -> bool:
    """
    Remove generated proto files.
    
    Returns:
        True if cleanup successful
    """
    logger.info("Cleaning generated proto files...")

    for f in GENERATED_FILES:
        if f.exists():
            f.unlink()
            logger.info(f"  ✓ Removed: {f.name}")
        else:
            logger.info(f"  ⊘ Not found: {f.name}")

    # Also remove __pycache__ if present
    pycache = PROTO_DIR / "__pycache__"
    if pycache.exists():
        import shutil
        shutil.rmtree(pycache)
        logger.info(f"  ✓ Removed: __pycache__")

    return True


def print_header() -> None:
    """Print script header."""
    print()
    print("=" * 60)
    print("Inference Arena - Proto Generation")
    print("=" * 60)
    print(f"  Proto file: {PROTO_FILE}")
    print(f"  Output dir: {PROTO_DIR}")
    print()


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Python files from proto definition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_proto.py           # Generate proto files
  python scripts/generate_proto.py --verify  # Verify generation
  python scripts/generate_proto.py --clean   # Remove generated files
        """,
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify proto files without generating",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove generated proto files",
    )

    args = parser.parse_args()

    print_header()

    if args.clean:
        success = clean_proto()
    elif args.verify:
        success = verify_proto()
    else:
        success = generate_proto()
        if success:
            print()
            success = verify_proto()

    print()
    print("=" * 60)
    if success:
        print("✓ Complete")
    else:
        print("✗ Failed")
    print("=" * 60)
    print()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
