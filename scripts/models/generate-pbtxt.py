#!/usr/bin/env python3
"""Generate Triton config.pbtxt files from experiment.yaml.

This CLI tool generates NVIDIA Triton Inference Server configuration files
based on model specifications defined in experiment.yaml.

Usage:
    # Generate all configs to stdout
    python scripts/models/generate-pbtxt.py --print

    # Generate specific model config
    python scripts/models/generate-pbtxt.py --model yolov5n --print

    # Save configs to directory
    python scripts/models/generate-pbtxt.py --output-dir models/triton

Author: Matthew Hong
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shared.triton.config import (
    generate_all_configs,
    generate_config_pbtxt,
    save_config_pbtxt,
    validate_config_pbtxt,
)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Triton config.pbtxt files from experiment.yaml"
    )
    parser.add_argument(
        "--model",
        choices=["yolov5n", "mobilenetv2", "all"],
        default="all",
        help="Model to generate config for (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (saves to {model}/config.pbtxt)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_config",
        help="Print config to stdout instead of saving",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated configs",
    )

    args = parser.parse_args()

    # Determine models to process
    if args.model == "all":
        models = ["yolov5n", "mobilenetv2"]
    else:
        models = [args.model]

    # Default output directory
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "models" / "triton"

    exit_code = 0

    for model_name in models:
        config = generate_config_pbtxt(model_name)

        if args.validate:
            errors = validate_config_pbtxt(config)
            if errors:
                print(f"Validation errors for {model_name}:")
                for error in errors:
                    print(f"  - {error}")
                exit_code = 1
                continue
            print(f"[OK] {model_name}: validation passed")

        if args.print_config:
            print()
            print("=" * 60)
            print(f"# {model_name}/config.pbtxt")
            print("=" * 60)
            print(config)
        elif not args.validate:
            output_path = save_config_pbtxt(model_name, args.output_dir)
            print(f"[OK] Generated: {output_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
