#!/usr/bin/env python3
"""Generate Triton config.pbtxt files from experiment.yaml.

This CLI tool generates NVIDIA Triton Inference Server configuration files
based on model specifications defined in experiment.yaml.

Usage:
    # Generate all configs to stdout (non-batched)
    python scripts/models/generate-pbtxt.py --print

    # Generate batching-enabled configs
    python scripts/models/generate-pbtxt.py --print --batched

    # Generate specific model config
    python scripts/models/generate-pbtxt.py --model yolov5n --print

    # Save configs to directory
    python scripts/models/generate-pbtxt.py --output-dir models/triton

    # Generate both non-batched and batched configs
    python scripts/models/generate-pbtxt.py --all-variants --output-dir models/triton

Author: Matthew Hong
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shared.triton.config import (  # noqa: E402
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
    parser.add_argument(
        "--batched",
        action="store_true",
        help="Generate batching-enabled config (model names will have _batched suffix)",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        dest="all_variants",
        help="Generate both non-batched and batched variants for all models",
    )

    args = parser.parse_args()

    # Determine models to process
    base_models = ["yolov5n", "mobilenetv2"] if args.model == "all" else [args.model]

    # Build list of (model_name, batching_enabled) tuples
    configs_to_generate = []
    if args.all_variants:
        # Generate both variants for each model
        for model in base_models:
            configs_to_generate.append((model, False))
            configs_to_generate.append((f"{model}_batched", True))
    elif args.batched:
        # Generate batched variants only
        for model in base_models:
            configs_to_generate.append((f"{model}_batched", True))
    else:
        # Generate non-batched variants only (default)
        for model in base_models:
            configs_to_generate.append((model, False))

    # Default output directory
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "models" / "triton"

    exit_code = 0

    for model_name, batching_enabled in configs_to_generate:
        config = generate_config_pbtxt(model_name, batching_enabled)

        if args.validate:
            errors = validate_config_pbtxt(config)
            if errors:
                print(f"Validation errors for {model_name}:")
                for error in errors:
                    print(f"  - {error}")
                exit_code = 1
                continue
            batching_status = "(batched)" if batching_enabled else "(non-batched)"
            print(f"[OK] {model_name}: validation passed {batching_status}")

        if args.print_config:
            print()
            print("=" * 60)
            batching_status = "BATCHED" if batching_enabled else "NON-BATCHED"
            print(f"# {model_name}/config.pbtxt [{batching_status}]")
            print("=" * 60)
            print(config)
        elif not args.validate:
            output_path = save_config_pbtxt(model_name, args.output_dir, batching_enabled)
            batching_status = "(batched)" if batching_enabled else "(non-batched)"
            print(f"[OK] Generated: {output_path} {batching_status}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
