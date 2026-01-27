#!/usr/bin/env python3
"""LOC counting script for RQ3 operational complexity analysis.

Counts logical LOC (code-only, excluding comments/docstrings/blanks) across
all three architectures (Monolithic, Microservices, Triton) using pygount.

Scoping rules per 03-CONTEXT.md:
- Application code: production services + dependencies
- Configuration: Dockerfile, docker-compose.yml, .env, .pbtxt
- Shared code counted by actual dependency
- Excludes: tests, auto-generated *_pb2.py, init scripts
"""

import csv
import json
import subprocess
from pathlib import Path


def count_loc_with_pygount(path, suffix):
    """Count logical LOC using pygount.

    Args:
        path: Path to count (file or directory)
        suffix: File extensions to include (comma-separated, e.g., "py" or "yml,yaml,env")

    Returns:
        dict: {'code': int, 'files': int} where code is logical LOC
    """
    cmd = ["pygount", "--format", "json", "--suffix", suffix, str(path)]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    # Extract totalCodeCount (logical LOC) from summary
    return {
        "code": data["summary"].get("totalCodeCount", 0),
        "files": data["summary"].get("totalFileCount", 0),
    }


def count_monolithic_loc(base_path):
    """Count LOC for Monolithic architecture.

    Application scope: architectures/monolithic/app/ (all .py files)
    Configuration scope: docker-compose.yml, Dockerfile, .env
    Exclusions: init_monolith_models.py (development script)
    """
    base = Path(base_path) / "architectures" / "monolithic"

    # Application code: app/ directory only
    app_result = count_loc_with_pygount(base / "app", "py")

    # Configuration: yml and env files in monolithic root
    config_result = count_loc_with_pygount(base, "yml,yaml,env")

    return {"architecture": "monolithic", "application": app_result, "configuration": config_result}


def count_microservices_loc(base_path):
    """Count LOC for Microservices architecture.

    Application scope:
    - architectures/microservices/detection/app/
    - architectures/microservices/classification/app/
    - src/shared/proto/inference.proto (proto definition, not generated)

    Configuration scope: docker-compose.yml, .env files
    Exclusions: init_microservices_models.py, *_pb2.py (auto-generated)
    """
    base = Path(base_path) / "architectures" / "microservices"
    proto_base = Path(base_path) / "src" / "shared" / "proto"

    # Application code: detection + classification apps
    detection_result = count_loc_with_pygount(base / "detection" / "app", "py")
    classification_result = count_loc_with_pygount(base / "classification" / "app", "py")

    # Shared proto definitions (exclude generated *_pb2.py by counting .proto source)
    proto_result = count_loc_with_pygount(proto_base / "inference.proto", "proto")

    # Sum application LOC
    total_app_loc = detection_result["code"] + classification_result["code"] + proto_result["code"]
    total_app_files = (
        detection_result["files"] + classification_result["files"] + proto_result["files"]
    )

    # Configuration: yml and env files
    config_result = count_loc_with_pygount(base, "yml,yaml,env")

    return {
        "architecture": "microservices",
        "application": {"code": total_app_loc, "files": total_app_files},
        "configuration": config_result,
    }


def count_triton_loc(base_path):
    """Count LOC for Triton architecture.

    Application scope:
    - architectures/triton/gateway/app/
    - scripts/models/generate-pbtxt.py (Triton-specific operational overhead)

    Configuration scope: docker-compose.yml, .env, .pbtxt files
    Exclusions: init_triton_models.py (development script)
    """
    base = Path(base_path) / "architectures" / "triton"
    scripts_base = Path(base_path) / "scripts" / "models"

    # Application code: gateway app
    gateway_result = count_loc_with_pygount(base / "gateway" / "app", "py")

    # Triton-specific: pbtxt generator (in shared/ but Triton-only dependency)
    pbtxt_gen_result = count_loc_with_pygount(scripts_base / "generate-pbtxt.py", "py")

    # Sum application LOC
    total_app_loc = gateway_result["code"] + pbtxt_gen_result["code"]
    total_app_files = gateway_result["files"] + pbtxt_gen_result["files"]

    # Configuration: yml, env, and pbtxt files
    config_result = count_loc_with_pygount(base, "yml,yaml,env")

    return {
        "architecture": "triton",
        "application": {"code": total_app_loc, "files": total_app_files},
        "configuration": config_result,
    }


def main():
    """Count LOC for all architectures and write to CSV.

    Output: results/metrics/loc_counts.csv
    Format: architecture,category,loc,files
    """
    base_path = Path(__file__).parent.parent.parent

    # Count for each architecture
    monolithic = count_monolithic_loc(base_path)
    microservices = count_microservices_loc(base_path)
    triton = count_triton_loc(base_path)

    # Prepare CSV data
    csv_data = []

    for arch_data in [monolithic, microservices, triton]:
        arch = arch_data["architecture"]

        # Application code row
        csv_data.append(
            {
                "architecture": arch,
                "category": "application",
                "loc": arch_data["application"]["code"],
                "files": arch_data["application"]["files"],
            }
        )

        # Configuration code row
        csv_data.append(
            {
                "architecture": arch,
                "category": "configuration",
                "loc": arch_data["configuration"]["code"],
                "files": arch_data["configuration"]["files"],
            }
        )

    # Write to CSV
    output_dir = base_path / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "loc_counts.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["architecture", "category", "loc", "files"])
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"LOC counts written to: {output_path}")
    print("\nSummary:")
    for row in csv_data:
        print(
            f"  {row['architecture']:15} {row['category']:15} {row['loc']:5} LOC ({row['files']} files)"
        )


if __name__ == "__main__":
    main()
