#!/usr/bin/env python3
"""File counting script for RQ3 operational complexity analysis.

Counts files by extension across all three architectures (Monolithic,
Microservices, Triton) using pathlib for recursive pattern matching.

File categories per 03-CONTEXT.md:
- Python source: .py files (per application code scope rules)
- Configuration: .yml, .yaml, .env files
- Triton-specific: .pbtxt files (model configuration)

Scoping rules:
- Count by actual dependency (shared code counted per architecture use)
- Exclude: tests, auto-generated *_pb2.py, development scripts
"""

import csv
from collections import Counter
from pathlib import Path


def count_files_by_extension(paths, extensions, exclude_patterns=None):
    """Count files by extension across multiple paths.

    Args:
        paths: List of Path objects to search
        extensions: List of file extensions to count (without dot)
        exclude_patterns: List of patterns to exclude (e.g., ['*_pb2.py', 'test_*'])

    Returns:
        Counter: {extension: count}
    """
    exclude_patterns = exclude_patterns or []
    counts = Counter()

    for path in paths:
        if not path.exists():
            continue

        for ext in extensions:
            # Recursive glob for extension
            for file_path in path.rglob(f"*.{ext}"):
                if not file_path.is_file():
                    continue

                # Apply exclusion patterns
                excluded = False
                for pattern in exclude_patterns:
                    if file_path.match(pattern):
                        excluded = True
                        break

                if not excluded:
                    counts[ext] += 1

    return counts


def count_monolithic_files(base_path):
    """Count files for Monolithic architecture.

    Scope:
    - Python: architectures/monolithic/app/*.py
    - Config: architectures/monolithic/*.yml, *.yaml, *.env
    - Exclude: init_monolith_models.py (development script)
    """
    base = Path(base_path) / "architectures" / "monolithic"

    # Python application files
    app_path = base / "app"
    py_counts = count_files_by_extension(
        [app_path], ["py"], exclude_patterns=["test_*.py", "*_test.py"]
    )

    # Configuration files
    config_counts = count_files_by_extension([base], ["yml", "yaml", "env"], exclude_patterns=[])

    # Merge counts
    all_counts = py_counts + config_counts

    return {ext: all_counts.get(ext, 0) for ext in ["py", "yml", "yaml", "env"]}


def count_microservices_files(base_path):
    """Count files for Microservices architecture.

    Scope:
    - Python: detection/app/*.py, classification/app/*.py, proto/*.proto
    - Config: *.yml, *.yaml, *.env
    - Exclude: init_microservices_models.py, *_pb2.py (generated)
    """
    base = Path(base_path) / "architectures" / "microservices"
    proto_base = Path(base_path) / "src" / "shared" / "proto"

    # Python application files from both services
    detection_path = base / "detection" / "app"
    classification_path = base / "classification" / "app"

    py_counts = count_files_by_extension(
        [detection_path, classification_path],
        ["py"],
        exclude_patterns=["test_*.py", "*_test.py", "*_pb2.py", "*_pb2_grpc.py"],
    )

    # Proto source files (exclude generated)
    proto_counts = count_files_by_extension([proto_base], ["proto"], exclude_patterns=[])

    # Configuration files
    config_counts = count_files_by_extension([base], ["yml", "yaml", "env"], exclude_patterns=[])

    # Merge all counts
    all_counts = py_counts + proto_counts + config_counts

    return {ext: all_counts.get(ext, 0) for ext in ["py", "proto", "yml", "yaml", "env"]}


def count_triton_files(base_path):
    """Count files for Triton architecture.

    Scope:
    - Python: gateway/app/*.py, scripts/models/generate-pbtxt.py
    - Config: *.yml, *.yaml, *.env
    - Triton-specific: .pbtxt files (if generated/stored)
    - Exclude: init_triton_models.py (development script)
    """
    base = Path(base_path) / "architectures" / "triton"
    scripts_base = Path(base_path) / "scripts" / "models"

    # Python application files
    gateway_path = base / "gateway" / "app"
    pbtxt_gen_path = scripts_base / "generate-pbtxt.py"

    py_counts = count_files_by_extension(
        [gateway_path], ["py"], exclude_patterns=["test_*.py", "*_test.py"]
    )

    # Add generate-pbtxt.py (single file, manual count)
    if pbtxt_gen_path.exists() and pbtxt_gen_path.is_file():
        py_counts["py"] += 1

    # Configuration files
    config_counts = count_files_by_extension([base], ["yml", "yaml", "env"], exclude_patterns=[])

    # Note: .pbtxt files are generated at runtime, not stored in repo
    # Count the generator script as the operational overhead instead

    # Merge counts
    all_counts = py_counts + config_counts

    return {ext: all_counts.get(ext, 0) for ext in ["py", "yml", "yaml", "env"]}


def main():
    """Count files for all architectures and write to CSV.

    Output: results/metrics/file_counts.csv
    Format: architecture,extension,count
    """
    base_path = Path(__file__).parent.parent.parent

    # Count for each architecture
    monolithic = count_monolithic_files(base_path)
    microservices = count_microservices_files(base_path)
    triton = count_triton_files(base_path)

    # Prepare CSV data
    csv_data = []

    # Monolithic rows
    for ext, count in monolithic.items():
        if count > 0:
            csv_data.append({"architecture": "monolithic", "extension": ext, "count": count})

    # Microservices rows
    for ext, count in microservices.items():
        if count > 0:
            csv_data.append({"architecture": "microservices", "extension": ext, "count": count})

    # Triton rows
    for ext, count in triton.items():
        if count > 0:
            csv_data.append({"architecture": "triton", "extension": ext, "count": count})

    # Write to CSV
    output_dir = base_path / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "file_counts.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["architecture", "extension", "count"])
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"File counts written to: {output_path}")
    print("\nSummary:")
    for row in csv_data:
        print(f"  {row['architecture']:15} .{row['extension']:6} {row['count']:3} files")


if __name__ == "__main__":
    main()
