#!/usr/bin/env python3
"""Count API endpoints across all three architectures.

This script counts the API surface area for operational complexity measurement:
- HTTP endpoints: FastAPI routes (@app.get, @app.post, etc.)
- gRPC endpoints: RPC method definitions in proto files

Outputs CSV with endpoint counts per architecture.

Author: RQ3 Data Collection (03-03)
"""

import csv
import json
import re
from pathlib import Path


def count_fastapi_endpoints(main_py_path: Path) -> tuple[int, list[str]]:
    """Count FastAPI route endpoints via @app decorator pattern.

    Args:
        main_py_path: Path to main.py file with FastAPI app

    Returns:
        Tuple of (count, list of endpoint paths with methods)
    """
    if not main_py_path.exists():
        return 0, []

    content = main_py_path.read_text()

    # Pattern: @app.{method}("{path}")
    # Matches: @app.get("/health"), @app.post("/predict"), etc.
    pattern = r'@app\.(get|post|put|delete|patch)\("([^"]+)"'
    matches = re.findall(pattern, content)

    endpoints = [f"{method.upper()} {path}" for method, path in matches]
    return len(endpoints), endpoints


def count_grpc_methods(proto_path: Path) -> tuple[int, list[str]]:
    """Count gRPC RPC method definitions in proto file.

    Args:
        proto_path: Path to .proto file

    Returns:
        Tuple of (count, list of rpc method names)
    """
    if not proto_path.exists():
        return 0, []

    content = proto_path.read_text()

    # Pattern: rpc MethodName(...) returns (...);
    # Exclude comment lines
    methods = []
    for line in content.splitlines():
        # Skip comment lines
        if line.strip().startswith("//"):
            continue

        # Match: rpc MethodName(Request) returns (Response);
        match = re.search(r"\s+rpc\s+(\w+)\s*\(", line)
        if match:
            methods.append(match.group(1))

    return len(methods), methods


def main():
    """Count API endpoints for all three architectures."""
    project_root = Path(__file__).parent.parent.parent

    results = []

    # 1. Monolithic architecture - FastAPI only
    print("Counting Monolithic endpoints...")
    monolithic_main = project_root / "architectures/monolithic/app/main.py"
    count, endpoints = count_fastapi_endpoints(monolithic_main)
    results.append(
        {
            "architecture": "monolithic",
            "endpoint_count": count,
            "endpoint_details": json.dumps(endpoints),
        }
    )
    print(f"  Found {count} HTTP endpoints: {endpoints}")

    # 2. Microservices architecture - Detection HTTP + Classification gRPC
    print("\nCounting Microservices endpoints...")

    # Detection service HTTP endpoints
    detection_main = project_root / "architectures/microservices/detection/app/main.py"
    detection_count, detection_endpoints = count_fastapi_endpoints(detection_main)
    print(f"  Detection HTTP: {detection_count} endpoints: {detection_endpoints}")

    # Classification service gRPC methods
    proto_file = project_root / "src/shared/proto/inference.proto"
    grpc_count, grpc_methods = count_grpc_methods(proto_file)
    print(f"  Classification gRPC: {grpc_count} methods: {grpc_methods}")

    # Total for microservices
    total_microservices = detection_count + grpc_count
    all_microservices_endpoints = detection_endpoints + [f"RPC {m}" for m in grpc_methods]
    results.append(
        {
            "architecture": "microservices",
            "endpoint_count": total_microservices,
            "endpoint_details": json.dumps(all_microservices_endpoints),
        }
    )
    print(f"  Total: {total_microservices} endpoints")

    # 3. Triton architecture - Gateway HTTP only
    print("\nCounting Triton endpoints...")
    triton_main = project_root / "architectures/triton/gateway/app/main.py"
    count, endpoints = count_fastapi_endpoints(triton_main)
    results.append(
        {
            "architecture": "triton",
            "endpoint_count": count,
            "endpoint_details": json.dumps(endpoints),
        }
    )
    print(f"  Found {count} HTTP endpoints: {endpoints}")

    # Write CSV output
    output_dir = project_root / "results/metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "api_endpoints.csv"

    with output_file.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["architecture", "endpoint_count", "endpoint_details"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ API endpoint counts written to {output_file}")
    print("\nSummary:")
    for result in results:
        print(f"  {result['architecture']}: {result['endpoint_count']} endpoints")


if __name__ == "__main__":
    main()
