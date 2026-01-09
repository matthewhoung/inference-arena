#!/usr/bin/env python3
"""Update Grafana dashboards with current Docker container IDs.

This script discovers running container IDs and updates the Grafana
dashboard JSON files to use the correct container_id values.

Usage:
    python update-dashboards.py

Author: Matthew Hong
"""

import json
import re
import subprocess
import sys
from pathlib import Path


def get_container_id(name_filter: str) -> str | None:
    """Get container ID by name filter."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={name_filter}", "--format", "{{.ID}}"],
            capture_output=True,
            text=True,
            check=True,
        )
        container_id = result.stdout.strip()
        return container_id if container_id else None
    except subprocess.CalledProcessError:
        return None


def update_container_ids_in_expr(content: str, old_pattern: str, new_id: str) -> str:
    """Replace container_id in PromQL expressions."""
    # Pattern matches container_id="<12-char-hex>" in JSON
    pattern = rf'container_id=\\"[a-f0-9]{{12}}\\"'
    replacement = f'container_id=\\"{new_id}\\"'
    return re.sub(pattern, replacement, content)


def update_dashboard_by_legend(
    dashboard_path: Path,
    updates: dict[str, str],
) -> bool:
    """Update dashboard container IDs based on legendFormat.

    Args:
        dashboard_path: Path to dashboard JSON file
        updates: Dict mapping legendFormat to new container_id

    Returns:
        True if updated, False if no changes made
    """
    if not dashboard_path.exists():
        return False

    with open(dashboard_path) as f:
        dashboard = json.load(f)

    modified = False

    # Walk through all panels and their targets
    for panel in dashboard.get("panels", []):
        for target in panel.get("targets", []):
            legend = target.get("legendFormat", "")
            if legend in updates:
                new_id = updates[legend]
                expr = target.get("expr", "")

                # Replace any 12-char hex container_id with the new one
                new_expr = re.sub(
                    r'container_id="[a-f0-9]{12}"',
                    f'container_id="{new_id}"',
                    expr,
                )

                if new_expr != expr:
                    target["expr"] = new_expr
                    modified = True

    if modified:
        with open(dashboard_path, "w") as f:
            json.dump(dashboard, f, indent=4)

    return modified


def main():
    # Discover script location and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    dashboard_dir = project_root / "infrastructure" / "grafana" / "provisioning" / "dashboards"

    print("Discovering container IDs...")

    # Get container IDs
    containers = {
        "monolithic": get_container_id("inference-arena-monolithic"),
        "micro-detect": get_container_id("inference-arena-detection"),
        "micro-classify": get_container_id("inference-arena-classification"),
        "triton-gateway": get_container_id("inference-arena-triton-gateway"),
        "triton-server": get_container_id("inference-arena-triton-server"),
    }

    print("\nContainer ID Mapping:")
    for name, cid in containers.items():
        print(f"  {name:20} {cid or '[NOT RUNNING]'}")
    print()

    # Update Monolithic Dashboard
    mono_path = dashboard_dir / "infrastructure-mono.json"
    if containers["monolithic"]:
        updates = {"monolithic": containers["monolithic"]}
        if update_dashboard_by_legend(mono_path, updates):
            print(f"Updated Monolithic dashboard: {containers['monolithic']}")
        else:
            print("Monolithic dashboard: no changes needed")
    else:
        print("Monolithic container not running - dashboard not updated")

    # Update Microservices Dashboard
    micro_path = dashboard_dir / "infrastructure-micro.json"
    if containers["micro-detect"] and containers["micro-classify"]:
        updates = {
            "micro-detect": containers["micro-detect"],
            "micro-classify": containers["micro-classify"],
        }
        if update_dashboard_by_legend(micro_path, updates):
            print(f"Updated Microservices dashboard:")
            print(f"  micro-detect: {containers['micro-detect']}")
            print(f"  micro-classify: {containers['micro-classify']}")
        else:
            print("Microservices dashboard: no changes needed")
    else:
        print("Microservices containers not running - dashboard not updated")

    # Update Triton Dashboard
    triton_path = dashboard_dir / "infrastructure-triton.json"
    if containers["triton-gateway"] and containers["triton-server"]:
        updates = {
            "triton-gateway": containers["triton-gateway"],
            "triton-server": containers["triton-server"],
        }
        if update_dashboard_by_legend(triton_path, updates):
            print(f"Updated Triton dashboard:")
            print(f"  triton-gateway: {containers['triton-gateway']}")
            print(f"  triton-server: {containers['triton-server']}")
        else:
            print("Triton dashboard: no changes needed")
    else:
        print("Triton containers not running - dashboard not updated")

    print("\nDashboard update complete!")
    print("\nNext steps:")
    print("   1. Restart Grafana: docker restart inference-arena-grafana")
    print("   2. Refresh browser to see updated dashboards")


if __name__ == "__main__":
    main()
