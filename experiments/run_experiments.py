#!/usr/bin/env python3
"""Automated experiment runner for Inference Arena.

This script orchestrates the full load testing experiment across all
architectures and load levels, producing CSV results for analysis.

Experiment Configuration (from experiment.yaml):
    - Architectures: Monolithic (8100), Microservices (8200), Triton (8300)
    - Load Levels: [1, 5, 10, 25, 50, 75, 100] concurrent users
    - Runs per configuration: 3
    - Duration: 240s (60s warmup + 180s measurement)

Prerequisites:
    1. Docker and docker-compose installed
    2. Infrastructure running: make start-infra
    3. Models initialized: make models-init-minio
    4. Locust installed: uv pip install locust

Usage:
    # Run all experiments (full matrix)
    python experiments/run_experiments.py

    # Run specific architecture
    python experiments/run_experiments.py --arch mono

    # Run specific load levels
    python experiments/run_experiments.py --users 10 25 50

    # Dry run (show commands without executing)
    python experiments/run_experiments.py --dry-run

    # Continue from a specific point
    python experiments/run_experiments.py --arch micro --users 25 --run 2

Author: Matthew Hong
Specification Reference: experiment.yaml, Ch3_Methodology_v4.md
"""

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Project root (parent of experiments/)
PROJECT_ROOT = Path(__file__).parent.parent

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results" / "raw"

# Architecture configurations
ARCHITECTURES = {
    "mono": {
        "name": "Monolithic",
        "port": 8100,
        "start_cmd": "make start-mono",
        "stop_cmd": "make stop-mono",
    },
    "micro": {
        "name": "Microservices",
        "port": 8200,
        "start_cmd": "make start-micro",
        "stop_cmd": "make stop-micro",
    },
    "triton": {
        "name": "Triton",
        "port": 8300,
        "start_cmd": "make start-triton",
        "stop_cmd": "make stop-triton",
    },
}

# Load levels and spawn rates (from experiment.yaml)
LOAD_LEVELS = {
    1: 1,
    5: 2,
    10: 3,
    25: 5,
    50: 10,
    75: 15,
    100: 20,
}

# Test duration (60s warmup + 180s measurement)
TEST_DURATION_SECONDS = 240

# Runs per configuration
RUNS_PER_CONFIG = 3

# Cooldown between runs (seconds)
COOLDOWN_BETWEEN_RUNS = 10

# Cooldown between architectures (seconds)
COOLDOWN_BETWEEN_ARCH = 30


# =============================================================================
# Helper Functions
# =============================================================================


def ensure_results_dir() -> None:
    """Create results directory if it doesn't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results directory: {RESULTS_DIR}")


def check_architecture_health(port: int, timeout: int = 30) -> bool:
    """Check if architecture is healthy and responding."""
    url = f"http://localhost:{port}/health"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)

    return False


def run_locust_test(
    arch_key: str, users: int, run_number: int, dry_run: bool = False
) -> bool:
    """Run a single Locust test and save results."""
    arch = ARCHITECTURES[arch_key]
    spawn_rate = LOAD_LEVELS[users]
    host = f"http://localhost:{arch['port']}"

    # Generate output filename
    # Format: mono_10users_run1
    csv_prefix = RESULTS_DIR / f"{arch_key}_{users}users_run{run_number}"

    # Build Locust command
    cmd = [
        "locust",
        "-f",
        str(PROJECT_ROOT / "experiments" / "locustfile.py"),
        f"--host={host}",
        "--headless",
        f"-u={users}",
        f"-r={spawn_rate}",
        f"-t={TEST_DURATION_SECONDS}s",
        f"--csv={csv_prefix}",
        "--csv-full-history",
    ]

    print(f"\n{'=' * 60}")
    print(f"Running: {arch['name']} | {users} users | Run {run_number}")
    print(f"{'=' * 60}")
    print(f"  Host:       {host}")
    print(f"  Spawn rate: {spawn_rate}/sec")
    print(f"  Duration:   {TEST_DURATION_SECONDS}s")
    print(f"  Output:     {csv_prefix}_stats.csv")

    if dry_run:
        print(f"  [DRY RUN] Command: {' '.join(cmd)}")
        return True

    # Run Locust
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=False,
            text=True,
        )
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Test cancelled by user")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run Locust: {e}")
        return False


def run_experiment_matrix(
    architectures: list[str],
    user_levels: list[int],
    start_run: int = 1,
    dry_run: bool = False,
    skip_health_check: bool = False,
) -> dict:
    """Run the full experiment matrix."""
    results = {"success": [], "failed": [], "skipped": []}

    total_tests = len(architectures) * len(user_levels) * (RUNS_PER_CONFIG - start_run + 1)
    current_test = 0

    print("\n" + "=" * 70)
    print("INFERENCE ARENA - EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"  Architectures:  {architectures}")
    print(f"  User levels:    {user_levels}")
    print(f"  Runs per config: {start_run} to {RUNS_PER_CONFIG}")
    print(f"  Total tests:    {total_tests}")
    print(f"  Dry run:        {dry_run}")
    print("=" * 70)

    for arch_key in architectures:
        arch = ARCHITECTURES[arch_key]
        print(f"\n\n{'#' * 70}")
        print(f"# Architecture: {arch['name']} (port {arch['port']})")
        print(f"{'#' * 70}")

        # Check health before running tests
        if not dry_run and not skip_health_check:
            print(f"\n[INFO] Checking {arch['name']} health...")
            if not check_architecture_health(arch["port"]):
                print(f"[WARNING] {arch['name']} not responding on port {arch['port']}")
                print(f"[INFO] Please start with: {arch['start_cmd']}")

                # Skip all tests for this architecture
                for users in user_levels:
                    for run in range(start_run, RUNS_PER_CONFIG + 1):
                        results["skipped"].append(
                            {"arch": arch_key, "users": users, "run": run}
                        )
                continue

        for users in user_levels:
            for run in range(start_run, RUNS_PER_CONFIG + 1):
                current_test += 1
                print(f"\n[{current_test}/{total_tests}] ", end="")

                success = run_locust_test(arch_key, users, run, dry_run)

                if success:
                    results["success"].append(
                        {"arch": arch_key, "users": users, "run": run}
                    )
                else:
                    results["failed"].append(
                        {"arch": arch_key, "users": users, "run": run}
                    )

                # Cooldown between runs
                if not dry_run and run < RUNS_PER_CONFIG:
                    print(f"[INFO] Cooldown: {COOLDOWN_BETWEEN_RUNS}s...")
                    time.sleep(COOLDOWN_BETWEEN_RUNS)

            # Reset start_run for subsequent user levels
            if start_run > 1:
                start_run = 1

        # Cooldown between architectures
        if not dry_run and arch_key != architectures[-1]:
            print(f"\n[INFO] Architecture cooldown: {COOLDOWN_BETWEEN_ARCH}s...")
            time.sleep(COOLDOWN_BETWEEN_ARCH)

    return results


def print_summary(results: dict) -> None:
    """Print experiment summary."""
    print("\n\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  Successful: {len(results['success'])}")
    print(f"  Failed:     {len(results['failed'])}")
    print(f"  Skipped:    {len(results['skipped'])}")

    if results["failed"]:
        print("\n  Failed tests:")
        for test in results["failed"]:
            print(f"    - {test['arch']}_{test['users']}users_run{test['run']}")

    if results["skipped"]:
        print("\n  Skipped tests (architecture not available):")
        for test in results["skipped"]:
            print(f"    - {test['arch']}_{test['users']}users_run{test['run']}")

    print("=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Inference Arena load testing experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all experiments
    python experiments/run_experiments.py

    # Run only monolithic architecture
    python experiments/run_experiments.py --arch mono

    # Run specific load levels
    python experiments/run_experiments.py --users 10 25 50

    # Dry run (preview commands)
    python experiments/run_experiments.py --dry-run

    # Skip health checks (assume architecture is running)
    python experiments/run_experiments.py --skip-health-check
        """,
    )

    parser.add_argument(
        "--arch",
        nargs="+",
        choices=list(ARCHITECTURES.keys()),
        default=list(ARCHITECTURES.keys()),
        help="Architectures to test (default: all)",
    )

    parser.add_argument(
        "--users",
        nargs="+",
        type=int,
        choices=list(LOAD_LEVELS.keys()),
        default=list(LOAD_LEVELS.keys()),
        help="User levels to test (default: all)",
    )

    parser.add_argument(
        "--run",
        type=int,
        choices=range(1, RUNS_PER_CONFIG + 1),
        default=1,
        help="Starting run number (default: 1)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without executing",
    )

    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip architecture health checks",
    )

    args = parser.parse_args()

    # Ensure results directory exists
    ensure_results_dir()

    # Run experiments
    try:
        results = run_experiment_matrix(
            architectures=args.arch,
            user_levels=args.users,
            start_run=args.run,
            dry_run=args.dry_run,
            skip_health_check=args.skip_health_check,
        )
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Experiment runner cancelled by user")
        return 1

    # Print summary
    print_summary(results)

    # Return exit code
    if results["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
