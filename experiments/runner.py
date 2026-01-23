"""Experiment runner for Inference Arena load testing.

This module provides the CLI for orchestrating load testing experiments
across the three ML serving architectures.

Features:
    - Click-based CLI with flexible options
    - Docker orchestration (start/stop architectures)
    - Health check polling before tests
    - Prometheus integration for resource metrics
    - Progress logging and estimated completion time

Usage:
    # Full experiment matrix (63 experiments, ~4.7 hours)
    python -m experiments.runner

    # Single architecture
    python -m experiments.runner -a monolithic

    # Specific configuration (debugging)
    python -m experiments.runner -a microservices -u 10 -r 1 --no-docker

    # Dry run (show plan without executing)
    python -m experiments.runner --dry-run

Author: Matthew Hong
Specification Reference: experiment.yaml, .claude/LOADTESTING.md
"""

import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

import click

from shared.health import HealthCheckTimeoutError, wait_for_healthy

from .config import (
    ARCHITECTURE_ENDPOINTS,
    COMPOSE_FILES,
    PROJECT_ROOT,
    get_phase_durations,
    get_runs_per_configuration,
    get_spawn_rate,
    get_total_duration,
    get_user_levels,
)
from .metrics import reset_collector
from .results import PrometheusClient, ResultsCollector, ResultsExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Cooldown times
COOLDOWN_BETWEEN_RUNS = 10  # seconds
COOLDOWN_BETWEEN_ARCH = 30  # seconds


@click.command()
@click.option(
    "-a",
    "--architecture",
    multiple=True,
    type=click.Choice(["monolithic", "microservices", "triton"]),
    help="Architecture(s) to test. Can be repeated. Default: all",
)
@click.option(
    "-u",
    "--users",
    multiple=True,
    type=int,
    help="User level(s) to test. Can be repeated. Default: all (1,5,10,25,50,75,100)",
)
@click.option(
    "-r",
    "--runs",
    default=None,
    type=int,
    help="Runs per configuration. Default: from experiment.yaml (3)",
)
@click.option(
    "--no-docker",
    is_flag=True,
    help="Skip docker-compose orchestration (assume containers are running)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show experiment plan without executing",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Results output directory. Default: results/experiment/",
)
@click.option(
    "--no-prometheus",
    is_flag=True,
    help="Skip Prometheus resource metrics collection",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    architecture: tuple[str, ...],
    users: tuple[int, ...],
    runs: int | None,
    no_docker: bool,
    dry_run: bool,
    output_dir: str | None,
    no_prometheus: bool,
    verbose: bool,
) -> None:
    """Run load testing experiments for Inference Arena.

    This CLI orchestrates the full experiment matrix across architectures,
    load levels, and multiple runs. Results are exported to JSON and CSV
    formats for analysis.

    Examples:
        # Full experiment matrix
        python -m experiments.runner

        # Single architecture, all loads
        python -m experiments.runner -a monolithic

        # Multiple architectures
        python -m experiments.runner -a monolithic -a microservices

        # Specific load levels
        python -m experiments.runner -a triton -u 1 -u 10 -u 50

        # Quick test (1 run, no docker)
        python -m experiments.runner -a monolithic -u 10 -r 1 --no-docker

        # Dry run to see plan
        python -m experiments.runner --dry-run
    """
    # Configure logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve parameters
    archs = list(architecture) if architecture else list(ARCHITECTURE_ENDPOINTS.keys())
    user_levels = list(users) if users else get_user_levels()
    num_runs = runs if runs is not None else get_runs_per_configuration()

    # Initialize exporter
    exporter = ResultsExporter(output_dir) if output_dir else ResultsExporter()

    # Initialize Prometheus client
    prometheus = None if no_prometheus else PrometheusClient()

    # Calculate totals
    total_experiments = len(archs) * len(user_levels) * num_runs
    duration_per_run = get_total_duration()
    cooldown_per_run = COOLDOWN_BETWEEN_RUNS
    estimated_time = total_experiments * (duration_per_run + cooldown_per_run)

    # Print plan
    click.echo()
    click.echo("=" * 70)
    click.echo("INFERENCE ARENA - EXPERIMENT RUNNER")
    click.echo("=" * 70)
    click.echo(f"  Architectures:      {archs}")
    click.echo(f"  User levels:        {user_levels}")
    click.echo(f"  Runs per config:    {num_runs}")
    click.echo(f"  Total experiments:  {total_experiments}")
    click.echo(f"  Duration per run:   {duration_per_run}s")
    click.echo(f"  Estimated time:     {_format_duration(estimated_time)}")
    click.echo(f"  Docker orchestration: {'No' if no_docker else 'Yes'}")
    click.echo(f"  Prometheus metrics:   {'No' if no_prometheus else 'Yes'}")
    click.echo(f"  Output directory:   {exporter.output_dir}")
    click.echo("=" * 70)
    click.echo()

    if dry_run:
        _print_dry_run_plan(archs, user_levels, num_runs)
        return

    # Validate user levels
    valid_levels = get_user_levels()
    for u in user_levels:
        if u not in valid_levels:
            click.echo(f"Error: Invalid user level {u}. Valid: {valid_levels}", err=True)
            sys.exit(1)

    # Run experiments
    all_results: list[dict[str, Any]] = []
    current = 0
    start_time = time.time()

    try:
        for arch in archs:
            click.echo()
            click.echo("#" * 70)
            click.echo(f"# ARCHITECTURE: {arch.upper()}")
            click.echo("#" * 70)

            # Start architecture (if docker orchestration enabled)
            if not no_docker:
                if not _start_architecture(arch):
                    click.echo(f"Failed to start {arch}, skipping...", err=True)
                    continue

            # Check health
            endpoint = ARCHITECTURE_ENDPOINTS[arch]
            if not _wait_for_health(endpoint):
                click.echo(f"{arch} not healthy, skipping...", err=True)
                if not no_docker:
                    _stop_architecture(arch)
                continue

            # Run experiments for this architecture
            for user_count in user_levels:
                for run_num in range(1, num_runs + 1):
                    current += 1
                    click.echo()
                    click.echo(f"[{current}/{total_experiments}] ", nl=False)
                    click.echo(f"{arch} - {user_count} users - run {run_num}")

                    # Run Locust experiment
                    result = _run_locust_experiment(
                        architecture=arch,
                        user_count=user_count,
                        run_number=run_num,
                        prometheus=prometheus,
                    )

                    if result:
                        all_results.append(result)
                        exporter.export_run(result)
                        _print_result_summary(result)
                    else:
                        click.echo("  [FAILED] Experiment did not produce results")

                    # Cooldown between runs
                    if current < total_experiments:
                        click.echo(f"  Cooldown: {COOLDOWN_BETWEEN_RUNS}s...")
                        time.sleep(COOLDOWN_BETWEEN_RUNS)

            # Stop architecture
            if not no_docker:
                _stop_architecture(arch)

            # Cooldown between architectures
            if arch != archs[-1] and not no_docker:
                click.echo(f"\nArchitecture cooldown: {COOLDOWN_BETWEEN_ARCH}s...")
                time.sleep(COOLDOWN_BETWEEN_ARCH)

    except KeyboardInterrupt:
        click.echo("\n\n[INTERRUPTED] Experiment runner cancelled by user")
        if not no_docker and archs:
            click.echo("Stopping any running architectures...")
            for arch in archs:
                _stop_architecture(arch)

    # Export summaries (regenerate from ALL existing runs, not just current)
    if all_results:
        click.echo()
        click.echo("=" * 70)
        click.echo("EXPORTING RESULTS")
        click.echo("=" * 70)
        csv_path, json_path = exporter.regenerate_summaries()
        click.echo(f"  CSV summary:  {csv_path}")
        click.echo(f"  Aggregated:   {json_path}")

    # Final summary
    elapsed = time.time() - start_time
    click.echo()
    click.echo("=" * 70)
    click.echo("EXPERIMENT SUMMARY")
    click.echo("=" * 70)
    click.echo(f"  Completed:    {len(all_results)} / {total_experiments}")
    click.echo(f"  Elapsed time: {_format_duration(elapsed)}")
    click.echo("=" * 70)


def _run_locust_experiment(
    architecture: str,
    user_count: int,
    run_number: int,
    prometheus: PrometheusClient | None,
) -> dict[str, Any] | None:
    """Run a single Locust experiment.

    Args:
        architecture: Architecture name
        user_count: Number of concurrent users
        run_number: Run number (1, 2, 3, ...)
        prometheus: PrometheusClient for resource metrics

    Returns:
        Result dictionary or None if failed
    """
    endpoint = ARCHITECTURE_ENDPOINTS[architecture]
    spawn_rate = get_spawn_rate(user_count)
    duration = get_total_duration()
    durations = get_phase_durations()

    # Set environment for Locust
    env = os.environ.copy()
    env["LOCUST_USERS"] = str(user_count)

    # Create temp file for metrics export from Locust subprocess
    # Use project's results/tmp/ instead of system /tmp/
    import tempfile

    tmp_dir = PROJECT_ROOT / "results" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="locust_metrics_", dir=str(tmp_dir)
    )
    metrics_file.close()
    env["LOCUST_METRICS_FILE"] = metrics_file.name

    # Build Locust command (use sys.executable to ensure correct Python/venv)
    locustfile = PROJECT_ROOT / "experiments" / "locustfile.py"
    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(locustfile),
        f"--host={endpoint}",
        "--headless",
        f"-u={user_count}",
        f"-r={spawn_rate}",
        f"-t={duration}s",
        "--only-summary",
    ]

    click.echo(f"  Host:       {endpoint}")
    click.echo(f"  Spawn rate: {spawn_rate}/sec")
    click.echo(
        f"  Duration:   {duration}s (warmup={durations['warmup']}s, measurement={durations['measurement']}s, cooldown={durations['cooldown']}s)"
    )

    # Record measurement phase timestamps
    measurement_start = time.time() + durations["warmup"]
    measurement_end = measurement_start + durations["measurement"]

    # Reset metrics collector
    reset_collector()

    try:
        # Run Locust
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

        # Log warning if Locust exited with error, but continue to collect metrics
        # This handles cases like Triton at 100 users where requests are still
        # in-flight when test duration expires (queue saturation scenario)
        if result.returncode != 0:
            click.echo(f"  [WARNING] Locust exited with code {result.returncode}")
            if result.stderr:
                click.echo(f"  stderr: {result.stderr[:200]}")
            # Continue anyway - metrics may still have been collected

        # Read metrics from file exported by Locust (even if Locust had non-zero exit)
        import json

        locust_stats = None
        phase_summary = None
        try:
            with open(metrics_file.name) as f:
                data = json.load(f)
                locust_stats = data.get("stats")
                phase_summary = data.get("phase_summary")
        except Exception as e:
            click.echo(f"  [WARNING] Could not read metrics file: {e}")
        finally:
            # Clean up temp file
            try:
                os.unlink(metrics_file.name)
            except Exception:
                pass

        # Collect results using the stats from Locust
        collector = ResultsCollector()
        experiment_result = collector.collect_from_stats(
            stats=locust_stats,
            phase_summary=phase_summary,
            prometheus=prometheus,
            architecture=architecture,
            user_count=user_count,
            run_number=run_number,
            measurement_start=measurement_start,
            measurement_end=measurement_end,
        )

        return experiment_result

    except Exception as e:
        click.echo(f"  [ERROR] {e}")
        return None


def _start_architecture(arch: str) -> bool:
    """Start architecture using docker-compose.

    Args:
        arch: Architecture name

    Returns:
        True if started successfully
    """
    compose_file = COMPOSE_FILES.get(arch)
    if not compose_file or not compose_file.exists():
        click.echo(f"Compose file not found: {compose_file}", err=True)
        return False

    click.echo(f"Starting {arch}...")
    try:
        subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "up", "-d"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        click.echo(f"Failed to start {arch}: {e.stderr.decode()[:200]}", err=True)
        return False


def _stop_architecture(arch: str) -> None:
    """Stop architecture using docker-compose.

    Args:
        arch: Architecture name
    """
    compose_file = COMPOSE_FILES.get(arch)
    if not compose_file:
        return

    click.echo(f"Stopping {arch}...")
    try:
        subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "down"],
            cwd=PROJECT_ROOT,
            capture_output=True,
        )
    except Exception:
        pass


def _wait_for_health(endpoint: str, timeout: float = 120.0) -> bool:
    """Wait for architecture health endpoint.

    Args:
        endpoint: Base URL (e.g., http://localhost:8100)
        timeout: Maximum wait time in seconds

    Returns:
        True if healthy, raises on timeout
    """
    health_url = f"{endpoint}/health"
    click.echo(f"Waiting for {health_url}...")

    def check_health() -> bool:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                return response.status == 200
        except (urllib.error.URLError, TimeoutError):
            return False

    # Extract service name from URL for logging
    # e.g., "http://localhost:8100" -> "localhost:8100"
    service_name = urlparse(endpoint).netloc

    try:
        wait_for_healthy(
            service_name,
            check_health,
            initial_delay=1.0,
            max_wait=timeout,
            backoff_multiplier=2.0,
            max_interval=5.0,
        )
        click.echo("  Health check passed!")
        return True
    except HealthCheckTimeoutError as e:
        click.echo(f"  Health check timeout: {e}", err=True)
        return False


def _print_dry_run_plan(
    archs: list[str],
    user_levels: list[int],
    num_runs: int,
) -> None:
    """Print experiment plan for dry run."""
    click.echo("DRY RUN - Experiment Plan:")
    click.echo()

    for arch in archs:
        click.echo(f"  {arch.upper()}:")
        for users in user_levels:
            for run in range(1, num_runs + 1):
                click.echo(f"    - {arch}_users{users}_run{run}")
        click.echo()


def _print_result_summary(result: dict[str, Any]) -> None:
    """Print summary of experiment result."""
    click.echo(f"  Throughput: {result['throughput_rps']} RPS")
    click.echo(f"  Error rate: {result['error_rate_percent']}%")

    if result.get("client_latency"):
        cl = result["client_latency"]
        click.echo(f"  Client P99: {cl['p99_ms']}ms")

    if result.get("server_latency"):
        sl = result["server_latency"]
        click.echo(f"  Server P99: {sl['p99_ms']}ms")


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


# Entry point for python -m experiments.runner
if __name__ == "__main__":
    main()
