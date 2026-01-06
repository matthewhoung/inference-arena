"""Locust load testing for Inference Arena.

This module provides load testing capabilities for the three ML serving
architectures (Monolithic, Microservices, Triton) using Locust.

Test images are loaded from the curated thesis test set (100 COCO images
with 3-5 detections each, μ=4, σ≈0.8).

Features:
    - Automatic stats reset after warmup period (60s)
    - Configurable load levels with spawn rates from experiment.yaml
    - CSV output for analysis

Usage:
    # Start architecture first (manually)
    make start-mono  # or start-micro, start-triton

    # Quick test (10 users, 30 seconds)
    locust -f experiments/locustfile.py --host=http://localhost:8100 \
           --headless -u 10 -r 3 -t 30s

    # Full run with CSV output (240s = 60s warmup + 180s measurement)
    locust -f experiments/locustfile.py --host=http://localhost:8100 \
           --headless -u 10 -r 3 -t 240s \
           --csv=results/raw/mono_10users_run1

    # With Web UI (for debugging)
    locust -f experiments/locustfile.py --host=http://localhost:8100
    # Open http://localhost:8089

Architecture Ports:
    - Monolithic:    http://localhost:8100
    - Microservices: http://localhost:8200
    - Triton:        http://localhost:8300

Load Levels (from experiment.yaml):
    Users   Spawn Rate
    1       1/sec
    5       2/sec
    10      3/sec
    25      5/sec
    50      10/sec
    75      15/sec
    100     20/sec

Author: Matthew Hong
Specification Reference: experiment.yaml, Ch3_Methodology_v4.md
"""

import random
import threading
from pathlib import Path

from locust import HttpUser, between, events, task

# =============================================================================
# Configuration
# =============================================================================

# Path to test images (curated thesis dataset)
DATA_DIR = Path(__file__).parent.parent / "data" / "thesis_test_set"

# Warmup duration in seconds (from experiment.yaml load_testing.phases.warmup)
WARMUP_DURATION_SECONDS = 60

# Load level to spawn rate mapping (from Ch3_Methodology)
SPAWN_RATES = {
    1: 1,
    5: 2,
    10: 3,
    25: 5,
    50: 10,
    75: 15,
    100: 20,
}

# Track warmup state
_warmup_complete = False
_stats_reset_timer: threading.Timer | None = None


# =============================================================================
# Locust User Definition
# =============================================================================


class InferenceUser(HttpUser):
    """Simulates a user sending inference requests.

    Each user:
    1. Loads test images on startup
    2. Sends random images to /predict endpoint
    3. Waits 0.1-0.5s between requests (think time)
    """

    wait_time = between(0.1, 0.5)  # Think time between requests

    def on_start(self) -> None:
        """Load test images when user starts."""
        self.images = list(DATA_DIR.glob("*.jpg"))
        if not self.images:
            raise RuntimeError(
                f"No test images found in {DATA_DIR}. "
                "Run 'python scripts/setup/download-data.py' first."
            )

    @task
    def predict(self) -> None:
        """Send prediction request with random image."""
        image_path = random.choice(self.images)

        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/jpeg")}
            self.client.post("/predict", files=files)


# =============================================================================
# Event Handlers
# =============================================================================


def _reset_stats(environment) -> None:
    """Reset statistics after warmup period."""
    global _warmup_complete
    _warmup_complete = True

    print("\n" + "=" * 60)
    print(f"WARMUP COMPLETE ({WARMUP_DURATION_SECONDS}s)")
    print("Resetting statistics for measurement phase...")
    print("=" * 60 + "\n")

    # Reset all stats
    environment.runner.stats.reset_all()


@events.test_start.add_listener
def on_test_start(environment, **kwargs) -> None:
    """Log test configuration and schedule stats reset after warmup."""
    global _warmup_complete, _stats_reset_timer
    _warmup_complete = False

    image_count = len(list(DATA_DIR.glob("*.jpg")))
    run_time = environment.parsed_options.run_time

    print("\n" + "=" * 60)
    print("Inference Arena - Load Test Starting")
    print("=" * 60)
    print(f"  Host:        {environment.host}")
    print(f"  Test images: {image_count} (from {DATA_DIR.name}/)")
    print(f"  Users:       {environment.parsed_options.num_users}")
    print(f"  Spawn rate:  {environment.parsed_options.spawn_rate}/sec")
    print(f"  Duration:    {run_time}")
    print(f"  Warmup:      {WARMUP_DURATION_SECONDS}s (stats will reset)")
    print("=" * 60 + "\n")

    # Schedule stats reset after warmup
    # Only if run_time > warmup duration
    if run_time:
        # Parse run_time (e.g., "240s", "4m", "1h")
        run_seconds = _parse_time_string(run_time)
        if run_seconds and run_seconds > WARMUP_DURATION_SECONDS:
            _stats_reset_timer = threading.Timer(
                WARMUP_DURATION_SECONDS, _reset_stats, args=[environment]
            )
            _stats_reset_timer.daemon = True
            _stats_reset_timer.start()
            print(f"[INFO] Stats reset scheduled in {WARMUP_DURATION_SECONDS}s")


def _parse_time_string(time_str: str | int | None) -> int | None:
    """Parse Locust time string to seconds (e.g., '240s', '4m', '1h')."""
    if time_str is None:
        return None

    # If already an int, return it directly
    if isinstance(time_str, int):
        return time_str

    time_str = str(time_str).strip().lower()

    try:
        if time_str.endswith("s"):
            return int(time_str[:-1])
        elif time_str.endswith("m"):
            return int(time_str[:-1]) * 60
        elif time_str.endswith("h"):
            return int(time_str[:-1]) * 3600
        else:
            # Assume seconds if no suffix
            return int(time_str)
    except ValueError:
        return None


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs) -> None:
    """Log summary at test completion."""
    global _stats_reset_timer

    # Cancel timer if test stopped early
    if _stats_reset_timer and _stats_reset_timer.is_alive():
        _stats_reset_timer.cancel()

    print("\n" + "=" * 60)
    print("Load Test Complete")
    if _warmup_complete:
        print("(Statistics reflect measurement phase only)")
    else:
        print("(Warning: Test ended before warmup completed)")
    print("=" * 60 + "\n")
