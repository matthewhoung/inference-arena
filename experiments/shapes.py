"""Load test shapes for Locust.

This module provides the ThreePhaseShape class that implements the
three-phase testing protocol: warmup → measurement → cooldown.

The shape maintains a constant user count throughout all phases,
with fast spawning at the start to reach target capacity quickly.

Usage:
    # In locustfile.py
    from experiments.shapes import ThreePhaseShape

    # ThreePhaseShape is automatically used when running via runner.py

Author: Matthew Hong
Specification Reference: experiment.yaml controlled_variables.load_testing
"""

import logging
import os
from typing import Literal

from locust import LoadTestShape

from .config import get_phase_durations, get_spawn_rate

logger = logging.getLogger(__name__)

# Phase names
PhaseType = Literal["warmup", "measurement", "cooldown"]


class ThreePhaseShape(LoadTestShape):
    """Constant user count across warmup/measurement/cooldown phases.

    This load test shape implements the three-phase protocol specified in
    the thesis methodology:

    1. Warmup (60s): Prime JIT, ONNX optimizations, CPU caches
    2. Measurement (180s): Collect performance metrics under steady state
    3. Cooldown (30s): System reset, garbage collection

    The shape maintains a constant user count throughout all phases.
    Users are spawned quickly at the start (spawn_rate = users × 10).

    Attributes:
        user_count: Target number of concurrent users
        warmup: Warmup phase duration in seconds
        measurement: Measurement phase duration in seconds
        cooldown: Cooldown phase duration in seconds
        total_duration: Total test duration in seconds

    Environment Variables:
        LOCUST_USERS: Override user count (set by runner.py)

    Example:
        # The shape is automatically detected by Locust when imported
        # in locustfile.py:
        from experiments.shapes import ThreePhaseShape

        # Or explicitly:
        shape = ThreePhaseShape()
        phase = shape.get_current_phase()  # "warmup", "measurement", or "cooldown"
    """

    def __init__(self):
        """Initialize the three-phase shape.

        Reads phase durations from experiment.yaml and user count from
        environment variable (set by runner.py).
        """
        super().__init__()

        # Get phase durations from config
        durations = get_phase_durations()
        self.warmup = durations["warmup"]
        self.measurement = durations["measurement"]
        self.cooldown = durations["cooldown"]
        self.total_duration = self.warmup + self.measurement + self.cooldown

        # User count from environment (set by runner) or default to 1
        self.user_count = int(os.environ.get("LOCUST_USERS", "1"))

        # Spawn rate: fast spawning to reach target quickly
        # Use 10x users or minimum from config
        try:
            self._spawn_rate = get_spawn_rate(self.user_count)
        except ValueError:
            # Fallback for non-standard user counts
            self._spawn_rate = max(self.user_count, 1)

        # Track phase transitions for logging
        self._last_phase: PhaseType | None = None
        self._phase_start_time: float | None = None

        logger.info(
            f"ThreePhaseShape initialized: {self.user_count} users, "
            f"phases=[{self.warmup}s/{self.measurement}s/{self.cooldown}s]"
        )

    def tick(self) -> tuple[int, float] | None:
        """Return the current user count and spawn rate.

        This method is called periodically by Locust to determine
        the desired number of users and spawn rate.

        Returns:
            Tuple of (user_count, spawn_rate) or None to stop the test
        """
        run_time = self.get_run_time()

        # Check if test should end
        if run_time >= self.total_duration:
            logger.info("ThreePhaseShape: Test complete, stopping")
            return None

        # Log phase transitions
        current_phase = self.get_current_phase()
        if current_phase != self._last_phase:
            self._log_phase_transition(current_phase, run_time)
            self._last_phase = current_phase

        # Return constant user count with fast spawn rate
        # Use high spawn rate initially, then maintain
        return (self.user_count, self._spawn_rate)

    def get_current_phase(self) -> PhaseType:
        """Get the current test phase based on elapsed time.

        Returns:
            Current phase: "warmup", "measurement", or "cooldown"
        """
        run_time = self.get_run_time()

        if run_time < self.warmup:
            return "warmup"
        elif run_time < self.warmup + self.measurement:
            return "measurement"
        else:
            return "cooldown"

    def get_phase_progress(self) -> dict:
        """Get detailed progress information for the current phase.

        Returns:
            Dictionary with phase info and progress
        """
        run_time = self.get_run_time()
        phase = self.get_current_phase()

        if phase == "warmup":
            phase_elapsed = run_time
            phase_remaining = self.warmup - run_time
            phase_duration = self.warmup
        elif phase == "measurement":
            phase_elapsed = run_time - self.warmup
            phase_remaining = (self.warmup + self.measurement) - run_time
            phase_duration = self.measurement
        else:
            phase_elapsed = run_time - self.warmup - self.measurement
            phase_remaining = self.total_duration - run_time
            phase_duration = self.cooldown

        return {
            "phase": phase,
            "phase_elapsed_seconds": round(phase_elapsed, 1),
            "phase_remaining_seconds": round(max(0, phase_remaining), 1),
            "phase_duration_seconds": phase_duration,
            "phase_progress_percent": round((phase_elapsed / phase_duration) * 100, 1),
            "total_elapsed_seconds": round(run_time, 1),
            "total_remaining_seconds": round(max(0, self.total_duration - run_time), 1),
            "total_progress_percent": round((run_time / self.total_duration) * 100, 1),
        }

    def is_measurement_phase(self) -> bool:
        """Check if currently in measurement phase.

        Returns:
            True if in measurement phase, False otherwise
        """
        return self.get_current_phase() == "measurement"

    def get_measurement_window(self) -> tuple[float, float]:
        """Get the time window for measurement phase.

        Returns:
            Tuple of (start_offset, end_offset) in seconds from test start
        """
        start = self.warmup
        end = self.warmup + self.measurement
        return (start, end)

    def _log_phase_transition(self, new_phase: PhaseType, run_time: float) -> None:
        """Log phase transition for debugging.

        Args:
            new_phase: The new phase being entered
            run_time: Current test run time in seconds
        """
        phase_info = {
            "warmup": f"Warmup ({self.warmup}s) - Prime JIT, caches",
            "measurement": f"Measurement ({self.measurement}s) - Collecting metrics",
            "cooldown": f"Cooldown ({self.cooldown}s) - System reset",
        }

        logger.info(
            f"\n{'='*60}\n"
            f"PHASE TRANSITION: {new_phase.upper()}\n"
            f"{phase_info[new_phase]}\n"
            f"Time: {run_time:.1f}s / {self.total_duration}s\n"
            f"{'='*60}"
        )


# Module-level singleton for accessing shape state
_shape_instance: ThreePhaseShape | None = None


def get_shape() -> ThreePhaseShape | None:
    """Get the active shape instance.

    This is set by Locust when the shape is used. Returns None if
    no shape is active (e.g., running without LoadTestShape).

    Returns:
        Active ThreePhaseShape instance or None
    """
    return _shape_instance


def set_shape(shape: ThreePhaseShape) -> None:
    """Set the active shape instance.

    Called internally when Locust initializes the shape.

    Args:
        shape: ThreePhaseShape instance to set as active
    """
    global _shape_instance
    _shape_instance = shape
