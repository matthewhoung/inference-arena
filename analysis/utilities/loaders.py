"""
Data Loading Utilities for Thesis Analysis

Provides clean interfaces to load experimental results from:
- summary.csv: Flat tabular data for pandas operations
- aggregate.json: Pre-computed mean/std by architecture and user count
- Individual run JSONs: For statistical hypothesis testing

Usage:
    from analysis.utilities.loaders import ResultsLoader

    loader = ResultsLoader()
    df = loader.load_summary()
    agg = loader.load_aggregate()
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Optional

from shared.config.loader import get_config


class ResultsLoader:
    """Load and normalize experimental results for analysis."""

    # Architecture display names for plots (presentation only)
    ARCH_DISPLAY_NAMES = {
        "monolithic": "Monolithic",
        "microservices": "Microservices",
        "triton": "Triton",
    }

    # Standard color palette for consistent visualizations (presentation only)
    ARCH_COLORS = {
        "monolithic": "#2ecc71",      # Green
        "microservices": "#3498db",   # Blue
        "triton": "#e74c3c",          # Red
    }

    # Configuration loaded from experiment.yaml (set by _load_config_from_yaml)
    VCPU_ALLOCATION: dict[str, int] = {}
    USER_LEVELS: list[int] = []
    SATURATION_THRESHOLD_MS: int = 500
    ARCHITECTURES: list[str] = []
    CONTAINER_COUNTS: dict[str, int] = {}

    _config_loaded = False

    @classmethod
    def _load_config_from_yaml(cls) -> None:
        """Load configuration values from experiment.yaml."""
        if cls._config_loaded:
            return

        config = get_config()

        # Load architectures from independent_variables
        cls.ARCHITECTURES = config["independent_variables"]["architecture"]["levels"]

        # Load user levels from independent_variables
        cls.USER_LEVELS = config["independent_variables"]["concurrent_users"]["levels"]

        # Load vCPU allocation from controlled_variables.resources
        resources = config["controlled_variables"]["resources"]
        cls.VCPU_ALLOCATION = {
            arch: resources[arch]["total_vcpu"]
            for arch in cls.ARCHITECTURES
        }

        # Load container counts from controlled_variables.resources
        cls.CONTAINER_COUNTS = {
            arch: resources[arch]["containers"]
            for arch in cls.ARCHITECTURES
        }

        # Load saturation threshold from hypotheses.H1d
        cls.SATURATION_THRESHOLD_MS = config["hypotheses"]["H1d"]["saturation_threshold_ms"]

        cls._config_loaded = True

    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialize loader with results directory path.

        Args:
            results_dir: Path to results/experiment. Defaults to project root.
        """
        # Ensure config is loaded from experiment.yaml
        self._load_config_from_yaml()

        if results_dir is None:
            # Find project root by looking for experiment.yaml
            current = Path(__file__).parent
            while current != current.parent:
                if (current / "experiment.yaml").exists():
                    results_dir = current / "results" / "experiment"
                    break
                current = current.parent
            else:
                raise FileNotFoundError("Could not find project root with experiment.yaml")

        self.results_dir = Path(results_dir)
        self.runs_dir = self.results_dir / "runs"
        self._validate_paths()

    def _validate_paths(self) -> None:
        """Validate that required data files exist."""
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")

        if not self.runs_dir.exists():
            raise FileNotFoundError(f"Runs directory not found: {self.runs_dir}")

        csv_path = self.results_dir / "summary.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"summary.csv not found: {csv_path}")

    def load_summary(self) -> pd.DataFrame:
        """
        Load summary.csv into a pandas DataFrame.

        Returns:
            DataFrame with columns:
            - architecture, concurrent_users, run_number, timestamp
            - throughput_rps, error_rate_percent
            - client_p50_ms, client_p95_ms, client_p99_ms
            - server_p50_ms, server_p95_ms, server_p99_ms
            - cpu_avg_percent, cpu_max_percent
            - memory_avg_mb, memory_max_mb
            - network_rx_bytes_per_sec, network_tx_bytes_per_sec
        """
        df = pd.read_csv(self.results_dir / "summary.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def load_aggregate(self) -> dict:
        """
        Load pre-aggregated results with mean/std by architecture and users.

        Returns:
            Dict with structure:
            {
                "architectures": {
                    "monolithic": {
                        "1": {"throughput": {"mean_rps", "std_rps"}, ...},
                        "5": {...},
                        ...
                    }
                }
            }
        """
        with open(self.results_dir / "aggregate.json") as f:
            return json.load(f)

    def load_all_runs(self) -> list[dict]:
        """
        Load all individual run JSON files.

        Returns:
            List of dicts, each containing full run details including
            per-container resource metrics.
        """
        runs = []
        for json_file in sorted(self.runs_dir.glob("**/run_*.json")):
            with open(json_file) as f:
                runs.append(json.load(f))
        return runs

    def get_aggregated_df(self) -> pd.DataFrame:
        """
        Convert aggregate.json to DataFrame with mean values per config.

        Returns:
            DataFrame indexed by (architecture, concurrent_users) with
            mean values for all metrics.
        """
        df = self.load_summary()

        # Group by architecture and concurrent_users, compute mean
        agg_df = df.groupby(["architecture", "concurrent_users"]).agg({
            "throughput_rps": ["mean", "std"],
            "error_rate_percent": "mean",
            "client_p50_ms": ["mean", "std"],
            "client_p95_ms": ["mean", "std"],
            "client_p99_ms": ["mean", "std"],
            "server_p50_ms": ["mean", "std"],
            "server_p95_ms": ["mean", "std"],
            "server_p99_ms": ["mean", "std"],
            "cpu_avg_percent": ["mean", "std"],
            "cpu_max_percent": "mean",
            "memory_avg_mb": ["mean", "std"],
            "memory_max_mb": "mean",
            "network_rx_bytes_per_sec": ["mean", "std"],
            "network_tx_bytes_per_sec": ["mean", "std"],
        })

        # Flatten column names
        agg_df.columns = ["_".join(col).strip("_") for col in agg_df.columns]

        return agg_df.reset_index()

    def compute_efficiency_metrics(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute derived efficiency metrics for RQ2 analysis.

        Adds columns:
        - total_vcpu: Total vCPU allocation for architecture
        - throughput_per_vcpu: Requests per second per vCPU
        - cpu_efficiency: Actual CPU usage / allocated CPU

        Args:
            df: Input DataFrame. If None, loads summary.csv.

        Returns:
            DataFrame with efficiency metrics added.
        """
        if df is None:
            df = self.load_summary()

        df = df.copy()

        # Add vCPU allocation
        df["total_vcpu"] = df["architecture"].map(self.VCPU_ALLOCATION)

        # Throughput efficiency: RPS per vCPU
        df["throughput_per_vcpu"] = df["throughput_rps"] / df["total_vcpu"]

        # CPU efficiency: actual usage vs allocation (200% = 2 vCPU)
        df["cpu_efficiency"] = df["cpu_avg_percent"] / (df["total_vcpu"] * 100)

        return df

    def get_latency_variance(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute latency variance metrics for H1c analysis.

        Adds:
        - p99_p50_gap_ms: P99 - P50 latency (variance indicator)

        Args:
            df: Input DataFrame. If None, loads summary.csv.

        Returns:
            DataFrame with latency variance metrics.
        """
        if df is None:
            df = self.load_summary()

        df = df.copy()
        df["p99_p50_gap_ms"] = df["client_p99_ms"] - df["client_p50_ms"]

        return df

    def get_throughput_at_load(self, concurrent_users: int) -> dict[str, float]:
        """
        Get mean throughput for each architecture at a specific load level.

        Args:
            concurrent_users: The concurrent user count to query.

        Returns:
            Dict mapping architecture name to mean throughput (RPS).
        """
        df = self.load_summary()
        filtered = df[df["concurrent_users"] == concurrent_users]
        throughput_by_arch = filtered.groupby("architecture")["throughput_rps"].mean()
        return throughput_by_arch.to_dict()
