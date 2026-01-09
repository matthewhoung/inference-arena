"""Results export for load testing experiments.

This module provides the ResultsExporter class that exports experiment
results to JSON and CSV formats for analysis.

Output Structure:
    results/experiment/
    ├── runs/
    │   ├── monolithic_users10_run1_20260106T120000.json
    │   ├── monolithic_users10_run2_20260106T121500.json
    │   └── ...
    ├── experiment_summary_20260106.csv
    └── experiment_aggregated_20260106.json

Usage:
    from experiments.results import ResultsExporter

    exporter = ResultsExporter()
    exporter.export_run(result)
    exporter.export_summary_csv(all_results)

Author: Matthew Hong
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

# CSV column order
CSV_COLUMNS = [
    "architecture",
    "concurrent_users",
    "run_number",
    "timestamp",
    "total_requests",
    "successful_requests",
    "error_rate_percent",
    "throughput_rps",
    "client_p50_ms",
    "client_p95_ms",
    "client_p99_ms",
    "server_p50_ms",
    "server_p95_ms",
    "server_p99_ms",
    "cpu_avg_percent",
    "cpu_max_percent",
    "memory_avg_mb",
    "memory_max_mb",
]


class ResultsExporter:
    """Export experiment results to JSON and CSV formats.

    This class handles exporting load test results in multiple formats:
    - Individual JSON files per run (detailed)
    - CSV summary of all runs (for pandas analysis)
    - Aggregated JSON grouped by architecture/users

    Attributes:
        output_dir: Base output directory
        runs_dir: Directory for individual run files

    Example:
        >>> exporter = ResultsExporter()
        >>> path = exporter.export_run(result)
        >>> print(f"Saved to: {path}")
    """

    def __init__(self, output_dir: Path | str | None = None):
        """Initialize the results exporter.

        Args:
            output_dir: Base directory for results output.
                        Defaults to results/experiment/
        """
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.runs_dir = self.output_dir / "runs"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ResultsExporter initialized: {self.output_dir}")

    def export_run(self, result: dict[str, Any]) -> Path:
        """Export a single run result to JSON.

        File naming: {arch}_users{N}_run{R}_{timestamp}.json

        Args:
            result: Result dictionary from ResultsCollector.collect()

        Returns:
            Path to the exported JSON file
        """
        # Generate filename
        arch = result["architecture"]
        users = result["concurrent_users"]
        run = result["run_number"]
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

        filename = f"{arch}_users{users}_run{run}_{timestamp}.json"
        filepath = self.runs_dir / filename

        # Write JSON with pretty formatting
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Exported run result: {filepath}")
        return filepath

    def export_summary_csv(
        self,
        results: list[dict[str, Any]],
        filename: str | None = None,
    ) -> Path:
        """Export all results to a CSV summary.

        Args:
            results: List of result dictionaries
            filename: Custom filename (optional)

        Returns:
            Path to the exported CSV file
        """
        if not filename:
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"experiment_summary_{date_str}.csv"

        filepath = self.output_dir / filename

        # Convert results to flat rows
        from .collector import ResultsCollector

        collector = ResultsCollector()
        rows = [collector.to_csv_row(r) for r in results]

        # Write CSV
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Exported CSV summary: {filepath} ({len(rows)} rows)")
        return filepath

    def export_aggregated_json(
        self,
        results: list[dict[str, Any]],
        filename: str | None = None,
    ) -> Path:
        """Export aggregated results grouped by architecture and users.

        Args:
            results: List of result dictionaries
            filename: Custom filename (optional)

        Returns:
            Path to the exported JSON file
        """
        if not filename:
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"experiment_aggregated_{date_str}.json"

        filepath = self.output_dir / filename

        # Group results
        grouped = self._group_results(results)

        # Calculate aggregated statistics
        aggregated = self._aggregate_grouped(grouped)

        # Write JSON
        with open(filepath, "w") as f:
            json.dump(aggregated, f, indent=2, default=str)

        logger.info(f"Exported aggregated results: {filepath}")
        return filepath

    def _group_results(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, dict[int, list[dict[str, Any]]]]:
        """Group results by architecture and user count.

        Returns:
            Nested dict: {arch: {users: [results]}}
        """
        grouped: dict[str, dict[int, list[dict[str, Any]]]] = {}

        for result in results:
            arch = result["architecture"]
            users = result["concurrent_users"]

            if arch not in grouped:
                grouped[arch] = {}
            if users not in grouped[arch]:
                grouped[arch][users] = []

            grouped[arch][users].append(result)

        return grouped

    def _aggregate_grouped(
        self,
        grouped: dict[str, dict[int, list[dict[str, Any]]]],
    ) -> dict[str, Any]:
        """Calculate aggregated statistics for grouped results.

        Args:
            grouped: Nested dict from _group_results

        Returns:
            Aggregated statistics structure
        """
        aggregated = {
            "generated_at": datetime.now().isoformat(),
            "architectures": {},
        }

        for arch, user_results in grouped.items():
            aggregated["architectures"][arch] = {}

            for users, runs in user_results.items():
                # Calculate mean and std for key metrics
                throughputs = [r["throughput_rps"] for r in runs]
                error_rates = [r["error_rate_percent"] for r in runs]

                # Client latency P99s
                client_p99s = [
                    r["client_latency"]["p99_ms"] for r in runs if r.get("client_latency")
                ]

                # Server latency P99s
                server_p99s = [
                    r["server_latency"]["p99_ms"] for r in runs if r.get("server_latency")
                ]

                aggregated["architectures"][arch][users] = {
                    "runs": len(runs),
                    "throughput": {
                        "mean_rps": round(self._mean(throughputs), 2),
                        "std_rps": round(self._std(throughputs), 2),
                        "values": throughputs,
                    },
                    "error_rate": {
                        "mean_percent": round(self._mean(error_rates), 2),
                        "values": error_rates,
                    },
                    "client_latency_p99": {
                        "mean_ms": round(self._mean(client_p99s), 2),
                        "std_ms": round(self._std(client_p99s), 2),
                        "values": client_p99s,
                    },
                    "server_latency_p99": {
                        "mean_ms": round(self._mean(server_p99s), 2),
                        "std_ms": round(self._std(server_p99s), 2),
                        "values": server_p99s,
                    },
                }

        return aggregated

    @staticmethod
    def _mean(values: list[float]) -> float:
        """Calculate mean of values."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _std(values: list[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5

    def get_existing_results(self) -> list[dict[str, Any]]:
        """Load all existing run results from runs directory.

        Returns:
            List of result dictionaries
        """
        results = []

        for filepath in sorted(self.runs_dir.glob("*.json")):
            try:
                with open(filepath) as f:
                    result = json.load(f)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

        logger.info(f"Loaded {len(results)} existing results")
        return results

    def regenerate_summaries(self) -> tuple[Path, Path]:
        """Regenerate summary files from existing run results.

        Returns:
            Tuple of (csv_path, json_path)
        """
        results = self.get_existing_results()

        if not results:
            logger.warning("No existing results found")
            return None, None

        csv_path = self.export_summary_csv(results)
        json_path = self.export_aggregated_json(results)

        return csv_path, json_path
