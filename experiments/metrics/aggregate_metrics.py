#!/usr/bin/env python3
"""Aggregate all RQ3 measurements into comprehensive summary table.

This script combines:
- LOC counts (application + configuration)
- File counts (Python + config files)
- Deployment times (mean + std across runs)
- API endpoint counts

Produces:
- CSV summary with absolute values and relative ratios
- LaTeX booktabs-formatted table for thesis integration

Author: RQ3 Data Collection (03-03)
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd  # noqa: E402

from analysis.utilities.loaders import ResultsLoader  # noqa: E402

# Load architectures from experiment.yaml via ResultsLoader
ResultsLoader._load_config_from_yaml()
ARCHITECTURES = ResultsLoader.ARCHITECTURES


def load_and_validate_csvs(metrics_dir: Path) -> dict:
    """Load all metrics measurement CSVs and validate structure.

    Args:
        metrics_dir: Path to results/metrics directory

    Returns:
        Dictionary of dataframes keyed by measurement type

    Raises:
        FileNotFoundError: If any required CSV is missing
        ValueError: If data validation fails
    """
    required_files = {
        "loc": "loc_counts.csv",
        "files": "file_counts.csv",
        "deployment": "deployment_times.csv",
        "endpoints": "api_endpoints.csv",
    }

    data = {}
    missing = []

    for key, filename in required_files.items():
        filepath = metrics_dir / filename
        if not filepath.exists():
            missing.append(filename)
        else:
            data[key] = pd.read_csv(filepath)

    if missing:
        raise FileNotFoundError(f"Missing required CSV files: {', '.join(missing)}")

    # Validate expected row counts (using architectures from experiment.yaml)
    architectures = ARCHITECTURES

    # LOC: 2 rows per architecture (application + configuration)
    expected_loc_rows = len(architectures) * 2
    if len(data["loc"]) != expected_loc_rows:
        print(f"Warning: Expected {expected_loc_rows} LOC rows, got {len(data['loc'])}")

    # Files: varies by architecture (extensions differ)
    min_file_rows = len(architectures)
    if len(data["files"]) < min_file_rows:
        raise ValueError(
            f"Expected at least {min_file_rows} file count rows, got {len(data['files'])}"
        )

    # Deployment: 3 runs per architecture (but Triton may have only 2 if run 3 failed)
    expected_deployment_rows = len(architectures) * 3
    if len(data["deployment"]) < (len(architectures) * 2):
        raise ValueError(
            f"Expected at least {len(architectures) * 2} deployment rows, got {len(data['deployment'])}"
        )
    if len(data["deployment"]) < expected_deployment_rows:
        print(
            f"Warning: Expected {expected_deployment_rows} deployment rows, got {len(data['deployment'])} (some runs may be missing)"
        )

    # Endpoints: 1 row per architecture
    if len(data["endpoints"]) != len(architectures):
        raise ValueError(
            f"Expected {len(architectures)} endpoint rows, got {len(data['endpoints'])}"
        )

    return data


def aggregate_summary(data: dict) -> pd.DataFrame:
    """Aggregate all measurements into summary table.

    Args:
        data: Dictionary of measurement dataframes

    Returns:
        Summary dataframe with all RQ3 metrics
    """
    # Use architectures from experiment.yaml
    architectures = ARCHITECTURES
    summary = []

    for arch in architectures:
        row = {"architecture": arch}

        # 1. LOC metrics (from loc_counts.csv)
        loc_data = data["loc"][data["loc"]["architecture"] == arch]
        app_loc = loc_data[loc_data["category"] == "application"]["loc"].sum()
        config_loc = loc_data[loc_data["category"] == "configuration"]["loc"].sum()
        total_loc = app_loc + config_loc

        row["app_code_loc"] = int(app_loc)
        row["config_loc"] = int(config_loc)
        row["total_loc"] = int(total_loc)

        # 2. File counts (from file_counts.csv)
        file_data = data["files"][data["files"]["architecture"] == arch]
        python_files = file_data[file_data["extension"] == "py"]["count"].sum()

        # Config files: yml, proto (for microservices), env, pbtxt
        config_extensions = ["yml", "proto", "env", "pbtxt"]
        config_files = file_data[file_data["extension"].isin(config_extensions)]["count"].sum()

        row["python_files"] = int(python_files)
        row["config_files"] = int(config_files)

        # 3. API endpoints (from api_endpoints.csv)
        endpoint_data = data["endpoints"][data["endpoints"]["architecture"] == arch]
        endpoint_count = endpoint_data["endpoint_count"].values[0]
        row["api_endpoints"] = int(endpoint_count)

        # 4. Deployment time statistics (from deployment_times.csv)
        deployment_data = data["deployment"][data["deployment"]["architecture"] == arch]
        times = deployment_data["total_time_seconds"].values

        if len(times) > 0:
            row["deployment_time_mean_s"] = float(times.mean())
            row["deployment_time_std_s"] = float(times.std())
        else:
            row["deployment_time_mean_s"] = 0.0
            row["deployment_time_std_s"] = 0.0

        summary.append(row)

    df = pd.DataFrame(summary)

    # 5. Calculate relative ratios vs Monolithic baseline
    monolithic_loc = df[df["architecture"] == "monolithic"]["total_loc"].values[0]
    monolithic_deployment = df[df["architecture"] == "monolithic"]["deployment_time_mean_s"].values[
        0
    ]

    df["loc_ratio_vs_mono"] = df["total_loc"] / monolithic_loc
    df["deployment_ratio_vs_mono"] = df["deployment_time_mean_s"] / monolithic_deployment

    # Set architecture as index
    df = df.set_index("architecture")

    return df


def generate_latex_table(df: pd.DataFrame, output_path: Path) -> None:
    """Generate booktabs-formatted LaTeX table.

    Args:
        df: Summary dataframe
        output_path: Path to write .tex file
    """
    # Prepare table for LaTeX output
    # Rename columns for better LaTeX display
    display_df = df.copy()
    display_df.columns = [
        "App LOC",
        "Config LOC",
        "Total LOC",
        "Python Files",
        "Config Files",
        "API Endpoints",
        "Deploy Time (s)",
        "Deploy Std (s)",
        "LOC Ratio",
        "Deploy Ratio",
    ]

    # Generate LaTeX with pandas
    latex_output = display_df.to_latex(
        float_format="%.1f",
        escape=False,
        column_format="l" + "r" * len(display_df.columns),
    )

    # Add booktabs package requirement comment
    header = (
        "% Requires \\usepackage{booktabs} in LaTeX preamble\n"
        "% RQ3: Operational Complexity Comparison Across Architectures\n"
        "% Generated by: experiments/metrics/aggregate_metrics.py\n\n"
    )

    # Add caption and label
    # Find the \begin{tabular} line and insert caption/label before it
    lines = latex_output.split("\n")
    table_start_idx = next(i for i, line in enumerate(lines) if "\\begin{tabular}" in line)

    # Insert table environment with caption
    lines.insert(table_start_idx, "\\begin{table}[htbp]")
    lines.insert(table_start_idx + 1, "\\centering")
    lines.insert(
        table_start_idx + 2,
        "\\caption{RQ3: Operational Complexity Comparison Across Architectures}",
    )
    lines.insert(table_start_idx + 3, "\\label{tab:rq3-operational-complexity}")

    # Find the \end{tabular} line and insert table end after it
    table_end_idx = next(i for i, line in enumerate(lines) if "\\end{tabular}" in line)
    lines.insert(table_end_idx + 1, "\\end{table}")

    final_latex = header + "\n".join(lines)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_latex)


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    metrics_dir = project_root / "results/metrics"

    print("Loading metrics measurement CSVs...")
    try:
        data = load_and_validate_csvs(metrics_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("✓ All CSV files loaded successfully")
    print(f"  - LOC counts: {len(data['loc'])} rows")
    print(f"  - File counts: {len(data['files'])} rows")
    print(f"  - Deployment times: {len(data['deployment'])} rows")
    print(f"  - API endpoints: {len(data['endpoints'])} rows")

    print("\nAggregating summary table...")
    summary_df = aggregate_summary(data)

    print("\n" + "=" * 80)
    print("RQ3 OPERATIONAL COMPLEXITY SUMMARY")
    print("=" * 80)
    print(summary_df.to_string())
    print("=" * 80)

    # Summary statistics
    print("\nKey Statistics:")
    print(f"  LOC range: {summary_df['total_loc'].min()} - {summary_df['total_loc'].max()}")
    print(
        f"  Deployment time range: {summary_df['deployment_time_mean_s'].min():.1f}s - {summary_df['deployment_time_mean_s'].max():.1f}s"
    )
    print(
        f"  API endpoint range: {summary_df['api_endpoints'].min()} - {summary_df['api_endpoints'].max()}"
    )

    # Write CSV output
    csv_output = metrics_dir / "metrics_summary.csv"
    summary_df.to_csv(csv_output)
    print(f"\n✓ Summary CSV written to {csv_output}")

    # Generate LaTeX table
    tables_dir = project_root / "analysis/tables"
    latex_output = tables_dir / "rq3_summary.tex"
    generate_latex_table(summary_df, latex_output)
    print(f"✓ LaTeX table written to {latex_output}")

    print("\n✓ RQ3 data aggregation complete!")


if __name__ == "__main__":
    main()
