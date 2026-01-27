"""
Thesis Analysis Module

Provides utilities and notebooks for analyzing ML serving architecture experiments.

Modules:
    utilities/loaders: Data loading utilities for summary.csv and aggregate.json

Notebooks:
    notebooks/01_rq1_performance.ipynb: Performance analysis (H1a-H1d)
    notebooks/02_rq2_resource_efficiency.ipynb: Resource efficiency (H2a-H2d)
    notebooks/03_rq3_cost_complexity.ipynb: Operational complexity (H3a-H3c)
"""

from analysis.utilities.loaders import ResultsLoader
from shared.config.loader import get_config

# Re-export get_config as load_experiment_config for backward compatibility
load_experiment_config = get_config

__all__ = ["ResultsLoader", "load_experiment_config", "get_config"]
