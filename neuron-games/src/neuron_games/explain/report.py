"""`python -m neuron_games explain --run <run_dir>`: explainability report for one run.

OWNER: William.
"""
from __future__ import annotations

from pathlib import Path


def run_report(run_dir: str | Path, device: str = "cpu") -> dict:
    """Load run_dir/config.json and checkpoints/best.pt, rebuild the net
    (model.build_net), build the *test* loader with _baseline.build_loaders
    (same split arguments as the run), and compute for the hidden layer:

    - faithfulness_curve with ks = [0, 8, 16, 32, 64]
    - stability between the last two epochs' credit (skip if only one epoch)
    - sparsity of the last epoch's credit

    Write run_dir/explain.json and return the same dict. Plot the faithfulness
    curve to run_dir/plots/faithfulness.png (matplotlib, Agg backend).
    """
    raise NotImplementedError("William: run_report (explain/report.py)")
