"""Post-hoc explanation and explainability metrics (proposal §8.5, §12.4).

OWNER: William. See Architecture_Work_14-10-2026.md, "William".

attribution.py  load per-neuron credit from a run; neuron masks for ablation
metrics.py      faithfulness, stability, sparsity
report.py       `python -m neuron_games explain --run <run_dir>` entry point
"""
