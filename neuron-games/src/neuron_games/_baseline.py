"""Bridge to the sibling `snn-surrogate-baseline/neuron_baseline` package.

The implementation plan (§4) reuses the baseline's data loaders, NVML energy
meter, seeding/fingerprint, and plots rather than copying them, so Row 2 and
the Neuron Games rows are measured by identical machinery. The baseline is not
an installed package, so we put its folder on `sys.path` once, here.
"""
from __future__ import annotations

import sys
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parents[3] / "snn-surrogate-baseline"
if not (BASELINE_ROOT / "neuron_baseline").is_dir():
    raise ImportError(f"snn-surrogate-baseline not found at {BASELINE_ROOT}")
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from neuron_baseline.data import build_loaders  # noqa: E402
from neuron_baseline.energy import EnergyMeter  # noqa: E402
from neuron_baseline.plots import save_training_plots  # noqa: E402
from neuron_baseline.reproducibility import fingerprint, seed_everything  # noqa: E402

__all__ = ["BASELINE_ROOT", "build_loaders", "EnergyMeter", "save_training_plots",
           "fingerprint", "seed_everything"]
