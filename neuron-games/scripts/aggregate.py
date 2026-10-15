"""Aggregate an ablation directory into one table.

OWNER: Deyvik. See Architecture_Work_14-10-2026.md, "Deyvik".

Usage (from neuron-games/):
    python scripts/aggregate.py runs/ablation-<ts> [--target-acc 0.90]

Input: every */*/results.json under the directory (layout from
run_ablation.sh: <dir>/<condition>/seed-<s>/results.json), each with
meta.condition, meta.row, test.accuracy, best_val_accuracy, wall_time_s,
energy, history (epoch_val_acc, epoch_time_s, epoch_val_firing_rate,
epoch_components, epoch_phi).

TODO (Deyvik):
  - group runs by condition; per condition report mean +- std over seeds of:
      test accuracy, best val accuracy, train wall time, train energy (J, or
      "n/a" when null), hidden / output firing rate at the best epoch,
      final-epoch J, V, E_met, E_bar, Phi,
      time_to_target: wall-clock seconds until epoch_val_acc first reaches
      --target-acc (cumulative epoch_time_s; "not reached" otherwise)
  - refuse to aggregate runs whose config fingerprint (torch version, GPU
    name) differs within a condition; print which runs disagree
  - write <dir>/summary.csv and <dir>/summary.md (rows sorted by meta.row)
  - plot val accuracy vs cumulative wall time, one line per condition (mean
    over seeds, shaded std), to <dir>/val_acc_vs_time.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ablation_dir", type=Path)
    p.add_argument("--target-acc", type=float, default=0.90)
    p.parse_args(argv)
    raise NotImplementedError("Deyvik: scripts/aggregate.py")


if __name__ == "__main__":
    sys.exit(main())
