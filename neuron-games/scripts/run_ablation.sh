#!/usr/bin/env bash
# Run the Neuron Games ablation grid: every condition x every seed.
#
# OWNER: Joohyun. See Architecture_Work_14-10-2026.md, "Joohyun".
#
# Usage (from neuron-games/):
#   scripts/run_ablation.sh [--device cuda|mps|cpu] [--epochs N] [--seeds "1 2 3"]
#                           [--conditions "three_factor tf_div tf_div_energy ng_full"]
#                           [--out runs/ablation-<ts>] [-- extra args passed to every run]
#
# TODO (Joohyun):
#  - parse the flags above; defaults: device=cuda, epochs=25, seeds="1 2 3",
#    conditions="three_factor tf_div tf_div_energy ng_full", out=runs/ablation-$(date +%Y%m%d-%H%M%S)
#  - for each condition and seed:
#      PYTHONPATH=src python -m neuron_games train --condition "$c" --seed "$s" \
#          --epochs "$EPOCHS" --device "$DEVICE" --run-dir "$OUT/$c/seed-$s" "${EXTRA[@]}"
#    and tee its stdout to "$OUT/$c/seed-$s/train.log"
#  - keep going if one run fails, but record failures and exit non-zero at the end
#  - write "$OUT/manifest.json" listing every run dir, condition, seed, exit code
#  - finish by calling: python scripts/aggregate.py "$OUT"
set -euo pipefail
echo "run_ablation.sh: not implemented yet (Joohyun)" >&2
exit 2
