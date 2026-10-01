"""Kernel sweep for the diversity pilot.

Runs 1 epoch at a fixed lambda for each kernel kind implemented in
`neuron_baseline.kernels`. All runs share the same seed, hyperparameters,
and data; only `diversity_kernel` changes.

Caveat: different kernels output values at different natural magnitudes
(count: products of totals; binned: dot products of raw spike vectors;
mci/vanrossum: dot products of smoothed traces; schoenberg: Gaussian in
[0, 1]). A fixed lambda therefore applies different *regularization
strength* to each kernel. The sweep reports which kernels do something at
this lambda, not which kernel is "best" at matched strength.

Usage (from snn-surrogate-baseline/):

    .venv/Scripts/python.exe scripts/kernel_sweep.py

Outputs:
    runs/kernel-sweep/<timestamp>/<kernel>/
    runs/kernel-sweep/<timestamp>/summary.{json,md,png}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuron_baseline.config import Config
from neuron_baseline.runner import run_training


KERNELS: list[str] = ["count", "binned", "mci", "vanrossum", "schoenberg"]
LAMBDA = 1e-3
BASE_SEED = 20260923


def main() -> int:
    sweep_dir = (
        Path("runs") / "kernel-sweep" / time.strftime("%Y%m%d-%H%M%S")
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] output dir: {sweep_dir.resolve()}")
    print(f"[sweep] lambda = {LAMBDA}, seed = {BASE_SEED}, kernels = {KERNELS}")

    rows: list[dict] = []
    # First row: lambda=0 baseline (kernel field is "none" since no kernel runs).
    baseline_cfg = Config(
        diversity_lambda=0.0,
        seed=BASE_SEED,
        epochs=1,
        run_dir=str(sweep_dir / "baseline_lambda0"),
    )
    print()
    print("=" * 72)
    print("[0/{N}] baseline (lambda = 0)".format(N=len(KERNELS)))
    print("=" * 72)
    out = run_training(baseline_cfg, verbose=True)
    r = out["results"]
    rows.append({
        "kernel": "none",
        "lambda": 0.0,
        "run_dir": baseline_cfg.run_dir,
        "val_acc": r["history"]["epoch_val_acc"][-1],
        "test_acc": r["test"]["accuracy"],
        "epoch_diversity_term": r["history"]["epoch_diversity_term"][-1],
        "train_wall_s": r["wall_time_s"]["train"],
        "train_energy_j": (r["energy"]["train"] or {}).get("energy_j"),
    })

    for i, kind in enumerate(KERNELS):
        cfg = Config(
            diversity_lambda=LAMBDA,
            diversity_kernel=kind,
            seed=BASE_SEED,
            epochs=1,
            run_dir=str(sweep_dir / kind),
        )
        print()
        print("=" * 72)
        print(f"[{i + 1}/{len(KERNELS)}] kernel = {kind} (lambda = {LAMBDA})")
        print("=" * 72)
        out = run_training(cfg, verbose=True)
        r = out["results"]
        rows.append({
            "kernel": kind,
            "lambda": LAMBDA,
            "run_dir": cfg.run_dir,
            "val_acc": r["history"]["epoch_val_acc"][-1],
            "test_acc": r["test"]["accuracy"],
            "epoch_diversity_term": r["history"]["epoch_diversity_term"][-1],
            "train_wall_s": r["wall_time_s"]["train"],
            "train_energy_j": (r["energy"]["train"] or {}).get("energy_j"),
        })

    summary = {
        "lambda": LAMBDA,
        "seed": BASE_SEED,
        "kernels": KERNELS,
        "rows": rows,
    }
    (sweep_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    md_lines = [
        "# Kernel sweep summary",
        "",
        f"- Fixed lambda: `{LAMBDA}`",
        f"- Seed: `{BASE_SEED}`",
        f"- Kernels: `{KERNELS}`",
        "",
        "Note: diversity_term across kernels is NOT comparable because each",
        "kernel outputs values at different natural magnitudes. Use it only",
        "to see whether a kernel is active at all under this lambda.",
        "",
        "| kernel | val_acc | test_acc | div_term | train_s | train_J |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        energy_s = (
            f"{row['train_energy_j']:.1f}"
            if row["train_energy_j"] is not None else "n/a"
        )
        md_lines.append(
            f"| {row['kernel']} | {row['val_acc']:.4f} | "
            f"{row['test_acc']:.4f} | {row['epoch_diversity_term']:.4f} | "
            f"{row['train_wall_s']:.1f} | {energy_s} |"
        )
    (sweep_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    _plot_sweep(sweep_dir, rows)

    print()
    print("=" * 72)
    print("KERNEL SWEEP SUMMARY")
    print("=" * 72)
    print(f"lambda = {LAMBDA} (same for all non-baseline rows)")
    print()
    print(f"{'kernel':>12}  {'val_acc':>8}  {'test_acc':>8}  "
          f"{'div_term':>12}  {'train_s':>8}  {'train_J':>8}")
    for row in rows:
        energy_s = (
            f"{row['train_energy_j']:>8.1f}"
            if row["train_energy_j"] is not None else "     n/a"
        )
        print(f"{row['kernel']:>12}  {row['val_acc']:>8.4f}  "
              f"{row['test_acc']:>8.4f}  "
              f"{row['epoch_diversity_term']:>12.4e}  "
              f"{row['train_wall_s']:>8.1f}  {energy_s}")
    print()
    print(f"[sweep] summary at: {sweep_dir.resolve()}")
    return 0


def _plot_sweep(sweep_dir: Path, rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[sweep] skipping plot ({type(e).__name__}: {e})")
        return

    baseline = next(r for r in rows if r["kernel"] == "none")
    kernels = [r["kernel"] for r in rows if r["kernel"] != "none"]
    val_accs = [r["val_acc"] for r in rows if r["kernel"] != "none"]
    test_accs = [r["test_acc"] for r in rows if r["kernel"] != "none"]
    divs = [max(r["epoch_diversity_term"], 1e-12)
            for r in rows if r["kernel"] != "none"]
    energies = [r["train_energy_j"] for r in rows if r["kernel"] != "none"]

    x = np.arange(len(kernels))
    width = 0.4

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(f"Kernel sweep (lambda = {rows[1]['lambda']}, 1 epoch, seed fixed)")

    ax = axes[0]
    ax.bar(x - width / 2, val_accs, width, label="val_acc")
    ax.bar(x + width / 2, test_accs, width, label="test_acc")
    ax.axhline(baseline["val_acc"], linestyle="--", color="C0", alpha=0.5,
               label=f"baseline val ({baseline['val_acc']:.4f})")
    ax.axhline(baseline["test_acc"], linestyle=":", color="C1", alpha=0.5,
               label=f"baseline test ({baseline['test_acc']:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels(kernels, rotation=20)
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x, divs, color="C2")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(kernels, rotation=20)
    ax.set_ylabel("end-of-epoch div_term (log)")
    ax.set_title("Diversity term (NOT comparable across kernels)")
    ax.grid(True, axis="y", alpha=0.3, which="both")

    ax = axes[2]
    if all(e is not None for e in energies):
        ax.bar(x, energies, color="C3")
        if baseline["train_energy_j"] is not None:
            ax.axhline(baseline["train_energy_j"], linestyle="--",
                       color="k", alpha=0.5,
                       label=f"baseline ({baseline['train_energy_j']:.0f} J)")
            ax.legend(fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(kernels, rotation=20)
        ax.set_ylabel("train energy (J)")
        ax.set_title("Train energy")
    else:
        ax.set_visible(False)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = sweep_dir / "kernel_sweep.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[sweep] plot: {out}")


if __name__ == "__main__":
    sys.exit(main())
