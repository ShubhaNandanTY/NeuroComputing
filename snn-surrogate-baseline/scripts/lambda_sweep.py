"""Phase B3: lambda sweep for the kernel-diversity pilot.

Runs 1 epoch at each lambda in LAMBDAS, single seed, and summarizes the
results. Picks lambda_0 for the pilot: the largest lambda that keeps
val accuracy within VAL_ACC_TOLERANCE of the baseline.

Usage (from snn-surrogate-baseline/):

    .venv/Scripts/python.exe scripts/lambda_sweep.py

Outputs:
    runs/lambda-sweep/<timestamp>/<lambda>/   (one run dir per lambda)
    runs/lambda-sweep/<timestamp>/summary.json
    runs/lambda-sweep/<timestamp>/summary.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure the project root is importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuron_baseline.config import Config
from neuron_baseline.runner import run_training


LAMBDAS: list[float] = [
    0.0,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
]
BASE_SEED = 20260923
VAL_ACC_TOLERANCE_PTS = 1.0


def _lambda_tag(lam: float) -> str:
    if lam == 0:
        return "lambda_0"
    return f"lambda_{lam:.0e}"


def _plot_sweep(sweep_dir: Path, rows: list[dict], baseline_val: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[sweep] skipping plot ({type(e).__name__}: {e})")
        return

    nonzero = [r for r in rows if r["lambda"] > 0]
    if not nonzero:
        return
    lams = [r["lambda"] for r in nonzero]
    val_accs = [r["val_acc"] for r in nonzero]
    test_accs = [r["test_acc"] for r in nonzero]
    divs = [r["epoch_diversity_term"] for r in nonzero]
    energies = [r["train_energy_j"] for r in nonzero]
    walls = [r["train_wall_s"] for r in nonzero]

    baseline = next(r for r in rows if r["lambda"] == 0.0)
    base_test = baseline["test_acc"]
    base_energy = baseline["train_energy_j"]
    base_wall = baseline["train_wall_s"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Lambda sweep (mci kernel, 1 epoch, seed fixed)")

    ax = axes[0, 0]
    ax.semilogx(lams, val_accs, "o-", label="val_acc")
    ax.semilogx(lams, test_accs, "s-", label="test_acc")
    ax.axhline(baseline_val, linestyle="--", color="C0", alpha=0.4,
               label=f"baseline val ({baseline_val:.4f})")
    ax.axhline(base_test, linestyle=":", color="C1", alpha=0.4,
               label=f"baseline test ({base_test:.4f})")
    ax.set_xlabel("lambda")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs lambda")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.loglog(lams, divs, "o-", color="C2")
    ax.set_xlabel("lambda")
    ax.set_ylabel("mean diversity term (end of epoch)")
    ax.set_title("Diversity term vs lambda")
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1, 0]
    if all(e is not None for e in energies):
        ax.semilogx(lams, energies, "o-", color="C3")
        if base_energy is not None:
            ax.axhline(base_energy, linestyle="--", color="k", alpha=0.4,
                       label=f"baseline ({base_energy:.0f} J)")
            ax.legend(fontsize=8)
        ax.set_xlabel("lambda")
        ax.set_ylabel("train energy (J)")
        ax.set_title("Train energy vs lambda")
    else:
        ax.set_visible(False)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogx(lams, walls, "o-", color="C4")
    ax.axhline(base_wall, linestyle="--", color="k", alpha=0.4,
               label=f"baseline ({base_wall:.1f} s)")
    ax.legend(fontsize=8)
    ax.set_xlabel("lambda")
    ax.set_ylabel("train wall time (s)")
    ax.set_title("Train wall vs lambda")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = sweep_dir / "lambda_sweep.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[sweep] plot: {out}")


def main() -> int:
    sweep_dir = (
        Path("runs") / "lambda-sweep" / time.strftime("%Y%m%d-%H%M%S")
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] output dir: {sweep_dir.resolve()}")

    rows: list[dict] = []
    for i, lam in enumerate(LAMBDAS):
        tag = _lambda_tag(lam)
        print()
        print("=" * 72)
        print(f"[{i + 1}/{len(LAMBDAS)}] lambda = {lam!r}  (tag: {tag})")
        print("=" * 72)
        cfg = Config(
            diversity_lambda=lam,
            seed=BASE_SEED,
            epochs=1,
            run_dir=str(sweep_dir / tag),
        )
        out = run_training(cfg, verbose=True)
        r = out["results"]
        train_energy = r["energy"]["train"]
        rows.append({
            "lambda": lam,
            "tag": tag,
            "run_dir": cfg.run_dir,
            "val_acc": r["history"]["epoch_val_acc"][-1],
            "test_acc": r["test"]["accuracy"],
            "epoch_diversity_term": r["history"]["epoch_diversity_term"][-1],
            "train_wall_s": r["wall_time_s"]["train"],
            "train_energy_j": (train_energy or {}).get("energy_j"),
            "inference_j_per_image": r["energy"].get("inference_per_image_j"),
        })

    baseline = next(row for row in rows if row["lambda"] == 0.0)
    baseline_val = baseline["val_acc"]
    tolerance = VAL_ACC_TOLERANCE_PTS / 100.0
    within_tol = [
        row for row in rows
        if row["lambda"] > 0.0 and row["val_acc"] >= baseline_val - tolerance
    ]
    if within_tol:
        lambda_0 = max(within_tol, key=lambda r: r["lambda"])
    else:
        lambda_0 = None

    summary = {
        "lambdas": LAMBDAS,
        "seed": BASE_SEED,
        "val_acc_tolerance_pts": VAL_ACC_TOLERANCE_PTS,
        "baseline_val_acc": baseline_val,
        "rows": rows,
        "lambda_0_selected": (
            {"lambda": lambda_0["lambda"], "val_acc": lambda_0["val_acc"]}
            if lambda_0 else None
        ),
    }
    (sweep_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    md_lines = [
        "# Lambda sweep summary",
        "",
        f"- Seed: `{BASE_SEED}`",
        f"- Lambdas: `{LAMBDAS}`",
        f"- Val-acc tolerance for lambda_0 selection: {VAL_ACC_TOLERANCE_PTS} pts",
        f"- Baseline (lambda=0) val_acc: `{baseline_val:.4f}`",
        "",
        "| lambda | val_acc | test_acc | div_term | train_s | train_J |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        energy_s = (
            f"{row['train_energy_j']:.1f}"
            if row["train_energy_j"] is not None else "n/a"
        )
        md_lines.append(
            f"| {row['lambda']:.0e} | {row['val_acc']:.4f} | "
            f"{row['test_acc']:.4f} | {row['epoch_diversity_term']:.4f} | "
            f"{row['train_wall_s']:.1f} | {energy_s} |"
        )
    md_lines += [""]
    if lambda_0:
        md_lines.append(
            f"**Selected lambda_0 = {lambda_0['lambda']:.0e}** "
            f"(val_acc {lambda_0['val_acc']:.4f}, "
            f"delta vs baseline = {(lambda_0['val_acc'] - baseline_val) * 100:+.2f} pts)."
        )
    else:
        md_lines.append(
            "**No lambda > 0 kept val_acc within tolerance.** "
            f"Baseline val_acc={baseline_val:.4f}; "
            "all diversity-enabled conditions degraded accuracy by more than "
            f"{VAL_ACC_TOLERANCE_PTS} pts."
        )
    (sweep_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    _plot_sweep(sweep_dir, rows, baseline_val)

    print()
    print("=" * 72)
    print("SWEEP SUMMARY")
    print("=" * 72)
    print(f"baseline val_acc (lambda=0): {baseline_val:.4f}")
    print()
    print(f"{'lambda':>10}  {'val_acc':>8}  {'test_acc':>8}  {'div_term':>10}  "
          f"{'train_s':>8}  {'train_J':>8}")
    for row in rows:
        energy_s = (
            f"{row['train_energy_j']:>8.1f}" if row["train_energy_j"] is not None
            else "     n/a"
        )
        print(f"{row['lambda']:>10.0e}  {row['val_acc']:>8.4f}  "
              f"{row['test_acc']:>8.4f}  {row['epoch_diversity_term']:>10.4f}  "
              f"{row['train_wall_s']:>8.1f}  {energy_s}")
    print()
    if lambda_0:
        print(f"[recommendation] lambda_0 = {lambda_0['lambda']:.0e}  "
              f"(largest lambda within {VAL_ACC_TOLERANCE_PTS}-pt tolerance)")
    else:
        print(f"[recommendation] no lambda within {VAL_ACC_TOLERANCE_PTS}-pt "
              "tolerance; consider finer search or wider tolerance")
    print(f"[sweep] summary at: {sweep_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
