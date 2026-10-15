"""Per-kernel lambda sweep.

Each kernel has its own lambda grid tuned to its natural K-value scale,
so the regularization strength is comparable across kernels. (A fixed
lambda across kernels like kernel_sweep.py applies vastly different
effective pressure, because `count` K-values are ~1e5 while `schoenberg`
K-values are in [0, 1].)

vanrossum is skipped: it is a numerical alias of mci under our smoothing
pipeline (same math, same output).

Usage (from snn-surrogate-baseline/):

    .venv/Scripts/python.exe scripts/kernel_lambda_sweep.py

Outputs:
    runs/kernel-lambda-sweep/<ts>/<kernel>_<lambda>/
    runs/kernel-lambda-sweep/<ts>/summary.{json,md,png}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuron_baseline.config import Config
from neuron_baseline.runner import run_training


PER_KERNEL_GRIDS: dict[str, list[float]] = {
    "count":      [1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
    "binned":     [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
    "mci":        [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "schoenberg": [1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0],
}
BASE_SEED = 20260923
VAL_ACC_TOLERANCE_PTS = 1.0


def _tag(kernel: str, lam: float) -> str:
    return f"{kernel}_{lam:.0e}"


def _run_one(sweep_dir: Path, kernel: str, lam: float) -> dict:
    cfg = Config(
        diversity_lambda=lam,
        diversity_kernel=kernel,
        seed=BASE_SEED,
        epochs=1,
        run_dir=str(sweep_dir / _tag(kernel, lam)),
    )
    out = run_training(cfg, verbose=True)
    r = out["results"]
    return {
        "kernel": kernel,
        "lambda": lam,
        "run_dir": cfg.run_dir,
        "val_acc": r["history"]["epoch_val_acc"][-1],
        "test_acc": r["test"]["accuracy"],
        "epoch_diversity_term": r["history"]["epoch_diversity_term"][-1],
        "train_wall_s": r["wall_time_s"]["train"],
        "train_energy_j": (r["energy"]["train"] or {}).get("energy_j"),
    }


def main() -> int:
    sweep_dir = (
        Path("runs") / "kernel-lambda-sweep" / time.strftime("%Y%m%d-%H%M%S")
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] output dir: {sweep_dir.resolve()}")

    runs: list[dict] = []

    # Shared baseline (lambda = 0).
    print()
    print("=" * 72)
    print("[baseline] lambda = 0")
    print("=" * 72)
    bl_cfg = Config(
        diversity_lambda=0.0, seed=BASE_SEED, epochs=1,
        run_dir=str(sweep_dir / "baseline_lambda0"),
    )
    bl_out = run_training(bl_cfg, verbose=True)
    br = bl_out["results"]
    baseline = {
        "kernel": "none",
        "lambda": 0.0,
        "run_dir": bl_cfg.run_dir,
        "val_acc": br["history"]["epoch_val_acc"][-1],
        "test_acc": br["test"]["accuracy"],
        "epoch_diversity_term": 0.0,
        "train_wall_s": br["wall_time_s"]["train"],
        "train_energy_j": (br["energy"]["train"] or {}).get("energy_j"),
    }
    runs.append(baseline)

    total_runs = sum(len(v) for v in PER_KERNEL_GRIDS.values())
    done = 0
    for kernel, lambdas in PER_KERNEL_GRIDS.items():
        for lam in lambdas:
            done += 1
            print()
            print("=" * 72)
            print(f"[{done}/{total_runs}] kernel={kernel} lambda={lam:.0e}")
            print("=" * 72)
            runs.append(_run_one(sweep_dir, kernel, lam))

    tolerance = VAL_ACC_TOLERANCE_PTS / 100.0
    baseline_val = baseline["val_acc"]
    recommendations: dict[str, dict | None] = {}
    for kernel in PER_KERNEL_GRIDS:
        in_tol = [r for r in runs
                  if r["kernel"] == kernel
                  and r["val_acc"] >= baseline_val - tolerance]
        recommendations[kernel] = (
            max(in_tol, key=lambda r: r["lambda"]) if in_tol else None
        )

    summary = {
        "per_kernel_grids": PER_KERNEL_GRIDS,
        "seed": BASE_SEED,
        "val_acc_tolerance_pts": VAL_ACC_TOLERANCE_PTS,
        "baseline_val_acc": baseline_val,
        "baseline_test_acc": baseline["test_acc"],
        "runs": runs,
        "recommended_lambda_per_kernel": {
            k: ({"lambda": v["lambda"], "val_acc": v["val_acc"]}
                if v else None)
            for k, v in recommendations.items()
        },
    }
    (sweep_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    _write_md(sweep_dir, summary, runs)
    _plot_sweep(sweep_dir, runs, baseline)

    print()
    print("=" * 72)
    print("PER-KERNEL SWEEP SUMMARY")
    print("=" * 72)
    print(f"baseline val_acc = {baseline_val:.4f}  "
          f"test_acc = {baseline['test_acc']:.4f}")
    print(f"tolerance = {VAL_ACC_TOLERANCE_PTS} pt")
    print()
    for kernel, lambdas in PER_KERNEL_GRIDS.items():
        print(f"--- {kernel} ---")
        print(f"  {'lambda':>10}  {'val_acc':>8}  {'test_acc':>8}  "
              f"{'div_term':>12}  {'train_J':>8}")
        for row in runs:
            if row["kernel"] != kernel:
                continue
            energy_s = (f"{row['train_energy_j']:>8.1f}"
                        if row["train_energy_j"] is not None else "     n/a")
            print(f"  {row['lambda']:>10.0e}  {row['val_acc']:>8.4f}  "
                  f"{row['test_acc']:>8.4f}  "
                  f"{row['epoch_diversity_term']:>12.4e}  {energy_s}")
        rec = recommendations[kernel]
        if rec:
            delta = (rec["val_acc"] - baseline_val) * 100
            print(f"  -> recommended lambda = {rec['lambda']:.0e}  "
                  f"(val_acc {rec['val_acc']:.4f}, delta {delta:+.2f} pts)")
        else:
            print(f"  -> no lambda within tolerance")
        print()
    print(f"[sweep] summary at: {sweep_dir.resolve()}")
    return 0


def _write_md(sweep_dir: Path, summary: dict, runs: list[dict]) -> None:
    baseline_val = summary["baseline_val_acc"]
    md: list[str] = [
        "# Per-kernel lambda sweep",
        "",
        f"- Seed: `{summary['seed']}`",
        f"- Val-acc tolerance for recommendation: {summary['val_acc_tolerance_pts']} pts",
        f"- Baseline (lambda=0) val_acc: `{baseline_val:.4f}` "
        f"test_acc `{summary['baseline_test_acc']:.4f}`",
        "",
    ]
    for kernel, lambdas in summary["per_kernel_grids"].items():
        md += [f"## {kernel}", "",
               "| lambda | val_acc | test_acc | div_term | train_J |",
               "|---|---|---|---|---|"]
        for row in runs:
            if row["kernel"] != kernel:
                continue
            energy_s = (f"{row['train_energy_j']:.1f}"
                        if row["train_energy_j"] is not None else "n/a")
            md.append(
                f"| {row['lambda']:.0e} | {row['val_acc']:.4f} | "
                f"{row['test_acc']:.4f} | "
                f"{row['epoch_diversity_term']:.4e} | {energy_s} |"
            )
        rec = summary["recommended_lambda_per_kernel"].get(kernel)
        md.append("")
        if rec:
            md.append(f"**Recommended lambda = {rec['lambda']:.0e}** "
                      f"(val_acc {rec['val_acc']:.4f}).")
        else:
            md.append("**No lambda within tolerance.**")
        md.append("")
    (sweep_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


def _plot_sweep(sweep_dir: Path, runs: list[dict], baseline: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[sweep] skipping plot ({type(e).__name__}: {e})")
        return

    kernels = list(PER_KERNEL_GRIDS.keys())
    fig, axes = plt.subplots(2, len(kernels), figsize=(4 * len(kernels), 7),
                             sharey="row")
    fig.suptitle("Per-kernel lambda sweep (1 epoch, single seed; baseline dashed)")

    for col, kernel in enumerate(kernels):
        rows = [r for r in runs if r["kernel"] == kernel]
        rows.sort(key=lambda r: r["lambda"])
        lams = [r["lambda"] for r in rows]
        vals = [r["val_acc"] for r in rows]
        tests = [r["test_acc"] for r in rows]
        divs = [max(r["epoch_diversity_term"], 1e-20) for r in rows]

        ax = axes[0, col]
        ax.semilogx(lams, vals, "o-", label="val_acc")
        ax.semilogx(lams, tests, "s-", label="test_acc")
        ax.axhline(baseline["val_acc"], linestyle="--", color="C0", alpha=0.4)
        ax.axhline(baseline["test_acc"], linestyle=":", color="C1", alpha=0.4)
        ax.set_title(f"{kernel}")
        ax.set_xlabel("lambda")
        if col == 0:
            ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)

        ax = axes[1, col]
        ax.loglog(lams, divs, "o-", color="C2")
        ax.set_xlabel("lambda")
        if col == 0:
            ax.set_ylabel("div_term (log)")
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out = sweep_dir / "kernel_lambda_sweep.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[sweep] plot: {out}")


if __name__ == "__main__":
    sys.exit(main())
