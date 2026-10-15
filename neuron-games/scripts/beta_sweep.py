"""Tuning sweep for V: beta over {kernel_offdiag, dpp_logdet} on a dense log grid.

OWNER: Deyvik. Satisfies the "Experiment" acceptance item in
Architecture_Work_14-10-2026.md section 6 ("Deyvik"), expanded from the
doc's 3-point grid {1e-4, 1e-3, 1e-2} to 9 half-decade points.

Runs (3 epochs each, three_factor base, spike feature unless noted):
  1. baseline              plain three_factor, no V
  2-10. dpp_<beta>         --diversity dpp_logdet,     beta in BETAS
  11-19. off_<beta>        --diversity kernel_offdiag, beta in BETAS
  20. best_plast           best (diversity, beta) with plasticity_signature

Prints a table (val acc, V, mean sec/epoch) and the recommended default beta.
Run outputs land in runs/sweep-<ts>/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuron_games.config import NGRunConfig  # noqa: E402
from neuron_games.runner import run_training  # noqa: E402


BETAS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
DIVERSITIES = [("dpp", "dpp_logdet"), ("off", "kernel_offdiag")]


def _run(name: str, root: Path, device: str, epochs: int, seed: int,
         diversity: str | None, beta: float | None, feature: str) -> dict:
    run_dir = root / name / f"seed-{seed}"
    kw: dict = {"epochs": epochs, "run_dir": str(run_dir), "device": device, "seed": seed}
    if diversity is not None:
        kw["diversity"] = diversity
        kw["beta"] = beta
        kw["diversity_feature"] = feature
    cfg = NGRunConfig.for_condition("three_factor", **kw)
    print(f"\n=== {name} (diversity={diversity} beta={beta} feature={feature}) ===")
    out = run_training(cfg, verbose=True)
    results = out["results"]
    history = results["history"]
    comps = history.get("epoch_components") or [{}]
    return {
        "name": name,
        "diversity": diversity,
        "beta": beta,
        "feature": feature,
        "best_val_acc": results["best_val_accuracy"],
        "test_acc": results["test"]["accuracy"],
        "mean_epoch_s": sum(history["epoch_time_s"]) / max(len(history["epoch_time_s"]), 1),
        "V_final": comps[-1].get("V"),
        "J_final": comps[-1].get("J"),
        "phi_final": (history.get("epoch_phi") or [None])[-1],
        "hidden_rate": history["epoch_val_firing_rate"][-1][0],
    }


def _fmt_beta(b: float | None) -> str:
    return "n/a" if b is None else f"{b:.0e}"


def _print_row(r: dict, baseline_s: float | None) -> None:
    overhead = f"{(r['mean_epoch_s'] - baseline_s) / baseline_s * 100:+.0f}%" if baseline_s else "-"
    V = "n/a" if r["V_final"] is None else f"{r['V_final']:+.4f}"
    beta_s = _fmt_beta(r["beta"])
    print(f"  {r['name']:<18} val={r['best_val_acc']:.4f}  test={r['test_acc']:.4f}  "
          f"beta={beta_s:>6}  V={V:>9}  s/epoch={r['mean_epoch_s']:.1f} ({overhead:>5})  "
          f"h={r['hidden_rate']:.3f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sweep-root", type=Path, default=None)
    args = p.parse_args(argv)

    ts = time.strftime("%Y%m%d-%H%M%S")
    root = args.sweep_root or Path("runs") / f"sweep-{ts}"
    root.mkdir(parents=True, exist_ok=True)
    print(f"Sweep root: {root.resolve()}")
    print(f"Betas: {BETAS}")

    rows: list[dict] = []
    rows.append(_run("baseline", root, args.device, args.epochs, args.seed, None, None, "spike"))

    for short, div in DIVERSITIES:
        for beta in BETAS:
            name = f"{short}_{beta:.0e}"
            rows.append(_run(name, root, args.device, args.epochs, args.seed,
                             div, beta, "spike"))

    baseline_s = next((r["mean_epoch_s"] for r in rows if r["name"] == "baseline"), None)

    v_rows = [r for r in rows if r["diversity"] is not None]
    best = max(v_rows, key=lambda r: r["best_val_acc"])
    print(f"\nBest spike-feature config: {best['name']} "
          f"(diversity={best['diversity']} beta={best['beta']} val={best['best_val_acc']:.4f})")

    plast = _run(f"plast_{best['name']}", root, args.device, args.epochs, args.seed,
                 best["diversity"], best["beta"], "plasticity_signature")
    rows.append(plast)

    print("\n=== Summary (sorted by val acc) ===")
    for r in sorted(rows, key=lambda r: -r["best_val_acc"]):
        _print_row(r, baseline_s)

    best_overall = max([r for r in rows if r["diversity"] is not None],
                       key=lambda r: r["best_val_acc"])
    print(f"\nRecommended default beta: {best_overall['beta']} "
          f"(from {best_overall['name']}, val={best_overall['best_val_acc']:.4f}).")
    print(f"Sweep artifacts under: {root.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
