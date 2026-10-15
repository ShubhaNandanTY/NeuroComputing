"""Aggregate an ablation directory into one table.

OWNER: Deyvik. See Architecture_Work_14-10-2026.md, "Deyvik".

Usage (from neuron-games/):
    python scripts/aggregate.py runs/ablation-<ts> [--target-acc 0.90]

Input: every */*/results.json under the directory (layout from
run_ablation.sh: <dir>/<condition>/seed-<s>/results.json), each with
meta.condition, meta.row, test.accuracy, best_val_accuracy, wall_time_s,
energy, history (epoch_val_acc, epoch_time_s, epoch_val_firing_rate,
epoch_components, epoch_phi).

Produces, under <dir>:
    summary.csv  — one row per condition, mean / std columns per metric
    summary.md   — the same table, Markdown, sorted by meta.row
    val_acc_vs_time.png — val acc (mean +- std band) vs cumulative wall time,
                          one line per condition.

Fingerprint check: within one condition, every run must share torch version
and GPU name (from config.json's fingerprint); mismatches are printed and
that condition is skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

METRIC_KEYS = [
    "test_acc", "best_val_acc", "wall_time_s", "energy_j",
    "hidden_rate_best", "output_rate_best",
    "final_J", "final_V", "final_E_met", "final_E_bar", "final_Phi",
    "time_to_target_s",
]


def _mean_std(xs: list[float | None]) -> tuple[float | None, float | None]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return (None, None)
    if len(xs) == 1:
        return (float(xs[0]), 0.0)
    return (statistics.mean(xs), statistics.stdev(xs))


def _best_epoch_rates(history: dict, best_epoch: int) -> tuple[float | None, float | None]:
    rates = history.get("epoch_val_firing_rate") or []
    idx = max(0, min(len(rates) - 1, best_epoch - 1))
    if not rates:
        return (None, None)
    row = rates[idx]
    if not row:
        return (None, None)
    return (float(row[0]), float(row[-1]))


def _final_components(history: dict) -> dict[str, float | None]:
    comps = history.get("epoch_components") or []
    if not comps:
        return {k: None for k in ("J", "V", "E_met", "E_bar")}
    last = comps[-1] or {}
    return {k: (float(last[k]) if k in last and last[k] is not None else None)
            for k in ("J", "V", "E_met", "E_bar")}


def _final_phi(history: dict) -> float | None:
    phi = history.get("epoch_phi") or []
    return float(phi[-1]) if phi else None


def _time_to_target(history: dict, target: float) -> float | None:
    times = history.get("epoch_time_s") or []
    accs = history.get("epoch_val_acc") or []
    cum = 0.0
    for t, v in zip(times, accs):
        cum += float(t)
        if float(v) >= target:
            return cum
    return None


def _train_energy(results: dict) -> float | None:
    energy = (results.get("energy") or {}).get("train")
    if not energy:
        return None
    v = energy.get("energy_j")
    return float(v) if v is not None else None


def _extract(results: dict, target_acc: float) -> dict[str, float | None]:
    meta = results.get("meta", {})
    history = results.get("history", {})
    hidden, output = _best_epoch_rates(history, int(meta.get("best_epoch", 1)))
    comps = _final_components(history)
    return {
        "test_acc": float(results.get("test", {}).get("accuracy", float("nan"))),
        "best_val_acc": float(results.get("best_val_accuracy", float("nan"))),
        "wall_time_s": float(results.get("wall_time_s", {}).get("train", float("nan"))),
        "energy_j": _train_energy(results),
        "hidden_rate_best": hidden,
        "output_rate_best": output,
        "final_J": comps["J"],
        "final_V": comps["V"],
        "final_E_met": comps["E_met"],
        "final_E_bar": comps["E_bar"],
        "final_Phi": _final_phi(history),
        "time_to_target_s": _time_to_target(history, target_acc),
    }


def _fingerprint_key(config: dict) -> tuple[str, str]:
    fp = config.get("fingerprint") or {}
    gpu = fp.get("gpu_name") or fp.get("device_name") or "cpu"
    torch_ver = fp.get("torch") or "unknown"
    return (torch_ver, gpu)


def _val_acc_curve(history: dict) -> tuple[list[float], list[float]]:
    """(cumulative wall time per epoch, val acc per epoch)."""
    times = history.get("epoch_time_s") or []
    accs = history.get("epoch_val_acc") or []
    cum, out_t = 0.0, []
    for t in times:
        cum += float(t)
        out_t.append(cum)
    return out_t, [float(a) for a in accs]


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _fmt_pair(mean: float | None, std: float | None, digits: int = 4) -> str:
    if mean is None:
        return "n/a"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _plot(cond_data: dict[str, list[dict]], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for condition, runs in sorted(cond_data.items(), key=lambda kv: kv[1][0]["row"]):
        curves = [_val_acc_curve(r["history"]) for r in runs]
        if not curves or not curves[0][0]:
            continue
        n_epochs = min(len(c[0]) for c in curves)
        if n_epochs == 0:
            continue
        times = [[c[0][e] for c in curves] for e in range(n_epochs)]
        accs = [[c[1][e] for c in curves] for e in range(n_epochs)]
        mean_t = [statistics.mean(ts) for ts in times]
        mean_a = [statistics.mean(a) for a in accs]
        std_a = [statistics.stdev(a) if len(a) > 1 else 0.0 for a in accs]
        line, = ax.plot(mean_t, mean_a, marker="o", label=condition)
        lo = [m - s for m, s in zip(mean_a, std_a)]
        hi = [m + s for m, s in zip(mean_a, std_a)]
        ax.fill_between(mean_t, lo, hi, color=line.get_color(), alpha=0.15)
    ax.set_xlabel("cumulative training wall time (s)")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Val accuracy vs training time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load(ablation_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """{condition: [{meta, history, metrics, fingerprint, path}, ...]}."""
    out: dict[str, list[dict[str, Any]]] = {}
    for cond_dir in sorted(p for p in ablation_dir.iterdir() if p.is_dir()):
        for seed_dir in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
            res_path = seed_dir / "results.json"
            cfg_path = seed_dir / "config.json"
            if not res_path.is_file() or not cfg_path.is_file():
                continue
            try:
                results = json.loads(res_path.read_text(encoding="utf-8"))
                config = json.loads(cfg_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                print(f"[skip] {seed_dir}: {e}", file=sys.stderr)
                continue
            meta = results.get("meta", {})
            condition = meta.get("condition", cond_dir.name)
            out.setdefault(condition, []).append({
                "path": seed_dir,
                "row": int(meta.get("row", 0)),
                "history": results.get("history", {}),
                "fingerprint": _fingerprint_key(config),
            })
    return out


def _aggregate(runs: list[dict], target_acc: float) -> dict[str, tuple[float | None, float | None]]:
    extracted = []
    for r in runs:
        results = json.loads((r["path"] / "results.json").read_text(encoding="utf-8"))
        extracted.append(_extract(results, target_acc))
    return {k: _mean_std([e[k] for e in extracted]) for k in METRIC_KEYS}


def _write_csv(path: Path, rows: list[dict], target_acc: float) -> None:
    fieldnames = ["condition", "row", "n_seeds"]
    for k in METRIC_KEYS:
        fieldnames += [f"{k}_mean", f"{k}_std"]
    fieldnames.append("fingerprint")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            flat = {"condition": row["condition"], "row": row["row"],
                    "n_seeds": row["n_seeds"], "fingerprint": row["fingerprint"]}
            for k, (m, s) in row["metrics"].items():
                flat[f"{k}_mean"] = "" if m is None else m
                flat[f"{k}_std"] = "" if s is None else s
            w.writerow(flat)


def _write_md(path: Path, rows: list[dict], target_acc: float) -> None:
    cols = [
        ("condition", "condition"),
        ("row", "row"),
        ("n_seeds", "n"),
        ("test_acc", "test acc"),
        ("best_val_acc", "best val"),
        ("wall_time_s", "wall s"),
        ("energy_j", "energy J"),
        ("hidden_rate_best", "h rate"),
        ("output_rate_best", "o rate"),
        ("final_J", "J"),
        ("final_V", "V"),
        ("final_E_met", "E_met"),
        ("final_E_bar", "E_bar"),
        ("final_Phi", "Phi"),
        ("time_to_target_s", f"t→{target_acc}s"),
    ]
    header = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [f"# Ablation summary (target acc {target_acc})", "", header, sep]
    for row in rows:
        cells = []
        for key, _ in cols:
            if key == "condition":
                cells.append(str(row["condition"]))
            elif key == "row":
                cells.append(str(row["row"]))
            elif key == "n_seeds":
                cells.append(str(row["n_seeds"]))
            else:
                cells.append(_fmt_pair(*row["metrics"][key]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Fingerprints:")
    for row in rows:
        lines.append(f"- {row['condition']}: {row['fingerprint']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ablation_dir", type=Path)
    p.add_argument("--target-acc", type=float, default=0.90)
    args = p.parse_args(argv)

    if not args.ablation_dir.is_dir():
        print(f"not a directory: {args.ablation_dir}", file=sys.stderr)
        return 2

    loaded = _load(args.ablation_dir)
    if not loaded:
        print(f"no runs found under {args.ablation_dir}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for condition, runs in loaded.items():
        fingerprints = {r["fingerprint"] for r in runs}
        if len(fingerprints) > 1:
            print(f"[skip] {condition}: mixed fingerprints:", file=sys.stderr)
            for r in runs:
                print(f"    {r['path']}: {r['fingerprint']}", file=sys.stderr)
            continue
        metrics = _aggregate(runs, args.target_acc)
        rows.append({
            "condition": condition,
            "row": runs[0]["row"],
            "n_seeds": len(runs),
            "fingerprint": next(iter(fingerprints)),
            "metrics": metrics,
        })

    rows.sort(key=lambda r: (r["row"], r["condition"]))

    csv_path = args.ablation_dir / "summary.csv"
    md_path = args.ablation_dir / "summary.md"
    plot_path = args.ablation_dir / "val_acc_vs_time.png"
    _write_csv(csv_path, rows, args.target_acc)
    _write_md(md_path, rows, args.target_acc)
    try:
        _plot({r["condition"]: loaded[r["condition"]] for r in rows}, plot_path)
    except Exception as e:
        print(f"[warn] plot failed: {e}", file=sys.stderr)

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    if plot_path.is_file():
        print(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
