"""CLI entrypoint.

Usage examples (from `snn-surrogate-baseline/`):

    # smoke test (very small — just prove everything wires up)
    .venv/Scripts/python.exe -m neuron_baseline smoke

    # full one-epoch baseline (defaults from Config)
    .venv/Scripts/python.exe -m neuron_baseline train

    # train longer with overrides
    .venv/Scripts/python.exe -m neuron_baseline train --epochs 5 --seed 42

    # evaluate a saved checkpoint on the test set
    .venv/Scripts/python.exe -m neuron_baseline eval --ckpt runs/<ts>/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

from .config import Config


def _now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    for f in fields(cfg):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    return cfg


def _add_config_overrides(p: argparse.ArgumentParser) -> None:
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--eval-batch-size", dest="eval_batch_size", type=int)
    p.add_argument("--num-steps", dest="num_steps", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--beta", type=float)
    p.add_argument("--surrogate-slope", dest="surrogate_slope", type=float)
    p.add_argument("--threshold", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--val-size", dest="val_size", type=int)
    p.add_argument("--data-dir", dest="data_dir")
    p.add_argument("--run-dir", dest="run_dir")
    p.add_argument("--device")
    p.add_argument("--num-workers", dest="num_workers", type=int)


def cmd_train(args: argparse.Namespace) -> int:
    import torch
    from .reproducibility import seed_everything, fingerprint
    from .data import build_loaders
    from .train import train
    from .eval import evaluate
    from .energy import EnergyMeter
    from .plots import save_training_plots

    cfg = _apply_overrides(Config(), args)
    if not cfg.run_dir:
        cfg.run_dir = str(Path("runs") / _now_stamp())
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    cfg.fingerprint = fingerprint()

    device = torch.device(cfg.device)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        print("FATAL: CUDA requested but unavailable. Refusing to fall back to CPU.",
              file=sys.stderr)
        return 3

    print(f"[cfg] run_dir={run_dir}")
    print(f"[cfg] {cfg.fingerprint.get('device_name', '(cpu)')}, "
          f"torch={cfg.fingerprint.get('torch')}, "
          f"snntorch={cfg.fingerprint.get('snntorch')}")
    cfg.save(run_dir / "config.json")

    loaders = build_loaders(
        cfg.data_dir, cfg.batch_size, cfg.eval_batch_size,
        cfg.val_size, cfg.seed, num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
    )
    print(f"[data] train_batches={len(loaders['train'])} "
          f"val_examples={len(loaders['val'].dataset)} "
          f"test_examples={len(loaders['test'].dataset)}")

    meter = EnergyMeter(device_index=0, sample_hz=cfg.energy_sample_hz)
    print(f"[energy] nvml_available={meter.available} "
          f"cumulative_energy={meter.has_cumulative} "
          f"device='{meter.device_name}' notes={meter._init_notes!r}")

    with meter.measure("train"):
        t0 = time.perf_counter()
        out = train(cfg, loaders, device, run_dir)
        train_wall_s = time.perf_counter() - t0
    train_energy = meter.result

    # Reload best.pt and evaluate on the reserved test set.
    best_path = run_dir / "checkpoints" / "best.pt"
    ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
    from .model import build_net
    eval_net = build_net(
        beta=cfg.beta, surrogate_slope=cfg.surrogate_slope,
        threshold=cfg.threshold, reset_mechanism=cfg.reset_mechanism,
    ).to(device)
    eval_net.load_state_dict(ckpt["model_state"])

    # Warm up before measuring inference energy
    for _ in range(cfg.warmup_batches):
        for data, targets in loaders["val"]:
            data = data.to(device, non_blocking=True)
            from .model import forward_pass
            with torch.no_grad():
                forward_pass(eval_net, cfg.num_steps, data)
            break

    with meter.measure("inference"):
        t0 = time.perf_counter()
        test_metrics = evaluate(eval_net, loaders["test"], cfg.num_steps, device)
        infer_wall_s = time.perf_counter() - t0
    infer_energy = meter.result

    meter.shutdown()

    energy_per_image_j = None
    if infer_energy is not None and infer_energy.energy_j is not None and test_metrics["n"]:
        energy_per_image_j = infer_energy.energy_j / test_metrics["n"]

    results = {
        "history": out["history"],
        "best_val_accuracy": out["best_val_acc"],
        "test": test_metrics,
        "wall_time_s": {"train": train_wall_s, "inference": infer_wall_s},
        "energy": {
            "train": train_energy.to_dict() if train_energy else None,
            "inference": infer_energy.to_dict() if infer_energy else None,
            "inference_per_image_j": energy_per_image_j,
            "measurement_notes": (
                "GPU-only NVML reading (RTX 5080 device power); "
                "does not include CPU / RAM / system PSU. "
                "torch.cuda.synchronize() at every measurement boundary. "
                "Validation eval is inside the training window (not separated). "
                "Warm-up: forward-only passes on val loader before inference measurement. "
                "Any competing GUI/GPU workloads on this desktop show up in the reading."
            ),
        },
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    save_training_plots(out["history"], run_dir / "plots")

    print()
    print(f"[done] test_accuracy={test_metrics['accuracy']:.4f} on n={test_metrics['n']} examples")
    print(f"[done] wall: train={train_wall_s:.1f}s  inference={infer_wall_s:.1f}s")
    print(f"[done] energy: train={train_energy.energy_j}J  "
          f"inference={infer_energy.energy_j}J  "
          f"per_image={energy_per_image_j}J")
    print(f"[done] artifacts under: {run_dir.resolve()}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    from tests.smoke import run_smoke
    return run_smoke()


def cmd_eval(args: argparse.Namespace) -> int:
    import torch
    from .data import build_loaders
    from .model import build_net
    from .eval import evaluate
    from .reproducibility import seed_everything

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ckpt["config"].items()
                    if k in {f.name for f in fields(Config)}})
    if isinstance(cfg.adam_betas, list):
        cfg.adam_betas = tuple(cfg.adam_betas)

    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    loaders = build_loaders(
        cfg.data_dir, cfg.batch_size, cfg.eval_batch_size,
        cfg.val_size, cfg.seed, num_workers=cfg.num_workers,
    )
    net = build_net(
        beta=cfg.beta, surrogate_slope=cfg.surrogate_slope,
        threshold=cfg.threshold, reset_mechanism=cfg.reset_mechanism,
    ).to(device)
    net.load_state_dict(ckpt["model_state"])
    m = evaluate(net, loaders["test"], cfg.num_steps, device)
    print(json.dumps({"ckpt": str(ckpt_path), "test": m, "cfg_epoch": ckpt.get("epoch")},
                     indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="neuron_baseline",
                                description="snnTorch Tutorial 6 surrogate-BP baseline")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="run baseline training")
    _add_config_overrides(p_train)
    p_train.set_defaults(func=cmd_train)

    p_smoke = sub.add_parser("smoke", help="run smoke test (small, fast)")
    p_smoke.set_defaults(func=cmd_smoke)

    p_eval = sub.add_parser("eval", help="evaluate a saved checkpoint on the test set")
    p_eval.add_argument("--ckpt", required=True, help="path to .pt checkpoint")
    p_eval.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
