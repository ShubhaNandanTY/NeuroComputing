"""CLI entrypoint. Run from `neuron-games/` with `src` on the path:

    PYTHONPATH=src python -m neuron_games train --condition bp_anchor --seed 1
    PYTHONPATH=src python -m neuron_games train --device mps --epochs 5
    PYTHONPATH=src python -m neuron_games eval --ckpt runs/bp_anchor/<ts>/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

from .config import CONDITIONS, NGRunConfig


def _add_config_overrides(p: argparse.ArgumentParser) -> None:
    p.add_argument("--condition", choices=sorted(CONDITIONS))
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--eval-batch-size", dest="eval_batch_size", type=int)
    p.add_argument("--num-steps", dest="num_steps", type=int)
    p.add_argument("--encoding", choices=["rate", "direct"])
    p.add_argument("--task-loss", dest="task_loss", choices=["ce_rate", "ce_count"])
    p.add_argument("--n-hidden", dest="n_hidden", type=int)
    p.add_argument("--alpha", type=float)
    p.add_argument("--threshold", type=float)
    p.add_argument("--surrogate-slope", dest="surrogate_slope", type=float)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--val-size", dest="val_size", type=int)
    p.add_argument("--data-dir", dest="data_dir")
    p.add_argument("--run-dir", dest="run_dir")
    p.add_argument("--device")
    p.add_argument("--num-workers", dest="num_workers", type=int)


def _cfg_from_args(args: argparse.Namespace) -> NGRunConfig:
    overrides = {f.name: getattr(args, f.name) for f in fields(NGRunConfig)
                 if getattr(args, f.name, None) is not None}
    return NGRunConfig(**overrides)


def cmd_train(args: argparse.Namespace) -> int:
    from .runner import run_training
    try:
        run_training(_cfg_from_args(args), verbose=True)
    except RuntimeError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 3
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import torch
    from ._baseline import build_loaders, seed_everything
    from .eval import evaluate
    from .model import build_net
    from .runner import resolve_device

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = NGRunConfig.from_dict(ckpt["config"])
    if args.device:
        cfg.device = args.device
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    loaders = build_loaders(cfg.data_dir, cfg.batch_size, cfg.eval_batch_size,
                            cfg.val_size, cfg.seed, num_workers=cfg.num_workers,
                            pin_memory=False)
    net = build_net(cfg).to(device)
    net.load_state_dict(ckpt["model_state"])
    m = evaluate(net, loaders["test"], cfg, device)
    print(json.dumps({"ckpt": str(Path(args.ckpt)), "epoch": ckpt.get("epoch"), "test": m},
                     indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="neuron_games",
                                description="Neuron Games: potential-game SNN training")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train one ablation condition")
    _add_config_overrides(p_train)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="evaluate a checkpoint on the test set")
    p_eval.add_argument("--ckpt", required=True)
    p_eval.add_argument("--device", help="override the device stored in the checkpoint")
    p_eval.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
