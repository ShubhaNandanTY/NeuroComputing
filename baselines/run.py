"""Train one backprop baseline: seed -> train -> reload best.pt -> test once -> results.json.

Usage (from baselines/):
    python run.py --model cnn --seed 20260923 --epochs 1
"""
import argparse
import json
import platform
import random
import time
from pathlib import Path

import torch
from torch import nn

from common.data import build_loaders
from common.train import evaluate, get_device, train
from models.cnn import CNN

MODELS = {"cnn": CNN}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--seed", type=int, default=20260923)  # snn-surrogate-baseline default
    p.add_argument("--epochs", type=int, default=25)      # N_BP in the energy plan
    p.add_argument("--lr", type=float, default=1e-3)      # Adam default; Deyvik's 1e-2 was tuned for SNNs, not CNNs
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--data-dir", default="data")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU, CUDA and MPS generators


def fingerprint(device: torch.device) -> dict:
    fp = {"python": platform.python_version(), "platform": platform.platform(),
          "torch": torch.__version__, "device": str(device)}
    if device.type == "cuda":
        fp["device_name"] = torch.cuda.get_device_name(0)
    return fp


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    run_dir = Path("results") / args.model / f"seed{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "fingerprint": fingerprint(device)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    loaders = build_loaders(args.data_dir, args.batch_size, args.eval_batch_size,
                            args.val_size, args.seed, pin_memory=device.type == "cuda")

    model = MODELS[args.model]().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[run] model={args.model} params={n_params} device={device} run_dir={run_dir}")

    t0 = time.perf_counter()
    out = train(model, loaders, args.epochs, args.lr, device, run_dir)
    train_wall_s = time.perf_counter() - t0

    # Test once, on the checkpoint selected by validation accuracy (not the last epoch).
    best = MODELS[args.model]().to(device)
    best.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    t0 = time.perf_counter()
    test = evaluate(best, loaders["test"], nn.CrossEntropyLoss(), device)
    test_wall_s = time.perf_counter() - t0

    results = {
        "model": args.model,
        "seed": args.seed,
        "n_params": n_params,
        "history": out["history"],
        "best_val_accuracy": out["best_val_acc"],
        "test": test,
        "wall_time_s": {"train": train_wall_s, "test": test_wall_s},
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[done] test_acc={test['accuracy']:.4f} (n={test['n']}) "
          f"train_time={train_wall_s:.1f}s -> {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
