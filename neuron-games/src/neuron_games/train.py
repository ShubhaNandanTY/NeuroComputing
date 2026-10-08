"""Training loop. Row 2 (`bp_anchor`): surrogate-gradient BPTT through the
vendored LIF recurrence, Adam. Structure mirrors neuron_baseline.train:
val eval after every epoch, `best.pt` on val accuracy, `last.pt` always.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import NGRunConfig
from .encoding import encode
from .eval import evaluate
from .model import build_net
from .potential.task import spike_count_predictions, task_loss


def _train_one_epoch_bptt(net, loader, optimizer, cfg: NGRunConfig, device) -> dict:
    net.train()
    loss_fn = task_loss(cfg.task_loss)
    out: dict[str, list] = {"loss": [], "acc": [], "rates": []}
    for data, targets in loader:
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        rec = net(encode(data, cfg.num_steps, cfg.encoding))
        loss = loss_fn(rec.out_spk, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        out["loss"].append(float(loss.item()))
        out["acc"].append(float(
            (spike_count_predictions(rec.out_spk) == targets).float().mean().item()))
        out["rates"].append([float(s.detach().mean().item()) for s in rec.spk])
    return out


def save_checkpoint(path: Path, net, optimizer, cfg: NGRunConfig, epoch: int,
                    val_accuracy: float, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(cfg),
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "extra": extra or {},
        },
        str(path),
    )


def train(cfg: NGRunConfig, loaders: dict, device: torch.device, run_dir: Path) -> dict:
    if cfg.condition != "bp_anchor":
        raise NotImplementedError(f"condition {cfg.condition!r} has no training rule yet")

    net = build_net(cfg).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, betas=cfg.adam_betas)

    # Key names match neuron_baseline so its plotting code works unchanged.
    history: dict[str, list] = {
        "train_loss_per_batch": [],
        "train_acc_per_batch":  [],
        "epoch_train_loss":     [],
        "epoch_train_acc":      [],
        "epoch_train_firing_rate": [],   # [epoch][layer]
        "epoch_val_loss":       [],
        "epoch_val_acc":        [],
        "epoch_val_firing_rate": [],
        "epoch_time_s":         [],
    }
    best_val_acc = -1.0

    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        ep = _train_one_epoch_bptt(net, loaders["train"], optimizer, cfg, device)
        val = evaluate(net, loaders["val"], cfg, device)
        dt = time.perf_counter() - t0

        nb = max(len(ep["loss"]), 1)
        rates = [sum(r[l] for r in ep["rates"]) / nb for l in range(len(net.layers))]
        history["train_loss_per_batch"].extend(ep["loss"])
        history["train_acc_per_batch"].extend(ep["acc"])
        history["epoch_train_loss"].append(sum(ep["loss"]) / nb)
        history["epoch_train_acc"].append(sum(ep["acc"]) / nb)
        history["epoch_train_firing_rate"].append(rates)
        history["epoch_val_loss"].append(val["loss"])
        history["epoch_val_acc"].append(val["accuracy"])
        history["epoch_val_firing_rate"].append(val["firing_rate"])
        history["epoch_time_s"].append(dt)

        print(f"[epoch {epoch+1}/{cfg.epochs}] "
              f"train_loss={history['epoch_train_loss'][-1]:.4f} "
              f"train_acc={history['epoch_train_acc'][-1]:.4f} "
              f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f} "
              f"rate(h,o)=({rates[0]:.3f},{rates[-1]:.3f}) "
              f"time={dt:.1f}s")

        save_checkpoint(run_dir / "checkpoints" / "last.pt",
                        net, optimizer, cfg, epoch + 1, val["accuracy"])
        if val["accuracy"] > best_val_acc:
            best_val_acc = val["accuracy"]
            save_checkpoint(run_dir / "checkpoints" / "best.pt",
                            net, optimizer, cfg, epoch + 1, val["accuracy"],
                            extra={"selection": "best_val_acc"})

    return {"history": history, "best_val_acc": best_val_acc,
            "net": net, "optimizer": optimizer}
