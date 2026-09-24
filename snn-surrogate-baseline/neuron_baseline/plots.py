"""Loss / accuracy plots. Kept minimal — the point is to eyeball convergence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless-safe for CLI runs
import matplotlib.pyplot as plt


def save_training_plots(history: dict[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss_per_batch"], lw=0.7, label="train (per batch)")
    ep_losses = history["epoch_train_loss"]
    if ep_losses:
        n_batches_per_epoch = max(1, len(history["train_loss_per_batch"]) // len(ep_losses))
        ax.plot(
            [n_batches_per_epoch * (i + 1) - 1 for i in range(len(ep_losses))],
            ep_losses, "o-", label="train (epoch mean)",
        )
    if history["epoch_val_loss"]:
        n_batches_per_epoch = max(1, len(history["train_loss_per_batch"]) // max(1, len(history["epoch_val_loss"])))
        ax.plot(
            [n_batches_per_epoch * (i + 1) - 1 for i in range(len(history["epoch_val_loss"]))],
            history["epoch_val_loss"], "s-", label="val (epoch)",
        )
    ax.set_xlabel("training batch")
    ax.set_ylabel("SF.ce_rate_loss (mean over T)")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = out_dir / "loss.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_acc_per_batch"], lw=0.7, label="train (per batch)")
    if history["epoch_val_acc"]:
        n_batches_per_epoch = max(1, len(history["train_acc_per_batch"]) // max(1, len(history["epoch_val_acc"])))
        ax.plot(
            [n_batches_per_epoch * (i + 1) - 1 for i in range(len(history["epoch_val_acc"]))],
            history["epoch_val_acc"], "s-", label="val (epoch)",
        )
    ax.set_xlabel("training batch")
    ax.set_ylabel("SF.accuracy_rate (spike-count argmax)")
    ax.set_title("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = out_dir / "accuracy.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    return written
