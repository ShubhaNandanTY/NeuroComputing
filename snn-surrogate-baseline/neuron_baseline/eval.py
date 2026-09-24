"""Evaluation utilities.

Both spike-rate accuracy and CE-rate loss are computed on every example
(no drop_last), including any incomplete final batch. Supports variable
eval batch sizes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import snntorch.functional as SF

from .model import forward_pass


@torch.no_grad()
def evaluate(net: nn.Sequential, loader, num_steps: int, device: torch.device,
             loss_fn: SF.LossFunctions | None = None
             ) -> dict[str, float | int]:
    net.eval()
    if loss_fn is None:
        loss_fn = SF.ce_rate_loss()

    total_correct = 0
    total_examples = 0
    weighted_loss_sum = 0.0

    for data, targets in loader:
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs = targets.numel()
        spk_rec, _ = forward_pass(net, num_steps, data)
        # spk_rec: [T, B, C]. Matches SF.accuracy_rate: sum spikes over T, argmax.
        pred = spk_rec.sum(dim=0).argmax(dim=1)
        total_correct += int((pred == targets).sum().item())
        total_examples += bs
        loss = loss_fn(spk_rec, targets)
        weighted_loss_sum += float(loss.item()) * bs

    if total_examples == 0:
        return {"accuracy": float("nan"), "loss": float("nan"), "n": 0}
    return {
        "accuracy": total_correct / total_examples,
        "loss": weighted_loss_sum / total_examples,
        "n": total_examples,
    }
