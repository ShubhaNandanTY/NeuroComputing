"""Full-set evaluation: every example (no drop_last), size-weighted loss,
plus mean firing rate per layer (spikes per neuron per step)."""
from __future__ import annotations

import torch

from .encoding import encode
from .model import FCSNN
from .potential.task import spike_count_predictions, task_loss


@torch.no_grad()
def evaluate(net: FCSNN, loader, cfg, device: torch.device) -> dict:
    net.eval()
    loss_fn = task_loss(cfg.task_loss)
    n = correct = 0
    loss_sum = 0.0
    rate_sums = [0.0 for _ in net.layers]

    for data, targets in loader:
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs = targets.numel()
        rec = net(encode(data, cfg.num_steps, cfg.encoding))
        correct += int((spike_count_predictions(rec.out_spk) == targets).sum().item())
        loss_sum += float(loss_fn(rec.out_spk, targets).item()) * bs
        for l, s in enumerate(rec.spk):
            rate_sums[l] += float(s.mean().item()) * bs
        n += bs

    if n == 0:
        return {"accuracy": float("nan"), "loss": float("nan"), "n": 0,
                "firing_rate": [float("nan")] * len(rate_sums)}
    return {"accuracy": correct / n, "loss": loss_sum / n, "n": n,
            "firing_rate": [r / n for r in rate_sums]}
