"""J — task objective on output spikes, plus the matching accuracy rule.

Implemented directly (no snnTorch) but numerically identical to snnTorch's:

- `ce_rate`  == `SF.ce_rate_loss()`:  mean over t of CE(spk[t], y).
  Default, because it has a per-step learning signal
  delta[t] = softmax(spk[t]) - onehot(y), which is what the e-prop
  modulator in plan §1.3 broadcasts.
- `ce_count` == `SF.ce_count_loss()`: CE(sum_t spk[t], y). Plan §2.1's
  original default; kept selectable.

Accuracy == `SF.accuracy_rate`: argmax of spike counts summed over T.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ce_rate_loss(spk: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """spk: [T, B, C]."""
    T, B, C = spk.shape
    return F.cross_entropy(spk.reshape(T * B, C), targets.repeat(T))


def ce_count_loss(spk: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(spk.sum(dim=0), targets)


TASK_LOSSES = {"ce_rate": ce_rate_loss, "ce_count": ce_count_loss}


def task_loss(kind: str):
    try:
        return TASK_LOSSES[kind]
    except KeyError:
        raise ValueError(f"unknown task_loss {kind!r}; known: {sorted(TASK_LOSSES)}") from None


def spike_count_predictions(spk: torch.Tensor) -> torch.Tensor:
    return spk.sum(dim=0).argmax(dim=1)
