"""Three-factor plasticity (proposal §3.2 eq. 3, §7): dW_ij = -eta * sum_t m_i[t] e_ij[t].

Every term of the cost C = -Phi enters the update the same way. A term that
depends on neuron i's spikes s_i[t] or membrane u_i[t] supplies a per-neuron
*learning signal* (its derivative with respect to s_i[t] or u_i[t]); the
synapse-local eligibility turns that signal into a weight change:

    ds_i[t]/dW_ij = h_i[t] * eps_j[t]        (spike pathway)
    du_i[t]/dW_ij = eps_j[t]                 (membrane pathway)

These are exact for the vendored LIF cell (detached reset, same-step input),
so no gradient is ever propagated through the recurrence or across layers.
Cross-layer credit comes only from the broadcast task signal
(modulator/random_feedback.py).

Shapes: signals are [T, B, N_l]; eps is [T, B, fan_in_l]; the result is
[N_l, fan_in_l], computed as one matmul over the flattened (T, B) axis. The
[T, B, N_l, fan_in_l] eligibility tensor is never formed.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import SNNRecord


@dataclass
class LayerSignals:
    """Learning signals for one layer, all already scaled by their coefficient
    in C. Any field may be None (= zero)."""
    spike: torch.Tensor | None = None    # dC/ds_i[t], [T, B, N_l]
    mem: torch.Tensor | None = None      # dC/du_i[t], [T, B, N_l]
    weight: torch.Tensor | None = None   # direct dC/dW (not through dynamics), [N_l, fan_in_l]

    def __add__(self, other: "LayerSignals") -> "LayerSignals":
        def _sum(a, b):
            if a is None:
                return b
            if b is None:
                return a
            return a + b
        return LayerSignals(_sum(self.spike, other.spike), _sum(self.mem, other.mem),
                            _sum(self.weight, other.weight))

    def scaled(self, c: float) -> "LayerSignals":
        return LayerSignals(*(None if x is None else c * x
                              for x in (self.spike, self.mem, self.weight)))


def membrane_drive(sig: LayerSignals, h: torch.Tensor) -> torch.Tensor | None:
    """Total per-neuron signal on u_i[t]: spike * h + mem. [T, B, N_l] or None."""
    out = None
    if sig.spike is not None:
        out = sig.spike * h
    if sig.mem is not None:
        out = sig.mem if out is None else out + sig.mem
    return out


def eligibility_grad(drive: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """sum_{t,b} drive[t,b,i] * eps[t,b,j] -> [N_l, fan_in_l]."""
    n_out, n_in = drive.shape[-1], eps.shape[-1]
    return drive.reshape(-1, n_out).transpose(0, 1) @ eps.reshape(-1, n_in)


def layer_grad(rec: SNNRecord, l: int, sig: LayerSignals,
               like: torch.Tensor) -> torch.Tensor:
    """dC/dW_l from one layer's signals. `like` gives shape/dtype/device."""
    if not rec.has_traces:
        raise ValueError("SNNRecord has no eligibility traces; run forward(record_traces=True)")
    g = torch.zeros_like(like)
    drive = membrane_drive(sig, rec.h[l])
    if drive is not None:
        g = g + eligibility_grad(drive, rec.eps[l])
    if sig.weight is not None:
        g = g + sig.weight
    return g


def three_factor_grads(rec: SNNRecord, signals: list[LayerSignals],
                       weights: list[torch.Tensor]) -> list[torch.Tensor]:
    """dC/dW_l for every layer."""
    return [layer_grad(rec, l, signals[l], weights[l]) for l in range(len(weights))]
