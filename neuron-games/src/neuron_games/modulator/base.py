"""Credit contract for Eligibility-Shapley style modulation (proposal §8).

Each window the trainer builds a `CreditContext` and asks the configured
credit method for a `CreditResult`:

- phi[l]:   [N_l] credit per neuron of layer l (None for layers not credited)
- gains[l]: [N_l] multiplicative gain on that neuron's *task* learning signal
            (None = gain 1). Only applied when cfg.credit_modulates is True;
            otherwise phi is logged for post-hoc analysis only.

The trainer applies a gain g_i by scaling row i of the layer's task-signal
gradient, which is exactly m_i(t) = g_i * L_i(t) in the three-factor rule
(the gain is per postsynaptic neuron, so it factors out of the sum over t, b).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch


@dataclass
class CreditContext:
    # G^(n): estimated sensitivity of Phi to each weight on this window,
    # G_l = -dC/dW_l from the three-factor rule *before* any gain. [N_l, fan_in_l]
    G: list[torch.Tensor]
    # dW_i^(n-1): the action each neuron actually took on the previous window,
    # W_after - W_before. None on the first window. [N_l, fan_in_l]
    prev_delta_w: list[torch.Tensor | None]
    # Task-only part of dC/dW_l (the part gains multiply). [N_l, fan_in_l]
    task_grads: list[torch.Tensor]
    layers: list[int]                    # which layers to credit (cfg.credit_layers)
    cfg: object
    window: int = 0


@dataclass
class CreditResult:
    phi: list[torch.Tensor | None] = field(default_factory=list)
    gains: list[torch.Tensor | None] = field(default_factory=list)


class CreditMethod(Protocol):
    def __call__(self, ctx: CreditContext) -> CreditResult: ...


def build_credit(cfg) -> CreditMethod | None:
    if cfg.credit == "none":
        return None
    if cfg.credit == "eligibility_shapley":
        from .eligibility_shapley import EligibilityShapley
        return EligibilityShapley(cfg)
    if cfg.credit == "shapley_taylor":
        from .shapley_taylor import ShapleyTaylor
        return ShapleyTaylor(cfg)
    raise ValueError(f"unknown credit {cfg.credit!r}")
