"""E_met - metabolic energy budget (proposal §6.1, eq. 15).

OWNER: Joohyun. See Architecture_Work_14-10-2026.md, "Joohyun".

E_met is a *cost*: potential.py adds +mu * E_met to C.
"""
from __future__ import annotations

import torch

from .base import TermContext, TermOutput, signals_from_autograd


def metabolic_value(spk: list[torch.Tensor], mem: list[torch.Tensor],
                    weights: list[torch.Tensor], cfg) -> torch.Tensor:
    """E_met = c_s * sum s + c_u * sum u^2 + c_w * sum |w|^p   (eq. 15)

    spk[l], mem[l]: [T, B, N_l]; weights[l]: [N_l, fan_in_l]; all layers.
    Sum over neurons and time, mean over the batch (base.py convention);
    the weight term has no batch axis. Coefficients: cfg.c_s, cfg.c_u,
    cfg.c_w, exponent cfg.p_w. Must return a differentiable scalar.
    """
    raise NotImplementedError("Joohyun: metabolic_value (metabolic.py)")


def metabolic_term(ctx: TermContext) -> TermOutput:
    return signals_from_autograd(
        ctx, lambda s, u, w: metabolic_value(s, u, w, ctx.cfg),
        wrt_spikes=True, wrt_mem=True, wrt_weights=True)
