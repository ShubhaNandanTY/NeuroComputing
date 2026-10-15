"""E_bar - barrier homeostasis on weights and membranes (proposal §6.2, eq. 16-18).

OWNER: Joohyun. See Architecture_Work_14-10-2026.md, "Joohyun".

E_bar is a *cost*: potential.py adds +mu * alpha_bar * E_bar to C. After
every optimizer step the trainer calls `project_` (when cfg.barrier != "none")
so weights stay inside (w_min, w_max).

Two forms (cfg.barrier):
- "quadratic": penalty outside the box + hard projection (plan §2.3 Phase 1).
- "log": eq. 17 through the log-barrier extension phi_t (Kervadec et al.
  2019, arXiv:1904.04205), t = barrier_t0 * barrier_mu ** epoch, so an
  infeasible membrane value gives a large finite cost instead of inf/NaN.
"""
from __future__ import annotations

import torch

from .base import TermContext, TermOutput, signals_from_autograd


def phi_t(z: torch.Tensor, t: float) -> torch.Tensor:
    """Log-barrier extension for a constraint z <= 0 (elementwise).

    phi_t(z) = -log(-z) / t                                  if z <= -1/t^2
             = t*z - log(1/t^2)/t + 1/t                      otherwise
    (C^1 at the switch point; matches Kervadec eq. 7.) Must not produce
    inf/NaN for any finite z, including z > 0.
    """
    raise NotImplementedError("Joohyun: phi_t (barrier.py)")


def barrier_value(spk: list[torch.Tensor], mem: list[torch.Tensor],
                  weights: list[torch.Tensor], cfg, epoch: int) -> torch.Tensor:
    """E_bar on weights (all layers) and membranes (all layers, all steps).

    quadratic:  sum relu(w - w_max)^2 + relu(w_min - w)^2
              + sum relu(u - u_max)^2 + relu(u_min - u)^2
    log:        eq. 17 with every log replaced by phi_t on the matching
                constraint, e.g. -log(w - w_min) -> phi_t(w_min - w, t).
    Membrane part: sum over neurons and time, mean over batch (base.py
    convention). Bounds: cfg.w_min/w_max/u_min/u_max.
    """
    raise NotImplementedError("Joohyun: barrier_value (barrier.py)")


@torch.no_grad()
def project_(weights: list[torch.Tensor], cfg) -> None:
    """In-place projection after each optimizer step.

    quadratic: clamp to [w_min, w_max].
    log: clamp to the open interval [w_min + d, w_max - d] with a small d
         (e.g. 1e-6 * (w_max - w_min)) so the barrier stays finite.
    `weights` are the live nn.Parameter tensors: modify with .clamp_().
    """
    raise NotImplementedError("Joohyun: project_ (barrier.py)")


def barrier_term(ctx: TermContext) -> TermOutput:
    return signals_from_autograd(
        ctx, lambda s, u, w: barrier_value(s, u, w, ctx.cfg, ctx.epoch),
        wrt_spikes=False, wrt_mem=True, wrt_weights=True)
