"""The contract every potential term implements.

A term is a function `term(ctx: TermContext) -> TermOutput`.

- `TermOutput.value` is the term's own scalar value on this window (V, E_met,
  E_bar, or R; *not* multiplied by beta / mu / lam - potential.py does that).
- `TermOutput.signals[l]` holds the derivatives of that value with respect to
  layer l's spikes s[t], membranes u[t], and (directly) weights W. Sign
  convention: plain derivatives of the term's value. potential.py applies the
  sign and coefficient from C = J - beta*V + mu*(E_met + alpha_bar*E_bar) + lam*R.

The easy way to write a term is `signals_from_autograd`: write the value as
a differentiable function of the recorded spikes / membranes / weights and
let autograd take the *local* derivatives. That autograd call only touches
the recorded tensors (treated as leaves); it never runs back through the LIF
recurrence or between layers, so it stays within the three-factor rule.

Normalization convention, so all terms are on the same scale as J (which is
a batch mean): sum over neurons and time, mean over the batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from ..learning_rule import LayerSignals
from ..model import SNNRecord


@dataclass
class TermContext:
    rec: SNNRecord                      # this window's U, S, E (with traces)
    weights: list[torch.Tensor]         # W_l, detached views of the parameters
    cfg: object                         # NGRunConfig
    epoch: int = 0                      # for schedules (log-barrier t)
    state: object | None = None         # GameState (Z: running rates), may be None in tests

    @property
    def n_layers(self) -> int:
        return len(self.weights)

    @property
    def hidden_layers(self) -> list[int]:
        return list(range(self.n_layers - 1))


@dataclass
class TermOutput:
    value: torch.Tensor                                  # scalar, detached
    signals: list[LayerSignals] = field(default_factory=list)
    info: dict = field(default_factory=dict)             # optional extra scalars to log

    @classmethod
    def zero(cls, ctx: TermContext) -> "TermOutput":
        return cls(value=torch.zeros((), device=ctx.weights[0].device),
                   signals=[LayerSignals() for _ in range(ctx.n_layers)])


TermFn = Callable[[TermContext], TermOutput]
ValueFn = Callable[[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]], torch.Tensor]


def signals_from_autograd(ctx: TermContext, value_fn: ValueFn,
                          wrt_spikes: bool = True, wrt_mem: bool = True,
                          wrt_weights: bool = True) -> TermOutput:
    """Local derivatives of `value_fn(spk, mem, weights) -> scalar`.

    spk[l], mem[l] are leaf copies of the recorded [T, B, N_l] tensors and
    weights[l] a leaf copy of W_l. Disable the paths a term does not use to
    save work; disabled paths come back as None.
    """
    with torch.enable_grad():
        spk = [s.detach().clone().requires_grad_(wrt_spikes) for s in ctx.rec.spk]
        mem = [u.detach().clone().requires_grad_(wrt_mem) for u in ctx.rec.mem]
        ws = [w.detach().clone().requires_grad_(wrt_weights) for w in ctx.weights]
        value = value_fn(spk, mem, ws)
        leaves = ([*spk] if wrt_spikes else []) + ([*mem] if wrt_mem else []) + \
                 ([*ws] if wrt_weights else [])
        grads = torch.autograd.grad(value, leaves, allow_unused=True) if leaves else []

    grads = list(grads)
    L = ctx.n_layers
    g_spk = grads[:L] if wrt_spikes else [None] * L
    grads = grads[L:] if wrt_spikes else grads
    g_mem = grads[:L] if wrt_mem else [None] * L
    grads = grads[L:] if wrt_mem else grads
    g_w = grads[:L] if wrt_weights else [None] * L
    return TermOutput(value=value.detach(),
                      signals=[LayerSignals(g_spk[l], g_mem[l], g_w[l]) for l in range(L)])
