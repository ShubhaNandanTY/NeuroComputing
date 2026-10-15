"""Shapley-Taylor order-2 interaction credit over circuits (proposal §8.3). STRETCH.

OWNER: William. See Architecture_Work_14-10-2026.md, "William".

Same interface as EligibilityShapley, so condition `ng_full_st2` only swaps
this class in. An additive surrogate has zero interactions, so order 2 needs
a non-additive coalition value, e.g. the second-order surrogate

    v(S) = <G, dW_S> + 1/2 * dW_S^T H dW_S,   dW_S = sum_{i in S} dW_i

with H a cheap curvature estimate (e.g. the change in G between windows
along dW, a finite difference). Compute pairwise indices within small blocks
of neurons (cfg-defined block size), not all C(N, 2) pairs, and turn
per-neuron totals into gains with eligibility_shapley.credit_gains.
"""
from __future__ import annotations

from .base import CreditContext, CreditResult


class ShapleyTaylor:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, ctx: CreditContext) -> CreditResult:
        raise NotImplementedError("William (stretch): ShapleyTaylor (shapley_taylor.py)")
