"""Phi composer (proposal eq. 8): Phi = -J + beta*V - mu*E - lam*R, E = E_met + alpha_bar*E_bar.

`evaluate_potential` runs every active non-task term on the window, scales its
local derivatives by its coefficient in the cost C = -Phi, and returns the
summed per-layer signals plus a breakdown for logging. The task term J is
handled by the trainer, because its hidden-layer signal is a broadcast
(modulator/random_feedback.py), not a local derivative.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..learning_rule import LayerSignals
from .base import TermContext, TermFn, TermOutput
from .barrier import barrier_term
from .diversity import diversity_term
from .metabolic import metabolic_term
from .regularizer import regularizer_term


@dataclass
class PotentialEval:
    signals: list[LayerSignals]                         # summed, already scaled into C
    components: dict[str, float] = field(default_factory=dict)  # raw term values: V, E_met, E_bar, R
    info: dict[str, float] = field(default_factory=dict)


def active_terms(cfg) -> list[tuple[str, float, TermFn]]:
    """(name, coefficient in C, term function) for every enabled term."""
    terms: list[tuple[str, float, TermFn]] = []
    if cfg.diversity != "none":
        terms.append(("V", -cfg.beta, diversity_term))
    if cfg.metabolic != "none":
        terms.append(("E_met", cfg.mu, metabolic_term))
    if cfg.barrier != "none":
        terms.append(("E_bar", cfg.mu * cfg.alpha_bar, barrier_term))
    if cfg.lam != 0.0:
        terms.append(("R", cfg.lam, regularizer_term))
    return terms


def evaluate_potential(ctx: TermContext) -> PotentialEval:
    signals = [LayerSignals() for _ in range(ctx.n_layers)]
    out = PotentialEval(signals=signals)
    for name, coef, fn in active_terms(ctx.cfg):
        res: TermOutput = fn(ctx)
        if len(res.signals) != ctx.n_layers:
            raise ValueError(f"term {name} returned {len(res.signals)} layer signals, "
                             f"expected {ctx.n_layers}")
        out.components[name] = float(res.value)
        for k, v in res.info.items():
            out.info[f"{name}/{k}"] = float(v)
        for l in range(ctx.n_layers):
            signals[l] = signals[l] + res.signals[l].scaled(coef)
    return out


def phi_value(J: float, components: dict[str, float], cfg) -> float:
    """Phi = -J + beta*V - mu*(E_met + alpha_bar*E_bar) - lam*R (missing terms are 0)."""
    V = components.get("V", 0.0)
    E = components.get("E_met", 0.0) + cfg.alpha_bar * components.get("E_bar", 0.0)
    R = components.get("R", 0.0)
    return -J + cfg.beta * V - cfg.mu * E - cfg.lam * R
