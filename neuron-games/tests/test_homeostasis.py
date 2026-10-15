"""E_met, E_bar, R (potential/metabolic.py, barrier.py, regularizer.py). OWNER: Joohyun.

These tests are the acceptance criteria. They skip until the functions stop
raising NotImplementedError. Add more as you see fit.
"""
from __future__ import annotations

import math

import pytest
import torch
from helpers import needs_impl, small_ctx

from neuron_games.config import NGRunConfig
from neuron_games.potential.barrier import barrier_term, barrier_value, phi_t, project_
from neuron_games.potential.metabolic import metabolic_term, metabolic_value
from neuron_games.potential.regularizer import regularizer_term, regularizer_value


def _cfg(**kw):
    return NGRunConfig.for_condition("tf_div_energy", n_in=12, n_hidden=9, n_out=4,
                                     device="cpu", **kw)


@needs_impl
def test_metabolic_value_hand_computed() -> None:
    cfg = _cfg(c_s=1.0, c_u=0.5, c_w=0.25, p_w=2.0)
    spk = [torch.tensor([[[1.0, 0.0]], [[1.0, 1.0]]])]          # [T=2, B=1, N=2]: 3 spikes
    mem = [torch.tensor([[[1.0, 2.0]], [[0.0, 1.0]]])]          # sum u^2 = 6
    w = [torch.tensor([[1.0, -2.0]])]                            # sum w^2 = 5
    v = metabolic_value(spk, mem, w, cfg)
    assert float(v) == pytest.approx(1.0 * 3 + 0.5 * 6 + 0.25 * 5)


@needs_impl
def test_metabolic_term_signals() -> None:
    cfg = _cfg(c_s=0.1, c_u=0.0, c_w=0.0)
    ctx = small_ctx(cfg)
    out = metabolic_term(ctx)
    B = ctx.rec.spk[0].shape[1]
    torch.testing.assert_close(out.signals[0].spike, torch.full_like(ctx.rec.spk[0], 0.1 / B))


@needs_impl
def test_phi_t_continuous_and_finite() -> None:
    for t in (0.5, 1.0, 5.0, 50.0):
        z0 = -1.0 / t ** 2
        left, right = phi_t(torch.tensor(z0 - 1e-7), t), phi_t(torch.tensor(z0 + 1e-7), t)
        assert float(left) == pytest.approx(float(right), abs=1e-4)
        assert torch.isfinite(phi_t(torch.tensor([-10.0, 0.0, 3.0, 1e6]), t)).all()
        z = torch.tensor(-4.0)
        assert float(phi_t(z, t)) == pytest.approx(-math.log(4.0) / t, rel=1e-6)


@needs_impl
@pytest.mark.parametrize("form", ["quadratic", "log"])
def test_barrier_zero_grad_deep_inside_and_pushes_back_outside(form) -> None:
    cfg = _cfg(barrier=form, w_min=-1.0, w_max=1.0, u_min=-2.0, u_max=2.0)
    w_in = [torch.zeros(2, 3, requires_grad=True)]
    w_out = [torch.full((2, 3), 1.5, requires_grad=True)]
    mem = [torch.zeros(4, 2, 2)]
    spk = [torch.zeros(4, 2, 2)]
    (g_out,) = torch.autograd.grad(barrier_value(spk, mem, w_out, cfg, epoch=0), w_out)
    assert (g_out > 0).all()                     # gradient descent moves w back below w_max
    if form == "quadratic":
        (g_in,) = torch.autograd.grad(barrier_value(spk, mem, w_in, cfg, epoch=0), w_in)
        assert g_in.abs().max() == 0


@needs_impl
@pytest.mark.parametrize("form", ["quadratic", "log"])
def test_project_keeps_weights_in_bounds(form) -> None:
    cfg = _cfg(barrier=form)
    w = torch.nn.Parameter(torch.tensor([[-3.0, 0.2, 7.0]]))
    project_([w], cfg)
    assert w.min() >= cfg.w_min and w.max() <= cfg.w_max
    if form == "log":
        assert w.min() > cfg.w_min and w.max() < cfg.w_max
    assert float(w[0, 1]) == pytest.approx(0.2)


@needs_impl
def test_barrier_term_has_mem_and_weight_signals() -> None:
    ctx = small_ctx(_cfg(barrier="quadratic", u_max=0.1))
    out = barrier_term(ctx)
    assert out.signals[0].mem is not None and out.signals[0].mem.abs().sum() > 0
    assert out.signals[0].weight is not None


@needs_impl
def test_regularizer() -> None:
    w = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0]])]
    assert float(regularizer_value(w)) == pytest.approx(14.0)
    ctx = small_ctx(_cfg(lam=1.0))
    out = regularizer_term(ctx)
    torch.testing.assert_close(out.signals[1].weight, 2.0 * ctx.weights[1])
