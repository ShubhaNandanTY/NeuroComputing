"""V - diversity (potential/diversity.py). OWNER: Deyvik.

These tests are the acceptance criteria. They skip until the functions stop
raising NotImplementedError. Add more as you see fit.
"""
from __future__ import annotations

import pytest
import torch
from helpers import needs_impl, small_ctx

from neuron_games.config import NGRunConfig
from neuron_games.potential.diversity import (diversity_term, dpp_logdet_value, feature_objects,
                                              kernel_offdiag_value)


def _cfg(**kw):
    return NGRunConfig.for_condition("tf_div", n_in=12, n_hidden=9, n_out=4, device="cpu", **kw)


@needs_impl
@pytest.mark.parametrize("feature", ["spike", "plasticity_signature"])
def test_feature_objects_shape(feature) -> None:
    ctx = small_ctx(_cfg(diversity_feature=feature))
    f = feature_objects(ctx, 0, ctx.rec.spk[0], ctx.rec.mem[0])
    assert f.dim() == 2 and f.shape[0] == 9 and torch.isfinite(f).all()


@needs_impl
def test_plasticity_signature_matches_explicit_traces() -> None:
    """Linear-kernel Gram from the shortcut == Gram of the explicit e_ij traces."""
    ctx = small_ctx(_cfg(diversity_feature="plasticity_signature", diversity_kernel="binned"))
    f = feature_objects(ctx, 0, ctx.rec.spk[0], ctx.rec.mem[0])
    h, eps = ctx.rec.h[0], ctx.rec.eps[0]                       # [T,B,N], [T,B,n_in]
    e = h.unsqueeze(-1) * eps.unsqueeze(-2)                     # [T,B,N,n_in]
    explicit = e.permute(2, 0, 1, 3).reshape(9, -1)
    torch.testing.assert_close(f @ f.T, explicit @ explicit.T, rtol=1e-4, atol=1e-6)


@needs_impl
def test_kernel_offdiag_rewards_orthogonal_features() -> None:
    same = torch.ones(4, 6)
    orth = torch.eye(4, 6)
    assert kernel_offdiag_value(orth @ orth.T) > kernel_offdiag_value(same @ same.T)
    assert kernel_offdiag_value(orth @ orth.T) == pytest.approx(0.0)


@needs_impl
def test_dpp_logdet_finite_on_singular_and_prefers_diverse() -> None:
    same = torch.ones(5, 8)
    diverse = torch.eye(5, 8)
    v_same = dpp_logdet_value(same @ same.T, 1e-4)        # rank 1: needs the jitter
    v_div = dpp_logdet_value(diverse @ diverse.T, 1e-4)
    assert torch.isfinite(v_same) and torch.isfinite(v_div)
    assert v_div > v_same


@needs_impl
def test_dpp_logdet_grad_matches_autograd_slogdet() -> None:
    torch.manual_seed(0)
    X = torch.randn(6, 10, dtype=torch.float64, requires_grad=True)
    K = X @ X.T
    v = dpp_logdet_value(K, 1e-4)
    (g,) = torch.autograd.grad(v, X)
    X2 = X.detach().clone().requires_grad_(True)
    K2 = X2 @ X2.T
    ref = torch.linalg.slogdet(K2 / K2.diagonal().mean()
                               + 1e-4 * torch.eye(6, dtype=torch.float64))[1]
    (g2,) = torch.autograd.grad(ref, X2)
    torch.testing.assert_close(g, g2, rtol=1e-6, atol=1e-8)


@needs_impl
@pytest.mark.parametrize("form", ["kernel_offdiag", "dpp_logdet"])
def test_diversity_term_signals(form) -> None:
    ctx = small_ctx(_cfg(diversity=form))
    out = diversity_term(ctx)
    assert torch.isfinite(out.value)
    s0 = out.signals[0]
    assert s0.spike is not None and s0.spike.shape == ctx.rec.spk[0].shape
    assert torch.isfinite(s0.spike).all() and s0.spike.abs().sum() > 0
    # output layer is not diversified
    assert out.signals[1].spike is None or out.signals[1].spike.abs().sum() == 0
