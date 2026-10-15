"""Sanity tests for `neuron_baseline.kernels`.

Runs via pytest:
    .venv/Scripts/python.exe -m pytest tests/test_kernels.py -v
"""
from __future__ import annotations

import pytest
import torch

from neuron_baseline.kernels import (
    kernel_matrix,
    pairwise_diversity_loss,
    smooth_exp,
)

KINDS = ["count", "binned", "mci", "vanrossum", "schoenberg"]


def _rand_features(B: int = 6, D: int = 32, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, D, generator=g)


def _params_for(kind: str) -> dict:
    if kind == "schoenberg":
        return {"sigma": 1.0}
    return {}


@pytest.mark.parametrize("kind", KINDS)
def test_symmetric(kind: str) -> None:
    K = kernel_matrix(_rand_features(), kind=kind, **_params_for(kind))
    assert torch.allclose(K, K.T, atol=1e-5), f"{kind} not symmetric"


@pytest.mark.parametrize("kind", KINDS)
def test_psd(kind: str) -> None:
    K = kernel_matrix(_rand_features(), kind=kind, **_params_for(kind))
    eigvals = torch.linalg.eigvalsh(K.double())
    tol = 1e-6 * max(eigvals.max().item(), 1.0)
    assert eigvals.min().item() >= -tol, (
        f"{kind} has negative eigenvalue: {eigvals.min().item()} "
        f"(tol {-tol}, max eig {eigvals.max().item()})"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_self_similarity_positive(kind: str) -> None:
    K = kernel_matrix(_rand_features(), kind=kind, **_params_for(kind))
    assert (K.diagonal() > 0).all(), f"{kind} has non-positive K[i,i]"


@pytest.mark.parametrize("kind", ["binned", "mci", "vanrossum"])
def test_identical_gives_max_similarity(kind: str) -> None:
    x = _rand_features(B=3)
    x[1] = x[0]
    K = kernel_matrix(x, kind=kind)
    assert torch.isclose(K[0, 1], K[0, 0], atol=1e-5), (
        f"{kind}: K[0,1]={K[0,1].item()} != K[0,0]={K[0,0].item()}"
    )


def test_schoenberg_identical_is_one() -> None:
    x = _rand_features(B=3)
    x[1] = x[0]
    K = kernel_matrix(x, kind="schoenberg", sigma=1.0)
    assert torch.isclose(K[0, 1], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(K[0, 0], torch.tensor(1.0), atol=1e-6)


@pytest.mark.parametrize("kind", ["binned", "mci", "vanrossum"])
def test_orthogonal_gives_zero_off_diagonal(kind: str) -> None:
    x = torch.zeros(2, 100)
    x[0, :10] = 1.0
    x[1, 60:70] = 1.0
    if kind in ("mci", "vanrossum"):
        x = smooth_exp(x, tau=5.0)
    K = kernel_matrix(x, kind=kind)
    assert K[0, 1].item() < 1e-3, (
        f"{kind} orthogonal off-diagonal not near zero: {K[0, 1].item()}"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_differentiable_wrt_features(kind: str) -> None:
    x = _rand_features().requires_grad_(True)
    K = kernel_matrix(x, kind=kind, **_params_for(kind))
    loss = pairwise_diversity_loss(K)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), f"{kind} produced non-finite gradients"
    assert (x.grad.abs() > 0).any(), f"{kind} produced zero gradient everywhere"


def test_diversity_loss_orders_identical_gt_orthogonal() -> None:
    x_same = torch.ones(4, 8)
    K_same = kernel_matrix(x_same, kind="binned")
    loss_same = pairwise_diversity_loss(K_same).item()

    x_orth = torch.zeros(4, 8)
    for i in range(4):
        x_orth[i, i * 2 : (i + 1) * 2] = 1.0
    K_orth = kernel_matrix(x_orth, kind="binned")
    loss_orth = pairwise_diversity_loss(K_orth).item()

    assert loss_orth < loss_same, (
        f"orthogonal loss ({loss_orth}) should be less than identical ({loss_same})"
    )


def test_diversity_loss_zero_for_single_object() -> None:
    x = _rand_features(B=1)
    K = kernel_matrix(x, kind="binned")
    loss = pairwise_diversity_loss(K)
    assert loss.item() == 0.0


def test_count_kernel_uses_only_totals() -> None:
    x_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    x_b = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    K_a = kernel_matrix(x_a, kind="count")
    K_b = kernel_matrix(x_b, kind="count")
    assert torch.isclose(K_a[0, 0], K_b[0, 0])


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown kernel kind"):
        kernel_matrix(_rand_features(), kind="bogus")


def test_bad_input_shape_raises() -> None:
    with pytest.raises(ValueError, match="features must be"):
        kernel_matrix(torch.zeros(4), kind="mci")
    with pytest.raises(ValueError, match="features must be"):
        kernel_matrix(torch.zeros(4, 5, 6), kind="mci")


def test_bad_K_shape_raises() -> None:
    with pytest.raises(ValueError, match="K must be"):
        pairwise_diversity_loss(torch.zeros(4, 5))


def test_smooth_exp_preserves_dtype_and_device() -> None:
    x = torch.rand(3, 20)
    y = smooth_exp(x, tau=5.0)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device


def test_smooth_exp_causal() -> None:
    x = torch.zeros(1, 20)
    x[0, 5] = 1.0
    y = smooth_exp(x, tau=3.0)
    assert (y[0, :5] == 0).all(), "smoothing leaked into pre-spike region (non-causal)"
    assert y[0, 5].item() > 0
    assert y[0, 6].item() > 0
    assert y[0, 6].item() < y[0, 5].item()


def test_smooth_exp_tau_zero_is_identity() -> None:
    x = torch.rand(2, 10)
    y = smooth_exp(x, tau=0.0)
    assert torch.allclose(x, y)
