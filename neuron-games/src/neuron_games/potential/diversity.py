"""V - within-layer diversity on spike or plasticity feature objects (proposal §5).

OWNER: Deyvik. See Architecture_Work_14-10-2026.md, "Deyvik".

V is a *reward*: potential.py adds -beta * V to the cost. Higher V = more
diverse layer. Computed per hidden layer and summed over layers (eq. 13, 14).

The glue (`diversity_term`) is done: it builds the feature objects for every
hidden layer, forms the [N_l, N_l] Gram matrix with the existing
`neuron_baseline.kernels.kernel_matrix`, reduces it with the configured form,
and gets the learning signals by local autograd. What is left is the four
functions marked TODO.
"""
from __future__ import annotations

import torch

from .._baseline import BASELINE_ROOT  # noqa: F401  (puts neuron_baseline on sys.path)
from neuron_baseline.kernels import kernel_matrix, smooth_exp  # noqa: E402
from ..model import fast_sigmoid_grad
from .base import TermContext, TermOutput, signals_from_autograd


def feature_objects(ctx: TermContext, l: int, spk: torch.Tensor,
                    mem: torch.Tensor) -> torch.Tensor:
    """Feature object tau_i for every neuron i of layer l, as rows of [N_l, D].

    spk, mem: [T, B, N_l] leaf tensors for layer l (differentiable).

    cfg.diversity_feature == "spike" (eq. 10): tau_i = (kappa * s_i)(t).
        Permute to [N_l, B, T]; for kernels mci / vanrossum / schoenberg
        smooth along time with smooth_exp(x, cfg.diversity_tau); flatten to
        [N_l, B*T]. Same construction as neuron_baseline/train.py
        `_diversity_term`, minus the conv reshape.

    cfg.diversity_feature == "plasticity_signature" (eq. 11): tau_i = {e_ij}_j
        with e_ij[t] = h_i[t] * eps_j[t]. Do NOT materialize [N_l, B*T*fan_in].
        For a linear kernel, <tau_i, tau_k> = sum_{t,b} h_i h_k ||eps[t,b]||^2,
        so return rows h_i[t,b] * ||eps[t,b]||, flattened to [N_l, B*T], where
        h = fast_sigmoid_grad(mem - threshold, surrogate_slope) (computed from
        the *leaf* `mem` so the signal reaches u) and eps = ctx.rec.eps[l]
        (presynaptic; treat as a constant). Document in the docstring which
        kernels this shortcut is valid for (linear ones); for schoenberg the
        same rows give the exact RKHS distance as well, since it only needs
        inner products.
    """
    raise NotImplementedError("Deyvik: feature_objects (diversity.py)")


def kernel_offdiag_value(K: torch.Tensor) -> torch.Tensor:
    """V_kernel = -R_l, R_l = sum_{i != j} K_ij (eq. 12-13), for one layer.

    Normalize by N_l * (N_l - 1) like neuron_baseline.kernels.pairwise_diversity_loss
    so beta does not have to change with layer width. Return the *reward*
    (negative redundancy).
    """
    raise NotImplementedError("Deyvik: kernel_offdiag_value (diversity.py)")


def dpp_logdet_value(K: torch.Tensor, eps_scale: float) -> torch.Tensor:
    """V_DPP = log det(K + eps I) for one layer (eq. 14).

    Required normalization (the tests assume it): K_n = K / mean(diag K), so
    the log-det cannot be raised just by making every neuron fire more; then
    V = log det(K_n + eps_scale * I). Use torch.linalg.cholesky_ex; if
    info != 0 retry once with 10x eps_scale, then raise. log det =
    2 * sum(log(diag(L))). Return a differentiable scalar.
    """
    raise NotImplementedError("Deyvik: dpp_logdet_value (diversity.py)")


def mmd_value(K: torch.Tensor) -> torch.Tensor:
    """Stretch (§5.4). MMD-based dispersion of the layer's feature objects
    against a reference distribution Q_l. Not needed for the first runs; leave
    raising until the other two forms are done."""
    raise NotImplementedError("Deyvik (stretch): mmd_value (diversity.py)")


def _layer_value(ctx: TermContext, l: int, spk: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
    cfg = ctx.cfg
    feats = feature_objects(ctx, l, spk, mem)
    params = {"sigma": cfg.diversity_sigma} if cfg.diversity_kernel == "schoenberg" else {}
    K = kernel_matrix(feats, kind=cfg.diversity_kernel, **params)
    if cfg.diversity == "kernel_offdiag":
        return kernel_offdiag_value(K)
    if cfg.diversity == "dpp_logdet":
        return dpp_logdet_value(K, cfg.dpp_eps_scale)
    if cfg.diversity == "mmd":
        return mmd_value(K)
    raise ValueError(f"unknown diversity {cfg.diversity!r}")


def diversity_term(ctx: TermContext) -> TermOutput:
    """V summed over hidden layers, with local derivatives wrt s and u."""
    layers = ctx.hidden_layers

    def value(spk, mem, _weights):
        return sum(_layer_value(ctx, l, spk[l], mem[l]) for l in layers)

    return signals_from_autograd(ctx, value, wrt_spikes=True, wrt_mem=True, wrt_weights=False)


__all__ = ["feature_objects", "kernel_offdiag_value", "dpp_logdet_value", "mmd_value",
           "diversity_term", "smooth_exp", "fast_sigmoid_grad"]
