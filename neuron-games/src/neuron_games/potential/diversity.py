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

    spk, mem: [T, B, N_l] leaf tensors (differentiable) for layer l.

    cfg.diversity_feature == "spike" (eq. 10):
        Permute [T,B,N] -> [N,B,T]; for mci / vanrossum / schoenberg smooth
        along time with smooth_exp(x, cfg.diversity_tau); flatten to [N, B*T].

    cfg.diversity_feature == "plasticity_signature" (eq. 11):
        Full feature tau_i has entries e_ij[t,b] = h_i[t,b] * eps_j[t,b] and
        would be shape [N, T*B*fan_in]. We never build it. The identity
            <tau_i, tau_k> = sum_{t,b} h_i h_k * ||eps[t,b]||^2
        means the compressed row r_i[t,b] = h_i[t,b] * ||eps[t,b]|| gives the
        same Gram matrix as the full tensor. Valid for linear kernels
        (binned / mci / vanrossum, i.e. inner-product kernels) and for
        schoenberg (RBF), since the RBF only depends on pairwise distances
        and ||tau_i - tau_k|| = ||r_i - r_k||. Not valid for count on these
        rows (it would sum h*||eps|| over time, which has no spike-count
        interpretation).

        h is computed from the leaf mem with fast_sigmoid_grad so the signal
        reaches u; eps = ctx.rec.eps[l] is treated as constant (presynaptic).
    """
    cfg = ctx.cfg
    T, B, N = spk.shape

    if cfg.diversity_feature == "spike":
        rows = spk.permute(2, 1, 0)  # [N, B, T]
        if cfg.diversity_kernel in ("mci", "vanrossum", "schoenberg"):
            rows = smooth_exp(rows, tau=cfg.diversity_tau, axis=-1)
        return rows.reshape(N, B * T)

    if cfg.diversity_feature == "plasticity_signature":
        eps = ctx.rec.eps[l].detach()                        # [T, B, fan_in]
        eps_norm = eps.pow(2).sum(dim=-1).sqrt()             # [T, B]
        h = fast_sigmoid_grad(mem - cfg.threshold, cfg.surrogate_slope)  # [T, B, N]
        rows = h * eps_norm.unsqueeze(-1)                    # [T, B, N]
        return rows.permute(2, 0, 1).reshape(N, T * B)

    raise ValueError(f"unknown diversity_feature {cfg.diversity_feature!r}")


def kernel_offdiag_value(K: torch.Tensor) -> torch.Tensor:
    """V_kernel = -R_l, R_l = sum_{i != j} K_ij / (N (N - 1)) (eq. 12-13).

    Reward form (negative redundancy); normalized so beta does not have to
    change with layer width. Returns 0 when N < 2.
    """
    if K.dim() != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be [N, N], got {tuple(K.shape)}")
    N = K.shape[0]
    if N < 2:
        return K.new_zeros(())
    off_diag = K.sum() - K.diagonal().sum()
    return -off_diag / (N * (N - 1))


def dpp_logdet_value(K: torch.Tensor, eps_scale: float) -> torch.Tensor:
    """V_DPP = log det(K_n + eps_scale * I) for one layer (eq. 14).

    K_n = K / mean(diag K) stops the log-det from being raised just by making
    every neuron fire more. Factorization uses torch.linalg.cholesky_ex; on
    failure (info != 0) we retry once with 10x eps_scale, then raise.
    Returned as a differentiable scalar: 2 * sum(log(diag(L))).
    """
    if K.dim() != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be [N, N], got {tuple(K.shape)}")
    N = K.shape[0]
    K_n = K / K.diagonal().mean()
    eye = torch.eye(N, device=K.device, dtype=K.dtype)
    L, info = torch.linalg.cholesky_ex(K_n + eps_scale * eye)
    if int(info) != 0:
        L, info = torch.linalg.cholesky_ex(K_n + (10.0 * eps_scale) * eye)
        if int(info) != 0:
            raise RuntimeError(
                f"dpp_logdet: Cholesky failed at eps={eps_scale} and 10x retry; info={int(info)}"
            )
    return 2.0 * torch.log(L.diagonal()).sum()


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
