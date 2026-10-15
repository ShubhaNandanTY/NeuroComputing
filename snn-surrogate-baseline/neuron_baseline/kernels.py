"""Feature-agnostic kernels for diversity regularization.

The module operates on a `[B, D]` feature tensor and returns a `[B, B]`
positive semi-definite similarity matrix. It does not know whether the B
objects are examples in a batch, neurons in a layer, or anything else --
that is the caller's problem (see docs/kernel-diversity-plan.md §4.1).

Five kernels, following Park et al. 2013 (arxiv 1302.5964):

- ``count``      -- ``K[i,j] = n_i * n_j``, where ``n = features.sum(-1)``.
                    Timing-blind. Cheapest possible sanity baseline.
- ``binned``     -- ``K[i,j] = <f_i, f_j>``. Linear kernel; caller passes
                    pre-binned spike counts.
- ``mci``        -- Memoryless cross-intensity: linear kernel on
                    (pre-smoothed) spike traces. Same math as ``binned``;
                    the alias signals caller intent -- features were
                    prepared via `smooth_exp`.
- ``vanrossum``  -- mCI restricted to an exponential smoothing window.
                    Alias of ``mci`` with a documented smoothing choice.
- ``schoenberg`` -- Gaussian on the mCI distance:
                    ``K[i,j] = exp(-||f_i - f_j||^2 / (2 sigma^2))``.
                    Universal kernel (dense in continuous functions).

Higher-order kernels and the Victor-Purpura distance are out of scope
(higher-order deferred; Victor-Purpura is not PSD -- cannot serve as a
loss term).

For mCI / van Rossum, the caller pre-smooths with `smooth_exp` before
calling `kernel_matrix`. Keeping smoothing outside the kernel keeps the
module feature-agnostic: the same code works on spike trains, eligibility
traces, or any other per-object feature.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_LINEAR_KINDS = ("binned", "mci", "vanrossum")
_ALL_KINDS = ("count", "binned", "mci", "vanrossum", "schoenberg")


def smooth_exp(x: torch.Tensor, tau: float, axis: int = -1) -> torch.Tensor:
    """Causal exponential smoothing along ``axis``.

    ``y[t] = sum_{k>=0} alpha * (1-alpha)^k * x[t-k]``, with
    ``alpha = 1 - exp(-1/tau)``. Truncated at 5 tau (~99% mass).

    Autograd-safe (no in-place ops).
    """
    if tau <= 0:
        return x
    if axis != -1:
        x = x.transpose(axis, -1)
    T = x.shape[-1]
    device = x.device
    dtype = x.dtype
    alpha_val = 1.0 - float(torch.exp(torch.tensor(-1.0 / tau, dtype=torch.float64)))
    window = max(1, min(T, int(round(5.0 * tau))))
    ks = torch.arange(window, device=device, dtype=dtype)
    h = alpha_val * (1.0 - alpha_val) ** ks
    kernel = h.flip(0).reshape(1, 1, window)
    orig_shape = x.shape
    x_flat = x.reshape(-1, 1, T)
    x_padded = F.pad(x_flat, (window - 1, 0))
    y = F.conv1d(x_padded, kernel)
    y = y.reshape(orig_shape)
    if axis != -1:
        y = y.transpose(axis, -1)
    return y


def _rbf_kernel(x: torch.Tensor, sigma: float) -> torch.Tensor:
    sq_norms = (x * x).sum(dim=-1, keepdim=True)
    sq_dists = sq_norms + sq_norms.transpose(-2, -1) - 2.0 * (x @ x.transpose(-2, -1))
    sq_dists = sq_dists.clamp(min=0.0)
    return torch.exp(-sq_dists / (2.0 * sigma * sigma))


def kernel_matrix(features: torch.Tensor, kind: str, **params) -> torch.Tensor:
    """Compute the ``[B, B]`` pairwise similarity matrix.

    Args:
        features: ``[B, D]`` tensor.
        kind: one of ``count``, ``binned``, ``mci``, ``vanrossum``, ``schoenberg``.
        params: ``sigma`` for ``schoenberg`` (default 1.0). Others take no params.

    Returns:
        ``[B, B]`` tensor, symmetric.
    """
    if features.dim() != 2:
        raise ValueError(
            f"features must be [B, D], got shape {tuple(features.shape)}"
        )
    if kind == "count":
        counts = features.sum(dim=-1, keepdim=True)
        return counts @ counts.transpose(-2, -1)
    if kind in _LINEAR_KINDS:
        return features @ features.transpose(-2, -1)
    if kind == "schoenberg":
        sigma = float(params.get("sigma", 1.0))
        if sigma <= 0:
            raise ValueError(f"schoenberg sigma must be > 0, got {sigma}")
        return _rbf_kernel(features, sigma)
    raise ValueError(
        f"unknown kernel kind: {kind!r}. Supported: {_ALL_KINDS}."
    )


def pairwise_diversity_loss(K: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal of ``K``.

    ``L = (sum_{i != j} K[i, j]) / (B * (B - 1))``.

    Small when features are diverse (off-diagonal similarities near zero);
    large when features are similar. Add to the training loss with
    ``+lambda * L`` to penalize redundancy.

    Returns a scalar of the same dtype/device as ``K``. Returns 0 for
    ``B < 2``.
    """
    if K.dim() != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be [B, B], got shape {tuple(K.shape)}")
    B = K.shape[0]
    if B < 2:
        return K.new_zeros(())
    off_diag = K.sum() - K.diagonal().sum()
    return off_diag / (B * (B - 1))
