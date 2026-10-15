"""Explainability metrics (proposal §12.4).

OWNER: William.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def masked_accuracy(net, loader, cfg, device, masks: list[torch.Tensor | None]) -> float:
    """Accuracy of `net` on `loader` with neurons removed by `masks`.

    Same encoding and accuracy rule as eval.evaluate (spike-count argmax),
    but pass masks to net(x, masks=masks). Every example counted.
    """
    raise NotImplementedError("William: masked_accuracy (explain/metrics.py)")


def faithfulness_curve(net, loader, cfg, device, credit: torch.Tensor, layer: int,
                       ks: list[int], n_random: int = 5, seed: int = 0) -> dict:
    """Removing the highest-credit neurons should hurt most (Neuron Shapley eval).

    For each k in ks: accuracy after removing the top-k, the bottom-k, and
    k random neurons (mean +- std over n_random draws) of `layer`.
    Return {"k": ks, "top": [...], "bottom": [...], "random_mean": [...],
    "random_std": [...], "full": acc_without_removal}.
    """
    raise NotImplementedError("William: faithfulness_curve (explain/metrics.py)")


def stability(credit_a: torch.Tensor, credit_b: torch.Tensor) -> dict:
    """Agreement of two credit vectors for the same layer (adjacent epochs, or
    two augmentations of the data). Return
    {"spearman": rank correlation, "top10_jaccard": Jaccard overlap of the
    top-10% neurons of each}."""
    raise NotImplementedError("William: stability (explain/metrics.py)")


def sparsity(credit: torch.Tensor) -> dict:
    """How concentrated credit is. Return {"gini": Gini coefficient of |phi|,
    "frac_50": smallest fraction of neurons holding 50% of total |phi|,
    "frac_90": same for 90%}."""
    raise NotImplementedError("William: sparsity (explain/metrics.py)")
