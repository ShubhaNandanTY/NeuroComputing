"""Broadcast learning signal (proposal §7.2; e-prop with random feedback, Bellec et al. 2020).

The output layer's task signal is its own error dJ/ds_out[t] (potential/task.py
`output_error`). Every hidden layer l receives that error through a fixed
random matrix B_l in R^{N_out x N_l}, drawn once and never trained:

    L_l[t, b, i] = sum_k delta[t, b, k] * B_l[k, i]

No weight transport (B is not W_out^T) and no backward pass. B_l uses the
same scale as nn.Linear's default init of an N_l -> N_out layer, times
cfg.feedback_scale, and is drawn from its own generator so changing B never
changes W's init.
"""
from __future__ import annotations

import math

import torch


class RandomFeedback:
    def __init__(self, layer_sizes: list[int], seed: int, scale: float = 1.0,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        """layer_sizes: [N_0, ..., N_out] neuron counts per layer (not the input)."""
        g = torch.Generator().manual_seed(seed)
        n_out = layer_sizes[-1]
        self.B: list[torch.Tensor] = []
        for n_l in layer_sizes[:-1]:
            bound = scale / math.sqrt(n_l)
            b = (torch.rand(n_out, n_l, generator=g, dtype=dtype) * 2.0 - 1.0) * bound
            self.B.append(b.to(device))

    def broadcast(self, out_err: torch.Tensor) -> list[torch.Tensor]:
        """out_err: [T, B, N_out] -> one [T, B, N_l] signal per hidden layer."""
        return [out_err @ b for b in self.B]

    def state_dict(self) -> dict:
        return {"B": [b.detach().cpu() for b in self.B]}

    def load_state_dict(self, sd: dict) -> None:
        self.B = [b.to(self.B[i].device) for i, b in enumerate(sd["B"])]
