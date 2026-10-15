"""Game state x = (W, U, S, E, Z) and the per-window bookkeeping (proposal §4.2-4.4).

W lives in the network parameters; U, S, E are the recordings of the current
window (`SNNRecord`); Z holds slow homeostatic variables. `GameState` also
keeps what the game needs across windows: the previous potential (for the
reward r = Phi(x_{n+1}) - Phi(x_n), eq. 9) and each neuron's previous action
dW_i (for Eligibility-Shapley, eq. 21-22).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .model import SNNRecord


@dataclass
class GameState:
    n_layers: int
    rate_ema: float = 0.99

    # Z: running per-neuron firing rate (spikes / step), one [N_l] per layer.
    rates: list[torch.Tensor | None] = field(default_factory=list)
    # Previous window's applied action per layer, W_after - W_before, [N_l, fan_in_l].
    prev_delta_w: list[torch.Tensor | None] = field(default_factory=list)
    # Phi at the previous window, for r = Phi_n - Phi_{n-1}.
    prev_phi: float | None = None
    window: int = 0

    def __post_init__(self) -> None:
        if not self.rates:
            self.rates = [None] * self.n_layers
        if not self.prev_delta_w:
            self.prev_delta_w = [None] * self.n_layers

    @torch.no_grad()
    def update_rates(self, rec: SNNRecord) -> None:
        for l, s in enumerate(rec.spk):
            r = s.mean(dim=(0, 1))
            self.rates[l] = r.clone() if self.rates[l] is None else (
                self.rate_ema * self.rates[l] + (1.0 - self.rate_ema) * r)

    def reward(self, phi: float) -> float | None:
        """r^(n) = Phi(x^(n)) - Phi(x^(n-1)); None on the first window."""
        r = None if self.prev_phi is None else phi - self.prev_phi
        self.prev_phi = phi
        return r

    def state_dict(self) -> dict:
        return {"rates": self.rates, "prev_delta_w": self.prev_delta_w,
                "prev_phi": self.prev_phi, "window": self.window}
