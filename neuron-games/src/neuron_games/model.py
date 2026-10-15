"""784-128-10 fully-connected LIF network with a vendored LIF cell.

Vendored rather than subclassing `snn.Leaky` (plan §8, decision 8): the
three-factor rule (Workstream A) needs the membrane potential, the surrogate
derivative `h`, and the previous spike as explicit tensors at every step,
and snnTorch keeps those as hidden module state. The same cell serves Row 2
(BPTT through autograd) and, later, Row 4 (three-factor, no autograd through
the recurrence), so the two rows train an identical model.

Dynamics of layer neuron i at step t (x = presynaptic input at step t):

    u_i[t] = alpha * u_i[t-1] + sum_j W_ij x_j[t] - v_th * s_i[t-1]
    s_i[t] = H(u_i[t] - v_th)

This matches snnTorch `Leaky(reset_mechanism="subtract")`. Two choices make
the factorized eligibility trace e_ij[t] = h_i[t] * eps_j[t] the *exact*
local derivative ds_i[t]/dW_ij (surrogate h), which the three-factor rule
relies on:

- The reset term uses a *detached* `s_i[t-1]`, exactly like snnTorch's
  `mem_reset`, so BPTT gradients do not flow through the reset.
- Input enters on the same step (no one-step synaptic delay). Plan §1 writes
  `s_j[t-1]`; this is the same model shifted by one index, and it makes the
  eligibility trace `eps_j[t] = alpha * eps_j[t-1] + x_j[t]`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def fast_sigmoid_grad(v: torch.Tensor, slope: float) -> torch.Tensor:
    """Surrogate derivative h = 1 / (1 + slope * |v|)^2, v = u - v_th.

    Same function as snnTorch `surrogate.fast_sigmoid`. Exposed so the
    three-factor rule computes `h` with the identical formula autograd uses.
    """
    return 1.0 / (1.0 + slope * v.abs()) ** 2


class _SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(v)
        ctx.slope = slope
        return (v > 0).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (v,) = ctx.saved_tensors
        return grad_out * fast_sigmoid_grad(v, ctx.slope), None


def spike(v: torch.Tensor, slope: float) -> torch.Tensor:
    """Heaviside forward, fast-sigmoid surrogate backward."""
    return _SpikeFn.apply(v, slope)


class LIFLayer(nn.Module):
    """Linear synapses followed by subtract-reset LIF neurons."""

    def __init__(self, fan_in: int, fan_out: int, alpha: float, threshold: float,
                 surrogate_slope: float, bias: bool = False):
        super().__init__()
        self.fc = nn.Linear(fan_in, fan_out, bias=bias)
        self.alpha = alpha
        self.threshold = threshold
        self.surrogate_slope = surrogate_slope

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(batch, self.fc.out_features, device=device, dtype=dtype)
        return z, z.clone()

    def step(self, x: torch.Tensor, u: torch.Tensor, s: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """One time step. Returns (s[t], u[t]) given input x[t] and (u[t-1], s[t-1])."""
        u = self.alpha * u + self.fc(x) - self.threshold * s.detach()
        s = spike(u - self.threshold, self.surrogate_slope)
        return s, u


@dataclass
class SNNRecord:
    """Per-layer recordings over the window; index 0 = hidden, -1 = output.

    spk[l], mem[l]: [T, B, N_l]. These are the S(.) and U(.) parts of the
    game state x = (W, U, S, E, Z) (proposal §4.2).

    With `record_traces=True` the forward pass also fills the eligibility
    factors E(.) (proposal §7.1, eq. 19), in factorized form:

    - inp[l]: [T, B, fan_in_l]   presynaptic input to layer l at step t
    - eps[l]: [T, B, fan_in_l]   eps_j[t] = alpha * eps_j[t-1] + inp_j[t]
    - h[l]:   [T, B, N_l]        h_i[t] = fast_sigmoid_grad(u_i[t] - v_th)

    The synapse-local eligibility trace is e_ij[t] = h_i[t] * eps_j[t]. It is
    never materialized at [T, B, N_l, fan_in_l]; see learning_rule.py.
    """
    spk: list[torch.Tensor]
    mem: list[torch.Tensor]
    inp: list[torch.Tensor] | None = None
    eps: list[torch.Tensor] | None = None
    h: list[torch.Tensor] | None = None

    @property
    def out_spk(self) -> torch.Tensor:
        return self.spk[-1]

    @property
    def has_traces(self) -> bool:
        return self.eps is not None and self.h is not None


class FCSNN(nn.Module):
    """n_in -> n_hidden -> n_out, LIF at both the hidden and output layer."""

    def __init__(self, n_in: int = 784, n_hidden: int = 128, n_out: int = 10,
                 alpha: float = 0.905, threshold: float = 1.0,
                 surrogate_slope: float = 25.0, bias: bool = False):
        super().__init__()
        kw = dict(alpha=alpha, threshold=threshold,
                  surrogate_slope=surrogate_slope, bias=bias)
        self.layers = nn.ModuleList([
            LIFLayer(n_in, n_hidden, **kw),
            LIFLayer(n_hidden, n_out, **kw),
        ])

    @property
    def weights(self) -> list[torch.Tensor]:
        """W_l for every layer, [N_l, fan_in_l]; the players' action variables."""
        return [layer.fc.weight for layer in self.layers]

    def forward(self, x_seq: torch.Tensor, record_traces: bool = False,
                masks: list[torch.Tensor | None] | None = None) -> SNNRecord:
        """x_seq: [T, B, n_in]. State starts at zero on every call.

        record_traces: also record inp / eps / h (see SNNRecord). Used by the
            three-factor rule; costs O(T * B * fan_in) memory per layer.
        masks: optional per-layer [N_l] multipliers applied to that layer's
            spikes (0 silences a neuron). Used by post-hoc attribution to
            remove neurons; None everywhere means the plain network.
        """
        T, B = x_seq.shape[:2]
        states = [layer.init_state(B, x_seq.device, x_seq.dtype) for layer in self.layers]
        spk: list[list[torch.Tensor]] = [[] for _ in self.layers]
        mem: list[list[torch.Tensor]] = [[] for _ in self.layers]
        inp_rec: list[list[torch.Tensor]] = [[] for _ in self.layers]
        eps_rec: list[list[torch.Tensor]] = [[] for _ in self.layers]
        h_rec: list[list[torch.Tensor]] = [[] for _ in self.layers]
        eps_state: list[torch.Tensor | None] = [None for _ in self.layers]

        for t in range(T):
            inp = x_seq[t]
            for l, layer in enumerate(self.layers):
                u, s = states[l]
                s, u = layer.step(inp, u, s)
                if masks is not None and masks[l] is not None:
                    s = s * masks[l]
                if record_traces:
                    e = inp if eps_state[l] is None else layer.alpha * eps_state[l] + inp
                    eps_state[l] = e
                    inp_rec[l].append(inp)
                    eps_rec[l].append(e)
                    h_rec[l].append(fast_sigmoid_grad(u - layer.threshold,
                                                      layer.surrogate_slope))
                states[l] = (u, s)
                spk[l].append(s)
                mem[l].append(u)
                inp = s

        rec = SNNRecord(spk=[torch.stack(r) for r in spk],
                        mem=[torch.stack(r) for r in mem])
        if record_traces:
            rec.inp = [torch.stack(r) for r in inp_rec]
            rec.eps = [torch.stack(r) for r in eps_rec]
            rec.h = [torch.stack(r) for r in h_rec]
        return rec


def build_net(cfg) -> FCSNN:
    return FCSNN(n_in=cfg.n_in, n_hidden=cfg.n_hidden, n_out=cfg.n_out,
                 alpha=cfg.alpha, threshold=cfg.threshold,
                 surrogate_slope=cfg.surrogate_slope, bias=cfg.bias)
