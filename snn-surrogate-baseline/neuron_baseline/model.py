"""snnTorch Tutorial 6 model verbatim + a static pixel-value forward pass.

Architecture (Tutorial 6 §3.1):

    Conv2d(1, 12, 5) → MaxPool2d(2) → Leaky
    Conv2d(12, 64, 5) → MaxPool2d(2) → Leaky
    Flatten() → Linear(64*4*4=1024, 10) → Leaky(output=True)

Every Leaky:
  beta=0.5, spike_grad=fast_sigmoid(slope=25),
  init_hidden=True, threshold=1.0, reset_mechanism='subtract'.
Only the last also sets output=True so it returns `(spk, mem)`.

Forward pass presents the SAME image at each of the T simulation steps
(no Bernoulli / latency encoding — the tutorial's "direct" input path).
Membrane state carries across the T loop within a batch and is reset
between batches by `utils.reset(net)`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, utils


def build_net(
    beta: float = 0.5,
    surrogate_slope: float = 25.0,
    threshold: float = 1.0,
    reset_mechanism: str = "subtract",
) -> nn.Sequential:
    spike_grad = surrogate.fast_sigmoid(slope=surrogate_slope)
    net = nn.Sequential(
        nn.Conv2d(1, 12, 5),
        nn.MaxPool2d(2),
        snn.Leaky(
            beta=beta, spike_grad=spike_grad, init_hidden=True,
            threshold=threshold, reset_mechanism=reset_mechanism,
        ),
        nn.Conv2d(12, 64, 5),
        nn.MaxPool2d(2),
        snn.Leaky(
            beta=beta, spike_grad=spike_grad, init_hidden=True,
            threshold=threshold, reset_mechanism=reset_mechanism,
        ),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 10),
        snn.Leaky(
            beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,
            threshold=threshold, reset_mechanism=reset_mechanism,
        ),
    )
    return net


def forward_pass(net: nn.Sequential, num_steps: int, data: torch.Tensor,
                 tap_module_idx: int | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the network for `num_steps` steps on the same static input.

    Returns:
        spk_rec: [T, B, num_classes]
        mem_rec: [T, B, num_classes]
        hidden_rec: [T, B, ...] from the module at `tap_module_idx` in `net`,
            or None if `tap_module_idx` is None. Used by the diversity
            regularizer (Phase B2) to compute a neuron-wise kernel matrix.

    Resets neuron hidden state ONCE at the start (between batches), then
    carries state across the T loop. The forward hook is registered only
    when `tap_module_idx` is not None, so lambda=0 training pays no cost
    for the diversity path.
    """
    spk_rec: list[torch.Tensor] = []
    mem_rec: list[torch.Tensor] = []
    hidden_rec: list[torch.Tensor] | None = None

    hook_handle = None
    captured: list[torch.Tensor] = []
    if tap_module_idx is not None:
        hidden_rec = []

        def _hook(_mod, _inp, out):
            captured.append(out)

        hook_handle = net[tap_module_idx].register_forward_hook(_hook)

    try:
        utils.reset(net)
        for _ in range(num_steps):
            if hidden_rec is not None:
                captured.clear()
            spk_out, mem_out = net(data)
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)
            if hidden_rec is not None:
                hidden_rec.append(captured[0])
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    hidden_tensor = torch.stack(hidden_rec) if hidden_rec is not None else None
    return torch.stack(spk_rec), torch.stack(mem_rec), hidden_tensor
