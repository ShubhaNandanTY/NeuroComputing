"""Smoke test — the goal is to catch wiring bugs BEFORE launching training.

Verifies (in order):
1. Model builds and moves to the configured device.
2. Forward pass produces output of shape [T, B, num_classes].
3. Loss is finite; grads are finite and at least some are nonzero.
4. Weights actually change after one optimizer step (not just grad exists).
5. Membrane state carries within a batch across T steps (progresses).
6. Between independent batches, `utils.reset(net)` restores hidden state
   (identical input twice with a reset between produces identical outputs).
7. Checkpoint round-trip: save → new net → load → same eval output.
"""
from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import snntorch.functional as SF
from snntorch import utils

from neuron_baseline.config import Config
from neuron_baseline.model import build_net, forward_pass
from neuron_baseline.reproducibility import seed_everything


CLASSES = 10


def _make_batch(bs: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.rand(bs, 1, 28, 28, device=device)
    y = torch.randint(0, CLASSES, (bs,), device=device)
    return x, y


def _fail(msg: str) -> None:
    print(f"SMOKE FAIL: {msg}", file=sys.stderr)


def run_smoke() -> int:
    seed_everything(20260923)
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        _fail("device=cuda but torch.cuda.is_available()=False")
        return 2

    T = 8            # small T for speed
    bs = 4
    print(f"[smoke] device={device} T={T} bs={bs}")

    net = build_net(cfg.beta, cfg.surrogate_slope, cfg.threshold, cfg.reset_mechanism).to(device)
    data, targets = _make_batch(bs, device)

    # 2. Forward shape
    spk_rec, mem_rec = forward_pass(net, T, data)
    if spk_rec.shape != (T, bs, CLASSES):
        _fail(f"spk_rec shape {tuple(spk_rec.shape)} != {(T, bs, CLASSES)}")
        return 10
    if mem_rec.shape != (T, bs, CLASSES):
        _fail(f"mem_rec shape {tuple(mem_rec.shape)} != {(T, bs, CLASSES)}")
        return 11
    print(f"[smoke] shapes ok: spk={tuple(spk_rec.shape)} mem={tuple(mem_rec.shape)}")

    # 3. Loss finite, grads finite, some nonzero
    loss_fn = SF.ce_rate_loss()
    loss = loss_fn(spk_rec, targets)
    if not torch.isfinite(loss):
        _fail(f"loss not finite: {loss}")
        return 12
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, betas=cfg.adam_betas)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    any_nonzero = False
    for name, p in net.named_parameters():
        if p.grad is None:
            continue
        if not torch.all(torch.isfinite(p.grad)):
            _fail(f"grad for {name} not all finite")
            return 13
        if torch.any(p.grad != 0):
            any_nonzero = True
    if not any_nonzero:
        _fail("no parameter has any nonzero grad")
        return 14
    print(f"[smoke] loss={loss.item():.4f} finite; grads finite; some nonzero")

    # 4. Optimizer step actually changes weights
    before = {n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad}
    optimizer.step()
    changed_any = False
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if not torch.equal(before[n], p.detach()):
            changed_any = True
            break
    if not changed_any:
        _fail("optimizer.step() did not change any weights")
        return 15
    print("[smoke] optimizer.step() moves weights: ok")

    # 5. Membrane state progresses across T within a batch
    net.eval()
    with torch.no_grad():
        _, mems = forward_pass(net, T, data)
    if torch.allclose(mems[0], mems[-1]):
        _fail("membrane potential is constant across T steps within a batch")
        return 16
    print(f"[smoke] mem trajectory nontrivial: |mem_T - mem_0|_mean={float((mems[-1]-mems[0]).abs().mean()):.4e}")

    # 6. utils.reset restores state — two identical batches with reset between should match bit-for-bit
    with torch.no_grad():
        spk_a, _ = forward_pass(net, T, data)
        spk_b, _ = forward_pass(net, T, data)  # forward_pass calls utils.reset internally
    if not torch.equal(spk_a, spk_b):
        _fail("reset between batches did not restore identical output for identical input")
        return 17
    # And confirm skipping the reset does NOT match (state carries -> different)
    with torch.no_grad():
        utils.reset(net)
        spk_c1, _ = forward_pass(net, T, data)
        # deliberately DON'T reset; run T more steps and compare against a fresh reset
        spk_c2, _ = forward_pass(net, T, data)
    # Because forward_pass calls reset, spk_c2 should equal spk_c1. Skip the anti-check.
    print("[smoke] utils.reset restores hidden state between batches: ok")

    # 7. Checkpoint save + load → identical eval logits
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "smoke.pt"
        torch.save({"model_state": net.state_dict()}, str(ckpt))
        net2 = build_net(cfg.beta, cfg.surrogate_slope, cfg.threshold, cfg.reset_mechanism).to(device)
        payload = torch.load(str(ckpt), map_location=device, weights_only=False)
        net2.load_state_dict(payload["model_state"])
        with torch.no_grad():
            spk_orig, _ = forward_pass(net, T, data)
            spk_load, _ = forward_pass(net2, T, data)
        if not torch.equal(spk_orig, spk_load):
            _fail("reload of saved checkpoint produced different output on the same batch")
            return 18
    print("[smoke] checkpoint save/load round-trip: bit-identical eval outputs")

    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke())
