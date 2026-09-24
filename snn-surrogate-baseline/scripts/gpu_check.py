"""Verify GPU is actually usable — not just importable.

Runs a small conv+linear forward+backward on CUDA, prints device/build info,
fails loudly (nonzero exit) if anything is wrong. Never silently falls back to CPU.
"""
from __future__ import annotations

import platform
import subprocess
import sys


def _nvidia_smi_line() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip().splitlines()
        return out[0] if out else "(no output)"
    except Exception as e:
        return f"(nvidia-smi failed: {e})"


def main() -> int:
    import torch
    import torch.nn as nn

    print(f"platform: {platform.platform()}")
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda (built with): {torch.version.cuda}")
    print(f"torch.backends.cudnn.version: {torch.backends.cudnn.version()}")
    print(f"nvidia-smi: {_nvidia_smi_line()}")

    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False", file=sys.stderr)
        return 1

    dev = torch.device("cuda:0")
    print(f"n_devices: {torch.cuda.device_count()}")
    print(f"device_name: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"capability: sm_{cap[0]}{cap[1]}")
    print(f"arch_list: {torch.cuda.get_arch_list()}")

    torch.manual_seed(0)
    x = torch.randn(4, 1, 28, 28, device=dev)
    target = torch.randint(0, 10, (4,), device=dev)

    net = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 28 * 28, 10),
    ).to(dev)

    torch.cuda.synchronize()
    logits = net(x)
    loss = nn.functional.cross_entropy(logits, target)
    loss.backward()
    torch.cuda.synchronize()

    ok_loss = torch.isfinite(loss).item()
    grads_any_nonzero = any(
        (p.grad is not None) and torch.any(p.grad != 0).item()
        for p in net.parameters()
    )
    grads_all_finite = all(
        (p.grad is None) or torch.all(torch.isfinite(p.grad)).item()
        for p in net.parameters()
    )
    print(f"loss: {loss.item():.6f} (finite={ok_loss})")
    print(f"grads_any_nonzero: {grads_any_nonzero}")
    print(f"grads_all_finite: {grads_all_finite}")

    if not (ok_loss and grads_any_nonzero and grads_all_finite):
        print("FAIL: forward/backward did not produce a healthy loss/grad", file=sys.stderr)
        return 2

    print("OK: GPU forward+backward completed cleanly on CUDA.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
