"""Seed everything we can and record what we can't.

CUDA convolution kernels remain nondeterministic across runs even with seeds set;
we document that as a known limit rather than pretending to fix it.
"""
from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from typing import Any


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Cudnn benchmark on → faster but nondeterministic; leaving on because
    # we document nondeterminism as a known limit anyway.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def _nvidia_smi() -> dict[str, str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip().splitlines()
        if not out:
            return {"gpu": "unknown"}
        name, drv, mem, cap = [x.strip() for x in out[0].split(",")]
        return {"gpu_name": name, "driver": drv, "vram_total": mem, "compute_cap": cap}
    except Exception as e:
        return {"gpu_query_error": str(e)}


def fingerprint() -> dict[str, Any]:
    import torch
    fp: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import snntorch  # noqa
        fp["snntorch"] = snntorch.__version__
    except Exception:
        pass
    try:
        import torchvision  # noqa
        fp["torchvision"] = torchvision.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        fp["device_name"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        fp["device_capability"] = f"sm_{cap[0]}{cap[1]}"
        fp["torch_arch_list"] = torch.cuda.get_arch_list()
    fp.update(_nvidia_smi())
    return fp
