"""Run configuration for Neuron Games experiments.

One `NGRunConfig` describes one row of the ablation grid
(docs/implementation-plan.md §5). Only Row 2 (`bp_anchor`) exists so far.
Serialized next to every run, like the baseline's `Config`.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# condition name -> ablation row in implementation-plan.md §5
CONDITIONS: dict[str, int] = {
    "bp_anchor": 2,
}


@dataclass
class NGRunConfig:
    condition: str = "bp_anchor"

    # --- Network: 784-128-10 fully-connected LIF (plan §1, §1.5) ---
    n_in: int = 784
    n_hidden: int = 128
    n_out: int = 10
    alpha: float = math.exp(-1.0 / 10.0)  # membrane decay, tau_m = 10 steps
    threshold: float = 1.0
    surrogate_slope: float = 25.0         # fast-sigmoid k
    bias: bool = False                    # every trainable parameter is a synapse w_ij

    # --- Input / task ---
    dataset: str = "MNIST"
    num_steps: int = 25                   # T
    encoding: str = "rate"                # rate (Bernoulli) | direct (same pixels each step)
    task_loss: str = "ce_rate"            # ce_rate (mean_t CE on spk[t]) | ce_count (CE on spike counts)

    # --- Optimization ---
    lr: float = 1e-3
    adam_betas: tuple[float, float] = (0.9, 0.999)
    epochs: int = 1
    batch_size: int = 128

    # --- Evaluation / reproducibility (same meaning as neuron_baseline.Config) ---
    val_size: int = 5000
    eval_batch_size: int = 256
    seed: int = 20260923
    data_dir: str = "./data/raw"
    run_dir: str = ""
    num_workers: int = 0
    device: str = "cuda"                  # never silently fall back
    energy_sample_hz: float = 20.0
    warmup_batches: int = 2

    fingerprint: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown condition {self.condition!r}; known: {sorted(CONDITIONS)}")
        if self.encoding not in ("rate", "direct"):
            raise ValueError(f"unknown encoding {self.encoding!r}")
        if self.task_loss not in ("ce_rate", "ce_count"):
            raise ValueError(f"unknown task_loss {self.task_loss!r}")

    @property
    def row(self) -> int:
        return CONDITIONS[self.condition]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NGRunConfig":
        data = dict(data)
        if isinstance(data.get("adam_betas"), list):
            data["adam_betas"] = tuple(data["adam_betas"])
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    @classmethod
    def load(cls, path: str | Path) -> "NGRunConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
