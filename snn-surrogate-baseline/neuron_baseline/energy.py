"""GPU energy measurement via NVML.

Two paths:
- Cumulative energy (preferred): `nvmlDeviceGetTotalEnergyConsumption` returns
  a monotonic mJ counter — subtract endpoints, divide by 1000.
- Power sampling (fallback): `nvmlDeviceGetPowerUsage` returns mW; a
  background sampler collects timestamped power, then trapezoidal integration
  gives energy:
      E ≈ Σ [(P_k + P_{k+1}) / 2] · (t_{k+1} − t_k)

If NVML is unavailable, `EnergyMeter` reports `energy_j=None` — we never
estimate joules from runtime or spike counts.

Caveats to keep in mind when interpreting results (also documented in the README):
- NVML power/energy is a **GPU-only** figure. It does not include CPU,
  memory, or system PSU overhead.
- Sampling introduces its own timing noise; keep the measurement window long
  enough (many batches) to dominate sensor granularity.
- Warm up before measuring (kernel compilation, cudnn autotune settle).
- `torch.cuda.synchronize()` is called at every measurement boundary so we
  don't attribute deferred work to the wrong window.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import pynvml as nvml
    _NVML_IMPORT_ERR: str | None = None
except Exception as e:
    nvml = None  # type: ignore
    _NVML_IMPORT_ERR = f"{type(e).__name__}: {e}"


@dataclass
class EnergyMeasurement:
    window: str                     # 'train' | 'inference' | ...
    duration_s: float
    energy_j: float | None
    method: str                     # 'cumulative_energy' | 'power_sampling' | 'unavailable'
    n_samples: int = 0
    mean_power_w: float | None = None
    device_index: int = 0
    device_name: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "duration_s": self.duration_s,
            "energy_j": self.energy_j,
            "method": self.method,
            "n_samples": self.n_samples,
            "mean_power_w": self.mean_power_w,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "notes": self.notes,
        }


@dataclass
class _SamplerState:
    times: list[float] = field(default_factory=list)
    powers_w: list[float] = field(default_factory=list)


class EnergyMeter:
    """Context-manager style meter for a single measurement window.

    Usage:
        meter = EnergyMeter(device_index=0)
        with meter.measure("train") as m:
            train_one_epoch(...)
        result = meter.result  # EnergyMeasurement
    """

    def __init__(self, device_index: int = 0, sample_hz: float = 20.0):
        self.device_index = device_index
        self.sample_hz = float(sample_hz)
        self.available = False
        self.has_cumulative = False
        self.device_name = ""
        self._handle = None
        self._init_notes: list[str] = []
        self._init_nvml()
        self.result: EnergyMeasurement | None = None

    def _init_nvml(self) -> None:
        if nvml is None:
            self._init_notes.append(f"pynvml import failed: {_NVML_IMPORT_ERR}")
            return
        try:
            nvml.nvmlInit()
            self._handle = nvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.device_name = nvml.nvmlDeviceGetName(self._handle)
            if isinstance(self.device_name, bytes):
                self.device_name = self.device_name.decode()
            # Feature detection: cumulative energy is optional per SKU/driver
            try:
                nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                self.has_cumulative = True
            except nvml.NVMLError as e:
                self.has_cumulative = False
                self._init_notes.append(f"cumulative energy unavailable: {e}")
            self.available = True
        except Exception as e:
            self._init_notes.append(f"NVML init failed: {e}")
            self.available = False

    def _sync_cuda(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def _read_power_w(self) -> float:
        # nvmlDeviceGetPowerUsage returns milliwatts
        return float(nvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0

    def _read_energy_mj(self) -> int:
        return int(nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))

    class _Ctx:
        def __init__(self, meter: "EnergyMeter", window: str):
            self.meter = meter
            self.window = window
            self.t0 = 0.0
            self.mj0 = 0
            self._sampler_thread: threading.Thread | None = None
            self._sampler_stop: threading.Event | None = None
            self._sampler_state = _SamplerState()

        def _sampler(self) -> None:
            interval = 1.0 / max(self.meter.sample_hz, 1e-3)
            next_t = time.perf_counter()
            while not (self._sampler_stop is not None and self._sampler_stop.is_set()):
                try:
                    self._sampler_state.times.append(time.perf_counter())
                    self._sampler_state.powers_w.append(self.meter._read_power_w())
                except Exception:
                    break
                next_t += interval
                sleep_for = next_t - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_t = time.perf_counter()

        def __enter__(self) -> "EnergyMeter._Ctx":
            self.meter._sync_cuda()
            self.t0 = time.perf_counter()
            if not self.meter.available:
                return self
            if self.meter.has_cumulative:
                self.mj0 = self.meter._read_energy_mj()
            else:
                # spin up sampler thread
                self._sampler_stop = threading.Event()
                self._sampler_thread = threading.Thread(target=self._sampler, daemon=True)
                # take one immediate sample so short windows still have data
                try:
                    self._sampler_state.times.append(time.perf_counter())
                    self._sampler_state.powers_w.append(self.meter._read_power_w())
                except Exception:
                    pass
                self._sampler_thread.start()
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.meter._sync_cuda()
            t1 = time.perf_counter()
            duration = t1 - self.t0

            if not self.meter.available:
                self.meter.result = EnergyMeasurement(
                    window=self.window, duration_s=duration, energy_j=None,
                    method="unavailable", device_index=self.meter.device_index,
                    device_name="", notes="; ".join(self.meter._init_notes) or "NVML unavailable",
                )
                return

            if self.meter.has_cumulative:
                mj1 = self.meter._read_energy_mj()
                d_mj = max(mj1 - self.mj0, 0)
                self.meter.result = EnergyMeasurement(
                    window=self.window, duration_s=duration,
                    energy_j=d_mj / 1000.0, method="cumulative_energy",
                    n_samples=1, mean_power_w=(d_mj / 1000.0) / max(duration, 1e-9),
                    device_index=self.meter.device_index, device_name=self.meter.device_name,
                    notes="nvmlDeviceGetTotalEnergyConsumption (mJ counter)",
                )
                return

            # Fallback: trapezoidal integration over sampled power
            assert self._sampler_stop is not None and self._sampler_thread is not None
            self._sampler_stop.set()
            self._sampler_thread.join(timeout=1.0)
            # take a final sample
            try:
                self._sampler_state.times.append(time.perf_counter())
                self._sampler_state.powers_w.append(self.meter._read_power_w())
            except Exception:
                pass
            ts = self._sampler_state.times
            ps = self._sampler_state.powers_w
            energy_j: float | None
            mean_p: float | None
            if len(ts) < 2:
                energy_j = None
                mean_p = ps[0] if ps else None
                note = "insufficient samples for integration"
            else:
                energy_j = 0.0
                for k in range(len(ts) - 1):
                    dt = ts[k + 1] - ts[k]
                    energy_j += 0.5 * (ps[k] + ps[k + 1]) * dt
                mean_p = energy_j / max(ts[-1] - ts[0], 1e-9)
                note = f"trapezoidal integration of power samples @ ~{self.meter.sample_hz:g} Hz"
            self.meter.result = EnergyMeasurement(
                window=self.window, duration_s=duration,
                energy_j=energy_j, method="power_sampling",
                n_samples=len(ts), mean_power_w=mean_p,
                device_index=self.meter.device_index, device_name=self.meter.device_name,
                notes=note,
            )

    def measure(self, window: str) -> "EnergyMeter._Ctx":
        return EnergyMeter._Ctx(self, window)

    def shutdown(self) -> None:
        if nvml is not None and self.available:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass
