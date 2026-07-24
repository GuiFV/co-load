"""Live VRAM measurement.

``VramProbe`` is the abstraction the orchestrator and watchdog depend on
(DIP). Two implementations ship: NVML (preferred) and an ``nvidia-smi``
parser as fallback. All figures are bytes.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Protocol

GIB = 2**30


class ProbeError(RuntimeError):
    """The GPU could not be measured."""


@dataclass(frozen=True)
class VramSnapshot:
    """Point-in-time VRAM figures for one GPU, in bytes."""

    total: int
    used: int

    @property
    def free(self) -> int:
        return self.total - self.used

    def budget(self, buffer_pct: float) -> int:
        """Loadable bytes right now: free minus the user-configured headroom.

        ``buffer_pct`` is a share of *total* VRAM kept free (default 10%,
        user-settable via config).
        """
        return max(0, self.free - int(self.total * buffer_pct))


class VramProbe(Protocol):
    def read(self) -> VramSnapshot: ...


class NvmlProbe:
    """Reads VRAM via NVML (pynvml). The nvml module is injectable for tests."""

    def __init__(self, gpu_index: int = 0, nvml: Any = None):
        if nvml is None:  # pragma: no cover - exercised only with a real GPU
            import pynvml

            nvml = pynvml
        self._nvml = nvml
        self._gpu_index = gpu_index
        self._initialized = False

    def read(self) -> VramSnapshot:
        try:
            if not self._initialized:
                self._nvml.nvmlInit()
                self._initialized = True
            handle = self._nvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            info = self._nvml.nvmlDeviceGetMemoryInfo(handle)
            return VramSnapshot(total=int(info.total), used=int(info.used))
        except Exception as exc:
            raise ProbeError(f"NVML probe failed: {exc}") from exc


def _run_nvidia_smi(cmd: list[str]) -> str:  # pragma: no cover - needs binary
    return subprocess.run(
        cmd, capture_output=True, text=True, check=True
    ).stdout


class NvidiaSmiProbe:
    """Fallback probe parsing ``nvidia-smi`` CSV output (MiB)."""

    def __init__(
        self,
        gpu_index: int = 0,
        runner: Callable[[list[str]], str] = _run_nvidia_smi,
    ):
        self._gpu_index = gpu_index
        self._runner = runner

    def read(self) -> VramSnapshot:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(self._gpu_index),
        ]
        try:
            out = self._runner(cmd)
            total_mib, used_mib = (int(part.strip()) for part in out.split(","))
        except Exception as exc:
            raise ProbeError(f"nvidia-smi probe failed: {exc}") from exc
        return VramSnapshot(total=total_mib * 1024**2, used=used_mib * 1024**2)


def default_probe(gpu_index: int = 0) -> VramProbe:  # pragma: no cover - env-dependent
    """NVML if it initializes, else nvidia-smi."""
    try:
        probe = NvmlProbe(gpu_index=gpu_index)
        probe.read()
        return probe
    except Exception:
        return NvidiaSmiProbe(gpu_index=gpu_index)
