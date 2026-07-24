"""VRAM probing: snapshot math, NVML probe, nvidia-smi fallback.

Probes are injected as dependencies (DIP) so everything is testable without a
physical GPU.
"""

import pytest

from coload.vram import (
    GIB,
    NvidiaSmiProbe,
    NvmlProbe,
    ProbeError,
    VramSnapshot,
)


class TestVramSnapshot:
    def test_free_is_total_minus_used(self):
        snap = VramSnapshot(total=24 * GIB, used=10 * GIB)
        assert snap.free == 14 * GIB

    def test_budget_subtracts_user_buffer_from_free(self):
        """budget = free - total * buffer_pct (buffer_pct is user-settable)."""
        snap = VramSnapshot(total=24 * GIB, used=10 * GIB)
        assert snap.budget(buffer_pct=0.10) == 14 * GIB - int(24 * GIB * 0.10)

    def test_budget_honors_custom_buffer_pct(self):
        snap = VramSnapshot(total=100, used=0)
        assert snap.budget(buffer_pct=0.25) == 75

    def test_budget_never_negative(self):
        snap = VramSnapshot(total=100, used=95)
        assert snap.budget(buffer_pct=0.10) == 0


class FakeNvml:
    """Stand-in for the pynvml module (DIP: NvmlProbe takes it injected)."""

    def __init__(self, total, used, fail_init=False):
        self._total, self._used = total, used
        self.fail_init = fail_init
        self.init_calls = 0

    def nvmlInit(self):
        self.init_calls += 1
        if self.fail_init:
            raise RuntimeError("no NVML driver")

    def nvmlDeviceGetHandleByIndex(self, index):
        return f"handle-{index}"

    def nvmlDeviceGetMemoryInfo(self, handle):
        class Info:
            pass

        info = Info()
        info.total, info.used = self._total, self._used
        return info


class TestNvmlProbe:
    def test_reads_snapshot(self):
        probe = NvmlProbe(gpu_index=0, nvml=FakeNvml(total=24 * GIB, used=6 * GIB))
        snap = probe.read()
        assert snap.total == 24 * GIB
        assert snap.used == 6 * GIB

    def test_init_only_once(self):
        nvml = FakeNvml(total=1, used=0)
        probe = NvmlProbe(gpu_index=0, nvml=nvml)
        probe.read()
        probe.read()
        assert nvml.init_calls == 1

    def test_init_failure_wrapped_as_probe_error(self):
        probe = NvmlProbe(gpu_index=0, nvml=FakeNvml(1, 0, fail_init=True))
        with pytest.raises(ProbeError):
            probe.read()


class TestNvidiaSmiProbe:
    def test_parses_csv_mib_output(self):
        def runner(cmd):
            assert "-i" in cmd and "1" in cmd
            return "24576, 10240\n"

        probe = NvidiaSmiProbe(gpu_index=1, runner=runner)
        snap = probe.read()
        assert snap.total == 24576 * 1024**2
        assert snap.used == 10240 * 1024**2

    def test_bad_output_raises_probe_error(self):
        probe = NvidiaSmiProbe(gpu_index=0, runner=lambda cmd: "garbage")
        with pytest.raises(ProbeError):
            probe.read()

    def test_runner_failure_raises_probe_error(self):
        def runner(cmd):
            raise FileNotFoundError("nvidia-smi not found")

        probe = NvidiaSmiProbe(gpu_index=0, runner=runner)
        with pytest.raises(ProbeError):
            probe.read()
