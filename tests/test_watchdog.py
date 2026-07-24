"""Watchdog: detects out-of-band VRAM usage the gateway didn't mediate.

Observe-only: it can alert, never prevent. Edge-triggered: one alert per
excursion over the safe line, re-armed when usage drops back under.
"""

from coload.vram import VramSnapshot
from coload.watchdog import Watchdog
from tests.conftest import GIB, FakeProbe


def snap(used_gib: float) -> VramSnapshot:
    return VramSnapshot(total=24 * GIB, used=int(used_gib * GIB))


def make_watchdog(probe, alerter, buffer_pct=0.10):
    return Watchdog(probe=probe, alerter=alerter, buffer_pct=buffer_pct)


class TestSafeLine:
    async def test_under_safe_line_no_alert(self, alerter):
        # safe line at 10% buffer = 21.6G
        wd = make_watchdog(FakeProbe(snap(20)), alerter)
        assert not await wd.check_once()
        assert alerter.alerts == []

    async def test_over_safe_line_alerts(self, alerter):
        wd = make_watchdog(FakeProbe(snap(23)), alerter)
        assert await wd.check_once()
        assert len(alerter.alerts) == 1
        assert alerter.alerts[0].severity == "critical"

    async def test_safe_line_honors_user_buffer_pct(self, alerter):
        """A 25% buffer moves the safe line down to 18G."""
        wd = make_watchdog(FakeProbe(snap(20)), alerter, buffer_pct=0.25)
        assert await wd.check_once()


class TestEdgeTriggering:
    async def test_sustained_overuse_alerts_once(self, alerter):
        probe = FakeProbe(snap(23))
        wd = make_watchdog(probe, alerter)
        await wd.check_once()
        await wd.check_once()
        await wd.check_once()
        assert len(alerter.alerts) == 1

    async def test_rearms_after_recovery(self, alerter):
        probe = FakeProbe(snap(23), snap(10), snap(23))
        wd = make_watchdog(probe, alerter)
        await wd.check_once()  # over -> alert
        await wd.check_once()  # recovered -> no alert
        await wd.check_once()  # over again -> alert
        assert len(alerter.alerts) == 2


class TestRobustness:
    async def test_probe_error_does_not_crash(self, alerter):
        class BrokenProbe:
            def read(self):
                raise RuntimeError("nvml gone")

        wd = make_watchdog(BrokenProbe(), alerter)
        assert not await wd.check_once()  # swallowed, logged
