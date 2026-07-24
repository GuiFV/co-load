"""Orchestrator: the serialized measure -> fit -> load -> observe spine.

Invariants under test:
1. VRAM is measured live, *inside* the load-mutex, right before allocating.
2. budget = free - total * buffer_pct (buffer_pct is the user setting).
3. When a model doesn't fit: alert + refuse, never auto-kill.
"""

import asyncio

import pytest

from coload.backends.base import BackendError
from coload.config import Config
from coload.estimates import EstimateStore
from coload.orchestrator import NotEnoughVram, Orchestrator
from coload.vram import VramSnapshot
from tests.conftest import GIB, FakeBackend, FakeProbe


def make_config(**over):
    base = {
        "buffer_pct": 0.10,
        "idle_ttl_seconds": 900,
        "engines": {
            "oll": {
                "kind": "ollama",
                "base_url": "http://localhost:11434",
                "models": {
                    "small": {"est_vram_gb": 8},
                    "tiny": {"est_vram_gb": 1},
                },
            },
            "vl": {
                "kind": "vllm",
                "base_url": "http://localhost:8000",
                "start": "vllm serve {model} --gpu-memory-utilization {budget_frac}",
                "models": {"big": {"est_vram_gb": 20}},
            },
        },
    }
    base.update(over)
    return Config.model_validate(base)


@pytest.fixture
def parts(tmp_path, clock, alerter):
    config = make_config()
    probe = FakeProbe(VramSnapshot(total=24 * GIB, used=4 * GIB))
    backends = {"oll": FakeBackend("oll"), "vl": FakeBackend("vl", url="http://fake:8000")}
    estimates = EstimateStore(
        tmp_path / "learned.json",
        seeds={"small": 8 * GIB, "tiny": 1 * GIB, "big": 20 * GIB},
    )
    orch = Orchestrator(
        config=config,
        probe=probe,
        backends=backends,
        estimates=estimates,
        alerter=alerter,
        clock=clock,
    )
    return orch, probe, backends, estimates, alerter, clock


class TestFitAndLoad:
    async def test_loads_when_model_fits(self, parts):
        orch, probe, backends, *_ = parts
        url = await orch.ensure_ready("small")
        assert url == backends["oll"].url
        # budget = free(20G) - 10% of total(2.4G)
        model, budget, total = backends["oll"].loads[0]
        assert model == "small"
        assert budget == 20 * GIB - int(24 * GIB * 0.10)
        assert total == 24 * GIB

    async def test_budget_uses_user_configured_buffer_pct(self, tmp_path, clock, alerter):
        """The user-settable buffer percentage flows into the fit check."""
        config = make_config(buffer_pct=0.25)
        probe = FakeProbe(VramSnapshot(total=24 * GIB, used=4 * GIB))
        backends = {"oll": FakeBackend("oll"), "vl": FakeBackend("vl")}
        estimates = EstimateStore(tmp_path / "l.json", seeds={"small": 8 * GIB})
        orch = Orchestrator(config, probe, backends, estimates, alerter, clock)

        await orch.ensure_ready("small")
        _, budget, _ = backends["oll"].loads[0]
        assert budget == 20 * GIB - int(24 * GIB * 0.25)

    async def test_fast_path_skips_probe_when_ready(self, parts):
        orch, probe, backends, *_ = parts
        backends["oll"].ready.add("small")
        await orch.ensure_ready("small")
        assert probe.reads == 0
        assert backends["oll"].loads == []

    async def test_unknown_model_raises_key_error(self, parts):
        orch, *_ = parts
        with pytest.raises(KeyError):
            await orch.ensure_ready("nope")


class TestFitRefusal:
    async def test_no_fit_raises_alerts_and_never_kills(self, parts):
        orch, probe, backends, _, alerter, _ = parts
        backends["oll"].ready.add("small")  # something else is resident
        probe.push(VramSnapshot(total=24 * GIB, used=20 * GIB))  # only 4G free

        with pytest.raises(NotEnoughVram) as exc_info:
            await orch.ensure_ready("big")  # needs 20G

        assert backends["vl"].loads == []          # nothing loaded
        assert backends["oll"].unloads == []       # nothing auto-killed
        assert len(alerter.alerts) == 1
        alert = alerter.alerts[0]
        assert "big" in alert.message
        # the alert names what's resident so the human can decide what to evict
        assert "small" in str(alert.context.get("resident"))
        assert exc_info.value.needed_bytes == 20 * GIB

    async def test_measures_inside_mutex_not_stale(self, parts):
        """Second queued load must see VRAM as the first load left it."""
        orch, probe, backends, *_ = parts
        # first read: 20G free; after "small" loads, 12G free
        probe.push(VramSnapshot(total=24 * GIB, used=4 * GIB))   # load measurement
        probe.push(VramSnapshot(total=24 * GIB, used=12 * GIB))  # post-load observe
        probe.push(VramSnapshot(total=24 * GIB, used=12 * GIB))  # big's measurement
        probe.push(VramSnapshot(total=24 * GIB, used=12 * GIB))

        results = await asyncio.gather(
            orch.ensure_ready("small"),
            orch.ensure_ready("big"),
            return_exceptions=True,
        )
        # "big" (20G est) must have been refused against the *post-load* 12G free
        assert any(isinstance(r, NotEnoughVram) for r in results)


class TestSerialization:
    async def test_loads_never_overlap(self, parts):
        orch, _, backends, *_ = parts
        await asyncio.gather(orch.ensure_ready("small"), orch.ensure_ready("tiny"))
        assert backends["oll"].max_concurrent_loads == 1

    async def test_concurrent_same_model_loads_once(self, parts):
        orch, _, backends, *_ = parts
        await asyncio.gather(orch.ensure_ready("small"), orch.ensure_ready("small"))
        assert len(backends["oll"].loads) == 1


class TestLearnedEstimates:
    async def test_observes_real_usage_after_load(self, parts):
        orch, probe, backends, estimates, *_ = parts
        # fixture snapshot (4G used) serves the pre-load read; this one the post-load
        probe.push(VramSnapshot(total=24 * GIB, used=13 * GIB))
        await orch.ensure_ready("small")
        assert estimates.estimate("small") == 9 * GIB  # learned 9G > seeded 8G

    async def test_failed_load_observes_nothing(self, parts, tmp_path, clock, alerter):
        orch, probe, backends, estimates, *_ = parts
        backends["oll"].fail_load = True
        with pytest.raises(BackendError):
            await orch.ensure_ready("small")
        assert estimates.estimate("small") == 8 * GIB  # still the seed


class TestIdleTtl:
    async def test_stops_idle_model_after_ttl(self, parts):
        orch, _, backends, _, _, clock = parts
        await orch.ensure_ready("small")
        clock.advance(901)
        stopped = await orch.stop_idle()
        assert stopped == ["small"]
        assert backends["oll"].unloads == ["small"]

    async def test_keeps_recently_used_model(self, parts):
        orch, _, backends, _, _, clock = parts
        await orch.ensure_ready("small")
        clock.advance(500)
        assert await orch.stop_idle() == []
        assert backends["oll"].unloads == []

    async def test_request_resets_ttl(self, parts):
        orch, _, backends, _, _, clock = parts
        await orch.ensure_ready("small")
        clock.advance(800)
        await orch.ensure_ready("small")  # touch via fast path
        clock.advance(200)
        assert await orch.stop_idle() == []

    async def test_zero_ttl_disables_idle_stop(self, tmp_path, clock, alerter):
        config = make_config(idle_ttl_seconds=0)
        probe = FakeProbe(VramSnapshot(total=24 * GIB, used=4 * GIB))
        backends = {"oll": FakeBackend("oll"), "vl": FakeBackend("vl")}
        estimates = EstimateStore(tmp_path / "l.json", seeds={"small": 8 * GIB})
        orch = Orchestrator(config, probe, backends, estimates, alerter, clock)
        await orch.ensure_ready("small")
        clock.advance(10_000)
        assert await orch.stop_idle() == []


class TestStatus:
    async def test_status_reports_vram_and_residents(self, parts):
        orch, _, backends, *_ = parts
        await orch.ensure_ready("small")
        status = await orch.status()
        assert status["vram"]["total_gb"] == 24.0
        assert status["buffer_pct"] == 0.10
        assert status["engines"]["oll"]["resident"] == ["small"]
