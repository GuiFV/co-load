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
        # The model needs 8G and the budget is far larger (free 20G less the
        # 10% buffer), so it is handed what it needs. Handing over the whole
        # budget is how an engine that sizes itself to it swallows the card.
        model, budget, total = backends["oll"].loads[0]
        assert model == "small"
        assert budget == 8 * GIB
        assert total == 24 * GIB

    async def test_headroom_is_left_for_everything_else(self, tmp_path, clock, alerter):
        """The point of handing over the estimate rather than the budget: a
        model needing a third of the card leaves the other two thirds for the
        next one, instead of an engine sizing itself to whatever was free."""
        config = make_config(buffer_pct=0.10)
        probe = FakeProbe(VramSnapshot(total=24 * GIB, used=4 * GIB))
        backends = {"oll": FakeBackend("oll"), "vl": FakeBackend("vl")}
        estimates = EstimateStore(tmp_path / "l.json", seeds={"tiny": 1 * GIB})
        orch = Orchestrator(config, probe, backends, estimates, alerter, clock)

        await orch.ensure_ready("tiny")
        _, budget, total = backends["oll"].loads[0]
        assert budget == 1 * GIB
        assert total - budget > 20 * GIB, "the rest of the card stays available"

    async def test_budget_uses_user_configured_buffer_pct(self, tmp_path, clock, alerter):
        """The user-settable buffer percentage flows into the fit check."""
        config = make_config(buffer_pct=0.25)
        probe = FakeProbe(VramSnapshot(total=24 * GIB, used=4 * GIB))
        backends = {"oll": FakeBackend("oll"), "vl": FakeBackend("vl")}
        # Fits the 17.6G budget at the default 10% buffer, not the 14G at 25%,
        # so the setting is what decides whether this load is allowed.
        estimates = EstimateStore(tmp_path / "l.json", seeds={"small": 16 * GIB})
        orch = Orchestrator(config, probe, backends, estimates, alerter, clock)

        with pytest.raises(NotEnoughVram):
            await orch.ensure_ready("small")

    async def test_an_elastic_engine_does_not_teach_its_own_budget(
        self, tmp_path, clock, alerter
    ):
        """vLLM keeps whatever slice it is handed, so measuring it afterwards
        reports our own decision back to us. Learning that is a ratchet: the
        next load is handed the measurement, exceeds it by the CUDA context it
        never counted, and teaches a bigger number again, until a model that
        ran an hour ago no longer fits the card."""
        config = make_config()
        # 4G used before, 28G after: the engine took its slice plus the CUDA
        # context overhead it does not count against its own utilisation.
        probe = FakeProbe(
            VramSnapshot(total=32 * GIB, used=4 * GIB),
            VramSnapshot(total=32 * GIB, used=28 * GIB),
        )
        elastic = FakeBackend("vl", url="http://fake:8000")
        elastic.sizes_to_budget = True
        backends = {"oll": FakeBackend("oll"), "vl": elastic}
        estimates = EstimateStore(tmp_path / "l.json", seeds={"big": 20 * GIB})
        orch = Orchestrator(config, probe, backends, estimates, alerter, clock)

        await orch.ensure_ready("big")
        assert estimates.estimate("big") == 20 * GIB, "learned its own allowance"

    async def test_a_fixed_engine_still_learns_what_it_measured(
        self, tmp_path, clock, alerter
    ):
        """Ollama uses what the weights need, so the measurement is real and
        worth keeping: that is how a seeded guess becomes a fact."""
        config = make_config()
        probe = FakeProbe(
            VramSnapshot(total=24 * GIB, used=4 * GIB),
            VramSnapshot(total=24 * GIB, used=13 * GIB),
        )
        backends = {"oll": FakeBackend("oll"), "vl": FakeBackend("vl")}
        estimates = EstimateStore(tmp_path / "l.json", seeds={"small": 8 * GIB})
        orch = Orchestrator(config, probe, backends, estimates, alerter, clock)

        await orch.ensure_ready("small")
        assert estimates.estimate("small") == 9 * GIB

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


class TestAutoEvictIdle:
    """Opt-in eviction of coload-loaded models when a request doesn't fit.

    Only models the orchestrator itself touched are candidates (LRU first);
    out-of-band residents are never evicted. Default is off.
    """

    def _parts(self, tmp_path, clock, alerter, probe, total_gib=32):
        config = Config.model_validate({
            "buffer_pct": 0.10,
            "auto_evict_idle": True,
            "engines": {
                "oll": {
                    "kind": "ollama",
                    "base_url": "http://localhost:11434",
                    "models": {
                        "tiny": {"est_vram_gb": 1},
                        "small": {"est_vram_gb": 8},
                        "big": {"est_vram_gb": 20},
                    },
                },
            },
        })
        backends = {"oll": FakeBackend("oll")}
        estimates = EstimateStore(
            tmp_path / "learned.json",
            seeds={"tiny": 1 * GIB, "small": 8 * GIB, "big": 20 * GIB},
        )
        orch = Orchestrator(config, probe, backends, estimates, alerter, clock)
        return orch, backends["oll"]

    async def test_evicts_idle_model_then_loads(self, tmp_path, clock, alerter, monkeypatch):
        monkeypatch.setattr("coload.orchestrator.EVICT_SETTLE_S", 0)
        G = GIB
        probe = FakeProbe(
            VramSnapshot(32 * G, 2 * G),   # small: pre-load
            VramSnapshot(32 * G, 10 * G),  # small: post-load observe
            VramSnapshot(32 * G, 10 * G),  # big: pre-load (no fit: budget 18.8)
            VramSnapshot(32 * G, 10 * G),  # big: evict-loop initial budget
            VramSnapshot(32 * G, 2 * G),   # big: after evicting small (fits)
            VramSnapshot(32 * G, 2 * G),   # big: re-measured 'before'
            VramSnapshot(32 * G, 22 * G),  # big: post-load observe
        )
        orch, backend = self._parts(tmp_path, clock, alerter, probe)

        await orch.ensure_ready("small")
        await orch.ensure_ready("big")

        assert backend.unloads == ["small"]
        assert [m for m, *_ in backend.loads] == ["small", "big"]

    async def test_evicts_lru_first_and_stops_when_enough(
        self, tmp_path, clock, alerter, monkeypatch
    ):
        monkeypatch.setattr("coload.orchestrator.EVICT_SETTLE_S", 0)
        G = GIB
        probe = FakeProbe(
            VramSnapshot(32 * G, 2 * G),   # small: pre-load
            VramSnapshot(32 * G, 10 * G),  # small: post-load
            VramSnapshot(32 * G, 10 * G),  # tiny: pre-load
            VramSnapshot(32 * G, 11 * G),  # tiny: post-load
            VramSnapshot(32 * G, 11 * G),  # big: pre-load (no fit)
            VramSnapshot(32 * G, 11 * G),  # big: evict-loop initial
            VramSnapshot(32 * G, 3 * G),   # big: after evicting small (fits)
            VramSnapshot(32 * G, 3 * G),   # big: re-measured 'before'
            VramSnapshot(32 * G, 23 * G),  # big: post-load observe
        )
        orch, backend = self._parts(tmp_path, clock, alerter, probe)

        await orch.ensure_ready("small")   # older -> LRU victim
        clock.advance(10)
        await orch.ensure_ready("tiny")    # newer -> survives
        await orch.ensure_ready("big")

        assert backend.unloads == ["small"]
        assert await backend.is_ready("tiny")

    async def test_never_evicts_out_of_band_residents(
        self, tmp_path, clock, alerter, monkeypatch
    ):
        monkeypatch.setattr("coload.orchestrator.EVICT_SETTLE_S", 0)
        G = GIB
        probe = FakeProbe(VramSnapshot(32 * G, 12 * G))  # budget 16.8 < 20
        orch, backend = self._parts(tmp_path, clock, alerter, probe)
        backend.ready.add("small")  # resident, but NOT loaded via coload

        with pytest.raises(NotEnoughVram):
            await orch.ensure_ready("big")

        assert backend.unloads == []          # out-of-band: never evicted
        assert len(alerter.alerts) == 1       # refusal still alerts the human

    async def test_refuses_when_eviction_is_not_enough(
        self, tmp_path, clock, alerter, monkeypatch
    ):
        monkeypatch.setattr("coload.orchestrator.EVICT_SETTLE_S", 0)
        G = GIB
        probe = FakeProbe(
            VramSnapshot(24 * G, 4 * G),   # small: pre-load
            VramSnapshot(24 * G, 12 * G),  # small: post-load
            VramSnapshot(24 * G, 12 * G),  # big: pre-load (no fit)
            VramSnapshot(24 * G, 12 * G),  # big: evict-loop initial
            VramSnapshot(24 * G, 4 * G),   # after evicting small: budget 17.6, still < 20
            VramSnapshot(24 * G, 4 * G),   # re-measured 'before'
        )
        orch, backend = self._parts(tmp_path, clock, alerter, probe, total_gib=24)

        await orch.ensure_ready("small")
        with pytest.raises(NotEnoughVram):
            await orch.ensure_ready("big")

        assert backend.unloads == ["small"]           # tried what it could
        assert "big" not in {m for m, *_ in backend.loads}
        assert len(alerter.alerts) == 1


class TestExplicitUnload:
    async def test_unload_model_frees_and_forgets(self, parts):
        orch, _, backends, _, _, clock = parts
        await orch.ensure_ready("small")
        assert await orch.unload_model("small") is True
        assert backends["oll"].unloads == ["small"]
        clock.advance(10_000)
        assert await orch.stop_idle() == []  # no longer tracked

    async def test_unload_model_not_resident_returns_false(self, parts):
        orch, _, backends, *_ = parts
        assert await orch.unload_model("small") is False
        assert backends["oll"].unloads == []

    async def test_unload_unknown_model_raises_key_error(self, parts):
        orch, *_ = parts
        with pytest.raises(KeyError):
            await orch.unload_model("nope")


class TestStatus:
    async def test_status_reports_vram_and_residents(self, parts):
        orch, _, backends, *_ = parts
        await orch.ensure_ready("small")
        status = await orch.status()
        assert status["vram"]["total_gb"] == 24.0
        assert status["buffer_pct"] == 0.10
        assert status["engines"]["oll"]["resident"] == ["small"]


class TestAdoptResident:
    """Coload must never lose track of a model it is responsible for.

    Both eviction paths iterate `_last_used`, which is in-memory. Restart the
    daemon with a model resident and that model became invisible: not idle-
    swept, not evictable to make room, holding VRAM until the engine itself
    was restarted. Adoption reconciles the table with what is actually loaded.
    """

    async def test_a_resident_model_is_adopted_and_becomes_idle_evictable(self, parts):
        orch, _probe, backends, _est, _alerter, clock = parts
        backends["oll"].ready.add("small")          # resident from a past process

        await orch.adopt_resident()

        assert "small" in (await orch.status())["last_used"]
        clock.advance(901)
        assert await orch.stop_idle() == ["small"]
        assert backends["oll"].unloads == ["small"]

    async def test_an_adopted_model_can_be_evicted_to_make_room(self, parts):
        orch, probe, backends, _est, _alerter, _clock = parts
        backends["oll"].ready.add("small")
        orch._config = make_config(auto_evict_idle=True)
        await orch.adopt_resident()
        # A 24 GiB card with 20 GiB held by the orphan, freed by the unload:
        # "big" fits only if the orphan can be offered up, which needs it to
        # have been adopted first.
        probe._queue[:] = [
            VramSnapshot(total=24 * GIB, used=20 * GIB),   # ensure_ready
            VramSnapshot(total=24 * GIB, used=20 * GIB),   # evict loop, before
            VramSnapshot(total=24 * GIB, used=1 * GIB),    # evict loop, after
            VramSnapshot(total=24 * GIB, used=1 * GIB),    # ensure_ready re-read
        ]

        await orch.ensure_ready("big")

        assert backends["oll"].unloads == ["small"]

    async def test_unconfigured_models_are_never_adopted(self, parts):
        """Out-of-band GPU users stay off limits: coload alerts, never evicts."""
        orch, _probe, backends, *_ = parts
        backends["oll"].ready.add("somebody-elses-model")

        await orch.adopt_resident()

        assert (await orch.status())["last_used"] == {}

    async def test_adoption_survives_an_unreachable_engine(self, parts):
        """A dead engine at boot must not stop the gateway from starting."""
        orch, _probe, backends, *_ = parts

        async def boom():
            raise BackendError("engine down")

        backends["vl"].resident_models = boom
        backends["oll"].ready.add("tiny")

        assert await orch.adopt_resident() == ["tiny"]
