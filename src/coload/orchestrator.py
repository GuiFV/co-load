"""The orchestrator: coload's brain.

Every load runs through one serialized routine (the load-mutex):

    acquire load-mutex
      free   = measured live via the probe, right now
      budget = free - total * buffer_pct        # buffer_pct: user setting, default 10%
      fits?  -> start backend sized to budget, wait healthy, learn real usage
      else   -> alert the human ("evict something and retry"); never auto-kill
    release load-mutex

Two invariants keep it safe:
1. Measurement happens *inside* the mutex, immediately before allocating, so
   the free figure cannot go stale under a concurrent load.
2. The buffer absorbs fragmentation, CUDA context growth, and fluctuations.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Mapping

from .alerts import Alert, Alerter
from .backends.base import Backend
from .config import Config
from .estimates import EstimateStore
from .vram import GIB, VramProbe

logger = logging.getLogger("coload.orchestrator")


def _gb(n: int) -> float:
    return round(n / GIB, 2)


class NotEnoughVram(RuntimeError):
    """The requested model does not fit in the current budget."""

    def __init__(
        self,
        model: str,
        needed_bytes: int,
        budget_bytes: int,
        free_bytes: int,
        resident: dict[str, list[str]],
    ):
        self.model = model
        self.needed_bytes = needed_bytes
        self.budget_bytes = budget_bytes
        self.free_bytes = free_bytes
        self.resident = resident
        super().__init__(
            f"'{model}' needs ~{_gb(needed_bytes)} GiB, but only "
            f"{_gb(budget_bytes)} GiB of budget is available "
            f"({_gb(free_bytes)} GiB free). Evict something and retry."
        )


class Orchestrator:
    def __init__(
        self,
        config: Config,
        probe: VramProbe,
        backends: Mapping[str, Backend],
        estimates: EstimateStore,
        alerter: Alerter,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._config = config
        self._probe = probe
        self._backends = backends
        self._estimates = estimates
        self._alerter = alerter
        self._clock = clock
        self._load_mutex = asyncio.Lock()
        self._last_used: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Admission control
    # ------------------------------------------------------------------ #

    async def ensure_ready(self, model: str) -> str:
        """Make ``model`` servable and return the URL to proxy to.

        Raises KeyError for unknown models, NotEnoughVram when the card is
        full (after alerting), BackendError when the engine fails to start.
        """
        backend = self._backends[self._config.engine_for_model(model)]

        if await backend.is_ready(model):  # fast path: no probe, no lock
            self._touch(model)
            return backend.proxy_url(model)

        async with self._load_mutex:
            if await backend.is_ready(model):  # a queued waiter already loaded it
                self._touch(model)
                return backend.proxy_url(model)

            before = self._probe.read()  # invariant 1: measured inside the mutex
            budget = before.budget(self._config.buffer_pct)
            needed = self._estimates.estimate(model)

            if needed > budget:
                await self._refuse(model, needed, budget, before.free)

            logger.info(
                "loading '%s' via %s (needs ~%s GiB, budget %s GiB)",
                model, backend.name, _gb(needed), _gb(budget),
            )
            await backend.load(model, budget, before.total)

            after = self._probe.read()
            self._estimates.observe(model, after.used - before.used)
            self._touch(model)
            return backend.proxy_url(model)

    async def _refuse(self, model: str, needed: int, budget: int, free: int) -> None:
        resident = await self._resident_map()
        error = NotEnoughVram(model, needed, budget, free, resident)
        await self._alerter.send(
            Alert(
                severity="warning",
                title="coload: model does not fit",
                message=str(error),
                context={"model": model, "resident": resident},
                dedup_key=f"fit:{model}",
            )
        )
        raise error

    # ------------------------------------------------------------------ #
    # Idle TTL
    # ------------------------------------------------------------------ #

    async def stop_idle(self) -> list[str]:
        """Unload models idle past the TTL; returns what was stopped."""
        ttl = self._config.idle_ttl_seconds
        if ttl <= 0:  # 0 disables idle-stop
            return []
        now = self._clock()
        expired = [m for m, ts in self._last_used.items() if now - ts >= ttl]

        stopped: list[str] = []
        for model in expired:
            backend = self._backends[self._config.engine_for_model(model)]
            async with self._load_mutex:
                if await backend.is_ready(model):
                    logger.info("idle TTL: unloading '%s' from %s", model, backend.name)
                    await backend.unload(model)
            self._last_used.pop(model, None)
            stopped.append(model)
        return stopped

    async def run_ttl_loop(self, interval_s: float = 30.0) -> None:  # pragma: no cover
        while True:
            await asyncio.sleep(interval_s)
            try:
                await self.stop_idle()
            except Exception:
                logger.exception("idle TTL sweep failed")

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    async def status(self) -> dict:
        snap = self._probe.read()
        return {
            "vram": {
                "total_gb": _gb(snap.total),
                "used_gb": _gb(snap.used),
                "free_gb": _gb(snap.free),
                "budget_gb": _gb(snap.budget(self._config.buffer_pct)),
            },
            "buffer_pct": self._config.buffer_pct,
            "engines": {
                name: {"resident": await backend.resident_models()}
                for name, backend in self._backends.items()
            },
            "last_used": dict(self._last_used),
        }

    async def _resident_map(self) -> dict[str, list[str]]:
        resident: dict[str, list[str]] = {}
        for name, backend in self._backends.items():
            try:
                models = await backend.resident_models()
            except Exception:  # a down engine must not break the alert
                models = []
            if models:
                resident[name] = models
        return resident

    def _touch(self, model: str) -> None:
        self._last_used[model] = self._clock()
