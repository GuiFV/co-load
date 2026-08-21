"""Composition root: build the whole object graph from a Config.

The only module that knows about concrete implementations; everything else
depends on abstractions.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass

from fastapi import FastAPI

from .alerts import RateLimitedAlerter, build_alerter
from .backends.base import Backend
from .backends.registry import build_backend
from .config import Config
from .estimates import EstimateStore
from .gateway import create_app
from .orchestrator import Orchestrator
from .room import CudaRoomMaker
from .vram import VramProbe, default_probe
from .watchdog import Watchdog


@dataclass
class Runtime:
    config: Config
    app: FastAPI
    orchestrator: Orchestrator
    watchdog: Watchdog
    backends: dict[str, Backend]
    estimates: EstimateStore


def build_runtime(config: Config, probe: VramProbe | None = None) -> Runtime:
    probe = probe or default_probe(config.gpu)
    backends = {name: build_backend(name, cfg) for name, cfg in config.engines.items()}
    seeds = {
        model: model_cfg.est_vram_bytes
        for engine in config.engines.values()
        for model, model_cfg in engine.models.items()
    }
    estimates = EstimateStore(config.estimates_path, seeds)
    alerter = RateLimitedAlerter(build_alerter(config.alert))
    orchestrator = Orchestrator(
        config=config,
        probe=probe,
        backends=backends,
        estimates=estimates,
        alerter=alerter,
        room_maker=CudaRoomMaker(),
    )
    watchdog = Watchdog(probe=probe, alerter=alerter, buffer_pct=config.buffer_pct)
    app = create_app(config=config, orchestrator=orchestrator)

    _attach_background_tasks(app, config, orchestrator, watchdog)
    return Runtime(
        config=config,
        app=app,
        orchestrator=orchestrator,
        watchdog=watchdog,
        backends=backends,
        estimates=estimates,
    )


def _attach_background_tasks(
    app: FastAPI, config: Config, orchestrator: Orchestrator, watchdog: Watchdog
) -> None:  # pragma: no cover - exercised only under a live server
    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        # Before anything else: reconcile with what is already on the card, so
        # a restart does not orphan the models this process loaded last time.
        await orchestrator.adopt_resident()
        tasks = [
            asyncio.create_task(watchdog.run(config.watchdog_interval_s)),
            asyncio.create_task(orchestrator.run_ttl_loop()),
        ]
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    app.router.lifespan_context = lifespan
