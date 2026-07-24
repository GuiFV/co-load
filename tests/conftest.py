"""Shared fakes for orchestrator/watchdog/gateway tests."""

import pytest

from coload.backends.base import Backend, BackendError
from coload.vram import VramSnapshot

GIB = 2**30


class FakeProbe:
    """Programmable probe: returns snapshots from a queue, repeats the last."""

    def __init__(self, *snapshots: VramSnapshot):
        self._queue = list(snapshots)
        self.reads = 0

    def push(self, snap: VramSnapshot):
        self._queue.append(snap)

    def read(self) -> VramSnapshot:
        self.reads += 1
        if len(self._queue) > 1:
            return self._queue.pop(0)
        return self._queue[0]


class FakeBackend(Backend):
    def __init__(self, name: str, url: str = "http://fake:1234", fail_load: bool = False):
        super().__init__(name)
        self.url = url
        self.ready: set[str] = set()
        self.loads: list[tuple[str, int, int]] = []
        self.unloads: list[str] = []
        self.fail_load = fail_load
        self.concurrent_loads = 0
        self.max_concurrent_loads = 0

    async def is_ready(self, model: str) -> bool:
        return model in self.ready

    async def load(self, model: str, budget_bytes: int, total_bytes: int) -> None:
        import asyncio

        self.concurrent_loads += 1
        self.max_concurrent_loads = max(self.max_concurrent_loads, self.concurrent_loads)
        await asyncio.sleep(0)  # yield, exposing races if the mutex is missing
        self.concurrent_loads -= 1
        if self.fail_load:
            raise BackendError(f"load of {model} failed")
        self.loads.append((model, budget_bytes, total_bytes))
        self.ready.add(model)

    async def unload(self, model: str) -> None:
        self.unloads.append(model)
        self.ready.discard(model)

    async def resident_models(self) -> list[str]:
        return sorted(self.ready)

    def proxy_url(self, model: str) -> str:
        return self.url


class RecordingAlerter:
    def __init__(self):
        self.alerts = []

    async def send(self, alert):
        self.alerts.append(alert)


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def alerter():
    return RecordingAlerter()
