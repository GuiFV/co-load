"""vLLM adapter.

vLLM grabs its VRAM slice at process startup (``gpu_memory_utilization``) and
holds it for the process lifetime, so coload runs one model per process,
sizes the slice from the *live* budget at load time, and frees VRAM by
stopping the process.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import Protocol

import httpx

from .base import Backend, BackendError


class ProcessHandle(Protocol):
    def is_running(self) -> bool: ...
    def terminate(self) -> None: ...


class ProcessLauncher(Protocol):
    def launch(self, command: str, env: dict[str, str] | None = None) -> ProcessHandle: ...


class _PopenHandle:  # pragma: no cover - thin wrapper over subprocess
    def __init__(self, proc: subprocess.Popen):
        self._proc = proc

    def is_running(self) -> bool:
        return self._proc.poll() is None

    def terminate(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()


class SubprocessLauncher:  # pragma: no cover - exercised only in integration
    def launch(self, command: str, env: dict[str, str] | None = None) -> ProcessHandle:
        import os

        merged = {**os.environ, **(env or {})}
        return _PopenHandle(subprocess.Popen(command, shell=True, env=merged))


class VllmBackend(Backend):
    # gpu_memory_utilization is a claim on the card, not a measurement of the
    # model: whatever it is handed, it takes and keeps.
    sizes_to_budget = True

    def __init__(
        self,
        name: str,
        base_url: str,
        start_template: str,
        stop_template: str | None = None,
        launcher: ProcessLauncher | None = None,
        client: httpx.AsyncClient | None = None,
        health_timeout_s: float = 180.0,
        health_poll_interval_s: float = 2.0,
        stop_timeout_s: float = 60.0,
    ):
        super().__init__(name)
        self._base_url = base_url.rstrip("/")
        self._start_template = start_template
        self._stop_template = stop_template
        self._launcher = launcher or SubprocessLauncher()
        self._client = client or httpx.AsyncClient(timeout=10.0)
        self._health_timeout_s = health_timeout_s
        self._health_poll_interval_s = health_poll_interval_s
        self._stop_timeout_s = stop_timeout_s
        self._process: ProcessHandle | None = None
        self._model: str | None = None
        self._rediscovered = False

    def _owns_process(self) -> bool:
        """Whether the handle we hold is the engine, or just its starter.

        A stop command is the declaration that something else owns the engine:
        `docker compose up -d` hands off to the daemon, `systemctl start` to
        systemd. In those cases the handle we kept belongs to a starter that
        exits immediately, so its liveness says nothing about the engine and
        health is the only evidence there is.
        """
        return self._stop_template is None

    async def is_ready(self, model: str) -> bool:
        if self._model is None:
            await self._rediscover()
        if self._model != model:
            return False
        if self._owns_process() and (
            self._process is None or not self._process.is_running()
        ):
            return False
        return await self._healthy()

    async def load(self, model: str, budget_bytes: int, total_bytes: int) -> None:
        if self._model is None:
            await self._rediscover()
        if self._model is not None:
            await self.unload(self._model)

        budget_frac = round(budget_bytes / total_bytes, 2)
        command = self._start_template.format(model=model, budget_frac=budget_frac)
        # Also exported as env vars so detached starters (docker compose) can
        # interpolate them where CLI placeholders can't reach.
        env = {"COLOAD_MODEL": model, "COLOAD_BUDGET_FRAC": str(budget_frac)}
        self._process = self._launcher.launch(command, env=env)
        self._model = model

        try:
            await self._wait_healthy()
        except BackendError:
            await self._teardown()
            raise

    async def unload(self, model: str) -> None:
        await self._teardown()

    async def resident_models(self) -> list[str]:
        if self._model is None:
            await self._rediscover()
        if self._model is None:
            return []
        if self._owns_process():
            running = self._process is not None and self._process.is_running()
            return [self._model] if running else []
        # Detached: ask the engine, because the starter is long gone. Getting
        # this wrong makes coload believe a resident model is absent, so it
        # loads it again and its VRAM accounting drifts from the card.
        return [self._model] if await self._healthy() else []

    async def _rediscover(self) -> None:
        """Re-learn what a detached engine is serving.

        The served model lives in this process, so a restart forgets it while
        the engine keeps running. For a detached engine the engine itself is
        the only witness left: ask its /v1/models rather than believe the
        card is empty. An owned process is different: a fresh backend owns
        none, so whatever answers the port belongs to somebody else and is
        left alone. Never raises; an engine that is down simply stays
        unknown.

        One attempt, ever: this is boot-time reconciliation, the counterpart
        of the orchestrator's adopt_resident. Retrying on every status poll
        or admission pass would pay a connection attempt against a port with
        nothing behind it for as long as the engine stays down.
        """
        if self._rediscovered or self._owns_process():
            return
        self._rediscovered = True
        try:
            resp = await self._client.get(f"{self._base_url}/v1/models")
            if resp.status_code != 200:
                return
            served = resp.json().get("data") or []
        except (httpx.HTTPError, ValueError):
            return
        if served:
            self._model = served[0].get("id")

    def proxy_url(self, model: str) -> str:
        return self._base_url

    async def _teardown(self) -> None:
        if self._stop_template is not None:
            # A stop command makes detached starts (docker compose, systemd)
            # first-class: run it and wait, bounded, for it to finish.
            command = self._stop_template.format(model=self._model or "")
            stopper = self._launcher.launch(command)
            deadline = asyncio.get_running_loop().time() + self._stop_timeout_s
            while stopper.is_running() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(self._health_poll_interval_s)
        if self._process is not None:
            self._process.terminate()
        self._process = None
        self._model = None

    async def _healthy(self) -> bool:
        try:
            resp = await self._client.get(f"{self._base_url}/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def _wait_healthy(self) -> None:
        # A detached starter exits the moment it has handed the engine off:
        # `docker compose up -d` returns as soon as the container is created,
        # long before vLLM has read its weights. That exit is success, not
        # death, so only a start we own says anything about the engine by
        # ending. Having a stop command is exactly what "someone else owns the
        # process" means here, which is why _teardown runs it.
        #
        # Without this, every detached start failed with "exited during
        # startup" unless the model happened to be healthy inside the first
        # poll, which no large model is.
        owns_process = self._owns_process()
        deadline = asyncio.get_running_loop().time() + self._health_timeout_s
        while asyncio.get_running_loop().time() < deadline:
            if owns_process and self._process is not None and not self._process.is_running():
                raise BackendError(f"vllm process for '{self._model}' exited during startup")
            if await self._healthy():
                return
            await asyncio.sleep(self._health_poll_interval_s)
        raise BackendError(
            f"vllm '{self._model}' failed health check within {self._health_timeout_s}s"
        )
