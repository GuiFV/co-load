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

    async def is_ready(self, model: str) -> bool:
        if self._model != model or self._process is None or not self._process.is_running():
            return False
        return await self._healthy()

    async def load(self, model: str, budget_bytes: int, total_bytes: int) -> None:
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
        if self._model is not None and self._process is not None and self._process.is_running():
            return [self._model]
        return []

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
        deadline = asyncio.get_running_loop().time() + self._health_timeout_s
        while asyncio.get_running_loop().time() < deadline:
            if self._process is not None and not self._process.is_running():
                raise BackendError(f"vllm process for '{self._model}' exited during startup")
            if await self._healthy():
                return
            await asyncio.sleep(self._health_poll_interval_s)
        raise BackendError(
            f"vllm '{self._model}' failed health check within {self._health_timeout_s}s"
        )
