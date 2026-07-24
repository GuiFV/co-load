"""Ollama adapter.

Ollama's daemon runs externally and manages its own model swaps; coload
triggers loads/unloads through the HTTP API's ``keep_alive`` mechanism:
``keep_alive: -1`` pins a model resident, ``keep_alive: 0`` evicts it.
Idle TTL is coload's job, so models are pinned and explicitly unloaded.
"""

from __future__ import annotations

import httpx

from .base import Backend, BackendError


class OllamaBackend(Backend):
    def __init__(self, name: str, base_url: str, client: httpx.AsyncClient | None = None):
        super().__init__(name)
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(base_url=self._base_url, timeout=300.0)

    async def is_ready(self, model: str) -> bool:
        # Ollama reports "name:tag"; a bare configured name matches its :latest.
        resident = await self.resident_models()
        return model in resident or f"{model}:latest" in resident

    async def load(self, model: str, budget_bytes: int, total_bytes: int) -> None:
        # An empty generate with keep_alive pins the model into VRAM. Ollama
        # sizes itself; budget_bytes is advisory here (the fit check already
        # ran against the estimate).
        try:
            resp = await self._client.post(
                "/api/generate", json={"model": model, "keep_alive": -1}
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendError(f"ollama failed to load '{model}': {exc}") from exc

    async def unload(self, model: str) -> None:
        try:
            resp = await self._client.post(
                "/api/generate", json={"model": model, "keep_alive": 0}
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendError(f"ollama failed to unload '{model}': {exc}") from exc

    async def resident_models(self) -> list[str]:
        try:
            resp = await self._client.get("/api/ps")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendError(f"ollama /api/ps failed: {exc}") from exc
        return [m["name"] for m in resp.json().get("models", [])]

    def proxy_url(self, model: str) -> str:
        return self._base_url
