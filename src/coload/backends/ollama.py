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

    async def _keep_alive(self, model: str, keep_alive: int, verb: str) -> None:
        """Pin (keep_alive=-1) or evict (0) a model.

        An empty /api/generate does this for generative models, but embedding
        models 400 on generate — for those, fall back to /api/embed, which
        honors keep_alive the same way. Ollama sizes itself; the fit check
        already ran against the estimate.
        """
        try:
            resp = await self._client.post(
                "/api/generate", json={"model": model, "keep_alive": keep_alive}
            )
            if resp.status_code == 400:  # embedding model: cannot generate
                resp = await self._client.post(
                    "/api/embed",
                    json={"model": model, "input": "", "keep_alive": keep_alive},
                )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendError(f"ollama failed to {verb} '{model}': {exc}") from exc

    async def load(self, model: str, budget_bytes: int, total_bytes: int) -> None:
        await self._keep_alive(model, -1, "load")

    async def unload(self, model: str) -> None:
        await self._keep_alive(model, 0, "unload")

    async def resident_models(self) -> list[str]:
        try:
            resp = await self._client.get("/api/ps")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendError(f"ollama /api/ps failed: {exc}") from exc
        return [m["name"] for m in resp.json().get("models", [])]

    def proxy_url(self, model: str) -> str:
        return self._base_url
