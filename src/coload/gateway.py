"""OpenAI-compatible gateway: the single admission point.

Apps talk to coload as if it were any OpenAI-style server; the gateway routes
by the ``model`` field, asks the orchestrator to make it servable (which may
summon an engine), then proxies the request through, streaming included.
Both Ollama and vLLM expose OpenAI-compatible ``/v1/*`` surfaces, so the path
is forwarded unchanged.

Refusals are honest: 503 with what's resident and what to do about it.
"""

from __future__ import annotations

import logging
from typing import Protocol

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .backends.base import BackendError
from .config import Config
from .orchestrator import NotEnoughVram

logger = logging.getLogger("coload.gateway")

_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


class OrchestratorPort(Protocol):
    """What the gateway needs from the orchestrator (ISP: nothing more)."""

    async def ensure_ready(self, model: str) -> str: ...
    async def status(self) -> dict: ...


def create_app(
    config: Config,
    orchestrator: OrchestratorPort,
    proxy_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    app = FastAPI(title="coload", version="0.1.0")
    client = proxy_client or httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))

    async def _admit_and_proxy(request: Request) -> JSONResponse | StreamingResponse:
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "request body must be JSON")
        model = body.get("model")
        if not model:
            return _error(400, "missing 'model' field")

        try:
            target = await orchestrator.ensure_ready(model)
        except KeyError:
            return _error(404, f"model '{model}' is not configured in coload")
        except NotEnoughVram as exc:
            return _error(503, str(exc), resident=exc.resident)
        except BackendError as exc:
            return _error(502, f"backend failed to start '{model}': {exc}")

        upstream = client.build_request(
            method=request.method,
            url=f"{target}{request.url.path}",
            headers={
                k: v for k, v in request.headers.items()
                if k.lower() not in _HOP_BY_HOP
            },
            content=await request.body(),
        )
        try:
            resp = await client.send(upstream, stream=True)
        except httpx.HTTPError as exc:
            return _error(502, f"proxy to '{model}' backend failed: {exc}")

        headers = {
            k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP
        }
        if resp.is_stream_consumed:  # body already materialized (e.g. buffered)
            return Response(resp.content, status_code=resp.status_code, headers=headers)
        return StreamingResponse(
            resp.aiter_raw(), status_code=resp.status_code, headers=headers
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _admit_and_proxy(request)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _admit_and_proxy(request)

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        return await _admit_and_proxy(request)

    @app.get("/v1/models")
    async def list_models():
        data = [
            {"id": model, "object": "model", "owned_by": engine_name}
            for engine_name, engine in config.engines.items()
            for model in engine.models
        ]
        return {"object": "list", "data": data}

    @app.get("/status")
    async def status():
        return await orchestrator.status()

    return app


def _error(status_code: int, message: str, **extra) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, **extra}},
    )
