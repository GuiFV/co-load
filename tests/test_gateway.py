"""Gateway: single admission point, OpenAI-compatible, honest 503s."""

import json

import httpx
import pytest

from coload.backends.base import BackendError
from coload.config import Config
from coload.gateway import create_app
from coload.orchestrator import NotEnoughVram

GIB = 2**30


class FakeOrchestrator:
    """Scripted orchestrator double keyed by model name."""

    def __init__(self):
        self.calls = []

    async def ensure_ready(self, model: str) -> str:
        self.calls.append(model)
        if model == "unknown-model":
            raise KeyError(model)
        if model == "too-big":
            raise NotEnoughVram(
                "too-big", 20 * GIB, 4 * GIB, 6 * GIB, {"oll": ["small"]}
            )
        if model == "broken":
            raise BackendError("engine crashed on start")
        return "http://backend.local"

    async def status(self):
        return {"vram": {"total_gb": 24.0}, "engines": {}}

    async def unload_model(self, model: str) -> bool:
        self.calls.append(f"unload:{model}")
        if model == "unknown-model":
            raise KeyError(model)
        return model == "small"


class BackendSim:
    def __init__(self):
        self.requests = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hi"}}]},
            headers={"x-backend": "sim"},
        )


@pytest.fixture
def config():
    return Config.model_validate(
        {
            "engines": {
                "oll": {
                    "kind": "ollama",
                    "base_url": "http://localhost:11434",
                    "models": {"small": {"est_vram_gb": 8}},
                }
            }
        }
    )


@pytest.fixture
def backend_sim():
    return BackendSim()


@pytest.fixture
def orch():
    return FakeOrchestrator()


@pytest.fixture
def client(config, orch, backend_sim):
    app = create_app(
        config=config,
        orchestrator=orch,
        proxy_client=httpx.AsyncClient(
            transport=httpx.MockTransport(backend_sim.handler)
        ),
    )
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gw"
    )


class TestChatCompletions:
    async def test_proxies_to_backend(self, client, orch, backend_sim):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "small", "messages": [{"role": "user", "content": "x"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "hi"
        assert orch.calls == ["small"]
        # request forwarded to the backend's URL, same path, body intact
        fwd = backend_sim.requests[0]
        assert str(fwd.url) == "http://backend.local/v1/chat/completions"
        assert json.loads(fwd.content)["model"] == "small"

    async def test_missing_model_field_is_400(self, client):
        resp = await client.post("/v1/chat/completions", json={"messages": []})
        assert resp.status_code == 400

    async def test_unknown_model_is_404(self, client):
        resp = await client.post(
            "/v1/chat/completions", json={"model": "unknown-model"}
        )
        assert resp.status_code == 404

    async def test_vram_full_is_503_with_guidance(self, client):
        resp = await client.post("/v1/chat/completions", json={"model": "too-big"})
        assert resp.status_code == 503
        err = resp.json()["error"]
        assert "Evict" in err["message"]
        assert err["resident"] == {"oll": ["small"]}

    async def test_backend_failure_is_502(self, client):
        resp = await client.post("/v1/chat/completions", json={"model": "broken"})
        assert resp.status_code == 502


class TestEmbeddings:
    async def test_embeddings_route_proxies(self, client, backend_sim):
        resp = await client.post(
            "/v1/embeddings", json={"model": "small", "input": "text"}
        )
        assert resp.status_code == 200
        assert str(backend_sim.requests[0].url).endswith("/v1/embeddings")


class TestModelLifecycleEndpoints:
    """CLI-facing endpoints: warm a model up, or evict it, explicitly."""

    async def test_load_endpoint_warms_model(self, client, orch):
        resp = await client.post("/models/load", json={"model": "small"})
        assert resp.status_code == 200
        assert resp.json() == {"model": "small", "status": "loaded"}
        assert orch.calls == ["small"]

    async def test_load_endpoint_maps_vram_full_to_503(self, client):
        resp = await client.post("/models/load", json={"model": "too-big"})
        assert resp.status_code == 503
        assert "Evict" in resp.json()["error"]["message"]

    async def test_load_endpoint_unknown_model_404(self, client):
        resp = await client.post("/models/load", json={"model": "unknown-model"})
        assert resp.status_code == 404

    async def test_unload_endpoint(self, client, orch):
        resp = await client.post("/models/unload", json={"model": "small"})
        assert resp.status_code == 200
        assert resp.json() == {"model": "small", "status": "unloaded"}
        assert orch.calls == ["unload:small"]

    async def test_unload_endpoint_not_resident(self, client):
        resp = await client.post("/models/unload", json={"model": "other"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "not-resident"

    async def test_unload_endpoint_missing_model_400(self, client):
        resp = await client.post("/models/unload", json={})
        assert resp.status_code == 400


class TestIntrospection:
    async def test_status_endpoint(self, client):
        resp = await client.get("/status")
        assert resp.status_code == 200
        assert resp.json()["vram"]["total_gb"] == 24.0

    async def test_v1_models_lists_configured(self, client):
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        ids = [m["id"] for m in resp.json()["data"]]
        assert ids == ["small"]
