"""CLI gateway client: list, warm up, evict, and chat with models over HTTP."""

import json

import httpx
import pytest

from coload.cli import GatewayClient, GatewayError, format_models_table


class GatewaySim:
    """Simulates the coload gateway HTTP API."""

    def __init__(self):
        self.resident = {"ollama": ["llama3.2"]}
        self.requests = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(200, json={
                "object": "list",
                "data": [
                    {"id": "llama3.2", "object": "model", "owned_by": "ollama"},
                    {"id": "big", "object": "model", "owned_by": "vllm"},
                ],
            })
        if path == "/status":
            return httpx.Response(200, json={
                "vram": {"total_gb": 24.0, "used_gb": 6.0, "free_gb": 18.0,
                         "budget_gb": 15.6},
                "buffer_pct": 0.1,
                "engines": {k: {"resident": v} for k, v in self.resident.items()},
            })
        if path == "/models/load":
            model = json.loads(request.content)["model"]
            if model == "big":
                return httpx.Response(503, json={"error": {
                    "message": "'big' needs ~20.0 GiB. Evict something and retry.",
                    "resident": self.resident,
                }})
            return httpx.Response(200, json={"model": model, "status": "loaded"})
        if path == "/models/unload":
            model = json.loads(request.content)["model"]
            status = "unloaded" if model == "llama3.2" else "not-resident"
            return httpx.Response(200, json={"model": model, "status": status})
        if path == "/v1/chat/completions":
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "hello from the model"}}],
            })
        return httpx.Response(404, json={"error": {"message": "no route"}})


@pytest.fixture
def sim():
    return GatewaySim()


@pytest.fixture
def gw(sim):
    return GatewayClient(
        "http://gw",
        client=httpx.Client(
            transport=httpx.MockTransport(sim.handler), base_url="http://gw"
        ),
    )


class TestModels:
    def test_merges_catalog_with_residency(self, gw):
        rows = gw.models()
        assert rows == [
            {"model": "llama3.2", "engine": "ollama", "resident": True},
            {"model": "big", "engine": "vllm", "resident": False},
        ]

    def test_table_marks_resident(self, gw):
        table = format_models_table(gw.models())
        lines = table.splitlines()
        assert any("llama3.2" in l and "yes" in l for l in lines)
        assert any("big" in l and "yes" not in l for l in lines)


class TestLoad:
    def test_load_success(self, gw):
        assert gw.load("llama3.2") == {"model": "llama3.2", "status": "loaded"}

    def test_load_full_card_raises_with_guidance(self, gw):
        with pytest.raises(GatewayError, match="Evict"):
            gw.load("big")


class TestUnload:
    def test_unload_resident(self, gw):
        assert gw.unload("llama3.2")["status"] == "unloaded"

    def test_unload_not_resident(self, gw):
        assert gw.unload("other")["status"] == "not-resident"


class TestChat:
    def test_chat_returns_content(self, gw, sim):
        reply = gw.chat("llama3.2", "hi there")
        assert reply == "hello from the model"
        body = json.loads(sim.requests[-1].content)
        assert body["model"] == "llama3.2"
        assert body["messages"] == [{"role": "user", "content": "hi there"}]


class TestErrors:
    def test_connection_error_wrapped(self):
        def down(request):
            raise httpx.ConnectError("refused")

        gw = GatewayClient(
            "http://gw",
            client=httpx.Client(
                transport=httpx.MockTransport(down), base_url="http://gw"
            ),
        )
        with pytest.raises(GatewayError, match="reach"):
            gw.status()
