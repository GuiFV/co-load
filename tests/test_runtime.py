"""Composition root: config wires probe/backends/orchestrator/watchdog/gateway."""

import httpx

from coload.backends.ollama import OllamaBackend
from coload.backends.vllm import VllmBackend
from coload.config import load_config
from coload.runtime import Runtime, build_runtime
from coload.vram import VramSnapshot
from tests.conftest import GIB, FakeProbe

CONFIG_YAML = """
buffer_pct: 0.2
estimates_path: "{estimates_path}"
engines:
  oll:
    kind: ollama
    base_url: "http://localhost:11434"
    models:
      "small": {{ est_vram_gb: 8 }}
  vl:
    kind: vllm
    base_url: "http://localhost:8000"
    start: "vllm serve {{model}}"
    models:
      "big": {{ est_vram_gb: 20 }}
"""


def write_config(tmp_path):
    est = (tmp_path / "learned.json").as_posix()
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_YAML.format(estimates_path=est), encoding="utf-8")
    return path


class TestBuildRuntime:
    def test_builds_backends_by_kind(self, tmp_path):
        rt = build_runtime(load_config(write_config(tmp_path)), probe=FakeProbe(
            VramSnapshot(total=24 * GIB, used=0)
        ))
        assert isinstance(rt, Runtime)
        assert isinstance(rt.backends["oll"], OllamaBackend)
        assert isinstance(rt.backends["vl"], VllmBackend)

    def test_estimates_seeded_from_config(self, tmp_path):
        rt = build_runtime(load_config(write_config(tmp_path)), probe=FakeProbe(
            VramSnapshot(total=24 * GIB, used=0)
        ))
        assert rt.estimates.estimate("small") == 8 * GIB
        assert rt.estimates.estimate("big") == 20 * GIB

    async def test_status_endpoint_reflects_user_buffer(self, tmp_path):
        """End-to-end wiring: the configured 20% buffer shows up in /status."""
        rt = build_runtime(load_config(write_config(tmp_path)), probe=FakeProbe(
            VramSnapshot(total=24 * GIB, used=4 * GIB)
        ))
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=rt.app), base_url="http://gw"
        )
        status = (await client.get("/status")).json()
        assert status["buffer_pct"] == 0.2
        # budget = 20G free - 20% of 24G = 15.2G
        assert status["vram"]["budget_gb"] == 15.2
