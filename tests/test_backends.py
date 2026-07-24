"""Backend adapters: Ollama (HTTP keep_alive) and vLLM (process lifecycle)."""

import json

import httpx
import pytest

from coload.backends.base import BackendError
from coload.backends.ollama import OllamaBackend
from coload.backends.registry import build_backend
from coload.backends.vllm import VllmBackend
from coload.config import EngineConfig

GIB = 2**30


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #


class OllamaSim:
    """Simulates the Ollama HTTP API."""

    def __init__(self):
        self.resident = []
        self.requests = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/ps":
            return httpx.Response(
                200, json={"models": [{"name": m} for m in self.resident]}
            )
        if path == "/api/generate":
            body = json.loads(request.content)
            if body.get("keep_alive") == 0:
                if body["model"] in self.resident:
                    self.resident.remove(body["model"])
            else:
                self.resident.append(body["model"])
            return httpx.Response(200, json={"done": True})
        return httpx.Response(404)


@pytest.fixture
def ollama_sim():
    return OllamaSim()


@pytest.fixture
def ollama(ollama_sim):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(ollama_sim.handler),
        base_url="http://localhost:11434",
    )
    return OllamaBackend("ollama", base_url="http://localhost:11434", client=client)


class TestOllamaBackend:
    async def test_not_ready_when_model_absent(self, ollama):
        assert not await ollama.is_ready("gemma:12b")

    async def test_load_requests_indefinite_keep_alive(self, ollama, ollama_sim):
        await ollama.load("gemma:12b", budget_bytes=10 * GIB, total_bytes=24 * GIB)
        gen = [r for r in ollama_sim.requests if r.url.path == "/api/generate"]
        body = json.loads(gen[0].content)
        assert body["model"] == "gemma:12b"
        assert body["keep_alive"] == -1
        assert await ollama.is_ready("gemma:12b")

    async def test_unload_sends_zero_keep_alive(self, ollama, ollama_sim):
        await ollama.load("gemma:12b", budget_bytes=10 * GIB, total_bytes=24 * GIB)
        await ollama.unload("gemma:12b")
        assert ollama_sim.resident == []

    async def test_resident_models_reflects_ps(self, ollama, ollama_sim):
        ollama_sim.resident = ["a", "b"]
        assert await ollama.resident_models() == ["a", "b"]

    async def test_load_failure_raises_backend_error(self):
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(500)),
            base_url="http://localhost:11434",
        )
        backend = OllamaBackend("ollama", "http://localhost:11434", client=client)
        with pytest.raises(BackendError):
            await backend.load("m", budget_bytes=GIB, total_bytes=2 * GIB)

    async def test_proxy_target_is_base_url(self, ollama):
        assert ollama.proxy_url("gemma:12b") == "http://localhost:11434"


# --------------------------------------------------------------------------- #
# vLLM
# --------------------------------------------------------------------------- #


class FakeProcess:
    def __init__(self):
        self.terminated = False

    def is_running(self):
        return not self.terminated

    def terminate(self):
        self.terminated = True


class FakeLauncher:
    def __init__(self):
        self.commands = []
        self.envs = []
        self.processes = []

    def launch(self, command: str, env=None):
        self.commands.append(command)
        self.envs.append(env or {})
        proc = FakeProcess()
        self.processes.append(proc)
        return proc


class HealthSim:
    """/health returns 503 for the first `fail_first` calls, then 200."""

    def __init__(self, fail_first=0):
        self.fail_first = fail_first
        self.calls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls <= self.fail_first:
            return httpx.Response(503)
        return httpx.Response(200)


def make_vllm(sim: HealthSim, launcher=None, **over):
    client = httpx.AsyncClient(transport=httpx.MockTransport(sim.handler))
    kwargs = dict(
        base_url="http://localhost:8000",
        start_template="vllm serve {model} --gpu-memory-utilization {budget_frac}",
        launcher=launcher or FakeLauncher(),
        client=client,
        health_timeout_s=1.0,
        health_poll_interval_s=0.001,
        stop_timeout_s=0.01,
    )
    kwargs.update(over)
    return VllmBackend("vllm", **kwargs)


class TestVllmBackend:
    async def test_load_launches_with_budget_fraction(self):
        launcher = FakeLauncher()
        backend = make_vllm(HealthSim(), launcher)
        await backend.load("gemma:31b", budget_bytes=18 * GIB, total_bytes=24 * GIB)
        assert launcher.commands == [
            "vllm serve gemma:31b --gpu-memory-utilization 0.75"
        ]

    async def test_ready_after_healthy(self):
        backend = make_vllm(HealthSim(fail_first=2))
        await backend.load("m", budget_bytes=GIB, total_bytes=2 * GIB)
        assert await backend.is_ready("m")

    async def test_not_ready_for_other_model(self):
        backend = make_vllm(HealthSim())
        await backend.load("m", budget_bytes=GIB, total_bytes=2 * GIB)
        assert not await backend.is_ready("other")

    async def test_health_timeout_raises_and_terminates(self):
        launcher = FakeLauncher()
        backend = make_vllm(
            HealthSim(fail_first=10_000), launcher, health_timeout_s=0.01
        )
        with pytest.raises(BackendError):
            await backend.load("m", budget_bytes=GIB, total_bytes=2 * GIB)
        assert launcher.processes[0].terminated

    async def test_unload_terminates_process(self):
        launcher = FakeLauncher()
        backend = make_vllm(HealthSim(), launcher)
        await backend.load("m", budget_bytes=GIB, total_bytes=2 * GIB)
        await backend.unload("m")
        assert launcher.processes[0].terminated
        assert not await backend.is_ready("m")

    async def test_resident_models(self):
        backend = make_vllm(HealthSim())
        assert await backend.resident_models() == []
        await backend.load("m", budget_bytes=GIB, total_bytes=2 * GIB)
        assert await backend.resident_models() == ["m"]

    async def test_second_load_replaces_first_process(self):
        launcher = FakeLauncher()
        backend = make_vllm(HealthSim(), launcher)
        await backend.load("a", budget_bytes=GIB, total_bytes=2 * GIB)
        await backend.load("b", budget_bytes=GIB, total_bytes=2 * GIB)
        assert launcher.processes[0].terminated
        assert await backend.resident_models() == ["b"]


class TestVllmStopCommand:
    """A `stop` template makes detached starts (docker compose) first-class:
    coload runs the stop command instead of terminating the launched process."""

    async def test_unload_runs_stop_command(self):
        launcher = FakeLauncher()
        backend = make_vllm(
            HealthSim(),
            launcher,
            start_template="docker compose up -d vllm",
            stop_template="docker compose stop vllm-{model}",
        )
        await backend.load("m", budget_bytes=GIB, total_bytes=2 * GIB)
        await backend.unload("m")
        assert "docker compose stop vllm-m" in launcher.commands
        # the (already-exited) start process is not the stop mechanism
        assert await backend.resident_models() == []

    async def test_failed_health_also_runs_stop_command(self):
        launcher = FakeLauncher()
        backend = make_vllm(
            HealthSim(fail_first=10_000),
            launcher,
            stop_template="docker compose stop vllm",
            health_timeout_s=0.01,
        )
        with pytest.raises(BackendError):
            await backend.load("m", budget_bytes=GIB, total_bytes=2 * GIB)
        assert "docker compose stop vllm" in launcher.commands

    async def test_start_receives_model_and_budget_env_vars(self):
        """docker compose can't take CLI placeholders, so the live budget and
        model also travel as env vars (COLOAD_MODEL, COLOAD_BUDGET_FRAC) that
        compose files interpolate natively."""
        launcher = FakeLauncher()
        backend = make_vllm(
            HealthSim(),
            launcher,
            start_template="docker compose up -d vllm",
        )
        await backend.load("qwen", budget_bytes=18 * GIB, total_bytes=24 * GIB)
        env = launcher.envs[0]
        assert env["COLOAD_MODEL"] == "qwen"
        assert env["COLOAD_BUDGET_FRAC"] == "0.75"

    async def test_without_stop_command_terminates_process(self):
        launcher = FakeLauncher()
        backend = make_vllm(HealthSim(), launcher)
        await backend.load("m", budget_bytes=GIB, total_bytes=2 * GIB)
        await backend.unload("m")
        assert launcher.processes[0].terminated
        assert len(launcher.commands) == 1  # no stop command launched


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class TestRegistry:
    def test_builds_ollama(self):
        cfg = EngineConfig.model_validate(
            {"kind": "ollama", "base_url": "http://x:11434", "models": {}}
        )
        backend = build_backend("o", cfg)
        assert isinstance(backend, OllamaBackend)

    def test_builds_vllm(self):
        cfg = EngineConfig.model_validate(
            {
                "kind": "vllm",
                "base_url": "http://x:8000",
                "start": "vllm serve {model}",
                "models": {},
            }
        )
        backend = build_backend("v", cfg)
        assert isinstance(backend, VllmBackend)

    def test_vllm_stop_command_wired_through(self):
        cfg = EngineConfig.model_validate(
            {
                "kind": "vllm",
                "base_url": "http://x:8000",
                "start": "docker compose up -d vllm",
                "stop": "docker compose stop vllm",
                "models": {},
            }
        )
        backend = build_backend("v", cfg)
        assert backend._stop_template == "docker compose stop vllm"
