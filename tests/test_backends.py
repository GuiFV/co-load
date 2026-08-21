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
        self.embed_only: set[str] = set()  # embedding models 400 on /api/generate

    def _apply_keep_alive(self, body):
        if body.get("keep_alive") == 0:
            if body["model"] in self.resident:
                self.resident.remove(body["model"])
        else:
            self.resident.append(body["model"])

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/ps":
            return httpx.Response(
                200, json={"models": [{"name": m} for m in self.resident]}
            )
        if path == "/api/generate":
            body = json.loads(request.content)
            if body["model"] in self.embed_only:  # real Ollama behavior
                return httpx.Response(
                    400, json={"error": f'"{body["model"]}" does not support generate'}
                )
            self._apply_keep_alive(body)
            return httpx.Response(200, json={"done": True})
        if path == "/api/embed":
            body = json.loads(request.content)
            self._apply_keep_alive(body)
            return httpx.Response(200, json={"embeddings": []})
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

    async def test_load_pins_for_a_finite_ttl(self, ollama, ollama_sim):
        """Never keep_alive: -1.

        Coload owns residency and evicts explicitly, so the pin only has to
        outlive its own idle sweep. Pinning indefinitely made the daemon the
        sole owner of the model's fate: both eviction paths iterate the
        in-memory `_last_used` map, so a model pinned by a previous coload
        process was invisible to them and stayed resident until Ollama itself
        restarted, holding VRAM no one could reclaim. A finite TTL is the
        dead-man's switch for exactly that.
        """
        await ollama.load("gemma:12b", budget_bytes=10 * GIB, total_bytes=24 * GIB)
        gen = [r for r in ollama_sim.requests if r.url.path == "/api/generate"]
        body = json.loads(gen[0].content)
        assert body["model"] == "gemma:12b"
        assert body["keep_alive"] == 3600
        assert body["keep_alive"] > 0
        assert await ollama.is_ready("gemma:12b")

    async def test_pin_ttl_zero_sends_the_wire_value_for_indefinite(self, ollama_sim):
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(ollama_sim.handler),
            base_url="http://localhost:11434",
        )
        backend = OllamaBackend(
            "ollama", base_url="http://localhost:11434", client=client, pin_ttl_s=0
        )
        await backend.load("gemma:12b", budget_bytes=GIB, total_bytes=24 * GIB)
        assert json.loads(ollama_sim.requests[-1].content)["keep_alive"] == -1

    async def test_pin_ttl_is_configurable(self, ollama_sim):
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(ollama_sim.handler),
            base_url="http://localhost:11434",
        )
        backend = OllamaBackend(
            "ollama", base_url="http://localhost:11434", client=client, pin_ttl_s=120
        )
        await backend.load("gemma:12b", budget_bytes=GIB, total_bytes=24 * GIB)
        body = json.loads(ollama_sim.requests[-1].content)
        assert body["keep_alive"] == 120

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

    async def test_load_embedding_model_falls_back_to_embed_api(self, ollama, ollama_sim):
        # Embedding models 400 on /api/generate; the adapter must pin them
        # via /api/embed instead.
        ollama_sim.embed_only.add("nomic-embed-text")
        await ollama.load("nomic-embed-text", budget_bytes=GIB, total_bytes=24 * GIB)
        emb = [r for r in ollama_sim.requests if r.url.path == "/api/embed"]
        body = json.loads(emb[0].content)
        assert body["keep_alive"] == 3600
        assert await ollama.is_ready("nomic-embed-text")

    async def test_unload_embedding_model_falls_back_to_embed_api(self, ollama, ollama_sim):
        ollama_sim.embed_only.add("nomic-embed-text")
        await ollama.load("nomic-embed-text", budget_bytes=GIB, total_bytes=24 * GIB)
        await ollama.unload("nomic-embed-text")
        assert ollama_sim.resident == []

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
    async def test_detached_starter_exiting_is_not_engine_death(self):
        """`docker compose up -d` returns as soon as the container exists,
        long before vLLM has read its weights. Treating that as the engine
        dying failed every detached start whose model was not healthy inside
        the first poll, which is every large model."""
        launcher = FakeLauncher()
        sim = HealthSim(fail_first=3)
        backend = make_vllm(
            sim, launcher,
            start_template="docker compose up -d vllm",
            stop_template="docker compose stop vllm",
        )
        launcher_launch = launcher.launch

        def launch_then_exit(command, env=None):
            proc = launcher_launch(command, env)
            proc.terminate()          # the starter returns immediately
            return proc

        launcher.launch = launch_then_exit
        await backend.load("big-model", budget_bytes=8, total_bytes=10)
        assert await backend.resident_models() == ["big-model"]

    async def test_owned_process_exiting_still_fails_fast(self):
        """With no stop command coload owns the process, so its exit is the
        engine dying and must not be waited out to the full timeout."""
        launcher = FakeLauncher()
        backend = make_vllm(HealthSim(fail_first=99), launcher, stop_template=None)
        launcher_launch = launcher.launch

        def launch_then_exit(command, env=None):
            proc = launcher_launch(command, env)
            proc.terminate()
            return proc

        launcher.launch = launch_then_exit
        with pytest.raises(BackendError, match="exited during startup"):
            await backend.load("m", budget_bytes=8, total_bytes=10)

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


class EngineSim:
    """A running detached engine: /health answers, /v1/models says what it serves."""

    def __init__(self, model="big-model"):
        self.model = model

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": self.model, "object": "model"}]},
            )
        return httpx.Response(404)


class DownSim:
    """No engine behind the port at all."""

    def __init__(self):
        self.calls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        raise httpx.ConnectError("connection refused", request=request)


class TestVllmRediscovery:
    """The served model lives in gateway memory, so a restart forgets it while
    a detached engine keeps running. The engine itself is the only witness
    left; a fresh backend asks it rather than believing the card is empty.
    Getting this wrong loads the model again over its own resident copy and
    the accounting drifts from the card."""

    _DETACHED = dict(
        start_template="docker compose up -d vllm",
        stop_template="docker compose stop vllm",
    )

    async def test_a_fresh_backend_rediscovers_a_detached_engine(self):
        backend = make_vllm(EngineSim("big-model"), **self._DETACHED)
        assert await backend.resident_models() == ["big-model"]

    async def test_rediscovery_makes_is_ready_true(self):
        backend = make_vllm(EngineSim("big-model"), **self._DETACHED)
        assert await backend.is_ready("big-model")

    async def test_no_engine_means_no_residents_and_no_error(self):
        backend = make_vllm(DownSim(), **self._DETACHED)
        assert await backend.resident_models() == []
        assert not await backend.is_ready("big-model")

    async def test_rediscovery_is_attempted_once_not_per_call(self):
        """Boot-time reconciliation, not a poll: a down engine must not cost
        a connection attempt on every status read or admission pass."""
        sim = DownSim()
        backend = make_vllm(sim, **self._DETACHED)
        await backend.resident_models()
        await backend.resident_models()
        assert not await backend.is_ready("big-model")
        assert sim.calls == 1

    async def test_an_owned_process_is_never_rediscovered(self):
        """With no stop command coload owns the process, and a fresh backend
        owns none: whatever answers the port belongs to somebody else."""
        backend = make_vllm(EngineSim("big-model"), stop_template=None)
        assert await backend.resident_models() == []

    async def test_a_rediscovered_model_is_stopped_before_a_different_load(self):
        launcher = FakeLauncher()
        backend = make_vllm(EngineSim("old-model"), launcher, **self._DETACHED)
        await backend.load("new-model", budget_bytes=GIB, total_bytes=2 * GIB)
        assert launcher.commands[0] == "docker compose stop vllm"
        assert launcher.commands[-1] == "docker compose up -d vllm"

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

    def test_vllm_health_timeout_wired_through(self):
        """A big checkpoint is slower to serve than the 180s default allows:
        weights, KV cache profiling and CUDA graph capture all precede the
        first health response. Measured at ~270s for a 21.7 GiB w4a16 model,
        which the default failed, tore down, and reported as a timeout."""
        cfg = EngineConfig.model_validate(
            {
                "kind": "vllm",
                "base_url": "http://x:8000",
                "start": "docker compose up -d vllm",
                "health_timeout_seconds": 900,
                "models": {},
            }
        )
        backend = build_backend("v", cfg)
        assert backend._health_timeout_s == 900

    def test_vllm_health_timeout_defaults_when_unset(self):
        cfg = EngineConfig.model_validate(
            {
                "kind": "vllm",
                "base_url": "http://x:8000",
                "start": "vllm serve {model}",
                "models": {},
            }
        )
        assert build_backend("v", cfg)._health_timeout_s == 180

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
