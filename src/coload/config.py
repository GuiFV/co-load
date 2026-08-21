"""Configuration models and YAML loading.

Single responsibility: parse and validate user configuration. Nothing here
touches the GPU, the network, or backend processes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

GIB = 2**30


class ModelConfig(BaseModel):
    """Per-model settings; ``est_vram_gb`` seeds the fit check."""

    est_vram_gb: float = Field(gt=0)
    max_model_len: int | None = None

    @property
    def est_vram_bytes(self) -> int:
        return int(self.est_vram_gb * GIB)


class EngineConfig(BaseModel):
    """One managed engine (backend) and the models it serves."""

    kind: Literal["ollama", "vllm"]
    base_url: str | None = None
    start: str | None = None
    stop: str | None = None  # optional; needed when `start` detaches (docker compose)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    # Ollama only: how long a loaded model stays pinned in the daemon.
    #
    # A backstop, not a policy. Coload evicts explicitly and long before this
    # fires, so in normal operation it never does; it covers the case where
    # coload is not running at all to do the evicting. Must exceed
    # idle_ttl_seconds, so coload's own sweep is never racing the daemon's.
    #
    # 0 pins indefinitely, which is the only honest reading of an
    # idle_ttl_seconds of 0 ("never evict for idleness"). It is safe because
    # adopt_resident() reclaims orphans at startup either way.
    pin_ttl_seconds: float = Field(default=3600, ge=0)
    # vLLM only: how long to wait for a started engine to answer /health.
    #
    # The default suits a small model. A large one is slower than it looks:
    # weights have to be read, the KV cache profiled and CUDA graphs captured
    # before the first request is served. Measured on a 5090, a 21.7 GiB
    # w4a16 checkpoint took ~270s cold, so the 180s default failed it, tore
    # the process down, and reported a health-check timeout, which reads like
    # a broken engine rather than an impatient one.
    health_timeout_seconds: float = Field(default=180, gt=0)

    @model_validator(mode="after")
    def _check_kind_requirements(self) -> "EngineConfig":
        if self.kind == "vllm" and not self.start:
            raise ValueError("vllm engines require a 'start' command template")
        if not self.base_url:
            raise ValueError(f"{self.kind} engines require a 'base_url'")
        return self


class LogConfig(BaseModel):
    """Where the gateway's own record of events ends up.

    The console handler is always on. ``path`` adds a rotating file, so
    decisions (loads, refusals, evictions) survive the console they scrolled
    off, or that was never attached on a box that starts the gateway at
    logon. Set it to null for console-only logging.
    """

    path: Path | None = Path(".coload/coload.log")
    level: str = "INFO"
    max_bytes: int = Field(default=10 * 2**20, gt=0)
    backup_count: int = Field(default=5, ge=0)

    @field_validator("level")
    @classmethod
    def _known_level(cls, value: str) -> str:
        name = value.upper()
        if name not in logging.getLevelNamesMapping():
            raise ValueError(f"unknown log level '{value}'")
        return name


class AlertConfig(BaseModel):
    channels: list[Literal["log", "webhook"]] = Field(default_factory=lambda: ["log"])
    webhook_url: str | None = None

    @model_validator(mode="after")
    def _webhook_needs_url(self) -> "AlertConfig":
        if "webhook" in self.channels and not self.webhook_url:
            raise ValueError("webhook alert channel requires 'webhook_url'")
        return self


class Config(BaseModel):
    """Top-level coload configuration.

    ``buffer_pct`` is the user-settable share of total VRAM kept free as
    headroom (fragmentation, CUDA context growth, fluctuations). Defaults to
    10% and must be in [0, 1).
    """

    gpu: int = 0
    buffer_pct: float = Field(default=0.10, ge=0.0, lt=1.0)
    # Opt-in: lock a fixed slice of the card for coload's models. When set,
    # the admission budget never falls below this figure minus what coload's
    # own configured models are estimated to hold, however much out-of-band
    # processes (a game, an editor, a browser) have taken. The OS makes the
    # guarantee physical: when the engine allocates, resident desktop memory
    # is paged out to system RAM, so out-of-band users lose performance,
    # never correctness. None (the default) keeps the purely measured budget.
    reserve_gb: float | None = Field(default=None, gt=0)
    # Opt-in: when an admitted model does not physically fit the live
    # measurement, briefly claim the bytes it needs so the OS pages
    # out-of-band memory to system RAM, release them, and then start the
    # engine against a card that measures clean. Without it (the default), a
    # physical shortfall refuses the load cleanly, with an alert, instead of
    # starting an engine that would fail its own sizing arithmetic. Only the
    # OS's own paging is used: out-of-band processes lose performance, never
    # correctness, and nothing is ever evicted.
    make_room: bool = False
    idle_ttl_seconds: float = Field(default=900, ge=0)
    # Opt-in: when a requested model doesn't fit, evict models COLOAD ITSELF
    # loaded (least-recently-used first) to make room, before refusing. Never
    # touches out-of-band GPU users — those are only ever alerted about.
    # Needed for workloads that legitimately alternate between models too big
    # to co-reside (e.g. a pipeline phase-switching 12b -> 31b on one card).
    auto_evict_idle: bool = False
    watchdog_interval_s: float = Field(default=10, gt=0)
    host: str = "127.0.0.1"
    port: int = 8800
    estimates_path: Path = Path(".coload/learned_estimates.json")
    log: LogConfig = Field(default_factory=LogConfig)
    alert: AlertConfig = Field(default_factory=AlertConfig)
    engines: dict[str, EngineConfig]

    @model_validator(mode="after")
    def _pin_outlives_the_idle_sweep(self) -> "Config":
        """coload must be the one that evicts; the daemon is only the backstop.

        With a pin shorter than the idle TTL, Ollama drops the model out from
        under a coload that still believes it is resident, and the next
        request pays a reload it never accounted for.
        """
        for name, engine in self.engines.items():
            if engine.kind != "ollama":
                continue
            if engine.pin_ttl_seconds == 0:  # indefinite, by request
                continue
            if engine.pin_ttl_seconds <= self.idle_ttl_seconds:
                raise ValueError(
                    f"engine '{name}': pin_ttl_seconds "
                    f"({engine.pin_ttl_seconds}) must exceed idle_ttl_seconds "
                    f"({self.idle_ttl_seconds}), so coload evicts before the "
                    f"daemon does"
                )
        return self

    @model_validator(mode="after")
    def _no_duplicate_models(self) -> "Config":
        seen: dict[str, str] = {}
        for engine_name, engine in self.engines.items():
            for model in engine.models:
                if model in seen:
                    raise ValueError(
                        f"duplicate model '{model}' in engines "
                        f"'{seen[model]}' and '{engine_name}'"
                    )
                seen[model] = engine_name
        return self

    @property
    def reserve_bytes(self) -> int | None:
        return None if self.reserve_gb is None else int(self.reserve_gb * GIB)

    def engine_for_model(self, model: str) -> str:
        """Name of the engine serving ``model``; raises KeyError if unknown."""
        for engine_name, engine in self.engines.items():
            if model in engine.models:
                return engine_name
        raise KeyError(f"no engine configured for model '{model}'")

    def model_config_for(self, model: str) -> ModelConfig:
        return self.engines[self.engine_for_model(model)].models[model]


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    text = Path(path).read_text(encoding="utf-8")
    return Config.model_validate(yaml.safe_load(text) or {})


def resolve_config_path(explicit: str | None = None, cwd: Path | None = None) -> Path:
    """Pick the config file for ``coload serve``.

    Machine-specific setups live in ``config.local.yaml`` (gitignored) and
    take precedence over the tracked ``config.yaml`` defaults, so personal
    changes never leak into the public repo. An explicit ``--config`` always
    wins.
    """
    if explicit:
        return Path(explicit)
    base = cwd or Path.cwd()
    local = base / "config.local.yaml"
    if local.exists():
        return local
    return base / "config.yaml"
