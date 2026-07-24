"""Configuration models and YAML loading.

Single responsibility: parse and validate user configuration. Nothing here
touches the GPU, the network, or backend processes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

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

    @model_validator(mode="after")
    def _check_kind_requirements(self) -> "EngineConfig":
        if self.kind == "vllm" and not self.start:
            raise ValueError("vllm engines require a 'start' command template")
        if not self.base_url:
            raise ValueError(f"{self.kind} engines require a 'base_url'")
        return self


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
    alert: AlertConfig = Field(default_factory=AlertConfig)
    engines: dict[str, EngineConfig]

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
