"""Backend factory keyed by engine ``kind`` (OCP: register, don't modify)."""

from __future__ import annotations

from typing import Callable

from ..config import EngineConfig
from .base import Backend
from .ollama import OllamaBackend
from .vllm import VllmBackend


def _build_ollama(name: str, cfg: EngineConfig) -> Backend:
    assert cfg.base_url is not None
    return OllamaBackend(
        name, base_url=cfg.base_url, pin_ttl_s=cfg.pin_ttl_seconds
    )


def _build_vllm(name: str, cfg: EngineConfig) -> Backend:
    assert cfg.base_url is not None and cfg.start is not None
    return VllmBackend(
        name,
        base_url=cfg.base_url,
        start_template=cfg.start,
        stop_template=cfg.stop,
        health_timeout_s=cfg.health_timeout_seconds,
    )


BACKEND_FACTORIES: dict[str, Callable[[str, EngineConfig], Backend]] = {
    "ollama": _build_ollama,
    "vllm": _build_vllm,
}


def build_backend(name: str, cfg: EngineConfig) -> Backend:
    try:
        factory = BACKEND_FACTORIES[cfg.kind]
    except KeyError:
        raise ValueError(f"unknown engine kind '{cfg.kind}'") from None
    return factory(name, cfg)
