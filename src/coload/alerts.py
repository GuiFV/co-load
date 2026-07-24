"""Alerting sinks.

``Alerter`` is a tiny protocol; sinks (log, webhook) are composed with
``MultiAlerter`` and decorated with ``RateLimitedAlerter``; new channels are
added without touching callers (OCP). Alert delivery must never crash the
orchestrator, so sinks swallow their own transport errors.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Protocol

import httpx

from .config import AlertConfig

logger = logging.getLogger("coload.alerts")

Severity = Literal["info", "warning", "critical"]


@dataclass(frozen=True)
class Alert:
    severity: Severity
    title: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    dedup_key: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "context": self.context,
        }


class Alerter(Protocol):
    async def send(self, alert: Alert) -> None: ...


class LogAlerter:
    _LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "critical": logging.CRITICAL}

    async def send(self, alert: Alert) -> None:
        logger.log(
            self._LEVELS.get(alert.severity, logging.WARNING),
            "%s: %s %s",
            alert.title,
            alert.message,
            alert.context or "",
        )


class WebhookAlerter:
    def __init__(self, url: str, client: httpx.AsyncClient | None = None):
        self._url = url
        self._client = client or httpx.AsyncClient(timeout=5.0)

    async def send(self, alert: Alert) -> None:
        try:
            await self._client.post(self._url, json=alert.to_payload())
        except httpx.HTTPError as exc:
            logger.error("webhook alert delivery failed: %s", exc)


class MultiAlerter:
    def __init__(self, sinks: Iterable[Alerter]):
        self.sinks = list(sinks)

    async def send(self, alert: Alert) -> None:
        for sink in self.sinks:
            try:
                await sink.send(alert)
            except Exception:
                logger.exception("alert sink %s failed", type(sink).__name__)


class RateLimitedAlerter:
    """Decorator suppressing repeats of the same ``dedup_key`` within a window."""

    def __init__(
        self,
        inner: Alerter,
        window_s: float = 300,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._inner = inner
        self._window_s = window_s
        self._clock = clock
        self._last_sent: dict[str, float] = {}

    async def send(self, alert: Alert) -> None:
        if alert.dedup_key is not None:
            now = self._clock()
            last = self._last_sent.get(alert.dedup_key)
            if last is not None and now - last < self._window_s:
                return
            self._last_sent[alert.dedup_key] = now
        await self._inner.send(alert)


def build_alerter(config: AlertConfig) -> MultiAlerter:
    """Compose sinks from user config."""
    sinks: list[Alerter] = []
    if "log" in config.channels:
        sinks.append(LogAlerter())
    if "webhook" in config.channels and config.webhook_url:
        sinks.append(WebhookAlerter(config.webhook_url))
    return MultiAlerter(sinks)
