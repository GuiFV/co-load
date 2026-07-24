"""Alerting: log + webhook sinks, composite fan-out, rate limiting."""

import json
import logging

import httpx
import pytest

from coload.alerts import (
    Alert,
    LogAlerter,
    MultiAlerter,
    RateLimitedAlerter,
    WebhookAlerter,
    build_alerter,
)
from coload.config import AlertConfig


def make_alert(**over):
    defaults = dict(
        severity="warning",
        title="VRAM full",
        message="model needs 20.0 GiB, only 4.0 GiB free",
        context={"model": "m"},
    )
    defaults.update(over)
    return Alert(**defaults)


class TestLogAlerter:
    async def test_logs_alert(self, caplog):
        with caplog.at_level(logging.WARNING, logger="coload.alerts"):
            await LogAlerter().send(make_alert())
        assert "VRAM full" in caplog.text
        assert "20.0 GiB" in caplog.text


class TestWebhookAlerter:
    async def test_posts_alert_json(self):
        received = {}

        def handler(request: httpx.Request) -> httpx.Response:
            received["url"] = str(request.url)
            received["body"] = json.loads(request.content)
            return httpx.Response(200)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        alerter = WebhookAlerter("http://alerts.local/hook", client=client)
        await alerter.send(make_alert())

        assert received["url"] == "http://alerts.local/hook"
        assert received["body"]["title"] == "VRAM full"
        assert received["body"]["severity"] == "warning"
        assert received["body"]["context"] == {"model": "m"}

    async def test_network_error_does_not_raise(self):
        def handler(request):
            raise httpx.ConnectError("down")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        alerter = WebhookAlerter("http://alerts.local/hook", client=client)
        await alerter.send(make_alert())  # must not propagate


class RecordingAlerter:
    def __init__(self, fail=False):
        self.alerts = []
        self.fail = fail

    async def send(self, alert):
        if self.fail:
            raise RuntimeError("sink broken")
        self.alerts.append(alert)


class TestMultiAlerter:
    async def test_fans_out_to_all_sinks(self):
        a, b = RecordingAlerter(), RecordingAlerter()
        await MultiAlerter([a, b]).send(make_alert())
        assert len(a.alerts) == len(b.alerts) == 1

    async def test_one_failing_sink_does_not_block_others(self):
        bad, good = RecordingAlerter(fail=True), RecordingAlerter()
        await MultiAlerter([bad, good]).send(make_alert())
        assert len(good.alerts) == 1


class TestRateLimitedAlerter:
    async def test_suppresses_duplicate_key_within_window(self):
        inner = RecordingAlerter()
        now = [0.0]
        limited = RateLimitedAlerter(inner, window_s=60, clock=lambda: now[0])

        await limited.send(make_alert(dedup_key="k"))
        now[0] = 30.0
        await limited.send(make_alert(dedup_key="k"))
        assert len(inner.alerts) == 1

    async def test_allows_after_window(self):
        inner = RecordingAlerter()
        now = [0.0]
        limited = RateLimitedAlerter(inner, window_s=60, clock=lambda: now[0])

        await limited.send(make_alert(dedup_key="k"))
        now[0] = 61.0
        await limited.send(make_alert(dedup_key="k"))
        assert len(inner.alerts) == 2

    async def test_distinct_keys_independent(self):
        inner = RecordingAlerter()
        limited = RateLimitedAlerter(inner, window_s=60, clock=lambda: 0.0)

        await limited.send(make_alert(dedup_key="a"))
        await limited.send(make_alert(dedup_key="b"))
        assert len(inner.alerts) == 2

    async def test_no_dedup_key_never_suppressed(self):
        inner = RecordingAlerter()
        limited = RateLimitedAlerter(inner, window_s=60, clock=lambda: 0.0)

        await limited.send(make_alert())
        await limited.send(make_alert())
        assert len(inner.alerts) == 2


class TestBuildAlerter:
    def test_log_only(self):
        alerter = build_alerter(AlertConfig(channels=["log"]))
        assert isinstance(alerter, MultiAlerter)

    def test_webhook_included_when_configured(self):
        alerter = build_alerter(
            AlertConfig(channels=["log", "webhook"], webhook_url="http://x/hook")
        )
        kinds = {type(s).__name__ for s in alerter.sinks}
        assert kinds == {"LogAlerter", "WebhookAlerter"}
