"""Background watchdog for out-of-band GPU usage.

Loads made *through* coload are safe by construction; this catches everything
else (a user loading a model directly into Ollama, a render job, ...). It is
observe-only: it fires an alert when total usage crosses the safe line
(``total * (1 - buffer_pct)``), it cannot prevent the overuse.

Edge-triggered so a sustained excursion produces one alert, not one per poll.
"""

from __future__ import annotations

import asyncio
import logging

from .alerts import Alert, Alerter
from .vram import GIB, VramProbe

logger = logging.getLogger("coload.watchdog")


class Watchdog:
    def __init__(self, probe: VramProbe, alerter: Alerter, buffer_pct: float):
        self._probe = probe
        self._alerter = alerter
        self._buffer_pct = buffer_pct
        self._over = False

    async def check_once(self) -> bool:
        """Poll once; returns True when usage is over the safe line."""
        try:
            snap = self._probe.read()
        except Exception:
            logger.exception("watchdog probe failed")
            return False

        safe_line = int(snap.total * (1 - self._buffer_pct))
        over = snap.used > safe_line

        if over and not self._over:  # rising edge
            await self._alerter.send(
                Alert(
                    severity="critical",
                    title="coload: GPU over safe line",
                    message=(
                        f"VRAM usage {snap.used / GIB:.1f} GiB exceeds the safe "
                        f"line {safe_line / GIB:.1f} GiB, likely an out-of-band "
                        f"load coload did not mediate."
                    ),
                    context={
                        "used_bytes": snap.used,
                        "total_bytes": snap.total,
                        "safe_line_bytes": safe_line,
                        "buffer_pct": self._buffer_pct,
                    },
                    dedup_key="watchdog:over-safe-line",
                )
            )
        self._over = over
        return over

    async def run(self, interval_s: float) -> None:  # pragma: no cover - loop shell
        while True:
            await self.check_once()
            await asyncio.sleep(interval_s)
