"""Logging setup for the gateway process.

Single responsibility: turn a ``LogConfig`` into handlers on the root
logger. Every other module just calls ``logging.getLogger`` and never knows
where its records go.

The rotating file exists because the gateway's decisions matter most when
nobody is watching: a load refused at five in the morning is diagnosed hours
later, and a console that scrolled away, or was never attached, leaves
nothing to read. Rotation bounds the disk the record can take; the console
handler keeps interactive runs readable.
"""

from __future__ import annotations

import logging
import logging.handlers

from .config import LogConfig

_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"

# Attribute stamped on every handler this module installs, so reconfiguring
# replaces exactly those and never touches handlers owned by anything else
# (a test runner, an embedding application).
_OWNED = "coload_owned"


def configure_logging(config: LogConfig) -> None:
    """Install a console handler, plus a rotating file handler when
    ``config.path`` is set, on the root logger.

    Idempotent: handlers a previous call installed are replaced, never
    stacked, so reconfiguring cannot double every line.
    """
    root = logging.getLogger()
    root.setLevel(config.level)

    for handler in list(root.handlers):
        if getattr(handler, _OWNED, False):
            root.removeHandler(handler)
            handler.close()

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if config.path is not None:
        config.path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                config.path,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding="utf-8",
            )
        )

    formatter = logging.Formatter(_FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _OWNED, True)
        root.addHandler(handler)
