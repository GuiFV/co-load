"""Logging setup: the rotating file that makes gateway decisions durable.

``configure_logging`` owns the root logger's handlers for the process. It
must be idempotent (reconfiguring replaces its own handlers, never stacks
them) and must leave handlers installed by anything else alone, because the
test runner and embedding applications have handlers of their own.
"""

import logging
import logging.handlers

import pytest

from coload.config import LogConfig
from coload.logs import configure_logging


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """Snapshot the root logger and put it back, whatever a test did."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    for handler in list(root.handlers):
        if handler not in saved_handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(saved_level)


def _coload_handlers() -> list[logging.Handler]:
    root = logging.getLogger()
    return [h for h in root.handlers if getattr(h, "coload_owned", False)]


def _file_handlers() -> list[logging.handlers.RotatingFileHandler]:
    return [
        h for h in _coload_handlers()
        if isinstance(h, logging.handlers.RotatingFileHandler)
    ]


class TestFileLogging:
    def test_records_reach_the_file(self, tmp_path):
        path = tmp_path / "coload.log"
        configure_logging(LogConfig(path=path))

        logging.getLogger("coload.orchestrator").info("loading 'm' via engine")
        for handler in _file_handlers():
            handler.flush()

        assert "loading 'm' via engine" in path.read_text(encoding="utf-8")

    def test_missing_parent_directories_are_created(self, tmp_path):
        path = tmp_path / "state" / "logs" / "coload.log"
        configure_logging(LogConfig(path=path))

        logging.getLogger("coload").info("hello")
        for handler in _file_handlers():
            handler.flush()

        assert path.exists()

    def test_rotation_settings_reach_the_handler(self, tmp_path):
        configure_logging(
            LogConfig(path=tmp_path / "c.log", max_bytes=1234, backup_count=7)
        )

        (handler,) = _file_handlers()
        assert handler.maxBytes == 1234
        assert handler.backupCount == 7

    def test_null_path_means_console_only(self):
        configure_logging(LogConfig(path=None))

        assert _file_handlers() == []
        assert len(_coload_handlers()) == 1  # the console handler


class TestIdempotence:
    def test_reconfiguring_replaces_rather_than_stacks(self, tmp_path):
        configure_logging(LogConfig(path=tmp_path / "a.log"))
        configure_logging(LogConfig(path=tmp_path / "b.log"))

        assert len(_coload_handlers()) == 2  # one console, one file
        (handler,) = _file_handlers()
        assert handler.baseFilename.endswith("b.log")

    def test_foreign_handlers_are_left_alone(self, tmp_path):
        sentinel = logging.NullHandler()
        logging.getLogger().addHandler(sentinel)

        configure_logging(LogConfig(path=tmp_path / "c.log"))

        assert sentinel in logging.getLogger().handlers


class TestLevel:
    def test_level_is_applied_to_the_root_logger(self, tmp_path):
        configure_logging(LogConfig(path=tmp_path / "c.log", level="WARNING"))
        assert logging.getLogger().level == logging.WARNING
