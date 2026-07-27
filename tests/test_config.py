"""Config: YAML loading, validation, and defaults.

The buffer percentage is a user-facing setting (``buffer_pct``) that defaults
to 10% of total VRAM and must stay within [0, 1).
"""

import pytest
from pydantic import ValidationError

from coload.config import Config, EngineConfig, ModelConfig, load_config

MINIMAL_YAML = """
engines:
  ollama:
    kind: ollama
    base_url: "http://localhost:11434"
    models:
      "gemma:12b": { est_vram_gb: 8 }
"""

FULL_YAML = """
gpu: 1
buffer_pct: 0.15
idle_ttl_seconds: 600
watchdog_interval_s: 5
host: "0.0.0.0"
port: 9000
alert:
  channels: [log, webhook]
  webhook_url: "http://localhost:9999/alert"
engines:
  vllm:
    kind: vllm
    base_url: "http://localhost:8000"
    start: "vllm serve {model} --gpu-memory-utilization {budget_frac}"
    models:
      "gemma:31b-awq": { est_vram_gb: 20, max_model_len: 16384 }
  ollama:
    kind: ollama
    base_url: "http://localhost:11434"
    models:
      "gemma:12b": { est_vram_gb: 8 }
      "nomic-embed-text": { est_vram_gb: 1 }
"""


def _write(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return p


class TestDefaults:
    def test_buffer_pct_defaults_to_ten_percent(self, tmp_path):
        cfg = load_config(_write(tmp_path, MINIMAL_YAML))
        assert cfg.buffer_pct == 0.10

    def test_other_defaults(self, tmp_path):
        cfg = load_config(_write(tmp_path, MINIMAL_YAML))
        assert cfg.gpu == 0
        assert cfg.idle_ttl_seconds == 900
        assert cfg.watchdog_interval_s == 10
        assert cfg.alert.channels == ["log"]
        assert cfg.port == 8800


class TestBufferPctSetting:
    """buffer_pct is user-settable (the user asked for this explicitly)."""

    def test_user_can_override_buffer_pct(self, tmp_path):
        cfg = load_config(_write(tmp_path, FULL_YAML))
        assert cfg.buffer_pct == 0.15

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_buffer_pct_out_of_range_rejected(self, bad):
        with pytest.raises(ValidationError):
            Config.model_validate(
                {
                    "buffer_pct": bad,
                    "engines": {
                        "ollama": {
                            "kind": "ollama",
                            "base_url": "http://localhost:11434",
                            "models": {"m": {"est_vram_gb": 1}},
                        }
                    },
                }
            )

    def test_buffer_pct_zero_is_allowed(self):
        cfg = Config.model_validate(
            {
                "buffer_pct": 0.0,
                "engines": {
                    "ollama": {
                        "kind": "ollama",
                        "base_url": "http://localhost:11434",
                        "models": {"m": {"est_vram_gb": 1}},
                    }
                },
            }
        )
        assert cfg.buffer_pct == 0.0


class TestFullConfig:
    def test_engines_parsed(self, tmp_path):
        cfg = load_config(_write(tmp_path, FULL_YAML))
        assert set(cfg.engines) == {"vllm", "ollama"}
        assert cfg.engines["vllm"].kind == "vllm"
        assert cfg.engines["vllm"].models["gemma:31b-awq"].max_model_len == 16384

    def test_alert_config(self, tmp_path):
        cfg = load_config(_write(tmp_path, FULL_YAML))
        assert cfg.alert.channels == ["log", "webhook"]
        assert cfg.alert.webhook_url == "http://localhost:9999/alert"

    def test_est_vram_bytes_conversion(self):
        m = ModelConfig(est_vram_gb=8)
        assert m.est_vram_bytes == 8 * 2**30


class TestModelIndex:
    def test_maps_model_to_engine_name(self, tmp_path):
        cfg = load_config(_write(tmp_path, FULL_YAML))
        assert cfg.engine_for_model("gemma:12b") == "ollama"
        assert cfg.engine_for_model("gemma:31b-awq") == "vllm"

    def test_unknown_model_raises_key_error(self, tmp_path):
        cfg = load_config(_write(tmp_path, FULL_YAML))
        with pytest.raises(KeyError):
            cfg.engine_for_model("nope")

    def test_duplicate_model_across_engines_rejected(self):
        with pytest.raises(ValidationError, match="duplicate"):
            Config.model_validate(
                {
                    "engines": {
                        "a": {
                            "kind": "ollama",
                            "base_url": "http://x",
                            "models": {"m": {"est_vram_gb": 1}},
                        },
                        "b": {
                            "kind": "vllm",
                            "base_url": "http://y",
                            "start": "cmd",
                            "models": {"m": {"est_vram_gb": 1}},
                        },
                    }
                }
            )


class TestEngineValidation:
    def test_vllm_accepts_optional_stop_command(self):
        cfg = EngineConfig.model_validate(
            {
                "kind": "vllm",
                "base_url": "http://x",
                "start": "docker compose up -d vllm",
                "stop": "docker compose stop vllm",
                "models": {},
            }
        )
        assert cfg.stop == "docker compose stop vllm"

    def test_stop_defaults_to_none(self):
        cfg = EngineConfig.model_validate(
            {"kind": "vllm", "base_url": "http://x", "start": "cmd", "models": {}}
        )
        assert cfg.stop is None

    def test_vllm_requires_start_command(self):
        with pytest.raises(ValidationError, match="start"):
            EngineConfig.model_validate(
                {"kind": "vllm", "base_url": "http://x", "models": {}}
            )

    def test_ollama_requires_base_url(self):
        with pytest.raises(ValidationError, match="base_url"):
            EngineConfig.model_validate({"kind": "ollama", "models": {}})

    def test_missing_config_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "missing.yaml")


class TestConfigPathResolution:
    """config.local.yaml (gitignored, machine-specific) beats the tracked
    config.yaml, so personal setups never need to touch public defaults."""

    def test_local_override_wins_when_present(self, tmp_path):
        (tmp_path / "config.yaml").write_text("a: 1", encoding="utf-8")
        (tmp_path / "config.local.yaml").write_text("a: 2", encoding="utf-8")
        from coload.config import resolve_config_path

        assert resolve_config_path(None, cwd=tmp_path).name == "config.local.yaml"

    def test_tracked_default_without_local(self, tmp_path):
        (tmp_path / "config.yaml").write_text("a: 1", encoding="utf-8")
        from coload.config import resolve_config_path

        assert resolve_config_path(None, cwd=tmp_path).name == "config.yaml"

    def test_explicit_path_always_wins(self, tmp_path):
        (tmp_path / "config.local.yaml").write_text("a: 2", encoding="utf-8")
        (tmp_path / "mine.yaml").write_text("a: 3", encoding="utf-8")
        from coload.config import resolve_config_path

        assert resolve_config_path("mine.yaml", cwd=tmp_path).name == "mine.yaml"


class TestPinTtl:
    """The pin is a backstop for a coload that is not running; it must not
    become the thing that does the evicting."""

    def _cfg(self, **engine):
        return {
            "idle_ttl_seconds": 900,
            "engines": {
                "ollama": {
                    "kind": "ollama",
                    "base_url": "http://localhost:11434",
                    "models": {"m": {"est_vram_gb": 1}},
                    **engine,
                }
            },
        }

    def test_defaults_to_a_finite_pin(self):
        cfg = Config.model_validate(self._cfg())
        assert cfg.engines["ollama"].pin_ttl_seconds == 3600

    def test_pin_shorter_than_the_idle_sweep_is_rejected(self):
        """Otherwise Ollama drops the model while coload still believes it is
        resident, and the next request silently pays a reload."""
        with pytest.raises(ValidationError, match="must exceed idle_ttl_seconds"):
            Config.model_validate(self._cfg(pin_ttl_seconds=300))

    def test_zero_opts_back_into_an_indefinite_pin(self):
        """Escape hatch for someone who has disabled the idle sweep and means
        it. Safe now only because adoption reclaims orphans on restart."""
        cfg = Config.model_validate(self._cfg(pin_ttl_seconds=0))
        assert cfg.engines["ollama"].pin_ttl_seconds == 0

    def test_a_negative_pin_is_rejected(self):
        """-1 is Ollama's wire value for forever, not a user-facing setting."""
        with pytest.raises(ValidationError):
            Config.model_validate(self._cfg(pin_ttl_seconds=-1))

    def test_vllm_engines_are_exempt(self):
        """keep_alive is an Ollama mechanism; vLLM lifecycle is a process."""
        cfg = Config.model_validate({
            "idle_ttl_seconds": 900,
            "engines": {
                "vllm": {
                    "kind": "vllm",
                    "base_url": "http://localhost:8000",
                    "start": "docker compose up -d vllm",
                    "models": {"m": {"est_vram_gb": 1}},
                }
            },
        })
        assert "vllm" in cfg.engines


class TestShippedDefaults:
    def test_repo_config_yaml_is_valid(self):
        """The out-of-the-box promise: the shipped config.yaml always parses."""
        from pathlib import Path

        repo_config = Path(__file__).parent.parent / "config.yaml"
        cfg = load_config(repo_config)
        assert cfg.buffer_pct == 0.10
        assert "ollama" in cfg.engines
        assert cfg.engines["vllm"].stop is not None  # compose needs a stop command
