"""Per-model VRAM estimates: seeded from config, refined by observation.

Learned figures are cached to disk so a restart keeps what real loads taught
us. Observation keeps the *peak* seen: KV cache growth means real usage can
only surprise upward.
"""

from __future__ import annotations

import json
from pathlib import Path


class EstimateStore:
    def __init__(self, path: str | Path, seeds: dict[str, int]):
        self._path = Path(path)
        self._seeds = dict(seeds)
        self._learned: dict[str, int] = self._load()

    def estimate(self, model: str) -> int:
        """Best-known VRAM need in bytes; learned beats seed. KeyError if unknown."""
        if model in self._learned:
            return self._learned[model]
        return self._seeds[model]

    def observe(self, model: str, measured_bytes: int) -> None:
        """Record a real measured footprint; keeps the peak and persists."""
        if measured_bytes <= 0:
            return
        current = self._learned.get(model, 0)
        if measured_bytes > current:
            self._learned[model] = measured_bytes
            self._save()

    def _load(self) -> dict[str, int]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return {str(k): int(v) for k, v in raw.items()}
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._learned, indent=2), encoding="utf-8")
        tmp.replace(self._path)
