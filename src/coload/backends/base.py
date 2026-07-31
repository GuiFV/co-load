"""Backend abstraction.

The orchestrator depends only on this interface (DIP); each engine adapter
implements it. Adding a new engine (llama.cpp, TGI, ...) means one new module
registered in the registry, with no orchestrator changes (OCP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BackendError(RuntimeError):
    """A backend failed to load, unload, or report health."""


class Backend(ABC):
    """One managed inference engine."""

    #: Whether this engine's footprint is decided by what it is handed rather
    #: than by the model.
    #:
    #: Ollama loads a model and uses what the weights need, so measuring it
    #: teaches us something. vLLM claims a slice of the card at startup and
    #: holds it for the process lifetime, so measuring it only tells us what
    #: we allowed it to take. Feeding that back as a learned estimate is a
    #: ratchet: each load is handed the last measurement, exceeds it by the
    #: CUDA context overhead it does not count, and teaches a bigger number
    #: until the model no longer fits the card it was running on an hour ago.
    sizes_to_budget: bool = False

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def is_ready(self, model: str) -> bool:
        """True if ``model`` is loaded and the engine can serve it now."""

    @abstractmethod
    async def load(self, model: str, budget_bytes: int, total_bytes: int) -> None:
        """Load ``model``, sized to at most ``budget_bytes``; block until healthy.

        Raises BackendError on failure. Must leave no half-started process
        behind when it raises.
        """

    @abstractmethod
    async def unload(self, model: str) -> None:
        """Release the VRAM ``model`` holds."""

    @abstractmethod
    async def resident_models(self) -> list[str]:
        """Models currently holding VRAM in this engine."""

    @abstractmethod
    def proxy_url(self, model: str) -> str:
        """Base URL requests for ``model`` should be proxied to."""
