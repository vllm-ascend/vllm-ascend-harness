"""Base model adapter interface.

Defines the contract that all inference backends must implement.
This is the primary extension point for plugging in new model
backends (vllm-ascend, vllm, or custom).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class GenerationRequest:
    """A single generation request."""

    prompt: str
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    stop: list[str] | None = None
    metadata: dict | None = None


@dataclass
class GenerationResult:
    """A single generation result."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    metadata: dict | None = None


class BaseModelAdapter(ABC):
    """Abstract base class for model backend adapters.

    Extension point: subclass and implement :meth:`generate` (and
    optionally :meth:`stream`) to integrate a new inference backend.
    """

    name: str = "base"

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Produce a single completion for *request*."""
        ...

    def stream(self, request: GenerationRequest) -> Iterator[GenerationResult]:
        """Stream tokens for *request*.

        Base implementation raises :class:`NotImplementedError`.
        Override in subclasses whose backend supports streaming
        (e.g. :class:`~vllm_ascend_harness.adapters.mock.MockAdapter`
        and
        :class:`~vllm_ascend_harness.adapters.vllm_ascend.VllmAscendAdapter`).
        """
        raise NotImplementedError("Streaming not supported by this adapter")

    def health_check(self) -> bool:
        """Return whether the backend is reachable and ready.

        Base implementation returns ``True`` (assumed healthy).
        Override in subclasses to perform a real backend probe
        (e.g. checking engine initialisation status).
        """
        return True

    def close(self) -> None:
        """Release backend resources.

        Override in subclasses that hold resources (engines, sockets).
        """
        return None
