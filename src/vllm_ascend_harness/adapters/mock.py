"""Mock model adapter for non-Ascend environments.

Provides a deterministic, dependency-free adapter that satisfies the
:class:`BaseModelAdapter` contract.  Used for:
- Local development without Ascend NPU hardware
- Unit testing across all modules
- CI pipelines without NPU access

The mock logic echoes the prompt back with a configurable prefix,
and simulates token counting based on whitespace splitting.
"""

from __future__ import annotations

from collections.abc import Iterator

from vllm_ascend_harness.adapters.base import (
    BaseModelAdapter,
    GenerationRequest,
    GenerationResult,
)


class MockAdapter(BaseModelAdapter):
    """Deterministic mock adapter for testing and development.

    Parameters
    ----------
    prefix:
        String prepended to the echoed prompt in the output.
        Defaults to ``"[mock] "``.
    **kwargs:
        Stored in ``self.config`` for inspection in tests.
    """

    name = "mock"

    def __init__(
        self,
        prefix: str = "[mock] ",
        **kwargs: object,
    ) -> None:
        self.prefix = prefix
        self.config = kwargs
        self._closed = False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Echo the prompt back with a prefix, simulating token counts."""
        if self._closed:
            raise RuntimeError("MockAdapter has been closed")

        text = f"{self.prefix}{request.prompt}"
        prompt_tokens = len(request.prompt.split())
        completion_tokens = len(text.split())

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="stop",
            metadata={
                "adapter": self.name,
                "model": "mock-model",
            },
        )

    def stream(self, request: GenerationRequest) -> Iterator[GenerationResult]:
        """Yield the mock output token-by-token (whitespace split)."""
        if self._closed:
            raise RuntimeError("MockAdapter has been closed")

        text = f"{self.prefix}{request.prompt}"
        tokens = text.split()
        prompt_tokens = len(request.prompt.split())

        for i, token in enumerate(tokens):
            is_last = i == len(tokens) - 1
            yield GenerationResult(
                text=token,
                prompt_tokens=prompt_tokens if i == 0 else 0,
                completion_tokens=1,
                finish_reason="stop" if is_last else "length",
                metadata={"adapter": self.name, "streaming": True},
            )

    def health_check(self) -> bool:
        """Always healthy unless closed."""
        return not self._closed

    def close(self) -> None:
        """Mark the adapter as closed."""
        self._closed = True
