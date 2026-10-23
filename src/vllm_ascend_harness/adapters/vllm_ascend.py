"""vLLM-Ascend inference backend adapter.

Primary adapter target — wraps the vLLM inference engine with the
vllm-ascend plugin to satisfy the :class:`BaseModelAdapter` contract
for Ascend NPU inference.

When vLLM is not installed (non-Ascend environments), the adapter
raises a clear ``ImportError`` on construction, directing users to
install the ``[ascend]`` extra or use :class:`MockAdapter` for testing.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from vllm_ascend_harness.adapters.base import (
    BaseModelAdapter,
    GenerationRequest,
    GenerationResult,
)


def _request_to_sampling_params(request: GenerationRequest) -> dict[str, Any]:
    """Map :class:`GenerationRequest` fields to vLLM SamplingParams kwargs.

    This dict is passed to ``vllm.SamplingParams(**kwargs)``.
    """
    params: dict[str, Any] = {
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
    }
    if request.stop is not None:
        params["stop"] = request.stop
    return params


class VllmAscendAdapter(BaseModelAdapter):
    """Adapter for the vLLM inference backend with Ascend NPU support.

    This adapter wraps ``vllm.LLM`` and relies on the ``vllm-ascend``
    plugin to route inference to Ascend NPU devices.

    Parameters
    ----------
    model:
        The model name or path (e.g. ``"Qwen/Qwen2.5-7B"``).
    **kwargs:
        Additional keyword arguments forwarded to ``vllm.LLM(...)``.
        Common options include ``tensor_parallel_size``, ``dtype``,
        ``trust_remote_code``, ``enforce_eager``, etc.

    Raises
    ------
    ImportError
        If ``vllm`` is not installed.  Install with ``pip install -e
        ".[ascend]"`` or use :class:`MockAdapter` for testing.
    """

    name = "vllm-ascend"

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self._llm_kwargs = kwargs
        self._engine: Any = None
        self._mock_mode: bool = False

        try:
            from vllm import LLM, SamplingParams  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "vllm is required for VllmAscendAdapter. "
                'Install with: pip install -e ".[ascend]" '
                "or use MockAdapter for testing."
            ) from exc

        self._SamplingParams = SamplingParams
        self._engine = LLM(model=model, **kwargs)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a single completion via the vLLM engine.

        Parameters
        ----------
        request:
            The generation request containing prompt and sampling params.

        Returns
        -------
        GenerationResult
            The generated text with token usage statistics.
        """
        if self._engine is None:
            raise RuntimeError("VllmAscendAdapter engine not initialized")

        sampling_params = self._SamplingParams(**_request_to_sampling_params(request))
        outputs = self._engine.generate(
            [request.prompt], sampling_params, use_tqdm=False
        )

        if not outputs:
            return GenerationResult(text="", finish_reason="error")

        output = outputs[0]
        completion = output.outputs[0]

        prompt_tokens = len(output.prompt_token_ids) if output.prompt_token_ids else 0
        completion_tokens = (
            len(completion.token_ids) if completion.token_ids else 0
        )

        return GenerationResult(
            text=completion.text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=completion.finish_reason or "stop",
            metadata={
                "model": self.model,
                "request_id": getattr(output, "request_id", None),
            },
        )

    def stream(self, request: GenerationRequest) -> Iterator[GenerationResult]:
        """Stream tokens from the vLLM engine.

        Uses vLLM's ``stream_output`` API when available to yield
        :class:`GenerationResult` chunks token-by-token.  Falls back
        to a single-shot ``generate()`` call wrapped in a one-element
        iterator when the streaming API is unavailable.
        """
        if self._engine is None:
            raise RuntimeError("VllmAscendAdapter engine not initialized")

        sampling_params = self._SamplingParams(**_request_to_sampling_params(request))

        stream_fn = getattr(self._engine, "stream_output", None)
        if stream_fn is not None:
            prompt_tokens = 0
            for i, chunk in enumerate(stream_fn([request.prompt], sampling_params)):
                if not chunk.outputs:
                    continue
                output = chunk.outputs[0]
                if i == 0 and output.prompt_token_ids:
                    prompt_tokens = len(output.prompt_ids)
                yield GenerationResult(
                    text=output.text,
                    prompt_tokens=prompt_tokens if i == 0 else 0,
                    completion_tokens=len(output.token_ids) if output.token_ids else 1,
                    finish_reason=output.finish_reason or "length",
                    metadata={
                        "model": self.model,
                        "streaming": True,
                    },
                )
            return

        result = self.generate(request)
        yield result

    def health_check(self) -> bool:
        """Return whether the vLLM engine is initialized and ready."""
        return self._engine is not None

    def close(self) -> None:
        """Release the vLLM engine resources."""
        if self._engine is not None:
            del self._engine
            self._engine = None
