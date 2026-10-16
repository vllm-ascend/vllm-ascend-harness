"""Model backend adapters.

Provides a uniform interface for different LLM inference backends.
The vllm-ascend adapter is the primary target; the mock adapter is
used for testing and non-Ascend environments.
"""

from vllm_ascend_harness.adapters.base import (
    BaseModelAdapter,
    GenerationRequest,
    GenerationResult,
)
from vllm_ascend_harness.adapters.mock import MockAdapter

__all__ = [
    "BaseModelAdapter",
    "GenerationRequest",
    "GenerationResult",
    "MockAdapter",
]
