#!/usr/bin/env python
"""Example: run a single-turn agent task.

Demonstrates the basic agent loop with a MockAdapter.
Replace the adapter with VllmAscendAdapter when Ascend NPU is available:

    from vllm_ascend_harness.adapters.vllm_ascend import VllmAscendAdapter
    adapter = VllmAscendAdapter(model="Qwen/Qwen2.5-7B")

Usage:
    python examples/run_agent.py --prompt "Hello, vllm-ascend!"
"""

from __future__ import annotations

import argparse
import sys

from vllm_ascend_harness.adapters.base import (
    BaseModelAdapter,
    GenerationRequest,
)
from vllm_ascend_harness.adapters.mock import MockAdapter
from vllm_ascend_harness.core.agent_loop import AgentContext, AgentLoop, AgentStep


class SingleTurnAgentLoop(AgentLoop):
    """Minimal agent loop that runs one generate step then stops."""

    def step(self) -> AgentStep:
        adapter: BaseModelAdapter = self.context.metadata["adapter"]
        req = GenerationRequest(prompt=self.context.prompt)
        result = adapter.generate(req)
        return AgentStep(
            skill="generate",
            input=self.context.prompt,
            output=result.text,
            metadata={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            },
        )

    def _is_done(self, step: AgentStep) -> bool:
        return step.skill == "generate"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single-turn agent task")
    parser.add_argument("--prompt", default="Hello, vllm-ascend!", help="Prompt text")
    args = parser.parse_args()

    adapter: BaseModelAdapter = MockAdapter()
    context = AgentContext(
        task_id="demo-001",
        prompt=args.prompt,
        max_steps=1,
        metadata={"adapter": adapter},
    )
    loop = SingleTurnAgentLoop(context)
    steps = loop.run()

    for s in steps:
        print(f"skill={s.skill}  output={s.output}")
    print("\nAgent task complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
