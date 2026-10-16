"""Task sandbox module.

Provides container isolation and command allowlist enforcement,
integrating with the "天罗地网" (sky-net) defense system to
ensure safe execution of agent-driven commands.

- :class:`ProcessSandbox` — subprocess-based, works anywhere (default)
- :class:`ContainerSandbox` — Docker-based, stronger isolation
- :class:`CommandWhitelist` — configurable command filtering
"""

from vllm_ascend_harness.sandbox.base import BaseSandbox, SandboxResult
from vllm_ascend_harness.sandbox.container import ContainerSandbox
from vllm_ascend_harness.sandbox.process import ProcessSandbox
from vllm_ascend_harness.sandbox.whitelist import CommandWhitelist

__all__ = [
    "BaseSandbox",
    "CommandWhitelist",
    "ContainerSandbox",
    "ProcessSandbox",
    "SandboxResult",
]
