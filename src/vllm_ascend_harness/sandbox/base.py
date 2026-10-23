"""Base sandbox interface.

Defines the contract for isolated task execution environments
(container-based or process-based).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SandboxLimits:
    """Resource and network policy constraints for a sandbox."""

    cpu_limit: str | None = None
    memory_limit: str | None = None
    timeout: int = 60
    network_enabled: bool = True
    max_file_size: int | None = None
    env: dict[str, str] | None = None


@dataclass
class SandboxResult:
    """Result of a sandboxed command execution."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    metadata: dict | None = None


class BaseSandbox(ABC):
    """Abstract sandbox for isolated command execution.

    Extension point: implement container-based or process-based
    isolation strategies.

    Parameters
    ----------
    limits:
        Optional :class:`SandboxLimits` controlling CPU, memory,
        network, and timeout constraints for every command executed
        in this sandbox.
    """

    def __init__(self, limits: SandboxLimits | None = None) -> None:
        self.limits = limits or SandboxLimits()

    @abstractmethod
    def run(self, command: str, timeout: int | None = None) -> SandboxResult:
        """Execute *command* inside the sandbox.

        The effective timeout is ``timeout`` if given, otherwise
        ``self.limits.timeout``.  Subclasses must enforce resource
        limits (CPU, memory) and network policies from
        :attr:`limits` when applicable.
        """
        ...

    @abstractmethod
    def setup(self) -> None:
        """Prepare the sandbox environment (e.g. start container)."""
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Destroy the sandbox environment."""
        ...

    def __enter__(self) -> BaseSandbox:
        self.setup()
        return self

    def __exit__(self, *exc: object) -> None:
        self.teardown()
