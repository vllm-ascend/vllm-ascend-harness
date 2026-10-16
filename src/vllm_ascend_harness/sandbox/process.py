"""Process-based sandbox implementation.

Uses ``subprocess`` to execute commands in an isolated process with
allowlist enforcement.  This is the default sandbox when Docker is
not available, suitable for development and CI without container
runtime.

Security note: process-based isolation is weaker than container-based
isolation.  Use :class:`ContainerSandbox` for untrusted workloads.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

from vllm_ascend_harness.sandbox.base import BaseSandbox, SandboxResult
from vllm_ascend_harness.sandbox.whitelist import CommandWhitelist


class ProcessSandbox(BaseSandbox):
    """Sandbox backed by ``subprocess`` process execution.

    Parameters
    ----------
    whitelist:
        :class:`CommandWhitelist` for command filtering.
        Defaults to :class:`CommandWhitelist` with built-in rules.
    workdir:
        Working directory for command execution.
        Defaults to current directory.
    env:
        Environment variables for the subprocess.
        Defaults to ``None`` (inherits parent environment).
    """

    def __init__(
        self,
        whitelist: CommandWhitelist | None = None,
        workdir: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.whitelist = whitelist or CommandWhitelist()
        self.workdir = str(workdir) if workdir else None
        self.env = env
        self._active = False

    def setup(self) -> None:
        """Mark the sandbox as active."""
        self._active = True

    def run(self, command: str, timeout: int = 60) -> SandboxResult:
        """Execute *command* in a subprocess.

        Returns
        -------
        SandboxResult
            Captures exit code, stdout, stderr, and duration.
        """
        if not self._active:
            self.setup()

        if not self.whitelist.is_allowed(command):
            return SandboxResult(
                exit_code=126,
                stderr="Command blocked by allowlist",
                metadata={"command": command, "blocked": True},
            )

        start = time.monotonic()
        try:
            if os.name == "nt":
                args = command
                shell = True
            else:
                args = shlex.split(command)
                shell = False

            proc = subprocess.run(  # noqa: S603
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workdir,
                env={**os.environ, **self.env} if self.env else None,
                shell=shell,
            )
            duration = int((time.monotonic() - start) * 1000)
            return SandboxResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_ms=duration,
                metadata={"command": command, "sandbox": "process"},
            )
        except subprocess.TimeoutExpired:
            duration = int((time.monotonic() - start) * 1000)
            return SandboxResult(
                exit_code=124,
                stderr=f"Command timed out after {timeout}s",
                duration_ms=duration,
                metadata={"command": command, "timeout": True},
            )
        except FileNotFoundError as exc:
            duration = int((time.monotonic() - start) * 1000)
            return SandboxResult(
                exit_code=127,
                stderr=str(exc),
                duration_ms=duration,
                metadata={"command": command, "error": "FileNotFoundError"},
            )

    def teardown(self) -> None:
        """Mark the sandbox as inactive."""
        self._active = False
