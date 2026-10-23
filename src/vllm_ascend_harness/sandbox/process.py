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

from vllm_ascend_harness.sandbox.base import BaseSandbox, SandboxLimits, SandboxResult
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
        limits: SandboxLimits | None = None,
    ) -> None:
        super().__init__(limits=limits)
        self.whitelist = whitelist or CommandWhitelist()
        self.workdir = str(workdir) if workdir else None
        if self.limits.env:
            self.env: dict[str, str] | None = {**(env or {}), **self.limits.env}
        else:
            self.env = env
        self._active = False

    def setup(self) -> None:
        """Mark the sandbox as active."""
        self._active = True

    def run(self, command: str, timeout: int | None = None) -> SandboxResult:
        """Execute *command* in a subprocess.

        Returns
        -------
        SandboxResult
            Captures exit code, stdout, stderr, and duration.
        """
        if not self._active:
            self.setup()

        effective_timeout = timeout if timeout is not None else self.limits.timeout

        if not self.whitelist.is_allowed(command):
            return SandboxResult(
                exit_code=126,
                stderr="Command blocked by allowlist",
                metadata={"command": command, "blocked": True},
            )

        start = time.monotonic()
        try:
            if os.name == "nt":
                args: list[str] | str = command
                shell = True
            else:
                args = shlex.split(command)
                shell = False

            proc = subprocess.run(  # noqa: S603
                args,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                cwd=self.workdir,
                env={**os.environ, **self.env} if self.env else None,
                shell=shell,
            )
            duration = int((time.monotonic() - start) * 1000)
            stdout = proc.stdout
            if self.limits.max_file_size and len(stdout) > self.limits.max_file_size:
                stdout = stdout[: self.limits.max_file_size]
            return SandboxResult(
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=proc.stderr,
                duration_ms=duration,
                metadata={"command": command, "sandbox": "process"},
            )
        except subprocess.TimeoutExpired:
            duration = int((time.monotonic() - start) * 1000)
            return SandboxResult(
                exit_code=124,
                stderr=f"Command timed out after {effective_timeout}s",
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
