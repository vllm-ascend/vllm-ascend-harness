"""Command execution skill.

Executes shell commands inside a sandbox, returning stdout/stderr.
Integrates with the sandbox module for isolation and resource limits.
"""

from __future__ import annotations

from typing import Any

from vllm_ascend_harness.skills.base import BaseSkill, SkillResult


class CommandExecSkill(BaseSkill):
    """Skill that executes shell commands in a sandbox.

    Parameters
    ----------
    sandbox:
        A :class:`~vllm_ascend_harness.sandbox.base.BaseSandbox`
        instance used for command execution.  If ``None``, the skill
        reports an error.
    max_output_chars:
        Maximum characters of stdout to return (default 10 000).
        Truncated output is annotated.
    """

    name = "command_exec"
    description = "Execute a shell command inside the sandbox."

    def __init__(
        self,
        sandbox: Any | None = None,
        max_output_chars: int = 10_000,
    ) -> None:
        self.sandbox = sandbox
        self.max_output_chars = max_output_chars

    def execute(self, command: str = "", timeout: int = 60, **kwargs: Any) -> SkillResult:
        """Execute *command* inside the sandbox.

        The command is run via ``sandbox.run()``.  The effective
        timeout is ``min(timeout, sandbox.limits.timeout)`` when the
        sandbox exposes resource limits.  Output is truncated if it
        exceeds ``max_output_chars``.
        """
        if not command.strip():
            return SkillResult(output="Empty command", success=False)

        if self.sandbox is None:
            return SkillResult(output="No sandbox configured", success=False)

        try:
            result = self.sandbox.run(command, timeout=timeout)
        except Exception as exc:
            return SkillResult(
                output=f"Execution error: {exc}",
                success=False,
                metadata={"error": str(exc)},
            )

        stdout = result.stdout or ""
        truncated = False
        if len(stdout) > self.max_output_chars:
            stdout = stdout[: self.max_output_chars] + "\n... [truncated]"
            truncated = True

        return SkillResult(
            output=stdout,
            success=(result.exit_code == 0),
            metadata={
                "exit_code": result.exit_code,
                "stderr": result.stderr,
                "duration_ms": result.duration_ms,
                "truncated": truncated,
            },
        )
