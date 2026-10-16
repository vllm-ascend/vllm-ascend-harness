"""Command allowlist enforcement.

Defines and enforces the set of commands permitted inside the
sandbox, blocking dangerous or unauthorised operations.

Supports loading from YAML configuration files with glob-style
pattern matching for flexible rule definitions.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ALLOWED = [
    "ls",
    "cat",
    "echo",
    "python",
    "python3",
    "pytest",
    "git",
    "ruff",
    "flake8",
    "mypy",
    "pip",
    "head",
    "tail",
    "wc",
    "grep",
    "find",
]

DEFAULT_BLOCKED = [
    "rm -rf /",
    "rm -rf ~",
    "sudo",
    "su ",
    "chmod 777",
    "curl ",
    "wget ",
    "dd if=",
    "mkfs",
    "shutdown",
    "reboot",
    "kill -9",
    "> /dev/sda",
]


@dataclass
class CommandWhitelist:
    """Allowlist of permitted sandbox commands.

    Parameters
    ----------
    allowed:
        Glob patterns for permitted base commands. ``None`` uses defaults.
    blocked:
        Substrings that block a command if found anywhere in it.
        ``None`` uses defaults.
    """

    allowed: list[str] | None = None
    blocked: list[str] | None = None

    def __post_init__(self) -> None:
        if self.allowed is None:
            self.allowed = list(DEFAULT_ALLOWED)
        if self.blocked is None:
            self.blocked = list(DEFAULT_BLOCKED)

    def is_allowed(self, command: str) -> bool:
        """Return whether *command* is permitted.

        Checks:
        1. If any blocked substring appears in the command → reject
        2. If the base command matches any allowed glob pattern → accept
        3. Otherwise → reject
        """
        command = command.strip()
        if not command:
            return False

        blocked = self.blocked or []
        if any(b in command for b in blocked):
            return False

        base = command.split()[0]
        allowed = self.allowed or []
        return any(fnmatch.fnmatch(base, pattern) for pattern in allowed)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> CommandWhitelist:
        """Build a whitelist from a config dict.

        Expected keys: ``allowed`` (list[str]), ``blocked`` (list[str]).
        Missing keys fall back to defaults.
        """
        return cls(
            allowed=config.get("allowed"),
            blocked=config.get("blocked"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> CommandWhitelist:
        """Load a whitelist from a YAML configuration file.

        Expected structure::

            allowed:
              - ls
              - cat
              - python*
            blocked:
              - "rm -rf /"
              - "sudo"
        """
        import yaml  # noqa: PLC0415

        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return cls.from_dict(config)
