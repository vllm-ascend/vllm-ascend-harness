"""Code reading skill.

Reads file contents from the workspace for agent analysis.
Enforces workspace boundary and file-size limits for safety.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vllm_ascend_harness.skills.base import BaseSkill, SkillResult

MAX_FILE_SIZE = 1024 * 1024
ALLOWED_EXTENSIONS: tuple[str, ...] | None = None
BLOCKED_PATTERNS = (
    ".env",
    ".ssh",
    ".aws",
    ".git/credentials",
    "id_rsa",
    "credentials",
)


class CodeReadSkill(BaseSkill):
    """Skill that reads file contents from the workspace.

    Parameters
    ----------
    workspace_root:
        The root directory the skill is allowed to read from.  Any
        path that resolves outside this root is rejected.
    max_file_size:
        Maximum file size in bytes (default 1 MiB).  Files exceeding
        this are rejected.
    allowed_extensions:
        Optional tuple of allowed file extensions (e.g.
        ``(".py", ".md")``).  ``None`` means all extensions allowed.
    """

    name = "code_read"
    description = "Read the contents of a file in the workspace."

    def __init__(
        self,
        workspace_root: str | Path = ".",
        max_file_size: int = MAX_FILE_SIZE,
        allowed_extensions: tuple[str, ...] | None = ALLOWED_EXTENSIONS,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions

    def _validate_path(self, file_path: str) -> Path:
        """Validate and resolve *file_path* within the workspace boundary."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        path = path.resolve()

        try:
            path.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError(
                f"Path '{file_path}' is outside the workspace boundary"
            ) from exc

        for pattern in BLOCKED_PATTERNS:
            if pattern in str(path).lower():
                raise ValueError(f"Access to sensitive file blocked: {file_path}")

        if self.allowed_extensions is not None:
            if path.suffix not in self.allowed_extensions:
                raise ValueError(
                    f"File extension '{path.suffix}' not allowed. "
                    f"Allowed: {', '.join(self.allowed_extensions)}"
                )

        if path.exists() and path.is_file():
            size = path.stat().st_size
            if size > self.max_file_size:
                raise ValueError(
                    f"File size {size} bytes exceeds limit "
                    f"of {self.max_file_size} bytes"
                )

        return path

    def execute(self, file_path: str = "", **kwargs: Any) -> SkillResult:
        """Read *file_path* and return its contents.

        Enforces workspace boundary, file size limits, and blocks
        access to sensitive files (credentials, keys, etc.).
        """
        try:
            path = self._validate_path(file_path)
        except ValueError as exc:
            return SkillResult(output=str(exc), success=False)

        if not path.exists():
            return SkillResult(output=f"File not found: {file_path}", success=False)
        try:
            content = path.read_text(encoding="utf-8")
            return SkillResult(
                output=content,
                metadata={
                    "path": str(path),
                    "size": path.stat().st_size,
                },
            )
        except Exception as exc:
            return SkillResult(output=str(exc), success=False)
