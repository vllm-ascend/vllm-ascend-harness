"""PR review skill.

Provides automated code review capabilities for pull requests,
supporting vLLM-Ascend community CI workflows.

Fetches PR diffs via GitHub API (``gh`` CLI), constructs a review
prompt, invokes an LLM via :class:`BaseModelAdapter`, and returns
structured review feedback.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Any

from vllm_ascend_harness.adapters.base import BaseModelAdapter, GenerationRequest
from vllm_ascend_harness.skills.base import BaseSkill, SkillResult

# Default review dimensions and their prompt guidance
DEFAULT_REVIEW_DIMENSIONS: dict[str, str] = {
    "style": "Code style: naming, formatting, docstrings, PEP 8 compliance.",
    "security": "Security risks: injection, secret leakage, unsafe operations.",
    "performance": "Performance: unnecessary allocations, O(n^2) loops, "
    "missing caching, redundant I/O.",
    "logic": "Logic correctness: edge cases, error handling, race conditions.",
}

# Maximum diff size (chars) to send to the LLM in a single request
MAX_DIFF_SIZE = 50000


@dataclass
class ReviewComment:
    """A single code review comment."""

    dimension: str  # e.g. "style", "security", "performance", "logic"
    severity: str  # "info", "warning", "error"
    file: str = ""
    line: int = 0
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "severity": self.severity,
            "file": self.file,
            "line": self.line,
            "message": self.message,
        }


@dataclass
class ReviewResult:
    """Structured result of a PR review."""

    comments: list[ReviewComment] = field(default_factory=list)
    summary: str = ""
    approved: bool = False

    def to_json(self) -> str:
        return json.dumps(
            {
                "summary": self.summary,
                "approved": self.approved,
                "comments": [c.to_dict() for c in self.comments],
            },
            indent=2,
            ensure_ascii=False,
        )


def _fetch_pr_diff(repo: str, pr_number: int) -> str:
    """Fetch PR diff via ``gh`` CLI.

    Returns
    -------
    str
        The unified diff text.

    Raises
    ------
    RuntimeError
        If ``gh`` CLI is unavailable or the API call fails.
    """
    try:
        result = subprocess.run(  # noqa: S603, S607
            [
                "gh",
                "api",
                f"repos/{repo}/pulls/{pr_number}",
                "-H",
                "Accept: application/vnd.github.v3.diff",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("gh CLI is not installed") from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to fetch PR diff: {result.stderr.strip()}"
        )
    return result.stdout


def _build_review_prompt(
    diff: str,
    dimensions: dict[str, str],
) -> str:
    """Construct the LLM review prompt from a diff.

    Parameters
    ----------
    diff:
        The unified diff text.
    dimensions:
        Review dimensions and their guidance text.
    """
    dims_text = "\n".join(
        f"- **{name}**: {guide}" for name, guide in dimensions.items()
    )
    return (
        "You are a senior code reviewer for the vLLM-Ascend project.\n"
        "Review the following pull request diff and provide structured feedback.\n\n"
        f"Review dimensions:\n{dims_text}\n\n"
        "Respond in JSON with this format:\n"
        "```json\n"
        "{\n"
        '  "summary": "Brief overall assessment",\n'
        '  "approved": true/false,\n'
        '  "comments": [\n'
        '    {"dimension": "...", "severity": "info|warning|error", '
        '"file": "...", "line": 0, "message": "..."}\n'
        "  ]\n"
        "}\n"
        "```\n\n"
        f"```diff\n{diff[:MAX_DIFF_SIZE]}\n```\n"
    )


def _parse_review_response(text: str) -> ReviewResult:
    """Parse LLM output into a :class:`ReviewResult`.

    Attempts to extract a JSON block from the response. Falls back
    to treating the raw text as the summary if parsing fails.
    """
    # Try to extract JSON from markdown code fences
    json_block = text
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            json_block = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            json_block = text[start:end].strip()

    try:
        data = json.loads(json_block)
        comments = [
            ReviewComment(
                dimension=c.get("dimension", "general"),
                severity=c.get("severity", "info"),
                file=c.get("file", ""),
                line=c.get("line", 0),
                message=c.get("message", ""),
            )
            for c in data.get("comments", [])
        ]
        return ReviewResult(
            comments=comments,
            summary=data.get("summary", ""),
            approved=data.get("approved", False),
        )
    except (json.JSONDecodeError, TypeError):
        return ReviewResult(
            summary=text,
            approved=False,
            comments=[],
        )


class PRReviewSkill(BaseSkill):
    """Skill that performs automated code review on a PR.

    Parameters
    ----------
    adapter:
        :class:`BaseModelAdapter` for LLM inference.
    dimensions:
        Review dimensions dict (name → guidance). Defaults to
        :data:`DEFAULT_REVIEW_DIMENSIONS`.
    """

    name = "pr_review"
    description = "Review a pull request and provide structured feedback."

    def __init__(
        self,
        adapter: BaseModelAdapter,
        dimensions: dict[str, str] | None = None,
    ) -> None:
        self.adapter = adapter
        self.dimensions = dimensions or dict(DEFAULT_REVIEW_DIMENSIONS)

    def fetch_diff(self, repo: str, pr_number: int) -> str:
        """Fetch the PR diff (exposed for testing/mocking)."""
        return _fetch_pr_diff(repo, pr_number)

    def execute(
        self,
        repo: str = "",
        pr_number: int = 0,
        diff: str | None = None,
        **kwargs: Any,
    ) -> SkillResult:
        """Review a PR and return structured feedback.

        Parameters
        ----------
        repo:
            GitHub repository in ``owner/name`` format.
        pr_number:
            Pull request number.
        diff:
            Pre-fetched diff text. If provided, skips GitHub API call.
            Useful for testing and offline review.

        Returns
        -------
        SkillResult
            ``output`` contains the JSON-serialized :class:`ReviewResult`.
        """
        if diff is None:
            if not repo or pr_number <= 0:
                return SkillResult(
                    output="",
                    success=False,
                    metadata={"error": "repo and pr_number are required"},
                )
            try:
                diff = self.fetch_diff(repo, pr_number)
            except RuntimeError as exc:
                return SkillResult(
                    output=str(exc),
                    success=False,
                    metadata={"error": str(exc)},
                )

        if not diff.strip():
            return SkillResult(
                output="Empty diff — nothing to review.",
                success=True,
                metadata={"review_result": ReviewResult().to_json()},
            )

        prompt = _build_review_prompt(diff, self.dimensions)
        request = GenerationRequest(prompt=prompt, max_tokens=2048, temperature=0.1)
        gen_result = self.adapter.generate(request)

        review = _parse_review_response(gen_result.text)
        return SkillResult(
            output=review.to_json(),
            success=True,
            metadata={
                "review_result": review.to_json(),
                "diff_size": len(diff),
                "prompt_tokens": gen_result.prompt_tokens,
                "completion_tokens": gen_result.completion_tokens,
                "num_comments": len(review.comments),
                "approved": review.approved,
            },
        )
