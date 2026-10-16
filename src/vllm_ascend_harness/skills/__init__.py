"""Built-in skills module.

Provides a collection of tool skills that the agent can invoke:
code reading, PR review, command execution, etc.
"""

from vllm_ascend_harness.skills.base import BaseSkill, SkillResult
from vllm_ascend_harness.skills.pr_review import (
    PRReviewSkill,
    ReviewComment,
    ReviewResult,
)

__all__ = [
    "BaseSkill",
    "PRReviewSkill",
    "ReviewComment",
    "ReviewResult",
    "SkillResult",
]
