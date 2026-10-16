"""Unit tests for PRReviewSkill.

Uses MockAdapter and pre-fetched diffs to test review logic
without requiring GitHub API access or real LLM inference.
"""

import json

import pytest

from vllm_ascend_harness.adapters.base import GenerationRequest, GenerationResult
from vllm_ascend_harness.adapters.mock import MockAdapter
from vllm_ascend_harness.skills.pr_review import (
    DEFAULT_REVIEW_DIMENSIONS,
    PRReviewSkill,
    ReviewComment,
    ReviewResult,
    _build_review_prompt,
    _parse_review_response,
)

SAMPLE_DIFF = """\
diff --git a/src/app.py b/src/app.py
index 1234567..abcdefg 100644
--- a/src/app.py
+++ b/src/app.py
@@ -10,3 +10,8 @@ def hello():
     print("hello")
+    import os
+    os.system("rm -rf /tmp/cache")
+    password = "admin123"
+    return password
"""

VALID_LLM_RESPONSE = """\
Here is my review:

```json
{
  "summary": "Security issues detected: command injection and hardcoded secret.",
  "approved": false,
  "comments": [
    {
      "dimension": "security",
      "severity": "error",
      "file": "src/app.py",
      "line": 13,
      "message": "os.system with rm -rf is dangerous. Use shutil.rmtree instead."
    },
    {
      "dimension": "security",
      "severity": "error",
      "file": "src/app.py",
      "line": 14,
      "message": "Hardcoded password detected. Use environment variables."
    },
    {
      "dimension": "style",
      "severity": "info",
      "file": "src/app.py",
      "line": 12,
      "message": "Avoid top-level import inside function."
    }
  ]
}
```
"""

NON_JSON_LLM_RESPONSE = "The code looks fine overall, no major issues found."


class _ScriptedAdapter(MockAdapter):
    """MockAdapter that returns a predefined response."""

    def __init__(self, response_text: str):
        super().__init__()
        self._scripted = response_text

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            text=self._scripted,
            prompt_tokens=100,
            completion_tokens=50,
            finish_reason="stop",
            metadata={"adapter": "scripted-mock"},
        )


@pytest.fixture
def adapter():
    return _ScriptedAdapter(VALID_LLM_RESPONSE)


@pytest.fixture
def skill(adapter):
    return PRReviewSkill(adapter=adapter)


class TestBuildReviewPrompt:
    def test_prompt_contains_diff(self):
        prompt = _build_review_prompt(SAMPLE_DIFF, DEFAULT_REVIEW_DIMENSIONS)
        assert "src/app.py" in prompt

    def test_prompt_contains_dimensions(self):
        prompt = _build_review_prompt(SAMPLE_DIFF, DEFAULT_REVIEW_DIMENSIONS)
        assert "style" in prompt
        assert "security" in prompt
        assert "performance" in prompt
        assert "logic" in prompt

    def test_prompt_contains_json_instruction(self):
        prompt = _build_review_prompt(SAMPLE_DIFF, DEFAULT_REVIEW_DIMENSIONS)
        assert "json" in prompt.lower()
        assert "approved" in prompt

    def test_prompt_truncates_large_diff(self):
        large_diff = "x" * 100000
        prompt = _build_review_prompt(large_diff, DEFAULT_REVIEW_DIMENSIONS)
        assert len(prompt) < 100000


class TestParseReviewResponse:
    def test_parse_valid_json_block(self):
        result = _parse_review_response(VALID_LLM_RESPONSE)
        assert result.approved is False
        assert "Security issues" in result.summary
        assert len(result.comments) == 3
        assert result.comments[0].dimension == "security"
        assert result.comments[0].severity == "error"

    def test_parse_plain_json(self):
        json_text = json.dumps({
            "summary": "OK",
            "approved": True,
            "comments": [],
        })
        result = _parse_review_response(json_text)
        assert result.approved is True
        assert result.summary == "OK"

    def test_parse_non_json_fallback(self):
        result = _parse_review_response(NON_JSON_LLM_RESPONSE)
        assert result.approved is False
        assert result.summary == NON_JSON_LLM_RESPONSE
        assert len(result.comments) == 0

    def test_parse_empty_response(self):
        result = _parse_review_response("")
        assert result.approved is False


class TestReviewResultSerialization:
    def test_to_json(self):
        review = ReviewResult(
            summary="Test summary",
            approved=True,
            comments=[
                ReviewComment(dimension="style", severity="info", message="ok"),
            ],
        )
        data = json.loads(review.to_json())
        assert data["summary"] == "Test summary"
        assert data["approved"] is True
        assert len(data["comments"]) == 1
        assert data["comments"][0]["dimension"] == "style"

    def test_review_comment_to_dict(self):
        comment = ReviewComment(
            dimension="security",
            severity="error",
            file="app.py",
            line=42,
            message="bad",
        )
        d = comment.to_dict()
        assert d["dimension"] == "security"
        assert d["severity"] == "error"
        assert d["file"] == "app.py"
        assert d["line"] == 42
        assert d["message"] == "bad"


class TestPRReviewSkillExecute:
    def test_execute_with_provided_diff(self, skill):
        result = skill.execute(diff=SAMPLE_DIFF)
        assert result.success is True
        review_data = json.loads(result.output)
        assert review_data["approved"] is False
        assert len(review_data["comments"]) == 3
        assert result.metadata["num_comments"] == 3

    def test_execute_returns_metadata(self, skill):
        result = skill.execute(diff=SAMPLE_DIFF)
        assert result.metadata["diff_size"] > 0
        assert result.metadata["prompt_tokens"] == 100
        assert result.metadata["completion_tokens"] == 50

    def test_execute_empty_diff(self, skill):
        result = skill.execute(diff="")
        assert result.success is True
        assert "Empty diff" in result.output

    def test_execute_no_repo_no_diff(self, skill):
        result = skill.execute()
        assert result.success is False
        assert "repo and pr_number" in result.metadata["error"]

    def test_execute_with_non_json_response(self):
        adapter = _ScriptedAdapter(NON_JSON_LLM_RESPONSE)
        skill = PRReviewSkill(adapter=adapter)
        result = skill.execute(diff=SAMPLE_DIFF)
        assert result.success is True
        review_data = json.loads(result.output)
        assert review_data["approved"] is False
        assert review_data["summary"] == NON_JSON_LLM_RESPONSE

    def test_execute_custom_dimensions(self):
        adapter = _ScriptedAdapter(VALID_LLM_RESPONSE)
        skill = PRReviewSkill(
            adapter=adapter,
            dimensions={"custom": "Check custom rules."},
        )
        prompt = _build_review_prompt(SAMPLE_DIFF, skill.dimensions)
        assert "custom" in prompt


class TestPRReviewSkillConfig:
    def test_default_dimensions(self, adapter):
        skill = PRReviewSkill(adapter=adapter)
        assert "style" in skill.dimensions
        assert "security" in skill.dimensions

    def test_custom_dimensions(self, adapter):
        skill = PRReviewSkill(
            adapter=adapter,
            dimensions={"lint": "Check lint compliance."},
        )
        assert "lint" in skill.dimensions
        assert "security" not in skill.dimensions

    def test_skill_name(self, skill):
        assert skill.name == "pr_review"

    def test_skill_description(self, skill):
        assert "pull request" in skill.description.lower()
