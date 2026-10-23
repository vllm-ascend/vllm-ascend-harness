"""Unit tests for CommandExecSkill."""

import pytest

from vllm_ascend_harness.sandbox.process import ProcessSandbox
from vllm_ascend_harness.skills.command_exec import CommandExecSkill


@pytest.fixture
def sandbox():
    sb = ProcessSandbox()
    sb.setup()
    return sb


class TestCommandExecSkillBasic:
    def test_execute_echo(self, sandbox):
        skill = CommandExecSkill(sandbox=sandbox)
        result = skill.execute(command="echo hello")
        assert result.success is True
        assert "hello" in result.output

    def test_execute_python(self, sandbox):
        skill = CommandExecSkill(sandbox=sandbox)
        result = skill.execute(command='python -c "print(42)"')
        assert result.success is True
        assert "42" in result.output

    def test_execute_blocked_command(self, sandbox):
        skill = CommandExecSkill(sandbox=sandbox)
        result = skill.execute(command="rm -rf /")
        assert result.success is False
        assert result.metadata["exit_code"] == 126

    def test_no_sandbox_configured(self):
        skill = CommandExecSkill(sandbox=None)
        result = skill.execute(command="echo hi")
        assert result.success is False
        assert "No sandbox" in result.output

    def test_empty_command(self, sandbox):
        skill = CommandExecSkill(sandbox=sandbox)
        result = skill.execute(command="")
        assert result.success is False
        assert "Empty" in result.output


class TestCommandExecSkillMetadata:
    def test_exit_code_in_metadata(self, sandbox):
        skill = CommandExecSkill(sandbox=sandbox)
        result = skill.execute(command="echo test")
        assert result.metadata["exit_code"] == 0
        assert "stderr" in result.metadata
        assert "duration_ms" in result.metadata

    def test_stderr_captured(self, sandbox):
        skill = CommandExecSkill(sandbox=sandbox)
        result = skill.execute(
            command='python -c "import sys; sys.stderr.write(\'err\')"'
        )
        assert "err" in result.metadata["stderr"]


class TestCommandExecSkillOutputTruncation:
    def test_truncation(self, sandbox):
        skill = CommandExecSkill(sandbox=sandbox, max_output_chars=10)
        result = skill.execute(
            command='python -c "print(\'x\' * 100)"'
        )
        assert result.success is True
        assert len(result.output) <= 30
        assert "truncated" in result.output.lower()
        assert result.metadata["truncated"] is True

    def test_no_truncation_for_short_output(self, sandbox):
        skill = CommandExecSkill(sandbox=sandbox, max_output_chars=1000)
        result = skill.execute(command="echo short")
        assert result.metadata["truncated"] is False
