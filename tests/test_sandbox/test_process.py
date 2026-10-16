"""Unit tests for ProcessSandbox."""

import pytest

from vllm_ascend_harness.sandbox.process import ProcessSandbox
from vllm_ascend_harness.sandbox.whitelist import CommandWhitelist


@pytest.fixture
def sandbox():
    sb = ProcessSandbox()
    sb.setup()
    return sb


class TestProcessSandboxRun:
    def test_run_echo(self, sandbox):
        result = sandbox.run("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.stdout

    def test_run_python(self, sandbox):
        result = sandbox.run('python -c "print(42)"')
        assert result.exit_code == 0
        assert "42" in result.stdout

    def test_run_blocked_command(self, sandbox):
        result = sandbox.run("rm -rf /")
        assert result.exit_code == 126
        assert "blocked" in result.stderr.lower()

    def test_run_unknown_command_blocked(self, sandbox):
        result = sandbox.run("nonexistent_command_xyz")
        assert result.exit_code == 126
        assert "blocked" in result.stderr.lower()

    def test_run_returns_metadata(self, sandbox):
        result = sandbox.run("echo test")
        assert result.metadata is not None
        assert result.metadata["sandbox"] == "process"

    def test_run_captures_stderr(self, sandbox):
        result = sandbox.run('python -c "import sys; sys.stderr.write(\'err\')"')
        assert "err" in result.stderr


class TestProcessSandboxLifecycle:
    def test_setup_marks_active(self):
        sb = ProcessSandbox()
        assert sb._active is False
        sb.setup()
        assert sb._active is True

    def test_teardown_marks_inactive(self, sandbox):
        assert sandbox._active is True
        sandbox.teardown()
        assert sandbox._active is False

    def test_context_manager(self):
        with ProcessSandbox() as sb:
            assert sb._active is True
            result = sb.run("echo ctx")
            assert result.exit_code == 0
        assert sb._active is False

    def test_auto_setup_on_run(self):
        sb = ProcessSandbox()
        assert sb._active is False
        result = sb.run("echo auto")
        assert result.exit_code == 0
        assert sb._active is True


class TestProcessSandboxCustomWhitelist:
    def test_custom_whitelist_allows(self):
        wl = CommandWhitelist(allowed=["echo"], blocked=[])
        sb = ProcessSandbox(whitelist=wl)
        result = sb.run("echo custom")
        assert result.exit_code == 0

    def test_custom_whitelist_blocks(self):
        wl = CommandWhitelist(allowed=["echo"], blocked=[])
        sb = ProcessSandbox(whitelist=wl)
        result = sb.run("ls")
        assert result.exit_code == 126
