"""Unit tests for CodeReadSkill."""

import pytest

from vllm_ascend_harness.skills.code_read import MAX_FILE_SIZE, CodeReadSkill


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "test.py").write_text("print('hello')", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# README", encoding="utf-8")
    (tmp_path / "large.txt").write_text("x" * (MAX_FILE_SIZE + 10), encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.py").write_text("y = 1", encoding="utf-8")
    return tmp_path


class TestCodeReadSkillBasic:
    def test_read_existing_file(self, workspace):
        skill = CodeReadSkill(workspace_root=workspace)
        result = skill.execute(file_path="test.py")
        assert result.success is True
        assert "print('hello')" in result.output

    def test_read_nested_file(self, workspace):
        skill = CodeReadSkill(workspace_root=workspace)
        result = skill.execute(file_path="sub/nested.py")
        assert result.success is True
        assert "y = 1" in result.output

    def test_read_nonexistent_file(self, workspace):
        skill = CodeReadSkill(workspace_root=workspace)
        result = skill.execute(file_path="nonexistent.py")
        assert result.success is False
        assert "not found" in result.output.lower()

    def test_read_with_metadata(self, workspace):
        skill = CodeReadSkill(workspace_root=workspace)
        result = skill.execute(file_path="test.py")
        assert result.metadata is not None
        assert "size" in result.metadata
        assert result.metadata["size"] == len("print('hello')")


class TestCodeReadSkillSecurity:
    def test_path_outside_workspace_blocked(self, workspace):
        skill = CodeReadSkill(workspace_root=workspace)
        result = skill.execute(file_path="../../etc/passwd")
        assert result.success is False
        assert "outside" in result.output.lower()

    def test_absolute_path_outside_blocked(self, workspace):
        skill = CodeReadSkill(workspace_root=workspace)
        result = skill.execute(file_path="/etc/passwd")
        assert result.success is False
        assert "outside" in result.output.lower()

    def test_dotdot_traversal_blocked(self, workspace):
        skill = CodeReadSkill(workspace_root=workspace)
        result = skill.execute(file_path="../../../some/secret")
        assert result.success is False

    def test_large_file_blocked(self, workspace):
        skill = CodeReadSkill(workspace_root=workspace)
        result = skill.execute(file_path="large.txt")
        assert result.success is False
        assert "exceeds" in result.output.lower()


class TestCodeReadSkillExtensions:
    def test_allowed_extension(self, workspace):
        skill = CodeReadSkill(
            workspace_root=workspace, allowed_extensions=(".py",)
        )
        result = skill.execute(file_path="test.py")
        assert result.success is True

    def test_blocked_extension(self, workspace):
        skill = CodeReadSkill(
            workspace_root=workspace, allowed_extensions=(".py",)
        )
        result = skill.execute(file_path="readme.md")
        assert result.success is False
        assert "not allowed" in result.output.lower()

    def test_custom_max_file_size(self, workspace):
        skill = CodeReadSkill(workspace_root=workspace, max_file_size=5)
        result = skill.execute(file_path="test.py")
        assert result.success is False
        assert "exceeds" in result.output.lower()
