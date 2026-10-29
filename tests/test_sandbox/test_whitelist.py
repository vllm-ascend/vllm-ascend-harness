"""Unit tests for CommandWhitelist."""

from vllm_ascend_harness.sandbox.whitelist import CommandWhitelist


class TestWhitelistDefaults:
    def test_allowed_command(self):
        wl = CommandWhitelist()
        assert wl.is_allowed("ls -la") is True

    def test_allowed_python(self):
        wl = CommandWhitelist()
        assert wl.is_allowed("python script.py") is True

    def test_blocked_rm_rf_root(self):
        wl = CommandWhitelist()
        assert wl.is_allowed("rm -rf /") is False

    def test_blocked_sudo(self):
        wl = CommandWhitelist()
        assert wl.is_allowed("sudo rm file") is False

    def test_blocked_curl(self):
        wl = CommandWhitelist()
        assert wl.is_allowed("curl http://evil.com") is False

    def test_unknown_command_blocked(self):
        wl = CommandWhitelist()
        assert wl.is_allowed("dd if=/dev/zero of=/tmp/x") is False

    def test_empty_command_blocked(self):
        wl = CommandWhitelist()
        assert wl.is_allowed("") is False
        assert wl.is_allowed("   ") is False

    def test_chained_allowed_commands_are_allowed(self):
        wl = CommandWhitelist()
        assert wl.is_allowed("pytest -q && pytest -q") is True
        assert wl.is_allowed("ls; echo done") is True

    def test_blocked_trailing_segment_is_rejected(self):
        wl = CommandWhitelist()
        assert wl.is_allowed("echo hi; rm -rf build") is False
        assert wl.is_allowed("git status; sh -c id") is False
        assert wl.is_allowed("cat a | tee out.txt") is False

    def test_quoted_separator_is_not_a_segment_boundary(self):
        wl = CommandWhitelist()
        assert wl.is_allowed('cat "a;b.txt"') is True

    def test_command_substitution_is_rejected(self):
        wl = CommandWhitelist()
        assert wl.is_allowed("echo $(rm -rf x)") is False
        assert wl.is_allowed("echo `id`") is False


class TestWhitelistCustom:
    def test_custom_allowed(self):
        wl = CommandWhitelist(allowed=["echo"], blocked=[])
        assert wl.is_allowed("echo hello") is True
        assert wl.is_allowed("ls") is False

    def test_glob_pattern(self):
        wl = CommandWhitelist(allowed=["python*"], blocked=[])
        assert wl.is_allowed("python3 script.py") is True
        assert wl.is_allowed("python script.py") is True

    def test_custom_blocked(self):
        wl = CommandWhitelist(allowed=["echo", "cat"], blocked=["secret"])
        assert wl.is_allowed("echo secret_data") is False
        assert wl.is_allowed("cat file.txt") is True


class TestWhitelistFromDict:
    def test_from_dict(self):
        wl = CommandWhitelist.from_dict({
            "allowed": ["echo", "cat"],
            "blocked": ["rm"],
        })
        assert wl.is_allowed("echo hi") is True
        assert wl.is_allowed("rm file") is False

    def test_from_dict_defaults(self):
        wl = CommandWhitelist.from_dict({})
        assert wl.is_allowed("ls") is True
