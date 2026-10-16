"""Unit tests for ContainerSandbox.

These tests verify the container sandbox logic without requiring
Docker to be installed.  Docker-dependent operations are tested
with mocked behavior.
"""

import pytest

from vllm_ascend_harness.sandbox.container import (
    DEFAULT_ASCEND_DEVICES,
    ContainerSandbox,
    _docker_available,
)


class TestContainerSandboxConfig:
    def test_default_config(self):
        sb = ContainerSandbox()
        assert sb.image == "vllm-ascend/harness:latest"
        assert sb.network == "none"
        assert sb.workdir == "/workspace"
        assert sb.devices == DEFAULT_ASCEND_DEVICES
        assert sb._container_id is None

    def test_custom_config(self):
        sb = ContainerSandbox(
            image="custom:latest",
            devices=["/dev/custom"],
            cpu_limit="4",
            memory_limit="8g",
            network="bridge",
            workdir="/app",
        )
        assert sb.image == "custom:latest"
        assert sb.devices == ["/dev/custom"]
        assert sb.cpu_limit == "4"
        assert sb.memory_limit == "8g"
        assert sb.network == "bridge"
        assert sb.workdir == "/app"

    def test_build_run_args(self):
        sb = ContainerSandbox(
            image="test:latest",
            cpu_limit="2",
            memory_limit="4g",
        )
        args = sb._build_run_args()
        assert "docker" in args[0]
        assert "run" in args
        assert "-d" in args
        assert "--rm" in args
        assert "test:latest" in args
        assert "--cpus" in args
        assert "2" in args
        assert "--memory" in args
        assert "4g" in args
        for dev in DEFAULT_ASCEND_DEVICES:
            assert dev in args

    def test_build_run_args_no_limits(self):
        sb = ContainerSandbox()
        args = sb._build_run_args()
        assert "--cpus" not in args
        assert "--memory" not in args


class TestContainerSandboxRunWithoutDocker:
    def test_setup_without_docker_raises(self):
        sb = ContainerSandbox()
        # Force docker check to fail
        sb._docker_ok = False
        with pytest.raises(RuntimeError, match="Docker is not available"):
            sb.setup()

    def test_run_without_container_raises(self):
        sb = ContainerSandbox()
        with pytest.raises(RuntimeError, match="not started"):
            sb.run("echo test")

    def test_blocked_command_returns_126(self):
        sb = ContainerSandbox()
        sb._container_id = "fake-id"
        result = sb.run("rm -rf /")
        assert result.exit_code == 126
        assert "blocked" in result.stderr.lower()


class TestDockerAvailable:
    def test_returns_bool(self):
        result = _docker_available()
        assert isinstance(result, bool)
