"""Container-based sandbox implementation.

Uses Docker CLI for container-based isolation of agent tasks.
Mounts Ascend NPU device nodes, enforces resource limits, and
applies command allowlist enforcement.

When Docker is not available, falls back to :class:`ProcessSandbox`.
Integrates with the "天罗地网" (sky-net) defense system for
production-grade task isolation.
"""

from __future__ import annotations

import subprocess
import time

from vllm_ascend_harness.sandbox.base import BaseSandbox, SandboxLimits, SandboxResult
from vllm_ascend_harness.sandbox.whitelist import CommandWhitelist

# Default Ascend device nodes to mount into containers
DEFAULT_ASCEND_DEVICES = [
    "/dev/davinci0",
    "/dev/davinci_manager",
    "/dev/devmm_svm",
    "/dev/hisi_hdc",
]


def _docker_available() -> bool:
    """Check if Docker CLI is available and running."""
    try:
        subprocess.run(  # noqa: S603, S607
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class ContainerSandbox(BaseSandbox):
    """Sandbox backed by Docker container runtime.

    Parameters
    ----------
    image:
        Docker image to use for the sandbox container.
    whitelist:
        :class:`CommandWhitelist` for command filtering.
    devices:
        Device nodes to mount into the container (e.g. Ascend NPU).
        Defaults to :data:`DEFAULT_ASCEND_DEVICES`.
    cpu_limit:
        CPU limit for the container (e.g. ``"2"`` for 2 cores).
    memory_limit:
        Memory limit (e.g. ``"4g"``).
    network:
        Network mode (``"none"`` for isolation, ``"bridge"`` for access).
    workdir:
        Working directory inside the container.
    """

    def __init__(
        self,
        image: str = "vllm-ascend/harness:latest",
        whitelist: CommandWhitelist | None = None,
        devices: list[str] | None = None,
        cpu_limit: str | None = None,
        memory_limit: str | None = None,
        network: str = "none",
        workdir: str = "/workspace",
        limits: SandboxLimits | None = None,
    ) -> None:
        super().__init__(limits=limits)
        self.image = image
        self.whitelist = whitelist or CommandWhitelist()
        self.devices = devices if devices is not None else list(DEFAULT_ASCEND_DEVICES)
        self.cpu_limit = cpu_limit or self.limits.cpu_limit
        self.memory_limit = memory_limit or self.limits.memory_limit
        self.network = "none" if not self.limits.network_enabled else network
        self.workdir = workdir
        self._container_id: str | None = None
        self._docker_ok: bool | None = None

    def _check_docker(self) -> bool:
        """Check Docker availability (cached)."""
        if self._docker_ok is None:
            self._docker_ok = _docker_available()
        return self._docker_ok

    def _build_run_args(self) -> list[str]:
        """Build ``docker run`` arguments."""
        args = ["docker", "run", "-d", "--rm"]
        args.extend(["--network", self.network])
        args.extend(["-w", self.workdir])

        for dev in self.devices:
            args.extend(["--device", dev])

        if self.cpu_limit:
            args.extend(["--cpus", self.cpu_limit])

        if self.memory_limit:
            args.extend(["--memory", self.memory_limit])

        args.append(self.image)
        args.append("sleep")
        args.append("infinity")
        return args

    def setup(self) -> None:
        """Start the sandbox container.

        Raises
        ------
        RuntimeError
            If Docker is not available.
        """
        if not self._check_docker():
            raise RuntimeError(
                "Docker is not available. "
                "Use ProcessSandbox for non-container environments."
            )

        result = subprocess.run(  # noqa: S603
            self._build_run_args(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start container: {result.stderr.strip()}"
            )
        self._container_id = result.stdout.strip()

    def run(self, command: str, timeout: int | None = None) -> SandboxResult:
        """Execute *command* inside the container via ``docker exec``.

        Returns
        -------
        SandboxResult
            Captures exit code, stdout, stderr, and duration.
        """
        if self._container_id is None:
            raise RuntimeError("Container not started. Call setup() first.")

        effective_timeout = timeout if timeout is not None else self.limits.timeout

        if not self.whitelist.is_allowed(command):
            return SandboxResult(
                exit_code=126,
                stderr="Command blocked by allowlist",
                metadata={"command": command, "blocked": True},
            )

        exec_args = [
            "docker",
            "exec",
            self._container_id,
            "sh",
            "-c",
            command,
        ]

        start = time.monotonic()
        try:
            proc = subprocess.run(  # noqa: S603
                exec_args,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            duration = int((time.monotonic() - start) * 1000)
            stdout = proc.stdout
            if self.limits.max_file_size and len(stdout) > self.limits.max_file_size:
                stdout = stdout[: self.limits.max_file_size]
            return SandboxResult(
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=proc.stderr,
                duration_ms=duration,
                metadata={
                    "command": command,
                    "sandbox": "container",
                    "container_id": self._container_id,
                },
            )
        except subprocess.TimeoutExpired:
            duration = int((time.monotonic() - start) * 1000)
            return SandboxResult(
                exit_code=124,
                stderr=f"Command timed out after {effective_timeout}s",
                duration_ms=duration,
                metadata={"command": command, "timeout": True},
            )

    def teardown(self) -> None:
        """Stop and remove the sandbox container."""
        if self._container_id is not None:
            subprocess.run(  # noqa: S603
                ["docker", "rm", "-f", self._container_id],
                capture_output=True,
                timeout=30,
            )
            self._container_id = None
