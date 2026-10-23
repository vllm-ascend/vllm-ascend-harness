"""Artifact collection utilities.

Collects and organises output artifacts (logs, traces, results)
produced during agent and evaluation runs.  Supports local storage
with optional remote upload (S3 / OBS) and retention policies.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


class ArtifactCollector:
    """Collects run artifacts to a local directory.

    Parameters
    ----------
    base_dir:
        Root directory for artifact storage.
    max_retention:
        Maximum number of run directories to retain.  Older runs are
        pruned on each :meth:`save` call.  ``None`` disables pruning.
    remote_uploader:
        Optional callable ``(path: Path, run_id: str) -> bool`` that
        uploads an artifact to a remote backend (S3, OBS, etc.).
        When set, each saved artifact is also passed to the uploader.
    """

    def __init__(
        self,
        base_dir: str | Path = "artifacts",
        max_retention: int | None = None,
        remote_uploader: Any = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_retention = max_retention
        self.remote_uploader = remote_uploader

    def save(self, run_id: str, name: str, content: str) -> Path:
        """Save *content* as artifact *name* under *run_id*."""
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / name
        path.write_text(content, encoding="utf-8")
        self._upload(path, run_id)
        self._prune()
        return path

    def save_bytes(self, run_id: str, name: str, data: bytes) -> Path:
        """Save binary *data* as artifact *name* under *run_id*."""
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / name
        path.write_bytes(data)
        self._upload(path, run_id)
        self._prune()
        return path

    def save_json(self, run_id: str, name: str, data: Any) -> Path:
        """Save *data* as a JSON artifact *name* under *run_id*."""
        return self.save(run_id, name, json.dumps(data, indent=2, ensure_ascii=False))

    def list_artifacts(self, run_id: str) -> list[Path]:
        """List artifacts for *run_id*."""
        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            return []
        return sorted(run_dir.iterdir())

    def list_runs(self) -> list[str]:
        """List all run IDs that have artifacts."""
        if not self.base_dir.exists():
            return []
        return sorted(
            d.name for d in self.base_dir.iterdir() if d.is_dir()
        )

    def delete_run(self, run_id: str) -> None:
        """Delete all artifacts for *run_id*."""
        run_dir = self.base_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)

    def _upload(self, path: Path, run_id: str) -> None:
        """Invoke the remote uploader if configured."""
        if self.remote_uploader is not None:
            try:
                self.remote_uploader(path, run_id)
            except Exception:
                pass

    def _prune(self) -> None:
        """Remove old run directories exceeding the retention limit."""
        if self.max_retention is None:
            return
        runs = sorted(
            (d for d in self.base_dir.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
        )
        while len(runs) > self.max_retention:
            oldest = runs.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)
