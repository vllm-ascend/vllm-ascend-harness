"""Unit tests for ArtifactCollector."""

import json
from pathlib import Path

from vllm_ascend_harness.telemetry.artifact import ArtifactCollector


class TestSave:
    def test_save_text(self, tmp_path):
        collector = ArtifactCollector(base_dir=tmp_path / "artifacts")
        path = collector.save("run-1", "log.txt", "hello world")
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "hello world"

    def test_save_creates_run_dir(self, tmp_path):
        collector = ArtifactCollector(base_dir=tmp_path / "artifacts")
        collector.save("run-1", "a.txt", "a")
        collector.save("run-1", "b.txt", "b")
        assert (tmp_path / "artifacts" / "run-1" / "a.txt").exists()
        assert (tmp_path / "artifacts" / "run-1" / "b.txt").exists()

    def test_save_bytes(self, tmp_path):
        collector = ArtifactCollector(base_dir=tmp_path / "artifacts")
        path = collector.save_bytes("run-1", "data.bin", b"\x00\x01\x02")
        assert path.read_bytes() == b"\x00\x01\x02"

    def test_save_json(self, tmp_path):
        collector = ArtifactCollector(base_dir=tmp_path / "artifacts")
        path = collector.save_json("run-1", "data.json", {"key": "value"})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {"key": "value"}


class TestListArtifacts:
    def test_list_artifacts(self, tmp_path):
        collector = ArtifactCollector(base_dir=tmp_path / "artifacts")
        collector.save("run-1", "a.txt", "a")
        collector.save("run-1", "b.txt", "b")
        artifacts = collector.list_artifacts("run-1")
        assert len(artifacts) == 2
        assert all(isinstance(a, Path) for a in artifacts)

    def test_list_artifacts_empty(self, tmp_path):
        collector = ArtifactCollector(base_dir=tmp_path / "artifacts")
        assert collector.list_artifacts("nonexistent") == []

    def test_list_runs(self, tmp_path):
        collector = ArtifactCollector(base_dir=tmp_path / "artifacts")
        collector.save("run-1", "a.txt", "a")
        collector.save("run-2", "b.txt", "b")
        runs = collector.list_runs()
        assert set(runs) == {"run-1", "run-2"}


class TestDeleteRun:
    def test_delete_run(self, tmp_path):
        collector = ArtifactCollector(base_dir=tmp_path / "artifacts")
        collector.save("run-1", "a.txt", "a")
        collector.delete_run("run-1")
        assert collector.list_artifacts("run-1") == []

    def test_delete_nonexistent_silent(self, tmp_path):
        collector = ArtifactCollector(base_dir=tmp_path / "artifacts")
        collector.delete_run("nonexistent")


class TestRetentionPolicy:
    def test_prune_old_runs(self, tmp_path):
        collector = ArtifactCollector(
            base_dir=tmp_path / "artifacts", max_retention=2
        )
        collector.save("run-1", "a.txt", "a")
        collector.save("run-2", "b.txt", "b")
        collector.save("run-3", "c.txt", "c")
        runs = set(collector.list_runs())
        assert "run-3" in runs
        assert len(runs) <= 2

    def test_no_prune_when_none(self, tmp_path):
        collector = ArtifactCollector(
            base_dir=tmp_path / "artifacts", max_retention=None
        )
        collector.save("run-1", "a.txt", "a")
        collector.save("run-2", "b.txt", "b")
        collector.save("run-3", "c.txt", "c")
        assert len(collector.list_runs()) == 3


class TestRemoteUploader:
    def test_remote_uploader_called(self, tmp_path):
        uploaded = []

        def uploader(path, run_id):
            uploaded.append((str(path), run_id))

        collector = ArtifactCollector(
            base_dir=tmp_path / "artifacts", remote_uploader=uploader
        )
        collector.save("run-1", "a.txt", "content")
        assert len(uploaded) == 1
        assert uploaded[0][1] == "run-1"

    def test_remote_uploader_error_isolated(self, tmp_path):
        def bad_uploader(path, run_id):
            raise RuntimeError("upload failed")

        collector = ArtifactCollector(
            base_dir=tmp_path / "artifacts", remote_uploader=bad_uploader
        )
        path = collector.save("run-1", "a.txt", "content")
        assert path.exists()
