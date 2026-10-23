"""Unit tests for TokenStatsCollector."""

import json

from vllm_ascend_harness.telemetry.token_stats import (
    TimeSeriesPoint,
    TokenStats,
    TokenStatsCollector,
)


class TestTokenStats:
    def test_defaults(self):
        stats = TokenStats()
        assert stats.prompt_tokens == 0
        assert stats.completion_tokens == 0
        assert stats.total_calls == 0


class TestRecordAndGet:
    def test_record_single(self):
        collector = TokenStatsCollector()
        collector.record("run-1", prompt_tokens=10, completion_tokens=20)
        stats = collector.get("run-1")
        assert stats is not None
        assert stats.prompt_tokens == 10
        assert stats.completion_tokens == 20
        assert stats.total_calls == 1

    def test_record_multiple_accumulates(self):
        collector = TokenStatsCollector()
        collector.record("run-1", 10, 20)
        collector.record("run-1", 5, 15)
        stats = collector.get("run-1")
        assert stats.prompt_tokens == 15
        assert stats.completion_tokens == 35
        assert stats.total_calls == 2

    def test_record_multiple_runs(self):
        collector = TokenStatsCollector()
        collector.record("run-1", 10, 20)
        collector.record("run-2", 30, 40)
        assert collector.get("run-1").completion_tokens == 20
        assert collector.get("run-2").prompt_tokens == 30

    def test_get_unknown_returns_none(self):
        collector = TokenStatsCollector()
        assert collector.get("nonexistent") is None


class TestSummary:
    def test_summary_returns_all(self):
        collector = TokenStatsCollector()
        collector.record("a", 1, 2)
        collector.record("b", 3, 4)
        summary = collector.summary()
        assert set(summary.keys()) == {"a", "b"}

    def test_summary_is_copy(self):
        collector = TokenStatsCollector()
        collector.record("a", 1, 2)
        summary = collector.summary()
        summary.clear()
        assert collector.get("a") is not None


class TestTimeSeries:
    def test_timeseries_appended(self):
        collector = TokenStatsCollector()
        collector.record("run-1", 10, 20)
        collector.record("run-1", 5, 15)
        ts = collector.timeseries()
        assert len(ts) == 2
        assert all(isinstance(p, TimeSeriesPoint) for p in ts)
        assert ts[0].prompt_tokens == 10
        assert ts[1].prompt_tokens == 5

    def test_timeseries_is_copy(self):
        collector = TokenStatsCollector()
        collector.record("run-1", 10, 20)
        ts = collector.timeseries()
        ts.clear()
        assert len(collector.timeseries()) == 1


class TestExportJson:
    def test_export_json_valid(self):
        collector = TokenStatsCollector()
        collector.record("run-1", 10, 20)
        collector.record("run-2", 30, 40)
        output = collector.export_json()
        data = json.loads(output)
        assert "stats" in data
        assert "timeseries" in data
        assert len(data["stats"]) == 2
        assert len(data["timeseries"]) == 2


class TestPersistence:
    def test_persist_and_load(self, tmp_path):
        path = tmp_path / "tokens.json"
        collector = TokenStatsCollector(persist_path=path)
        collector.record("run-1", 10, 20)
        assert path.exists()

        collector2 = TokenStatsCollector(persist_path=path)
        stats = collector2.get("run-1")
        assert stats is not None
        assert stats.prompt_tokens == 10
        assert stats.total_calls == 1
        assert len(collector2.timeseries()) == 1

    def test_persist_on_each_record(self, tmp_path):
        path = tmp_path / "tokens.json"
        collector = TokenStatsCollector(persist_path=path)
        collector.record("a", 1, 2)
        collector.record("b", 3, 4)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["stats"]) == 2


class TestReset:
    def test_reset_clears_all(self):
        collector = TokenStatsCollector()
        collector.record("a", 1, 2)
        collector.reset()
        assert collector.get("a") is None
        assert collector.timeseries() == []

    def test_reset_with_persistence(self, tmp_path):
        path = tmp_path / "tokens.json"
        collector = TokenStatsCollector(persist_path=path)
        collector.record("a", 1, 2)
        collector.reset()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["stats"] == {}
