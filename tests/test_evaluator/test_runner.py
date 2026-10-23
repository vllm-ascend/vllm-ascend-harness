"""Unit tests for EvalRunner."""

import json

import pytest

from vllm_ascend_harness.adapters.mock import MockAdapter
from vllm_ascend_harness.evaluator.runner import (
    EvalReport,
    EvalResult,
    EvalRunner,
    EvalTask,
)
from vllm_ascend_harness.evaluator.scorer import ExactMatchScorer


@pytest.fixture
def adapter():
    return MockAdapter()


@pytest.fixture
def scorer():
    return ExactMatchScorer()


@pytest.fixture
def tasks():
    return [
        EvalTask(task_id="t1", prompt="hello", expected="[mock] hello", category="greeting"),
        EvalTask(task_id="t2", prompt="world", expected="[mock] world", category="greeting"),
        EvalTask(task_id="t3", prompt="code", expected="ascend", category="code"),
    ]


class TestEvalTask:
    def test_defaults(self):
        task = EvalTask(task_id="t1", prompt="hello")
        assert task.expected is None
        assert task.category == "default"
        assert task.metadata == {}


class TestEvalResult:
    def test_defaults(self):
        result = EvalResult(task_id="t1")
        assert result.output == ""
        assert result.score == 0.0
        assert result.category == "default"


class TestEvalRunnerBasic:
    def test_run_basic(self, adapter, scorer, tasks):
        runner = EvalRunner(adapter=adapter, scorer=scorer)
        report = runner.run(tasks)
        assert len(report.results) == 3
        assert report.aggregate_score > 0.0

    def test_run_no_scorer(self, adapter, tasks):
        runner = EvalRunner(adapter=adapter, scorer=None)
        report = runner.run(tasks)
        assert all(r.score == 0.0 for r in report.results)

    def test_run_empty_tasks(self, adapter, scorer):
        runner = EvalRunner(adapter=adapter, scorer=scorer)
        report = runner.run([])
        assert report.results == []
        assert report.aggregate_score == 0.0


class TestCategoryBreakdown:
    def test_category_breakdowns_computed(self, adapter, scorer, tasks):
        runner = EvalRunner(adapter=adapter, scorer=scorer)
        report = runner.run(tasks)
        assert "greeting" in report.category_breakdowns
        assert "code" in report.category_breakdowns
        greeting = report.category_breakdowns["greeting"]
        assert greeting.count == 2
        assert greeting.total_score == 2.0
        assert greeting.average_score == 1.0

    def test_category_breakdown_empty(self):
        report = EvalReport()
        report.compute_category_breakdowns()
        assert report.category_breakdowns == {}


class TestRegressionDiff:
    def test_regression_diff_detects_changes(self):
        report = EvalReport(
            results=[
                EvalResult(task_id="t1", score=1.0),
                EvalResult(task_id="t2", score=0.0),
            ],
            baseline_scores={"t1": 0.5, "t2": 1.0},
        )
        diffs = report.regression_diff()
        assert "t1" in diffs
        assert diffs["t1"]["delta"] == 0.5
        assert "t2" in diffs
        assert diffs["t2"]["delta"] == -1.0

    def test_regression_diff_no_changes(self):
        report = EvalReport(
            results=[EvalResult(task_id="t1", score=1.0)],
            baseline_scores={"t1": 1.0},
        )
        diffs = report.regression_diff()
        assert diffs == {}


class TestEvalRunnerRetry:
    def test_retry_on_failure(self):
        class FailOnceAdapter(MockAdapter):
            def __init__(self):
                super().__init__()
                self._calls = 0

            def generate(self, request):
                self._calls += 1
                if self._calls == 1:
                    raise RuntimeError("transient failure")
                return super().generate(request)

        adapter = FailOnceAdapter()
        runner = EvalRunner(adapter=adapter, scorer=None, max_retries=2)
        report = runner.run([EvalTask(task_id="t1", prompt="hi")])
        assert len(report.results) == 1
        assert report.results[0].output != ""
        assert report.results[0].metadata["attempts"] == 2

    def test_all_retries_exhausted(self):
        class AlwaysFailAdapter(MockAdapter):
            def generate(self, request):
                raise RuntimeError("permanent failure")

        adapter = AlwaysFailAdapter()
        runner = EvalRunner(adapter=adapter, scorer=None, max_retries=1)
        report = runner.run([EvalTask(task_id="t1", prompt="hi")])
        assert report.results[0].score == 0.0
        assert "error" in report.results[0].metadata


class TestEvalRunnerCheckpoint:
    def test_checkpoint_save_and_resume(self, adapter, scorer, tasks, tmp_path):
        ckpt = tmp_path / "checkpoint.json"
        runner = EvalRunner(adapter=adapter, scorer=scorer, checkpoint_path=ckpt)
        runner.run(tasks)
        assert ckpt.exists()
        data = json.loads(ckpt.read_text())
        assert len(data["results"]) == 3

    def test_checkpoint_resume_skips_cached(self, adapter, scorer, tasks, tmp_path):
        ckpt = tmp_path / "checkpoint.json"
        ckpt.write_text(
            json.dumps({
                "results": [
                    {
                        "task_id": "t1",
                        "output": "[mock] hello",
                        "score": 1.0,
                        "category": "greeting",
                        "metadata": {},
                    }
                ]
            }),
            encoding="utf-8",
        )
        runner = EvalRunner(adapter=adapter, scorer=scorer, checkpoint_path=ckpt)
        report = runner.run(tasks)
        assert len(report.results) == 3
        cached = [r for r in report.results if r.task_id == "t1"][0]
        assert cached.metadata == {}


class TestEvalRunnerParallel:
    def test_parallel_execution(self, adapter, scorer, tasks):
        runner = EvalRunner(
            adapter=adapter, scorer=scorer, parallel=True, max_workers=3
        )
        report = runner.run(tasks)
        assert len(report.results) == 3
        assert report.aggregate_score > 0.0
