"""Unit tests for AISBenchScorer.

Tests all metric implementations (accuracy, pass_at_k, bleu, rouge_l,
contains) and the AIS-Bench framework delegation extension point.
"""

import json

import pytest

from vllm_ascend_harness.evaluator.ais_bench import (
    SUPPORTED_METRICS,
    AISBenchScorer,
    MetricReport,
    _accuracy,
    _bleu,
    _contains,
    _normalize,
    _pass_at_k,
    _rouge_l,
    _tokenize,
    set_ais_bench_evaluator,
)


class _Task:
    """Minimal task object for testing."""

    def __init__(self, expected: str):
        self.expected = expected


class TestNormalize:
    def test_strips_whitespace(self):
        assert _normalize("  hello  ") == "hello"

    def test_lowercases(self):
        assert _normalize("Hello WORLD") == "hello world"

    def test_collapses_whitespace(self):
        assert _normalize("hello    world\n") == "hello world"


class TestTokenize:
    def test_simple(self):
        assert _tokenize("hello world") == ["hello", "world"]

    def test_punctuation(self):
        tokens = _tokenize("hello, world!")
        assert tokens == ["hello", "world"]

    def test_empty(self):
        assert _tokenize("") == []


class TestAccuracy:
    def test_exact_match(self):
        assert _accuracy("hello", "hello") == 1.0

    def test_case_insensitive(self):
        assert _accuracy("Hello", "HELLO") == 1.0

    def test_whitespace_normalized(self):
        assert _accuracy("  hello  ", "hello") == 1.0

    def test_mismatch(self):
        assert _accuracy("hello", "world") == 0.0


class TestContains:
    def test_exact_contains(self):
        assert _contains("hello world", "hello") == 1.0

    def test_substring(self):
        assert _contains("the answer is 42", "42") == 1.0

    def test_no_match(self):
        assert _contains("hello", "world") == 0.0

    def test_case_insensitive(self):
        assert _contains("Hello World", "hello") == 1.0


class TestBleu:
    def test_perfect_match(self):
        score = _bleu("the cat sat on the mat", "the cat sat on the mat")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_partial_match(self):
        score = _bleu("the cat sat on the mat", "the dog sat on the mat")
        assert 0.0 < score < 1.0

    def test_no_match(self):
        score = _bleu("completely different text here", "the cat sat on the mat")
        assert score == 0.0

    def test_empty_output(self):
        assert _bleu("", "hello world") == 0.0

    def test_empty_expected(self):
        assert _bleu("hello", "") == 0.0

    def test_short_output_brevity_penalty(self):
        # Short candidate should get brevity penalty
        score = _bleu("the cat", "the cat sat on the mat")
        assert score < 1.0


class TestRougeL:
    def test_perfect_match(self):
        score = _rouge_l("the cat sat", "the cat sat")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_partial_match(self):
        score = _rouge_l("the cat sat on the mat", "the cat")
        assert 0.0 < score < 1.0

    def test_no_match(self):
        assert _rouge_l("hello world", "goodbye universe") == 0.0

    def test_empty(self):
        assert _rouge_l("", "hello") == 0.0
        assert _rouge_l("hello", "") == 0.0


class TestPassAtK:
    def test_match_k1(self):
        assert _pass_at_k("hello", "hello", k=1) == 1.0

    def test_no_match_k1(self):
        assert _pass_at_k("world", "hello", k=1) == 0.0

    def test_zero_k(self):
        assert _pass_at_k("hello", "hello", k=0) == 0.0


class TestAISBenchScorer:
    def test_accuracy_metric(self):
        scorer = AISBenchScorer(metric="accuracy")
        task = _Task(expected="hello")
        assert scorer.score(task, "hello") == 1.0
        assert scorer.score(task, "world") == 0.0

    def test_bleu_metric(self):
        scorer = AISBenchScorer(metric="bleu")
        task = _Task(expected="the cat sat on the mat")
        score = scorer.score(task, "the cat sat on the mat")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_rouge_l_metric(self):
        scorer = AISBenchScorer(metric="rouge_l")
        task = _Task(expected="the cat sat")
        score = scorer.score(task, "the cat sat")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_contains_metric(self):
        scorer = AISBenchScorer(metric="contains")
        task = _Task(expected="42")
        assert scorer.score(task, "the answer is 42") == 1.0
        assert scorer.score(task, "no answer here") == 0.0

    def test_pass_at_k_metric(self):
        scorer = AISBenchScorer(metric="pass_at_k", k=1)
        task = _Task(expected="hello")
        assert scorer.score(task, "hello") == 1.0

    def test_no_expected_returns_zero(self):
        scorer = AISBenchScorer(metric="accuracy")
        task = type("T", (), {"expected": None})()
        assert scorer.score(task, "anything") == 0.0

    def test_unsupported_metric_raises(self):
        with pytest.raises(ValueError, match="Unsupported metric"):
            AISBenchScorer(metric="invalid")

    def test_supported_metrics_list(self):
        assert "accuracy" in SUPPORTED_METRICS
        assert "bleu" in SUPPORTED_METRICS
        assert "rouge_l" in SUPPORTED_METRICS
        assert "pass_at_k" in SUPPORTED_METRICS
        assert "contains" in SUPPORTED_METRICS

    def test_score_with_report(self):
        scorer = AISBenchScorer(metric="accuracy")
        task = _Task(expected="hello")
        report = scorer.score_with_report(task, "hello")
        assert isinstance(report, MetricReport)
        assert report.metric == "accuracy"
        assert report.score == 1.0
        assert report.details["uses_ais_bench"] is False


class TestAISBenchDelegation:
    def test_delegates_to_registered_evaluator(self):
        calls: list[tuple] = []

        def evaluator(output, expected, metric):
            calls.append((output, expected, metric))
            return 0.75

        set_ais_bench_evaluator(evaluator)
        try:
            scorer = AISBenchScorer(metric="accuracy")
            task = _Task(expected="hello")
            assert scorer.score(task, "hello") == 0.75
            assert len(calls) == 1
            assert calls[0] == ("hello", "hello", "accuracy")
        finally:
            # Reset to None for other tests
            set_ais_bench_evaluator(None)

    def test_score_with_report_uses_ais_bench_flag(self):
        def evaluator(output, expected, metric):
            return 0.5

        set_ais_bench_evaluator(evaluator)
        try:
            scorer = AISBenchScorer(metric="bleu")
            task = _Task(expected="hello")
            report = scorer.score_with_report(task, "hello")
            assert report.details["uses_ais_bench"] is True
            assert report.score == 0.5
        finally:
            set_ais_bench_evaluator(None)


class TestMetricReportSerialization:
    def test_to_json(self):
        report = MetricReport(metric="accuracy", score=0.95, details={"k": None})
        data = json.loads(report.to_json())
        assert data["metric"] == "accuracy"
        assert data["score"] == 0.95
        assert data["details"]["k"] is None
