"""AIS-Bench integration layer.

Provides integration with the AIS-Bench evaluation framework for
collecting standardised regression metrics on Ascend NPU.

Includes pure-Python metric implementations (accuracy, pass@k, bleu,
rouge, contains) that work without Ascend hardware, with an extension
point to delegate to the real AIS-Bench framework when available.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from vllm_ascend_harness.evaluator.scorer import BaseScorer

# Supported metric names
SUPPORTED_METRICS = ("accuracy", "pass_at_k", "bleu", "rouge_l", "contains")

# Extension point: override this to integrate real AIS-Bench framework.
# Set to a callable(output: str, expected: str, metric: str) -> float.
_ais_bench_evaluator: Callable[[str, str, str], float] | None = None


def set_ais_bench_evaluator(evaluator: Any) -> None:
    """Register a real AIS-Bench evaluator callable.

    Extension point for integrating the actual AIS-Bench framework.
    The callable signature is::

        evaluator(output: str, expected: str, metric: str) -> float

    When set, :class:`AISBenchScorer` delegates to it instead of
    using the built-in pure-Python metric implementations.
    """
    global _ais_bench_evaluator
    _ais_bench_evaluator = evaluator


@dataclass
class MetricReport:
    """Structured report of a single metric evaluation."""

    metric: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"metric": self.metric, "score": self.score, "details": self.details},
            indent=2,
            ensure_ascii=False,
        )


def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower().strip()
    tokens = re.findall(r"\w+", text)
    return tokens


def _accuracy(output: str, expected: str) -> float:
    """Exact match after normalization."""
    return 1.0 if _normalize(output) == _normalize(expected) else 0.0


def _contains(output: str, expected: str) -> float:
    """Check if expected is a substring of output (normalized)."""
    return 1.0 if _normalize(expected) in _normalize(output) else 0.0


def _bleu(output: str, expected: str, max_n: int = 4) -> float:
    """Compute BLEU score (simplified, up to 4-gram).

    Uses brevity penalty and geometric mean of n-gram precisions.
    """
    ref_tokens = _tokenize(expected)
    cand_tokens = _tokenize(output)

    if not cand_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    bp = 1.0 if len(cand_tokens) >= len(ref_tokens) else math.exp(
        1 - len(ref_tokens) / len(cand_tokens)
    )

    # N-gram precisions
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        if len(cand_tokens) < n:
            precisions.append(0.0)
            continue

        cand_ngrams = Counter(
            tuple(cand_tokens[i : i + n])
            for i in range(len(cand_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i : i + n])
            for i in range(len(ref_tokens) - n + 1)
        )

        total = sum(cand_ngrams.values())
        if total == 0:
            precisions.append(0.0)
            continue

        matches = sum(min(c, ref_ngrams.get(ng, 0)) for ng, c in cand_ngrams.items())
        precisions.append(matches / total)

    # Geometric mean (filter zero precisions)
    nonzero = [p for p in precisions if p > 0]
    if not nonzero:
        return 0.0

    geo_mean = math.exp(sum(math.log(p) for p in nonzero) / len(nonzero))
    return bp * geo_mean


def _rouge_l(output: str, expected: str) -> float:
    """Compute ROUGE-L (F1) based on longest common subsequence."""
    ref_tokens = _tokenize(expected)
    cand_tokens = _tokenize(output)

    if not ref_tokens or not cand_tokens:
        return 0.0

    # LCS via DP
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    if lcs == 0:
        return 0.0

    recall = lcs / m
    precision = lcs / n
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def _pass_at_k(output: str, expected: str, k: int = 1) -> float:
    """Compute pass@k for code generation tasks.

    For k=1, this is a simple accuracy check (normalized exact match).
    For k>1, the standard pass@k formula is applied assuming *output*
    contains ``k`` newline-separated candidate answers:

    .. math::
        pass@k = 1 - \\binom{n-c}{k} / \\binom{n}{k}

    where *n* is the total number of samples and *c* is the number
    of correct samples.  When only a single output is provided (no
    newline), pass@k reduces to a single-sample accuracy check.
    """
    if k <= 0:
        return 0.0

    candidates = output.split("\n") if "\n" in output else [output]
    n = len(candidates)
    correct = sum(
        1 for c in candidates if _normalize(c) == _normalize(expected)
    )

    if n == 0 or correct == 0:
        return 0.0
    if correct >= k:
        return 1.0

    if k == 1:
        return _accuracy(output, expected)

    if n < k:
        return float(correct) / n

    from functools import reduce
    from operator import mul

    def _comb(a: int, b: int) -> int:
        if b > a or b < 0:
            return 0
        if b == 0:
            return 1
        b = min(b, a - b)
        num = reduce(mul, range(a, a - b, -1), 1)
        den = reduce(mul, range(1, b + 1), 1)
        return num // den

    total = _comb(n, k)
    fail = _comb(n - correct, k)
    if total == 0:
        return 0.0
    return 1.0 - fail / total


# Metric name → computation function
_METRIC_FUNCTIONS: dict[str, Callable[..., float]] = {
    "accuracy": _accuracy,
    "pass_at_k": _pass_at_k,
    "bleu": _bleu,
    "rouge_l": _rouge_l,
    "contains": _contains,
}


class AISBenchScorer(BaseScorer):
    """Scorer that evaluates model outputs using AIS-Bench metrics.

    Supports the following metrics (selected at construction time):

    - ``accuracy`` — normalized exact match
    - ``pass_at_k`` — pass@k for code generation (k=1 default)
    - ``bleu`` — BLEU score (up to 4-gram with brevity penalty)
    - ``rouge_l`` — ROUGE-L F1 based on longest common subsequence
    - ``contains`` — substring containment check

    When a real AIS-Bench evaluator is registered via
    :func:`set_ais_bench_evaluator`, scoring is delegated to it.

    Parameters
    ----------
    metric:
        Metric name. Must be one of :data:`SUPPORTED_METRICS`.
    k:
        Value of k for pass@k metric (default 1).
    """

    name = "ais-bench"

    def __init__(self, metric: str = "accuracy", k: int = 1) -> None:
        if metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric: {metric}. "
                f"Supported: {', '.join(SUPPORTED_METRICS)}"
            )
        self.metric = metric
        self.k = k

    def score(self, task: Any, output: str) -> float:
        """Compute the configured AIS-Bench metric for *output*.

        Returns
        -------
        float
            Score in [0, 1].
        """
        expected = getattr(task, "expected", None)
        if expected is None:
            return 0.0

        # Delegate to real AIS-Bench framework if registered
        if _ais_bench_evaluator is not None:
            return float(_ais_bench_evaluator(output, expected, self.metric))

        # Fall back to pure-Python implementation
        func = _METRIC_FUNCTIONS[self.metric]
        if self.metric == "pass_at_k":
            return float(func(output, expected, k=self.k))
        return float(func(output, expected))

    def score_with_report(self, task: Any, output: str) -> MetricReport:
        """Score and return a detailed :class:`MetricReport`."""
        score = self.score(task, output)
        return MetricReport(
            metric=self.metric,
            score=score,
            details={
                "k": self.k if self.metric == "pass_at_k" else None,
                "uses_ais_bench": _ais_bench_evaluator is not None,
            },
        )
