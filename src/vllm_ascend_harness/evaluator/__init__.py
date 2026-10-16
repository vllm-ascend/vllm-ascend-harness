"""Evaluation module.

Provides batch task execution, scorers, AIS-Bench integration,
and regression metric collection for vLLM-Ascend model evaluation.
"""

from vllm_ascend_harness.evaluator.ais_bench import (
    SUPPORTED_METRICS,
    AISBenchScorer,
    MetricReport,
    set_ais_bench_evaluator,
)
from vllm_ascend_harness.evaluator.runner import EvalRunner
from vllm_ascend_harness.evaluator.scorer import BaseScorer

__all__ = [
    "AISBenchScorer",
    "BaseScorer",
    "EvalRunner",
    "MetricReport",
    "SUPPORTED_METRICS",
    "set_ais_bench_evaluator",
]
