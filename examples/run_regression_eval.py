#!/usr/bin/env python
"""Example: run a nightly regression evaluation.

Demonstrates the EvalRunner with a MockAdapter and ExactMatchScorer.
Replace the adapter with VllmAscendAdapter for real Ascend NPU evaluation:

    from vllm_ascend_harness.adapters.vllm_ascend import VllmAscendAdapter
    adapter = VllmAscendAdapter(model="Qwen/Qwen2.5-7B")

Usage:
    python examples/run_regression_eval.py
"""

from __future__ import annotations

import argparse
import sys

from vllm_ascend_harness.adapters.base import BaseModelAdapter
from vllm_ascend_harness.adapters.mock import MockAdapter
from vllm_ascend_harness.evaluator.runner import EvalRunner, EvalTask
from vllm_ascend_harness.evaluator.scorer import ExactMatchScorer


def build_tasks() -> list[EvalTask]:
    """Build a small set of regression tasks.

    TODO: load from YAML/JSON config in real usage.
    """
    return [
        EvalTask(task_id="reg-001", prompt="hello", expected="[mock] hello"),
        EvalTask(task_id="reg-002", prompt="world", expected="[mock] world"),
        EvalTask(task_id="reg-003", prompt="vllm", expected="ascend"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run nightly regression evaluation")
    parser.add_argument("--config", default=None, help="Path to regression config (YAML)")
    args = parser.parse_args()

    adapter: BaseModelAdapter = MockAdapter()
    scorer = ExactMatchScorer()
    runner = EvalRunner(adapter=adapter, scorer=scorer)

    # TODO: load tasks from config when provided
    tasks = build_tasks()
    report = runner.run(tasks)

    print(f"Tasks evaluated: {len(report.results)}")
    for r in report.results:
        print(f"  {r.task_id}: score={r.score:.2f}  output={r.output!r}")
    print(f"\nAggregate score: {report.aggregate_score:.2f}")
    print("Regression evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
