#!/usr/bin/env python
"""Example: run a nightly regression evaluation.

Demonstrates the EvalRunner with a MockAdapter and ExactMatchScorer.
Replace the adapter with VllmAscendAdapter for real Ascend NPU evaluation:

    from vllm_ascend_harness.adapters.vllm_ascend import VllmAscendAdapter
    adapter = VllmAscendAdapter(model="Qwen/Qwen2.5-7B")

Usage:
    python examples/run_regression_eval.py
    python examples/run_regression_eval.py --config examples/regression_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from vllm_ascend_harness.adapters.base import BaseModelAdapter
from vllm_ascend_harness.adapters.mock import MockAdapter
from vllm_ascend_harness.evaluator.runner import EvalRunner, EvalTask
from vllm_ascend_harness.evaluator.scorer import ExactMatchScorer


def build_tasks() -> list[EvalTask]:
    """Build a small set of regression tasks (used when no config)."""
    return [
        EvalTask(task_id="reg-001", prompt="hello", expected="[mock] hello"),
        EvalTask(task_id="reg-002", prompt="world", expected="[mock] world"),
        EvalTask(task_id="reg-003", prompt="vllm", expected="ascend"),
    ]


def load_config(config_path: str | Path) -> dict:
    """Load a regression config from a YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_adapter_from_config(cfg: dict) -> BaseModelAdapter:
    """Build a model adapter from the config ``adapter`` section."""
    name = cfg.get("name", "mock")
    kwargs = cfg.get("kwargs", {})
    if name == "mock":
        return MockAdapter(**kwargs)
    if name == "vllm-ascend":
        from vllm_ascend_harness.adapters.vllm_ascend import VllmAscendAdapter
        return VllmAscendAdapter(**kwargs)
    raise ValueError(f"Unknown adapter: {name}")


def build_scorer_from_config(cfg: dict):
    """Build a scorer from the config ``scorer`` section."""
    name = cfg.get("name", "exact-match")
    kwargs = cfg.get("kwargs", {})
    if name == "exact-match":
        return ExactMatchScorer()
    if name == "ais-bench":
        from vllm_ascend_harness.evaluator.ais_bench import AISBenchScorer
        return AISBenchScorer(**kwargs)
    raise ValueError(f"Unknown scorer: {name}")


def build_tasks_from_config(tasks_cfg: list[dict]) -> list[EvalTask]:
    """Build EvalTask list from the config ``tasks`` section."""
    return [
        EvalTask(
            task_id=t["task_id"],
            prompt=t["prompt"],
            expected=t.get("expected"),
            category=t.get("category", "default"),
        )
        for t in tasks_cfg
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run nightly regression evaluation")
    parser.add_argument(
        "--config", default=None, help="Path to regression config (YAML)"
    )
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        adapter = build_adapter_from_config(cfg["adapter"])
        scorer = build_scorer_from_config(cfg["scorer"])
        tasks = build_tasks_from_config(cfg["tasks"])
    else:
        adapter: BaseModelAdapter = MockAdapter()
        scorer = ExactMatchScorer()
        tasks = build_tasks()

    runner = EvalRunner(adapter=adapter, scorer=scorer)
    report = runner.run(tasks)

    print(f"Tasks evaluated: {len(report.results)}")
    for r in report.results:
        print(f"  {r.task_id}: score={r.score:.2f}  output={r.output!r}")
    print(f"\nAggregate score: {report.aggregate_score:.2f}")

    if report.category_breakdowns:
        print("\nCategory breakdown:")
        for cat, bd in sorted(report.category_breakdowns.items()):
            print(
                f"  {cat}: count={bd.count} avg={bd.average_score:.2f}"
            )

    print("Regression evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
