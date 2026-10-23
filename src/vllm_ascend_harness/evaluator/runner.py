"""Batch evaluation runner.

Drives batch execution of evaluation tasks across a dataset,
collecting per-task results and aggregate metrics.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm_ascend_harness.adapters.base import BaseModelAdapter, GenerationRequest
from vllm_ascend_harness.telemetry.logger import get_logger

_logger = get_logger("evaluator.runner")


@dataclass
class EvalTask:
    """A single evaluation task."""

    task_id: str
    prompt: str
    expected: str | None = None
    category: str = "default"
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of a single evaluation task."""

    task_id: str
    output: str = ""
    score: float = 0.0
    category: str = "default"
    metadata: dict = field(default_factory=dict)


@dataclass
class CategoryBreakdown:
    """Per-category aggregate metrics."""

    category: str
    count: int = 0
    total_score: float = 0.0

    @property
    def average_score(self) -> float:
        return self.total_score / self.count if self.count else 0.0


@dataclass
class EvalReport:
    """Aggregate report across all evaluation tasks."""

    results: list[EvalResult] = field(default_factory=list)
    aggregate_score: float = 0.0
    category_breakdowns: dict[str, CategoryBreakdown] = field(default_factory=dict)
    baseline_scores: dict[str, float] = field(default_factory=dict)

    def compute_category_breakdowns(self) -> None:
        """Compute per-category aggregates from ``results``."""
        breakdowns: dict[str, CategoryBreakdown] = {}
        for r in self.results:
            bd = breakdowns.setdefault(
                r.category, CategoryBreakdown(category=r.category)
            )
            bd.count += 1
            bd.total_score += r.score
        self.category_breakdowns = breakdowns

    def regression_diff(self) -> dict[str, dict[str, float]]:
        """Compare current results against ``baseline_scores``.

        Returns a dict mapping ``task_id`` to ``{"baseline": ...,
        "current": ..., "delta": ...}`` for tasks where the score
        changed.
        """
        diffs: dict[str, dict[str, float]] = {}
        for r in self.results:
            if r.task_id in self.baseline_scores:
                baseline = self.baseline_scores[r.task_id]
                delta = r.score - baseline
                if abs(delta) > 1e-9:
                    diffs[r.task_id] = {
                        "baseline": baseline,
                        "current": r.score,
                        "delta": delta,
                    }
        return diffs


class EvalRunner:
    """Batch evaluation runner.

    Iterates tasks, invokes the model adapter, applies scorers,
    and aggregates results into an :class:`EvalReport`.

    Parameters
    ----------
    adapter:
        The model backend adapter used for generation.
    scorer:
        Optional scorer applied to each task output.
    max_retries:
        Number of retry attempts for failed generations (default 0).
    parallel:
        Whether to run tasks in parallel using a thread pool (default
        ``False``).
    max_workers:
        Maximum worker threads when *parallel* is ``True``.
    checkpoint_path:
        Optional path to a JSON file for checkpointing.  Completed
        results are saved after each task so a crashed run can be
        resumed.
    """

    def __init__(
        self,
        adapter: BaseModelAdapter,
        scorer: Any | None = None,
        *,
        max_retries: int = 0,
        parallel: bool = False,
        max_workers: int = 4,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.adapter = adapter
        self.scorer = scorer
        self.max_retries = max_retries
        self.parallel = parallel
        self.max_workers = max_workers
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    def _run_single(self, task: EvalTask) -> EvalResult:
        """Execute a single task with retry logic."""
        last_error: str = ""
        for attempt in range(self.max_retries + 1):
            try:
                request = GenerationRequest(prompt=task.prompt)
                gen_result = self.adapter.generate(request)
                score = 0.0
                if self.scorer is not None:
                    score = self.scorer.score(task, gen_result.text)
                return EvalResult(
                    task_id=task.task_id,
                    output=gen_result.text,
                    score=score,
                    category=task.category,
                    metadata={
                        "attempts": attempt + 1,
                        "prompt_tokens": gen_result.prompt_tokens,
                        "completion_tokens": gen_result.completion_tokens,
                    },
                )
            except Exception as exc:
                last_error = str(exc)
                _logger.warning(
                    "Task %s attempt %d/%d failed: %s",
                    task.task_id,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                time.sleep(0.1 * (attempt + 1))
        return EvalResult(
            task_id=task.task_id,
            output=f"[error] {last_error}",
            score=0.0,
            category=task.category,
            metadata={"error": last_error, "attempts": self.max_retries + 1},
        )

    def _load_checkpoint(self) -> dict[str, EvalResult]:
        """Load previously saved results from the checkpoint file."""
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            return {}
        data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        return {
            d["task_id"]: EvalResult(**d) for d in data.get("results", [])
        }

    def _save_checkpoint(self, results: list[EvalResult]) -> None:
        """Save results to the checkpoint file."""
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "results": [
                {
                    "task_id": r.task_id,
                    "output": r.output,
                    "score": r.score,
                    "category": r.category,
                    "metadata": r.metadata,
                }
                for r in results
            ]
        }
        self.checkpoint_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def run(self, tasks: list[EvalTask]) -> EvalReport:
        """Execute all *tasks* and return an aggregate report.

        Supports parallel execution via a thread pool, configurable
        retries per task, and checkpoint/resume via a JSON file.
        """
        cached = self._load_checkpoint()
        results: list[EvalResult] = []
        pending: list[EvalTask] = []

        for task in tasks:
            if task.task_id in cached:
                results.append(cached[task.task_id])
            else:
                pending.append(task)

        if pending:
            if self.parallel and len(pending) > 1:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(self.max_workers, len(pending))
                ) as pool:
                    futures = {
                        pool.submit(self._run_single, t): t for t in pending
                    }
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        results.append(result)
                        self._save_checkpoint(results)
            else:
                for task in pending:
                    result = self._run_single(task)
                    results.append(result)
                    self._save_checkpoint(results)

        results.sort(key=lambda r: r.task_id)
        total_score = sum(r.score for r in results)
        aggregate = total_score / len(results) if results else 0.0
        report = EvalReport(results=results, aggregate_score=aggregate)
        report.compute_category_breakdowns()
        return report
