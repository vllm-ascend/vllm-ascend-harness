"""Scorer interfaces.

Defines the contract for evaluating model outputs.  This is the
primary extension point for custom scoring strategies (exact-match,
LLM-judge, AIS-Bench metrics, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseScorer(ABC):
    """Abstract base class for scorers.

    Extension point: subclass and implement :meth:`score` to provide
    a custom scoring strategy.
    """

    name: str = "base"

    @abstractmethod
    def score(self, task: Any, output: str) -> float:
        """Return a score in [0, 1] for *output* given *task*."""
        ...

    def batch_score(
        self,
        tasks: list[Any],
        outputs: list[str],
        parallel: bool = False,
        max_workers: int = 4,
    ) -> list[float]:
        """Score a batch of task/output pairs.

        When *parallel* is ``True``, scoring is dispatched via a
        :class:`~concurrent.futures.ThreadPoolExecutor` with up to
        *max_workers* worker threads.
        """
        if len(tasks) != len(outputs):
            raise ValueError("tasks and outputs must have equal length")
        if not parallel or len(tasks) <= 1:
            return [self.score(t, o) for t, o in zip(tasks, outputs)]

        import concurrent.futures

        results: list[float] = [0.0] * len(tasks)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max_workers, len(tasks))
        ) as pool:
            future_to_idx = {
                pool.submit(self.score, t, o): i
                for i, (t, o) in enumerate(zip(tasks, outputs))
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results


class ExactMatchScorer(BaseScorer):
    """Simple exact-match scorer."""

    name = "exact-match"

    def score(self, task: Any, output: str) -> float:
        expected = getattr(task, "expected", None)
        if expected is None:
            return 0.0
        return 1.0 if output.strip() == expected.strip() else 0.0
