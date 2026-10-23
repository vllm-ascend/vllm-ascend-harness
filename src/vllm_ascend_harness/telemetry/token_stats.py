"""Token usage statistics collector.

Tracks prompt and completion token usage across agent steps and
evaluation runs for cost and resource observability.  Supports
optional persistence to JSON and time-series snapshots.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TokenStats:
    """Aggregated token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_calls: int = 0


@dataclass
class TimeSeriesPoint:
    """A single time-series data point."""

    timestamp: float
    run_id: str
    prompt_tokens: int
    completion_tokens: int


class TokenStatsCollector:
    """Collects and aggregates token usage across runs.

    Parameters
    ----------
    persist_path:
        Optional path to a JSON file.  When set, stats are loaded on
        construction and saved after every :meth:`record` call.
    """

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._stats: dict[str, TokenStats] = {}
        self._timeseries: list[TimeSeriesPoint] = []
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    def record(self, run_id: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage for *run_id*."""
        stats = self._stats.setdefault(run_id, TokenStats())
        stats.prompt_tokens += prompt_tokens
        stats.completion_tokens += completion_tokens
        stats.total_calls += 1
        self._timeseries.append(
            TimeSeriesPoint(
                timestamp=time.time(),
                run_id=run_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
        self._save()

    def get(self, run_id: str) -> TokenStats | None:
        """Return accumulated stats for *run_id*."""
        return self._stats.get(run_id)

    def summary(self) -> dict[str, TokenStats]:
        """Return all collected stats."""
        return dict(self._stats)

    def timeseries(self) -> list[TimeSeriesPoint]:
        """Return the full time-series of recorded data points."""
        return list(self._timeseries)

    def export_json(self) -> str:
        """Export all stats and time-series as a JSON string."""
        data: dict[str, Any] = {
            "stats": {k: asdict(v) for k, v in self._stats.items()},
            "timeseries": [asdict(p) for p in self._timeseries],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _save(self) -> None:
        """Persist current stats to the JSON file if configured."""
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(self.export_json(), encoding="utf-8")

    def _load(self) -> None:
        """Load stats from the JSON file if it exists."""
        assert self._persist_path is not None
        data = json.loads(self._persist_path.read_text(encoding="utf-8"))
        for run_id, d in data.get("stats", {}).items():
            self._stats[run_id] = TokenStats(**d)
        for p in data.get("timeseries", []):
            self._timeseries.append(TimeSeriesPoint(**p))

    def reset(self) -> None:
        """Clear all collected stats and time-series."""
        self._stats.clear()
        self._timeseries.clear()
        self._save()
