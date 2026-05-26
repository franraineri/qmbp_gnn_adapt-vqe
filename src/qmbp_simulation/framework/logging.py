"""Structured event logging for experiment execution.

Provides machine-parseable event logging for post-hoc analysis of
experiment behavior (timing, convergence, failures).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ExperimentEvent:
    """A single structured event during experiment execution."""

    timestamp: str
    event_type: str
    experiment_id: str
    seed: int | None = None
    h_value: float | None = None
    data: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.floating | np.integer):
                d[k] = float(v) if isinstance(v, np.floating) else int(v)
        return d


class StructuredLogger:
    """Machine-parseable event logger for experiment execution.

    Captures structured events that can be analyzed post-hoc for:
    - Timing breakdowns (which h-points are slow?)
    - Convergence patterns (which seeds fail?)
    - Resource usage (how many function evaluations per point?)

    Usage:
        slog = StructuredLogger("A3")
        slog.log("vqe_start", seed=42, h_value=1.5)
        slog.start_timer("vqe_point")
        # ... do VQE ...
        slog.stop_timer("vqe_point", event_type="vqe_complete",
                        seed=42, h_value=1.5,
                        data={"energy": -5.2, "n_evals": 120})
        slog.save(path)
    """

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.events: list[ExperimentEvent] = []
        self._timers: dict[str, float] = {}

    def log(
        self,
        event_type: str,
        seed: int | None = None,
        h_value: float | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Record a structured event."""
        event = ExperimentEvent(
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            event_type=event_type,
            experiment_id=self.experiment_id,
            seed=seed,
            h_value=h_value,
            data=data or {},
        )
        self.events.append(event)

    def start_timer(self, label: str) -> None:
        """Start a named timer."""
        self._timers[label] = time.time()

    def stop_timer(
        self,
        label: str,
        event_type: str | None = None,
        seed: int | None = None,
        h_value: float | None = None,
        data: dict[str, Any] | None = None,
    ) -> float:
        """Stop a named timer and optionally log the elapsed time."""
        start = self._timers.pop(label, time.time())
        elapsed = time.time() - start
        if event_type:
            event_data = data or {}
            event_data["elapsed_s"] = round(elapsed, 3)
            self.log(event_type, seed=seed, h_value=h_value, data=event_data)
        return elapsed

    def get_events(self, event_type: str | None = None) -> list[ExperimentEvent]:
        """Get events, optionally filtered by type."""
        if event_type is None:
            return list(self.events)
        return [e for e in self.events if e.event_type == event_type]

    def get_timing_summary(self) -> dict[str, Any]:
        """Compute timing breakdown from logged events."""
        by_type: dict[str, list[float]] = {}
        for event in self.events:
            elapsed = event.data.get("elapsed_s", 0)
            if elapsed > 0:
                if event.event_type not in by_type:
                    by_type[event.event_type] = []
                by_type[event.event_type].append(elapsed)

        summary: dict[str, Any] = {}
        for etype, times in by_type.items():
            summary[etype] = {
                "count": len(times),
                "total_s": round(sum(times), 2),
                "mean_s": round(float(np.mean(times)), 3),
                "max_s": round(max(times), 3),
            }
        return summary

    def get_failure_summary(self) -> dict[str, Any]:
        """Summarize failures from logged events."""
        failures = [e for e in self.events if "fail" in e.event_type.lower()]
        by_seed: dict[int, int] = {}
        by_h: dict[float, int] = {}
        reasons: list[str] = []

        for f in failures:
            if f.seed is not None:
                by_seed[f.seed] = by_seed.get(f.seed, 0) + 1
            if f.h_value is not None:
                by_h[f.h_value] = by_h.get(f.h_value, 0) + 1
            reason = f.data.get("reason", "unknown")
            reasons.append(reason)

        return {
            "total_failures": len(failures),
            "by_seed": by_seed,
            "by_h_value": dict(sorted(by_h.items())),
            "reasons": reasons,
        }

    def save(self, path: Path) -> None:
        """Save all events to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "experiment_id": self.experiment_id,
            "n_events": len(self.events),
            "timing_summary": self.get_timing_summary(),
            "failure_summary": self.get_failure_summary(),
            "events": [e.to_dict() for e in self.events],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for embedding in result JSON."""
        return {
            "n_events": len(self.events),
            "timing_summary": self.get_timing_summary(),
            "failure_summary": self.get_failure_summary(),
            "events": [e.to_dict() for e in self.events],
        }


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
