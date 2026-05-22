"""Auto-registration and structured event logging for V8 experiments.

Provides:
- @register_experiment decorator for automatic registry population
- StructuredLogger for machine-parseable event logging
- ExperimentEvent dataclass for typed event records
- RunSummary for post-hoc analysis of experiment execution
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Auto-Registration ────────────────────────────────────────────────────────

# Global registry populated by @register_experiment decorator
_AUTO_REGISTRY: dict[str, tuple[str, str]] = {}


def register_experiment(experiment_id: str, category: str = "X"):
    """Class decorator that auto-registers an experiment.

    Usage:
        @register_experiment("B5", category="B")
        class ExperimentB5(BaseExperiment):
            ...

    The experiment is then discoverable via get_experiment_class("B5")
    without manually editing __init__.py.
    """

    def decorator(cls):
        module = cls.__module__
        class_name = cls.__name__
        _AUTO_REGISTRY[experiment_id.upper()] = (module, class_name)
        # Attach metadata to the class for introspection
        cls._experiment_id = experiment_id.upper()
        cls._category = category
        return cls

    return decorator


def get_auto_registered() -> dict[str, tuple[str, str]]:
    """Return all auto-registered experiments."""
    return dict(_AUTO_REGISTRY)


# ── Structured Event Logging ─────────────────────────────────────────────────


@dataclass
class ExperimentEvent:
    """A single structured event during experiment execution.

    Events are machine-parseable and enable post-hoc analysis of
    experiment behavior (timing, convergence, failures).
    """

    timestamp: str
    event_type: str  # "phase_start", "phase_end", "vqe_converged", "vqe_failed", etc.
    experiment_id: str
    seed: int | None = None
    h_value: float | None = None
    data: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Clean numpy types
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

    Events are stored in-memory during execution and flushed to JSON
    alongside the experiment results.

    Usage:
        slog = StructuredLogger("A3")
        slog.log("vqe_start", seed=42, h_value=1.5)
        slog.log("vqe_converged", seed=42, h_value=1.5,
                 data={"energy": -5.2, "n_evals": 120, "n_restarts": 3})
        slog.log("vqe_failed", seed=42, h_value=1.0,
                 data={"reason": "maxiter reached"})
        events = slog.get_events()
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

        summary = {}
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


# ── Run Summary (post-hoc analysis) ─────────────────────────────────────────


@dataclass
class RunSummary:
    """Aggregated summary of an experiment run for quick comparison.

    Generated automatically after analyze() and stored in the result JSON.
    Enables fast querying without loading full results.
    """

    experiment_id: str
    run_id: str
    timestamp: str
    status: str  # "success", "partial", "failed"

    # Core metrics
    n_seeds: int = 0
    n_seeds_passed: int = 0
    n_total_points: int = 0
    n_points_passing: int = 0

    # Accuracy
    mean_de_gap: float = 0.0
    std_de_gap: float = 0.0
    best_de_gap: float = float("inf")
    worst_de_gap: float = 0.0
    pass_rate: float = 0.0

    # Timing
    total_time_s: float = 0.0
    mean_time_per_point_s: float = 0.0

    # Convergence
    convergence_rate: float = 1.0  # Fraction of VQE runs that converged
    mean_evaluations: float = 0.0

    # Comparison
    vs_baseline_improvement_pct: float = 0.0
    vs_baseline_verdict: str = "neutral"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_analysis(
        cls,
        experiment_id: str,
        run_id: str,
        analysis: dict[str, Any],
        results: dict[int, list] | None = None,
    ) -> RunSummary:
        """Build RunSummary from analyze() output."""
        summary = analysis.get("summary", {})
        per_seed = analysis.get("per_seed", {})

        n_seeds = len(per_seed)
        n_seeds_passed = sum(
            1 for s in per_seed.values() if s.get("n_passing", 0) == s.get("n_total", 0)
        )

        # Determine status
        if "error" in summary:
            status = "failed"
        elif n_seeds_passed == n_seeds:
            status = "success"
        else:
            status = "partial"

        # Convergence stats from results
        convergence_rate = 1.0
        mean_evals = 0.0
        if results:
            all_metrics = [m for ms in results.values() for m in ms]
            if all_metrics:
                converged = [m for m in all_metrics if m.converged]
                convergence_rate = len(converged) / len(all_metrics)
                evals = [m.n_evaluations for m in all_metrics if m.n_evaluations > 0]
                mean_evals = float(np.mean(evals)) if evals else 0.0

        n_total = summary.get("n_total_points", 0)
        total_time = summary.get("total_time_s", 0)

        return cls(
            experiment_id=experiment_id,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            status=status,
            n_seeds=n_seeds,
            n_seeds_passed=n_seeds_passed,
            n_total_points=n_total,
            n_points_passing=int(summary.get("pass_rate", 0) * n_total),
            mean_de_gap=summary.get("mean_de_gap", 0),
            std_de_gap=summary.get("std_de_gap", 0),
            best_de_gap=summary.get("min_de_gap", float("inf"))
            if "min_de_gap" in summary
            else float("inf"),
            worst_de_gap=summary.get("max_de_gap", 0) if "max_de_gap" in summary else 0,
            pass_rate=summary.get("pass_rate", 0),
            total_time_s=total_time,
            mean_time_per_point_s=total_time / n_total if n_total > 0 else 0,
            convergence_rate=convergence_rate,
            mean_evaluations=mean_evals,
        )


def _json_default(obj):
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
