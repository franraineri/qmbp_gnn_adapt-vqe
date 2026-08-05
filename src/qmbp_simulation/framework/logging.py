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

from qmbp_simulation.utils.helpers import json_serialize


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
            json.dump(data, f, indent=2, default=json_serialize)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for embedding in result JSON."""
        return {
            "n_events": len(self.events),
            "timing_summary": self.get_timing_summary(),
            "failure_summary": self.get_failure_summary(),
            "events": [e.to_dict() for e in self.events],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Progress Reporting
# ─────────────────────────────────────────────────────────────────────────────


class ProgressReporter:
    """Standardized progress reporting with phase tracking.

    Provides consistent console output for multi-phase pipeline execution.
    Tracks elapsed time per phase and produces a final summary.

    Usage:
        reporter = ProgressReporter("Pipeline N=6")
        with reporter.phase(1, "Exact diagonalization") as p:
            do_phase1()
            p.detail("17 h-points computed")
        with reporter.phase(2, "VQE optimization") as p:
            do_phase2()
            p.detail("mean fidelity = 0.98")
        reporter.summary({"total_points": 17, "mean_de_gap": 0.014})
    """

    def __init__(self, title: str = "", width: int = 60) -> None:
        self._title = title
        self._width = width
        self._phases: list[dict[str, Any]] = []
        self._t0 = time.time()

        if title:
            print("=" * width)
            print(f"  {title}")
            print("=" * width)

    def phase(self, phase_num: int, description: str) -> _PhaseContext:
        """Start a named phase with timing.

        Parameters
        ----------
        phase_num : int
            Phase number (for display).
        description : str
            Human-readable phase description.

        Returns
        -------
        _PhaseContext
            Context manager that tracks elapsed time.
        """
        return _PhaseContext(self, phase_num, description)

    def _record_phase(self, phase_num: int, description: str, elapsed_s: float) -> None:
        """Record a completed phase (called by _PhaseContext)."""
        self._phases.append(
            {
                "phase": phase_num,
                "description": description,
                "elapsed_s": elapsed_s,
            }
        )

    def checkpoint(self, label: str, value: str = "") -> None:
        """Print a checkpoint message.

        Parameters
        ----------
        label : str
            Checkpoint label.
        value : str
            Optional value to display.
        """
        if value:
            print(f"    {label}: {value}")
        else:
            print(f"    {label}")

    def summary(self, metrics: dict[str, Any] | None = None) -> None:
        """Print final summary with timing breakdown.

        Parameters
        ----------
        metrics : dict | None
            Key metrics to display in the summary.
        """
        total_elapsed = time.time() - self._t0
        print()
        print("=" * self._width)
        print(f"  Complete in {total_elapsed:.1f}s")
        print("=" * self._width)

        if self._phases:
            for p in self._phases:
                print(f"    Phase {p['phase']}: {p['description']} ({p['elapsed_s']:.1f}s)")

        if metrics:
            print()
            for key, val in metrics.items():
                if isinstance(val, float):
                    print(f"    {key}: {val:.4f}")
                else:
                    print(f"    {key}: {val}")

    @property
    def total_elapsed_s(self) -> float:
        """Total elapsed time since reporter creation."""
        return time.time() - self._t0


class _PhaseContext:
    """Context manager for a single phase within ProgressReporter."""

    def __init__(self, reporter: ProgressReporter, phase_num: int, description: str) -> None:
        self._reporter = reporter
        self._phase_num = phase_num
        self._description = description
        self._start: float = 0.0
        self.elapsed_s: float = 0.0

    def __enter__(self) -> _PhaseContext:
        print(f"\n  Phase {self._phase_num}: {self._description}...")
        self._start = time.time()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.elapsed_s = time.time() - self._start
        print(f"    Done in {self.elapsed_s:.1f}s")
        self._reporter._record_phase(self._phase_num, self._description, self.elapsed_s)

    def detail(self, message: str) -> None:
        """Print a detail message within the phase.

        Parameters
        ----------
        message : str
            Detail to display (indented under the phase).
        """
        print(f"    {message}")
