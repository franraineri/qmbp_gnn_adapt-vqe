"""Result storage and cross-experiment comparison.

Provides:
- Loading/querying results across experiments
- Baseline caching (reference results)
- Cross-experiment comparison
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default results root relative to working directory
_DEFAULT_RESULTS_ROOT = Path("results/experiments")


class ResultStore:
    """Query and compare experiment results."""

    def __init__(self, results_root: Path | None = None):
        self.root = results_root or _DEFAULT_RESULTS_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    def list_experiments(self) -> list[str]:
        """List all experiment IDs with saved results."""
        experiments: list[str] = []
        if not self.root.exists():
            return experiments
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and d.name.startswith("exp_"):
                experiments.append(d.name.replace("exp_", "").upper())
        return experiments

    def load_latest(self, experiment_id: str) -> dict[str, Any] | None:
        """Load the most recent result for an experiment."""
        exp_dir = self.root / f"exp_{experiment_id.lower()}"
        if not exp_dir.exists():
            return None

        runs = sorted(exp_dir.glob("run_*.json"), reverse=True)
        if not runs:
            return None

        with open(runs[0]) as f:
            return json.load(f)

    def load_all_runs(self, experiment_id: str) -> list[dict[str, Any]]:
        """Load all runs for an experiment (chronological)."""
        exp_dir = self.root / f"exp_{experiment_id.lower()}"
        if not exp_dir.exists():
            return []

        results: list[dict[str, Any]] = []
        for run_file in sorted(exp_dir.glob("run_*.json")):
            with open(run_file) as f:
                results.append(json.load(f))
        return results

    def get_baseline(self, system_key: str) -> dict[str, Any] | None:
        """Load cached baseline for a system configuration.

        Parameters
        ----------
        system_key : str
            Key like "n6_h1.5", "n10_h1.5", "n20_h2.0"
        """
        baselines_dir = self.root / "baselines"
        baselines_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = baselines_dir / f"baseline_{system_key}.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                return json.load(f)
        return None

    def save_baseline(self, system_key: str, data: dict[str, Any]) -> None:
        """Cache a baseline result."""
        baselines_dir = self.root / "baselines"
        baselines_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = baselines_dir / f"baseline_{system_key}.json"
        with open(baseline_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_baseline_de_gap(self, n_qubits: int, h_value: float) -> float:
        """Get baseline ΔE/gap for a system configuration.

        Tries cached baseline first, falls back to known reference values.
        """
        system_key = f"n{n_qubits}_h{h_value}"
        cached = self.get_baseline(system_key)
        if cached is not None:
            return cached.get("mean_de_gap", 0.03)

        # Known reference values from validated experiments
        _KNOWN_BASELINES: dict[tuple[int, float], float] = {
            (6, 1.5): 0.014,
            (10, 1.5): 0.027,
            (20, 2.0): 0.0175,
        }
        return _KNOWN_BASELINES.get((n_qubits, h_value), 0.03)

    def compare_experiments(
        self,
        experiment_ids: list[str],
        baseline_id: str = "v61",
    ) -> list[dict[str, Any]]:
        """Compare multiple experiments against a baseline.

        Returns list of comparison dicts with improvement metrics.
        """
        comparisons: list[dict[str, Any]] = []
        for exp_id in experiment_ids:
            result = self.load_latest(exp_id)
            if result is None:
                logger.warning(f"No results found for {exp_id}")
                continue

            analysis = result.get("analysis", {})
            summary = analysis.get("summary", {})
            if "mean_de_gap" not in summary:
                continue

            config = result.get("config", {})
            system = config.get("system", {})
            n = system.get("n_qubits", 6)
            h_test = system.get("h_test", [1.5])
            h_val = h_test[0] if h_test else 1.5

            baseline_de_gap = self.get_baseline_de_gap(n, h_val)
            exp_de_gap = summary["mean_de_gap"]

            improvement_pct = (
                (baseline_de_gap - exp_de_gap) / baseline_de_gap * 100
                if baseline_de_gap > 0
                else 0.0
            )

            if improvement_pct > 10:
                verdict = "improvement"
            elif improvement_pct < -10:
                verdict = "regression"
            else:
                verdict = "neutral"

            comparisons.append(
                {
                    "experiment_id": exp_id,
                    "baseline_id": baseline_id,
                    "system_desc": f"N={n}, h={h_val}",
                    "exp_de_gap": exp_de_gap,
                    "exp_de_gap_std": summary.get("std_de_gap", 0),
                    "baseline_de_gap": baseline_de_gap,
                    "improvement_pct": improvement_pct,
                    "n_seeds": analysis.get("n_seeds", 1),
                    "verdict": verdict,
                }
            )

        return comparisons

    def generate_comparison_table(self, comparisons: list[dict[str, Any]]) -> str:
        """Generate markdown comparison table."""
        lines = [
            "| Experiment | ΔE/gap | Baseline | Improvement | Verdict |",
            "|------------|--------|----------|-------------|---------|",
        ]
        for c in comparisons:
            emoji = {
                "improvement": "✅",
                "regression": "❌",
                "neutral": "➖",
            }[c["verdict"]]
            lines.append(
                f"| {c['experiment_id']} | "
                f"{c['exp_de_gap']:.4f}±{c['exp_de_gap_std']:.4f} | "
                f"{c['baseline_de_gap']:.4f} | "
                f"{c['improvement_pct']:+.1f}% | "
                f"{emoji} {c['verdict']} |"
            )
        return "\n".join(lines)
