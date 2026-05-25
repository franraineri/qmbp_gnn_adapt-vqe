"""Unified result storage and cross-experiment comparison engine.

Provides:
- Loading/querying results across experiments
- Baseline caching (V6.1 reference results)
- Cross-experiment comparison tables
- Markdown report generation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from scripts.experiments_v8.core.metrics import ComparisonResult

logger = logging.getLogger(__name__)

# Navigate: core/result_store.py -> core/ -> experiments_v8/ -> results/
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
BASELINES_DIR = RESULTS_ROOT / "baselines"


class ResultStore:
    """Query and compare experiment results."""

    def __init__(self, results_root: Path | None = None):
        self.root = results_root or RESULTS_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    def list_experiments(self) -> list[str]:
        """List all experiment IDs with saved results."""
        experiments = []
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

        results = []
        for run_file in sorted(exp_dir.glob("run_*.json")):
            with open(run_file) as f:
                results.append(json.load(f))
        return results

    def get_baseline(self, system_key: str) -> dict[str, Any] | None:
        """Load cached V6.1 baseline for a system configuration.

        Parameters
        ----------
        system_key : str
            Key like "n6_h1.5", "n10_h1.5", "n20_h2.0"
        """
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        baseline_path = BASELINES_DIR / f"baseline_{system_key}.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                return json.load(f)
        return None

    def save_baseline(self, system_key: str, data: dict[str, Any]) -> None:
        """Cache a baseline result."""
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        baseline_path = BASELINES_DIR / f"baseline_{system_key}.json"
        with open(baseline_path, "w") as f:
            json.dump(data, f, indent=2)

    def compare_experiments(
        self,
        experiment_ids: list[str],
        baseline_id: str = "v61",
    ) -> list[ComparisonResult]:
        """Compare multiple experiments against a baseline.

        Returns list of ComparisonResult objects.
        """
        comparisons = []
        for exp_id in experiment_ids:
            result = self.load_latest(exp_id)
            if result is None:
                logger.warning(f"No results found for {exp_id}")
                continue

            analysis = result.get("analysis", {})
            summary = analysis.get("summary", {})
            if "mean_de_gap" not in summary:
                continue

            # Build comparison (simplified — uses summary stats)
            comp = ComparisonResult(
                experiment_id=exp_id,
                baseline_id=baseline_id,
                system_desc=self._extract_system_desc(result),
                exp_de_gap=summary["mean_de_gap"],
                baseline_de_gap=self._get_baseline_de_gap(result),
                improvement_pct=0.0,  # Computed below
                exp_time_s=summary.get("total_time_s", 0),
                baseline_time_s=0.0,
                speedup=1.0,
                exp_de_gap_std=summary.get("std_de_gap", 0),
                n_seeds=analysis.get("n_seeds", 1),
            )

            # Compute improvement
            if comp.baseline_de_gap > 0:
                comp.improvement_pct = (
                    (comp.baseline_de_gap - comp.exp_de_gap) / comp.baseline_de_gap * 100
                )

            # Verdict
            if comp.improvement_pct > 10:
                comp.verdict = "improvement"
            elif comp.improvement_pct < -10:
                comp.verdict = "regression"
            else:
                comp.verdict = "neutral"

            comparisons.append(comp)

        return comparisons

    def generate_comparison_table(self, comparisons: list[ComparisonResult]) -> str:
        """Generate markdown comparison table."""
        lines = [
            "| Experiment | ΔE/gap | Baseline | Improvement | Speedup | Verdict |",
            "|------------|--------|----------|-------------|---------|---------|",
        ]
        for c in comparisons:
            emoji = {"improvement": "✅", "regression": "❌", "neutral": "➖"}[c.verdict]
            lines.append(
                f"| {c.experiment_id} | {c.exp_de_gap:.4f}±{c.exp_de_gap_std:.4f} | "
                f"{c.baseline_de_gap:.4f} | {c.improvement_pct:+.1f}% | "
                f"{c.speedup:.1f}× | {emoji} {c.verdict} |"
            )
        return "\n".join(lines)

    # ── Private ──────────────────────────────────────────────────────────────

    def _extract_system_desc(self, result: dict) -> str:
        """Extract system description from result config."""
        config = result.get("config", {})
        system = config.get("system", {})
        return f"N={system.get('n_qubits', '?')}, h={system.get('h_test', ['?'])}"

    def _get_baseline_de_gap(self, result: dict) -> float:
        """Get baseline DE/gap for the system in this result.

        Attempts to load from cached baseline file first, falls back to
        known V6.1 reference values from project-status.md.
        """
        config = result.get("config", {})
        system = config.get("system", {})
        n = system.get("n_qubits", 6)
        h_test = system.get("h_test", [1.5])
        h_val = h_test[0] if h_test else 1.5

        # Try loading from cached baseline file
        system_key = f"n{n}_h{h_val}"
        cached = self.get_baseline(system_key)
        if cached is not None:
            return cached.get("mean_de_gap", 0.03)

        # Fallback: known V6.1 reference values (from RESULTS_SUMMARY_V61_V7.md)
        _KNOWN_BASELINES = {
            (6, 1.5): 0.014,  # N=6, h=1.5: 1.4%
            (10, 1.5): 0.027,  # N=10, h=1.5: 2.7%
            (20, 2.0): 0.0175,  # N=20, h=2.0: 1.75%
        }
        return _KNOWN_BASELINES.get((n, h_val), 0.03)
