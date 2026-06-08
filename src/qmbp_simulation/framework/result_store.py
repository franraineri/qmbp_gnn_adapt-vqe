"""Result storage, querying, and cross-experiment comparison.

Provides:
- Discovery and loading of experiment and pipeline results
- Experiment-aware evaluation (each experiment judged by its own criteria)
- Noisy/ZNE result analysis with group-by and correlation tools
- Baseline caching for reference values
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from qmbp_simulation.framework.criteria import compute_verdict

logger = logging.getLogger(__name__)

_DEFAULT_RESULTS_ROOT = Path("results/experiments")

# Experiment category mapping: category_name → list of experiment ID prefixes
CATEGORY_MAP: dict[str, list[str]] = {
    "optimization": ["B", "C3", "G4"],
    "scaling": ["A", "G3"],
    "landscape": ["F"],
    "predictor": ["C1", "D", "E3", "G1", "G2", "G5"],
    "hardware": [],
    "generalization": ["E4"],
    "mps_scaling": ["SCALE"],
}

# Known reference baselines from validated V6.1 experiments (ΔE/gap)
_KNOWN_BASELINES: dict[tuple[int, float], float] = {
    (6, 1.25): 0.030,
    (6, 1.5): 0.014,
    (6, 1.75): 0.008,
    (6, 2.0): 0.005,
    (10, 1.5): 0.027,
    (10, 2.0): 0.012,
    (20, 2.0): 0.0175,
    (40, 3.0): 0.0333,  # Viability test: COBYLA + aer_mps
}


class ResultStore:
    """Query, compare, and analyze experiment results.

    Supports three result types:
    - Experiment runs (from BaseExperiment): results/experiments/exp_<id>/run_*.json
    - Pipeline runs (from PipelineRunner): results/pipeline/pipeline_run_*.json
    - Noisy/ZNE results: results/experiments/exp_noisy_variants/*.json
    """

    def __init__(self, results_root: Path | None = None) -> None:
        self.root = results_root or _DEFAULT_RESULTS_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Discovery ────────────────────────────────────────────────────────

    def list_categories(self) -> dict[str, list[str]]:
        """List experiment categories and their ID prefixes.

        Returns
        -------
        dict[str, list[str]]
            Category name → list of experiment ID prefixes.
        """
        return dict(CATEGORY_MAP)

    def resolve_category(
        self,
        category: str,
        available: list[str] | None = None,
    ) -> list[str]:
        """Resolve a category name or letter to experiment IDs.

        Parameters
        ----------
        category : str
            Category name (e.g., "optimization") or letter prefix (e.g., "G").
        available : list[str] | None
            Available experiment IDs to filter against. If None, uses
            list_experiments().

        Returns
        -------
        list[str]
            Matching experiment IDs.
        """
        if available is None:
            available = self.list_experiments()

        cat_lower = category.lower()
        if cat_lower in CATEGORY_MAP:
            prefixes = CATEGORY_MAP[cat_lower]
            return [e for e in available if any(e.startswith(p) for p in prefixes)]

        # Try as a letter prefix
        prefix = category.upper()
        return [e for e in available if e.startswith(prefix)]

    # Known test artifact experiment IDs (from test_runner_base.py)
    _TEST_ARTIFACTS: set[str] = {
        "TEST",
        "FAIL",
        "XFAIL",
        "NONE",
        "CNT",
        "T1A",
        "T1B",
        "T1C",
    }

    def list_experiments(self, *, exclude_tests: bool = True) -> list[str]:
        """List experiment IDs that have at least one run_*.json file.

        Parameters
        ----------
        exclude_tests : bool
            If True (default), filter out known test artifacts.
        """
        experiments: list[str] = []
        if not self.root.exists():
            return experiments
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and d.name.startswith("exp_") and list(d.glob("run_*.json")):
                exp_id = d.name.replace("exp_", "").upper()
                if not exp_id:
                    continue  # Skip empty IDs
                if exclude_tests and exp_id in self._TEST_ARTIFACTS:
                    continue
                experiments.append(exp_id)
        return experiments

    def list_pipeline_runs(self) -> list[Path]:
        """List pipeline run files (most recent first)."""
        pipeline_dir = self.root.parent / "pipeline"
        if not pipeline_dir.exists():
            return []
        return sorted(pipeline_dir.glob("pipeline_run_*.json"), reverse=True)

    # ── Loading ──────────────────────────────────────────────────────────

    def load_latest(self, experiment_id: str) -> dict[str, Any] | None:
        """Load the most recent result for an experiment."""
        exp_dir = self.root / f"exp_{experiment_id.lower()}"
        if not exp_dir.exists():
            return None
        runs = sorted(exp_dir.glob("run_*.json"), reverse=True)
        if not runs:
            return None
        with open(runs[0]) as f:
            return json.load(f)  # type: ignore[no-any-return]

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

    def load_noisy_results(self, filename: str | None = None) -> list[dict[str, Any]]:
        """Load noisy/ZNE experiment results.

        Parameters
        ----------
        filename : str | None
            Specific file to load. If None, loads the most recent file
            that contains a "results" array.
        """
        noisy_dir = self.root / "exp_noisy_variants"
        if not noisy_dir.exists():
            return []

        if filename:
            path = noisy_dir / filename
            files = [path] if path.exists() else []
        else:
            files = sorted(noisy_dir.glob("*.json"), reverse=True)

        for f in files:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if isinstance(data.get("results"), list) and data["results"]:
                    return data["results"]  # type: ignore[no-any-return]
            except (json.JSONDecodeError, OSError):
                continue
        return []

    def load_zne_results(self) -> list[dict[str, Any]]:
        """Load GF-ZNE and PEA-ZNE results from all new-format experiments.

        Scans experiment directories matching the ZNE validation experiments
        and extracts per-h-point comparison data (gains, R², method).

        Returns
        -------
        list[dict]
            Per-h-point results with keys: experiment, topology, n_qubits, h,
            gf_gain, gf_r2, pea_gain, pea_r2, ces_gain, ces_r2.
        """
        zne_dirs = [
            "exp_gf_zne_cmp",
            "exp_zne_3way",
            "exp_pea_zne_val",
            "exp_pea_hw_ready",
            "exp_pea_pipeline",
            "exp_zne_cross_topology",
        ]

        all_points: list[dict[str, Any]] = []

        for dirname in zne_dirs:
            exp_dir = self.root / dirname
            if not exp_dir.exists():
                continue
            for f in sorted(exp_dir.glob("run_*.json")):
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                except (json.JSONDecodeError, OSError):
                    continue

                config = data.get("config", {})
                system = config.get("system", {})
                topology = system.get("topology", "")
                n_qubits = system.get("n_qubits", 0)
                exp_id = config.get("experiment_id", dirname)

                # Extract comparison data from section_4 or section_5
                results_sections = data.get("results", {})
                comparison = self._extract_zne_comparison(results_sections)

                for row in comparison:
                    point: dict[str, Any] = {
                        "experiment": exp_id,
                        "file": f.name,
                        "topology": topology or row.get("topology", ""),
                        "n_qubits": n_qubits or row.get("n_qubits", 0),
                        "h": row.get("h", 0),
                    }
                    # GF data (multiple possible key patterns)
                    gf_gain = row.get("gf_zne_gain") or row.get("gf_gain") or row.get("gain_gf_zne")
                    if gf_gain is not None:
                        point["gf_gain"] = gf_gain
                        point["gf_r2"] = (
                            row.get("gf_zne_r2") or row.get("gf_r2") or row.get("r2_gf_zne")
                        )
                    # PEA data
                    pea_gain = (
                        row.get("pea_zne_gain") or row.get("pea_gain") or row.get("gain_pea_zne")
                    )
                    if pea_gain is not None:
                        point["pea_gain"] = pea_gain
                        point["pea_r2"] = (
                            row.get("pea_zne_r2") or row.get("pea_r2") or row.get("r2_pea_zne")
                        )
                    # CES data
                    ces_gain = (
                        row.get("ces_zne_gain") or row.get("ces_gain") or row.get("gain_ces_zne")
                    )
                    if ces_gain is not None:
                        point["ces_gain"] = ces_gain
                        point["ces_r2"] = (
                            row.get("ces_zne_r2") or row.get("ces_r2") or row.get("r2_ces_zne")
                        )

                    all_points.append(point)

        return all_points

    @staticmethod
    def _extract_zne_comparison(results_sections: dict) -> list[dict]:
        """Extract per-h comparison rows from ValidationRunner result sections."""
        # Try section_5, section_4 (comparison/verdict sections)
        for sec_key in ["section_5", "section_4"]:
            sec = results_sections.get(sec_key, {}).get("data", {})
            comparison = sec.get("comparison", [])
            if comparison:
                return comparison  # type: ignore[no-any-return]
            # Also check nested in summary
            summary = sec.get("summary", {})
            comparison = summary.get("comparison", [])
            if comparison:
                return comparison  # type: ignore[no-any-return]
        return []

    def analyze_zne_summary(self) -> dict[str, Any]:
        """Analyze all ZNE experiments and return consolidated summary.

        Returns
        -------
        dict with keys: n_evaluations, methods (CES/GF/PEA stats),
        by_topology, coverage_matrix, gaps.
        """
        points = self.load_zne_results()
        if not points:
            return {"n_evaluations": 0, "status": "no_data"}

        gf_gains = [p["gf_gain"] for p in points if "gf_gain" in p]
        pea_gains = [p["pea_gain"] for p in points if "pea_gain" in p]
        ces_gains = [p["ces_gain"] for p in points if "ces_gain" in p]
        gf_r2s = [p["gf_r2"] for p in points if p.get("gf_r2") is not None]
        pea_r2s = [p["pea_r2"] for p in points if p.get("pea_r2") is not None]

        summary: dict[str, Any] = {
            "n_evaluations": len(points),
            "methods": {},
        }

        if ces_gains:
            summary["methods"]["CES-ZNE"] = {
                "n": len(ces_gains),
                "mean_gain": float(np.mean(ces_gains)),
                "always_positive": sum(1 for g in ces_gains if g > 0) == len(ces_gains),
                "positive_rate": sum(1 for g in ces_gains if g > 0) / len(ces_gains),
            }
        if gf_gains:
            summary["methods"]["GF-ZNE"] = {
                "n": len(gf_gains),
                "mean_gain": float(np.mean(gf_gains)),
                "mean_r2": float(np.mean(gf_r2s)) if gf_r2s else None,
                "always_positive": sum(1 for g in gf_gains if g > 0) == len(gf_gains),
                "positive_rate": sum(1 for g in gf_gains if g > 0) / len(gf_gains),
            }
        if pea_gains:
            summary["methods"]["PEA-ZNE"] = {
                "n": len(pea_gains),
                "mean_gain": float(np.mean(pea_gains)),
                "mean_r2": float(np.mean(pea_r2s)) if pea_r2s else None,
                "always_positive": sum(1 for g in pea_gains if g > 0) == len(pea_gains),
                "positive_rate": sum(1 for g in pea_gains if g > 0) / len(pea_gains),
            }

        # By topology
        topologies = sorted(set(p["topology"] for p in points if p.get("topology")))
        summary["by_topology"] = {}
        for topo in topologies:
            topo_pts = [p for p in points if p["topology"] == topo]
            topo_gf = [p["gf_gain"] for p in topo_pts if "gf_gain" in p]
            topo_pea = [p["pea_gain"] for p in topo_pts if "pea_gain" in p]
            summary["by_topology"][topo] = {
                "n_points": len(topo_pts),
                "gf_mean_gain": float(np.mean(topo_gf)) if topo_gf else None,
                "pea_mean_gain": float(np.mean(topo_pea)) if topo_pea else None,
            }

        # Coverage gaps
        configs = sorted(set((p["topology"], p["n_qubits"]) for p in points if p["topology"]))
        gaps = []
        for topo, n in configs:
            cfg_pts = [p for p in points if p["topology"] == topo and p["n_qubits"] == n]
            if not any("pea_gain" in p for p in cfg_pts):
                gaps.append(f"PEA on {topo} N={n}")
            if not any("gf_gain" in p for p in cfg_pts):
                gaps.append(f"GF on {topo} N={n}")
        summary["gaps"] = gaps

        return summary

    # ── Baselines ────────────────────────────────────────────────────────

    def get_baseline_de_gap(self, n_qubits: int, h_value: float) -> float:
        """Get reference ΔE/gap for a system configuration."""
        system_key = f"n{n_qubits}_h{h_value}"
        baselines_dir = self.root / "baselines"
        baseline_path = baselines_dir / f"baseline_{system_key}.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                return json.load(f).get("mean_de_gap", 0.03)  # type: ignore[no-any-return]
        return _KNOWN_BASELINES.get((n_qubits, h_value), 0.03)

    def save_baseline(self, system_key: str, data: dict[str, Any]) -> None:
        """Cache a baseline result."""
        baselines_dir = self.root / "baselines"
        baselines_dir.mkdir(parents=True, exist_ok=True)
        with open(baselines_dir / f"baseline_{system_key}.json", "w") as f:
            json.dump(data, f, indent=2)

    # ── Experiment Comparison ────────────────────────────────────────────

    def compare_experiments(self, experiment_ids: list[str]) -> list[dict[str, Any]]:
        """Compare experiments using their own success criteria.

        Each experiment is evaluated against its hypothesis — not a blanket
        ΔE/gap baseline. Returns structured comparison with verdict.
        """
        comparisons: list[dict[str, Any]] = []
        for exp_id in experiment_ids:
            if not exp_id:
                continue  # Skip empty IDs

            result = self.load_latest(exp_id)
            if result is None:
                logger.warning(f"No results for {exp_id}")
                continue

            analysis = result.get("analysis", {})
            summary = analysis.get("summary", {})
            if not summary or "error" in summary:
                continue

            config = result.get("config", {})
            comparisons.append(self._evaluate_single(exp_id, config, summary, analysis))
        return comparisons

    def _evaluate_single(
        self,
        exp_id: str,
        config: dict[str, Any],
        summary: dict[str, Any],
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate one experiment against its criteria."""
        verdict, criteria_desc = compute_verdict(exp_id, summary)

        return {
            "experiment_id": exp_id,
            "category": config.get("category", exp_id[0] if exp_id else ""),
            "hypothesis": config.get("hypothesis", "N/A"),
            "criteria": criteria_desc,
            "n_seeds": analysis.get("n_seeds", 1),
            "mean_de_gap": summary.get("mean_de_gap"),
            "std_de_gap": summary.get("std_de_gap", 0),
            "pass_rate": summary.get("pass_rate"),
            "total_time_s": summary.get("total_time_s", 0),
            "verdict": verdict,
        }

    # ── Noisy / ZNE Analysis ─────────────────────────────────────────────

    def analyze_noisy_by_group(
        self,
        results: list[dict[str, Any]],
        group_key: str,
    ) -> dict[Any, dict[str, float]]:
        """Group noisy results by a key and compute aggregate statistics.

        Parameters
        ----------
        results : list[dict]
            Per-evaluation result dicts (e.g., from ZNE robustness).
        group_key : str
            Key to group by (e.g., "seed_layout", "n_layouts", "h_test").

        Returns
        -------
        dict mapping group_value → {mean_r2, std_r2, mean_gain, helps_rate, n}
        """
        groups: dict[Any, list[dict]] = {}
        for r in results:
            val = r.get(group_key)
            if val is not None:
                groups.setdefault(val, []).append(r)

        out: dict[Any, dict[str, float]] = {}
        for val, group in sorted(groups.items(), key=lambda x: str(x[0])):
            r2s = np.array([r.get("r2", 0) for r in group])
            gains = np.array([r.get("gain", 0) for r in group])
            out[val] = {
                "mean_r2": float(np.mean(r2s)),
                "std_r2": float(np.std(r2s)),
                "mean_gain": float(np.mean(gains)),
                "helps_rate": float(np.mean(gains > 0)),
                "n": len(group),
            }
        return out

    def analyze_noisy_correlations(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Compute key correlations and summary stats for noisy results."""
        if len(results) < 3:
            return {}

        r2_arr = np.array([r.get("r2", 0) for r in results])
        gain_arr = np.array([r.get("gain", 0) for r in results])

        stats: dict[str, float] = {
            "n_evaluations": len(results),
            "mean_r2": float(np.mean(r2_arr)),
            "pct_r2_gt_08": float(np.mean(r2_arr > 0.8) * 100),
            "pct_helps": float(np.mean(gain_arr > 0) * 100),
            "mean_gain_pct": float(np.mean(gain_arr) * 100),
        }

        # R² vs gain correlation
        if np.std(r2_arr) > 1e-10 and np.std(gain_arr) > 1e-10:
            stats["corr_r2_gain"] = float(np.corrcoef(r2_arr, gain_arr)[0, 1])

        # CES ratio vs R²
        ratios = []
        for r in results:
            ces_range = r.get("ces_range", [])
            if len(ces_range) == 2 and ces_range[0] > 0:
                ratios.append(ces_range[1] / ces_range[0])
        if len(ratios) == len(results):
            ratio_arr = np.array(ratios)
            if np.std(ratio_arr) > 1e-10:
                stats["corr_ces_ratio_r2"] = float(np.corrcoef(ratio_arr, r2_arr)[0, 1])

        return stats

    # ── Formatting ───────────────────────────────────────────────────────

    def format_experiment_table(self, comparisons: list[dict[str, Any]]) -> str:
        """Format experiment comparison as aligned text table."""
        if not comparisons:
            return "No results to compare."

        lines = [
            f"{'Exp':<4} {'Cat':<4} {'Verdict':<12} {'ΔE/gap':<16} {'Pass%':<7} {'Criteria'}",
            "-" * 80,
        ]
        for c in comparisons:
            emoji = {"confirmed": "✅", "rejected": "⚠️", "failed": "❌"}.get(c["verdict"], "?")
            de = (
                f"{c['mean_de_gap']:.4f}±{c['std_de_gap']:.4f}"
                if c["mean_de_gap"] is not None
                else "N/A"
            )
            pr = f"{c['pass_rate'] * 100:.0f}%" if c["pass_rate"] is not None else "N/A"
            lines.append(
                f"{c['experiment_id']:<4} {c['category']:<4} "
                f"{emoji} {c['verdict']:<9} {de:<16} {pr:<7} {c['criteria']}"
            )
        return "\n".join(lines)

    def format_noisy_table(
        self,
        grouped: dict[Any, dict[str, float]],
        group_key: str,
    ) -> str:
        """Format grouped noisy analysis as aligned text table."""
        lines = [
            f"  {'Value':<12} {'R²':<14} {'Gain':<10} {'Helps':<8} {'N'}",
            f"  {'-' * 50}",
        ]
        for val, s in grouped.items():
            lines.append(
                f"  {str(val):<12} "
                f"{s['mean_r2']:.4f}±{s['std_r2']:.4f}  "
                f"{s['mean_gain'] * 100:+5.1f}%   "
                f"{s['helps_rate'] * 100:4.0f}%    "
                f"{s['n']}"
            )
        return "\n".join(lines)
