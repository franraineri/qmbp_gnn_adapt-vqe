"""Mitigation Benchmark Analyzer.

Post-execution analysis pipeline for the mitigation benchmark.
Consolidates all result JSONs, validates entries, computes derived metrics,
generates thesis figures, and exports ranked comparison tables.

Stages:
    1. Scan — recursively load all result JSONs from results/mitigation_benchmark/
    2. Validate — check required fields, filter corrupted entries
    3. Derive — compute improvement_vs_raw, overhead_factor, precision_per_shot, net_benefit
    4. Rank — sort configs by mean_delta_e_gap, compute Pareto frontier
    5. Ablate — measure contribution of each technique individually
    6. Transfer — compute sim↔hardware transfer ratios and Spearman correlation
    7. Sensitivity — PEA budget and twirling curves with saturation points
    8. Hypotheses — validate H1-H19 against observed data
    9. Export — comparison_table.json, ablation_study.json, figures, LaTeX tables

Scans:
    results/mitigation_benchmark/{fake_backend,hardware}/{config_id}/*.json

Usage:
    python -m project_health.analysis.mitigation_benchmark_analyzer
    python -m project_health.analysis.mitigation_benchmark_analyzer --thesis-table
    python -m project_health.analysis.mitigation_benchmark_analyzer --figures
    python -m project_health.analysis.mitigation_benchmark_analyzer --statistical
    python -m project_health.analysis.mitigation_benchmark_analyzer --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_RESULTS_DIR = Path("results/mitigation_benchmark")

# Required top-level sections in a ResultEnvelope
REQUIRED_SECTIONS = ("benchmark_metadata", "circuit_stats", "results")

# Required keys within benchmark_metadata
REQUIRED_METADATA_KEYS = ("config_id", "h_value", "execution_mode")

# Required keys within results (at least one energy metric + delta_e_gap)
REQUIRED_RESULTS_KEYS = ("delta_e_gap",)

# Directories/files to skip during scan
SKIP_DIRS = {"configs", "analysis", "__pycache__"}
SKIP_FILES = {"manifest.json"}


# ═══════════════════════════════════════════════════════════════════════════════
# Hypothesis Map — H1-H19 linking config pairs to expected direction
# ═══════════════════════════════════════════════════════════════════════════════

HYPOTHESIS_MAP: dict[str, dict] = {
    "H1": {
        "description": "DD reduces raw error vs no mitigation",
        "config_pair": ("C0_raw", "C1_dd_only"),
        "direction": "lower",
        "metric": "delta_e_gap",
    },
    "H2": {
        "description": "Twirling + TREX improves over DD alone",
        "config_pair": ("C1_dd_only", "C2_dd_tw"),
        "direction": "lower",
        "metric": "delta_e_gap",
    },
    "H3": {
        "description": "Gate-folding ZNE improves over DD+Tw+TREX",
        "config_pair": ("C2_dd_tw", "C3_full_gf"),
        "direction": "lower",
        "metric": "delta_e_gap",
    },
    "H4": {
        "description": "PEA-ZNE superior to gate-folding ZNE",
        "config_pair": ("C3_full_gf", "C5_full_pea_balanced"),
        "direction": "lower",
        "metric": "delta_e_gap",
    },
    "H5": {
        "description": "Heavier PEA budget improves precision",
        "config_pair": ("C4_full_pea_light", "C6_full_pea_heavy"),
        "direction": "lower",
        "metric": "delta_e_gap",
    },
    "H6": {
        "description": "DD beneficial with PEA (vs no-DD PEA)",
        "config_pair": ("C7_pea_no_dd", "C5_full_pea_balanced"),
        "direction": "lower",
        "metric": "delta_e_gap",
    },
    "H7": {
        "description": "XY4 DD comparable to XpXm",
        "config_pair": ("C5_full_pea_balanced", "C8_pea_xy4"),
        "direction": "lower",
        "metric": "delta_e_gap",
    },
    "H8": {
        "description": "Affine never worsens (even without ZNE)",
        "config_pair": ("C15_pea_no_affine", "C5_full_pea_balanced"),
        "direction": "lower",
        "metric": "delta_e_gap",
    },
    "H9": {
        "description": "GNN-QEM after PEA does not help (regression)",
        "config_pair": ("C5_full_pea_balanced", "C10_kitchen_sink"),
        "direction": "higher",
        "metric": "delta_e_gap",
    },
    "H10": {
        "description": "Mitiq ZNE competitive with IBM gate-folding",
        "config_pair": ("C3_full_gf", "C11_mitiq_zne"),
        "direction": "lower",
        "metric": "delta_e_gap",
    },
    "H11": {
        "description": "Mitiq CDR provides meaningful correction",
        "config_pair": ("C0_raw", "C12_mitiq_cdr"),
        "direction": "lower",
        "metric": "delta_e_gap",
    },
    "H12": {
        "description": "DDD+ZNE composition beneficial",
        "config_pair": ("C11_mitiq_zne", "C13_mitiq_ddd_zne"),
        "direction": "lower",
        "metric": "delta_e_gap",
    },
    "H13": {
        "description": "DD+Tw improves Mitiq CDR over bare CDR",
        "config_pair": ("C12_mitiq_cdr", "C14_dd_mitiq_cdr"),
        "direction": "lower",
        "metric": "delta_e_gap",
    },
    "H14": {
        "description": "PEA saturates at balanced budget",
        "config_pair": ("C5_full_pea_balanced", "C6_full_pea_heavy"),
        "direction": "lower",
        "metric": "delta_e_gap",
    },
    "H15": {
        "description": "AQC compression preserves accuracy with PEA",
        "config_pair": ("C5_full_pea_balanced", "C16_aqc_pea"),
        "direction": "lower",
        "metric": "delta_e_gap",
    },
    "H16": {
        "description": "AQC + Mitiq CDR viable alternative",
        "config_pair": ("C12_mitiq_cdr", "C17_aqc_mitiq_cdr"),
        "direction": "lower",
        "metric": "delta_e_gap",
    },
    "H17": {
        "description": "AQC alone (no ZNE) still beneficial over raw",
        "config_pair": ("C0_raw", "C18_aqc_raw"),
        "direction": "lower",
        "metric": "delta_e_gap",
    },
    "H18": {
        "description": "Balanced PEA achieves ΔE/gap < 3%",
        "config_pair": ("C5_full_pea_balanced",),
        "direction": "threshold",
        "threshold": 0.03,
        "metric": "delta_e_gap",
    },
    "H19": {
        "description": "Top config achieves correct phase labels",
        "config_pair": ("C5_full_pea_balanced",),
        "direction": "accuracy",
        "threshold": 1.0,
        "metric": "correct_label_rate",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# MitigationBenchmarkAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════


class MitigationBenchmarkAnalyzer:
    """Post-execution analysis pipeline for the mitigation benchmark.

    Stages:
    1. Scan — recursively load all result JSONs from results/mitigation_benchmark/
    2. Validate — check required fields, filter corrupted entries
    3. Derive — compute improvement_vs_raw, overhead_factor, precision_per_shot, net_benefit
    4. Rank — sort configs by mean_delta_e_gap, compute ranking
    5. (Later tasks add: ablate, transfer, sensitivity, hypotheses, export)
    """

    def __init__(self, results_dir: Path | None = None):
        self.results_dir = results_dir or DEFAULT_RESULTS_DIR
        self.entries: list[dict] = []  # Valid result envelopes
        self.errors: list[dict] = []  # Entries with validation errors
        self.derived_metrics: dict[str, dict] = {}  # Keyed by config_id

    def scan(self) -> None:
        """Recursively load all JSON result files from results_dir.

        Walks the results directory, loads each .json file (excluding
        manifest.json and configs/ directory), validates the entry, and
        adds valid entries to self.entries. Invalid entries are logged
        to self.errors with reason and path.
        """
        self.entries.clear()
        self.errors.clear()

        if not self.results_dir.exists():
            logger.warning(f"Results directory not found: {self.results_dir}")
            return

        for json_path in sorted(self.results_dir.rglob("*.json")):
            # Skip excluded files and directories
            if json_path.name in SKIP_FILES:
                continue
            if any(skip_dir in json_path.parts for skip_dir in SKIP_DIRS):
                continue

            try:
                data = json.loads(json_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                self.errors.append(
                    {
                        "path": str(json_path),
                        "reason": f"Failed to parse JSON: {e}",
                    }
                )
                logger.debug(f"Skipping {json_path.name}: parse error: {e}")
                continue

            if not isinstance(data, dict):
                self.errors.append(
                    {
                        "path": str(json_path),
                        "reason": "Top-level JSON is not a dict",
                    }
                )
                continue

            if self._validate_entry(data, json_path):
                # Attach source path for traceability
                data["_source_path"] = str(json_path)
                self.entries.append(data)

    def _validate_entry(self, entry: dict, path: Path) -> bool:
        """Validate a single result envelope has required fields.

        Checks:
        - Required top-level sections: benchmark_metadata, circuit_stats, results
        - Required metadata keys: config_id, h_value, execution_mode
        - Required results keys: delta_e_gap
        - At least one energy metric: e_mitigated or e_raw

        Parameters
        ----------
        entry : dict
            Parsed JSON result envelope.
        path : Path
            Source file path (for error reporting).

        Returns
        -------
        bool
            True if valid, False if validation failed (error logged to self.errors).
        """
        # Check required top-level sections
        missing_sections = [s for s in REQUIRED_SECTIONS if s not in entry]
        if missing_sections:
            self.errors.append(
                {
                    "path": str(path),
                    "reason": f"Missing required sections: {missing_sections}",
                }
            )
            return False

        # Check required metadata keys
        metadata = entry["benchmark_metadata"]
        if not isinstance(metadata, dict):
            self.errors.append(
                {
                    "path": str(path),
                    "reason": "benchmark_metadata is not a dict",
                }
            )
            return False

        missing_meta = [k for k in REQUIRED_METADATA_KEYS if k not in metadata]
        if missing_meta:
            self.errors.append(
                {
                    "path": str(path),
                    "reason": f"Missing metadata keys: {missing_meta}",
                }
            )
            return False

        # Check required results keys
        results = entry["results"]
        if not isinstance(results, dict):
            self.errors.append(
                {
                    "path": str(path),
                    "reason": "results section is not a dict",
                }
            )
            return False

        missing_results = [k for k in REQUIRED_RESULTS_KEYS if k not in results]
        if missing_results:
            self.errors.append(
                {
                    "path": str(path),
                    "reason": f"Missing results keys: {missing_results}",
                }
            )
            return False

        # At least one energy metric must be present
        has_energy = ("e_mitigated" in results) or ("e_raw" in results)
        if not has_energy:
            self.errors.append(
                {
                    "path": str(path),
                    "reason": "No energy metric (e_mitigated or e_raw) in results",
                }
            )
            return False

        return True

    # ───────────────────────────────────────────────────────────────────────
    # Aggregation helpers
    # ───────────────────────────────────────────────────────────────────────

    def _entries_for_config(self, config_id: str) -> list[dict]:
        """Return all valid entries matching a given config_id."""
        return [
            e for e in self.entries if e.get("benchmark_metadata", {}).get("config_id") == config_id
        ]

    def _mean_metric_for_config(self, config_id: str, metric: str = "delta_e_gap") -> float | None:
        """Compute mean of a metric across all h-values for a config.

        Returns None if no entries found for that config_id.
        """
        entries = self._entries_for_config(config_id)
        if not entries:
            return None

        if metric == "correct_label_rate":
            labels = [
                e["results"].get("correct_label")
                for e in entries
                if "results" in e and "correct_label" in e.get("results", {})
            ]
            if not labels:
                return None
            return sum(1 for lb in labels if lb) / len(labels)

        values = [
            e["results"][metric]
            for e in entries
            if "results" in e
            and metric in e.get("results", {})
            and e["results"][metric] is not None
        ]
        if not values:
            return None
        return mean(values)

    def _shot_noise_estimate(self, config_id: str) -> float:
        """Estimate shot noise level for a config.

        Uses std across h-values if >= 3 data points available,
        otherwise falls back to 1/sqrt(shots) approximation.
        """
        entries = self._entries_for_config(config_id)
        values = [
            e["results"]["delta_e_gap"]
            for e in entries
            if "results" in e
            and "delta_e_gap" in e.get("results", {})
            and e["results"]["delta_e_gap"] is not None
        ]
        if len(values) >= 3:
            return stdev(values)

        # Fallback: 1/sqrt(shots) approximation
        for e in entries:
            shots = e.get("shots")
            if shots and shots > 0:
                return 1.0 / math.sqrt(shots)

        # Ultimate fallback: conservative default
        return 1.0 / math.sqrt(16384)

    # ───────────────────────────────────────────────────────────────────────
    # Hypothesis validation (Requirement 18)
    # ───────────────────────────────────────────────────────────────────────

    def compute_hypothesis_verdicts(self) -> list[dict]:
        """Evaluate all 19 hypotheses using HYPOTHESIS_MAP.

        Each hypothesis maps to:
          - config_pair: tuple of config_ids to compare
          - direction: "lower"|"higher"|"threshold"|"accuracy"
          - metric: field to compare (default: "delta_e_gap")

        Verdict logic for paired comparisons (direction="lower" or "higher"):
          - CONFIRMED: observed matches expected direction beyond shot_noise_estimate
          - REFUTED: observed opposite to expected direction beyond shot_noise_estimate
          - INCONCLUSIVE: |difference| < shot_noise_estimate

        For threshold hypotheses (direction="threshold"):
          - CONFIRMED: metric value < threshold
          - REFUTED: metric value >= threshold beyond noise
          - INCONCLUSIVE: metric ≈ threshold within noise

        For accuracy hypotheses (direction="accuracy"):
          - CONFIRMED: correct_label_rate >= threshold
          - REFUTED: correct_label_rate < threshold
          - INCONCLUSIVE: insufficient data

        Returns list of dicts with: hypothesis_id, description, configs_tested,
            observed_delta, verdict.
        """
        verdicts: list[dict] = []

        for h_id, h_def in HYPOTHESIS_MAP.items():
            config_pair = h_def["config_pair"]
            direction = h_def["direction"]
            metric = h_def.get("metric", "delta_e_gap")
            description = h_def["description"]

            # --- Single-config hypotheses (threshold / accuracy) ---
            if direction in ("threshold", "accuracy"):
                config_id = config_pair[0]
                value = self._mean_metric_for_config(config_id, metric)

                if value is None:
                    verdicts.append(
                        {
                            "hypothesis_id": h_id,
                            "description": description,
                            "configs_tested": list(config_pair),
                            "observed_delta": None,
                            "verdict": "INCONCLUSIVE",
                        }
                    )
                    continue

                threshold = h_def["threshold"]
                noise = self._shot_noise_estimate(config_id)

                if direction == "threshold":
                    if value < threshold - noise:
                        verdict = "CONFIRMED"
                    elif value > threshold + noise:
                        verdict = "REFUTED"
                    else:
                        verdict = "INCONCLUSIVE"
                    observed_delta = value
                else:  # accuracy
                    if value >= threshold:
                        verdict = "CONFIRMED"
                    elif value < threshold - noise:
                        verdict = "REFUTED"
                    else:
                        verdict = "INCONCLUSIVE"
                    observed_delta = value

                verdicts.append(
                    {
                        "hypothesis_id": h_id,
                        "description": description,
                        "configs_tested": list(config_pair),
                        "observed_delta": observed_delta,
                        "verdict": verdict,
                    }
                )
                continue

            # --- Paired-config hypotheses (lower / higher) ---
            config_a, config_b = config_pair[0], config_pair[1]
            mean_a = self._mean_metric_for_config(config_a, metric)
            mean_b = self._mean_metric_for_config(config_b, metric)

            if mean_a is None or mean_b is None:
                verdicts.append(
                    {
                        "hypothesis_id": h_id,
                        "description": description,
                        "configs_tested": [config_a, config_b],
                        "observed_delta": None,
                        "verdict": "INCONCLUSIVE",
                    }
                )
                continue

            # delta = mean_a - mean_b (positive means B is better/lower)
            observed_delta = mean_a - mean_b
            noise = max(
                self._shot_noise_estimate(config_a),
                self._shot_noise_estimate(config_b),
            )

            if direction == "lower":
                # Expect config_b to have LOWER metric than config_a
                # i.e., observed_delta > 0 means B is lower (confirmation)
                if observed_delta > noise:
                    verdict = "CONFIRMED"
                elif observed_delta < -noise:
                    verdict = "REFUTED"
                else:
                    verdict = "INCONCLUSIVE"
            elif direction == "higher":
                # Expect config_b to have HIGHER metric than config_a
                # i.e., observed_delta < 0 means B is higher (confirmation)
                if observed_delta < -noise:
                    verdict = "CONFIRMED"
                elif observed_delta > noise:
                    verdict = "REFUTED"
                else:
                    verdict = "INCONCLUSIVE"
            else:
                verdict = "INCONCLUSIVE"

            verdicts.append(
                {
                    "hypothesis_id": h_id,
                    "description": description,
                    "configs_tested": [config_a, config_b],
                    "observed_delta": observed_delta,
                    "verdict": verdict,
                }
            )

        return verdicts

    # ═══════════════════════════════════════════════════════════════════════════
    # Derived Metrics (Stage 3)
    # ═══════════════════════════════════════════════════════════════════════════

    def compute_derived_metrics(self) -> dict[str, dict]:
        """Compute derived comparison metrics per config_id.

        For each config_id found in self.entries, computes:
          - mean_delta_e_gap: average delta_e_gap across h-values
          - improvement_vs_raw: (E_raw - E_mitigated) / (E_raw - E_exact), mean
          - overhead_factor: total_shots / baseline_shots (C0_raw shots)
          - precision_per_shot: (1 - delta_e_gap) / total_shots
          - net_benefit: fidelity_estimate × (1 - delta_e_gap)

        Results are cached in self.derived_metrics (dict keyed by config_id).
        If C0_raw baseline is missing, improvement_vs_raw and overhead_factor
        are omitted (set to None) and a warning is logged.

        Returns
        -------
        dict[str, dict]
            Mapping config_id → {mean_delta_e_gap, improvement_vs_raw,
            overhead_factor, precision_per_shot, net_benefit, n_entries}.
        """
        if self.derived_metrics:
            return self.derived_metrics

        # Group entries by config_id
        by_config: dict[str, list[dict]] = defaultdict(list)
        for entry in self.entries:
            config_id = entry["benchmark_metadata"]["config_id"]
            by_config[config_id].append(entry)

        # Find C0_raw baseline shots for overhead computation
        c0_shots: float | None = None
        has_baseline = "C0_raw" in by_config
        if has_baseline:
            c0_entries = by_config["C0_raw"]
            shots_list = [self._get_shots(e) for e in c0_entries]
            valid_shots = [s for s in shots_list if s > 0]
            c0_shots = mean(valid_shots) if valid_shots else None
        else:
            logger.warning(
                "C0_raw baseline not found in results — "
                "improvement_vs_raw and overhead_factor will be omitted"
            )

        self.derived_metrics = {}
        for config_id, entries in by_config.items():
            delta_gaps = [
                e["results"]["delta_e_gap"]
                for e in entries
                if e["results"].get("delta_e_gap") is not None
            ]
            mean_gap = mean(delta_gaps) if delta_gaps else None

            # improvement_vs_raw — only when C0_raw baseline is present
            mean_improvement: float | None = None
            if has_baseline:
                improvements: list[float] = []
                for e in entries:
                    r = e["results"]
                    e_raw = r.get("e_raw")
                    e_mit = r.get("e_mitigated")
                    e_exact = r.get("e_exact")
                    if e_raw is not None and e_mit is not None and e_exact is not None:
                        denom = e_raw - e_exact
                        if abs(denom) > 1e-12:
                            improvements.append((e_raw - e_mit) / denom)
                mean_improvement = mean(improvements) if improvements else None

            # overhead_factor — only when C0_raw baseline is present
            overhead: float | None = None
            if has_baseline and c0_shots is not None and c0_shots > 0:
                config_shots = [self._get_shots(e) for e in entries]
                valid_config_shots = [s for s in config_shots if s > 0]
                if valid_config_shots:
                    mean_shots = sum(valid_config_shots) / len(valid_config_shots)
                    overhead = mean_shots / c0_shots

            # precision_per_shot
            precisions: list[float] = []
            for e in entries:
                deg = e["results"].get("delta_e_gap")
                total_shots = self._get_shots(e)
                if deg is not None and total_shots > 0:
                    precisions.append((1.0 - deg) / total_shots)
            precision = mean(precisions) if precisions else None

            # net_benefit: fidelity_estimate × (1 - delta_e_gap)
            fidelities = [e.get("circuit_stats", {}).get("fidelity_estimate") for e in entries]
            fidelities = [f for f in fidelities if f is not None]
            if fidelities and mean_gap is not None:
                net_ben = mean(fidelities) * (1.0 - mean_gap)
            else:
                net_ben = None

            self.derived_metrics[config_id] = {
                "config_id": config_id,
                "mean_delta_e_gap": mean_gap,
                "improvement_vs_raw": mean_improvement,
                "overhead_factor": overhead,
                "precision_per_shot": precision,
                "net_benefit": net_ben,
                "n_entries": len(entries),
            }

        return self.derived_metrics

    # ═══════════════════════════════════════════════════════════════════════════
    # Ranking (Stage 4)
    # ═══════════════════════════════════════════════════════════════════════════

    def rank_configs(self) -> list[dict]:
        """Sort configs by mean_delta_e_gap ascending (best first).

        Must call compute_derived_metrics() before this method.

        Returns list of dicts with: config_id, mean_delta_e_gap,
        improvement_vs_raw, overhead_factor, precision_per_shot,
        net_benefit, rank.
        """
        if not self.derived_metrics:
            logger.warning("No derived metrics — call compute_derived_metrics() first")
            return []

        # Filter out configs with no mean_delta_e_gap
        ranked = [m for m in self.derived_metrics.values() if m.get("mean_delta_e_gap") is not None]

        # Sort ascending by mean_delta_e_gap
        ranked.sort(key=lambda x: x["mean_delta_e_gap"])

        # Assign rank (1-based)
        result: list[dict] = []
        for i, m in enumerate(ranked, start=1):
            result.append(
                {
                    "rank": i,
                    "config_id": m["config_id"],
                    "mean_delta_e_gap": m["mean_delta_e_gap"],
                    "improvement_vs_raw": m.get("improvement_vs_raw"),
                    "overhead_factor": m.get("overhead_factor"),
                    "precision_per_shot": m.get("precision_per_shot"),
                    "net_benefit": m.get("net_benefit"),
                }
            )

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Export (Stage 9)
    # ═══════════════════════════════════════════════════════════════════════════

    def export_comparison_table(self, output_dir: Path | None = None) -> None:
        """Export comparison_table.json with ranking and global metadata.

        Output path: results/mitigation_benchmark/analysis/comparison_table.json
        (or custom output_dir if provided).

        Contains:
        - ranking: list of ranked config dicts
        - metadata: n_configs, n_entries, baseline_config (C0_raw or null)
        """
        if output_dir is None:
            output_dir = self.results_dir / "analysis"

        output_dir.mkdir(parents=True, exist_ok=True)

        ranking = self.rank_configs()

        # Determine baseline config
        baseline_config = "C0_raw" if "C0_raw" in self.derived_metrics else None

        table = {
            "ranking": ranking,
            "metadata": {
                "n_configs": len(self.derived_metrics),
                "n_entries": len(self.entries),
                "baseline_config": baseline_config,
            },
        }

        output_path = output_dir / "comparison_table.json"
        output_path.write_text(json.dumps(table, indent=2))
        logger.info(f"Exported comparison table to {output_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _get_shots(entry: dict) -> int:
        """Extract total shots from an entry.

        Checks top-level 'shots' field first, falls back to
        benchmark_metadata.shots, then mitigation_config for PEA
        total shots (num_randomizations × shots_per_randomization).
        """
        shots = entry.get("shots")
        if shots and shots > 0:
            return int(shots)

        shots = entry.get("benchmark_metadata", {}).get("shots")
        if shots and shots > 0:
            return int(shots)

        # For PEA configs, total shots = num_rand × shots_per_rand
        mit_config = entry.get("mitigation_config", {})
        num_rand = mit_config.get("pea_num_randomizations")
        shots_per = mit_config.get("pea_shots_per_randomization")
        if num_rand and shots_per:
            return int(num_rand * shots_per)

        return 0

    # ═══════════════════════════════════════════════════════════════════════════
    # Pareto Frontier (Stage 4)
    # ═══════════════════════════════════════════════════════════════════════════

    def compute_pareto_frontier(self) -> list[str]:
        """Compute Pareto frontier of non-dominated configs.

        Considers the 2D objective space (overhead_factor, mean_delta_e_gap).
        A config is non-dominated if no other config has BOTH:
          - lower or equal overhead_factor AND
          - lower or equal mean_delta_e_gap
        with at least one strictly lower.

        Must call compute_derived_metrics() first (uses cached results if
        already computed).

        Configs missing overhead_factor or mean_delta_e_gap are skipped
        gracefully (e.g., when C0_raw is absent, overhead uses shots=1).

        Returns
        -------
        list[str]
            Config IDs on the Pareto frontier, sorted by overhead_factor.
        """
        # Ensure derived metrics are computed
        if not self.derived_metrics:
            self.compute_derived_metrics()

        # Collect valid 2D points: (overhead_factor, mean_delta_e_gap, config_id)
        points: list[tuple[float, float, str]] = []
        for config_id, metrics in self.derived_metrics.items():
            overhead = metrics.get("overhead_factor")
            gap = metrics.get("mean_delta_e_gap")
            if overhead is not None and gap is not None:
                points.append((overhead, gap, config_id))

        if not points:
            logger.warning("No valid points for Pareto frontier computation")
            return []

        # Find non-dominated configs
        frontier: list[str] = []
        for i, (oh_i, gap_i, cid_i) in enumerate(points):
            dominated = False
            for j, (oh_j, gap_j, _) in enumerate(points):
                if i == j:
                    continue
                # j dominates i if j is ≤ in both AND strictly < in at least one
                if oh_j <= oh_i and gap_j <= gap_i:
                    if oh_j < oh_i or gap_j < gap_i:
                        dominated = True
                        break
            if not dominated:
                frontier.append(cid_i)

        # Sort frontier by overhead_factor for consistent output
        frontier.sort(key=lambda c: self.derived_metrics[c].get("overhead_factor", 0.0))
        return frontier

    # ═══════════════════════════════════════════════════════════════════════════
    # Ablation Study (Stage 5)
    # ═══════════════════════════════════════════════════════════════════════════

    # Ablation pairs: technique → (config_without, config_with)
    # Contribution = delta_e_gap(without) - delta_e_gap(with)
    # Positive contribution means the technique helps (reduces error).
    ABLATION_PAIRS: dict[str, tuple[str, str]] = {
        "DD": ("C0_raw", "C1_dd_only"),
        "Twirling+TREX": ("C1_dd_only", "C2_dd_tw"),
        "ZNE (GF)": ("C2_dd_tw", "C3_full_gf"),
        "ZNE (PEA)": ("C2_dd_tw", "C5_full_pea_balanced"),
        "Affine": ("C15_pea_no_affine", "C5_full_pea_balanced"),
        "GNN-QEM": ("C5_full_pea_balanced", "C10_kitchen_sink"),
        "AQC": ("C5_full_pea_balanced", "C16_aqc_pea"),
    }

    def compute_ablation(self) -> dict[str, dict]:
        """Compute individual contribution of each technique via ablation.

        For each technique, finds the config pair where one has the technique
        and the other doesn't. Contribution is measured as:
          contribution = mean_delta_e_gap(without) - mean_delta_e_gap(with)

        A positive contribution means the technique reduces error (is helpful).

        For Affine: C15_pea_no_affine (no affine) vs C5 (with affine).
        For GNN-QEM: C5 (no GNN) vs C10_kitchen_sink (with GNN).
        For AQC: C5 (no AQC) vs C16_aqc_pea (with AQC).

        Results are exported to results/mitigation_benchmark/analysis/ablation_study.json.

        Returns
        -------
        dict[str, dict]
            Mapping technique_name → {contribution, config_pair,
            delta_observed, delta_without, delta_with}.
        """
        # Ensure derived metrics are computed
        if not self.derived_metrics:
            self.compute_derived_metrics()

        # Group entries by config_id for per-h-value comparison
        by_config: dict[str, list[dict]] = defaultdict(list)
        for entry in self.entries:
            config_id = entry["benchmark_metadata"]["config_id"]
            by_config[config_id].append(entry)

        ablation_results: dict[str, dict] = {}

        for technique, (config_without, config_with) in self.ABLATION_PAIRS.items():
            # Get mean_delta_e_gap from derived metrics
            metrics_without = self.derived_metrics.get(config_without)
            metrics_with = self.derived_metrics.get(config_with)

            if metrics_without is None or metrics_with is None:
                logger.warning(
                    f"Ablation '{technique}': missing data for "
                    f"{config_without} or {config_with} — skipping"
                )
                ablation_results[technique] = {
                    "contribution": None,
                    "config_pair": [config_without, config_with],
                    "delta_observed": None,
                    "delta_without": None,
                    "delta_with": None,
                    "status": "missing_data",
                }
                continue

            gap_without = metrics_without.get("mean_delta_e_gap")
            gap_with = metrics_with.get("mean_delta_e_gap")

            if gap_without is None or gap_with is None:
                logger.warning(
                    f"Ablation '{technique}': no delta_e_gap for "
                    f"{config_without} or {config_with} — skipping"
                )
                ablation_results[technique] = {
                    "contribution": None,
                    "config_pair": [config_without, config_with],
                    "delta_observed": None,
                    "delta_without": gap_without,
                    "delta_with": gap_with,
                    "status": "incomplete_data",
                }
                continue

            contribution = gap_without - gap_with
            ablation_results[technique] = {
                "contribution": contribution,
                "config_pair": [config_without, config_with],
                "delta_observed": contribution,
                "delta_without": gap_without,
                "delta_with": gap_with,
                "status": "computed",
            }

        # Export to analysis directory
        self._export_ablation_study(ablation_results)

        return ablation_results

    def _export_ablation_study(self, ablation_results: dict[str, dict]) -> None:
        """Export ablation study results to JSON.

        Output path: results/mitigation_benchmark/analysis/ablation_study.json
        """
        analysis_dir = self.results_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        output_path = analysis_dir / "ablation_study.json"
        output_data = {
            "description": (
                "Ablation study: individual contribution of each "
                "mitigation technique. Positive contribution = technique "
                "reduces error (helpful)."
            ),
            "techniques": ablation_results,
        }

        try:
            output_path.write_text(json.dumps(output_data, indent=2, default=str))
            logger.info(f"Exported ablation study to {output_path}")
        except OSError as e:
            logger.error(f"Failed to export ablation study: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure Generation (Stage 9 — Thesis Figures)
    # ═══════════════════════════════════════════════════════════════════════════

    # Method type classification for color-coding barplot
    _METHOD_TYPE_MAP: dict[str, str] = {
        "C0_raw": "raw",
        "C1_dd_only": "DD",
        "C2_dd_tw": "DD",
        "C3_full_gf": "ZNE",
        "C4_full_pea_light": "ZNE",
        "C5_full_pea_balanced": "ZNE",
        "C6_full_pea_heavy": "ZNE",
        "C7_pea_no_dd": "ZNE",
        "C8_pea_xy4": "ZNE",
        "C9_gnn_qem": "ZNE",
        "C10_kitchen_sink": "ZNE",
        "C11_mitiq_zne": "Mitiq",
        "C12_mitiq_cdr": "Mitiq",
        "C13_mitiq_ddd_zne": "Mitiq",
        "C14_dd_mitiq_cdr": "Mitiq",
        "C15_pea_no_affine": "ZNE",
        "C16_aqc_pea": "AQC",
        "C17_aqc_mitiq_cdr": "AQC",
        "C18_aqc_raw": "AQC",
    }

    _METHOD_COLORS: dict[str, str] = {
        "raw": "#888888",
        "DD": "#2196F3",
        "ZNE": "#4CAF50",
        "Mitiq": "#FF9800",
        "AQC": "#9C27B0",
    }

    def generate_figures(
        self, output_dir: Path | None = None, figure_name: str | None = None
    ) -> None:
        """Generate thesis figures for the mitigation benchmark analysis.

        Produces figures at 300 DPI (selectively or all):
          - precision_vs_config.png: barplot of ΔE/gap ranked by config
          - cost_benefit.png: scatter with Pareto frontier
          - technique_ablation.png: heatmap of technique contributions
          - sensitivity_pea_budget.png / sensitivity_twirling.png

        Parameters
        ----------
        output_dir : Path | None
            Output directory for figures. Defaults to
            results/mitigation_benchmark/analysis/figures/.
        figure_name : str | None
            Specific figure to generate. None or "all" generates all figures.
        """
        if output_dir is None:
            output_dir = self.results_dir / "analysis" / "figures"

        # Ensure derived metrics are computed
        if not self.derived_metrics:
            self.compute_derived_metrics()

        # Check minimum data requirement
        configs_with_data = [
            cid for cid, m in self.derived_metrics.items() if m.get("mean_delta_e_gap") is not None
        ]
        if len(configs_with_data) < 2:
            logger.warning(
                f"Only {len(configs_with_data)} configurations with derived "
                f"metrics — need at least 2 to generate figures. Skipping."
            )
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy import of matplotlib
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning(
                "matplotlib not available — cannot generate figures. "
                "Install with: pip install matplotlib"
            )
            return

        # Set consistent thesis style
        plt.rcParams.update({"font.size": 12})

        if figure_name is None or figure_name in ("precision", "all"):
            self._generate_precision_barplot(plt, output_dir)
        if figure_name is None or figure_name in ("pareto", "all"):
            self._generate_cost_benefit_scatter(plt, output_dir)
        if figure_name is None or figure_name in ("ablation", "all"):
            self._generate_ablation_heatmap(plt, output_dir)
        if figure_name is None or figure_name in ("sensitivity_pea", "sensitivity_twirling", "all"):
            self._generate_sensitivity_figures(output_dir)

    def _generate_precision_barplot(self, plt, output_dir: Path) -> None:
        """Generate precision_vs_config.png: barplot of ΔE/gap by config.

        Ranked configs on x-axis, mean_delta_e_gap on y-axis.
        Error bars from std across h-values.
        Color-coded by method type (raw, DD, ZNE, Mitiq, AQC).
        """
        from collections import defaultdict as _defaultdict

        # Get ranking (sorted by mean_delta_e_gap ascending)
        ranked = self.rank_configs()
        if len(ranked) < 2:
            logger.warning("precision_vs_config: fewer than 2 ranked configs — skipping")
            return

        # Compute std across h-values for each config
        by_config: dict[str, list[float]] = _defaultdict(list)
        for entry in self.entries:
            config_id = entry["benchmark_metadata"]["config_id"]
            delta = entry["results"].get("delta_e_gap")
            if delta is not None:
                by_config[config_id].append(delta)

        config_ids = [r["config_id"] for r in ranked]
        means = [r["mean_delta_e_gap"] for r in ranked]
        stds = []
        for cid in config_ids:
            values = by_config.get(cid, [])
            if len(values) >= 2:
                stds.append(stdev(values))
            else:
                stds.append(0.0)

        # Color by method type
        colors = [
            self._METHOD_COLORS.get(self._METHOD_TYPE_MAP.get(cid, "raw"), "#888888")
            for cid in config_ids
        ]

        # Create figure
        n_configs = len(config_ids)
        figwidth = max(10, n_configs * 0.6)
        fig, ax = plt.subplots(figsize=(figwidth, 6))

        x_pos = range(n_configs)
        bars = ax.bar(
            x_pos, means, yerr=stds, color=colors, edgecolor="black", linewidth=0.5, capsize=3
        )

        ax.set_xlabel("Configuration (ranked)")
        ax.set_ylabel("Mean ΔE/gap")
        ax.set_title("ΔE/gap by Mitigation Configuration")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [cid.replace("_", "\n", 1) for cid in config_ids],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        ax.axhline(y=0.03, color="red", linestyle="--", alpha=0.7, label="3% threshold")

        # Legend for method types
        from matplotlib.patches import Patch

        legend_patches = [
            Patch(facecolor=color, edgecolor="black", label=method)
            for method, color in self._METHOD_COLORS.items()
            if any(self._METHOD_TYPE_MAP.get(cid) == method for cid in config_ids)
        ]
        if legend_patches:
            ax.legend(handles=legend_patches, loc="upper right")

        plt.tight_layout()
        out_path = output_dir / "precision_vs_config.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Generated {out_path}")

    def _generate_cost_benefit_scatter(self, plt, output_dir: Path) -> None:
        """Generate cost_benefit.png: scatter with Pareto frontier.

        x-axis: overhead_factor, y-axis: mean_delta_e_gap.
        Each point labeled with config shortname.
        Pareto frontier connected as a line.
        """
        # Collect valid 2D points
        points: list[tuple[float, float, str]] = []
        for config_id, metrics in self.derived_metrics.items():
            overhead = metrics.get("overhead_factor")
            gap = metrics.get("mean_delta_e_gap")
            if overhead is not None and gap is not None:
                points.append((overhead, gap, config_id))

        if len(points) < 2:
            logger.warning("cost_benefit: fewer than 2 configs with overhead data — skipping")
            return

        fig, ax = plt.subplots(figsize=(10, 7))

        # Color by method type
        for overhead, gap, cid in points:
            method = self._METHOD_TYPE_MAP.get(cid, "raw")
            color = self._METHOD_COLORS.get(method, "#888888")
            ax.scatter(overhead, gap, c=color, s=80, edgecolors="black", linewidths=0.5, zorder=3)
            # Label with shortname (remove C*_ prefix noise)
            shortname = cid.split("_", 1)[0] if "_" in cid else cid
            ax.annotate(
                shortname,
                (overhead, gap),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                alpha=0.8,
            )

        # Pareto frontier line
        frontier_ids = self.compute_pareto_frontier()
        if len(frontier_ids) >= 2:
            frontier_points = [
                (
                    self.derived_metrics[cid]["overhead_factor"],
                    self.derived_metrics[cid]["mean_delta_e_gap"],
                )
                for cid in frontier_ids
                if self.derived_metrics[cid].get("overhead_factor") is not None
                and self.derived_metrics[cid].get("mean_delta_e_gap") is not None
            ]
            if len(frontier_points) >= 2:
                frontier_points.sort(key=lambda p: p[0])
                fx = [p[0] for p in frontier_points]
                fy = [p[1] for p in frontier_points]
                ax.plot(fx, fy, "r--", linewidth=1.5, alpha=0.8, label="Pareto frontier", zorder=2)

        ax.set_xlabel("Overhead Factor (shots / baseline)")
        ax.set_ylabel("Mean ΔE/gap")
        ax.set_title("Cost-Benefit Analysis (Pareto Frontier)")
        ax.axhline(y=0.03, color="gray", linestyle=":", alpha=0.5, label="3% threshold")
        ax.legend(loc="upper right")

        plt.tight_layout()
        out_path = output_dir / "cost_benefit.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Generated {out_path}")

    def _generate_ablation_heatmap(self, plt, output_dir: Path) -> None:
        """Generate technique_ablation.png: heatmap of technique contributions.

        Rows: techniques (DD, Twirling, ZNE, Affine, etc.)
        Columns: h-values
        Color: contribution Δ (improvement from adding technique)
        """
        import numpy as np

        # Build per-h-value ablation data
        # For each technique, compute delta_e_gap(without) - delta_e_gap(with)
        # at each h_value
        by_config_h: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
        for entry in self.entries:
            config_id = entry["benchmark_metadata"]["config_id"]
            h_val = entry["benchmark_metadata"]["h_value"]
            delta = entry["results"].get("delta_e_gap")
            if delta is not None:
                by_config_h[config_id][h_val].append(delta)

        # Collect all h-values present in data
        all_h_values: set[float] = set()
        for h_map in by_config_h.values():
            all_h_values.update(h_map.keys())

        if not all_h_values:
            logger.warning("technique_ablation: no h-value data — skipping")
            return

        h_values_sorted = sorted(all_h_values)
        techniques_ordered = list(self.ABLATION_PAIRS.keys())

        # Build heatmap matrix: rows=techniques, cols=h_values
        matrix: list[list[float | None]] = []
        valid_techniques: list[str] = []

        for technique in techniques_ordered:
            config_without, config_with = self.ABLATION_PAIRS[technique]
            row: list[float | None] = []
            has_any_data = False

            for h_val in h_values_sorted:
                vals_without = by_config_h.get(config_without, {}).get(h_val, [])
                vals_with = by_config_h.get(config_with, {}).get(h_val, [])

                if vals_without and vals_with:
                    mean_without = sum(vals_without) / len(vals_without)
                    mean_with = sum(vals_with) / len(vals_with)
                    contribution = mean_without - mean_with
                    row.append(contribution)
                    has_any_data = True
                else:
                    row.append(None)

            if has_any_data:
                matrix.append(row)
                valid_techniques.append(technique)

        if len(valid_techniques) < 1:
            logger.warning("technique_ablation: no technique has paired data — skipping")
            return

        # Convert to numpy array (replace None with NaN)
        data = np.array(
            [[v if v is not None else float("nan") for v in row] for row in matrix],
            dtype=float,
        )

        fig, ax = plt.subplots(
            figsize=(max(6, len(h_values_sorted) * 1.2), max(4, len(valid_techniques) * 0.8))
        )

        # Use diverging colormap (green=positive/helpful, red=harmful)
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", interpolation="nearest")

        ax.set_xticks(range(len(h_values_sorted)))
        ax.set_xticklabels([f"{h:.2f}" for h in h_values_sorted])
        ax.set_yticks(range(len(valid_techniques)))
        ax.set_yticklabels(valid_techniques)
        ax.set_xlabel("h-value")
        ax.set_ylabel("Technique")
        ax.set_title("Technique Ablation Study")

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Contribution Δ (positive = helpful)")

        # Annotate cells with values
        for i in range(len(valid_techniques)):
            for j in range(len(h_values_sorted)):
                val = data[i, j]
                if not np.isnan(val):
                    text_color = "white" if abs(val) > 0.5 * np.nanmax(np.abs(data)) else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=text_color,
                    )

        plt.tight_layout()
        out_path = output_dir / "technique_ablation.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Generated {out_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Transfer Analysis (Stage 6)
    # ═══════════════════════════════════════════════════════════════════════════

    def compute_transfer_ratios(self) -> dict[str, float]:
        """Compute ΔE/gap(hardware) / ΔE/gap(fake_backend) per config_id.

        Groups entries by (config_id, execution_mode). For each config_id
        with results in BOTH "fake_backend" and "hardware":
          - Computes mean delta_e_gap for fake_backend entries
          - Computes mean delta_e_gap for hardware entries
          - transfer_ratio = mean_delta_hw / mean_delta_sim

        Returns
        -------
        dict[str, float]
            Mapping config_id → transfer_ratio. Empty dict if no dual-mode
            configs exist.
        """
        # Group delta_e_gap values by (config_id, execution_mode)
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for entry in self.entries:
            config_id = entry["benchmark_metadata"]["config_id"]
            mode = entry["benchmark_metadata"]["execution_mode"]
            delta = entry["results"]["delta_e_gap"]
            if delta is not None:
                grouped[config_id][mode].append(delta)

        # Compute transfer ratio for configs with both modes
        ratios: dict[str, float] = {}
        for config_id, modes in grouped.items():
            if "fake_backend" in modes and "hardware" in modes:
                sim_values = modes["fake_backend"]
                hw_values = modes["hardware"]
                mean_sim = sum(sim_values) / len(sim_values)
                mean_hw = sum(hw_values) / len(hw_values)
                if mean_sim > 0:
                    ratios[config_id] = mean_hw / mean_sim
                else:
                    logger.warning(
                        f"Skipping transfer ratio for {config_id}: "
                        f"mean_delta_e_gap(sim) = {mean_sim} (non-positive)"
                    )

        if not ratios:
            logger.warning("No dual-mode configs found — cannot compute transfer ratios")

        return ratios

    def compute_spearman_correlation(self) -> float | None:
        """Compute Spearman ρ between sim and hw rankings of configs.

        Ranks configs by mean_delta_e_gap in each mode separately.
        Requires ≥5 configs with results in both modes.

        Returns
        -------
        float | None
            Spearman ρ value, or None if fewer than 5 dual-mode configs.
        """
        # Group delta_e_gap by (config_id, mode)
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for entry in self.entries:
            config_id = entry["benchmark_metadata"]["config_id"]
            mode = entry["benchmark_metadata"]["execution_mode"]
            delta = entry["results"]["delta_e_gap"]
            if delta is not None:
                grouped[config_id][mode].append(delta)

        # Find configs with both modes
        dual_configs: list[str] = []
        for config_id, modes in grouped.items():
            if "fake_backend" in modes and "hardware" in modes:
                dual_configs.append(config_id)

        if len(dual_configs) < 5:
            logger.warning(
                f"Only {len(dual_configs)} dual-mode configs found "
                f"(minimum 5 required) — skipping Spearman correlation"
            )
            return None

        # Compute mean delta_e_gap per mode for dual configs
        sim_means: list[float] = []
        hw_means: list[float] = []
        for config_id in dual_configs:
            sim_vals = grouped[config_id]["fake_backend"]
            hw_vals = grouped[config_id]["hardware"]
            sim_means.append(sum(sim_vals) / len(sim_vals))
            hw_means.append(sum(hw_vals) / len(hw_vals))

        # Compute Spearman rank correlation
        try:
            from scipy.stats import spearmanr
        except ImportError:
            logger.warning("scipy not available — cannot compute Spearman correlation")
            return None

        rho, _ = spearmanr(sim_means, hw_means)
        return float(rho)

    # ═══════════════════════════════════════════════════════════════════════════
    # Sensitivity Analysis (Stage 7)
    # ═══════════════════════════════════════════════════════════════════════════

    # PEA budget mapping: config_id → (num_randomizations × shots_per_randomization)
    _PEA_BUDGET_CONFIGS: dict[str, int] = {
        "C4_full_pea_light": 32 * 128,  # 4096
        "C5_full_pea_balanced": 48 * 192,  # 9216
        "C6_full_pea_heavy": 64 * 256,  # 16384
    }

    # Twirling configs: config_id → twirling_num_randomizations
    # None means twirling disabled; only include configs that have twirling set.
    _TWIRLING_CONFIGS: dict[str, int | None] = {
        "C0_raw": None,
        "C1_dd_only": None,
        "C2_dd_tw": 32,
        "C3_full_gf": 32,
        "C4_full_pea_light": None,
        "C5_full_pea_balanced": 48,
        "C6_full_pea_heavy": 64,
    }

    def compute_transpilation_summary(self) -> dict[str, dict]:
        """Compute per-config transpilation metrics summary.

        Groups results by optimization_level and returns key metrics
        (depth_2q, n_2q, fidelity) for transpilation decision-making.

        Returns
        -------
        dict with keys:
          - per_config: dict[config_id → circuit_stats]
          - by_opt_level: dict[opt_level → aggregated stats]
          - conclusion: str (human-readable summary)
        """
        per_config: dict[str, dict] = {}
        for entry in self.entries:
            cid = entry.get("benchmark_metadata", {}).get("config_id", "")
            cs = entry.get("circuit_stats", {})
            if cid and cs and cid not in per_config:
                per_config[cid] = {
                    "optimization_level": cs.get("optimization_level"),
                    "depth": cs.get("depth"),
                    "depth_2q": cs.get("depth_2q"),
                    "n_2q_gates": cs.get("n_2q_gates"),
                    "n_1q_gates": cs.get("n_1q_gates"),
                    "fidelity_estimate": cs.get("fidelity_estimate"),
                    "routing_overhead_pct": cs.get("routing_overhead_pct"),
                }

        # Group by opt_level
        from collections import defaultdict

        by_opt: dict[int, list] = defaultdict(list)
        for cid, s in per_config.items():
            opt = s.get("optimization_level", 2)
            by_opt[opt].append(s)

        import numpy as np

        by_opt_summary = {}
        for opt, entries in by_opt.items():
            d2qs = [s["depth_2q"] for s in entries if s.get("depth_2q")]
            n2qs = [s["n_2q_gates"] for s in entries if s.get("n_2q_gates")]
            fids = [s["fidelity_estimate"] for s in entries if s.get("fidelity_estimate")]
            by_opt_summary[opt] = {
                "n_configs": len(entries),
                "mean_depth_2q": float(np.mean(d2qs)) if d2qs else 0,
                "mean_n_2q": float(np.mean(n2qs)) if n2qs else 0,
                "mean_fidelity": float(np.mean(fids)) if fids else 0,
            }

        conclusion = ""
        if 0 in by_opt_summary and 2 in by_opt_summary:
            ratio = by_opt_summary[0]["mean_depth_2q"] / max(by_opt_summary[2]["mean_depth_2q"], 1)
            conclusion = (
                f"opt_level=0 produces {ratio:.1f}x deeper circuits than opt_level=2. "
                f"Use opt_level=2 for hardware (PEA), opt_level=0 only for Mitiq."
            )

        return {
            "per_config": per_config,
            "by_opt_level": by_opt_summary,
            "conclusion": conclusion,
        }

    def compute_sensitivity_curves(self) -> dict[str, dict]:
        """Compute parameter sensitivity curves for PEA budget and twirling.

        PEA budget curve:
          Groups entries for C4/C5/C6 by their total PEA budget
          (num_randomizations × shots_per_randomization). Computes
          mean delta_e_gap per budget level.

        Twirling curve:
          Groups configs by twirling_num_randomizations value.
          Computes mean delta_e_gap per randomization count.

        Saturation logic:
          For consecutive sorted points (x1, y1) and (x2, y2):
            improvement = y1 - y2 (positive means y decreased = better)
            cost_delta = x2 - x1
          Saturation where |improvement| / cost_delta < 0.1 (normalized).

        Returns
        -------
        dict[str, dict]
            Keys: "pea_budget", "twirling". Each value is a dict with:
            - "data_points": list of (parameter_value, mean_delta_e_gap) sorted
            - "saturation_point": parameter value where saturation occurs or None
            - "sufficient_data": bool (True if ≥3 data points)
        """
        return {
            "pea_budget": self._compute_pea_budget_curve(),
            "twirling": self._compute_twirling_curve(),
        }

    def _compute_pea_budget_curve(self) -> dict:
        """Compute PEA budget sensitivity curve from C4/C5/C6 entries."""
        # Group delta_e_gap values by PEA budget
        budget_deltas: dict[int, list[float]] = defaultdict(list)

        for entry in self.entries:
            config_id = entry["benchmark_metadata"]["config_id"]
            if config_id not in self._PEA_BUDGET_CONFIGS:
                continue
            delta = entry["results"]["delta_e_gap"]
            if delta is not None:
                budget = self._PEA_BUDGET_CONFIGS[config_id]
                budget_deltas[budget].append(delta)

        # Build data points: (budget, mean_delta_e_gap) sorted by budget
        data_points: list[tuple[float, float]] = []
        for budget in sorted(budget_deltas.keys()):
            values = budget_deltas[budget]
            if values:
                data_points.append((float(budget), sum(values) / len(values)))

        sufficient = len(data_points) >= 3
        if not sufficient:
            logger.warning(
                f"PEA budget sensitivity: only {len(data_points)} data points "
                f"(minimum 3 required) — curve may be incomplete"
            )

        saturation = self._find_saturation_point(data_points)

        return {
            "data_points": data_points,
            "saturation_point": saturation,
            "sufficient_data": sufficient,
        }

    def _compute_twirling_curve(self) -> dict:
        """Compute twirling sensitivity curve from configs with varying twirling."""
        # Group delta_e_gap values by twirling_num_randomizations
        twirl_deltas: dict[int, list[float]] = defaultdict(list)

        for entry in self.entries:
            config_id = entry["benchmark_metadata"]["config_id"]
            if config_id not in self._TWIRLING_CONFIGS:
                continue
            twirl_value = self._TWIRLING_CONFIGS[config_id]
            # Skip configs with twirling disabled (None)
            if twirl_value is None:
                continue
            delta = entry["results"]["delta_e_gap"]
            if delta is not None:
                twirl_deltas[twirl_value].append(delta)

        # Build data points: (num_randomizations, mean_delta_e_gap) sorted
        data_points: list[tuple[float, float]] = []
        for num_rand in sorted(twirl_deltas.keys()):
            values = twirl_deltas[num_rand]
            if values:
                data_points.append((float(num_rand), sum(values) / len(values)))

        sufficient = len(data_points) >= 3
        if not sufficient:
            logger.warning(
                f"Twirling sensitivity: only {len(data_points)} data points "
                f"(minimum 3 required) — curve may be incomplete"
            )

        saturation = self._find_saturation_point(data_points)

        return {
            "data_points": data_points,
            "saturation_point": saturation,
            "sufficient_data": sufficient,
        }

    @staticmethod
    def _find_saturation_point(
        data_points: list[tuple[float, float]],
    ) -> float | None:
        """Find the saturation point in a sensitivity curve.

        Saturation occurs at the first point (x2) where:
          |improvement| / cost_delta < 0.1

        Where:
          improvement = y1 - y2 (positive = y decreased = getting better)
          cost_delta = x2 - x1 (always positive for sorted points)

        Parameters
        ----------
        data_points : list[tuple[float, float]]
            Sorted list of (parameter_value, mean_delta_e_gap).

        Returns
        -------
        float | None
            The parameter value at which saturation begins, or None if
            no saturation detected or fewer than 2 points.
        """
        if len(data_points) < 2:
            return None

        for i in range(1, len(data_points)):
            x1, y1 = data_points[i - 1]
            x2, y2 = data_points[i]
            cost_delta = x2 - x1
            if cost_delta <= 0:
                continue
            improvement = y1 - y2  # positive = delta_e_gap decreased
            # Normalize: |improvement| / cost_delta
            ratio = abs(improvement) / cost_delta
            if ratio < 0.1:
                return x2

        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # Sensitivity Figures (Stage 9 — Requirement 14.4, 17.4)
    # ═══════════════════════════════════════════════════════════════════════════

    def _generate_sensitivity_figures(self, output_dir: Path) -> None:
        """Generate sensitivity curve figures for PEA budget and twirling.

        Produces:
          - sensitivity_pea_budget.png: ΔE/gap vs total PEA budget (if ≥3 points)
          - sensitivity_twirling.png: ΔE/gap vs twirling_num_randomizations (if ≥3 points)

        Saturation point is marked with a vertical dashed line if detected.
        Style: 300 DPI, font size 12, consistent with thesis figures.

        Parameters
        ----------
        output_dir : Path
            Directory to save figures (typically analysis/figures/).
        """
        sensitivity = self.compute_sensitivity_curves()
        output_dir.mkdir(parents=True, exist_ok=True)

        # PEA budget sensitivity
        pea = sensitivity["pea_budget"]
        if pea["sufficient_data"]:
            self._plot_sensitivity_curve(
                data_points=pea["data_points"],
                saturation_point=pea["saturation_point"],
                xlabel="PEA Total Budget (num_rand × shots_per_rand)",
                ylabel="Mean ΔE/gap",
                title="Sensitivity: PEA Budget vs ΔE/gap",
                output_path=output_dir / "sensitivity_pea_budget.png",
            )
            logger.info("Generated sensitivity_pea_budget.png")
        else:
            logger.warning(
                "Skipping sensitivity_pea_budget.png: insufficient data "
                f"({len(pea['data_points'])} points, need ≥3)"
            )

        # Twirling sensitivity
        twirl = sensitivity["twirling"]
        if twirl["sufficient_data"]:
            self._plot_sensitivity_curve(
                data_points=twirl["data_points"],
                saturation_point=twirl["saturation_point"],
                xlabel="Twirling num_randomizations",
                ylabel="Mean ΔE/gap",
                title="Sensitivity: Twirling Randomizations vs ΔE/gap",
                output_path=output_dir / "sensitivity_twirling.png",
            )
            logger.info("Generated sensitivity_twirling.png")
        else:
            logger.warning(
                "Skipping sensitivity_twirling.png: insufficient data "
                f"({len(twirl['data_points'])} points, need ≥3)"
            )

    @staticmethod
    def _plot_sensitivity_curve(
        data_points: list[tuple[float, float]],
        saturation_point: float | None,
        xlabel: str,
        ylabel: str,
        title: str,
        output_path: Path,
    ) -> None:
        """Plot a single sensitivity curve with optional saturation marker.

        Parameters
        ----------
        data_points : list[tuple[float, float]]
            Sorted (x, y) pairs for the curve.
        saturation_point : float | None
            x-value where saturation begins (vertical dashed line).
        xlabel, ylabel, title : str
            Axis labels and figure title.
        output_path : Path
            Full path to save the PNG file.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 13,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
            }
        )

        xs = [p[0] for p in data_points]
        ys = [p[1] for p in data_points]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(xs, ys, "o-", color="#1b9e77", linewidth=2, markersize=8)

        if saturation_point is not None:
            ax.axvline(
                x=saturation_point,
                color="#d95f02",
                linestyle="--",
                linewidth=1.5,
                label=f"Saturation ({saturation_point:.0f})",
            )
            ax.legend(loc="upper right")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════════
    # LaTeX Table Export (Stage 9 — Requirement 14.4)
    # ═══════════════════════════════════════════════════════════════════════════

    def generate_latex_table(self) -> str:
        """Generate LaTeX table for Chapter 5 with ranking and metrics.

        Columns: Rank, Config, ΔE/gap (%), Overhead, Precision/Shot, Net Benefit.
        Formatted as \\begin{table}...\\end{table} with caption and label.

        Also writes to results/mitigation_benchmark/analysis/comparison_table.tex.

        Returns
        -------
        str
            Complete LaTeX table string.
        """
        # Ensure data is ready
        self.compute_derived_metrics()
        ranking = self.rank_configs()

        if not ranking:
            logger.warning("No ranking data — cannot generate LaTeX table")
            return ""

        # Build LaTeX
        lines: list[str] = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"  \centering")
        lines.append(
            r"  \caption{Mitigation configuration ranking by mean "
            r"$\Delta E / \text{gap}$. Lower is better.}"
        )
        lines.append(r"  \label{tab:mitigation_ranking}")
        lines.append(r"  \begin{tabular}{r l r r r r}")
        lines.append(r"    \toprule")
        lines.append(
            r"    Rank & Config & $\Delta E$/gap (\%) "
            r"& Overhead & Prec./Shot & Net Benefit \\"
        )
        lines.append(r"    \midrule")

        for entry in ranking:
            rank = entry["rank"]
            config_id = entry["config_id"].replace("_", r"\_")
            gap_pct = (
                f"{entry['mean_delta_e_gap'] * 100:.2f}"
                if entry["mean_delta_e_gap"] is not None
                else "---"
            )
            overhead = (
                f"{entry['overhead_factor']:.1f}x"
                if entry.get("overhead_factor") is not None
                else "---"
            )
            precision = (
                f"{entry['precision_per_shot']:.2e}"
                if entry.get("precision_per_shot") is not None
                else "---"
            )
            net_ben = (
                f"{entry['net_benefit']:.4f}" if entry.get("net_benefit") is not None else "---"
            )
            lines.append(
                f"    {rank} & {config_id} & {gap_pct} & {overhead} & {precision} & {net_ben} \\\\"
            )

        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")
        lines.append(r"\end{table}")

        latex_str = "\n".join(lines)

        # Write to file
        output_dir = self.results_dir / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "comparison_table.tex"
        try:
            output_path.write_text(latex_str)
            logger.info(f"Exported LaTeX table to {output_path}")
        except OSError as e:
            logger.error(f"Failed to export LaTeX table: {e}")

        return latex_str

    # ═══════════════════════════════════════════════════════════════════════════
    # Per-Regime Analysis
    # ═══════════════════════════════════════════════════════════════════════════

    def compute_per_regime_analysis(self) -> dict[str, dict]:
        """Split results into 3 regimes and compute per-config statistics.

        Regimes based on h-value (excluding h < 0.75 — ansatz failure):
          - "critical" (0.75 ≤ h < 2.0): near phase transition
          - "transition" (2.0 ≤ h < 3.0): intermediate
          - "paramagnetic" (h ≥ 3.0): deep paramagnetic (production target)

        For each regime, computes per-config mean ΔE/gap, std, ranking,
        and identifies the best configuration.

        Returns
        -------
        dict[str, dict]
            Keys: "critical", "transition", "paramagnetic". Each contains:
            - h_range: (min_h, max_h) observed in that regime
            - n_entries: number of entries in the regime
            - ranking: list of {config_id, mean_delta_e_gap, std, n} sorted ascending
            - best_config: config_id with lowest mean ΔE/gap
            - best_delta_e_gap: the best mean value
        """
        regime_bounds = {
            "critical": (0.75, 2.0),
            "transition": (2.0, 3.0),
            "paramagnetic": (3.0, float("inf")),
        }

        result: dict[str, dict] = {}

        for regime_name, (h_low, h_high) in regime_bounds.items():
            # Filter entries for this regime
            regime_entries: list[dict] = []
            for entry in self.entries:
                h_val = entry["benchmark_metadata"].get("h_value")
                if h_val is None:
                    continue
                # Exclude h < 0.75 (ansatz failure)
                if h_val < 0.75:
                    continue
                if h_low <= h_val < h_high:
                    regime_entries.append(entry)

            if not regime_entries:
                result[regime_name] = {
                    "h_range": (h_low, h_high),
                    "n_entries": 0,
                    "ranking": [],
                    "best_config": None,
                    "best_delta_e_gap": None,
                }
                continue

            # Compute h_range actually observed
            h_values = [e["benchmark_metadata"]["h_value"] for e in regime_entries]
            observed_h_range = (min(h_values), max(h_values))

            # Group by config_id
            by_config: dict[str, list[float]] = defaultdict(list)
            for entry in regime_entries:
                config_id = entry["benchmark_metadata"]["config_id"]
                delta = entry["results"].get("delta_e_gap")
                if delta is not None:
                    by_config[config_id].append(delta)

            # Compute ranking
            ranking: list[dict] = []
            for config_id, values in by_config.items():
                mean_val = mean(values)
                std_val = stdev(values) if len(values) >= 2 else 0.0
                ranking.append(
                    {
                        "config_id": config_id,
                        "mean_delta_e_gap": mean_val,
                        "std": std_val,
                        "n": len(values),
                    }
                )

            # Sort ascending by mean_delta_e_gap
            ranking.sort(key=lambda x: x["mean_delta_e_gap"])

            best = ranking[0] if ranking else None

            result[regime_name] = {
                "h_range": observed_h_range,
                "n_entries": len(regime_entries),
                "ranking": ranking,
                "best_config": best["config_id"] if best else None,
                "best_delta_e_gap": best["mean_delta_e_gap"] if best else None,
            }

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # H-Sweep Table
    # ═══════════════════════════════════════════════════════════════════════════

    _DEFAULT_H_SWEEP_CONFIGS = [
        "C0_raw",
        "C3_full_gf",
        "C5_full_pea_balanced",
        "C16_aqc_pea",
    ]

    def compute_h_sweep_table(
        self, configs: list[str] | None = None
    ) -> dict[str, dict[float, dict]]:
        """Compute per-h-value statistics for selected configurations.

        For each config, groups entries by h-value and computes mean, std,
        n, and improvement relative to C0_raw at the same h-value.

        Parameters
        ----------
        configs : list[str] | None
            Config IDs to include. Defaults to C0_raw, C3_full_gf,
            C5_full_pea_balanced, C16_aqc_pea.

        Returns
        -------
        dict[str, dict[float, dict]]
            Mapping config_id → {h_value → {mean, std, n, improvement_vs_raw}}.
        """
        if configs is None:
            configs = self._DEFAULT_H_SWEEP_CONFIGS

        # First, build C0_raw per-h baseline for improvement computation
        c0_by_h: dict[float, list[float]] = defaultdict(list)
        for entry in self.entries:
            if entry["benchmark_metadata"]["config_id"] == "C0_raw":
                h_val = entry["benchmark_metadata"]["h_value"]
                delta = entry["results"].get("delta_e_gap")
                if delta is not None:
                    c0_by_h[h_val].append(delta)

        c0_means: dict[float, float] = {h: mean(vals) for h, vals in c0_by_h.items() if vals}

        # Compute per-config, per-h stats
        result: dict[str, dict[float, dict]] = {}

        for config_id in configs:
            by_h: dict[float, list[float]] = defaultdict(list)
            for entry in self.entries:
                if entry["benchmark_metadata"]["config_id"] != config_id:
                    continue
                h_val = entry["benchmark_metadata"]["h_value"]
                delta = entry["results"].get("delta_e_gap")
                if delta is not None:
                    by_h[h_val].append(delta)

            config_table: dict[float, dict] = {}
            for h_val in sorted(by_h.keys()):
                values = by_h[h_val]
                mean_val = mean(values)
                std_val = stdev(values) if len(values) >= 2 else 0.0

                # Improvement vs C0_raw at same h
                improvement: float | None = None
                c0_mean_h = c0_means.get(h_val)
                if c0_mean_h is not None and c0_mean_h > 1e-12:
                    improvement = (c0_mean_h - mean_val) / c0_mean_h

                config_table[h_val] = {
                    "mean": mean_val,
                    "std": std_val,
                    "n": len(values),
                    "improvement_vs_raw": improvement,
                }

            result[config_id] = config_table

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Shot Sensitivity
    # ═══════════════════════════════════════════════════════════════════════════

    def compute_shot_sensitivity(self) -> dict[str, list[dict]]:
        """Group entries by (config_id, shots) and compute mean ΔE/gap.

        Returns
        -------
        dict[str, list[dict]]
            Mapping config_id → [{shots, mean_delta_e_gap, std, n}, ...]
            sorted by shots ascending.
        """
        # Group by (config_id, shots)
        grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

        for entry in self.entries:
            config_id = entry["benchmark_metadata"]["config_id"]
            shots = self._get_shots(entry)
            delta = entry["results"].get("delta_e_gap")
            if delta is not None and shots > 0:
                grouped[config_id][shots].append(delta)

        result: dict[str, list[dict]] = {}

        for config_id, shots_map in sorted(grouped.items()):
            entries_list: list[dict] = []
            for shots in sorted(shots_map.keys()):
                values = shots_map[shots]
                mean_val = mean(values)
                std_val = stdev(values) if len(values) >= 2 else 0.0
                entries_list.append(
                    {
                        "shots": shots,
                        "mean_delta_e_gap": mean_val,
                        "std": std_val,
                        "n": len(values),
                    }
                )
            result[config_id] = entries_list

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Full Analysis Export (JSON)
    # ═══════════════════════════════════════════════════════════════════════════

    def export_full_analysis(self, output_path: Path) -> None:
        """Export complete analysis to a single JSON file.

        Includes:
          - per_regime: from compute_per_regime_analysis()
          - h_sweep: from compute_h_sweep_table()
          - shot_sensitivity: from compute_shot_sensitivity()
          - ranking: from rank_configs()
          - hypothesis_verdicts: from compute_hypothesis_verdicts()
          - metadata: {n_entries, n_errors, h_range, configs, seeds}

        Parameters
        ----------
        output_path : Path
            Destination file path for the JSON export.
        """
        # Ensure derived metrics are computed for ranking
        self.compute_derived_metrics()

        # Collect all h-values and configs
        all_h: list[float] = []
        all_configs: set[str] = set()
        all_seeds: set[int] = set()
        for entry in self.entries:
            h_val = entry["benchmark_metadata"].get("h_value")
            if h_val is not None:
                all_h.append(h_val)
            config_id = entry["benchmark_metadata"].get("config_id")
            if config_id:
                all_configs.add(config_id)
            seed = entry.get("benchmark_metadata", {}).get("seed")
            if seed is not None:
                all_seeds.add(seed)

        h_range = (min(all_h), max(all_h)) if all_h else (0.0, 0.0)

        # Build h_sweep with float keys converted to strings for JSON
        h_sweep_raw = self.compute_h_sweep_table()
        h_sweep_json: dict[str, dict[str, dict]] = {}
        for config_id, h_map in h_sweep_raw.items():
            h_sweep_json[config_id] = {str(h_val): stats for h_val, stats in h_map.items()}

        export_data = {
            "per_regime": self.compute_per_regime_analysis(),
            "h_sweep": h_sweep_json,
            "shot_sensitivity": self.compute_shot_sensitivity(),
            "ranking": self.rank_configs(),
            "hypothesis_verdicts": self.compute_hypothesis_verdicts(),
            "metadata": {
                "n_entries": len(self.entries),
                "n_errors": len(self.errors),
                "h_range": list(h_range),
                "configs": sorted(all_configs),
                "seeds": sorted(all_seeds),
            },
        }

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            output_path.write_text(json.dumps(export_data, indent=2, default=str))
            logger.info(f"Exported full analysis to {output_path}")
        except OSError as e:
            logger.error(f"Failed to export full analysis: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point for MitigationBenchmarkAnalyzer."""
    parser = argparse.ArgumentParser(
        description="Mitigation Benchmark Analyzer — consolidate results and generate analysis"
    )
    parser.add_argument(
        "--thesis-table",
        action="store_true",
        help="Print thesis-ready LaTeX/Markdown comparison table",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Generate analysis figures (PNG 300 DPI)",
    )
    parser.add_argument(
        "--figure",
        type=str,
        default=None,
        choices=[
            "precision",
            "pareto",
            "ablation",
            "sensitivity_pea",
            "sensitivity_twirling",
            "all",
        ],
        help="Generate a specific figure (default: all when --figures is passed)",
    )
    parser.add_argument(
        "--statistical",
        action="store_true",
        help="Run statistical significance tests between configurations",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed per-regime analysis (critical/transition/paramagnetic)",
    )
    parser.add_argument(
        "--h-sweep",
        action="store_true",
        help="Show per-h-value table for key configs (C0, C3, C5, C16)",
    )
    parser.add_argument(
        "--shot-sensitivity",
        action="store_true",
        help="Show shot budget sensitivity analysis",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Export full analysis to JSON file",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override results directory (default: results/mitigation_benchmark/)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["fake_backend", "hardware"],
        help="Filter results by execution mode. If not set, loads all modes.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    results_dir = Path(args.results_dir) if args.results_dir else None
    analyzer = MitigationBenchmarkAnalyzer(results_dir=results_dir)
    analyzer.scan()

    # Filter by execution mode if specified
    if args.mode:
        before_count = len(analyzer.entries)
        analyzer.entries = [
            e
            for e in analyzer.entries
            if e.get("benchmark_metadata", {}).get("execution_mode") == args.mode
        ]
        filtered = before_count - len(analyzer.entries)
        if filtered > 0:
            print(
                f"  Filtered to --mode={args.mode}: {len(analyzer.entries)} entries "
                f"({filtered} from other modes excluded)"
            )

    print(f"Loaded {len(analyzer.entries)} valid results ({len(analyzer.errors)} errors)")

    if analyzer.errors and args.verbose:
        print("\n  Validation errors:")
        for err in analyzer.errors[:10]:
            print(f"    {err['path']}: {err['reason']}")
        if len(analyzer.errors) > 10:
            print(f"    ... and {len(analyzer.errors) - 10} more")

    # Later tasks will add: derive, figures, thesis-table, statistical logic
    if args.thesis_table:
        verdicts = analyzer.compute_hypothesis_verdicts()
        _print_hypothesis_table(verdicts)
        # Generate LaTeX table for Chapter 5
        latex = analyzer.generate_latex_table()
        if latex:
            tex_path = analyzer.results_dir / "analysis" / "comparison_table.tex"
            print(f"\n  [thesis-table] LaTeX table written to {tex_path}")

    if args.figures:
        analyzer.compute_derived_metrics()
        figure_name = args.figure if args.figure else "all"
        analyzer.generate_figures(figure_name=figure_name)
        print("  [figures] Generation complete")

    if args.statistical:
        print("\n  [statistical] Not yet implemented (task 8.7)")

    if args.detailed:
        per_regime = analyzer.compute_per_regime_analysis()
        _print_per_regime(per_regime)

    if args.h_sweep:
        h_sweep = analyzer.compute_h_sweep_table()
        _print_h_sweep(h_sweep)

    if args.shot_sensitivity:
        shot_data = analyzer.compute_shot_sensitivity()
        _print_shot_sensitivity(shot_data)

    if args.json:
        analyzer.export_full_analysis(Path(args.json))
        print(f"  [json] Full analysis exported to {args.json}")


def _print_hypothesis_table(verdicts: list[dict]) -> None:
    """Print hypothesis validation table to stdout."""
    if not verdicts:
        print("\n  [thesis-table] No hypothesis verdicts (no data loaded)")
        return

    # Header
    print(
        "\n  ┌─────────────────────────────────────────────────────────────"
        "──────────────────────────────────────────────────┐"
    )
    print(
        f"  │ {'ID':<4} │ {'Description':<50} │ {'Configs':<30} │ "
        f"{'Δ Observed':<12} │ {'Verdict':<12} │"
    )
    print(
        "  ├─────────────────────────────────────────────────────────────"
        "──────────────────────────────────────────────────┤"
    )

    for v in verdicts:
        h_id = v["hypothesis_id"]
        desc = v["description"][:50]
        configs = ", ".join(v["configs_tested"])[:30]
        delta = v["observed_delta"]
        delta_str = f"{delta:.5f}" if delta is not None else "N/A"
        verdict = v["verdict"]

        # Color-code verdict for terminal
        if verdict == "CONFIRMED":
            verdict_display = f"✅ {verdict}"
        elif verdict == "REFUTED":
            verdict_display = f"❌ {verdict}"
        else:
            verdict_display = f"⚠️  {verdict}"

        print(
            f"  │ {h_id:<4} │ {desc:<50} │ {configs:<30} │ "
            f"{delta_str:<12} │ {verdict_display:<12} │"
        )

    print(
        "  └─────────────────────────────────────────────────────────────"
        "──────────────────────────────────────────────────┘"
    )

    # Summary counts
    confirmed = sum(1 for v in verdicts if v["verdict"] == "CONFIRMED")
    refuted = sum(1 for v in verdicts if v["verdict"] == "REFUTED")
    inconclusive = sum(1 for v in verdicts if v["verdict"] == "INCONCLUSIVE")
    print(
        f"\n  Summary: {confirmed} CONFIRMED, {refuted} REFUTED, "
        f"{inconclusive} INCONCLUSIVE (total: {len(verdicts)})"
    )


def _print_per_regime(per_regime: dict) -> None:
    """Print per-regime analysis showing top-5 configs per regime."""
    if not per_regime:
        print("\n  [detailed] No per-regime data available")
        return

    for regime_name, data in per_regime.items():
        n_entries = data.get("n_entries", 0)
        h_range = data.get("h_range", (0, 0))
        ranking = data.get("ranking", [])
        best = data.get("best_config", "N/A")
        best_gap = data.get("best_delta_e_gap")

        print(
            f"\n  ── {regime_name.upper()} regime "
            f"(h ∈ [{h_range[0]:.2f}, {h_range[1]:.2f}]) "
            f"── {n_entries} entries ──"
        )

        if not ranking:
            print("    No data in this regime")
            continue

        best_str = f"{best_gap:.5f}" if best_gap is not None else "N/A"
        print(f"    Best: {best} (ΔE/gap = {best_str})")
        print(f"    {'Rank':<5} {'Config':<28} {'Mean ΔE/gap':<14} {'Std':<10} {'N':<4}")
        print(f"    {'─' * 5} {'─' * 28} {'─' * 14} {'─' * 10} {'─' * 4}")

        top_n = ranking[:5]
        for i, entry in enumerate(top_n, start=1):
            config_id = entry["config_id"]
            mean_val = f"{entry['mean_delta_e_gap']:.5f}"
            std_val = f"{entry['std']:.5f}"
            n = entry["n"]
            print(f"    {i:<5} {config_id:<28} {mean_val:<14} {std_val:<10} {n:<4}")


def _print_h_sweep(h_sweep: dict) -> None:
    """Print h-sweep table with h-values as rows, configs as columns."""
    if not h_sweep:
        print("\n  [h-sweep] No h-sweep data available")
        return

    # Collect all h-values across configs
    all_h: set[float] = set()
    for config_table in h_sweep.values():
        all_h.update(config_table.keys())

    if not all_h:
        print("\n  [h-sweep] No h-value data found")
        return

    h_values = sorted(all_h)
    configs = list(h_sweep.keys())

    print("\n  ── H-SWEEP TABLE (Mean ΔE/gap) ──")
    # Header
    header = f"    {'h':<8}"
    for cid in configs:
        short = cid[:16]
        header += f" {short:<18}"
    print(header)
    print(f"    {'─' * 8}" + f" {'─' * 18}" * len(configs))

    # Rows
    for h_val in h_values:
        row = f"    {h_val:<8.3f}"
        for cid in configs:
            stats = h_sweep[cid].get(h_val)
            if stats is not None:
                val = stats["mean"]
                imp = stats.get("improvement_vs_raw")
                if imp is not None and cid != "C0_raw":
                    row += f" {val:.5f} ({imp:+.0%})"[:18].ljust(19)
                else:
                    row += f" {val:.5f}".ljust(19)
            else:
                row += " ---".ljust(19)
        print(row)


def _print_shot_sensitivity(shot_data: dict) -> None:
    """Print shot sensitivity table for each config."""
    if not shot_data:
        print("\n  [shot-sensitivity] No shot sensitivity data available")
        return

    print("\n  ── SHOT SENSITIVITY ──")

    for config_id, entries in sorted(shot_data.items()):
        if not entries:
            continue
        # Only print configs with multiple shot levels
        if len(entries) < 2:
            continue

        print(f"\n    {config_id}:")
        print(f"      {'Shots':<12} {'Mean ΔE/gap':<14} {'Std':<12} {'N':<4}")
        print(f"      {'─' * 12} {'─' * 14} {'─' * 12} {'─' * 4}")

        for entry in entries:
            shots = entry["shots"]
            mean_val = f"{entry['mean_delta_e_gap']:.5f}"
            std_val = f"{entry['std']:.5f}"
            n = entry["n"]
            print(f"      {shots:<12} {mean_val:<14} {std_val:<12} {n:<4}")


if __name__ == "__main__":
    main()
