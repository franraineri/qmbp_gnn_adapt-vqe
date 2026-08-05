"""VQE Quality Predictor — Predict convergence probability BEFORE running.

Uses historical data from ResultIndex to predict whether a given (model,
topology, N, p, h_range) configuration will achieve ΔE/gap < 5%.

Features used for prediction:
  - N/p ratio (circuit expressibility proxy)
  - n_params/n_edges ratio (parameter density per interaction)
  - topology complexity (coordination number, gate_neighborhood_cv)
  - h-range position relative to known valid regime boundary
  - historical pass_rate for similar configurations (recency-weighted)

This saves compute by aborting configs that are predicted to fail.

Integration points:
  - PipelineRunner: gate on predicted pass probability before VQE sweep
  - AcceleratedVQE: estimate h_min via `report.estimated_h_min`
  - PreflightChecker: emit Issue if pass_probability < 0.3
  - VariantRunner: sort configs by pass_probability (descending)
  - project_health: suggest next experiments via suggest_viable_configs()

Usage:
    from qmbp_simulation.analysis.quality_predictor import (
        QualityPredictor, PredictionReport,
    )

    predictor = QualityPredictor()  # loads from ResultIndex
    report = predictor.predict(
        model="tfim", topology="heavy_hex", n_qubits=10, p_layers=2,
        h_min=1.0, h_max=3.5,
    )
    print(report)
    # PredictionReport(pass_probability=0.72, confidence="medium",
    #   recommendation="PROCEED", estimated_h_min=1.30,
    #   confidence_interval=(0.55, 0.85), reasons=[...])

    if report.should_run:
        # proceed with VQE
        ...

    # Batch mode for campaign planning
    reports = predictor.batch_predict(configs)
    viable = predictor.suggest_viable_configs(model="tfim")

    # Priority sorting for variant runners
    sorted_configs = predictor.prioritize_configs(configs)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Topology metadata (coordination numbers, typical CX overhead)
# ═══════════════════════════════════════════════════════════════════════════════

TOPOLOGY_COMPLEXITY: dict[str, dict[str, float]] = {
    "chain_1d": {"coordination": 2.0, "cx_factor": 1.0, "difficulty": 0.2},
    "ladder": {"coordination": 3.0, "cx_factor": 1.5, "difficulty": 0.4},
    "square": {"coordination": 4.0, "cx_factor": 2.0, "difficulty": 0.6},
    "triangular": {"coordination": 6.0, "cx_factor": 3.0, "difficulty": 0.8},
    "heavy_hex": {"coordination": 2.57, "cx_factor": 1.3, "difficulty": 0.5},
    "kagome": {"coordination": 4.0, "cx_factor": 2.5, "difficulty": 0.9},
}

# Approximate edge count per qubit for each topology (used for param density)
_EDGES_PER_QUBIT: dict[str, float] = {
    "chain_1d": 1.0,
    "ladder": 1.5,
    "square": 2.0,
    "triangular": 3.0,
    "heavy_hex": 1.29,
    "kagome": 2.0,
}

# Recency half-life in days — runs older than this get half weight
_RECENCY_HALF_LIFE_DAYS: float = 30.0


@dataclass
class PredictionReport:
    """Result of a VQE quality prediction.

    Attributes
    ----------
    pass_probability : float
        Estimated probability (0-1) that the pipeline will achieve ΔE/gap < 5%.
    confidence : str
        "high", "medium", or "low" — based on amount of historical data.
    confidence_interval : tuple[float, float]
        (lower, upper) bounds on pass_probability using Wilson score interval.
        Provides calibrated uncertainty: "70% with 3 runs" vs "70% with 20 runs".
    recommendation : str
        "PROCEED", "CAUTION", or "ABORT".
    should_run : bool
        Convenience flag: True if recommendation != "ABORT".
    reasons : list[str]
        Human-readable explanations for the prediction.
    similar_runs : int
        Number of historical runs used for this prediction.
    estimated_time_s : float
        Rough estimate of wall-clock time based on similar runs.
    estimated_h_min : float
        Estimated minimum h-value where VQE converges for this config.
        Used by AcceleratedVQE to set sweep boundaries.
    feature_vector : dict[str, float]
        The features used for prediction (for debugging/transparency).
    """

    pass_probability: float = 0.0
    confidence: str = "low"
    confidence_interval: tuple[float, float] = (0.0, 1.0)
    recommendation: str = "CAUTION"
    should_run: bool = True
    reasons: list[str] = field(default_factory=list)
    similar_runs: int = 0
    estimated_time_s: float = 0.0
    estimated_h_min: float = 0.0
    feature_vector: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        emoji = {"PROCEED": "✅", "CAUTION": "⚠️", "ABORT": "❌"}
        ci_lo, ci_hi = self.confidence_interval
        lines = [
            f"{emoji.get(self.recommendation, '?')} {self.recommendation} "
            f"(pass_prob={self.pass_probability:.0%} "
            f"[{ci_lo:.0%}, {ci_hi:.0%}], confidence={self.confidence})",
        ]
        for reason in self.reasons:
            lines.append(f"  • {reason}")
        if self.estimated_h_min > 0:
            lines.append(f"  📐 Estimated h_min (valid regime): {self.estimated_h_min:.2f}")
        if self.estimated_time_s > 0:
            lines.append(f"  ⏱️  Estimated time: {self.estimated_time_s:.0f}s")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict for result persistence."""
        return {
            "pass_probability": self.pass_probability,
            "confidence": self.confidence,
            "confidence_interval": list(self.confidence_interval),
            "recommendation": self.recommendation,
            "should_run": self.should_run,
            "reasons": self.reasons,
            "similar_runs": self.similar_runs,
            "estimated_time_s": self.estimated_time_s,
            "estimated_h_min": self.estimated_h_min,
            "feature_vector": self.feature_vector,
        }


class QualityPredictor:
    """Predicts VQE convergence probability from historical run data.

    Loads historical results from ResultIndex and builds a simple
    feature-based predictor (no ML needed — Bayesian update + kNN from
    actual data with recency weighting). Intentionally simple and interpretable.

    Parameters
    ----------
    root : Path | None
        Project root (auto-detected if None).
    min_similar_runs : int
        Minimum similar runs required for "high" confidence.

    Integration Examples
    --------------------
    # In PipelineRunner — gate execution on prediction
    predictor = QualityPredictor()
    report = predictor.predict(model=m, topology=t, n_qubits=n, p_layers=p)
    if not report.should_run:
        logger.warning("Skipping: %s", report)
        return

    # In AcceleratedVQE — estimate h_min boundary
    report = predictor.predict(...)
    h_min_safe = report.estimated_h_min

    # In VariantRunner — sort configs by likelihood
    sorted_configs = predictor.prioritize_configs(configs)

    # In PreflightChecker — emit diagnostic issue
    from qmbp_simulation.framework.preflight import Issue, Severity
    if report.pass_probability < 0.3:
        issues.append(Issue(Severity.ERROR, "Quality predictor: ABORT"))
    """

    def __init__(self, root: Path | None = None, min_similar_runs: int = 5) -> None:
        self._root = root or Path(__file__).resolve().parents[3]
        self._min_similar = min_similar_runs
        self._entries: list[dict[str, Any]] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load historical run data from ResultIndex."""
        try:
            from qmbp_simulation.framework.result_index import ResultIndex

            idx = ResultIndex(self._root)
            self._entries = idx.valid_entries
            logger.info("QualityPredictor: loaded %d historical runs", len(self._entries))
        except Exception as e:
            logger.warning("QualityPredictor: could not load ResultIndex: %s", e)
            self._entries = []

    @staticmethod
    def _recency_weight(timestamp_str: str) -> float:
        """Compute exponential decay weight based on entry age.

        Recent runs are more representative (post-bugfix, updated pipeline).
        Weight = exp(-age_days * ln(2) / half_life).
        Returns 1.0 if timestamp is missing/invalid.
        """
        if not timestamp_str:
            return 0.5  # Unknown age — half weight
        try:
            ts = datetime.fromisoformat(timestamp_str)
            age_days = (datetime.now() - ts).total_seconds() / 86400.0
            if age_days < 0:
                age_days = 0.0
            return math.exp(-age_days * math.log(2) / _RECENCY_HALF_LIFE_DAYS)
        except (ValueError, TypeError):
            return 0.5

    @staticmethod
    def _wilson_interval(
        n_success: float, n_total: float, z: float = 1.96
    ) -> tuple[float, float]:
        """Wilson score confidence interval for binomial proportion.

        More robust than normal approximation for small samples.
        Returns (lower, upper) bounds on pass probability.
        """
        if n_total <= 0:
            return (0.0, 1.0)
        p_hat = n_success / n_total
        denom = 1 + z**2 / n_total
        center = (p_hat + z**2 / (2 * n_total)) / denom
        spread = z * math.sqrt(
            (p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total
        ) / denom
        lower = max(0.0, center - spread)
        upper = min(1.0, center + spread)
        return (lower, upper)

    def _compute_features(
        self,
        model: str,
        topology: str,
        n_qubits: int,
        p_layers: int,
        h_min: float,
        h_max: float,
    ) -> dict[str, float]:
        """Compute prediction features for a configuration."""
        topo_info = TOPOLOGY_COMPLEXITY.get(topology, TOPOLOGY_COMPLEXITY["chain_1d"])

        # Model-aware critical point (TFIM: h_c≈1.0, others vary)
        H_CRITICAL_MAP = {
            "tfim": 1.0,
            "tfim_longitudinal": 1.0,
            "tfim_frustrated": 0.8,
            "tfim_bond_resolved": 1.0,
            "heisenberg": 0.0,
            "heisenberg_transverse": 1.0,
            "xy": 0.0,
        }
        h_critical = H_CRITICAL_MAP.get(model, 1.0)
        h_fraction_below_hc = max(0.0, h_critical - h_min) / (h_max - h_min + 1e-10)

        # Valid regime boundary from preflight
        h_min_safe = 1.0  # Default
        try:
            from qmbp_simulation.framework.preflight import get_regime_threshold

            h_min_safe = get_regime_threshold(topology, n_qubits, p_layers)
        except (ImportError, ValueError):
            pass
        # Ensure h_min_safe is positive
        if h_min_safe <= 0:
            h_min_safe = 1.0

        fraction_in_valid_regime = max(0.0, h_max - max(h_min, h_min_safe)) / (
            h_max - h_min + 1e-10
        )

        # Model-specific difficulty multiplier
        MODEL_DIFFICULTY = {
            "tfim": 0.0,
            "tfim_longitudinal": 0.05,
            "tfim_frustrated": 0.3,
            "tfim_bond_resolved": 0.0,
            "heisenberg": 0.4,
            "heisenberg_transverse": 0.3,
            "xy": 0.4,
            "kitaev": 0.5,
        }
        model_difficulty = MODEL_DIFFICULTY.get(model, 0.2)

        # Parameter density: n_params / n_edges
        edges_per_qubit = _EDGES_PER_QUBIT.get(topology, 1.0)
        n_edges = n_qubits * edges_per_qubit
        n_params = n_qubits * p_layers * 2  # Rough param count
        param_density = n_params / max(n_edges, 1.0)

        return {
            "n_over_p": n_qubits / max(p_layers, 1),
            "topo_difficulty": topo_info["difficulty"],
            "model_difficulty": model_difficulty,
            "coordination": topo_info["coordination"],
            "cx_factor": topo_info["cx_factor"],
            "h_fraction_below_hc": h_fraction_below_hc,
            "fraction_in_valid_regime": fraction_in_valid_regime,
            "h_min_safe": h_min_safe,
            "n_params": n_params,
            "param_density": param_density,
            "n_edges": n_edges,
        }

    def _find_similar_runs(
        self,
        model: str,
        topology: str,
        n_qubits: int,
        p_layers: int,
    ) -> list[dict[str, Any]]:
        """Find historical runs with similar configuration."""
        similar = []
        for entry in self._entries:
            # Exact match on model + topology
            if entry.get("model") != model:
                continue
            if entry.get("topology") != topology:
                continue
            # Allow ±2 qubits and same p
            e_n = entry.get("n_qubits", 0)
            e_p = entry.get("p_layers", 0)
            if abs(e_n - n_qubits) <= 2 and e_p == p_layers:
                similar.append(entry)
        return similar

    def _estimate_h_min_from_history(
        self,
        similar: list[dict[str, Any]],
        features: dict[str, float],
    ) -> float:
        """Estimate minimum h where VQE converges from historical data.

        Strategy:
        1. If we have similar runs that passed (pass_rate >= 0.8), use
           the preflight boundary as baseline (most calibrated source).
        2. For AcceleratedVQE integration — this provides a data-driven
           h_min that avoids wasting compute in the non-convergent regime.

        Returns the estimated h_min (always > 0).
        """
        h_min_safe = features["h_min_safe"]

        if not similar:
            return h_min_safe

        # Look at passing runs — their configs tell us where convergence works
        passing_runs = [
            e for e in similar if e.get("pass_rate", 0) >= 0.8
        ]
        failing_runs = [
            e for e in similar if e.get("pass_rate", 0) < 0.3
        ]

        if passing_runs and not failing_runs:
            # All similar runs pass — regime boundary is conservative, trust it
            return max(0.5, h_min_safe * 0.9)

        if failing_runs and not passing_runs:
            # All fail — push boundary higher
            return h_min_safe * 1.3

        # Mixed results — use preflight boundary (best calibrated)
        return h_min_safe

    def predict(
        self,
        model: str = "tfim",
        topology: str = "chain_1d",
        n_qubits: int = 10,
        p_layers: int = 2,
        h_min: float = 1.0,
        h_max: float = 3.5,
    ) -> PredictionReport:
        """Predict whether this configuration will pass.

        Parameters
        ----------
        model : str
            Hamiltonian model name.
        topology : str
            Lattice topology.
        n_qubits : int
            System size.
        p_layers : int
            HVA depth.
        h_min, h_max : float
            Field range for the sweep.

        Returns
        -------
        PredictionReport
            Prediction with probability, confidence, CI, estimated_h_min,
            and recommendations. The report is JSON-serializable via
            ``report.to_dict()`` for persistence in result envelopes.
        """
        features = self._compute_features(model, topology, n_qubits, p_layers, h_min, h_max)
        similar = self._find_similar_runs(model, topology, n_qubits, p_layers)

        report = PredictionReport(feature_vector=features, similar_runs=len(similar))
        reasons: list[str] = []

        # ── Rule 1: Historical pass rate (recency-weighted) ───────────────
        if similar:
            weights = np.array([
                self._recency_weight(e.get("timestamp", "")) for e in similar
            ])
            pass_rates = np.array([e.get("pass_rate", 0.0) for e in similar])
            total_weight = float(weights.sum())
            if total_weight > 0:
                historical_rate = float(np.average(pass_rates, weights=weights))
            else:
                historical_rate = float(np.mean(pass_rates))
            report.pass_probability = historical_rate

            # Estimate time from similar runs (recency-weighted median)
            times = [
                e.get("elapsed_s", 0.0) for e in similar
                if e.get("elapsed_s", 0) > 0
            ]
            if times:
                report.estimated_time_s = float(np.median(times))

            # Wilson confidence interval
            # Effective sample size from weighted observations
            effective_n = total_weight if total_weight > 0 else float(len(similar))
            n_success = historical_rate * effective_n
            report.confidence_interval = self._wilson_interval(n_success, effective_n)

            reasons.append(
                f"Historical pass rate: {historical_rate:.0%} "
                f"({len(similar)} runs, recency-weighted)"
            )
        else:
            # No history — use heuristic model (Bayesian prior)
            report.pass_probability = self._heuristic_estimate(features)
            # Wide CI when no data
            report.confidence_interval = (
                max(0.0, report.pass_probability - 0.3),
                min(1.0, report.pass_probability + 0.3),
            )
            reasons.append("No historical data — using heuristic estimate")

        # ── Rule 2: Valid regime check ────────────────────────────────────
        if features["fraction_in_valid_regime"] < 0.5:
            report.pass_probability *= 0.5
            reasons.append(
                f"Only {features['fraction_in_valid_regime']:.0%} of h-range is in "
                f"valid regime (h_min_safe={features['h_min_safe']:.2f})"
            )

        # ── Rule 3: Topology difficulty ───────────────────────────────────
        if features["topo_difficulty"] > 0.7 and p_layers <= 2:
            report.pass_probability *= 0.7
            reasons.append(
                f"High-difficulty topology ({topology}) with shallow ansatz (p={p_layers})"
            )

        # ── Rule 4: System size vs depth ──────────────────────────────────
        # Only penalize if truly under-parameterized (global HVA: 2*p params).
        # Bond-resolved HVA (2N-1 params) is NOT under-parameterized at large N.
        is_bond_resolved = "bond_resolved" in model
        if n_qubits > 16 and p_layers <= 1 and not is_bond_resolved:
            report.pass_probability *= 0.6
            reasons.append(f"Large system (N={n_qubits}) with p=1 may lack expressibility")

        # ── Rule 5: Parameter density ─────────────────────────────────────
        if features["param_density"] < 1.5 and features["topo_difficulty"] > 0.4:
            report.pass_probability *= 0.85
            reasons.append(
                f"Low parameter density ({features['param_density']:.1f} params/edge) "
                f"for complex topology"
            )

        # ── Confidence and recommendation ─────────────────────────────────
        if len(similar) >= self._min_similar:
            report.confidence = "high"
        elif len(similar) >= 2:
            report.confidence = "medium"
        else:
            report.confidence = "low"

        if report.pass_probability >= 0.6:
            report.recommendation = "PROCEED"
        elif report.pass_probability >= 0.15:
            report.recommendation = "CAUTION"
        else:
            report.recommendation = "ABORT"

        report.should_run = report.recommendation != "ABORT"
        report.reasons = reasons

        # ── Estimated h_min (for AcceleratedVQE integration) ──────────────
        report.estimated_h_min = self._estimate_h_min_from_history(similar, features)

        return report

    def _heuristic_estimate(self, features: dict[str, float]) -> float:
        """Estimate pass probability from features alone (no historical data).

        Uses a Bayesian-inspired prior calibrated against project status data:
          - chain_1d p=2: ~80% base rate
          - triangular p=2: ~50% base rate
          - valid regime fraction is the strongest predictor
          - param_density modulates for under-parameterized configs
        """
        base = 0.8

        # Topology penalty
        base -= features["topo_difficulty"] * 0.3

        # Model penalty (Heisenberg/XY harder than TFIM)
        base -= features.get("model_difficulty", 0.0) * 0.4

        # Valid regime bonus/penalty
        base *= (0.5 + 0.5 * features["fraction_in_valid_regime"])

        # Parameter density correction
        pd = features.get("param_density", 2.0)
        if pd < 1.5:
            base *= (0.7 + 0.2 * pd)  # penalty for under-parameterized

        # p=1 penalty for large N — only if truly under-parameterized
        # Bond-resolved has 2N-1 params regardless of p, so it's always expressive
        if features["n_over_p"] > 15 and features.get("param_density", 2.0) < 2.0:
            base *= 0.7

        return float(np.clip(base, 0.05, 0.99))

    def batch_predict(
        self,
        configs: list[dict[str, Any]],
    ) -> list[PredictionReport]:
        """Predict quality for multiple configurations.

        Parameters
        ----------
        configs : list[dict]
            Each dict must have keys: model, topology, n_qubits, p_layers,
            h_min, h_max.

        Returns
        -------
        list[PredictionReport]
            One report per config, same order as input.
        """
        return [
            self.predict(
                model=c.get("model", "tfim"),
                topology=c.get("topology", "chain_1d"),
                n_qubits=c.get("n_qubits", 10),
                p_layers=c.get("p_layers", 2),
                h_min=c.get("h_min", 1.0),
                h_max=c.get("h_max", 3.5),
            )
            for c in configs
        ]

    def prioritize_configs(
        self,
        configs: list[dict[str, Any]],
        *,
        exclude_abort: bool = False,
    ) -> list[dict[str, Any]]:
        """Sort configs by predicted pass probability (highest first).

        Useful for VariantRunner integration — run the most likely configs
        first for faster feedback loops.

        Parameters
        ----------
        configs : list[dict]
            Configs with standard keys (model, topology, n_qubits, p_layers, ...).
        exclude_abort : bool
            If True, filter out configs predicted to ABORT.

        Returns
        -------
        list[dict]
            Sorted configs, each enriched with '_prediction' key containing
            the full PredictionReport.to_dict().
        """
        reports = self.batch_predict(configs)
        enriched = []
        for cfg, report in zip(configs, reports):
            if exclude_abort and not report.should_run:
                continue
            enriched_cfg = {**cfg, "_prediction": report.to_dict()}
            enriched.append((report.pass_probability, enriched_cfg))
        enriched.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in enriched]

    def suggest_viable_configs(
        self,
        model: str = "tfim",
        n_qubits_range: tuple[int, int] = (4, 20),
        p_range: tuple[int, int] = (1, 3),
        min_pass_prob: float = 0.6,
    ) -> list[dict[str, Any]]:
        """Suggest configurations likely to succeed.

        Scans the config space and returns configs predicted to pass.
        Useful for planning experiment campaigns and project_health
        "next recommended experiments" generation.

        Parameters
        ----------
        model : str
            Hamiltonian model to scan.
        n_qubits_range : tuple
            (min_n, max_n) inclusive. Scanned in steps of 2.
        p_range : tuple
            (min_p, max_p) inclusive.
        min_pass_prob : float
            Minimum pass probability to include.

        Returns
        -------
        list[dict]
            Configs sorted by pass_probability descending. Each includes
            'confidence_interval' and 'estimated_h_min' for downstream use.
        """
        from qmbp_simulation.models.constants import SUPPORTED_TOPOLOGIES

        viable = []
        for topo in SUPPORTED_TOPOLOGIES:
            for n in range(n_qubits_range[0], n_qubits_range[1] + 1, 2):
                for p in range(p_range[0], p_range[1] + 1):
                    report = self.predict(
                        model=model, topology=topo, n_qubits=n, p_layers=p
                    )
                    if report.pass_probability >= min_pass_prob:
                        viable.append({
                            "model": model,
                            "topology": topo,
                            "n_qubits": n,
                            "p_layers": p,
                            "pass_probability": report.pass_probability,
                            "confidence_interval": list(report.confidence_interval),
                            "estimated_h_min": report.estimated_h_min,
                            "recommendation": report.recommendation,
                        })
        return sorted(viable, key=lambda x: x["pass_probability"], reverse=True)

    def calibration_report(self) -> dict[str, Any]:
        """Evaluate predictor accuracy against actual historical outcomes.

        Performs leave-one-out cross-validation: for each historical run,
        predicts its outcome using all OTHER runs, then compares against
        the actual pass_rate. This reveals systematic biases.

        Returns
        -------
        dict
            Calibration statistics:
            - n_evaluated: number of runs assessed
            - mean_absolute_error: average |predicted - actual|
            - bias: mean(predicted - actual) — positive = overconfident
            - brier_score: mean((predicted - actual)^2)
            - calibration_bins: list of (predicted_range, actual_rate, count)
            - accuracy_at_threshold: fraction correctly classified at 0.3/0.7

        Use this to tune heuristic weights or identify model/topology
        combos where the predictor is systematically wrong.
        """
        if len(self._entries) < 5:
            return {"n_evaluated": 0, "error": "insufficient data (<5 entries)"}

        predictions = []
        actuals = []

        for i, entry in enumerate(self._entries):
            model = entry.get("model", "")
            topology = entry.get("topology", "")
            n_qubits = entry.get("n_qubits", 0)
            p_layers = entry.get("p_layers", 0)
            actual_rate = entry.get("pass_rate", 0.0)

            if not model or not topology or not n_qubits or not p_layers:
                continue

            # Find similar runs EXCLUDING this entry
            similar = []
            for j, other in enumerate(self._entries):
                if j == i:
                    continue
                if other.get("model") != model:
                    continue
                if other.get("topology") != topology:
                    continue
                e_n = other.get("n_qubits", 0)
                e_p = other.get("p_layers", 0)
                if abs(e_n - n_qubits) <= 2 and e_p == p_layers:
                    similar.append(other)

            # Predict using similar (or heuristic if none)
            features = self._compute_features(
                model, topology, n_qubits, p_layers, h_min=1.0, h_max=3.5
            )
            if similar:
                weights = np.array([
                    self._recency_weight(e.get("timestamp", "")) for e in similar
                ])
                pass_rates = np.array([e.get("pass_rate", 0.0) for e in similar])
                total_w = float(weights.sum())
                if total_w > 0:
                    predicted = float(np.average(pass_rates, weights=weights))
                else:
                    predicted = float(np.mean(pass_rates))
            else:
                predicted = self._heuristic_estimate(features)

            predictions.append(predicted)
            actuals.append(float(actual_rate))

        if not predictions:
            return {"n_evaluated": 0, "error": "no evaluable entries"}

        preds = np.array(predictions)
        acts = np.array(actuals)
        n = len(preds)

        mae = float(np.mean(np.abs(preds - acts)))
        bias = float(np.mean(preds - acts))
        brier = float(np.mean((preds - acts) ** 2))

        # Accuracy at thresholds
        # "ABORT" = predicted < 0.3 → actual should be < 0.5
        abort_mask = preds < 0.3
        proceed_mask = preds >= 0.7
        abort_correct = float(np.mean(acts[abort_mask] < 0.5)) if abort_mask.any() else None
        proceed_correct = float(np.mean(acts[proceed_mask] >= 0.5)) if proceed_mask.any() else None

        # Calibration bins
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        cal_bins = []
        for lo, hi in bins:
            mask = (preds >= lo) & (preds < hi)
            if mask.any():
                cal_bins.append({
                    "range": f"[{lo:.1f}, {hi:.1f})",
                    "predicted_mean": float(preds[mask].mean()),
                    "actual_mean": float(acts[mask].mean()),
                    "count": int(mask.sum()),
                })

        return {
            "n_evaluated": n,
            "mean_absolute_error": mae,
            "bias": bias,
            "brier_score": brier,
            "abort_accuracy": abort_correct,
            "proceed_accuracy": proceed_correct,
            "calibration_bins": cal_bins,
        }
