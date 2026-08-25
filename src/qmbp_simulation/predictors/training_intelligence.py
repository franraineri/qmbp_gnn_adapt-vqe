"""Training Intelligence — automated retrain triggers, h-range validation, and data strategies.

This module provides the logic layer for deciding WHEN and HOW to retrain models,
enforcing data quality gates before training starts, and incorporating extrapolation
feedback into the training loop.

Integration points:
- `post_experiment_sync()` calls `check_retrain_triggers()` → auto-queues
- `runner_base.run()` calls `validate_training_readiness()` before training sections
- `run_multi_topology_training.py` calls `prepare_training_config()` for data strategy

Usage:
    from qmbp_simulation.predictors.training_intelligence import (
        check_retrain_triggers,
        validate_training_readiness,
        prepare_training_config,
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Auto-Retrain Triggers
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RetrainTrigger:
    """A triggered retrain recommendation with full context."""

    topology: str
    reason: str
    priority: int  # 1=critical, 2=high, 3=medium
    data_growth_pct: float  # How much NPZ data grew since last train
    n_values_available: list[int] = field(default_factory=list)
    n_training_points: int = 0
    current_model_pass_rate: float = 0.0
    h_range_coverage: float = 0.0  # [0,1] fraction of eval h-range covered by training
    command: str = ""  # CLI command to execute

    @property
    def should_auto_execute(self) -> bool:
        """True if this retrain is safe to auto-execute without human review."""
        return (
            self.priority <= 3
            and self.data_growth_pct >= 0.30
            and self.n_training_points >= 50
            and self.h_range_coverage >= 0.60
        )


def check_retrain_triggers(
    *,
    data_growth_threshold: float = 0.30,
    min_points_for_retrain: int = 50,
) -> list[RetrainTrigger]:
    """Check all zoo models for retrain triggers.

    Compares current NPZ data availability against what each model was
    trained on. Triggers fire when:
    - NPZ data grew >30% since model was trained
    - New N values are available that model doesn't cover
    - Model pass_rate_by_n shows 0% at some N where data exists
    - H-range of training data doesn't cover evaluation range

    Parameters
    ----------
    data_growth_threshold : float
        Fraction of data growth that triggers retrain (default: 30%).
    min_points_for_retrain : int
        Minimum training points needed to justify a retrain.

    Returns
    -------
    list[RetrainTrigger]
        Sorted by priority (lowest number = highest urgency).
    """
    from qmbp_simulation.predictors.model_zoo import _load_manifest

    dashboard_path = _PROJECT_ROOT / "data" / "model_quality_dashboard.json"
    if not dashboard_path.exists():
        return []

    with open(dashboard_path) as f:
        dashboard = json.load(f)

    configs = dashboard.get("configs", [])
    entries = _load_manifest()
    multi_n = [e for e in entries if e.n_qubits == 0 and e.p_layers == 1]

    # Group dashboard configs by topology
    configs_by_topo: dict[str, list[dict]] = {}
    for c in configs:
        configs_by_topo.setdefault(c["topology"], []).append(c)

    triggers: list[RetrainTrigger] = []

    for entry in multi_n:
        topo = entry.topology
        topo_configs = configs_by_topo.get(topo, [])
        if not topo_configs:
            continue

        # Current training data available
        total_points_available = sum(c.get("n_points", 0) for c in topo_configs)
        n_values_available = sorted(set(c["n_qubits"] for c in topo_configs))
        useful_configs = [c for c in topo_configs if c.get("training_utility") == "useful"]
        useful_points = sum(c.get("n_points", 0) for c in useful_configs)

        # Data growth detection
        data_growth = 0.0
        if entry.n_training_points > 0:
            data_growth = (
                total_points_available - entry.n_training_points
            ) / entry.n_training_points
        elif total_points_available >= min_points_for_retrain:
            data_growth = 1.0  # Model has 0 pts registered → definitely retrain

        # H-range coverage: what fraction of [2.0, 5.5] (typical eval range) is covered?
        h_ranges = [c.get("h_range", [0, 0]) for c in topo_configs if c.get("h_range")]
        h_range_coverage = _compute_h_range_coverage(h_ranges, eval_range=(2.0, 5.5))

        # N-value gap detection: model has 0% at N where useful data exists
        n_with_zero_pass = []
        if entry.pass_rate_by_n:
            for n_str, pr in entry.pass_rate_by_n.items():
                n = int(n_str)
                if float(pr) == 0 and n in n_values_available:
                    # Check if we have useful data for this N
                    has_useful = any(
                        c["n_qubits"] == n and c.get("training_utility") == "useful"
                        for c in topo_configs
                    )
                    if has_useful:
                        n_with_zero_pass.append(n)

        # Determine priority and reason
        priority = 4  # default: low
        reasons: list[str] = []

        if data_growth >= data_growth_threshold:
            priority = min(priority, 3)
            reasons.append(
                f"data grew {data_growth:.0%} ({entry.n_training_points}→{total_points_available}pts)"
            )

        if n_with_zero_pass:
            priority = min(priority, 2)
            reasons.append(f"0% pass at N={n_with_zero_pass} despite useful data")

        if h_range_coverage < 0.60:
            priority = min(priority, 2)
            reasons.append(f"h-range covers only {h_range_coverage:.0%} of eval range")

        if useful_points >= min_points_for_retrain and entry.pass_rate < 0.20:
            priority = min(priority, 2)
            reasons.append(
                f"pass_rate={entry.pass_rate:.0%} with {useful_points} useful pts available"
            )

        # QPT h_c accuracy trigger: if model fails near the critical point,
        # it cannot track the phase transition — a fundamental quality issue.
        # Uses get_h_critical() to find the topology-specific h_c, then checks
        # if pass_rate near h_c is significantly worse than far from h_c.
        try:
            from scripts.analysis.qpt_detection import get_h_critical

            h_c = get_h_critical(topo)
            if h_c is not None:
                # Check configs near h_c (within ±0.5) vs far from h_c
                near_hc = [
                    c
                    for c in topo_configs
                    if c.get("h_frontier") is not None and abs(c.get("h_frontier", 99) - h_c) < 0.5
                ]
                far_hc = [
                    c
                    for c in topo_configs
                    if c.get("h_frontier") is not None and abs(c.get("h_frontier", 99) - h_c) >= 0.5
                ]
                if near_hc and far_hc:
                    pass_near = np.mean([c.get("pass_rate_dual_criterion", 0) for c in near_hc])
                    pass_far = np.mean([c.get("pass_rate_dual_criterion", 0) for c in far_hc])
                    # If accuracy near h_c is >30% worse than far from h_c → retrain
                    if pass_far > 0.3 and (pass_far - pass_near) > 0.30:
                        priority = min(priority, 2)
                        reasons.append(
                            f"QPT h_c accuracy gap: pass_near_hc={pass_near:.0%} "
                            f"vs pass_far={pass_far:.0%} (h_c≈{h_c:.2f})"
                        )
        except Exception:
            pass  # QPT detection not available — skip this trigger

        if not reasons:
            continue  # No trigger for this model

        # Build CLI command
        cmd = (
            f".venv/bin/python scripts/experiment_runners/cross_topology/"
            f"run_multi_topology_training.py --topologies {topo} "
            f"--max-n {max(n_values_available) if n_values_available else 20}"
        )

        triggers.append(
            RetrainTrigger(
                topology=topo,
                reason=" | ".join(reasons),
                priority=priority,
                data_growth_pct=data_growth,
                n_values_available=n_values_available,
                n_training_points=total_points_available,
                current_model_pass_rate=entry.pass_rate,
                h_range_coverage=h_range_coverage,
                command=cmd,
            )
        )

    return sorted(triggers, key=lambda t: t.priority)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. H-Range Alignment Validation
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class HRangeValidation:
    """Result of h-range alignment validation."""

    is_valid: bool
    coverage: float  # [0,1] fraction of eval range covered
    training_range: tuple[float, float]
    eval_range: tuple[float, float]
    gap_regions: list[tuple[float, float]]  # uncovered intervals
    recommendations: list[str]


def validate_h_range_alignment(
    topology: str,
    *,
    eval_h_min: float = 2.0,
    eval_h_max: float = 5.5,
    min_coverage: float = 0.70,
) -> HRangeValidation:
    """Validate that training data h-range covers the evaluation range.

    The MPNN cannot extrapolate to h-values it never saw during training.
    This function checks if the training NPZ data covers the h-range used
    in model comparisons and extrapolation evaluations.

    Parameters
    ----------
    topology : str
        Topology to check.
    eval_h_min, eval_h_max : float
        The h-range used in evaluation/comparison (default: [2.0, 5.5]).
    min_coverage : float
        Minimum fraction of eval range that must be covered (default: 70%).

    Returns
    -------
    HRangeValidation
        Validation result with coverage metrics and recommendations.
    """
    dashboard_path = _PROJECT_ROOT / "data" / "model_quality_dashboard.json"
    if not dashboard_path.exists():
        return HRangeValidation(
            is_valid=False,
            coverage=0.0,
            training_range=(0, 0),
            eval_range=(eval_h_min, eval_h_max),
            gap_regions=[(eval_h_min, eval_h_max)],
            recommendations=["Dashboard not found. Run an experiment first."],
        )

    with open(dashboard_path) as f:
        dashboard = json.load(f)

    configs = [c for c in dashboard.get("configs", []) if c["topology"] == topology]
    if not configs:
        return HRangeValidation(
            is_valid=False,
            coverage=0.0,
            training_range=(0, 0),
            eval_range=(eval_h_min, eval_h_max),
            gap_regions=[(eval_h_min, eval_h_max)],
            recommendations=[f"No training data found for topology '{topology}'."],
        )

    # Compute union of all h-ranges for this topology
    h_ranges = [c.get("h_range", [0, 0]) for c in configs if c.get("h_range")]
    coverage = _compute_h_range_coverage(h_ranges, eval_range=(eval_h_min, eval_h_max))

    # Find the effective training range (union)
    all_h_min = min(r[0] for r in h_ranges) if h_ranges else 0
    all_h_max = max(r[1] for r in h_ranges) if h_ranges else 0
    training_range = (all_h_min, all_h_max)

    # Identify gap regions
    gap_regions = _find_gap_regions(h_ranges, eval_range=(eval_h_min, eval_h_max))

    # Recommendations
    recommendations: list[str] = []
    if coverage < min_coverage:
        recommendations.append(
            f"Training h-range covers only {coverage:.0%} of evaluation range "
            f"[{eval_h_min}, {eval_h_max}]. Need at least {min_coverage:.0%}."
        )
    if gap_regions:
        for lo, hi in gap_regions:
            recommendations.append(
                f"Gap: no training data in h=[{lo:.2f}, {hi:.2f}]. Run VQE to fill this region."
            )
    if all_h_min > eval_h_min + 0.5:
        recommendations.append(
            f"Training starts at h={all_h_min:.2f} but evaluation uses h≥{eval_h_min}. "
            f"Add data at lower h values."
        )

    return HRangeValidation(
        is_valid=coverage >= min_coverage,
        coverage=coverage,
        training_range=training_range,
        eval_range=(eval_h_min, eval_h_max),
        gap_regions=gap_regions,
        recommendations=recommendations,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Training Configuration with Extrapolation Feedback
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrainingConfig:
    """Optimized training configuration based on data analysis."""

    topologies: list[str]
    max_n: int
    recommended_epochs: int
    h_range: tuple[float, float]
    use_extrapolation_data: bool
    extrapolation_weight: float  # 0.0-1.0, how much to weight extrap data
    excluded_n_values: dict[str, list[int]]  # {topo: [N values to exclude]}
    n_total_points: int
    n_useful_points: int
    confidence: str  # "high", "medium", "low"
    warnings: list[str]
    data_sources: dict[str, int]  # {source_type: n_points}


def prepare_training_config(
    topologies: list[str] | None = None,
    *,
    max_n: int = 20,
    include_extrapolation: bool = True,
    target_eval_h_range: tuple[float, float] = (2.0, 5.5),
) -> TrainingConfig:
    """Prepare an optimized training configuration using all available intelligence.

    Analyzes dashboard, exclusion registry, extrapolation results, and h-range
    coverage to produce a training config that maximizes deployment quality.

    Parameters
    ----------
    topologies : list[str] | None
        Topologies to include. Default: all with useful data.
    max_n : int
        Maximum system size to include in training (default: 20).
    include_extrapolation : bool
        If True, incorporate extrapolation NPZ data (approximate quality) to
        improve generalization at larger N.
    target_eval_h_range : tuple[float, float]
        The h-range the model will be evaluated on.

    Returns
    -------
    TrainingConfig
        Complete training configuration with recommendations.
    """
    dashboard_path = _PROJECT_ROOT / "data" / "model_quality_dashboard.json"
    extrap_dir = _PROJECT_ROOT / "data" / "large_n_extrapolation"

    # Load dashboard
    dashboard: dict = {}
    if dashboard_path.exists():
        with open(dashboard_path) as f:
            dashboard = json.load(f)

    configs = dashboard.get("configs", [])

    # Determine topologies
    available_topos = sorted(set(c["topology"] for c in configs))
    if topologies is None:
        topologies = available_topos
    else:
        topologies = [t for t in topologies if t in available_topos]

    # Load exclusion registry
    excluded_n: dict[str, list[int]] = {}
    try:
        from qmbp_simulation.analysis.metrics import load_training_exclusions

        registry = load_training_exclusions()
        for exc in registry.get("excluded", []):
            topo = exc.get("topology", "")
            n = exc.get("n_qubits", 0)
            if topo and n:
                excluded_n.setdefault(topo, []).append(n)
    except Exception:
        pass

    # Analyze training data per topology
    warnings: list[str] = []
    n_total_points = 0
    n_useful_points = 0
    data_sources: dict[str, int] = {"npz_verified": 0, "npz_approximate": 0, "extrapolation": 0}
    all_h_values: list[float] = []

    for topo in topologies:
        topo_configs = [c for c in configs if c["topology"] == topo and c["n_qubits"] <= max_n]
        excluded_ns = excluded_n.get(topo, [])
        topo_configs = [c for c in topo_configs if c["n_qubits"] not in excluded_ns]

        for c in topo_configs:
            pts = c.get("n_points", 0)
            n_total_points += pts
            if c.get("training_utility") == "useful":
                n_useful_points += pts
                data_sources["npz_verified"] += pts
            elif c.get("training_utility") == "insufficient_signal":
                data_sources["npz_approximate"] += pts

            h_range = c.get("h_range", [])
            if h_range:
                all_h_values.extend([h_range[0], h_range[1]])

        # H-range validation for this topology
        h_val = validate_h_range_alignment(
            topo,
            eval_h_min=target_eval_h_range[0],
            eval_h_max=target_eval_h_range[1],
        )
        if not h_val.is_valid:
            for rec in h_val.recommendations:
                warnings.append(f"[{topo}] {rec}")

    # Extrapolation data analysis
    extrap_points = 0
    use_extrapolation = False
    extrap_weight = 0.0
    if include_extrapolation and extrap_dir.exists():
        for topo in topologies:
            for npz_file in extrap_dir.glob(f"{topo}_N*_p1.npz"):
                try:
                    stem = npz_file.stem
                    n_str = stem.split("_N")[1].split("_")[0]
                    n_val = int(n_str)
                    if n_val > max_n:
                        continue
                    data = np.load(str(npz_file), allow_pickle=True)
                    pts = len(data["h_values"])
                    # Only include if quality is reasonable
                    if "de_gaps" in data:
                        pass_rate = float((data["de_gaps"] < 0.10).mean())
                        if pass_rate >= 0.30:
                            extrap_points += pts
                            data_sources["extrapolation"] += pts
                except Exception:
                    continue

        if extrap_points > 0:
            use_extrapolation = True
            # Weight extrapolation data less (approximate quality)
            extrap_weight = min(0.5, extrap_points / max(n_useful_points, 1))

    # Compute effective h-range for training
    if all_h_values:
        h_range = (min(all_h_values), max(all_h_values))
    else:
        h_range = target_eval_h_range

    # Recommended epochs based on data size
    total_usable = n_useful_points + int(extrap_points * extrap_weight)
    if total_usable >= 500:
        recommended_epochs = 2000
    elif total_usable >= 200:
        recommended_epochs = 3000
    elif total_usable >= 100:
        recommended_epochs = 4000
    else:
        recommended_epochs = 5000
        warnings.append(f"Low data ({total_usable} usable points). Consider more VQE runs first.")

    # Confidence assessment
    if n_useful_points >= 300 and len(topologies) >= 3:
        confidence = "high"
    elif n_useful_points >= 100:
        confidence = "medium"
    else:
        confidence = "low"

    return TrainingConfig(
        topologies=topologies,
        max_n=max_n,
        recommended_epochs=recommended_epochs,
        h_range=h_range,
        use_extrapolation_data=use_extrapolation,
        extrapolation_weight=extrap_weight,
        excluded_n_values=excluded_n,
        n_total_points=n_total_points,
        n_useful_points=n_useful_points,
        confidence=confidence,
        warnings=warnings,
        data_sources=data_sources,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Validation Gate: Pre-Training Readiness Check
# ═══════════════════════════════════════════════════════════════════════════════


def validate_training_readiness(
    topology: str | list[str],
    *,
    min_useful_points: int = 50,
    min_h_coverage: float = 0.60,
) -> tuple[bool, list[str]]:
    """Gate check before starting any training.

    Returns (is_ready, issues). If is_ready=False, training should NOT proceed
    because the data quality is insufficient for a meaningful model.

    Parameters
    ----------
    topology : str | list[str]
        Topology or topologies to validate.
    min_useful_points : int
        Minimum number of useful training points required.
    min_h_coverage : float
        Minimum h-range coverage fraction.

    Returns
    -------
    tuple[bool, list[str]]
        (is_ready, list_of_issues). Empty list if ready.
    """
    if isinstance(topology, str):
        topologies = [topology]
    else:
        topologies = list(topology)

    issues: list[str] = []

    dashboard_path = _PROJECT_ROOT / "data" / "model_quality_dashboard.json"
    if not dashboard_path.exists():
        issues.append("Dashboard not found — cannot validate data quality.")
        return False, issues

    with open(dashboard_path) as f:
        dashboard = json.load(f)

    configs = dashboard.get("configs", [])

    for topo in topologies:
        topo_configs = [c for c in configs if c["topology"] == topo]
        if not topo_configs:
            issues.append(f"[{topo}] No training data found.")
            continue

        # Check useful points
        useful = [c for c in topo_configs if c.get("training_utility") == "useful"]
        useful_pts = sum(c.get("n_points", 0) for c in useful)
        if useful_pts < min_useful_points:
            issues.append(
                f"[{topo}] Only {useful_pts} useful points (need {min_useful_points}+). "
                f"Run more VQE experiments."
            )

        # Check h-range coverage
        h_val = validate_h_range_alignment(topo, min_coverage=min_h_coverage)
        if not h_val.is_valid:
            issues.append(
                f"[{topo}] H-range coverage={h_val.coverage:.0%} < {min_h_coverage:.0%}. "
                f"Training range [{h_val.training_range[0]:.1f}, {h_val.training_range[1]:.1f}] "
                f"doesn't cover eval range [{h_val.eval_range[0]:.1f}, {h_val.eval_range[1]:.1f}]."
            )

        # Check GT coherence for this topology
        gt_issues = _check_gt_coherence_for_topology(topo)
        if gt_issues:
            issues.append(f"[{topo}] GT coherence: {gt_issues}")

        # Check for contaminated data
        not_useful = [c for c in topo_configs if c.get("training_utility") == "not_useful"]
        if len(not_useful) > len(useful):
            issues.append(
                f"[{topo}] More bad configs ({len(not_useful)}) than good ({len(useful)}). "
                f"Data quality too low."
            )

    is_ready = len(issues) == 0
    return is_ready, issues


# ═══════════════════════════════════════════════════════════════════════════════
# Internal Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_h_range_coverage(
    h_ranges: list[list | tuple],
    eval_range: tuple[float, float],
) -> float:
    """Compute what fraction of eval_range is covered by the union of h_ranges."""
    if not h_ranges:
        return 0.0

    eval_lo, eval_hi = eval_range
    eval_width = eval_hi - eval_lo
    if eval_width <= 0:
        return 1.0

    # Discretize and compute coverage
    n_bins = 100
    bins = np.linspace(eval_lo, eval_hi, n_bins)
    covered = np.zeros(n_bins, dtype=bool)

    for h_range in h_ranges:
        if len(h_range) < 2:
            continue
        lo, hi = float(h_range[0]), float(h_range[1])
        covered |= (bins >= lo) & (bins <= hi)

    return float(covered.mean())


def _find_gap_regions(
    h_ranges: list[list | tuple],
    eval_range: tuple[float, float],
    min_gap_width: float = 0.3,
) -> list[tuple[float, float]]:
    """Find uncovered intervals within eval_range."""
    if not h_ranges:
        return [eval_range]

    eval_lo, eval_hi = eval_range
    n_bins = 200
    bins = np.linspace(eval_lo, eval_hi, n_bins)
    covered = np.zeros(n_bins, dtype=bool)

    for h_range in h_ranges:
        if len(h_range) < 2:
            continue
        lo, hi = float(h_range[0]), float(h_range[1])
        covered |= (bins >= lo) & (bins <= hi)

    # Find contiguous uncovered regions
    gaps: list[tuple[float, float]] = []
    in_gap = False
    gap_start = 0.0

    for i, (b, c) in enumerate(zip(bins, covered, strict=False)):
        if not c and not in_gap:
            gap_start = float(b)
            in_gap = True
        elif c and in_gap:
            gap_end = float(b)
            if gap_end - gap_start >= min_gap_width:
                gaps.append((gap_start, gap_end))
            in_gap = False

    if in_gap:
        gap_end = float(bins[-1])
        if gap_end - gap_start >= min_gap_width:
            gaps.append((gap_start, gap_end))

    return gaps


def _check_gt_coherence_for_topology(topology: str) -> str:
    """Quick GT coherence check for a specific topology."""
    gt_path = _PROJECT_ROOT / "data" / "ground_truth_cache.json"
    npz_dir = _PROJECT_ROOT / "data" / "multi_n_training"

    if not gt_path.exists() or not npz_dir.exists():
        return ""

    try:
        with open(gt_path) as f:
            raw = json.load(f)
        gt = raw.get("entries", raw)

        n_stale = 0
        for npz_file in npz_dir.glob(f"{topology}_N*_p1.npz"):
            data = np.load(str(npz_file), allow_pickle=True)
            if "e_exact" not in data or "h_values" not in data:
                continue
            h_vals = data["h_values"]
            e_exact_npz = data["e_exact"]

            parts = npz_file.stem.split("_")
            n_idx = next((i for i, p in enumerate(parts) if p.startswith("N")), None)
            if n_idx is None:
                continue
            n_val = int(parts[n_idx][1:])

            for i, h in enumerate(h_vals):
                key = f"{topology}|{n_val}|tfim_bond_resolved|{float(h):.2f}"
                if key in gt:
                    gt_e = gt[key].get("energy", gt[key].get("e_exact"))
                    if gt_e is not None and abs(float(e_exact_npz[i]) - float(gt_e)) > 1e-6:
                        n_stale += 1

        if n_stale > 0:
            return f"{n_stale} stale e_exact points (run validate_gt_npz_coherence(fix=True))"
    except Exception:
        pass

    return ""
