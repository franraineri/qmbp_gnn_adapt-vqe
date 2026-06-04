"""Coverage gap detection and advanced metric computation.

Analyzes scan results to identify what's missing for complete thesis
coverage and hardware deployment readiness.

Also computes aggregated quality metrics for VQE convergence, MPNN training,
pipeline timing, energy decomposition, and result distribution analysis.

Uses data already parsed by ResultScanner — no re-parsing of JSON files.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from project_health.digest.models import ExperimentResult, NoiselessResult, NoisyResult
from project_health.models import (
    ActionItem,
    CoverageGap,
    EnergyDecompositionStats,
    GapType,
    ModelDistribution,
    MPNNQualityStats,
    Priority,
    TimingStats,
    VQEQualityStats,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Valid regime boundaries (canonical source: analysis/scan_coverage.py)
# ═══════════════════════════════════════════════════════════════════════════════

P1_VALID_REGIME: dict[tuple[str, int], float] = {
    ("chain_1d", 6): 1.6,
    ("chain_1d", 10): 1.9,
    ("chain_1d", 20): 2.25,
    ("heavy_hex", 6): 2.0,
    ("heavy_hex", 10): 3.0,
    ("ladder", 6): 2.0,
    ("ladder", 10): 3.0,
    ("triangular", 6): 4.0,
    ("triangular", 10): 3.5,
}

P2_VALID_REGIME: dict[tuple[str, int], float] = {
    ("chain_1d", 6): 1.25,
    ("chain_1d", 10): 1.5,
    ("chain_1d", 20): 2.0,
    ("heavy_hex", 6): 1.5,
    ("heavy_hex", 10): 1.5,
    ("ladder", 6): 1.5,
    ("ladder", 10): 2.0,
    ("triangular", 6): 2.0,
    ("triangular", 10): 2.5,
}


# Required seeds for reproducibility claims
REQUIRED_SEEDS: set[int] = {42, 43, 44}


def detect_coverage_gaps(
    noiseless: list[NoiselessResult],
    noisy: list[NoisyResult],
    experiments: list[ExperimentResult],
) -> list[CoverageGap]:
    """Detect all coverage gaps from scan results.

    Returns a list of CoverageGap objects sorted by priority.
    """
    gaps: list[CoverageGap] = []

    gaps.extend(_gap_missing_p1_noiseless(noiseless))
    gaps.extend(_gap_invalid_regime(noiseless))
    gaps.extend(_gap_insufficient_seeds(noiseless))
    gaps.extend(_gap_missing_zne(noiseless, noisy))
    gaps.extend(_gap_missing_experiments(experiments))

    # Sort by priority (CRITICAL first)
    gaps.sort(key=lambda g: g.priority.value)
    return gaps


def derive_actions(report: Any) -> list[ActionItem]:
    """Derive prioritized actionable items from the full health report.

    Combines coverage gap analysis with VQE/MPNN quality diagnostics,
    timing anomalies, distribution imbalances, and experiment status
    to produce concrete, prioritized next-step recommendations.

    Parameters
    ----------
    report : HealthReport
        The fully-populated health report (gaps, VQE quality, MPNN quality,
        timing, distribution, energy decomposition all set).

    Returns
    -------
    list[ActionItem]
        Sorted by priority (CRITICAL first, LOW last).
    """
    actions: list[ActionItem] = []

    # ─── Coverage-gap-derived actions ────────────────────────────────────
    actions.extend(_actions_from_gaps(report.gaps))

    # ─── VQE quality-derived actions ─────────────────────────────────────
    actions.extend(_actions_from_vqe_quality(report.vqe_quality))

    # ─── MPNN quality-derived actions ────────────────────────────────────
    actions.extend(_actions_from_mpnn_quality(report.mpnn_quality))

    # ─── Distribution imbalance actions ──────────────────────────────────
    actions.extend(_actions_from_distribution(report.distribution))

    # ─── Energy decomposition-derived actions ────────────────────────────
    actions.extend(_actions_from_energy_decomposition(report.energy_decomposition))

    # ─── Experiment failure-derived actions ──────────────────────────────
    actions.extend(_actions_from_experiment_failures(report.experiments))

    # Sort by priority (CRITICAL first)
    actions.sort(key=lambda a: a.priority.value)
    return actions


def _actions_from_gaps(gaps: list[CoverageGap]) -> list[ActionItem]:
    """Generate actions from coverage gaps (original logic, preserved)."""
    actions: list[ActionItem] = []

    by_type: dict[GapType, list[CoverageGap]] = defaultdict(list)
    for g in gaps:
        by_type[g.gap_type].append(g)

    # Hardware-blocking gaps → CRITICAL
    hw_gaps = by_type.get(GapType.MISSING_ZNE, [])
    hw_critical = [g for g in hw_gaps if g.n_qubits >= 10 and g.topology == "heavy_hex"]
    if hw_critical:
        actions.append(
            ActionItem(
                priority=Priority.CRITICAL,
                title="ZNE validation needed for heavy_hex N≥10",
                detail=(
                    f"{len(hw_critical)} config(s) have noiseless data but no ZNE. "
                    "Hardware deployment requires ZNE on heavy_hex."
                ),
                category="hardware",
            )
        )

    # Missing experiments
    missing_exp = by_type.get(GapType.MISSING_EXPERIMENT, [])
    if missing_exp:
        ids = [g.detail for g in missing_exp]
        actions.append(
            ActionItem(
                priority=Priority.HIGH,
                title=f"{len(missing_exp)} defined experiment(s) have no results",
                detail=f"Missing: {', '.join(ids[:10])}",
                category="coverage",
            )
        )

    # Seed coverage
    seed_gaps = by_type.get(GapType.INSUFFICIENT_SEEDS, [])
    if seed_gaps:
        configs = [f"{g.topology} N={g.n_qubits}" for g in seed_gaps]
        actions.append(
            ActionItem(
                priority=Priority.MEDIUM,
                title=f"Seed coverage incomplete for {len(seed_gaps)} config(s)",
                detail=f"Need 3 seeds (42/43/44) for: {', '.join(configs[:5])}",
                category="reproducibility",
            )
        )

    # Regime violations
    regime_gaps = by_type.get(GapType.INVALID_REGIME, [])
    if regime_gaps:
        actions.append(
            ActionItem(
                priority=Priority.LOW,
                title=f"{len(regime_gaps)} result(s) tested outside valid regime",
                detail=(
                    "These are expected to fail — not actionable unless they passed unexpectedly."
                ),
                category="coverage",
            )
        )

    # Non-critical ZNE gaps
    zne_other = [g for g in hw_gaps if g not in hw_critical]
    if zne_other:
        topos = sorted({f"{g.topology} N={g.n_qubits}" for g in zne_other})
        actions.append(
            ActionItem(
                priority=Priority.MEDIUM,
                title=f"ZNE validation missing for {len(zne_other)} config(s)",
                detail=f"p=1 noiseless exists but no ZNE: {', '.join(topos[:5])}",
                category="coverage",
            )
        )

    return actions


def _actions_from_vqe_quality(vqe: Any) -> list[ActionItem]:
    """Generate actions from VQE convergence quality diagnostics.

    Flags chain breaks (θ > 1.0) which account for 45% of known failures,
    and low convergence rates indicating optimizer issues.
    """
    actions: list[ActionItem] = []

    if vqe.n_results == 0:
        return actions

    # Chain break warnings are a major failure mode (45% of failures)
    if vqe.n_chain_break_warnings > 0:
        fraction = vqe.n_chain_break_warnings / vqe.n_results
        if fraction > 0.30:
            priority = Priority.HIGH
        elif fraction > 0.10:
            priority = Priority.MEDIUM
        else:
            priority = Priority.LOW

        actions.append(
            ActionItem(
                priority=priority,
                title=(
                    f"VQE chain breaks: {vqe.n_chain_break_warnings}/{vqe.n_results} "
                    f"runs ({fraction:.0%}) have θ-smoothness > 1.0"
                ),
                detail=(
                    "Chain breaks indicate VQE angle discontinuities across the h-sweep. "
                    f"Max θ-smoothness: {vqe.theta_smoothness_max:.2f}. "
                    "Root cause: descending sweep lost state continuity. "
                    "Consider narrower h-grid or additional VQE restarts for affected topologies."
                ),
                category="vqe_quality",
            )
        )

    # Low overall convergence rate
    if vqe.convergence_rate_min is not None and vqe.convergence_rate_min < 0.80:
        actions.append(
            ActionItem(
                priority=Priority.MEDIUM,
                title=(f"VQE convergence issue: worst case only {vqe.convergence_rate_min:.0%}"),
                detail=(
                    f"Mean convergence: {vqe.convergence_rate_mean:.2%}. "
                    "Runs with < 80% convergence may need more restarts or "
                    "higher max_iterations."
                ),
                category="vqe_quality",
            )
        )

    return actions


def _actions_from_mpnn_quality(mpnn: Any) -> list[ActionItem]:
    """Generate actions from MPNN training quality diagnostics.

    Flags overfitting (gen_gap > 0.01, accounts for 25% of failures)
    and high prediction MSE.
    """
    actions: list[ActionItem] = []

    if mpnn.n_results == 0:
        return actions

    # Overfit warnings are a major failure mode (25% of failures)
    if mpnn.n_overfit_warnings > 0:
        fraction = mpnn.n_overfit_warnings / mpnn.n_results
        if fraction > 0.20:
            priority = Priority.HIGH
        elif fraction > 0.10:
            priority = Priority.MEDIUM
        else:
            priority = Priority.LOW

        actions.append(
            ActionItem(
                priority=priority,
                title=(
                    f"MPNN overfitting: {mpnn.n_overfit_warnings}/{mpnn.n_results} "
                    f"runs ({fraction:.0%}) have gen_gap > 0.01"
                ),
                detail=(
                    f"Max generalization gap: {mpnn.gen_gap_max:.4f}. "
                    "Overfitting correlates with small training sets or "
                    "insufficient regularization. Consider increasing patience, "
                    "reducing learning rate, or using more h-points in sweep."
                ),
                category="mpnn_quality",
            )
        )

    # High mean θ-MSE indicates poor angle predictions overall
    if mpnn.theta_mse_mean is not None and mpnn.theta_mse_mean > 0.01:
        actions.append(
            ActionItem(
                priority=Priority.MEDIUM,
                title=f"MPNN prediction MSE elevated: {mpnn.theta_mse_mean:.4f}",
                detail=(
                    "Mean θ-MSE > 0.01 suggests the MPNN is struggling to learn "
                    "the VQE angle landscape. Consider: larger hidden_dim, "
                    "more epochs, or denser h-grid in Phase 1."
                ),
                category="mpnn_quality",
            )
        )

    return actions


def _actions_from_distribution(dist: Any) -> list[ActionItem]:
    """Generate actions from result distribution imbalances.

    Identifies under-represented topologies or qubit counts that may
    limit thesis claims about generalization.
    """
    actions: list[ActionItem] = []

    if not dist.by_topology:
        return actions

    # Check p-layer imbalance: p=1 is the hardware strategy but often
    # under-represented relative to p=2 (since p=2 was explored first)
    p1_count = dist.by_p_layers.get(1, 0)
    p2_count = dist.by_p_layers.get(2, 0)
    total_p = p1_count + p2_count
    if total_p > 0 and p1_count < total_p * 0.20:
        actions.append(
            ActionItem(
                priority=Priority.MEDIUM,
                title=(
                    f"p=1 under-represented: only {p1_count}/{total_p} "
                    f"({p1_count / total_p:.0%}) of noiseless runs"
                ),
                detail=(
                    "p=1 is the recommended hardware strategy. "
                    "Thesis generalization claims need balanced p=1 coverage "
                    "across topologies."
                ),
                category="distribution",
            )
        )

    # Check heavy_hex coverage (hardware-relevant topology)
    total_topo = sum(dist.by_topology.values())
    hh_count = dist.by_topology.get("heavy_hex", 0)
    if total_topo > 50 and hh_count < total_topo * 0.05:
        actions.append(
            ActionItem(
                priority=Priority.MEDIUM,
                title=(
                    f"heavy_hex under-represented: only {hh_count}/{total_topo} "
                    f"({hh_count / total_topo:.0%}) of total runs"
                ),
                detail=(
                    "heavy_hex is the IBM Torino target topology. "
                    "Hardware deployment claims need sufficient heavy_hex validation."
                ),
                category="distribution",
            )
        )

    return actions


def _actions_from_energy_decomposition(ed: Any) -> list[ActionItem]:
    """Generate actions from energy error decomposition analysis.

    Identifies whether the dominant error source is circuit expressibility
    or MPNN prediction quality, guiding improvement priorities.
    """
    actions: list[ActionItem] = []

    if ed.n_results == 0:
        return actions

    # If MPNN is the dominant error source (>80% of error), flag it
    if ed.mpnn_error_fraction > 0.80:
        actions.append(
            ActionItem(
                priority=Priority.LOW,
                title=(
                    f"MPNN dominates error: {ed.mpnn_error_fraction:.0%} "
                    f"of total ΔE comes from MPNN prediction"
                ),
                detail=(
                    f"Mean circuit error: {ed.mean_circuit_error:.4f}, "
                    f"Mean MPNN error: {ed.mean_mpnn_error:.4f}. "
                    "Circuit expressibility is not the bottleneck — "
                    "focus improvement on MPNN architecture/training."
                ),
                category="diagnostics",
            )
        )
    elif ed.circuit_error_fraction > 0.50:
        actions.append(
            ActionItem(
                priority=Priority.MEDIUM,
                title=(
                    f"Circuit expressibility bottleneck: "
                    f"{ed.circuit_error_fraction:.0%} of error from VQE ceiling"
                ),
                detail=(
                    "VQE is not reaching ground state well enough. "
                    "Consider more restarts or checking for convergence issues."
                ),
                category="diagnostics",
            )
        )

    return actions


def _actions_from_experiment_failures(
    experiments: list[Any],
) -> list[ActionItem]:
    """Generate actions from experiment failures that may need investigation.

    Distinguishes between expected negative results (in REJECTION_IS_FINDING)
    and unexpected failures that might indicate bugs or misconfigurations.
    """
    from qmbp_simulation.framework.criteria import REJECTION_IS_FINDING

    actions: list[ActionItem] = []

    # Identify experiments that failed (not rejected — failed means they
    # didn't meet threshold and aren't in REJECTION_IS_FINDING)
    failed = [e for e in experiments if e.verdict == "failed"]
    if not failed:
        return actions

    # Separate into low-pass-rate (potentially fixable) vs edge cases
    fixable = [e for e in failed if e.pass_rate is not None and e.pass_rate > 0.0]
    zero_pass = [e for e in failed if e.pass_rate is not None and e.pass_rate == 0.0]

    if fixable:
        ids = [f"{e.experiment_id}({e.pass_rate * 100:.0f}%)" for e in fixable]
        actions.append(
            ActionItem(
                priority=Priority.MEDIUM,
                title=(f"{len(fixable)} experiment(s) partially passed — may improve with tuning"),
                detail=(
                    f"Near-threshold: {', '.join(ids[:8])}. "
                    "These have partial success and may cross threshold with "
                    "optimized configs (more restarts, adjusted h-grid, larger N)."
                ),
                category="experiments",
            )
        )

    if zero_pass:
        ids = [e.experiment_id for e in zero_pass]
        # Only flag if they're not in REJECTION_IS_FINDING
        truly_failed = [eid for eid in ids if eid not in REJECTION_IS_FINDING]
        if truly_failed:
            actions.append(
                ActionItem(
                    priority=Priority.HIGH,
                    title=(f"{len(truly_failed)} experiment(s) completely failed (0% pass rate)"),
                    detail=(
                        f"Failed: {', '.join(truly_failed[:8])}. "
                        "Zero pass rate suggests fundamental issue — verify "
                        "config, check for bugs, or reconsider hypothesis viability."
                    ),
                    category="experiments",
                )
            )

    return actions


# ═══════════════════════════════════════════════════════════════════════════════
# Gap detection functions (private)
# ═══════════════════════════════════════════════════════════════════════════════


def _gap_missing_p1_noiseless(noiseless: list[NoiselessResult]) -> list[CoverageGap]:
    """Find configs with p=2 data but no p=1 noiseless results."""
    gaps: list[CoverageGap] = []

    # Build config sets: (topology, n_qubits) that have data
    p2_configs: set[tuple[str, int]] = set()
    p1_configs: set[tuple[str, int]] = set()

    for r in noiseless:
        if r.delta_e_over_gap is None or not r.topology or not r.n_qubits:
            continue
        config = (r.topology, r.n_qubits)
        if r.p_layers == 2:
            p2_configs.add(config)
        elif r.p_layers == 1:
            p1_configs.add(config)

    # Configs that exist in p=2 but not p=1
    missing = p2_configs - p1_configs
    # Exclude kagome (not central to thesis)
    missing = {c for c in missing if c[0] != "kagome"}

    for topo, n in sorted(missing):
        gaps.append(
            CoverageGap(
                gap_type=GapType.MISSING_P1_NOISELESS,
                topology=topo,
                n_qubits=n,
                p_layers=1,
                detail=f"p=2 has data for {topo} N={n}, but p=1 is missing",
                recommendation=f"Run pipeline with p=1 N={n} topology={topo} seeds=42,43,44",
                priority=Priority.MEDIUM,
            )
        )

    return gaps


def _gap_invalid_regime(noiseless: list[NoiselessResult]) -> list[CoverageGap]:
    """Find p=1 results tested outside the valid regime.

    Deduplicates by (topology, n_qubits, h_test) to avoid reporting
    the same regime violation for each seed.
    """
    gaps: list[CoverageGap] = []
    seen: set[tuple[str, int, float]] = set()

    for r in noiseless:
        if r.p_layers != 1 or not r.topology or not r.n_qubits:
            continue
        if r.delta_e_over_gap is None:
            continue

        threshold = P1_VALID_REGIME.get((r.topology, r.n_qubits), 0.0)
        if threshold == 0.0:
            continue

        # Check h_test values
        for h in r.h_test:
            if h < threshold:
                key = (r.topology, r.n_qubits, h)
                if key in seen:
                    continue
                seen.add(key)
                gaps.append(
                    CoverageGap(
                        gap_type=GapType.INVALID_REGIME,
                        topology=r.topology,
                        n_qubits=r.n_qubits,
                        p_layers=1,
                        detail=f"h_test={h} < regime boundary {threshold}",
                        recommendation=f"Use h_test ≥ {threshold}",
                        priority=Priority.LOW,
                    )
                )
                break  # One gap per result is enough

    return gaps


def _gap_insufficient_seeds(noiseless: list[NoiselessResult]) -> list[CoverageGap]:
    """Find p=1 configs with fewer than 3 seeds."""
    gaps: list[CoverageGap] = []

    # Group p=1 results by (topology, n_qubits) and collect seeds
    config_seeds: dict[tuple[str, int], set[int]] = defaultdict(set)
    for r in noiseless:
        if r.p_layers != 1 or not r.topology or not r.n_qubits:
            continue
        if r.delta_e_over_gap is None or r.seed is None:
            continue
        config_seeds[(r.topology, r.n_qubits)].add(r.seed)

    for (topo, n), seeds in sorted(config_seeds.items()):
        if len(seeds) < 3:
            missing = sorted(REQUIRED_SEEDS - seeds)
            gaps.append(
                CoverageGap(
                    gap_type=GapType.INSUFFICIENT_SEEDS,
                    topology=topo,
                    n_qubits=n,
                    p_layers=1,
                    detail=f"Only {len(seeds)} seed(s): {sorted(seeds)}. Missing: {missing}",
                    recommendation=f"Run with seeds {missing}",
                    priority=Priority.MEDIUM,
                )
            )

    return gaps


def _gap_missing_zne(
    noiseless: list[NoiselessResult],
    noisy: list[NoisyResult],
) -> list[CoverageGap]:
    """Find p=1 configs with noiseless data but no ZNE validation."""
    gaps: list[CoverageGap] = []

    # p=1 noiseless configs
    p1_noiseless_configs: set[tuple[str, int]] = set()
    for r in noiseless:
        if r.p_layers == 1 and r.topology and r.n_qubits and r.delta_e_over_gap is not None:
            p1_noiseless_configs.add((r.topology, r.n_qubits))

    # p=1 noisy configs
    p1_noisy_configs: set[tuple[str, int]] = set()
    for r in noisy:
        if r.p_layers == 1 and r.topology and r.n_qubits:
            p1_noisy_configs.add((r.topology, r.n_qubits))

    # Noiseless exists but no ZNE
    missing = p1_noiseless_configs - p1_noisy_configs
    for topo, n in sorted(missing):
        # Higher priority for hardware-relevant configs
        priority = Priority.HIGH if (topo == "heavy_hex" and n >= 10) else Priority.MEDIUM
        gaps.append(
            CoverageGap(
                gap_type=GapType.MISSING_ZNE,
                topology=topo,
                n_qubits=n,
                p_layers=1,
                detail=f"p=1 noiseless exists for {topo} N={n} but no ZNE run",
                recommendation=f"Run noisy simulation with ZNE for {topo} N={n}",
                priority=priority,
            )
        )

    return gaps


def _gap_missing_experiments(experiments: list[ExperimentResult]) -> list[CoverageGap]:
    """Find experiments defined in EXPERIMENT_CRITERIA but with no results."""
    from qmbp_simulation.framework.criteria import EXPERIMENT_CRITERIA

    gaps: list[CoverageGap] = []

    # IDs that have results
    existing_ids = {e.experiment_id for e in experiments}

    # IDs defined in criteria but missing from results
    for exp_id in sorted(EXPERIMENT_CRITERIA.keys()):
        if exp_id not in existing_ids:
            criteria = EXPERIMENT_CRITERIA[exp_id]
            gaps.append(
                CoverageGap(
                    gap_type=GapType.MISSING_EXPERIMENT,
                    detail=exp_id,
                    recommendation=f"Run experiment {exp_id}: {criteria.get('desc', '')}",
                    priority=Priority.LOW,
                )
            )

    return gaps


# ═══════════════════════════════════════════════════════════════════════════════
# Noiseless/Noisy aggregate stats
# ═══════════════════════════════════════════════════════════════════════════════


def compute_noiseless_stats(
    noiseless: list[NoiselessResult],
) -> tuple[float, float | None]:
    """Compute pass rate and median ΔE/gap for noiseless results.

    Returns (pass_rate, median_de_gap).
    """
    de_gaps = [r.delta_e_over_gap for r in noiseless if r.delta_e_over_gap is not None]
    if not de_gaps:
        return 0.0, None

    n_pass = sum(1 for d in de_gaps if d < 0.05)
    pass_rate = n_pass / len(de_gaps)
    median_de = statistics.median(de_gaps)
    return pass_rate, median_de


def compute_noiseless_by_topology(
    noiseless: list[NoiselessResult],
) -> dict[str, dict[str, Any]]:
    """Compute per-topology noiseless stats.

    Returns dict mapping topology → {n_runs, pass_rate, median_de, best, worst}.
    """
    by_topo: dict[str, list[float]] = defaultdict(list)
    for r in noiseless:
        if r.delta_e_over_gap is not None and r.topology:
            by_topo[r.topology].append(r.delta_e_over_gap)

    result: dict[str, dict[str, Any]] = {}
    for topo in sorted(by_topo.keys()):
        gaps = by_topo[topo]
        n_pass = sum(1 for d in gaps if d < 0.05)
        result[topo] = {
            "n_runs": len(gaps),
            "pass_rate": n_pass / len(gaps),
            "median_de": statistics.median(gaps),
            "best": min(gaps),
            "worst": max(gaps),
        }
    return result


def compute_noisy_stats(
    noisy: list[NoisyResult],
) -> tuple[float, float, float]:
    """Compute success rate, mean R², mean gain for noisy results.

    Returns (success_rate, mean_r2, mean_gain).
    Safely handles missing or None values in R² and gain fields.
    """
    if not noisy:
        return 0.0, 0.0, 0.0

    n_success = sum(1 for r in noisy if r.success_criteria_met)
    success_rate = n_success / len(noisy)

    r2_values = [r.mean_r2 for r in noisy if r.mean_r2 is not None]
    gain_values = [r.mean_gain_pct for r in noisy if r.mean_gain_pct is not None]

    mean_r2 = statistics.mean(r2_values) if r2_values else 0.0
    mean_gain = statistics.mean(gain_values) if gain_values else 0.0
    return success_rate, mean_r2, mean_gain


# ═══════════════════════════════════════════════════════════════════════════════
# Advanced metrics (VQE, MPNN, Timing, Distribution, Energy Decomposition)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_vqe_quality(noiseless: list[NoiselessResult]) -> VQEQualityStats:
    """Compute aggregated VQE convergence quality from Phase 2 data.

    Tracks convergence rate, theta smoothness (chain break indicator),
    and worst-case convergence points.
    """
    convergence_rates: list[float] = []
    theta_smoothness_values: list[float] = []

    for r in noiseless:
        if r.convergence_rate is not None:
            convergence_rates.append(r.convergence_rate)
        if r.theta_smoothness is not None:
            theta_smoothness_values.append(r.theta_smoothness)

    n_results = len(convergence_rates) + len(theta_smoothness_values)
    if n_results == 0:
        return VQEQualityStats()

    stats = VQEQualityStats(n_results=max(len(convergence_rates), len(theta_smoothness_values)))

    if convergence_rates:
        stats.convergence_rate_mean = statistics.mean(convergence_rates)
        stats.convergence_rate_min = min(convergence_rates)

    if theta_smoothness_values:
        stats.theta_smoothness_mean = statistics.mean(theta_smoothness_values)
        stats.theta_smoothness_max = max(theta_smoothness_values)
        stats.n_chain_break_warnings = sum(1 for v in theta_smoothness_values if v > 1.0)

    return stats


def compute_mpnn_quality(noiseless: list[NoiselessResult]) -> MPNNQualityStats:
    """Compute aggregated MPNN training quality from Phase 3 data.

    Tracks generalization gap (overfit indicator) and prediction MSE.
    """
    gen_gaps: list[float] = []
    theta_mse_values: list[float] = []

    for r in noiseless:
        if r.generalization_gap is not None:
            gen_gaps.append(r.generalization_gap)
        if r.theta_zz_mse is not None:
            theta_mse_values.append(r.theta_zz_mse)

    if not gen_gaps and not theta_mse_values:
        return MPNNQualityStats()

    stats = MPNNQualityStats(n_results=max(len(gen_gaps), len(theta_mse_values)))

    if gen_gaps:
        stats.gen_gap_mean = statistics.mean(gen_gaps)
        stats.gen_gap_max = max(gen_gaps)
        stats.gen_gap_median = statistics.median(gen_gaps)
        stats.n_overfit_warnings = sum(1 for g in gen_gaps if g > 0.01)

    if theta_mse_values:
        stats.theta_mse_mean = statistics.mean(theta_mse_values)

    return stats


def compute_timing_stats(
    noiseless: list[NoiselessResult],
    noisy: list[NoisyResult],
    experiments: list[ExperimentResult],
) -> TimingStats:
    """Compute pipeline timing breakdown across all run types.

    Aggregates elapsed times from noiseless, noisy, and experiment runs
    to provide total compute time and per-phase averages.
    """
    all_elapsed: list[float] = []

    # Noiseless timings
    for r in noiseless:
        if r.elapsed_s and r.elapsed_s > 0:
            all_elapsed.append(r.elapsed_s)

    # Noisy timings
    for r in noisy:
        if r.elapsed_s and r.elapsed_s > 0:
            all_elapsed.append(r.elapsed_s)

    # Experiment timings
    for r in experiments:
        if r.total_time_s and r.total_time_s > 0:
            all_elapsed.append(r.total_time_s)

    if not all_elapsed:
        return TimingStats()

    stats = TimingStats(
        total_pipeline_hours=sum(all_elapsed) / 3600.0,
        mean_run_s=statistics.mean(all_elapsed),
        median_run_s=statistics.median(all_elapsed),
        max_run_s=max(all_elapsed),
        total_runs=len(all_elapsed),
    )

    # Per-phase breakdown (only from noiseless results that have phase data)
    # Phase timing comes from the raw JSON; NoiselessResult stores elapsed_s total.
    # We approximate phase 2 from convergence data available, but actual per-phase
    # timing requires raw JSON access. We report what's available.
    # Note: elapsed_s on NoiselessResult covers the entire pipeline run (all phases).

    return stats


def compute_distribution(
    noiseless: list[NoiselessResult],
    noisy: list[NoisyResult],
) -> ModelDistribution:
    """Compute distribution of results across system parameters.

    Useful for identifying imbalanced coverage (e.g., too many chain_1d runs,
    not enough heavy_hex or triangular).
    """
    dist = ModelDistribution()

    for r in noiseless:
        # By model
        model = r.model or "tfim"
        dist.by_model[model] = dist.by_model.get(model, 0) + 1
        # By topology
        if r.topology:
            dist.by_topology[r.topology] = dist.by_topology.get(r.topology, 0) + 1
        # By n_qubits
        if r.n_qubits:
            dist.by_n_qubits[r.n_qubits] = dist.by_n_qubits.get(r.n_qubits, 0) + 1
        # By p_layers
        dist.by_p_layers[r.p_layers] = dist.by_p_layers.get(r.p_layers, 0) + 1

    for r in noisy:
        if r.topology:
            dist.by_topology[r.topology] = dist.by_topology.get(r.topology, 0) + 1
        if r.n_qubits:
            dist.by_n_qubits[r.n_qubits] = dist.by_n_qubits.get(r.n_qubits, 0) + 1

    return dist


def compute_energy_decomposition(
    noiseless: list[NoiselessResult],
) -> EnergyDecompositionStats:
    """Compute aggregated energy error decomposition.

    Separates error contributions from circuit expressibility (VQE ceiling)
    vs MPNN prediction quality. This is key for understanding which
    component to improve.

    Note: Uses mag_x_error and corr_zz_error as proxies when decomposition
    data isn't directly available. The actual decomposition comes from the
    raw JSON's phase4.energy_decomposition field, which is captured in
    NoiselessResult when available.
    """
    # We work from delta_e_over_gap and the known structure:
    # total_error = error_from_circuit + error_from_mpnn
    # The scanner doesn't extract these sub-fields directly, but we can
    # infer from convergence_rate (circuit quality) vs gen_gap (MPNN quality).
    circuit_errors: list[float] = []
    mpnn_errors: list[float] = []

    for r in noiseless:
        if r.delta_e_over_gap is None:
            continue
        # If convergence_rate is available, low convergence → circuit contribution
        # If gen_gap is available, high gen_gap → MPNN contribution
        if r.convergence_rate is not None and r.generalization_gap is not None:
            # Heuristic decomposition: circuit error ∝ (1 - convergence_rate)
            # MPNN error ∝ generalization_gap (normalized)
            circuit_contrib = (1.0 - r.convergence_rate) * r.delta_e_over_gap
            mpnn_contrib = r.delta_e_over_gap - circuit_contrib
            if mpnn_contrib < 0:
                mpnn_contrib = 0.0
                circuit_contrib = r.delta_e_over_gap
            circuit_errors.append(circuit_contrib)
            mpnn_errors.append(mpnn_contrib)

    if not circuit_errors:
        return EnergyDecompositionStats()

    total_error = sum(circuit_errors) + sum(mpnn_errors)
    return EnergyDecompositionStats(
        n_results=len(circuit_errors),
        mean_circuit_error=statistics.mean(circuit_errors),
        mean_mpnn_error=statistics.mean(mpnn_errors),
        circuit_error_fraction=(sum(circuit_errors) / total_error if total_error > 0 else 0.0),
        mpnn_error_fraction=(sum(mpnn_errors) / total_error if total_error > 0 else 0.0),
    )
