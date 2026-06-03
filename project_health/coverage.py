"""Coverage gap detection logic.

Analyzes scan results to identify what's missing for complete thesis
coverage and hardware deployment readiness.

Uses data already parsed by ResultScanner — no re-parsing of JSON files.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from project_health.models import ActionItem, CoverageGap, GapType, Priority
from scripts.digest.models import ExperimentResult, NoiselessResult, NoisyResult

# ═══════════════════════════════════════════════════════════════════════════════
# Valid regime boundaries (canonical source: analysis/scan_coverage.py)
# Imported here as constants to avoid coupling to that script's global state.
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


def derive_actions(gaps: list[CoverageGap]) -> list[ActionItem]:
    """Convert coverage gaps into prioritized actionable items.

    Groups related gaps and produces concrete next-step recommendations.
    """
    actions: list[ActionItem] = []

    # Group gaps by type for smarter action generation
    by_type: dict[GapType, list[CoverageGap]] = defaultdict(list)
    for g in gaps:
        by_type[g.gap_type].append(g)

    # Hardware-blocking gaps → CRITICAL actions
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

    # Missing experiments (expected in EXPERIMENT_CRITERIA but no results)
    missing_exp = by_type.get(GapType.MISSING_EXPERIMENT, [])
    if missing_exp:
        ids = [g.detail for g in missing_exp]
        actions.append(
            ActionItem(
                priority=Priority.HIGH,
                title=f"{len(missing_exp)} defined experiments have no results",
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
                detail="These are expected to fail — not actionable unless they passed unexpectedly.",
                category="coverage",
            )
        )

    # Non-critical ZNE gaps (not heavy_hex or N<10)
    zne_other = [g for g in hw_gaps if g not in hw_critical]
    if zne_other:
        actions.append(
            ActionItem(
                priority=Priority.MEDIUM,
                title=f"ZNE validation missing for {len(zne_other)} config(s)",
                detail="p=1 noiseless exists but no noisy/ZNE run.",
                category="coverage",
            )
        )

    actions.sort(key=lambda a: a.priority.value)
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
    """
    if not noisy:
        return 0.0, 0.0, 0.0

    n_success = sum(1 for r in noisy if r.success_criteria_met)
    success_rate = n_success / len(noisy)
    mean_r2 = statistics.mean(r.mean_r2 for r in noisy)
    mean_gain = statistics.mean(r.mean_gain_pct for r in noisy)
    return success_rate, mean_r2, mean_gain
