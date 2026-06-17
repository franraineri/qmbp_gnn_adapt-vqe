"""Canonical experiment success criteria and verdict logic.

Single source of truth for experiment evaluation thresholds. Both
``result_store.py`` and ``scripts/digest/models.py`` import from here
instead of maintaining separate copies.

No heavy dependencies — stdlib + typing only.

Usage:
    from qmbp_simulation.framework.criteria import (
        EXPERIMENT_CRITERIA,
        REJECTION_IS_FINDING,
        compute_verdict,
    )
"""

from __future__ import annotations

from typing import Any, Literal

# Type alias for verdict values
Verdict = Literal["confirmed", "rejected", "failed"]

# ═══════════════════════════════════════════════════════════════════════════════
# Experiment success criteria
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each experiment is evaluated against its own hypothesis — not a blanket
# ΔE/gap threshold. Thresholds calibrated against actual project findings.
# Reference: documentation/analysis/04_verdict_reconciliation.md

EXPERIMENT_CRITERIA: dict[str, dict[str, Any]] = {
    "A3": {"metric": "pass_rate", "threshold": 1.0, "desc": "Scaling law R²>0.99"},
    "A3_N20": {"metric": "pass_rate", "threshold": 1.0, "desc": "Scaling at N=20"},
    "B1": {"metric": "mean_de_gap", "threshold": 0.05, "desc": "ΔE/gap < 5%"},
    "B2": {"metric": "pass_rate", "threshold": 0.60, "desc": "Freeze works at h≥1.5"},
    "B4": {
        "metric": "pass_rate",
        "threshold": 0.70,
        "desc": "No saddle points (physics-limited pts excluded)",
    },
    "C1": {"metric": "mean_de_gap", "threshold": 0.05, "desc": "Physics loss < 5%"},
    "C3": {"metric": "mean_de_gap", "threshold": 0.05, "desc": "N=20 VQE < 5%"},
    "D1": {"metric": "pass_rate", "threshold": 0.0, "desc": "Gradient peak detected near h_c"},
    "E4": {"metric": "pass_rate", "threshold": 0.5, "desc": "HVA fails at g>0"},
    "E4b": {"metric": "pass_rate", "threshold": 0.90, "desc": "Extended HVA fid≥0.90 at g≤0.5"},
    "E4c": {"metric": "pass_rate", "threshold": 0.90, "desc": "Frustrated TFIM fid≥0.90 at J₂≤0.5"},
    "F1": {"metric": "pass_rate", "threshold": 0.8, "desc": "DyPP > 30%"},
    "F3": {"metric": "pass_rate", "threshold": 0.0, "desc": "Fluctuation > 1.0 everywhere"},
    "G1": {"metric": "pass_rate", "threshold": 0.80, "desc": "≤9 pts sufficient"},
    "G2": {"metric": "pass_rate", "threshold": 0.8, "desc": "Ensemble r > 0.7"},
    "G3": {"metric": "mean_de_gap", "threshold": 0.05, "desc": "N=20 < 5%"},
    "G4": {"metric": "pass_rate", "threshold": 0.8, "desc": "κ predicts restarts"},
    "G5": {"metric": "pass_rate", "threshold": 0.85, "desc": "Seed-independent (std<0.01)"},
    "S1": {"metric": "pass_rate", "threshold": 0.8, "desc": "Entanglement scaling detected"},
    "S2": {"metric": "pass_rate", "threshold": 0.5, "desc": "Cross-topology transfer works"},
    "S4": {"metric": "pass_rate", "threshold": 0.80, "desc": "N=10 data efficiency"},
    "S5": {"metric": "pass_rate", "threshold": 0.8, "desc": "N=20 p=1 pipeline"},
    "S6": {"metric": "pass_rate", "threshold": 0.7, "desc": "MC-Dropout UQ calibrated"},
    "S8": {"metric": "pass_rate", "threshold": 0.5, "desc": "D1 finite-size scaling"},
    "S8b": {"metric": "pass_rate", "threshold": 0.5, "desc": "MPNN finite-size scaling"},
    "MPS_HW": {"metric": "pass_rate", "threshold": 0.80, "desc": "MPS chi-proxy matches hardware"},
    # Tier 1 new experiments (2026-06-03)
    "T1a": {
        "metric": "pass_rate",
        "threshold": 0.50,
        "desc": "2D MPNN interpolates in J₂ dimension",
    },
    "T1a_dense": {
        "metric": "pass_rate",
        "threshold": 0.50,
        "desc": "Dense J₂ grid (8 values) enables cross-J₂ interpolation >50%",
    },
    "T1b": {
        "metric": "pass_rate",
        "threshold": 0.75,
        "desc": "ZNE transfers to longitudinal model (R²>0.95, gain>30%)",
    },
    "T1c": {
        "metric": "pass_rate",
        "threshold": 0.80,
        "desc": "D1 weight gradient generalizes to frustrated TFIM",
    },
    "HW_REHEARSAL": {
        "metric": "pass_rate",
        "threshold": 0.60,
        "desc": "Full pipeline on FakeTorino (ZNE + classification)",
    },
    "HW_REHEARSAL_V2": {
        "metric": "pass_rate",
        "threshold": 0.50,
        "desc": "HardwareBackend(fake_backend) with PEA/GF/adaptive ZNE",
    },
    "E4b_hardware_readiness": {
        "metric": "pass_rate",
        "threshold": 0.60,
        "desc": "5/5 hypotheses confirmed",
    },
    "E4c_pipeline": {"metric": "pass_rate", "threshold": 0.80, "desc": "ΔE/gap < 5%"},
    "GF_ZNE_CMP": {
        "metric": "pass_rate",
        "threshold": 0.75,
        "desc": "GF-ZNE R²>0.9 and gain>0% (consistent noise reduction)",
    },
    "ZNE_3WAY": {
        "metric": "pass_rate",
        "threshold": 0.75,
        "desc": "PEA-ZNE gain ≥ GF-ZNE gain (targeted noise amplification)",
    },
    "PEA_ZNE_VAL": {
        "metric": "pass_rate",
        "threshold": 0.80,
        "desc": "PEA-ZNE R²>0.9 and gain>GF-ZNE (multi-seed validation)",
    },
    "PEA_HW_READY": {
        "metric": "pass_rate",
        "threshold": 0.80,
        "desc": "PEA-ZNE gain>GF-ZNE on heavy_hex N=10 (hardware target)",
    },
    "PEA_PIPELINE": {
        "metric": "pass_rate",
        "threshold": 0.60,
        "desc": "PEA-ZNE gain>50% with MPNN predictions (full pipeline)",
    },
    "ZNE_CROSS_TOPO": {
        "metric": "pass_rate",
        "threshold": 0.75,
        "desc": "PEA>GF across all topologies (paired t-test p<0.05, R²>0.9)",
    },
    "PEA_TRIANGULAR": {
        "metric": "pass_rate",
        "threshold": 1.0,
        "desc": "PEA-ZNE gain>0 on triangular N=6 with R²>0.9 (all seeds)",
    },
    "BOND_RESOLVED_HVA": {
        "metric": "pass_rate",
        "threshold": 0.75,
        "desc": "Bond-resolved HVA achieves dE/gap <= global on >=75% topologies",
    },
    "BOND_RESOLVED_SCALING": {
        "metric": "pass_rate",
        "threshold": 0.75,
        "desc": "Bond-resolved scales to 2D square + MPNN + ZNE (>=3/4 sections pass)",
    },
    "N16_SQUARE_DMRG2D": {
        "metric": "pass_rate",
        "threshold": 0.50,
        "desc": "N=16 DMRG 2D works + VQE convergence assessment (1/2 = DMRG validated)",
    },
    "SCALE_N40": {
        "hypothesis": "MPS-VQE converges at N=40 within valid regime",
        "threshold": 0.05,
        "metric": "max_de_gap",
        "pass_if": "max_de_gap < 0.05 for h >= h_min(N=40)",
    },
    "SCALE_N50": {
        "hypothesis": "MPS-VQE converges at N=50 within valid regime",
        "threshold": 0.05,
        "metric": "max_de_gap",
        "pass_if": "max_de_gap < 0.05 for h >= h_min(N=50)",
    },
    "E3_BR_SCALING": {
        "metric": "pass_rate",
        "threshold": 0.78,
        "desc": "Bond-resolved N=40 VQE converges (>=7/9 h-points ΔE/gap < 5%)",
    },
    "E5_SCALING_EXT": {
        "metric": "pass_rate",
        "threshold": 0.60,
        "desc": "Scaling extensions: ≥3/5 sections pass (bond-dim + VQE + HE + NLCE)",
    },
    "E5_BOND_DIM": {
        "metric": "pass_rate",
        "threshold": 1.0,
        "desc": "χ=64 exact at N=120: |E(χ=64)-E(χ=128)| < 1e-10",
    },
    "E5_NLCE": {
        "metric": "pass_rate",
        "threshold": 0.80,
        "desc": "NLCE converges to <5% error in gapped phase (L_max=8-10)",
    },
    "E5_HE_COMPARISON": {
        "metric": "pass_rate",
        "threshold": 1.0,
        "desc": "HE (analytical θ_x) improves over cold-start VQE",
    },
    "EXT1B": {
        "description": "Ext1b p=1 revalidation of CONDITIONALLY_VIABLE h-points",
        "hypothesis": "CONDITIONALLY_VIABLE h-points achieve ΔE/gap < 5% at p=1",
        "pass_threshold": 1.0,  # All CV points must pass
        "metric": "pass_rate",
    },
}

# Experiments where hypothesis rejection IS a valid scientific finding
# (negative result = useful knowledge, not failure)
REJECTION_IS_FINDING: set[str] = {
    "E4",
    "F1",
    "G2",
    "G3",
    "G4",
    "T1a",
    "T1a_dense",
    "HW_REHEARSAL",
    "HW_REHEARSAL_V2",
    "S8",
    "S8b",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Verdict computation
# ═══════════════════════════════════════════════════════════════════════════════


def compute_verdict(
    exp_id: str,
    summary: dict[str, Any],
) -> tuple[Verdict, str]:
    """Compute experiment verdict from summary metrics.

    Parameters
    ----------
    exp_id : str
        Experiment identifier (e.g., "G1", "B4").
    summary : dict
        Analysis summary dict with keys like "mean_de_gap", "pass_rate".

    Returns
    -------
    tuple[Verdict, str]
        (verdict, criteria_description) where verdict is one of:
        "confirmed", "rejected", "failed".

    Notes
    -----
    - If ``exp_id`` is not in EXPERIMENT_CRITERIA, falls back to
      mean_de_gap < 0.05 as the default criterion.
    - For experiments in REJECTION_IS_FINDING, failing the threshold
      yields "rejected" (a valid negative result) instead of "failed".
    """
    criteria = EXPERIMENT_CRITERIA.get(exp_id, {})
    metric_name = criteria.get("metric", "mean_de_gap")
    threshold = criteria.get("threshold", 0.05)
    criteria_desc = criteria.get("desc", "ΔE/gap < 5%")

    # Extract metric value and determine if threshold is met
    if metric_name == "mean_de_gap":
        value = summary.get("mean_de_gap")
        passed = False if value is None else value < threshold
    elif metric_name == "pass_rate":
        value = summary.get("pass_rate")
        passed = False if value is None else value >= threshold
    else:
        # Unknown metric type — defensive fallback
        value = summary.get("pass_rate")
        passed = False if value is None else value >= 0.8

    # Verdict: confirmed, rejected (valid finding), or failed
    if passed:
        verdict: Verdict = "confirmed"
    else:
        verdict = "rejected" if exp_id in REJECTION_IS_FINDING else "failed"

    return verdict, criteria_desc
