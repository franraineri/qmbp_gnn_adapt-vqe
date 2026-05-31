"""Comparative analysis and regime discovery for multi-model evaluation.

Provides dataclasses and helper functions for:
- Regime discovery (finding valid h-ranges for different models)
- Comparative metrics (TFIM vs Heisenberg side-by-side)
- Result classification (success/partial/failure/negative)
- Staggered magnetization computation

Usage:
    from qmbp_simulation.analysis.comparative import (
        RegimeDiscoveryResult, ComparativeMetrics,
        find_h_min, classify_result, compute_staggered_magnetization,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ── Regime Discovery ─────────────────────────────────────────────────────


@dataclass
class RegimeDiscoveryResult:
    """Result of regime discovery for a specific model and anisotropy.

    Attributes
    ----------
    model_type : str
        Model identifier ("heisenberg", "xy", etc.).
    delta : float
        Anisotropy parameter.
    n_qubits : int
        System size.
    h_values : list[float]
        Field values evaluated.
    fidelities : list[float]
        Best VQE fidelity at each h-point.
    entropies : list[float]
        Entanglement entropy at each h-point.
    h_min : float | None
        Minimum h where fidelity ≥ threshold (None if no valid regime).
    max_fidelity : float
        Maximum fidelity achieved across all h-points.
    valid_regime_width : int
        Number of h-points with fidelity ≥ threshold.
    threshold_used : float
        Fidelity threshold applied.
    is_negative_result : bool
        True if no valid regime exists (max_fidelity < 0.60).
    """

    model_type: str
    delta: float
    n_qubits: int
    h_values: list[float] = field(default_factory=list)
    fidelities: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    h_min: float | None = None
    max_fidelity: float = 0.0
    valid_regime_width: int = 0
    threshold_used: float = 0.93
    is_negative_result: bool = False


def find_h_min(
    fidelities_by_seed: dict[int, list[float]],
    h_values: list[float],
    threshold: float = 0.93,
    min_seeds: int = 2,
) -> float | None:
    """Find minimum h where at least min_seeds achieve fidelity ≥ threshold.

    Parameters
    ----------
    fidelities_by_seed : dict[int, list[float]]
        Mapping seed → list of fidelities (one per h-value).
    h_values : list[float]
        Field values (same order as fidelity lists).
    threshold : float
        Fidelity threshold.
    min_seeds : int
        Minimum number of seeds that must pass at a given h.

    Returns
    -------
    float | None
        Minimum h where criterion is met, or None if no h qualifies.
    """
    seeds = list(fidelities_by_seed.keys())
    n_h = len(h_values)

    for i, h in enumerate(h_values):
        passing_seeds = sum(
            1
            for s in seeds
            if i < len(fidelities_by_seed[s]) and fidelities_by_seed[s][i] >= threshold
        )
        if passing_seeds >= min_seeds:
            return float(h)

    return None


def classify_result(max_fidelity: float) -> str:
    """Classify the regime discovery result based on maximum fidelity.

    Parameters
    ----------
    max_fidelity : float
        Maximum fidelity achieved across all h-points and seeds.

    Returns
    -------
    str
        Classification: "fundamental_expressibility_limitation" if max < 0.60,
        "partial_expressibility" if 0.60 ≤ max < 0.93,
        "viable_regime" if max ≥ 0.93.
    """
    if max_fidelity < 0.60:
        return "fundamental_expressibility_limitation"
    elif max_fidelity < 0.93:
        return "partial_expressibility"
    else:
        return "viable_regime"


def filter_by_threshold(
    fidelities: list[float],
    threshold: float,
) -> list[int]:
    """Return indices of results with fidelity ≥ threshold, preserving order.

    Parameters
    ----------
    fidelities : list[float]
        Fidelity values.
    threshold : float
        Minimum fidelity threshold.

    Returns
    -------
    list[int]
        Indices of qualifying points.
    """
    return [i for i, f in enumerate(fidelities) if f >= threshold]


def find_minimum_viable_threshold(
    fidelities: list[float],
    thresholds: list[float] | None = None,
    min_points: int = 5,
) -> float | None:
    """Find highest threshold yielding at least min_points qualifying points.

    Parameters
    ----------
    fidelities : list[float]
        Fidelity values from VQE sweep.
    thresholds : list[float] | None
        Descending list of thresholds to try. Default: [0.93, 0.80, 0.70, 0.60].
    min_points : int
        Minimum number of points needed for viable MPNN training.

    Returns
    -------
    float | None
        Highest viable threshold, or None if no threshold yields enough points.
    """
    if thresholds is None:
        thresholds = [0.93, 0.80, 0.70, 0.60]

    for threshold in sorted(thresholds, reverse=True):
        n_qualifying = sum(1 for f in fidelities if f >= threshold)
        if n_qualifying >= min_points:
            return threshold

    return None


# ── Comparative Analysis ─────────────────────────────────────────────────


@dataclass
class ComparativeMetrics:
    """Metrics for comparing pipeline performance across models.

    Attributes
    ----------
    model_type : str
        Model identifier.
    delta_e_over_gap : float | None
        Average ΔE/gap (None if Phase 4 not executed).
    avg_fidelity : float
        Average VQE fidelity in valid regime.
    mpnn_mse : float | None
        MPNN training MSE (None if Phase 3 not executed).
    valid_regime_width : int
        Number of h-points in valid regime.
    vqe_time_s : float
        Total VQE computation time.
    cx_budget_per_layer : int
        Number of 2-qubit gates per HVA layer.
    max_entanglement : float
        Maximum entanglement entropy in the sweep.
    hva_capacity : float | None
        Maximum entropy representable by HVA (None if no valid point).
    """

    model_type: str
    delta_e_over_gap: float | None = None
    avg_fidelity: float = 0.0
    mpnn_mse: float | None = None
    valid_regime_width: int = 0
    vqe_time_s: float = 0.0
    cx_budget_per_layer: int = 0
    max_entanglement: float = 0.0
    hva_capacity: float | None = None


def compute_cx_budget(n_edges: int, model_type: str) -> int:
    """Compute the number of 2-qubit gates per HVA layer.

    Parameters
    ----------
    n_edges : int
        Number of lattice edges.
    model_type : str
        Model type ("tfim" or "heisenberg"/"xy").

    Returns
    -------
    int
        Number of 2-qubit gates per layer.
    """
    if model_type in ("heisenberg", "xy"):
        return 3 * n_edges  # RXX + RYY + RZZ per edge
    return n_edges  # RZZ only for TFIM


def classify_outcome(
    delta_e_over_gap: float | None,
    has_valid_regime: bool,
) -> str:
    """Classify pipeline outcome for a model.

    Parameters
    ----------
    delta_e_over_gap : float | None
        Best ΔE/gap achieved (None if pipeline didn't reach Phase 4).
    has_valid_regime : bool
        Whether any valid regime was found.

    Returns
    -------
    str
        "full_success" if ΔE/gap < 0.05,
        "partial_success" if pipeline runs but ΔE/gap ≥ 0.05,
        "failure" if no valid regime exists.
    """
    if not has_valid_regime:
        return "failure"
    if delta_e_over_gap is not None and delta_e_over_gap < 0.05:
        return "full_success"
    return "partial_success"


def compute_staggered_magnetization(z_expectations: list[float] | np.ndarray) -> float:
    """Compute staggered (Néel) magnetization from per-site Z expectations.

    M_s = (1/N) Σ_i (-1)^i ⟨Z_i⟩

    Parameters
    ----------
    z_expectations : list[float] or np.ndarray
        Per-site Z expectation values ⟨Z_i⟩.

    Returns
    -------
    float
        Staggered magnetization M_s.
    """
    z = np.asarray(z_expectations)
    n = len(z)
    signs = np.array([(-1) ** i for i in range(n)])
    return float(np.sum(signs * z) / n)


def generate_comparison_table(
    tfim_metrics: ComparativeMetrics,
    heisenberg_metrics: ComparativeMetrics,
) -> str:
    """Generate a formatted comparison table between two models.

    Parameters
    ----------
    tfim_metrics : ComparativeMetrics
        TFIM baseline metrics.
    heisenberg_metrics : ComparativeMetrics
        Heisenberg model metrics.

    Returns
    -------
    str
        Formatted markdown table.
    """

    def _fmt(val, fmt=".4f"):
        if val is None:
            return "N/A"
        return f"{val:{fmt}}"

    lines = [
        "| Metric | TFIM | Heisenberg | Ratio |",
        "|--------|------|------------|-------|",
        f"| ΔE/gap | {_fmt(tfim_metrics.delta_e_over_gap)} | "
        f"{_fmt(heisenberg_metrics.delta_e_over_gap)} | "
        f"{_fmt(heisenberg_metrics.delta_e_over_gap / tfim_metrics.delta_e_over_gap, '.1f') if tfim_metrics.delta_e_over_gap and heisenberg_metrics.delta_e_over_gap else 'N/A'} |",
        f"| Avg Fidelity | {tfim_metrics.avg_fidelity:.4f} | "
        f"{heisenberg_metrics.avg_fidelity:.4f} | "
        f"{heisenberg_metrics.avg_fidelity / tfim_metrics.avg_fidelity:.2f} |"
        if tfim_metrics.avg_fidelity > 0
        else f"| Avg Fidelity | {tfim_metrics.avg_fidelity:.4f} | "
        f"{heisenberg_metrics.avg_fidelity:.4f} | N/A |",
        f"| MPNN MSE | {_fmt(tfim_metrics.mpnn_mse)} | {_fmt(heisenberg_metrics.mpnn_mse)} | - |",
        f"| Valid Regime Width | {tfim_metrics.valid_regime_width} | "
        f"{heisenberg_metrics.valid_regime_width} | - |",
        f"| VQE Time (s) | {tfim_metrics.vqe_time_s:.1f} | "
        f"{heisenberg_metrics.vqe_time_s:.1f} | "
        f"{heisenberg_metrics.vqe_time_s / tfim_metrics.vqe_time_s:.1f}× |"
        if tfim_metrics.vqe_time_s > 0
        else f"| VQE Time (s) | {tfim_metrics.vqe_time_s:.1f} | "
        f"{heisenberg_metrics.vqe_time_s:.1f} | N/A |",
        f"| CX/layer | {tfim_metrics.cx_budget_per_layer} | "
        f"{heisenberg_metrics.cx_budget_per_layer} | "
        f"{heisenberg_metrics.cx_budget_per_layer / tfim_metrics.cx_budget_per_layer:.0f}× |"
        if tfim_metrics.cx_budget_per_layer > 0
        else f"| CX/layer | {tfim_metrics.cx_budget_per_layer} | "
        f"{heisenberg_metrics.cx_budget_per_layer} | N/A |",
        f"| Max Entanglement | {tfim_metrics.max_entanglement:.4f} | "
        f"{heisenberg_metrics.max_entanglement:.4f} | - |",
    ]
    return "\n".join(lines)
