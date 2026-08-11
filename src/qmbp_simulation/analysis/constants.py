"""Analysis Constants — Single source of truth for quality criteria thresholds.

This module is the LOWEST-LEVEL module in the analysis subpackage.
It has NO imports from any qmbp_simulation module (only stdlib/builtins),
which makes it safe to import from anywhere without circular dependencies.

All quality thresholds, metric boundaries, and diagnostic constants live here.
Other modules (metrics.py, failures_tests.py, cross_n_validator.py) import
from this file rather than defining their own copies.

Usage:
    from qmbp_simulation.analysis.constants import DE_GAP_THRESHOLD, MAX_ABS_ERROR
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# Dual Quality Criterion (pass/fail decisions)
# ═══════════════════════════════════════════════════════════════════════════════

DE_GAP_THRESHOLD: float = 0.05
"""Relative energy error threshold: |ΔE|/gap < 5%.
A point passes if its relative error normalized by the spectral gap
is below this threshold. This is criterion 1 of the dual criterion."""

MAX_ABS_ERROR: float = 0.10
"""Absolute energy error cap (Hartree). Points above this are failures
regardless of ΔE/gap. Prevents gap masking: for large gaps (h >> h_c),
ΔE/gap can be small while |ΔE| is physically significant."""

MIN_FIDELITY: float = 0.97
"""Minimum state overlap for acceptable quality (noiseless only).
Fidelity check is skipped when N > 22 (statevector unavailable)."""

# ═══════════════════════════════════════════════════════════════════════════════
# Data Quality Audit Thresholds
# ═══════════════════════════════════════════════════════════════════════════════

TRAINING_BAD_RATIO_THRESHOLD: float = 0.20
"""If n_bad/n_total > 20% in a NPZ but zoo model has pass_rate > 0.60,
flag training/zoo incoherence."""

ZOO_PASS_FOR_INCOHERENCE_FLAG: float = 0.60
"""Zoo pass_rate above which a high bad-ratio in NPZ is flagged as incoherence."""

PASS_RATE_REGRESSION_THRESHOLD: float = 0.15
"""If max pass_rate_dual for a topology drops > 15% vs previous dashboard,
flag as regression."""

H_FRONTIER_MONOTONICITY_TOLERANCE: float = 0.10
"""h_frontier should increase (or stay flat) with N. If it drops more than
this tolerance, flag as data quality anomaly."""

GAP_MASKING_THRESHOLD: float = 0.10
"""Difference between pass_rate_5pct and pass_rate_dual above which
gap masking is considered significant."""

MIN_TRAINING_DUAL_PASS_RATE: float = 0.30
"""Minimum fraction of points passing dual criterion for a NPZ to be
considered useful for MPNN training."""

MIN_TRAINING_POINTS_FOR_SIGNAL: int = 5
"""Minimum absolute count of dual-passing points for a NPZ to contribute
useful training signal."""
