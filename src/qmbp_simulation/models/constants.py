"""
Physics constants for the GNN-HVA quantum simulation framework.

Canonical location for all physics-level constants used across the package.
Hardware/analysis constants live in ``qmbp_simulation.analysis.data_models``.

Sources:
    - src/poc/v6/config.py (SUPPORTED_TOPOLOGIES, MAX_P_LAYERS, qubit limits)
    - src/poc/v6/config_v61.py (gradient analysis thresholds)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Topology and circuit depth constraints
# ---------------------------------------------------------------------------

SUPPORTED_TOPOLOGIES: tuple[str, ...] = (
    "chain_1d",
    "heavy_hex",
    "kagome",
    "square",
    "triangular",
    "ladder",
)
"""Valid lattice topologies for the framework."""

MAX_P_LAYERS: int = 10
"""Maximum HVA circuit depth (Mele et al. constraint)."""

# ---------------------------------------------------------------------------
# Solver dispatch thresholds
# ---------------------------------------------------------------------------

EXACT_DIAG_QUBIT_LIMIT: int = 15
"""n_qubits ≤ 15 → exact diagonalization."""

DMRG_QUBIT_LIMIT: int = 200
"""n_qubits > 15 → DMRG (up to 200 qubits). 1D TFIM validated at N=120 with χ=64."""

# ---------------------------------------------------------------------------
# VQE optimizer methods
# ---------------------------------------------------------------------------

SUPPORTED_VQE_METHODS: tuple[str, ...] = ("L-BFGS-B", "COBYLA", "Nelder-Mead")
"""Optimizer methods supported by VQEOptimizer.

- L-BFGS-B: gradient-based (finite differences), best for exact backends.
- COBYLA: gradient-free, tolerant to shot noise, for MPS shot-based backends.
- Nelder-Mead: gradient-free simplex, alternative to COBYLA.
"""

# ---------------------------------------------------------------------------
# Statevector / MPS thresholds
# ---------------------------------------------------------------------------

STATEVECTOR_MAX_N: int = 22
"""Maximum N for sparse eigsh ground state vector extraction.

At N=22, the statevector has 2^22 = 4M complex128 amplitudes (~64 MB).
Above this, statevector-based methods (fidelity, entropy) are infeasible.
Use DMRG energy + skip fidelity for N > 22.
"""

MPS_DEFAULT_CHI_MAX: int = 64
"""Default MPS bond dimension for Aer/TeNPy evaluation.

Validated exact (|MPS - Statevector| ≈ 1e-14) for HVA p≤2 on 1D TFIM
at any N. Sufficient for heavy_hex and ladder topologies.
Reference: V7 experiments 3A/3B scaling validation.
"""

# ---------------------------------------------------------------------------
# Default experiment parameters
# ---------------------------------------------------------------------------

DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44)
"""Default random seeds for reproducibility across experiments."""


# ---------------------------------------------------------------------------
# Pipeline thresholds and defaults
# ---------------------------------------------------------------------------

DE_GAP_THRESHOLD: float = 0.05
"""Primary success criterion: ΔE/gap < 5%."""

FIDELITY_THRESHOLD_TFIM: float = 0.93
"""Fidelity filter for TFIM (Phase 3 training data quality)."""

FIDELITY_THRESHOLD_HEISENBERG: float = 0.60
"""Relaxed fidelity filter for Heisenberg/XY models."""

THETA_SMOOTHNESS_CHAIN_BREAK: float = 1.0
"""θ_smoothness above this indicates a warm-start chain break."""

GEN_GAP_WARNING: float = 0.01
"""Generalization gap above this triggers a warning (85% failure rate)."""

GEN_GAP_CATASTROPHIC: float = 0.05
"""Generalization gap above this aborts Phase 4 (95%+ failure rate)."""

COBYLA_AUTO_SWITCH_THRESHOLD: int = 8
"""Auto-switch from L-BFGS-B to COBYLA when n_params > this value."""

VQE_WALL_CLOCK_LIMIT_PER_POINT: float = 600.0
"""Wall-clock timeout (seconds) per optimize() call in VQE sweeps.
If a single optimize() call exceeds this, the COBYLA cost function
returns a stale value to force convergence. The effective per-restart
timeout is this value / (n_restarts + 1).

For reference: N=10 p=1 chain_1d → ~5-10s/point, N=20 p=4 → ~60-120s/point.
600s gives 10x margin for the hardest known configs."""

VQE_RESTART_STAGNATION_THRESHOLD: int = 3
"""Number of consecutive restarts without improvement before early-stopping.
If the last N restarts all failed to beat the best energy, remaining restarts
are skipped. Prevents wasted compute on landscapes where warm-start already
found the global minimum."""

VQE_RESTART_IMPROVEMENT_TOL: float = 1e-10
"""Minimum energy improvement to count a restart as 'useful'.
Energy changes smaller than this are treated as stagnation."""

# ---------------------------------------------------------------------------
# MPNN training defaults
# ---------------------------------------------------------------------------

MPNN_DEFAULT_HIDDEN_DIM: int = 64
"""Default MPNN hidden dimension (128 recommended for N≥10)."""

MPNN_DEFAULT_LR: float = 1e-3
"""Default MPNN learning rate."""

MPNN_DEFAULT_WEIGHT_DECAY: float = 1e-4
"""Default L2 regularization weight for Adam optimizer."""

MPNN_DEFAULT_PATIENCE: int = 150
"""Default ReduceLROnPlateau patience (epochs without improvement)."""

MPNN_DEFAULT_DROPOUT: float = 0.1
"""Dropout rate in MPNN output heads (hardcoded in architecture)."""

# ---------------------------------------------------------------------------
# Data augmentation defaults
# ---------------------------------------------------------------------------

AUGMENTATION_NOISE_SIGMA: float = 0.02
"""Gaussian noise σ for θ data augmentation."""

AUGMENTATION_N_COPIES: int = 2
"""Number of noisy copies per training point."""

AUGMENTATION_DATASET_THRESHOLD: int = 40
"""Apply augmentation when dataset has fewer than this many points."""

# ---------------------------------------------------------------------------
# SWA (Stochastic Weight Averaging) defaults
# ---------------------------------------------------------------------------

SWA_EXTRA_EPOCHS: int = 200
"""Number of extra low-LR epochs for SWA averaging."""

SWA_LR: float = 1e-4
"""Learning rate for SWA extra training phase."""

SWA_AVERAGE_WEIGHT: float = 0.5
"""Mixing weight: model = w*original + (1-w)*swa_model."""

# ---------------------------------------------------------------------------
# VQE refinement defaults
# ---------------------------------------------------------------------------

VQE_REFINEMENT_DE_GAP_MIN: float = 0.05
"""Minimum ΔE/gap to trigger VQE refinement (below this = already passing)."""

VQE_REFINEMENT_DE_GAP_MAX: float = 5.0
"""Maximum ΔE/gap for refinement (above this = too far gone)."""

# ---------------------------------------------------------------------------
# Outlier detection defaults
# ---------------------------------------------------------------------------

OUTLIER_THRESHOLD: float = 2.0
"""MAD-based outlier detection threshold (std deviations)."""

OUTLIER_FIDELITY_FLOOR: float = 0.5
"""Fidelity below this (with high-fidelity neighbors) = outlier."""
