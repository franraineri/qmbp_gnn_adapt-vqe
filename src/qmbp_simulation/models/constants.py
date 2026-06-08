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

MAX_P_LAYERS: int = 2
"""Maximum HVA circuit depth (Mele et al. constraint)."""

# ---------------------------------------------------------------------------
# Solver dispatch thresholds
# ---------------------------------------------------------------------------

EXACT_DIAG_QUBIT_LIMIT: int = 15
"""n_qubits ≤ 15 → exact diagonalization."""

DMRG_QUBIT_LIMIT: int = 100
"""n_qubits > 15 → DMRG (up to 100 qubits). 1D TFIM validated at N=50 in <3s."""

# ---------------------------------------------------------------------------
# VQE optimizer methods
# ---------------------------------------------------------------------------

SUPPORTED_VQE_METHODS: tuple[str, ...] = ("L-BFGS-B", "COBYLA", "Nelder-Mead")
"""Optimizer methods supported by VQEOptimizer.

- L-BFGS-B: gradient-based (finite differences), best for exact backends.
- COBYLA: gradient-free, tolerant to shot noise, for MPS shot-based backends.
- Nelder-Mead: gradient-free simplex, alternative to COBYLA.
"""
