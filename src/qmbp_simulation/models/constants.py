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
    "kagome",
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

DMRG_QUBIT_LIMIT: int = 40
"""n_qubits > 15 → DMRG (up to 40 qubits)."""
