"""Trotter step circuit builder for time evolution.

Provides reusable Suzuki-Trotter decomposition circuits for quench dynamics,
hardware time evolution, and DQPT analysis.

Supports:
- First-order Trotter: U₁(dt) = e^{-i dt H_ZZ} · e^{-i dt H_X}
- Second-order (symmetric) Trotter: U₂(dt) = e^{-i(dt/2)H_ZZ} · e^{-i dt H_X} · e^{-i(dt/2)H_ZZ}

The second-order decomposition has error O(dt³) per step vs O(dt²) for first-order,
significantly improving accuracy for the same number of steps.

Functions
---------
build_trotter_step : Build a single Trotter step circuit from a LatticeConfig
build_trotter_step_from_topology : Convenience wrapper using make_lattice internally

Usage
-----
>>> from qmbp_simulation.circuits.trotter import build_trotter_step
>>> from qmbp_simulation import make_lattice
>>> lattice = make_lattice("chain_1d", 10, J=1.0, h=2.0)
>>> trotter_qc = build_trotter_step(lattice, dt=0.1, order=2)
>>> # Use in time evolution: compose repeatedly
>>> full_circuit = init_qc.copy()
>>> for _ in range(n_steps):
...     full_circuit = full_circuit.compose(trotter_qc)
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit

from qmbp_simulation.models.data_models import LatticeConfig


def build_trotter_step(
    lattice: LatticeConfig,
    dt: float,
    *,
    order: int = 2,
    model: str = "tfim",
) -> QuantumCircuit:
    """Build a single Trotter step circuit for time evolution under H.


    For the standard TFIM: H = -J sum_{(i,j)} Z_i Z_j - h sum_i X_i

    The unitary is U(dt) = e^{-i H dt}, decomposed via Suzuki-Trotter.
    """
    if order not in (1, 2):
        raise ValueError(f"Trotter order must be 1 or 2, got {order}")
    if model not in ("tfim", "tfim_longitudinal"):
        raise ValueError(f"Unsupported model '{model}'. Use 'tfim' or 'tfim_longitudinal'.")

    qc = QuantumCircuit(lattice.n_qubits)

    if model == "tfim":
        _apply_tfim_trotter(qc, lattice, dt, order)
    elif model == "tfim_longitudinal":
        _apply_tfim_longitudinal_trotter(qc, lattice, dt, order)

    return qc


def build_trotter_step_from_topology(
    topology: str,
    n_qubits: int,
    h: float,
    dt: float,
    *,
    J: float = 1.0,
    order: int = 2,
    model: str = "tfim",
) -> QuantumCircuit:
    """Convenience: build Trotter step from topology parameters.


    Internally calls make_lattice and build_trotter_step.
    """
    from qmbp_simulation import make_lattice

    lattice = make_lattice(topology, n_qubits, J=J, h=h)
    return build_trotter_step(lattice, dt, order=order, model=model)


# ═══════════════════════════════════════════════════════════════════════════════
# Internal: model-specific Trotter decompositions
# ═══════════════════════════════════════════════════════════════════════════════


def _apply_tfim_trotter(
    qc: QuantumCircuit,
    lattice: LatticeConfig,
    dt: float,
    order: int,
) -> None:
    """Apply TFIM Trotter step: H = -J sum Z_iZ_j - h sum X_i.

    Gate mapping:
    - ZZ term: exp(+i J dt Z_iZ_j) = RZZ(-2 J dt)
    - X term: exp(+i h dt X_i) = RX(-2 h dt)
    """
    n_qubits = lattice.n_qubits
    edges = lattice.edges

    if order == 1:
        # First-order: U_ZZ(dt) · U_X(dt)
        _apply_zz_layer(qc, edges, lattice.J, dt)
        _apply_x_layer(qc, n_qubits, lattice.h, dt)
    else:
        # Second-order (symmetric): U_ZZ(dt/2) · U_X(dt) · U_ZZ(dt/2)
        _apply_zz_layer(qc, edges, lattice.J, dt / 2.0)
        _apply_x_layer(qc, n_qubits, lattice.h, dt)
        _apply_zz_layer(qc, edges, lattice.J, dt / 2.0)


def _apply_tfim_longitudinal_trotter(
    qc: QuantumCircuit,
    lattice: LatticeConfig,
    dt: float,
    order: int,
) -> None:
    """Apply TFIM + longitudinal field Trotter step.

    H = -J sum Z_iZ_j - h sum X_i - g sum Z_i

    Additional Z-rotation layer for the longitudinal field.
    The g parameter is extracted from the lattice config if available,
    otherwise defaults to 0 (falls back to standard TFIM).
    """
    n_qubits = lattice.n_qubits
    edges = lattice.edges
    # g is stored as an extra attribute on extended LatticeConfig;
    # default to 0 if not present (standard TFIM behavior)
    g = getattr(lattice, "g", 0.0)

    if order == 1:
        _apply_zz_layer(qc, edges, lattice.J, dt)
        _apply_x_layer(qc, n_qubits, lattice.h, dt)
        if abs(g) > 1e-15:
            _apply_z_layer(qc, n_qubits, g, dt)
    else:
        _apply_zz_layer(qc, edges, lattice.J, dt / 2.0)
        _apply_x_layer(qc, n_qubits, lattice.h, dt)
        if abs(g) > 1e-15:
            _apply_z_layer(qc, n_qubits, g, dt)
        _apply_zz_layer(qc, edges, lattice.J, dt / 2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Internal: gate layers
# ═══════════════════════════════════════════════════════════════════════════════


def _apply_zz_layer(
    qc: QuantumCircuit,
    edges: list[tuple[int, int]],
    J,
    dt: float,
) -> None:
    """Apply RZZ gates on all edges: exp(+i J dt Z_iZ_j) → RZZ(-2*J*dt)."""
    for bond_idx, (i, j) in enumerate(edges):
        j_val = J[bond_idx] if isinstance(J, np.ndarray) else J
        qc.rzz(-2 * j_val * dt, i, j)


def _apply_x_layer(
    qc: QuantumCircuit,
    n_qubits: int,
    h,
    dt: float,
) -> None:
    """Apply RX gates on all sites: exp(+i h dt X_i) → RX(-2*h*dt)."""
    for site in range(n_qubits):
        h_val = h[site] if isinstance(h, np.ndarray) else h
        qc.rx(-2 * h_val * dt, site)


def _apply_z_layer(
    qc: QuantumCircuit,
    n_qubits: int,
    g: float,
    dt: float,
) -> None:
    """Apply RZ gates on all sites: exp(+i g dt Z_i) → RZ(-2*g*dt)."""
    for site in range(n_qubits):
        qc.rz(-2 * g * dt, site)
