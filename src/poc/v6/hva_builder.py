"""
HVA Circuit Builder — Shallow Hamiltonian Variational Ansatz with lattice-aware
gate placement.

Constructs HVA circuits that respect the physical topology of arbitrary spin
systems.  Enforces the Mele et al. (Nature Physics, 2026) depth constraint:
p ≤ 2 layers.

Gate convention (matching V4 PoC):
  - Initial state: |+⟩^N  (Hadamard layer)
  - Per layer: RZZ(2θ_zz) on each lattice edge, then RX(2θ_x) on each qubit
  - Physical 2θ scaling: qc.rzz(2*θ, i, j), qc.rx(2*θ, i)
  - Parameters: 2*p total (one θ_zz and one θ_x per layer)
"""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit, ParameterVector

from .config import LatticeConfig, MAX_P_LAYERS


class HVACircuitBuilder:
    """Build lattice-aware HVA circuits with strict depth compliance."""

    def create(
        self,
        n_qubits: int,
        p_layers: int,
        lattice: LatticeConfig,
    ) -> tuple[QuantumCircuit, ParameterVector]:
        """Build an HVA circuit for the given lattice topology.

        Parameters
        ----------
        n_qubits : int
            Number of qubits (must match ``lattice.n_qubits``).
        p_layers : int
            Number of HVA layers (MUST be ≤ 2).
        lattice : LatticeConfig
            Lattice specification with edge list.

        Returns
        -------
        (qc, theta)
            qc : QuantumCircuit with 2*p_layers parameters.
            theta : ParameterVector of length 2*p_layers.

        Raises
        ------
        ValueError
            If ``p_layers > 2`` (Mele et al. depth constraint).
        """
        # ── Task 4.2: depth validation ────────────────────────────────
        if p_layers > MAX_P_LAYERS:
            raise ValueError(
                f"p_layers={p_layers} exceeds the maximum of {MAX_P_LAYERS}. "
                f"Mele et al. (Nature Physics, 2026) show that non-unital noise "
                f"truncates effective circuit depth to O(log n). HVA circuits "
                f"MUST have p ≤ {MAX_P_LAYERS} layers for NISQ viability."
            )

        if n_qubits != lattice.n_qubits:
            raise ValueError(
                f"n_qubits={n_qubits} does not match lattice.n_qubits="
                f"{lattice.n_qubits}."
            )

        qc = QuantumCircuit(n_qubits)
        theta = ParameterVector("θ", 2 * p_layers)

        # Initial state: |+⟩^N (paramagnetic ground state at h → ∞)
        qc.h(range(n_qubits))

        # HVA layers: e^{-iθ_x H_X} · e^{-iθ_zz H_ZZ}
        for layer in range(p_layers):
            theta_zz = theta[layer * 2]
            theta_x = theta[layer * 2 + 1]

            # RZZ(2θ_zz) on each lattice edge
            for i, j in lattice.edges:
                qc.rzz(2 * theta_zz, i, j)

            # RX(2θ_x) on all qubits
            for i in range(n_qubits):
                qc.rx(2 * theta_x, i)

        return qc, theta
