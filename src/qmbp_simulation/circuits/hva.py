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

Two circuit representations:
  - ``create()``: Explicit RZZ/RX gates (default, backward-compatible).
  - ``create_pauli_evolution()``: PauliEvolutionGate representation. Exposes
    commuting structure to the transpiler, yielding ~11% lower 2Q-depth with
    identical gate count. Recommended for hardware deployment.
    Ref: IBM tutorial "Compilation methods for Hamiltonian simulation circuits".
"""

from __future__ import annotations

from qiskit.circuit import ParameterVector, QuantumCircuit

from qmbp_simulation.models import MAX_P_LAYERS, LatticeConfig


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
        if p_layers > MAX_P_LAYERS:
            raise ValueError(
                f"p_layers={p_layers} exceeds the maximum of {MAX_P_LAYERS}. "
                f"Mele et al. (Nature Physics, 2026) show that non-unital noise "
                f"truncates effective circuit depth to O(log n). HVA circuits "
                f"MUST have p ≤ {MAX_P_LAYERS} layers for NISQ viability."
            )

        if n_qubits != lattice.n_qubits:
            raise ValueError(
                f"n_qubits={n_qubits} does not match lattice.n_qubits={lattice.n_qubits}."
            )

        if not lattice.edges:
            raise ValueError(
                f"Lattice has no edges (topology='{lattice.topology}', N={n_qubits}). "
                f"Cannot build HVA circuit without ZZ interaction terms."
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

    def create_pauli_evolution(
        self,
        n_qubits: int,
        p_layers: int,
        lattice: LatticeConfig,
    ) -> tuple[QuantumCircuit, ParameterVector]:
        """Build an HVA circuit using PauliEvolutionGate for better transpilation.

        Functionally identical to ``create()`` but uses ``PauliEvolutionGate``
        to represent each commuting layer (ZZ interactions, X field). This
        gives the transpiler structural information about gate commutativity,
        enabling ~11% lower 2Q-depth via better parallel scheduling.

        Validated 2026-06-05: same layout, same n_2Q gates (34), but
        2Q-depth 24 vs 27 (original). Recommended for hardware deployment.

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
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp

        if p_layers > MAX_P_LAYERS:
            raise ValueError(
                f"p_layers={p_layers} exceeds the maximum of {MAX_P_LAYERS}. "
                f"Mele et al. (Nature Physics, 2026) depth constraint."
            )

        if n_qubits != lattice.n_qubits:
            raise ValueError(
                f"n_qubits={n_qubits} does not match lattice.n_qubits={lattice.n_qubits}."
            )

        if not lattice.edges:
            raise ValueError(
                f"Lattice has no edges (topology='{lattice.topology}', N={n_qubits}). "
                f"Cannot build HVA circuit without ZZ interaction terms."
            )

        # Build ZZ operator: sum of ZZ on each edge (commuting group)
        zz_terms = []
        for i, j in lattice.edges:
            label = ["I"] * n_qubits
            label[n_qubits - 1 - i] = "Z"
            label[n_qubits - 1 - j] = "Z"
            zz_terms.append(("".join(label), 1.0))
        H_zz = SparsePauliOp.from_list(zz_terms)

        # Build X operator: sum of X on each site (commuting group)
        x_terms = []
        for i in range(n_qubits):
            label = ["I"] * n_qubits
            label[n_qubits - 1 - i] = "X"
            x_terms.append(("".join(label), 1.0))
        H_x = SparsePauliOp.from_list(x_terms)

        qc = QuantumCircuit(n_qubits)
        theta = ParameterVector("θ", 2 * p_layers)

        # Initial state: |+⟩^N
        qc.h(range(n_qubits))

        # HVA layers using PauliEvolutionGate
        for layer in range(p_layers):
            theta_zz = theta[layer * 2]
            theta_x = theta[layer * 2 + 1]

            # e^{-i * 2*theta_zz * H_ZZ} (factor 2 matches RZZ(2θ) convention)
            qc.append(
                PauliEvolutionGate(H_zz, time=2 * theta_zz),
                range(n_qubits),
            )
            # e^{-i * 2*theta_x * H_X}
            qc.append(
                PauliEvolutionGate(H_x, time=2 * theta_x),
                range(n_qubits),
            )

        return qc, theta

    def create_frustrated_tfim(
        self,
        n_qubits: int,
        p_layers: int,
        lattice: LatticeConfig,
    ) -> tuple[QuantumCircuit, ParameterVector]:
        """Build an HVA circuit for frustrated TFIM (J1-J2).

        Gate structure per layer:
        - RZZ(2θ_nn) on NN bonds (from lattice.edges)
        - RZZ(2θ_nnn) on NNN bonds (computed from topology)
        - RX(2θ_x) on all sites (transverse field)

        This mirrors H = -J₁·ZZ_nn + J₂·ZZ_nnn - h·X.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        p_layers : int
            Number of HVA layers (MUST be ≤ 2).
        lattice : LatticeConfig
            Lattice specification with edge list (defines NN bonds).

        Returns
        -------
        (qc, theta)
            qc : QuantumCircuit with 3*p_layers parameters.
            theta : ParameterVector of length 3*p_layers.

        Notes
        -----
        Hardware viability: NNN bonds add extra CX gates. At N=6 p=1, this
        circuit uses ~27 CZ gates (exceeds ZNE budget of 18). Viable for
        noiseless simulation; hardware deployment limited to N=4.
        """
        if p_layers > MAX_P_LAYERS:
            raise ValueError(
                f"p_layers={p_layers} exceeds the maximum of {MAX_P_LAYERS}. "
                f"Mele et al. (Nature Physics, 2026) depth constraint."
            )

        if n_qubits != lattice.n_qubits:
            raise ValueError(
                f"n_qubits={n_qubits} does not match lattice.n_qubits={lattice.n_qubits}."
            )

        if not lattice.edges:
            raise ValueError(
                f"Lattice has no edges (topology='{lattice.topology}', N={n_qubits}). "
                f"Cannot build HVA circuit without ZZ interaction terms."
            )

        # Compute NNN edges from topology
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        nnn_edges = HamiltonianBuilder._generate_nnn_edges(lattice)

        qc = QuantumCircuit(n_qubits)
        theta = ParameterVector("θ", 3 * p_layers)

        # Initial state: |+⟩^N
        qc.h(range(n_qubits))

        for layer in range(p_layers):
            theta_nn = theta[layer * 3]
            theta_nnn = theta[layer * 3 + 1]
            theta_x = theta[layer * 3 + 2]

            # RZZ(2θ_nn) on NN bonds
            for i, j in lattice.edges:
                qc.rzz(2 * theta_nn, i, j)

            # RZZ(2θ_nnn) on NNN bonds
            for i, j in nnn_edges:
                qc.rzz(2 * theta_nnn, i, j)

            # RX(2θ_x) on all qubits
            for i in range(n_qubits):
                qc.rx(2 * theta_x, i)

        return qc, theta

    def create_bond_resolved(
        self,
        n_qubits: int,
        p_layers: int,
        lattice: LatticeConfig,
    ) -> tuple[QuantumCircuit, ParameterVector]:
        """Build a bond-resolved HVA circuit with per-bond and per-site parameters.

        Unlike the standard HVA which uses one θ_zz for ALL bonds and one θ_x
        for ALL sites (2 params per layer), this variant assigns independent
        parameters to each bond and each site:

        - θ_zz_k for each edge k ∈ {0, ..., E-1}
        - θ_x_i for each qubit i ∈ {0, ..., N-1}

        This increases expressibility without increasing circuit depth or gate
        count (same CX budget as global HVA). The motivation is:
        1. Capture symmetry-broken states near QPT (Fusco et al., 2026).
        2. Make the parameter space high-dimensional (classical-hard to search).
        3. Make the GNN predictor *essential* (interpolation fails in 20+ dims).

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
            qc : QuantumCircuit with (n_edges + n_qubits) * p_layers parameters.
            theta : ParameterVector of length (n_edges + n_qubits) * p_layers.

        Raises
        ------
        ValueError
            If ``p_layers > 2`` (Mele et al. depth constraint).

        Notes
        -----
        Gate count is IDENTICAL to standard HVA — same RZZ on each edge, same
        RX on each site. Only the parametrization differs (local vs global).
        This means ZNE budget is unchanged: p=1 N=10 ≈ 18 CX → ZNE works.

        The parameter ordering convention is:
            [θ_zz_0, θ_zz_1, ..., θ_zz_{E-1}, θ_x_0, θ_x_1, ..., θ_x_{N-1}]
        repeated for each layer.
        """
        if p_layers > MAX_P_LAYERS:
            raise ValueError(
                f"p_layers={p_layers} exceeds the maximum of {MAX_P_LAYERS}. "
                f"Mele et al. (Nature Physics, 2026) depth constraint."
            )

        if n_qubits != lattice.n_qubits:
            raise ValueError(
                f"n_qubits={n_qubits} does not match lattice.n_qubits={lattice.n_qubits}."
            )

        if not lattice.edges:
            raise ValueError(
                f"Lattice has no edges (topology='{lattice.topology}', N={n_qubits}). "
                f"Cannot build HVA circuit without ZZ interaction terms."
            )

        n_edges = len(lattice.edges)
        params_per_layer = n_edges + n_qubits

        qc = QuantumCircuit(n_qubits)
        theta = ParameterVector("θ", params_per_layer * p_layers)

        # Initial state: |+⟩^N (paramagnetic ground state at h → ∞)
        qc.h(range(n_qubits))

        # HVA layers with per-bond / per-site parameters
        for layer in range(p_layers):
            offset = layer * params_per_layer

            # RZZ(2·θ_zz_k) on each lattice edge k
            for k, (i, j) in enumerate(lattice.edges):
                qc.rzz(2 * theta[offset + k], i, j)

            # RX(2·θ_x_i) on each qubit i
            for i in range(n_qubits):
                qc.rx(2 * theta[offset + n_edges + i], i)

        return qc, theta

    def create_tfim_longitudinal(
        self,
        n_qubits: int,
        p_layers: int,
        lattice: LatticeConfig,
    ) -> tuple[QuantumCircuit, ParameterVector]:
        """Build an HVA circuit for TFIM + longitudinal field.

        Gate structure per layer: RZZ on edges, then RX on sites, then RZ on sites.
        This mirrors H = -J·ZZ - h·X - g·Z, adding the RZ layer that the
        standard TFIM HVA lacks (E4 showed this is necessary for g>0).

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
            qc : QuantumCircuit with 3*p_layers parameters.
            theta : ParameterVector of length 3*p_layers.

        Raises
        ------
        ValueError
            If ``p_layers > 2`` (Mele et al. depth constraint).
        """
        if p_layers > MAX_P_LAYERS:
            raise ValueError(
                f"p_layers={p_layers} exceeds the maximum of {MAX_P_LAYERS}. "
                f"Mele et al. (Nature Physics, 2026) depth constraint."
            )

        if n_qubits != lattice.n_qubits:
            raise ValueError(
                f"n_qubits={n_qubits} does not match lattice.n_qubits={lattice.n_qubits}."
            )

        if not lattice.edges:
            raise ValueError(
                f"Lattice has no edges (topology='{lattice.topology}', N={n_qubits}). "
                f"Cannot build HVA circuit without ZZ interaction terms."
            )

        qc = QuantumCircuit(n_qubits)
        theta = ParameterVector("θ", 3 * p_layers)

        # Initial state: |+⟩^N (paramagnetic ground state at h → ∞)
        qc.h(range(n_qubits))

        # HVA layers: e^{-iθ_z H_Z} · e^{-iθ_x H_X} · e^{-iθ_zz H_ZZ}
        for layer in range(p_layers):
            theta_zz = theta[layer * 3]
            theta_x = theta[layer * 3 + 1]
            theta_z = theta[layer * 3 + 2]

            # RZZ(2θ_zz) on each lattice edge
            for i, j in lattice.edges:
                qc.rzz(2 * theta_zz, i, j)

            # RX(2θ_x) on all qubits (transverse field)
            for i in range(n_qubits):
                qc.rx(2 * theta_x, i)

            # RZ(2θ_z) on all qubits (longitudinal field)
            for i in range(n_qubits):
                qc.rz(2 * theta_z, i)

        return qc, theta

    def create_heisenberg(
        self,
        n_qubits: int,
        p_layers: int,
        lattice: LatticeConfig,
        initial_state: str = "neel",
    ) -> tuple[QuantumCircuit, ParameterVector]:
        """Build an HVA circuit for the Heisenberg XXZ model.

        Gate structure per layer: RXX + RYY + RZZ on edges, then RZ on sites.
        This mirrors the Hamiltonian H = J(XX + YY + Δ·ZZ) - h·Z.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        p_layers : int
            Number of HVA layers (MUST be ≤ 2).
        lattice : LatticeConfig
            Lattice specification with edge list.
        initial_state : str
            Initial state preparation:
            - "neel": |↑↓↑↓...⟩ (antiferromagnetic, natural for Heisenberg)
            - "plus": |+⟩^N (paramagnetic, same as TFIM)
            - "zero": |0⟩^N (all spin-up, ferromagnetic)

        Returns
        -------
        (qc, theta)
            qc : QuantumCircuit with 4*p_layers parameters.
            theta : ParameterVector of length 4*p_layers.

        Raises
        ------
        ValueError
            If ``p_layers > 2`` (Mele et al. depth constraint).
        """
        if p_layers > MAX_P_LAYERS:
            raise ValueError(
                f"p_layers={p_layers} exceeds the maximum of {MAX_P_LAYERS}. "
                f"Mele et al. (Nature Physics, 2026) depth constraint."
            )

        if n_qubits != lattice.n_qubits:
            raise ValueError(
                f"n_qubits={n_qubits} does not match lattice.n_qubits={lattice.n_qubits}."
            )

        qc = QuantumCircuit(n_qubits)
        theta = ParameterVector("θ", 4 * p_layers)

        # Initial state preparation
        if initial_state == "neel":
            # Néel state: |↑↓↑↓...⟩ = |0101...⟩ (X on odd sites)
            for i in range(1, n_qubits, 2):
                qc.x(i)
        elif initial_state == "plus":
            # |+⟩^N (paramagnetic — same as TFIM)
            qc.h(range(n_qubits))
        elif initial_state == "zero":
            # |0⟩^N = |↑↑↑...⟩ (ferromagnetic, no gates needed)
            pass
        else:
            raise ValueError(
                f"Unknown initial_state '{initial_state}'. Use 'neel', 'plus', or 'zero'."
            )

        # HVA layers: e^{-iθ_z H_Z} · e^{-iθ_zz H_ZZ} · e^{-iθ_yy H_YY} · e^{-iθ_xx H_XX}
        for layer in range(p_layers):
            theta_xx = theta[layer * 4]
            theta_yy = theta[layer * 4 + 1]
            theta_zz = theta[layer * 4 + 2]
            theta_z = theta[layer * 4 + 3]

            # RXX(2θ_xx) on each lattice edge
            for i, j in lattice.edges:
                qc.rxx(2 * theta_xx, i, j)

            # RYY(2θ_yy) on each lattice edge
            for i, j in lattice.edges:
                qc.ryy(2 * theta_yy, i, j)

            # RZZ(2θ_zz) on each lattice edge
            for i, j in lattice.edges:
                qc.rzz(2 * theta_zz, i, j)

            # RZ(2θ_z) on all qubits
            for i in range(n_qubits):
                qc.rz(2 * theta_z, i)

        return qc, theta
