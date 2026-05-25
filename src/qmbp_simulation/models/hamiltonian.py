"""
Hamiltonian Builder — Construct parameterized spin Hamiltonians for arbitrary
lattice topologies using SparsePauliOp (Qiskit 2.x).

Supports: chain_1d, kagome, triangular, ladder.

References
----------
- Mele et al., Nature Physics (2026): shallow-circuit depth constraint.
- V4 PoC: build_tfim_hamiltonian() pattern with SparsePauliOp.from_sparse_list.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qmbp_simulation.models.constants import SUPPORTED_TOPOLOGIES
from qmbp_simulation.models.data_models import LatticeConfig

# ── Lattice generators ────────────────────────────────────────────────────


def generate_chain_1d(n: int, periodic: bool = False) -> list[tuple[int, int]]:
    """Open or periodic 1D chain edges."""
    edges = [(i, i + 1) for i in range(n - 1)]
    if periodic:
        edges.append((n - 1, 0))
    return edges


def generate_ladder(n: int, periodic: bool = False) -> list[tuple[int, int]]:
    """Two-leg ladder: n must be even. Sites 0..n/2-1 = leg 0, n/2..n-1 = leg 1."""
    if n % 2 != 0:
        raise ValueError(f"Ladder topology requires even n_qubits, got {n}.")
    leg = n // 2
    edges: list[tuple[int, int]] = []
    # Intra-leg (leg 0)
    for i in range(leg - 1):
        edges.append((i, i + 1))
    # Intra-leg (leg 1)
    for i in range(leg, n - 1):
        edges.append((i, i + 1))
    # Rungs
    for i in range(leg):
        edges.append((i, i + leg))
    if periodic:
        edges.append((leg - 1, 0))
        edges.append((n - 1, leg))
    return edges


def generate_triangular(n: int) -> list[tuple[int, int]]:
    """Triangular lattice on a roughly-square grid.

    Maps *n* sites onto rows of width ``cols = ceil(sqrt(n))``.
    Edges: horizontal, vertical, and diagonal (NE) within the grid.
    Sites beyond *n* are excluded.
    """
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    edges: list[tuple[int, int]] = []

    def idx(r: int, c: int) -> int | None:
        s = r * cols + c
        return s if s < n else None

    for r in range(rows):
        for c in range(cols):
            s = idx(r, c)
            if s is None:
                continue
            # Right neighbour
            right = idx(r, c + 1)
            if right is not None:
                edges.append((s, right))
            # Down neighbour
            down = idx(r + 1, c)
            if down is not None:
                edges.append((s, down))
            # Diagonal (down-right)
            diag = idx(r + 1, c + 1)
            if diag is not None:
                edges.append((s, diag))
    return edges


def generate_kagome(n: int) -> list[tuple[int, int]]:
    """Kagome lattice (corner-sharing triangles).

    The Kagome lattice has 3 sites per unit cell.  We tile unit cells on a
    rectangular grid until we have at least *n* sites, then trim.  This gives
    coordination number 4 for bulk sites.
    """
    if n < 3:
        raise ValueError("Kagome lattice requires at least 3 sites.")
    # Number of unit cells along each direction
    cells = int(np.ceil(np.sqrt(n / 3)))
    # Generate sites: 3 per unit cell (A, B, C)
    site_map: dict[tuple[int, int, int], int] = {}
    counter = 0
    for cy in range(cells):
        for cx in range(cells):
            for sub in range(3):
                if counter >= n:
                    break
                site_map[(cx, cy, sub)] = counter
                counter += 1
            if counter >= n:
                break
        if counter >= n:
            break

    edges: list[tuple[int, int]] = []

    for cy in range(cells):
        for cx in range(cells):
            a = site_map.get((cx, cy, 0))
            b = site_map.get((cx, cy, 1))
            c = site_map.get((cx, cy, 2))
            # Intra-cell triangle
            if a is not None and b is not None:
                edges.append((a, b))
            if a is not None and c is not None:
                edges.append((a, c))
            if b is not None and c is not None:
                edges.append((b, c))
            # Inter-cell: B connects to A of right cell
            a_right = site_map.get((cx + 1, cy, 0))
            if b is not None and a_right is not None:
                edges.append((b, a_right))
            # Inter-cell: C connects to A of upper cell
            a_up = site_map.get((cx, cy + 1, 0))
            if c is not None and a_up is not None:
                edges.append((c, a_up))
    return edges


def compute_coordination_numbers(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Compute per-site degree from edge list."""
    coord = np.zeros(n, dtype=int)
    for i, j in edges:
        coord[i] += 1
        coord[j] += 1
    return coord


# ── Factory: build LatticeConfig from topology name ──────────────────────


def make_lattice(
    topology: str,
    n_qubits: int,
    J: float | np.ndarray = 1.0,
    h: float | np.ndarray = 1.0,
    periodic: bool = False,
) -> LatticeConfig:
    """Convenience factory that generates edges and coordination numbers
    automatically from the topology name.
    """
    if topology == "chain_1d":
        edges = generate_chain_1d(n_qubits, periodic)
    elif topology == "ladder":
        edges = generate_ladder(n_qubits, periodic)
    elif topology == "triangular":
        edges = generate_triangular(n_qubits)
    elif topology == "kagome":
        edges = generate_kagome(n_qubits)
    else:
        raise ValueError(f"Unknown topology '{topology}'. Supported: {SUPPORTED_TOPOLOGIES}")
    coord = compute_coordination_numbers(n_qubits, edges)
    return LatticeConfig(
        topology=topology,
        n_qubits=n_qubits,
        J=J,
        h=h,
        edges=edges,
        coordination_numbers=coord,
        periodic=periodic,
    )


# ── HamiltonianBuilder ───────────────────────────────────────────────────


class HamiltonianBuilder:
    """Construct spin Hamiltonians for arbitrary lattice topologies.

    All operators are returned as ``SparsePauliOp`` (Qiskit 2.x compliant).
    """

    # ── Task 2.1: build() ─────────────────────────────────────────────

    def build(self, lattice: LatticeConfig) -> SparsePauliOp:
        """Build H = -J Σ_{(i,j)} Z_i Z_j  -  h Σ_i X_i.

        Parameters
        ----------
        lattice : LatticeConfig
            Lattice specification (topology, edges, couplings, field).

        Returns
        -------
        SparsePauliOp
            The Hamiltonian as a Qiskit SparsePauliOp.
        """
        n = lattice.n_qubits
        terms: list[tuple[str, list[int], complex]] = []

        # ZZ interaction terms on lattice edges
        for bond_idx, (i, j) in enumerate(lattice.edges):
            j_val = lattice.J[bond_idx] if isinstance(lattice.J, np.ndarray) else lattice.J
            terms.append(("ZZ", [i, j], -j_val))

        # Transverse field X terms on all sites
        for site in range(n):
            h_val = lattice.h[site] if isinstance(lattice.h, np.ndarray) else lattice.h
            terms.append(("X", [site], -h_val))

        H = SparsePauliOp.from_sparse_list(terms, num_qubits=n)

        # Task 2.4: validate Hermiticity and dimensions
        self.validate(H, n)

        return H

    # ── Task 2.2: build_local_observables() ───────────────────────────

    def build_local_observables(
        self, lattice: LatticeConfig
    ) -> tuple[list[SparsePauliOp], list[SparsePauliOp]]:
        """Return per-site X operators and per-bond ZZ operators.

        Parameters
        ----------
        lattice : LatticeConfig

        Returns
        -------
        (ops_X, ops_ZZ)
            ops_X : list of SparsePauliOp, one per site.
            ops_ZZ : list of SparsePauliOp, one per bond in lattice.edges.
        """
        n = lattice.n_qubits
        ops_x = [SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=n) for i in range(n)]
        ops_zz = [
            SparsePauliOp.from_sparse_list([("ZZ", [i, j], 1.0)], num_qubits=n)
            for i, j in lattice.edges
        ]
        return ops_x, ops_zz

    # ── Task 2.3: build_graph_data() ──────────────────────────────────

    def build_graph_data(self, lattice: LatticeConfig) -> tuple[np.ndarray, np.ndarray]:
        """Return symmetric edge_index (COO) and coordination numbers for MPNN.

        Parameters
        ----------
        lattice : LatticeConfig

        Returns
        -------
        (edge_index, coordination_numbers)
            edge_index : np.ndarray of shape [2, 2*n_edges] — symmetric (undirected).
            coordination_numbers : np.ndarray of shape [n_qubits].
        """
        src = [i for i, j in lattice.edges]
        dst = [j for i, j in lattice.edges]
        # Make symmetric: add reverse edges
        edge_index = np.array([src + dst, dst + src], dtype=np.int64)
        return edge_index, lattice.coordination_numbers.copy()

    # ── Heisenberg XXZ Hamiltonian ───────────────────────────────────

    def build_heisenberg(self, lattice: LatticeConfig, delta: float = 1.0) -> SparsePauliOp:
        """Build H = J Σ_{(i,j)} (X_iX_j + Y_iY_j + Δ·Z_iZ_j) - h Σ_i Z_i.

        The Heisenberg XXZ model with external field in Z direction.
        At Δ=1 (isotropic), this is the standard Heisenberg model.

        Parameters
        ----------
        lattice : LatticeConfig
            Lattice specification (topology, edges, couplings, field).
        delta : float
            Anisotropy parameter. Δ=1 is isotropic (XXX), Δ=0 is XY model.

        Returns
        -------
        SparsePauliOp
            The Heisenberg XXZ Hamiltonian.
        """
        n = lattice.n_qubits
        terms: list[tuple[str, list[int], complex]] = []

        # Exchange interaction terms on lattice edges: J(XX + YY + Δ·ZZ)
        for bond_idx, (i, j) in enumerate(lattice.edges):
            j_val = lattice.J[bond_idx] if isinstance(lattice.J, np.ndarray) else lattice.J
            terms.append(("XX", [i, j], j_val))
            terms.append(("YY", [i, j], j_val))
            terms.append(("ZZ", [i, j], j_val * delta))

        # External field in Z direction on all sites
        for site in range(n):
            h_val = lattice.h[site] if isinstance(lattice.h, np.ndarray) else lattice.h
            terms.append(("Z", [site], -h_val))

        H = SparsePauliOp.from_sparse_list(terms, num_qubits=n)
        self.validate(H, n)
        return H

    # ── Heisenberg observables ───────────────────────────────────────

    def build_heisenberg_observables(
        self, lattice: LatticeConfig
    ) -> tuple[list[SparsePauliOp], list[SparsePauliOp]]:
        """Return per-site Z operators and per-bond S·S operators for Heisenberg.

        Parameters
        ----------
        lattice : LatticeConfig

        Returns
        -------
        (ops_Z, ops_SS)
            ops_Z : list of SparsePauliOp, one Z_i per site (magnetization).
            ops_SS : list of SparsePauliOp, one (XX+YY+ZZ)_{ij} per bond.
        """
        n = lattice.n_qubits
        ops_z = [SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], num_qubits=n) for i in range(n)]
        ops_ss = [
            SparsePauliOp.from_sparse_list(
                [("XX", [i, j], 1.0), ("YY", [i, j], 1.0), ("ZZ", [i, j], 1.0)],
                num_qubits=n,
            )
            for i, j in lattice.edges
        ]
        return ops_z, ops_ss

    # ── Task 2.4: validate() ─────────────────────────────────────────

    @staticmethod
    def validate(H: SparsePauliOp, n_qubits: int) -> None:
        """Verify Hermiticity and correct dimensions.

        Uses Pauli-level checks (O(n_terms)) instead of to_matrix() (O(2^n))
        so validation stays efficient for large systems.

        Raises
        ------
        ValueError
            If H is not Hermitian or has wrong dimensions.
        """
        if H.num_qubits != n_qubits:
            raise ValueError(f"Hamiltonian has {H.num_qubits} qubits, expected {n_qubits}.")
        # Hermiticity: all coefficients must be real for Pauli operators
        # (Pauli strings are self-adjoint, so H = H† iff all coeffs are real)
        if not np.allclose(H.coeffs.imag, 0, atol=1e-12):
            raise ValueError("Hamiltonian is not Hermitian (complex coefficients found).")
