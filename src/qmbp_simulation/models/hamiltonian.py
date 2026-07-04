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

import logging

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qmbp_simulation.models.constants import SUPPORTED_TOPOLOGIES
from qmbp_simulation.models.data_models import LatticeConfig

logger = logging.getLogger(__name__)

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


def generate_square(n: int, periodic: bool = False) -> list[tuple[int, int]]:
    """Square lattice (2D grid with nearest-neighbor edges only).

    Maps *n* sites onto a rectangular grid with `cols = ceil(sqrt(n))`.
    Only horizontal and vertical edges (no diagonals — that's triangular).
    Coordination number z=4 for bulk, z=3 for edges, z=2 for corners.

    This is the canonical 2D lattice for studying QPTs. At 4×4 (N=16)
    or larger, it becomes challenging for classical tensor network methods.
    """
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    edges: list[tuple[int, int]] = []

    def idx(r: int, c: int) -> int | None:
        if r < 0 or c < 0 or r >= rows or c >= cols:
            return None
        s = r * cols + c
        return s if s < n else None

    for r in range(rows):
        for c in range(cols):
            s = idx(r, c)
            if s is None:
                continue
            # Right neighbour (same row, next column)
            right = idx(r, c + 1)
            if right is not None:
                edges.append((s, right))
            # Down neighbour (next row, same column)
            down = idx(r + 1, c)
            if down is not None:
                edges.append((s, down))
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


def generate_heavy_hex(n: int) -> list[tuple[int, int]]:
    """Heavy-hex lattice — native topology of IBM Eagle/Heron/Torino processors.

    The heavy-hex lattice is a degree-3 graph where hexagons are connected by
    "bridge" qubits. Coordination number z=2 for bridge qubits and z=3 for
    hexagon junction vertices.

    This is the native coupling map of IBM quantum processors (127-qubit Eagle,
    133-qubit Heron/Torino). Using this topology means the HVA circuit maps
    directly to hardware without SWAP routing overhead.

    Construction: We build a linear heavy-hex chain — a backbone of qubits
    connected linearly, with "bridge" qubits branching off at regular intervals
    (every 2nd backbone site). This matches the 1D slice of IBM's heavy-hex
    coupling map.

    For n qubits:
    - Backbone: sites 0, 1, 2, ..., k-1 connected linearly
    - Bridges: sites k, k+1, ... branching off backbone at every 2nd site
    - Result: max degree 3 (backbone sites with bridge), degree 2 (others), degree 1 (bridge tips)

    Parameters
    ----------
    n : int
        Number of qubits (must be >= 4).

    Returns
    -------
    list[tuple[int, int]]
        Edge list for the heavy-hex lattice.
    """
    if n < 4:
        raise ValueError("Heavy-hex lattice requires at least 4 sites.")

    edges: list[tuple[int, int]] = []

    # Strategy: allocate backbone sites first, then bridge sites
    # Bridges branch off every 2 backbone sites (at positions 1, 3, 5, ...)
    # Each bridge is a single qubit hanging off the backbone

    # Determine how many backbone vs bridge sites we can fit
    # For n sites: backbone_len + n_bridges = n
    # n_bridges = backbone_len // 2 (one bridge per 2 backbone sites)
    # So: backbone_len + backbone_len//2 = n → backbone_len ≈ 2n/3

    backbone_len = max(3, (2 * n + 2) // 3)  # At least 3 backbone sites
    n_bridges = n - backbone_len

    # Clamp: can't have more bridges than available branch points
    max_bridges = (backbone_len - 1) // 2
    if n_bridges > max_bridges:
        # Redistribute: extend backbone
        backbone_len = n - max_bridges + (n - max_bridges - 1) // 2
        backbone_len = min(backbone_len, n)
        n_bridges = n - backbone_len

    # Ensure we use exactly n sites
    if backbone_len + n_bridges > n:
        n_bridges = n - backbone_len
    if backbone_len + n_bridges < n:
        backbone_len = n - n_bridges

    # Backbone edges (linear chain)
    for i in range(backbone_len - 1):
        edges.append((i, i + 1))

    # Bridge edges: branch off backbone at odd-indexed sites (1, 3, 5, ...)
    bridge_idx = backbone_len  # First bridge site index
    for i in range(1, backbone_len, 2):
        if bridge_idx >= n:
            break
        edges.append((i, bridge_idx))
        bridge_idx += 1

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
    logger.debug("make_lattice: topology=%s, n_qubits=%d, J=%s, h=%s", topology, n_qubits, J, h)
    if not isinstance(n_qubits, int | np.integer) or n_qubits < 2:
        raise ValueError(f"n_qubits must be an integer >= 2, got {n_qubits}")

    if topology == "chain_1d":
        edges = generate_chain_1d(n_qubits, periodic)
    elif topology == "ladder":
        edges = generate_ladder(n_qubits, periodic)
    elif topology == "square":
        edges = generate_square(n_qubits, periodic)
    elif topology == "triangular":
        edges = generate_triangular(n_qubits)
    elif topology == "kagome":
        edges = generate_kagome(n_qubits)
    elif topology == "heavy_hex":
        edges = generate_heavy_hex(n_qubits)
    else:
        raise ValueError(f"Unknown topology '{topology}'. Supported: {SUPPORTED_TOPOLOGIES}")
    # Physics guard: detect self-loops and duplicate edges (unphysical).
    # Self-loops are always invalid; duplicates silently double interaction strength.
    edge_set: set[tuple[int, int]] = set()
    for i, j in edges:
        if i == j:
            raise ValueError(
                f"Self-loop detected at site {i} in topology '{topology}'. "
                f"Self-interactions are unphysical for spin Hamiltonians."
            )
        key = (min(i, j), max(i, j))
        if key in edge_set:
            logger.warning(
                "make_lattice: duplicate edge (%d, %d) in topology '%s'. "
                "This doubles the ZZ interaction on that bond.",
                i,
                j,
                topology,
            )
        edge_set.add(key)

    # Disconnected site detection: every qubit should participate in ≥1 edge
    sites_in_edges = {s for edge in edges for s in edge}
    isolated = set(range(n_qubits)) - sites_in_edges
    if isolated:
        logger.warning(
            "make_lattice: isolated sites %s in topology '%s' (N=%d). "
            "These qubits have no ZZ interactions and act as free spins.",
            sorted(isolated),
            topology,
            n_qubits,
        )

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
        logger.debug(
            "HamiltonianBuilder.build: topology=%s, n=%d, edges=%d, h=%s",
            lattice.topology,
            n,
            len(lattice.edges),
            lattice.h,
        )
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
        logger.debug("build_local_observables: n=%d, n_bonds=%d", n, len(lattice.edges))
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
        logger.debug("build_graph_data: n=%d, edges=%d", lattice.n_qubits, len(lattice.edges))
        src = [i for i, j in lattice.edges]
        dst = [j for i, j in lattice.edges]
        # Make symmetric: add reverse edges
        edge_index = np.array([src + dst, dst + src], dtype=np.int64)
        return edge_index, lattice.coordination_numbers.copy()

    # ── TFIM + Longitudinal Field ────────────────────────────────────

    def build_tfim_longitudinal(self, lattice: LatticeConfig, g: float = 0.0) -> SparsePauliOp:
        """Build H = -J Σ_{(i,j)} Z_i Z_j  -  h Σ_i X_i  -  g Σ_i Z_i.

        Extension of the standard TFIM with a longitudinal field g in Z.
        At g=0 this reduces to the standard TFIM. The longitudinal field
        breaks the Z₂ symmetry and shifts the critical point.

        Parameters
        ----------
        lattice : LatticeConfig
            Lattice specification (topology, edges, couplings, field).
        g : float
            Longitudinal field strength. g=0 recovers standard TFIM.

        Returns
        -------
        SparsePauliOp
            The TFIM + longitudinal field Hamiltonian.
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

        # Longitudinal field Z terms on all sites
        if abs(g) > 1e-15:
            for site in range(n):
                terms.append(("Z", [site], -g))

        H = SparsePauliOp.from_sparse_list(terms, num_qubits=n)
        self.validate(H, n)
        return H

    # ── TFIM + Longitudinal observables ──────────────────────────────

    def build_tfim_longitudinal_observables(
        self, lattice: LatticeConfig
    ) -> tuple[list[SparsePauliOp], list[SparsePauliOp], list[SparsePauliOp]]:
        """Return per-site X, per-site Z, and per-bond ZZ operators.

        Parameters
        ----------
        lattice : LatticeConfig

        Returns
        -------
        (ops_X, ops_Z, ops_ZZ)
            ops_X : list of SparsePauliOp, one X_i per site.
            ops_Z : list of SparsePauliOp, one Z_i per site (longitudinal mag).
            ops_ZZ : list of SparsePauliOp, one ZZ_{ij} per bond.
        """
        n = lattice.n_qubits
        ops_x = [SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=n) for i in range(n)]
        ops_z = [SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], num_qubits=n) for i in range(n)]
        ops_zz = [
            SparsePauliOp.from_sparse_list([("ZZ", [i, j], 1.0)], num_qubits=n)
            for i, j in lattice.edges
        ]
        return ops_x, ops_z, ops_zz

    # ── Frustrated TFIM (J1-J2) ──────────────────────────────────────

    def build_frustrated_tfim(self, lattice: LatticeConfig, J2: float = 0.0) -> SparsePauliOp:
        """Build H = -J₁ Σ_{nn} Z_iZ_j + J₂ Σ_{nnn} Z_iZ_j - h Σ_i X_i.

        Frustrated TFIM with next-nearest-neighbor (NNN) antiferromagnetic
        coupling. At J₂=0 this reduces to the standard TFIM. Frustration
        arises from competing ferromagnetic NN and antiferromagnetic NNN
        interactions.

        Parameters
        ----------
        lattice : LatticeConfig
            Lattice specification (topology, edges, couplings, field).
            The edges define the NN bonds; NNN bonds are computed automatically
            for chain topologies.
        J2 : float
            Next-nearest-neighbor coupling. J₂>0 is antiferromagnetic (frustrating).
            J₂=0 recovers standard TFIM.

        Returns
        -------
        SparsePauliOp
            The frustrated TFIM Hamiltonian.

        Notes
        -----
        Hardware viability: NNN bonds add CX gates (N=6 p=1: 27 CZ vs 10 for
        standard TFIM). ZNE viable only at N=4. Intended for noiseless simulation
        to demonstrate pipeline extensibility with frustration physics.
        """
        n = lattice.n_qubits
        terms: list[tuple[str, list[int], complex]] = []

        # NN: ferromagnetic (-J₁ ZZ) on lattice edges
        for bond_idx, (i, j) in enumerate(lattice.edges):
            j_val = lattice.J[bond_idx] if isinstance(lattice.J, np.ndarray) else lattice.J
            terms.append(("ZZ", [i, j], -j_val))

        # NNN: antiferromagnetic (+J₂ ZZ) — generated from topology
        if abs(J2) > 1e-15:
            nnn_edges = self._generate_nnn_edges(lattice)
            for i, j in nnn_edges:
                terms.append(("ZZ", [i, j], J2))

        # Transverse field X terms on all sites
        for site in range(n):
            h_val = lattice.h[site] if isinstance(lattice.h, np.ndarray) else lattice.h
            terms.append(("X", [site], -h_val))

        H = SparsePauliOp.from_sparse_list(terms, num_qubits=n)
        self.validate(H, n)
        return H

    @staticmethod
    def _generate_nnn_edges(lattice: LatticeConfig) -> list[tuple[int, int]]:
        """Generate next-nearest-neighbor edges from the lattice topology.

        For chain_1d: (i, i+2) for all valid i.
        For other topologies: 2-hop paths in the bond graph (nodes connected
        by exactly 2 bonds but NOT directly connected).

        Returns
        -------
        list[tuple[int, int]]
            Sorted list of NNN edges as (min_idx, max_idx) tuples.
            All indices are guaranteed to be in [0, n_qubits).
        """
        n = lattice.n_qubits
        if lattice.topology == "chain_1d":
            return [(i, i + 2) for i in range(n - 2)]

        # Generic: find all pairs at graph distance exactly 2
        # Build adjacency from lattice edges
        adj: dict[int, set[int]] = {i: set() for i in range(n)}
        for i, j in lattice.edges:
            adj[i].add(j)
            adj[j].add(i)

        nnn: set[tuple[int, int]] = set()
        for node in range(n):
            for neighbor in adj[node]:
                for hop2 in adj[neighbor]:
                    # Skip self-loops and direct neighbors
                    if hop2 != node and hop2 not in adj[node]:
                        edge = (min(node, hop2), max(node, hop2))
                        nnn.add(edge)

        return sorted(nnn)

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

    # ── Heisenberg XXZ with transverse field ────────────────────────

    def build_heisenberg_transverse(
        self, lattice: LatticeConfig, delta: float = 0.5
    ) -> SparsePauliOp:
        """Build H = J Σ_{(i,j)} (X_iX_j + Y_iY_j + Δ·Z_iZ_j) - h Σ_i X_i.

        Heisenberg XXZ model with external field in the X direction (transverse).
        The transverse field breaks the U(1) symmetry of the XXZ model and creates
        a QPT between an antiferromagnetic phase (low h) and a paramagnetic phase
        (high h), analogous to the TFIM but with richer spin interactions.

        At Δ=0.5 (default), this is the anisotropic XXZ in a transverse field —
        a model that is genuinely distinct from TFIM while remaining accessible
        to shallow HVA circuits with |+⟩^N initial state.

        Parameters
        ----------
        lattice : LatticeConfig
            Lattice specification (topology, edges, couplings, field).
        delta : float
            ZZ anisotropy. Δ=1 is isotropic XXX, Δ=0.5 is anisotropic (default),
            Δ=0 reduces to XY model in transverse field.

        Returns
        -------
        SparsePauliOp
            The Heisenberg XXZ transverse-field Hamiltonian.
        """
        print("INFO: building heisenberg transverse on hamiltonian.py")
        n = lattice.n_qubits
        terms: list[tuple[str, list[int], complex]] = []

        # Exchange interaction: J(XX + YY + Δ·ZZ) on lattice edges
        for bond_idx, (i, j) in enumerate(lattice.edges):
            j_val = lattice.J[bond_idx] if isinstance(lattice.J, np.ndarray) else lattice.J
            terms.append(("XX", [i, j], j_val))
            terms.append(("YY", [i, j], j_val))
            terms.append(("ZZ", [i, j], j_val * delta))

        # Transverse field in X direction on all sites
        for site in range(n):
            h_val = lattice.h[site] if isinstance(lattice.h, np.ndarray) else lattice.h
            terms.append(("X", [site], -h_val))

        H = SparsePauliOp.from_sparse_list(terms, num_qubits=n)
        self.validate(H, n)
        return H

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
