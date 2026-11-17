"""Unit tests for tilted square clusters with PBC (models.tilted_square)."""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.circuits.symmetry import bond_orbits, detect_symmetry_group, site_orbits
from qmbp_simulation.models.tilted_square import (
    MAGIC_TILTS,
    generate_square_tilted,
    tilted_for_n,
)


def _degrees(edges, n):
    d = [0] * n
    for i, j in edges:
        d[i] += 1
        d[j] += 1
    return d


class TestGeometry:
    @pytest.mark.parametrize("n,tilt", sorted(MAGIC_TILTS.items()))
    def test_magic_table_is_consistent(self, n, tilt):
        l1, l2 = tilt
        assert l1 * l1 + l2 * l2 == n

    @pytest.mark.parametrize("n,tilt", [(n, t) for n, t in sorted(MAGIC_TILTS.items()) if n <= 20])
    def test_site_count_matches_tilt(self, n, tilt):
        assert generate_square_tilted(*tilt).n_qubits == n

    @pytest.mark.parametrize("n", [9, 10, 16, 18, 20])
    def test_uniform_coordination_z4(self, n):
        cl = tilted_for_n(n)
        assert set(_degrees(cl.nn_edges, n)) == {4}
        assert set(_degrees(cl.nnn_edges, n)) == {4}

    @pytest.mark.parametrize("n", [9, 16, 18])
    def test_no_self_loops_or_duplicates(self, n):
        cl = tilted_for_n(n)
        for edges in (cl.nn_edges, cl.nnn_edges):
            assert all(i != j for i, j in edges)
            assert len(edges) == len(set(edges))
        assert not (set(cl.nn_edges) & set(cl.nnn_edges))

    def test_n8_collapses_nnn_degree(self):
        # Known caveat: the 2x2 cell is too small, the two diagonals wrap onto the
        # same pairs and the nnn degree drops to 2. N=8 is not a study target.
        cl = tilted_for_n(8)
        assert set(_degrees(cl.nn_edges, 8)) == {4}
        assert set(_degrees(cl.nnn_edges, 8)) == {2}

    def test_invalid_tilt_raises(self):
        with pytest.raises(ValueError):
            generate_square_tilted(0, 0)
        with pytest.raises(ValueError):
            generate_square_tilted(-1, 2)

    def test_non_magic_n_raises(self):
        with pytest.raises(ValueError, match="not a tilted magic number"):
            tilted_for_n(12)


class TestSymmetry:
    @pytest.mark.parametrize("n,order", [(9, 72), (16, 128), (18, 144)])
    def test_full_group_order(self, n, order):
        cl = tilted_for_n(n)
        # The default max_syms=16 cap truncates the order to a spurious 17.
        g = detect_symmetry_group(cl.nn_edges, cl.nnn_edges, n, max_syms=100000)
        assert g.order == order

    def test_default_cap_truncates_order(self):
        cl = tilted_for_n(16)
        g = detect_symmetry_group(cl.nn_edges, cl.nnn_edges, 16)
        assert g.order == 17  # 16 non-identity perms + identity: a cap artifact

    @pytest.mark.parametrize("n", [9, 16])
    def test_single_orbit_per_block(self, n):
        cl = tilted_for_n(n)
        g = detect_symmetry_group(cl.nn_edges, cl.nnn_edges, n, max_syms=100000)
        assert len(bond_orbits(cl.nn_edges, g)) == 1
        assert len(bond_orbits(cl.nnn_edges, g)) == 1
        assert len(site_orbits(n, g)) == 1

    def test_group_preserves_both_graphs(self):
        cl = tilted_for_n(9)
        g = detect_symmetry_group(cl.nn_edges, cl.nnn_edges, 9, max_syms=100000)
        nn = {tuple(sorted(e)) for e in cl.nn_edges}
        nnn = {tuple(sorted(e)) for e in cl.nnn_edges}
        for perm in g.perms:
            assert {tuple(sorted((perm[i], perm[j]))) for i, j in cl.nn_edges} == nn
            assert {tuple(sorted((perm[i], perm[j]))) for i, j in cl.nnn_edges} == nnn


class TestHamiltonianSymmetry:
    def test_ground_state_is_group_invariant_n9(self):
        # Phase-0 finding: the tilted N=9 ground state is fully symmetric, which is
        # what makes the orbit-tied ansatz structurally able to reach it.
        from qiskit.quantum_info import SparsePauliOp
        from scipy.sparse.linalg import eigsh

        from qmbp_simulation.circuits.symmetry import apply_permutation_to_state

        cl = tilted_for_n(9)
        terms = [("ZZ", [i, j], -1.0) for i, j in cl.nn_edges]
        terms += [("ZZ", [i, j], 0.5) for i, j in cl.nnn_edges]
        terms += [("X", [s], -0.5) for s in range(9)]
        H = SparsePauliOp.from_sparse_list(terms, num_qubits=9)
        ev, evec = eigsh(H.to_matrix(sparse=True), k=2, which="SA")
        order = np.argsort(ev)
        assert ev[order[1]] - ev[order[0]] > 1e-3  # non-degenerate
        gs = evec[:, order[0]].astype(complex)
        g = detect_symmetry_group(cl.nn_edges, cl.nnn_edges, 9, max_syms=100000)
        for perm in g.perms[:20]:
            ov = abs(np.vdot(gs, apply_permutation_to_state(gs, perm, 9))) ** 2
            assert ov == pytest.approx(1.0, abs=1e-8)
