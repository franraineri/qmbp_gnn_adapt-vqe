"""Unit tests for the lattice symmetry analysis module (analysis.symmetry).

Deterministic and light: uses small hand-built edge sets with KNOWN symmetry
groups (a 2x2 plaquette, a square ring) rather than the heavy make_lattice +
eigsh path, so the orbit/tie contracts are verified without any circuit build.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from qmbp_simulation.circuits.symmetry import (
    LatticeSymmetryGroup,
    SymmetryReport,
    analyze_symmetry,
    apply_permutation_to_state,
    bond_orbits,
    compress_seed_to_orbits,
    detect_symmetry_group,
    generate_subgroups,
    ground_state_invariance,
    orbit_index_map,
    recommend_symmetry_strategy,
    site_index_map,
    site_orbits,
    symmetrize_seed,
)

# A 2x2 plaquette (4-cycle): qubits 0-1-3-2-0. nn = the ring edges; nnn = the two
# diagonals (graph distance 2). Its symmetry group is D4 (order 8).
RING_NN = [(0, 1), (1, 3), (3, 2), (2, 0)]
RING_NNN = [(0, 3), (1, 2)]


class TestDetectGroup:
    def test_ring_has_nontrivial_symmetries(self):
        g = detect_symmetry_group(RING_NN, RING_NNN, 4)
        assert isinstance(g, LatticeSymmetryGroup)
        assert g.order > 1  # a 4-cycle is highly symmetric
        assert not g.is_trivial
        # every perm is a valid permutation of 0..3
        for p in g.perms:
            assert sorted(p) == [0, 1, 2, 3]

    def test_identity_excluded(self):
        g = detect_symmetry_group(RING_NN, RING_NNN, 4)
        identity = [0, 1, 2, 3]
        assert identity not in g.perms

    def test_asymmetric_graph_is_trivial(self):
        # A rigid tree (branches of distinct lengths at distinct nodes) has no
        # non-trivial automorphism → the symmetry group is just the identity.
        nn = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (3, 6)]
        g = detect_symmetry_group(nn, [], 7)
        assert g.order == 1
        assert g.is_trivial


class TestOrbits:
    def test_ring_nn_single_orbit(self):
        # All 4 ring edges are equivalent under D4 → one orbit.
        g = detect_symmetry_group(RING_NN, RING_NNN, 4)
        orb = bond_orbits(RING_NN, g)
        assert len(orb) == 1
        assert sum(len(o) for o in orb) == 4

    def test_ring_nnn_single_orbit(self):
        g = detect_symmetry_group(RING_NN, RING_NNN, 4)
        orb = bond_orbits(RING_NNN, g)
        assert len(orb) == 1  # both diagonals equivalent
        assert sum(len(o) for o in orb) == 2

    def test_site_orbit_all_equivalent(self):
        g = detect_symmetry_group(RING_NN, RING_NNN, 4)
        orb = site_orbits(4, g)
        assert len(orb) == 1  # all 4 corners equivalent under D4

    def test_trivial_group_gives_singletons(self):
        g = LatticeSymmetryGroup(n_qubits=4, perms=[], names=[])
        orb = bond_orbits(RING_NN, g)
        assert len(orb) == 4  # no tying → one orbit per edge
        s = site_orbits(4, g)
        assert len(s) == 4


class TestOrbitIndexMap:
    def test_contract_aligned_and_contiguous(self):
        g = detect_symmetry_group(RING_NN, RING_NNN, 4)
        idx = orbit_index_map(RING_NN, g)
        assert len(idx) == len(RING_NN)
        # all ring edges in one orbit → all same id
        assert len(set(idx)) == 1
        # ids are contiguous from 0
        assert set(idx) == set(range(len(set(idx))))

    def test_distinct_orbits_distinct_ids(self):
        # nn (ring) and nnn (diag) are different orbits; within each, one id.
        g = detect_symmetry_group(RING_NN, RING_NNN, 4)
        nn_idx = orbit_index_map(RING_NN, g)
        nnn_idx = orbit_index_map(RING_NNN, g)
        assert len(set(nn_idx)) == 1
        assert len(set(nnn_idx)) == 1

    def test_site_index_map_contract(self):
        g = detect_symmetry_group(RING_NN, RING_NNN, 4)
        sidx = site_index_map(4, g)
        assert len(sidx) == 4
        assert len(set(sidx)) == 1  # all sites equivalent

    def test_trivial_group_identity_map(self):
        g = LatticeSymmetryGroup(n_qubits=4, perms=[], names=[])
        idx = orbit_index_map(RING_NN, g)
        assert idx == [0, 1, 2, 3]  # each edge its own orbit, in order


class TestStateInvariance:
    def test_permutation_preserves_norm(self):
        rng = np.random.default_rng(0)
        psi = rng.normal(size=16) + 1j * rng.normal(size=16)
        psi /= np.linalg.norm(psi)
        perm = [1, 0, 3, 2]  # swap qubit 0<->1, 2<->3
        out = apply_permutation_to_state(psi, perm, 4)
        np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-12)

    def test_identity_permutation_is_noop(self):
        rng = np.random.default_rng(1)
        psi = rng.normal(size=16).astype(complex)
        out = apply_permutation_to_state(psi, [0, 1, 2, 3], 4)
        np.testing.assert_allclose(out, psi, atol=1e-12)

    def test_symmetric_state_is_invariant(self):
        # The uniform |+>^N state is invariant under ANY qubit permutation.
        n = 4
        psi = np.ones(1 << n, dtype=complex) / np.sqrt(1 << n)
        g = detect_symmetry_group(RING_NN, RING_NNN, n)
        inv = ground_state_invariance(psi, g)
        for v in inv.values():
            np.testing.assert_allclose(v, 1.0, atol=1e-10)


class TestAnalyzeSymmetryReport:
    def test_report_fields_and_reduction(self):
        g, rep = analyze_symmetry(RING_NN, RING_NNN, 4, topology="square", h=0.5, j2=0.5)
        assert rep.group_order == g.order
        assert rep.nn_orbits == 1 and rep.nnn_orbits == 1 and rep.site_orbits == 1
        # 4 nn + 2 nnn + 4 sites = 10 independent → 3 tied → ×3.33
        assert rep.params_per_layer_independent == 10
        assert rep.params_per_layer_tied == 3
        assert rep.reduction_factor == pytest.approx(10 / 3)

    def test_report_json_roundtrip(self):
        _g, rep = analyze_symmetry(RING_NN, RING_NNN, 4)
        rep2 = SymmetryReport.from_dict(json.loads(json.dumps(rep.to_dict())))
        assert rep2.reduction_factor == pytest.approx(rep.reduction_factor)
        assert rep2.nn_orbit_sizes == rep.nn_orbit_sizes
        assert rep2.group_order == rep.group_order

    def test_gs_invariance_included_when_state_given(self):
        n = 4
        psi = np.ones(1 << n, dtype=complex) / np.sqrt(1 << n)
        _g, rep = analyze_symmetry(RING_NN, RING_NNN, n, ground_state=psi, manifold_dim=1)
        assert rep.gs_invariance is not None
        assert rep.manifold_dim == 1
        for v in rep.gs_invariance.values():
            assert v == pytest.approx(1.0, abs=1e-10)


class TestCompressSeedToOrbits:
    """compress_seed_to_orbits — an independent per-bond warm-start averaged to
    the tied (one-per-orbit) layout, slot-aligned with the tied builder."""

    def test_averages_each_orbit_per_block(self):
        # D4 plaquette: nn=1 orbit (4 edges), nnn=1 orbit (2), site=1 orbit (4).
        g = detect_symmetry_group(RING_NN, RING_NNN, 4)
        # blocks [nn, nnn, x] → independent seed length 4+2+4 = 10.
        seed = np.array(
            [
                0.2,
                0.4,
                0.6,
                0.8,  # nn block (mean 0.5)
                1.0,
                2.0,  # nnn block (mean 1.5)
                0.0,
                0.1,
                0.2,
                0.3,  # x block (mean 0.15)
            ]
        )
        tied = compress_seed_to_orbits(seed, ["nn", "nnn", "x"], RING_NN, RING_NNN, 4, g)
        # one value per orbit per block → 3 values, each the orbit mean
        np.testing.assert_allclose(tied, [0.5, 1.5, 0.15])

    def test_rx_final_adds_one_site_orbit_value(self):
        g = detect_symmetry_group(RING_NN, RING_NNN, 4)
        # [nn, nnn, x] + rx_final(x) → independent length 4+2+4+4 = 14.
        seed = np.concatenate(
            [
                np.full(4, 0.5),  # nn
                np.full(2, 1.5),  # nnn
                np.full(4, 0.2),  # x
                np.full(4, 0.9),  # rx_final (x over sites)
            ]
        )
        tied = compress_seed_to_orbits(seed, ["nn", "nnn", "x"], RING_NN, RING_NNN, 4, g, rx_final=True)
        # 3 base orbit params + 1 extra site-orbit param from rx_final
        assert len(tied) == 4
        np.testing.assert_allclose(tied, [0.5, 1.5, 0.2, 0.9])

    def test_length_mismatch_raises_clear_error(self):
        # A seed that does not match the block layout raises a clear ValueError
        # (not a cryptic IndexError) — regression for the N=20 tied run crash.
        g = detect_symmetry_group(RING_NN, RING_NNN, 4)
        with pytest.raises(ValueError):
            compress_seed_to_orbits(np.zeros(3), ["nn", "nnn", "x"], RING_NN, RING_NNN, 4, g)


class TestGenerateSubgroups:
    """generate_subgroups — the closed subgroup lattice for a symmetry sweep."""

    def test_ring_d4_subgroup_orders(self):
        # The 2x2 plaquette has the full D4 (order 8); its subgroup orders are
        # the divisors realized by closed subsets: 1, 2, 4, 8.
        g = detect_symmetry_group(RING_NN, RING_NNN, 4)
        assert g.order == 8
        subs = generate_subgroups(g)
        orders = sorted({s.order for s in subs})
        assert orders == [1, 2, 4, 8]
        # the trivial subgroup (order 1) and the full group (order 8) are present
        assert any(s.order == 1 and s.is_trivial for s in subs)
        assert any(s.order == 8 for s in subs)
        # every subgroup is closed: composing any two members stays in the set
        for s in subs:
            elems = {tuple(range(4))} | {tuple(p) for p in s.perms}
            for a in elems:
                for b in elems:
                    c = tuple(a[b[i]] for i in range(4))
                    assert c in elems

    def test_trivial_group_only_yields_itself(self):
        g = LatticeSymmetryGroup(n_qubits=4, perms=[], names=[])
        subs = generate_subgroups(g)
        assert len(subs) == 1
        assert subs[0].order == 1


class TestSymmetrizeSeed:
    """symmetrize_seed — project a per-bond seed onto the symmetric subspace
    (orbit-average) while KEEPING full per-bond width (no tying)."""

    def test_keeps_length_and_averages_within_orbit(self):
        # D4 plaquette: nn=1 orbit(4), nnn=1 orbit(2), site=1 orbit(4).
        g = detect_symmetry_group(RING_NN, RING_NNN, 4)
        seed = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 0.0, 0.1, 0.2, 0.3])
        out = symmetrize_seed(seed, ["nn", "nnn", "x"], RING_NN, RING_NNN, 4, g)
        # same length as input (projection, NOT compression)
        assert out.size == seed.size == 10
        # every bond in an orbit takes the orbit mean
        np.testing.assert_allclose(out[:4], 0.5)  # nn mean
        np.testing.assert_allclose(out[4:6], 1.5)  # nnn mean
        np.testing.assert_allclose(out[6:10], 0.15)  # x mean

    def test_idempotent_and_trivial_group_noop(self):
        g = detect_symmetry_group(RING_NN, RING_NNN, 4)
        seed = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 0.0, 0.1, 0.2, 0.3])
        once = symmetrize_seed(seed, ["nn", "nnn", "x"], RING_NN, RING_NNN, 4, g)
        twice = symmetrize_seed(once, ["nn", "nnn", "x"], RING_NN, RING_NNN, 4, g)
        np.testing.assert_allclose(once, twice)  # idempotent
        # trivial group: every orbit is a singleton → seed unchanged
        trivial = LatticeSymmetryGroup(n_qubits=4, perms=[], names=[])
        noop = symmetrize_seed(seed, ["nn", "nnn", "x"], RING_NN, RING_NNN, 4, trivial)
        np.testing.assert_allclose(noop, seed)


class TestRecommendSymmetryStrategy:
    """recommend_symmetry_strategy — the empirical tied-vs-loose rule by phase."""

    def test_ordered_phase_recommends_orbit_tied(self):
        # Non-trivial group + low h (ordered/frustrated) → orbit_tied, with a
        # real parameter reduction computed from the orbit counts.
        g = detect_symmetry_group(RING_NN, RING_NNN, 4)  # D4, order 8
        r = recommend_symmetry_strategy(g, 0.3, nn_edges=RING_NN, nnn_edges=RING_NNN)
        assert r.strategy == "orbit_tied"
        assert r.expected_param_reduction > 1.0
        assert r.group_order == 8

    def test_paramagnet_recommends_loose_and_trivial_always_loose(self):
        g = detect_symmetry_group(RING_NN, RING_NNN, 4)
        assert recommend_symmetry_strategy(g, 1.5, nn_edges=RING_NN, nnn_edges=RING_NNN).strategy == "loose"
        # a trivial group always falls back to loose regardless of h
        triv = LatticeSymmetryGroup(n_qubits=4, perms=[], names=[])
        rt = recommend_symmetry_strategy(triv, 0.3)
        assert rt.strategy == "loose"
        assert rt.expected_param_reduction == 1.0
