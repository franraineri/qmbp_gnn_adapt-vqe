"""Tests for the reusable bond-selection primitive and the masked HVA builder.

Covers the shared contract behind T1 (per-bond pruning), T2 (highest-weight
subset), and future ADAPT-VQE growth:

- ``BondSelection`` bookkeeping (bond/2q counts, normalization, emptiness).
- ``prune_by_theta`` keeps only above-tolerance bonds; drops near-zero / NaN.
- ``top_k_by_weight`` keeps the strongest k bonds, stable, original order.
- ``rank_by_gradient`` orders bonds by |grad| descending (ADAPT growth signal).
- ``create_bond_resolved_masked`` equals the configurable builder for a full
  selection (strict generalization) and yields fewer 2q gates for a subset.

Pure ``src/`` modules; the equivalence test builds tiny circuits only.
"""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.circuits.bond_mask import (
    BondSelection,
    full_selection,
    prune_by_theta,
    rank_by_gradient,
    top_k_by_weight,
)


class TestBondSelection:
    def test_counts_and_2q(self):
        sel = BondSelection(nn_edges=[(0, 1), (1, 2)], nnn_edges=[(0, 2)])
        assert sel.n_bonds == 3
        assert sel.n_2q == 6  # two CX per RZZ
        assert not sel.is_empty()

    def test_empty_selection(self):
        sel = BondSelection()
        assert sel.is_empty()
        assert sel.n_2q == 0

    def test_edges_normalized_to_int_tuples(self):
        sel = BondSelection(nn_edges=[[0, 1]], nnn_edges=[(np.int64(1), np.int64(3))])
        assert sel.nn_edges == [(0, 1)]
        assert sel.nnn_edges == [(1, 3)]
        assert all(isinstance(i, int) and isinstance(j, int) for i, j in sel.nn_edges)


class TestPruneByTheta:
    def test_keeps_above_tolerance(self):
        nn = [(0, 1), (1, 2), (2, 3)]
        nnn = [(0, 2), (1, 3)]
        sel = prune_by_theta(nn, nnn, np.array([0.5, 0.01, 0.3]),
                             np.array([0.2, 0.001]), tol=0.05)
        assert sel.nn_edges == [(0, 1), (2, 3)]
        assert sel.nnn_edges == [(0, 2)]
        assert "prune_by_theta" in sel.provenance

    def test_drops_non_finite_theta(self):
        nn = [(0, 1), (1, 2)]
        sel = prune_by_theta(nn, [], np.array([np.nan, 0.9]), np.array([]), tol=0.05)
        assert sel.nn_edges == [(1, 2)]

    def test_all_pruned_yields_empty(self):
        nn = [(0, 1), (1, 2)]
        sel = prune_by_theta(nn, [], np.array([0.001, 0.002]), np.array([]), tol=0.05)
        assert sel.is_empty()

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            prune_by_theta([(0, 1)], [], np.array([0.1, 0.2]), np.array([]))


class TestTopKByWeight:
    def test_keeps_strongest_k(self):
        nnn = [(0, 2), (1, 3), (2, 4)]
        sel = top_k_by_weight([], nnn, np.array([]), np.array([0.1, 0.9, 0.5]), k_nnn=2)
        # strongest two are (1,3)=0.9 and (2,4)=0.5 → kept in original edge order
        assert sel.nnn_edges == [(1, 3), (2, 4)]

    def test_k_none_keeps_all(self):
        nn = [(0, 1), (1, 2)]
        sel = top_k_by_weight(nn, [], np.array([0.1, 0.2]), np.array([]), k_nn=None)
        assert sel.nn_edges == nn

    def test_k_zero_keeps_none(self):
        nn = [(0, 1), (1, 2)]
        sel = top_k_by_weight(nn, [], np.array([0.1, 0.2]), np.array([]), k_nn=0)
        assert sel.nn_edges == []

    def test_non_finite_ranks_last(self):
        nn = [(0, 1), (1, 2), (2, 3)]
        sel = top_k_by_weight(nn, [], np.array([np.nan, 0.9, 0.1]), np.array([]), k_nn=2)
        # NaN ranks last → keep (1,2)=0.9 and (2,3)=0.1
        assert sel.nn_edges == [(1, 2), (2, 3)]

    def test_preserves_original_order_in_kept_set(self):
        nn = [(0, 1), (1, 2), (2, 3), (3, 4)]
        # weights pick indices 3 and 1 as strongest → output keeps edge order 1,3
        sel = top_k_by_weight(nn, [], np.array([0.1, 0.8, 0.2, 0.9]), np.array([]), k_nn=2)
        assert sel.nn_edges == [(1, 2), (3, 4)]


class TestRankByGradient:
    def test_orders_by_abs_grad_descending(self):
        nn = [(0, 1), (1, 2)]
        nnn = [(0, 2)]
        ranked = rank_by_gradient(nn, nnn, np.array([0.1, 0.9]), np.array([0.5]))
        assert ranked[0] == ("nn", (1, 2), 0.9)
        assert ranked[1] == ("nnn", (0, 2), 0.5)
        assert ranked[2] == ("nn", (0, 1), pytest.approx(0.1))

    def test_non_finite_ranks_last(self):
        nn = [(0, 1), (1, 2)]
        ranked = rank_by_gradient(nn, [], np.array([np.nan, 0.3]), np.array([]))
        assert ranked[0][1] == (1, 2)
        assert ranked[-1][1] == (0, 1)  # NaN → -1.0 score, last


class TestMaskedBuilderEquivalence:
    """The masked builder is a strict generalization of the configurable one."""

    @pytest.fixture
    def lattice_and_edges(self):
        from qmbp_simulation.models import make_lattice
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        lat = make_lattice("square", 10, J=1.0, h=0.5)
        nnn = HamiltonianBuilder._generate_nnn_edges(lat)
        return lat, lat.edges, nnn

    def _n2q(self, qc):
        from qmbp_simulation.analysis.circuit_visualizer import circuit_summary

        bound = qc.assign_parameters(np.full(qc.num_parameters, 0.37))
        return circuit_summary(bound)["n_2q_gates"]

    def test_full_selection_matches_configurable(self, lattice_and_edges):
        from qmbp_simulation.circuits import HVACircuitBuilder

        lat, nn, nnn = lattice_and_edges
        b = HVACircuitBuilder()
        sel = full_selection(nn, nnn)
        qc_masked, _ = b.create_bond_resolved_masked(
            10, lat, blocks=["nn", "nnn", "x"], bond_selection=sel)
        qc_conf, _ = b.create_bond_resolved_frustrated_configurable(
            10, lat, blocks=["nn", "nnn", "x"])
        assert qc_masked.num_parameters == qc_conf.num_parameters
        assert self._n2q(qc_masked) == self._n2q(qc_conf)

    def test_subset_reduces_2q(self, lattice_and_edges):
        from qmbp_simulation.circuits import HVACircuitBuilder

        lat, nn, nnn = lattice_and_edges
        b = HVACircuitBuilder()
        full = full_selection(nn, nnn)
        half = BondSelection(nn_edges=nn, nnn_edges=nnn[: len(nnn) // 2])
        qc_full, _ = b.create_bond_resolved_masked(
            10, lat, blocks=["nn", "nnn", "x"], bond_selection=full)
        qc_half, _ = b.create_bond_resolved_masked(
            10, lat, blocks=["nn", "nnn", "x"], bond_selection=half)
        assert self._n2q(qc_half) < self._n2q(qc_full)

    def test_invalid_edge_raises(self, lattice_and_edges):
        from qmbp_simulation.circuits import HVACircuitBuilder

        lat, _, _ = lattice_and_edges
        b = HVACircuitBuilder()
        bad = BondSelection(nn_edges=[(0, 99)], nnn_edges=[])  # 99 >= N=10
        with pytest.raises(ValueError):
            b.create_bond_resolved_masked(
                10, lat, blocks=["nn", "x"], bond_selection=bad)


class TestVariantRouting:
    """AnsatzVariant.bond_selection routes build_variant to the masked engine."""

    def test_registry_variant_unchanged(self):
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.circuits.hva_variants import VARIANTS, build_variant
        from qmbp_simulation.models import make_lattice

        lat = make_lattice("square", 10, J=1.0, h=0.5)
        qc, _ = build_variant(HVACircuitBuilder(), 10, lat, VARIANTS["p2_base"])
        # p2_base has no bond_selection → full configurable build (76 params)
        assert VARIANTS["p2_base"].bond_selection is None
        assert qc.num_parameters == 76

    def test_masked_variant_builds_via_mask(self):
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.circuits.hva_variants import build_variant, make_masked_variant
        from qmbp_simulation.models import make_lattice
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        lat = make_lattice("square", 10, J=1.0, h=0.5)
        nnn = HamiltonianBuilder._generate_nnn_edges(lat)
        sel = BondSelection(nn_edges=lat.edges, nnn_edges=nnn[: len(nnn) // 2],
                            provenance="half_nnn")
        v = make_masked_variant("p1_half_nnn_topk", "half nnn", ["nn", "nnn", "x"], sel)
        qc, _ = build_variant(HVACircuitBuilder(), 10, lat, v)
        assert v.bond_selection is not None
        assert "half_nnn" in v.description
        # fewer params than full nnn (subset of nnn bonds)
        n_full = len(lat.edges) + len(nnn) + 10
        assert qc.num_parameters < n_full


class TestAdaptGrowthLogic:
    """The bond-growth loop ADAPT uses: rank remaining bonds, add the top ones,
    grow the selection monotonically until full. Pure selection-level logic
    (no circuit build) — mirrors run_adapt_bonds' growth step."""

    def test_growth_adds_highest_gradient_bond(self):
        nnn_all = [(0, 2), (1, 3), (2, 4)]
        sel = BondSelection(nn_edges=[(0, 1)], nnn_edges=[])
        remaining = [e for e in nnn_all if e not in set(sel.nnn_edges)]
        # gradient magnitudes: (1,3) strongest
        ranked = rank_by_gradient([], remaining, np.array([]), np.array([0.1, 0.9, 0.4]))
        top = [edge for _k, edge, score in ranked if score > 1e-6][:1]
        grown = BondSelection(nn_edges=sel.nn_edges, nnn_edges=sel.nnn_edges + top)
        assert grown.nnn_edges == [(1, 3)]
        assert grown.n_bonds == sel.n_bonds + 1

    def test_growth_is_monotonic_until_full(self):
        nnn_all = [(0, 2), (1, 3), (2, 4)]
        sel = BondSelection(nn_edges=[(0, 1)], nnn_edges=[])
        sizes = [sel.n_bonds]
        for _ in range(10):  # bounded; must terminate at full
            remaining = [e for e in nnn_all if e not in set(sel.nnn_edges)]
            if not remaining:
                break
            ranked = rank_by_gradient([], remaining, np.array([]),
                                      np.abs(np.arange(len(remaining), dtype=float) + 1))
            top = [edge for _k, edge, s in ranked if s > 1e-6][:2]
            sel = BondSelection(nn_edges=sel.nn_edges, nnn_edges=sel.nnn_edges + top)
            sizes.append(sel.n_bonds)
        # strictly increasing, ends at the full nnn set (+ the nn backbone)
        assert sizes == sorted(sizes)
        assert len(set(sizes)) == len(sizes)
        assert sel.nnn_edges == nnn_all or set(sel.nnn_edges) == set(nnn_all)

    def test_growth_stops_when_all_gradients_below_tol(self):
        remaining = [(0, 2), (1, 3)]
        ranked = rank_by_gradient([], remaining, np.array([]), np.array([1e-9, 1e-9]))
        top = [edge for _k, edge, s in ranked if s > 1e-6]
        assert top == []  # nothing to add → ADAPT stops

    def test_full_growth_equals_full_selection(self):
        nnn_all = [(0, 2), (1, 3)]
        nn = [(0, 1)]
        grown = BondSelection(nn_edges=nn, nnn_edges=list(nnn_all))
        full = full_selection(nn, nnn_all)
        assert set(grown.nnn_edges) == set(full.nnn_edges)
        assert grown.n_2q == full.n_2q


class TestRegressionBondTopkRegimePipeline:
    """Four regression tests pinning the bugs hit while building the N=18
    regime-aware top-k validation, so they cannot recur.

    Bugs covered:
    (a) cold-start reference — the p2 full reference was converged from θ=0 /
        random at a tiny gap and collapsed (fid~0.46), poisoning the bond
        ranking. The seed MUST be the regime seed, length-matched to the circuit.
    (b) θ_x indices leaking into ZZ coordinates — exploration must perturb only
        the RX block, never the (renormalized, pinned) ZZ angles.
    (c) masked seed length mismatch — the regime seed must have exactly the
        masked circuit's parameter count (fewer nnn bonds than the full lattice).
    (d) regime-seed boundary selection — the h thresholds must map to the exact
        documented seed per phase.
    """

    @staticmethod
    def _lattice(n=10, h=0.5):
        from qmbp_simulation.models import make_lattice
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        lat = make_lattice("square", n, J=1.0, h=h)
        nnn = HamiltonianBuilder._generate_nnn_edges(lat)
        return lat, lat.edges, nnn

    def test_a_regime_seed_matches_full_reference_params(self):
        """(a) The full p2 reference's regime seed length == circuit params, so
        the reference is never cold-started from a mismatched/empty seed."""

        from qmbp_simulation.analysis.warmstart import select_regime_seed
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.circuits.hva_variants import AnsatzVariant, build_variant

        lat, nn, nnn = self._lattice()
        ref = AnsatzVariant("ref", "", blocks=["nn", "nnn", "x"] * 2)
        qc, _ = build_variant(HVACircuitBuilder(), 10, lat, ref)
        seed, name = select_regime_seed(len(nn), len(nnn), 10, 2, 0.5, J=1.0, J2=0.5)
        assert len(seed) == qc.num_parameters
        # at h=0.5 the ZZ angles must be non-zero (a real renormalized seed, not θ=0)
        assert name == "second_order"
        assert abs(seed[0]) > 1e-6

    def test_b_theta_x_indices_exclude_zz_coordinates(self):
        """(b) θ_x indices point only at RX coordinates: in a regime seed the ZZ
        entries are negative, the X entries positive — the index sets must not
        overlap and x-indices must land on the positive (X) block."""
        import numpy as np

        from qmbp_simulation.analysis.warmstart import select_regime_seed, theta_x_indices

        n_nn, n_nnn, n, p = 13, 15, 10, 2
        seed, _ = select_regime_seed(n_nn, n_nnn, n, p, 0.5, J=1.0, J2=0.5)
        x_idx = set(theta_x_indices(n_nn, n_nnn, n, p))
        per = n_nn + n_nnn + n
        zz_idx = set()
        for layer in range(p):
            o = layer * per
            zz_idx.update(range(o, o + n_nn + n_nnn))  # nn + nnn blocks
        assert x_idx.isdisjoint(zz_idx), "θ_x indices overlap ZZ coordinates"
        arr = np.asarray(seed)
        assert np.all(arr[list(x_idx)] > 0), "θ_x entries should be the positive RX angle"
        assert np.all(arr[list(zz_idx)] < 0), "ZZ entries should be negative (renorm)"

    def test_c_masked_regime_seed_length_matches_masked_circuit(self):
        """(c) For a masked top-k ansatz (fewer nnn bonds) the regime seed length
        must equal the masked circuit's parameter count, not the full one."""
        from qmbp_simulation.analysis.warmstart import select_regime_seed
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.circuits.bond_mask import BondSelection
        from qmbp_simulation.circuits.hva_variants import build_variant, make_masked_variant

        lat, nn, nnn = self._lattice()
        k = len(nnn) // 2
        sel = BondSelection(nn_edges=nn, nnn_edges=nnn[:k])
        v = make_masked_variant("tk", "", ["nn", "nnn", "x"] * 2, sel)
        qc, _ = build_variant(HVACircuitBuilder(), 10, lat, v)
        seed, _ = select_regime_seed(len(sel.nn_edges), len(sel.nnn_edges), 10, 2,
                                     0.5, J=1.0, J2=0.5)
        assert len(seed) == qc.num_parameters
        # and it must be SHORTER than the full-lattice seed (fewer nnn bonds)
        full_seed, _ = select_regime_seed(len(nn), len(nnn), 10, 2, 0.5, J2=0.5)
        assert len(seed) < len(full_seed)

    def test_d_regime_seed_boundaries_map_to_documented_phase(self):
        """(d) The h thresholds select the exact documented seed per phase."""
        from qmbp_simulation.analysis.warmstart import select_regime_seed

        names = {h: select_regime_seed(13, 15, 10, 2, h, J2=0.5)[1]
                 for h in (0.30, 0.45, 0.50, 1.00, 1.20, 1.50)}
        assert "flat_renorm" in names[0.30]
        assert "flat_renorm" in names[0.45]   # inclusive ordered upper edge
        assert names[0.50] == "second_order"
        assert names[1.00] == "second_order"
        assert "so_nn_shrink" in names[1.20]  # inclusive paramagnet lower edge
        assert "so_nn_shrink" in names[1.50]


class TestBondWeightsFromTheta:
    """Per-bond |θ| aggregation for the standard [nn,nnn,x]*p layout."""

    def test_max_abs_across_layers(self):
        from qmbp_simulation.circuits.bond_mask import bond_weights_from_theta

        # p=2, n_nn=2, n_nnn=1, n_qubits=2 → per layer = 5, total 10.
        theta = np.array([
            0.1, -0.9, 0.3, 7.0, 8.0,     # layer 0: nn=[0.1,-0.9] nnn=[0.3] x=..
            -0.4, 0.2, -0.8, 9.0, 9.0,    # layer 1: nn=[-0.4,0.2] nnn=[-0.8] x=..
        ])
        w_nn, w_nnn = bond_weights_from_theta(theta, 2, 1, 2, 2)
        np.testing.assert_allclose(w_nn, [0.4, 0.9])   # max|.| per nn bond
        np.testing.assert_allclose(w_nnn, [0.8])       # max|.| per nnn bond

    def test_ignores_x_block(self):
        from qmbp_simulation.circuits.bond_mask import bond_weights_from_theta

        theta = np.array([0.5, 0.2, 99.0, 99.0])  # n_nn=1 n_nnn=1 n_qubits=2, p=1
        w_nn, w_nnn = bond_weights_from_theta(theta, 1, 1, 2, 1)
        assert w_nn[0] == 0.5 and w_nnn[0] == 0.2


class TestBondWeightsForBlocks:
    """Per-bond |θ| for an arbitrary (asymmetric) block sequence."""

    def test_half_nn_rx_layout_offsets(self):
        """p2_half_nn_rx = [nn,nnn,x, nn,nnn,x, nn,x] + rx_final: nn appears 3×,
        nnn twice. The aggregator must slice each block at its real offset."""
        from qmbp_simulation.circuits.bond_mask import bond_weights_for_blocks

        n_nn, n_nnn, nq = 2, 1, 2
        blocks = ["nn", "nnn", "x", "nn", "nnn", "x", "nn", "x"]
        # Build θ with a known maximum per bond placed in a specific block.
        parts = [
            [0.1, 0.2],  # nn block 1
            [0.3],       # nnn block 1
            [0.0, 0.0],  # x
            [0.1, 0.9],  # nn block 2 → bond1 max here
            [0.7],       # nnn block 2 → max here
            [0.0, 0.0],  # x
            [0.8, 0.1],  # nn block 3 → bond0 max here
            [0.0, 0.0],  # x
            [0.0, 0.0],  # rx_final
        ]
        theta = np.concatenate([np.array(p) for p in parts])
        w_nn, w_nnn = bond_weights_for_blocks(
            theta, blocks, n_nn, n_nnn, nq, rx_final=True)
        np.testing.assert_allclose(w_nn, [0.8, 0.9])  # per-bond max across 3 nn blocks
        np.testing.assert_allclose(w_nnn, [0.7])      # per-bond max across 2 nnn blocks

    def test_length_mismatch_raises(self):
        from qmbp_simulation.circuits.bond_mask import bond_weights_for_blocks

        with pytest.raises(ValueError):
            bond_weights_for_blocks(np.zeros(3), ["nn", "nnn", "x"], 2, 1, 2)

    def test_matches_standard_aggregator_on_standard_layout(self):
        """On a plain [nn,nnn,x]*p sequence the block-walker equals the
        uniform-layer aggregator (strict generalization)."""
        from qmbp_simulation.circuits.bond_mask import (
            bond_weights_for_blocks,
            bond_weights_from_theta,
        )

        rng = np.random.default_rng(0)
        n_nn, n_nnn, nq, p = 3, 2, 4, 2
        theta = rng.uniform(-1, 1, (n_nn + n_nnn + nq) * p)
        blocks = ["nn", "nnn", "x"] * p
        a_nn, a_nnn = bond_weights_from_theta(theta, n_nn, n_nnn, nq, p)
        b_nn, b_nnn = bond_weights_for_blocks(theta, blocks, n_nn, n_nnn, nq)
        np.testing.assert_allclose(a_nn, b_nn)
        np.testing.assert_allclose(a_nnn, b_nnn)


class TestSelectionFromVariantTheta:
    """Bridge from a structure-variant θ to a BondSelection (T1/T2 on any layout)."""

    def _setup(self):
        # p2_half_nn_rx-like: nn appears 3×, nnn 2×, + rx_final.
        nn = [(0, 1), (1, 2), (2, 3)]
        nnn = [(0, 2), (1, 3)]
        nq = 4
        blocks = ["nn", "nnn", "x", "nn", "nnn", "x", "nn", "x"]
        parts = [
            [0.9, 0.1, 0.1],  # nn1
            [0.8, 0.05],      # nnn1 → bond0 strong
            [0.0] * nq,       # x
            [0.1, 0.9, 0.1],  # nn2
            [0.02, 0.7],      # nnn2 → bond1 strong
            [0.0] * nq,       # x
            [0.1, 0.1, 0.9],  # nn3
            [0.0] * nq,       # x
            [0.0] * nq,       # rx_final
        ]
        theta = np.concatenate([np.array(p) for p in parts])
        return theta, blocks, nn, nnn, nq

    def test_top_k_keeps_strongest_nnn(self):
        from qmbp_simulation.circuits.bond_mask import selection_from_variant_theta

        theta, blocks, nn, nnn, nq = self._setup()
        sel = selection_from_variant_theta(
            theta, blocks, nn, nnn, nq, rx_final=True, method="top_k", keep_frac=0.5)
        assert sel.nn_edges == nn            # all nn kept
        assert len(sel.nnn_edges) == 1       # round(0.5*2)=1 strongest nnn
        assert "top_k_by_weight" in sel.provenance

    def test_prune_drops_below_tol(self):
        from qmbp_simulation.circuits.bond_mask import selection_from_variant_theta

        theta, blocks, nn, nnn, nq = self._setup()
        sel = selection_from_variant_theta(
            theta, blocks, nn, nnn, nq, rx_final=True, method="prune", tol=0.5)
        # per-bond max|θ|: nn all reach 0.9 (kept); nnn = [0.8, 0.7] both > 0.5 (kept)
        assert len(sel.nn_edges) == 3
        assert len(sel.nnn_edges) == 2
        assert "prune_by_theta" in sel.provenance

    def test_unknown_method_raises(self):
        from qmbp_simulation.circuits.bond_mask import selection_from_variant_theta

        theta, blocks, nn, nnn, nq = self._setup()
        with pytest.raises(ValueError):
            selection_from_variant_theta(theta, blocks, nn, nnn, nq, rx_final=True,
                                         method="bogus")


class TestHalfNnRxFamily:
    """The half_nn_rx structure family (p1/p2/p3): p full layers + a half nn
    layer + trailing RX. The most expressive-per-2q structure in the study."""

    def test_family_follows_the_recipe(self):
        from qmbp_simulation.circuits.hva_variants import VARIANTS

        for p, name in [(1, "p1_half_nn_rx"), (2, "p2_half_nn_rx"),
                        (3, "p3_half_nn_rx")]:
            v = VARIANTS[name]
            # p full [nn,nnn,x] layers + a half [nn,x] layer, with free RX.
            assert v.blocks == ["nn", "nnn", "x"] * p + ["nn", "x"]
            assert v.rx_final is True

    def test_p3_half_nn_rx_builds_and_prunes(self):
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.circuits.bond_mask import selection_from_variant_theta
        from qmbp_simulation.circuits.hva_variants import (
            VARIANTS,
            build_variant,
            make_masked_variant,
        )
        from qmbp_simulation.models import make_lattice
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        lat = make_lattice("square", 10, J=1.0, h=0.5)
        nn, nnn = lat.edges, HamiltonianBuilder._generate_nnn_edges(lat)
        base = VARIANTS["p3_half_nn_rx"]
        qc_full, _ = build_variant(HVACircuitBuilder(), 10, lat, base)
        # a prune selection yields a strictly smaller circuit
        theta = np.full(qc_full.num_parameters, 0.4)
        sel = selection_from_variant_theta(theta, list(base.blocks), nn, nnn, 10,
                                           rx_final=True, method="prune", tol=0.2)
        v = make_masked_variant("p3_prune", "x", list(base.blocks), sel,
                                rx_final=True)
        qc_m, _ = build_variant(HVACircuitBuilder(), 10, lat, v)
        assert qc_m.num_parameters <= qc_full.num_parameters
