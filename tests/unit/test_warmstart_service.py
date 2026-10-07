"""Unit tests for the promoted warm-start core (qmbp_simulation.analysis.warmstart).

Tests the experiment-agnostic seed formulas, the calibrated second-order shrink,
the regime gate, and the multi-seed aggregation math. These live in ``src/`` per
the repo rule that tests must not import from ``scripts/``; this is the core that
the ``vl_vs_hva`` study service now re-exports.
"""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.analysis.warmstart import (
    DEFAULT_CURV_COEF,
    DEFAULT_SHRINK_COEF,
    SECOND_ORDER_H_MAX,
    SECOND_ORDER_H_MIN,
    aggregate_seed_stats,
    first_order_warmstart_theta,
    second_order_correction_magnitude,
    second_order_regime_gate,
    second_order_warmstart_theta,
)

# Layout used throughout: frustrated square N=9, nnn p=2 (matches the report).
N_NN, N_NNN, N_QUBITS, P = 12, 8, 9, 2
PER = N_NN + N_NNN + N_QUBITS
J, J2 = 1.0, 0.5


def _slices(layer: int = 0):
    o = layer * PER
    return (slice(o, o + N_NN), slice(o + N_NN, o + N_NN + N_NNN), slice(o + N_NN + N_NNN, o + PER))


class TestFirstOrderWarmStart:
    def test_length_and_layout(self):
        th = first_order_warmstart_theta(N_NN, N_NNN, N_QUBITS, P, 1.0, J=J, J2=J2)
        assert th.shape == (PER * P,)

    def test_formula_values_at_h_one(self):
        h = 1.0
        th = first_order_warmstart_theta(N_NN, N_NNN, N_QUBITS, P, h, J=J, J2=J2)
        nn, nnn, x = _slices(0)
        np.testing.assert_allclose(th[nn], -J / (4 * h))
        np.testing.assert_allclose(th[nnn], -J2 / (4 * h))
        np.testing.assert_allclose(th[x], np.arctan(J / h))

    def test_layers_are_identical(self):
        th = first_order_warmstart_theta(N_NN, N_NNN, N_QUBITS, P, 0.7, J=J, J2=J2)
        np.testing.assert_allclose(th[:PER], th[PER : 2 * PER])

    def test_limit_h_to_infinity_goes_to_zero(self):
        # theta_nn ~ -J/4h -> 0, theta_x = arctan(J/h) -> 0
        th = first_order_warmstart_theta(N_NN, N_NNN, N_QUBITS, P, 1e6, J=J, J2=J2)
        assert np.abs(th).max() < 1e-5

    def test_limit_h_to_zero_theta_x_saturates(self):
        # arctan(J/h) -> pi/2 as h -> 0+; the ZZ angles blow up (~1/h), so we
        # only assert the bounded transverse-field component saturates.
        h = 1e-4
        th = first_order_warmstart_theta(N_NN, N_NNN, N_QUBITS, P, h, J=J, J2=J2)
        _, _, x = _slices(0)
        np.testing.assert_allclose(th[x], np.pi / 2, atol=1e-3)

    def test_rejects_nonpositive_h(self):
        with pytest.raises(ValueError):
            first_order_warmstart_theta(N_NN, N_NNN, N_QUBITS, P, 0.0)


class TestSecondOrderWarmStart:
    def test_shrink_at_calibrated_coefficient(self):
        # At h=0.5, J=1, c=1/3: shrink = 1 - (1/(2*0.5))^2 * 1/3 = 1 - 1/3 = 2/3.
        h = 0.5
        th = second_order_warmstart_theta(
            N_NN,
            N_NNN,
            N_QUBITS,
            P,
            h,
            J=J,
            J2=J2,
            shrink_coef=DEFAULT_SHRINK_COEF,
            curv_coef=DEFAULT_CURV_COEF,
        )
        nn, nnn, x = _slices(0)
        shrink = 1.0 - (J / (2 * h)) ** 2 * DEFAULT_SHRINK_COEF
        np.testing.assert_allclose(shrink, 2.0 / 3.0)
        np.testing.assert_allclose(th[nn], -J / (4 * h) * shrink)
        np.testing.assert_allclose(th[nnn], -J2 / (4 * h) * shrink)
        curv = 1.0 - (J / (2 * h)) ** 2 * DEFAULT_CURV_COEF
        np.testing.assert_allclose(th[x], np.arctan(J / h) * curv)

    def test_reduces_to_first_order_when_coefs_zero(self):
        h = 0.5
        first = first_order_warmstart_theta(N_NN, N_NNN, N_QUBITS, P, h, J=J, J2=J2)
        second = second_order_warmstart_theta(N_NN, N_NNN, N_QUBITS, P, h, J=J, J2=J2, shrink_coef=0.0, curv_coef=0.0)
        np.testing.assert_allclose(second, first)

    def test_shrink_reduces_zz_magnitude_near_transition(self):
        h = 0.5
        first = first_order_warmstart_theta(N_NN, N_NNN, N_QUBITS, P, h, J=J, J2=J2)
        second = second_order_warmstart_theta(N_NN, N_NNN, N_QUBITS, P, h, J=J, J2=J2)
        nn, _, _ = _slices(0)
        # ZZ angles are negative; shrink pulls them toward zero => smaller |.|.
        assert np.abs(second[nn]).max() < np.abs(first[nn]).max()

    def test_correction_magnitude_grows_as_h_falls(self):
        assert (
            second_order_correction_magnitude(0.3)
            > second_order_correction_magnitude(0.5)
            > second_order_correction_magnitude(0.7)
        )


class TestRegimeGate:
    def test_inside_window(self):
        assert second_order_regime_gate(0.4) is True
        assert second_order_regime_gate(0.5) is True
        assert second_order_regime_gate(0.6) is True

    def test_below_window_disabled(self):
        # Deep ordered phase: second-order over-shrinks and hurts (report P3).
        assert second_order_regime_gate(0.3) is False

    def test_above_window_disabled(self):
        assert second_order_regime_gate(0.7) is False
        assert second_order_regime_gate(1.0) is False

    def test_default_window_matches_report(self):
        assert (SECOND_ORDER_H_MIN, SECOND_ORDER_H_MAX) == (0.4, 0.6)


class TestAggregation:
    def test_separation_sigma_and_helps_flag(self):
        stats = aggregate_seed_stats([0.80, 0.82, 0.81], [0.90, 0.91, 0.92])
        assert stats["n_seeds"] == 3
        np.testing.assert_allclose(stats["delta_mean"], 0.10, atol=1e-9)
        assert stats["separation_sigma"] > 1.0
        assert stats["second_order_helps"] is True

    def test_noise_level_difference_not_significant(self):
        # Overlapping distributions -> not significant.
        rng = np.random.default_rng(0)
        a = 0.81 + rng.normal(0, 0.05, 8)
        b = 0.82 + rng.normal(0, 0.05, 8)
        stats = aggregate_seed_stats(a, b)
        assert stats["second_order_helps"] is False

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            aggregate_seed_stats([], [0.9])


class TestVariantWarmStart:
    """variant_warmstart_theta builds block-wise seeds for arbitrary layouts."""

    def test_matches_standard_p1_for_base_blocks(self):
        from qmbp_simulation.analysis.warmstart import (
            second_order_warmstart_theta,
            variant_warmstart_theta,
        )

        n_nn, n_nnn, nq, h = 20, 20, 10, 0.5
        v = variant_warmstart_theta(["nn", "nnn", "x"], n_nn, n_nnn, nq, h, J=1.0, J2=0.5)
        s = second_order_warmstart_theta(n_nn, n_nnn, nq, 1, h, J=1.0, J2=0.5)
        np.testing.assert_allclose(v, s, atol=1e-12)

    def test_length_matches_block_sequence(self):
        from qmbp_simulation.analysis.warmstart import variant_warmstart_theta

        n_nn, n_nnn, nq, h = 12, 8, 9, 0.5
        blocks = ["nn", "nnn", "x", "nn", "x"]
        v = variant_warmstart_theta(blocks, n_nn, n_nnn, nq, h, rx_final=True, J2=0.5)
        expected = n_nn + n_nnn + nq + n_nn + nq + nq  # blocks + rx_final
        assert len(v) == expected

    def test_trailing_rotations_seeded_zero(self):
        from qmbp_simulation.analysis.warmstart import variant_warmstart_theta

        n_nn, n_nnn, nq, h = 4, 2, 4, 0.5
        v = variant_warmstart_theta(["nn"], n_nn, n_nnn, nq, h, rx_final=True, rz_final=True, J2=0.5)
        # last 2*nq entries (rx_final + rz_final) must be exactly zero
        np.testing.assert_allclose(v[-2 * nq :], 0.0, atol=1e-12)

    def test_unknown_block_raises(self):
        from qmbp_simulation.analysis.warmstart import variant_warmstart_theta

        with pytest.raises(ValueError):
            variant_warmstart_theta(["bogus"], 4, 2, 4, 0.5)

    def test_rejects_nonpositive_h(self):
        from qmbp_simulation.analysis.warmstart import variant_warmstart_theta

        with pytest.raises(ValueError):
            variant_warmstart_theta(["nn"], 4, 2, 4, 0.0)


class TestComposeExtendTheta:
    """compose_extend_theta seeds base prefix verbatim + near-identity tail."""

    def test_prefix_preserved_and_tail_zero(self):
        from qmbp_simulation.analysis.warmstart import compose_extend_theta

        base = np.arange(10, dtype=float)  # a fake p2 θ of length 10
        n_nn, n_nnn, nq = 4, 3, 5
        seed = compose_extend_theta(base, ["nn", "x"], n_nn, n_nnn, nq, 0.5, J2=0.5)
        # prefix identical to base
        np.testing.assert_allclose(seed[:10], base, atol=1e-12)
        # tail (nn + x = 4 + 5 = 9) all zeros → extra layer starts as identity
        assert len(seed) == 10 + n_nn + nq
        np.testing.assert_allclose(seed[10:], 0.0, atol=1e-12)

    def test_tail_length_counts_all_blocks(self):
        from qmbp_simulation.analysis.warmstart import compose_extend_theta

        base = np.zeros(5)
        n_nn, n_nnn, nq = 4, 3, 6
        seed = compose_extend_theta(base, ["nnn", "x"], n_nn, n_nnn, nq, 0.5, extra_rx_final=True, J2=0.5)
        # 5 base + nnn(3) + x(6) + rx_final(6)
        assert len(seed) == 5 + n_nnn + nq + nq

    def test_unknown_extra_block_raises(self):
        from qmbp_simulation.analysis.warmstart import compose_extend_theta

        with pytest.raises(ValueError):
            compose_extend_theta(np.zeros(4), ["bogus"], 4, 3, 5, 0.5)


class TestSeedVsOptimumReport:
    """seed_vs_optimum_report: sign-degeneracy-aware seed-vs-optimum comparison."""

    def test_identity_when_opt_equals_seed(self):
        from qmbp_simulation.analysis.warmstart import (
            second_order_warmstart_theta,
            seed_vs_optimum_report,
        )

        n_nn, n_nnn, nq, p, h = 12, 8, 9, 2, 0.5
        seed = second_order_warmstart_theta(n_nn, n_nnn, nq, p, h, J=1.0, J2=0.5)
        rep = seed_vs_optimum_report(seed, n_nn, n_nnn, nq, p, h, J=1.0, J2=0.5)
        assert rep["distance"] < 1e-12
        for ly in rep["layers"]:
            for b in ("nn", "nnn", "x"):
                assert abs(ly[b]["ratio_abs"] - 1.0) < 1e-9
                assert ly[b]["sign_flip_frac"] == 0.0

    def test_sign_flip_detected_not_magnitude_change(self):
        from qmbp_simulation.analysis.warmstart import (
            second_order_warmstart_theta,
            seed_vs_optimum_report,
        )

        n_nn, n_nnn, nq, p, h = 4, 2, 4, 1, 0.5
        seed = second_order_warmstart_theta(n_nn, n_nnn, nq, p, h, J=1.0, J2=0.5)
        flipped = -seed  # every angle sign-flipped, same magnitude
        rep = seed_vs_optimum_report(flipped, n_nn, n_nnn, nq, p, h, J=1.0, J2=0.5)
        ly = rep["layers"][0]
        # magnitude unchanged (ratio ~1), but sign flip flagged for signed blocks
        for b in ("nn", "nnn", "x"):
            assert abs(ly[b]["ratio_abs"] - 1.0) < 1e-9
            assert ly[b]["sign_flip_frac"] == 1.0

    def test_length_mismatch_raises(self):
        from qmbp_simulation.analysis.warmstart import seed_vs_optimum_report

        with pytest.raises(ValueError):
            seed_vs_optimum_report(np.zeros(5), 4, 2, 4, 1, 0.5)


class TestSecondOrderNnShrink:
    """second_order_nn_shrink_theta shrinks only θ_nn, leaving nnn/x intact."""

    def test_shrink_one_reproduces_standard_seed(self):
        from qmbp_simulation.analysis.warmstart import (
            second_order_nn_shrink_theta,
            second_order_warmstart_theta,
        )

        n_nn, n_nnn, nq, p, h = 12, 8, 9, 2, 0.5
        std = second_order_warmstart_theta(n_nn, n_nnn, nq, p, h, J=1.0, J2=0.5)
        v = second_order_nn_shrink_theta(n_nn, n_nnn, nq, p, h, J=1.0, J2=0.5, nn_extra_shrink=1.0)
        np.testing.assert_allclose(v, std, atol=1e-12)

    def test_only_nn_block_scaled(self):
        from qmbp_simulation.analysis.warmstart import (
            second_order_nn_shrink_theta,
            second_order_warmstart_theta,
        )

        n_nn, n_nnn, nq, p, h = 6, 4, 5, 2, 0.5
        std = second_order_warmstart_theta(n_nn, n_nnn, nq, p, h, J=1.0, J2=0.5)
        v = second_order_nn_shrink_theta(n_nn, n_nnn, nq, p, h, J=1.0, J2=0.5, nn_extra_shrink=0.4)
        per = n_nn + n_nnn + nq
        for layer in range(p):
            o = layer * per
            # nn block scaled by 0.4
            np.testing.assert_allclose(v[o : o + n_nn], 0.4 * std[o : o + n_nn], atol=1e-12)
            # nnn + x blocks unchanged
            np.testing.assert_allclose(v[o + n_nn : o + per], std[o + n_nn : o + per], atol=1e-12)


class TestRegimeSeedSelector:
    """select_regime_seed picks the calibrated seed per h-regime (square J2=0.5)."""

    def test_ordered_phase_uses_flat_renorm(self):
        from qmbp_simulation.analysis.warmstart import FLAT_RENORM_ZZ_COEF, select_regime_seed

        seed, name = select_regime_seed(13, 15, 10, 2, 0.3, J=1.0, J2=0.5)
        assert "flat_renorm" in name
        # θ_nn[0] == -zz_coef/h == -0.11/0.3
        assert abs(seed[0] - (-FLAT_RENORM_ZZ_COEF / 0.3)) < 1e-9

    def test_near_hc_uses_second_order(self):
        from qmbp_simulation.analysis.warmstart import select_regime_seed

        _seed, name = select_regime_seed(13, 15, 10, 2, 0.5, J=1.0, J2=0.5)
        assert name == "second_order"

    def test_paramagnet_uses_nn_shrink(self):
        from qmbp_simulation.analysis.warmstart import select_regime_seed

        _seed, name = select_regime_seed(13, 15, 10, 2, 1.5, J=1.0, J2=0.5)
        assert "so_nn_shrink" in name

    def test_boundaries(self):
        from qmbp_simulation.analysis.warmstart import (
            ORDERED_H_MAX,
            PARAMAGNETIC_H_MIN,
            select_regime_seed,
        )

        _s, n_lo = select_regime_seed(13, 15, 10, 1, ORDERED_H_MAX, J2=0.5)
        assert "flat_renorm" in n_lo  # inclusive lower edge → ordered
        _s, n_hi = select_regime_seed(13, 15, 10, 1, PARAMAGNETIC_H_MIN, J2=0.5)
        assert "so_nn_shrink" in n_hi  # inclusive → paramagnet

    def test_seed_length_matches_layout(self):
        from qmbp_simulation.analysis.warmstart import select_regime_seed

        seed, _ = select_regime_seed(13, 15, 10, 2, 0.5, J2=0.5)
        assert len(seed) == (13 + 15 + 10) * 2


class TestThetaXIndices:
    """theta_x_indices selects the RX (soft) coordinates in the flat θ vector."""

    def test_count_and_positions_p1(self):
        from qmbp_simulation.analysis.warmstart import theta_x_indices

        idx = theta_x_indices(13, 15, 10, 1)
        assert len(idx) == 10
        assert idx == list(range(28, 38))  # after 13 nn + 15 nnn

    def test_count_p2(self):
        from qmbp_simulation.analysis.warmstart import theta_x_indices

        idx = theta_x_indices(13, 15, 10, 2)
        assert len(idx) == 20
        # second layer's x block starts at 38 (per-layer) + 28
        assert 38 + 28 in idx


class TestTransferTheta:
    """transfer_theta maps a converged donor θ onto a different layout by role."""

    def _donor(self, n_nn, n_nnn, n_q, p):
        return np.arange((n_nn + n_nnn + n_q) * p, dtype=float)

    def test_same_layout_is_identity_like(self):
        from qmbp_simulation.analysis.warmstart import transfer_theta

        donor = self._donor(5, 6, 4, 2)
        out = transfer_theta(
            donor, donor_n_nn=5, donor_n_nnn=6, donor_p=2, target_n_nn=5, target_n_nnn=6, target_p=2, n_qubits=4
        )
        # same sizes, no edge map → nn/nnn/x copied verbatim (within clip range)
        assert out.shape == donor.shape

    def test_p3_to_p2_drops_extra_layer(self):
        from qmbp_simulation.analysis.warmstart import transfer_theta

        donor = self._donor(27, 39, 18, 3)  # p3
        out = transfer_theta(
            donor, donor_n_nn=27, donor_n_nnn=39, donor_p=3, target_n_nn=27, target_n_nnn=39, target_p=2, n_qubits=18
        )
        assert out.shape[0] == (27 + 39 + 18) * 2

    def test_nnn_subset_aligned_by_edge(self):
        from qmbp_simulation.analysis.warmstart import transfer_theta

        n_q = 6
        donor_nnn = [(0, 2), (1, 3), (2, 4), (3, 5)]
        target_nnn = [(2, 4), (0, 2)]  # subset, reordered
        # donor θ: give each nnn bond a recognizable value in layer 0
        donor = np.zeros((2 + len(donor_nnn) + n_q) * 1)
        # layout layer0: [nn(2), nnn(4), x(6)] → nnn at idx 2..6 (values in [-π,π])
        donor[2:6] = [0.10, 0.11, 0.12, 0.13]  # (0,2)=.10 (1,3)=.11 (2,4)=.12 (3,5)=.13
        out = transfer_theta(
            donor,
            donor_n_nn=2,
            donor_n_nnn=4,
            donor_p=1,
            target_n_nn=2,
            target_n_nnn=2,
            target_p=1,
            n_qubits=n_q,
            donor_nnn_edges=donor_nnn,
            target_nnn_edges=target_nnn,
        )
        # target nnn at idx 2..4 → should be [.12 (2,4), .10 (0,2)]
        assert out[2] == pytest.approx(0.12)
        assert out[3] == pytest.approx(0.10)

    def test_missing_edge_gets_fill(self):
        from qmbp_simulation.analysis.warmstart import transfer_theta

        donor_nnn = [(0, 2)]
        target_nnn = [(0, 2), (9, 9)]  # second not in donor
        # clean donor: nn=1, nnn=1, x=1, p=1 → len 3 (values in [-π,π])
        donor = np.array([0.5, 0.7, 0.1])
        out = transfer_theta(
            donor,
            donor_n_nn=1,
            donor_n_nnn=1,
            donor_p=1,
            target_n_nn=1,
            target_n_nnn=2,
            target_p=1,
            n_qubits=1,
            donor_nnn_edges=donor_nnn,
            target_nnn_edges=target_nnn,
            fill_value=0.0,
        )
        # layout: [nn(1), nnn(2), x(1)] → out[1]=(0,2)=0.7, out[2]=(9,9) missing=0.0
        assert out[1] == pytest.approx(0.7)
        assert out[2] == pytest.approx(0.0)

    def test_bad_donor_length_returns_none(self):
        from qmbp_simulation.analysis.warmstart import transfer_theta

        out = transfer_theta(
            np.zeros(5),
            donor_n_nn=27,
            donor_n_nnn=39,
            donor_p=3,
            target_n_nn=27,
            target_n_nnn=20,
            target_p=2,
            n_qubits=18,
        )
        assert out is None


class TestBestWarmStartSeed:
    """best_warm_start_seed — the single reusable warm-start chooser."""

    def test_fallback_to_regime_when_no_donors(self):
        from qmbp_simulation.analysis.warmstart import best_warm_start_seed

        seed, prov, fi = best_warm_start_seed(n_nn=13, n_nnn=15, n_qubits=10, p_layers=2, h=0.5, J2=0.5)
        assert len(seed) == (13 + 15 + 10) * 2
        assert prov == "second_order"
        assert fi is None

    def test_prefers_donor_transfer_without_fid_fn(self):
        from qmbp_simulation.analysis.warmstart import best_warm_start_seed

        donor = np.zeros((13 + 15 + 10) * 2)
        seed, prov, _ = best_warm_start_seed(
            n_nn=13,
            n_nnn=15,
            n_qubits=10,
            p_layers=2,
            h=0.5,
            J2=0.5,
            donors=[{"theta": donor, "n_nn": 13, "n_nnn": 15, "p": 2, "label": "p2x"}],
        )
        assert "transfer" in prov

    def test_fid_fn_selects_highest_fidelity_candidate(self):
        from qmbp_simulation.analysis.warmstart import best_warm_start_seed

        donor = np.full((13 + 15 + 10) * 2, 0.9)  # distinctive donor

        # fid_fn rewards the donor transfer over regime. best_warm_start_seed now
        # Z2-canonicalizes the donor before transfer (sign of the ZZ blocks is a
        # gauge), so key on |θ[0]| (invariant under the Z2 flip) not its sign.
        def fid_fn(theta):
            return 0.95 if abs(abs(float(theta[0])) - 0.9) < 1e-6 else 0.40

        seed, prov, fi = best_warm_start_seed(
            n_nn=13,
            n_nnn=15,
            n_qubits=10,
            p_layers=2,
            h=0.5,
            J2=0.5,
            donors=[{"theta": donor, "n_nn": 13, "n_nnn": 15, "p": 2, "label": "d"}],
            fid_fn=fid_fn,
        )
        assert "transfer" in prov
        assert fi == pytest.approx(0.95)

    def test_seed_length_always_matches_target(self):
        from qmbp_simulation.analysis.warmstart import best_warm_start_seed

        donor = np.zeros((27 + 39 + 18) * 3)  # p3 donor
        seed, _prov, _ = best_warm_start_seed(
            n_nn=27,
            n_nnn=20,
            n_qubits=18,
            p_layers=2,
            h=0.5,
            J2=0.5,
            donors=[
                {
                    "theta": donor,
                    "n_nn": 27,
                    "n_nnn": 39,
                    "p": 3,
                    "label": "p3",
                    "nnn_edges": [(i, i + 2) for i in range(39)],
                }
            ],
            target_nnn_edges=[(i, i + 2) for i in range(20)],
        )
        assert len(seed) == (27 + 20 + 18) * 2


class TestWarmStartImprovements:
    """Regression tests pinning the three warm-start transfer improvements:
    (a) Z2 canonicalization, (c) regime-gated fill, (e) micro-descent cascade."""

    def test_canonicalize_flips_positive_zz_donor(self):
        # (a) A donor with positive-sum NN block is a Z2 gauge copy; canonicalize
        # must flip the ZZ blocks (nn+nnn) to the negative-NN gauge, leaving θ_x.
        from qmbp_simulation.analysis.warmstart import transfer_theta

        # layout layer0: [nn(2), nnn(1), x(2)] → len 5, p=1
        donor = np.array([0.3, 0.5, 0.2, 1.1, 1.2])  # nn-sum = +0.8 > 0
        out = transfer_theta(
            donor,
            donor_n_nn=2,
            donor_n_nnn=1,
            donor_p=1,
            target_n_nn=2,
            target_n_nnn=1,
            target_p=1,
            n_qubits=2,
            canonicalize=True,
        )
        # ZZ blocks flipped, θ_x untouched
        assert out[0] == pytest.approx(-0.3)
        assert out[1] == pytest.approx(-0.5)
        assert out[2] == pytest.approx(-0.2)
        assert out[3] == pytest.approx(1.1)
        assert out[4] == pytest.approx(1.2)

    def test_canonicalize_off_by_default_keeps_verbatim(self):
        # Default (canonicalize=False) must preserve the verbatim-copy contract.
        from qmbp_simulation.analysis.warmstart import transfer_theta

        donor = np.array([0.3, 0.5, 0.2, 1.1, 1.2])
        out = transfer_theta(
            donor, donor_n_nn=2, donor_n_nnn=1, donor_p=1, target_n_nn=2, target_n_nnn=1, target_p=1, n_qubits=2
        )
        np.testing.assert_allclose(out, donor)

    def test_regime_fill_used_for_missing_bond(self):
        # (c) A bond absent from the donor is filled with the regime angle at that
        # position (from fill_theta), not a flat zero.
        from qmbp_simulation.analysis.warmstart import transfer_theta

        donor_nnn = [(0, 2)]
        target_nnn = [(0, 2), (9, 9)]  # second edge not in donor
        donor = np.array([0.5, 0.7, 0.1])  # nn=1 nnn=1 x=1 p=1
        # fill_theta layout: [nn(1), nnn(2), x(1)] → regime nnn angles at idx 1,2
        fill = np.array([-0.11, -0.22, -0.33, 0.4])
        out = transfer_theta(
            donor,
            donor_n_nn=1,
            donor_n_nnn=1,
            donor_p=1,
            target_n_nn=1,
            target_n_nnn=2,
            target_p=1,
            n_qubits=1,
            donor_nnn_edges=donor_nnn,
            target_nnn_edges=target_nnn,
            fill_theta=fill,
        )
        assert out[1] == pytest.approx(0.7)  # donor bond kept
        assert out[2] == pytest.approx(-0.33)  # missing bond → regime fill (not 0)

    def test_micro_descent_cascade_picks_best_basin(self):
        # (e) With descent_fn, the candidate whose MICRO-DESCENT yields the higher
        # fidelity is chosen (and its refined θ returned), even if its raw seed
        # fidelity would lose.
        from qmbp_simulation.analysis.warmstart import best_warm_start_seed

        donor = np.full((2 + 1 + 2) * 1, 0.9)

        def descent_fn(theta):
            # The regime seed (negative NN) relaxes into a better basin here.
            refined = np.asarray(theta, float)
            fid = 0.99 if float(refined[0]) < 0 else 0.50
            return refined, fid

        def fid_fn(theta):  # raw-seed ranking would prefer the donor
            return 0.95 if abs(abs(float(theta[0])) - 0.9) < 1e-6 else 0.10

        seed, prov, fi = best_warm_start_seed(
            n_nn=2,
            n_nnn=1,
            n_qubits=2,
            p_layers=1,
            h=0.5,
            J2=0.5,
            donors=[{"theta": donor, "n_nn": 2, "n_nnn": 1, "p": 1, "label": "d"}],
            fid_fn=fid_fn,
            descent_fn=descent_fn,
        )
        assert "descent" in prov
        assert fi == pytest.approx(0.99)
        assert float(seed[0]) < 0  # regime (negative-NN) basin won

    def test_descent_fn_falls_back_when_all_fail(self):
        # If every descent raises, selection falls back to fid_fn ranking.
        from qmbp_simulation.analysis.warmstart import best_warm_start_seed

        def descent_fn(theta):
            raise RuntimeError("descent blew up")

        def fid_fn(theta):
            return 0.60

        seed, prov, fi = best_warm_start_seed(
            n_nn=2, n_nnn=1, n_qubits=2, p_layers=1, h=0.5, J2=0.5, fid_fn=fid_fn, descent_fn=descent_fn
        )
        assert "descent" not in prov
        assert fi == pytest.approx(0.60)


class TestGapAdaptiveBudget:
    """restarts_for_gap + topk_frac_for_gap scale the budget by spectral gap."""

    def test_restarts_increase_as_gap_shrinks(self):
        from qmbp_simulation.analysis.warmstart import SMALL_GAP, restarts_for_gap

        big = restarts_for_gap(1.0, base_restarts=1)
        mid = restarts_for_gap(SMALL_GAP / 2, base_restarts=1)
        tiny = restarts_for_gap(SMALL_GAP / 100, base_restarts=1)
        assert big == 1
        assert mid == 2
        assert tiny == 3
        assert big <= mid <= tiny  # monotone non-decreasing as gap shrinks

    def test_restarts_clipped_to_max(self):
        from qmbp_simulation.analysis.warmstart import restarts_for_gap

        assert restarts_for_gap(1e-9, base_restarts=4, max_restarts=4) == 4

    def test_restarts_none_gap_uses_base(self):
        from qmbp_simulation.analysis.warmstart import restarts_for_gap

        assert restarts_for_gap(None, base_restarts=2) == 2

    def test_topk_frac_shrinks_only_at_tiny_gap(self):
        from qmbp_simulation.analysis.warmstart import SMALL_GAP, topk_frac_for_gap

        assert topk_frac_for_gap(1.0, 0.5) == 0.5  # large gap → unchanged
        assert topk_frac_for_gap(SMALL_GAP / 2, 0.5) == 0.5  # small but not tiny
        assert topk_frac_for_gap(SMALL_GAP / 100, 0.5) == pytest.approx(0.33)  # tiny

    def test_topk_frac_never_grows(self):
        from qmbp_simulation.analysis.warmstart import topk_frac_for_gap

        # already below aggressive floor → kept, not raised
        assert topk_frac_for_gap(1e-9, 0.25) == pytest.approx(0.25)

    def test_topk_frac_rejects_bad_base(self):
        from qmbp_simulation.analysis.warmstart import topk_frac_for_gap

        with pytest.raises(ValueError):
            topk_frac_for_gap(0.5, 1.5)


class TestCrossesTransition:
    """crosses_transition detects phase-boundary crossings for the h-sweep."""

    def test_within_ordered_phase_no_cross(self):
        from qmbp_simulation.analysis.warmstart import crosses_transition

        assert crosses_transition(0.4, 0.3) is False  # both ordered (<=0.45)

    def test_within_paramagnet_no_cross(self):
        from qmbp_simulation.analysis.warmstart import crosses_transition

        assert crosses_transition(1.5, 1.3) is False  # both paramagnet (>=1.2)

    def test_ordered_to_near_hc_crosses(self):
        from qmbp_simulation.analysis.warmstart import crosses_transition

        assert crosses_transition(0.5, 0.3) is True  # near-h_c → ordered

    def test_near_hc_to_paramagnet_crosses(self):
        from qmbp_simulation.analysis.warmstart import crosses_transition

        assert crosses_transition(0.5, 1.3) is True  # near-h_c → paramagnet

    def test_symmetric(self):
        from qmbp_simulation.analysis.warmstart import crosses_transition

        assert crosses_transition(0.3, 0.5) == crosses_transition(0.5, 0.3)


class TestBondResolvedRegimeSeed:
    """bond_resolved_regime_seed modulates the regime seed by a donor's shape."""

    def test_no_donor_returns_plain_regime_seed(self):
        from qmbp_simulation.analysis.warmstart import (
            bond_resolved_regime_seed,
            select_regime_seed,
        )

        seed, name = bond_resolved_regime_seed(13, 15, 10, 2, 0.5, J2=0.5)
        ref, ref_name = select_regime_seed(13, 15, 10, 2, 0.5, J2=0.5)
        np.testing.assert_allclose(seed, ref)
        assert name == ref_name

    def test_donor_modulates_and_preserves_length(self):
        from qmbp_simulation.analysis.warmstart import (
            bond_resolved_regime_seed,
            select_regime_seed,
        )

        rng = np.random.default_rng(1)
        donor = rng.normal(0.0, 0.2, (13 + 15 + 10) * 2)
        seed, name = bond_resolved_regime_seed(
            13, 15, 10, 2, 0.5, J2=0.5, donor_theta=donor, donor_n_nn=13, donor_n_nnn=15, donor_p=2
        )
        plain, _ = select_regime_seed(13, 15, 10, 2, 0.5, J2=0.5)
        assert len(seed) == (13 + 15 + 10) * 2
        assert "bondshape" in name
        assert not np.allclose(seed, plain)  # shape actually changed the seed

    def test_strength_zero_recovers_plain_seed(self):
        from qmbp_simulation.analysis.warmstart import (
            bond_resolved_regime_seed,
            select_regime_seed,
        )

        rng = np.random.default_rng(2)
        donor = rng.normal(0.0, 0.2, (13 + 15 + 10) * 2)
        seed, _ = bond_resolved_regime_seed(
            13, 15, 10, 2, 0.5, J2=0.5, donor_theta=donor, donor_n_nn=13, donor_n_nnn=15, donor_p=2, strength=0.0
        )
        # strength=0 → canonicalized plain regime seed (sign-canonical compare)
        from qmbp_simulation.analysis.theta_patterns import canonicalize_z2

        plain, _ = select_regime_seed(13, 15, 10, 2, 0.5, J2=0.5)
        np.testing.assert_allclose(
            canonicalize_z2(seed, 13, 15, 10, 2), canonicalize_z2(plain, 13, 15, 10, 2), atol=1e-9
        )

    def test_mismatched_layout_falls_back_to_plain(self):
        from qmbp_simulation.analysis.warmstart import bond_resolved_regime_seed

        donor = np.zeros((27 + 39 + 18) * 3)  # different layout
        seed, name = bond_resolved_regime_seed(
            13, 15, 10, 2, 0.5, J2=0.5, donor_theta=donor, donor_n_nn=27, donor_n_nnn=39, donor_p=3
        )
        assert len(seed) == (13 + 15 + 10) * 2
        assert "bondshape" not in name  # fell back to the plain regime seed

    def test_theta_x_block_untouched_by_modulation(self):
        from qmbp_simulation.analysis.warmstart import (
            bond_resolved_regime_seed,
            select_regime_seed,
            theta_x_indices,
        )

        rng = np.random.default_rng(3)
        donor = rng.normal(0.0, 0.2, (13 + 15 + 10) * 2)
        seed, _ = bond_resolved_regime_seed(
            13, 15, 10, 2, 0.5, J2=0.5, donor_theta=donor, donor_n_nn=13, donor_n_nnn=15, donor_p=2
        )
        plain, _ = select_regime_seed(13, 15, 10, 2, 0.5, J2=0.5)
        from qmbp_simulation.analysis.theta_patterns import canonicalize_z2

        x_idx = theta_x_indices(13, 15, 10, 2)
        # θ_x is Z2-invariant; compare on the canonical seeds at x positions
        s_c = canonicalize_z2(seed, 13, 15, 10, 2)
        p_c = canonicalize_z2(plain, 13, 15, 10, 2)
        np.testing.assert_allclose(s_c[x_idx], p_c[x_idx], atol=1e-9)


class TestWarmstartInitFidelity:
    """warmstart_init_fidelity: cheap transfer+evaluate probe (no reoptimize)."""

    def test_returns_fid_from_injected_fn(self):
        from qmbp_simulation.analysis.warmstart import warmstart_init_fidelity

        donor = np.full((13 + 15 + 10) * 2, 0.1)
        fi, seed = warmstart_init_fidelity(
            donor,
            donor_n_nn=13,
            donor_n_nnn=15,
            donor_p=2,
            target_n_nn=13,
            target_n_nnn=15,
            target_p=2,
            n_qubits=10,
            fid_fn=lambda t: 0.873,
        )
        assert fi == pytest.approx(0.873)
        assert seed is not None and seed.size == (13 + 15 + 10) * 2

    def test_fid_fn_receives_transferred_seed(self):
        from qmbp_simulation.analysis.warmstart import warmstart_init_fidelity

        seen = {}

        def fid_fn(theta):
            seen["len"] = len(theta)
            return 0.5

        warmstart_init_fidelity(
            np.zeros((2 + 1 + 4) * 1),
            donor_n_nn=2,
            donor_n_nnn=1,
            donor_p=1,
            target_n_nn=2,
            target_n_nnn=1,
            target_p=1,
            n_qubits=4,
            fid_fn=fid_fn,
        )
        assert seen["len"] == (2 + 1 + 4) * 1

    def test_bad_donor_returns_none(self):
        from qmbp_simulation.analysis.warmstart import warmstart_init_fidelity

        fi, seed = warmstart_init_fidelity(
            np.zeros(5),
            donor_n_nn=27,
            donor_n_nnn=39,
            donor_p=3,
            target_n_nn=27,
            target_n_nnn=39,
            target_p=2,
            n_qubits=18,
            fid_fn=lambda t: 1.0,
        )
        assert fi is None and seed is None

    def test_fid_fn_exception_returns_none_fid_but_keeps_seed(self):
        from qmbp_simulation.analysis.warmstart import warmstart_init_fidelity

        def boom(theta):
            raise RuntimeError("eval failed")

        fi, seed = warmstart_init_fidelity(
            np.zeros((2 + 1 + 4) * 1),
            donor_n_nn=2,
            donor_n_nnn=1,
            donor_p=1,
            target_n_nn=2,
            target_n_nnn=1,
            target_p=1,
            n_qubits=4,
            fid_fn=boom,
        )
        assert fi is None
        assert seed is not None  # transfer succeeded, only the eval failed


class TestThetaXArctanDeviation:
    """theta_x_arctan_deviation: θ_x block mean vs arctan(J/h)."""

    def test_exact_arctan_seed_has_zero_deviation(self):
        from qmbp_simulation.analysis.warmstart import (
            first_order_warmstart_theta,
            theta_x_arctan_deviation,
        )

        # first-order seed sets θ_x = arctan(J/h) exactly → deviation ~0
        seed = first_order_warmstart_theta(13, 15, 10, 2, 0.5, J2=0.5)
        dev = theta_x_arctan_deviation(seed, 13, 15, 10, 2, 0.5, J=1.0)
        assert dev["rel_deviation"] == pytest.approx(0.0, abs=1e-9)
        assert dev["tracks_arctan"] is True
        assert dev["arctan_pred"] == pytest.approx(np.arctan(1.0 / 0.5))

    def test_arctan_pred_decreases_with_h(self):
        from qmbp_simulation.analysis.warmstart import theta_x_arctan_deviation

        seed = np.zeros((13 + 15 + 10) * 2)
        lo = theta_x_arctan_deviation(seed, 13, 15, 10, 2, 0.3)["arctan_pred"]
        hi = theta_x_arctan_deviation(seed, 13, 15, 10, 2, 1.3)["arctan_pred"]
        assert lo > hi  # arctan(1/h) decreases as h grows

    def test_rejects_nonpositive_h(self):
        from qmbp_simulation.analysis.warmstart import theta_x_arctan_deviation

        with pytest.raises(ValueError):
            theta_x_arctan_deviation(np.zeros(76), 13, 15, 10, 2, 0.0)

    def test_per_layer_means_length_matches_p(self):
        from qmbp_simulation.analysis.warmstart import theta_x_arctan_deviation

        dev = theta_x_arctan_deviation(np.zeros((13 + 15 + 10) * 2), 13, 15, 10, 2, 0.5)
        assert len(dev["theta_x_mean_per_layer"]) == 2


class TestDifficultyIndex:
    """difficulty_index + phase_proximity + budget_for_difficulty (Fase A)."""

    def test_proximity_peaks_at_h_critical(self):
        from qmbp_simulation.analysis.warmstart import H_CRITICAL, phase_proximity

        assert phase_proximity(H_CRITICAL) == pytest.approx(1.0)
        assert phase_proximity(0.3) < 1.0
        assert phase_proximity(1.3) < phase_proximity(0.3)  # paramagnet furthest

    def test_proximity_zero_far_from_hc(self):
        from qmbp_simulation.analysis.warmstart import phase_proximity

        # one+ half-width away → 0
        assert phase_proximity(1.3) == pytest.approx(0.0)

    def test_difficulty_grows_with_n_in_transition(self):
        from qmbp_simulation.analysis.warmstart import difficulty_index

        d8 = difficulty_index(8, 0.5, 0.006)
        d12 = difficulty_index(12, 0.5, 0.006)
        d18 = difficulty_index(18, 0.5, 0.006)
        assert d8 < d12 < d18  # monotone in N at the transition

    def test_difficulty_near_zero_in_paramagnet(self):
        from qmbp_simulation.analysis.warmstart import difficulty_index

        # paramagnet is easy at any N → difficulty ~0
        assert difficulty_index(18, 1.3, 1.1) == pytest.approx(0.0)

    def test_difficulty_ignores_gap_value(self):
        from qmbp_simulation.analysis.warmstart import difficulty_index

        # same N,h but wildly different gaps → identical difficulty (gap unused)
        a = difficulty_index(12, 0.5, 1e-5)
        b = difficulty_index(12, 0.5, 0.5)
        assert a == pytest.approx(b)

    def test_budget_more_restarts_in_transition(self):
        from qmbp_simulation.analysis.warmstart import budget_for_difficulty

        r_small, _, D_small = budget_for_difficulty(8, 0.5, 0.006)
        r_big, _, D_big = budget_for_difficulty(18, 0.5, 0.006)
        assert D_big > D_small
        assert r_big >= r_small  # bigger N at transition → more restarts

    def test_budget_base_restarts_in_easy_phase(self):
        from qmbp_simulation.analysis.warmstart import budget_for_difficulty

        r, _frac, D = budget_for_difficulty(18, 1.3, 1.1, base_restarts=1)
        assert pytest.approx(0.0) == D
        assert r == 1  # paramagnet stays at base, no extra restarts

    def test_budget_clips_to_max(self):
        from qmbp_simulation.analysis.warmstart import budget_for_difficulty

        r, _frac, _D = budget_for_difficulty(40, 0.5, 0.006, base_restarts=3, max_restarts=4)
        assert r == 4


class TestExtractThetaMetrics:
    """extract_theta_metrics: flat tidy per-θ metrics row."""

    def test_phase_labels_by_h(self):
        from qmbp_simulation.analysis.theta_patterns import extract_theta_metrics

        z = np.zeros((13 + 15 + 10) * 2)
        assert extract_theta_metrics(z, 13, 15, 10, 2, 0.3)["phase"] == "ordered"
        assert extract_theta_metrics(z, 13, 15, 10, 2, 0.5)["phase"] == "near_hc"
        assert extract_theta_metrics(z, 13, 15, 10, 2, 1.3)["phase"] == "paramag"

    def test_d_theta_to_seed_zero_for_same_theta(self):
        from qmbp_simulation.analysis.theta_patterns import extract_theta_metrics

        rng = np.random.default_rng(0)
        th = rng.normal(0, 0.2, (13 + 15 + 10) * 2)
        m = extract_theta_metrics(th, 13, 15, 10, 2, 0.5, seed_theta=th)
        assert m["d_theta_to_seed"] == pytest.approx(0.0, abs=1e-9)

    def test_d_theta_none_without_seed(self):
        from qmbp_simulation.analysis.theta_patterns import extract_theta_metrics

        m = extract_theta_metrics(np.zeros((13 + 15 + 10) * 2), 13, 15, 10, 2, 0.5)
        assert m["d_theta_to_seed"] is None

    def test_absmeans_nonnegative_and_keys_present(self):
        from qmbp_simulation.analysis.theta_patterns import extract_theta_metrics

        rng = np.random.default_rng(1)
        th = rng.normal(0, 0.3, (13 + 15 + 10) * 2)
        m = extract_theta_metrics(th, 13, 15, 10, 2, 0.5)
        for k in ("nn_absmean", "nnn_absmean", "x_absmean", "l2_l1_nn_ratio"):
            assert m[k] >= 0.0
        assert m["N"] == 10 and m["p_layers"] == 2


class TestCalibratedWarmstart:
    """calibrated_zz_coef / calibrated_x_scale / calibrated_warmstart_theta."""

    def test_zz_coef_bounds(self):
        from qmbp_simulation.analysis.warmstart import calibrated_zz_coef

        for h in (0.1, 0.3, 0.5, 1.0, 2.0, 5.0):
            c = calibrated_zz_coef(h)
            assert 1e-3 <= c <= 0.25

    def test_zz_coef_rises_toward_paramagnet(self):
        from qmbp_simulation.analysis.warmstart import calibrated_zz_coef

        # ZZ prefactor grows from the transition region to the paramagnet
        assert calibrated_zz_coef(0.5) < calibrated_zz_coef(2.0)

    def test_x_scale_bounds(self):
        from qmbp_simulation.analysis.warmstart import calibrated_x_scale

        for h in (0.1, 0.3, 0.5, 0.9, 1.0, 1.5, 3.0):
            xs = calibrated_x_scale(h)
            assert 0.7 <= xs <= 1.6

    def test_x_scale_amplifies_approaching_hc_break(self):
        from qmbp_simulation.analysis.warmstart import calibrated_x_scale

        # below the break, x_scale increases with h (θ_x amplified toward h≈1)
        assert calibrated_x_scale(0.3) < calibrated_x_scale(0.9)
        # regime break: paramagnet side starts lower than the pre-break peak
        assert calibrated_x_scale(1.1) < calibrated_x_scale(1.0)

    def test_seed_length_and_finiteness(self):
        from qmbp_simulation.analysis.warmstart import calibrated_warmstart_theta

        seed = calibrated_warmstart_theta(13, 15, 10, 2, 0.5, J2=0.5)
        assert seed.shape[0] == (13 + 15 + 10) * 2
        assert np.all(np.isfinite(seed))
        assert np.all(np.abs(seed) <= np.pi)

    def test_seed_nn_uses_calibrated_coef(self):
        from qmbp_simulation.analysis.warmstart import (
            calibrated_warmstart_theta,
            calibrated_zz_coef,
        )

        h = 0.5
        seed = calibrated_warmstart_theta(13, 15, 10, 2, h, J=1.0, J2=0.5)
        # θ_nn[0] == -J * zz_coef(h) / h
        assert seed[0] == pytest.approx(-1.0 * calibrated_zz_coef(h) / h, abs=1e-9)

    def test_seed_x_block_scaled(self):
        from qmbp_simulation.analysis.warmstart import (
            calibrated_warmstart_theta,
            calibrated_x_scale,
            calibrated_zz_coef,
            first_order_warmstart_theta,
        )

        h = 0.7
        cal = calibrated_warmstart_theta(13, 15, 10, 2, h, J2=0.5)
        base = first_order_warmstart_theta(13, 15, 10, 2, h, J2=0.5, zz_coef=calibrated_zz_coef(h))
        # x block (last 10 of layer 0) is the base θ_x times x_scale(h)
        x0_cal = cal[13 + 15 : 13 + 15 + 10]
        x0_base = base[13 + 15 : 13 + 15 + 10]
        np.testing.assert_allclose(x0_cal, x0_base * calibrated_x_scale(h), atol=1e-9)


class TestRegimeSeedCalibratedFlag:
    """select_regime_seed(use_calibrated=...) opt-in integration."""

    def test_default_preserves_regime_gating(self):
        from qmbp_simulation.analysis.warmstart import select_regime_seed

        # off (default) → the established regime names, unchanged
        assert select_regime_seed(13, 15, 10, 2, 0.3, J2=0.5)[1].startswith("flat_renorm")
        assert select_regime_seed(13, 15, 10, 2, 0.5, J2=0.5)[1] == "second_order"
        assert select_regime_seed(13, 15, 10, 2, 1.5, J2=0.5)[1].startswith("so_nn_shrink")

    def test_calibrated_flag_switches_seed(self):
        from qmbp_simulation.analysis.warmstart import (
            calibrated_warmstart_theta,
            select_regime_seed,
        )

        seed, name = select_regime_seed(13, 15, 10, 2, 0.5, J2=0.5, use_calibrated=True)
        assert name == "calibrated"
        np.testing.assert_allclose(seed, calibrated_warmstart_theta(13, 15, 10, 2, 0.5, J2=0.5))

    def test_calibrated_flag_length_matches(self):
        from qmbp_simulation.analysis.warmstart import select_regime_seed

        for h in (0.3, 0.7, 1.3):
            seed, _ = select_regime_seed(13, 15, 10, 2, h, J2=0.5, use_calibrated=True)
            assert len(seed) == (13 + 15 + 10) * 2


class TestBestCombinedWarmstart:
    """best_combined_warmstart — the one-call combined cascade."""

    L = (13, 15, 10, 2)  # n_nn, n_nnn, n_qubits, p

    def _kw(self, **extra):
        n_nn, n_nnn, n_q, p = self.L
        base = dict(n_nn=n_nn, n_nnn=n_nnn, n_qubits=n_q, p_layers=p, h=0.5, J2=0.5)
        base.update(extra)
        return base

    def test_no_evaluator_returns_calibrated_first(self):
        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        r = best_combined_warmstart(**self._kw())
        assert r["provenance"] == "calibrated"
        assert r["seed"].size == (13 + 15 + 10) * 2
        assert r["init_fidelity"] is None
        # report always lists the built-in candidates
        labels = [x["label"] for x in r["report"]]
        assert "calibrated" in labels

    def test_descent_selects_best_basin_and_refines(self):
        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        # descent favors the candidate with the most-negative first entry,
        # and returns a refined θ (here: the input unchanged + its score)
        def descent_fn(theta):
            return theta, -float(theta[0])

        r = best_combined_warmstart(**self._kw(descent_fn=descent_fn))
        assert r["provenance"].endswith("+descent")
        assert r["init_fidelity"] is not None
        selected = [x for x in r["report"] if x["selected"]]
        assert len(selected) == 1
        assert selected[0]["descent_fid"] == pytest.approx(r["init_fidelity"])

    def test_fid_fn_ranks_raw_candidates(self):
        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        # reward the regime seed specifically (nn[0] == -0.11/0.5 for flat_renorm
        # is not the regime at h=0.5; at h=0.5 regime is second_order). Just make
        # fid_fn depend on the seed so the max is well-defined and recorded.
        def fid_fn(theta):
            return float(1.0 / (1.0 + abs(theta[0])))

        r = best_combined_warmstart(**self._kw(fid_fn=fid_fn))
        assert r["init_fidelity"] is not None
        # every candidate got a raw_fid recorded
        assert all(x["raw_fid"] is not None for x in r["report"])

    def test_cross_n_donor_included(self):
        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        # a donor from a smaller N (9 qubits) at a different layout, carrying its
        # own n_qubits → transferred via donor_n_qubits
        donor = {
            "theta": np.zeros((11 + 13 + 9) * 2),
            "n_nn": 11,
            "n_nnn": 13,
            "p": 2,
            "n_qubits": 9,
            "label": "crossN9",
        }
        r = best_combined_warmstart(**self._kw(fid_fn=lambda t: 0.5, donors=[donor]))
        labels = [x["label"] for x in r["report"]]
        assert any("crossN9" in lbl for lbl in labels)

    def test_cross_n_donor_auto_yields_full_and_nnx(self):
        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        # rigorous divide-&-conquer: a cross-N donor automatically contributes
        # BOTH a full transfer and an nn+x-only transfer (nnn left at regime).
        donor = {
            "theta": np.zeros((11 + 13 + 9) * 2),
            "n_nn": 11,
            "n_nnn": 13,
            "p": 2,
            "n_qubits": 9,
            "label": "crossN9",
        }
        r = best_combined_warmstart(**self._kw(fid_fn=lambda t: 0.5, donors=[donor]))
        labels = [x["label"] for x in r["report"]]
        assert any(lbl.endswith("|nn+x") for lbl in labels)  # nn+x variant
        assert any("crossN9>" in lbl and not lbl.endswith("|nn+x") for lbl in labels)  # full variant

    def test_same_layout_donor_full_only(self):
        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        # same N and same nnn count → only the full transfer (nnn is reliable)
        n_nn, n_nnn, n_q, _p = self.L
        donor = {
            "theta": np.zeros((n_nn + n_nnn + n_q) * 2),
            "n_nn": n_nn,
            "n_nnn": n_nnn,
            "p": 2,
            "n_qubits": n_q,
            "label": "sameL",
        }
        r = best_combined_warmstart(**self._kw(fid_fn=lambda t: 0.5, donors=[donor]))
        labels = [x["label"] for x in r["report"]]
        assert not any(lbl.endswith("|nn+x") for lbl in labels)

    def test_extra_candidates_compete(self):
        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        guess = np.full((13 + 15 + 10) * 2, 0.05)
        r = best_combined_warmstart(**self._kw(fid_fn=lambda t: 0.5, extra_candidates=[(guess, "myguess")]))
        labels = [x["label"] for x in r["report"]]
        assert "myguess" in labels

    def test_toggles_exclude_builtins(self):
        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        r = best_combined_warmstart(**self._kw(include_calibrated=False, include_regime=True))
        labels = [x["label"] for x in r["report"]]
        assert "calibrated" not in labels

    def test_degenerate_guard_regime_always_available(self):
        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        # exclude both built-ins and give no donors/extras → still returns a valid
        # seed (regime fallback), never empty
        r = best_combined_warmstart(**self._kw(include_calibrated=False, include_regime=False))
        assert r["seed"].size == (13 + 15 + 10) * 2

    def test_descent_exception_falls_back_to_fid(self):
        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        def boom(theta):
            raise RuntimeError("descent failed")

        r = best_combined_warmstart(**self._kw(descent_fn=boom, fid_fn=lambda t: 0.7))
        # all descents failed → fid_fn path, provenance has no +descent
        assert not r["provenance"].endswith("+descent")
        assert r["init_fidelity"] == pytest.approx(0.7)


class TestTransferThetaForBlocks:
    """Structure-aware full→masked θ transfer (p2_half_nn_rx and friends)."""

    def test_copies_nn_x_verbatim_and_selects_nnn(self):
        import numpy as np

        from qmbp_simulation.analysis.warmstart import transfer_theta_for_blocks

        # blocks [nn, nnn, x] once; donor has 2 nnn edges, target keeps 1.
        n_nn, nq = 2, 2
        donor_nnn = [(0, 2), (1, 3)]
        target_nnn = [(1, 3)]
        blocks = ["nn", "nnn", "x"]
        # donor θ: nn=[0.1,0.2] nnn=[0.3,0.4] x=[0.5,0.6]
        donor = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        out = transfer_theta_for_blocks(
            donor, blocks, donor_nnn_edges=donor_nnn, target_nnn_edges=target_nnn, n_nn=n_nn, n_qubits=nq
        )
        # target layout: nn=[0.1,0.2] nnn=[0.4] (edge (1,3)) x=[0.5,0.6]
        np.testing.assert_allclose(out, [0.1, 0.2, 0.4, 0.5, 0.6])

    def test_rx_final_copied(self):
        import numpy as np

        from qmbp_simulation.analysis.warmstart import transfer_theta_for_blocks

        n_nn, nq = 1, 2
        donor_nnn = [(0, 2)]
        blocks = ["nn", "nnn", "x"]
        # donor: nn=[0.1] nnn=[0.2] x=[0.3,0.4] rx_final=[0.7,0.8]
        donor = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8])
        out = transfer_theta_for_blocks(
            donor, blocks, donor_nnn_edges=donor_nnn, target_nnn_edges=donor_nnn, n_nn=n_nn, n_qubits=nq, rx_final=True
        )
        # full-bond target (same nnn) → identical to donor
        np.testing.assert_allclose(out, donor)

    def test_length_mismatch_returns_none(self):
        import numpy as np

        from qmbp_simulation.analysis.warmstart import transfer_theta_for_blocks

        out = transfer_theta_for_blocks(
            np.zeros(3), ["nn", "nnn", "x"], donor_nnn_edges=[(0, 2)], target_nnn_edges=[(0, 2)], n_nn=2, n_qubits=2
        )
        assert out is None


class TestGeometricCrossNMatching:
    """Grid-cell alignment for cross-N donor transfer (M1 migrated to core)."""

    def test_lattice_coords_square_grid(self):
        from qmbp_simulation.analysis.warmstart import lattice_coords

        # square N=18 → ceil(sqrt(18))=5 cols; qubit 7 is row1,col2.
        c = lattice_coords("square", 18)
        assert c[0] == (0, 0)
        assert c[7] == (1, 2)
        assert c[17] == (3, 2)
        assert len(c) == 18

    def test_lattice_coords_nongrid_is_none(self):
        from qmbp_simulation.analysis.warmstart import lattice_coords

        # Non-grid topologies have no trivial embedding → index-match fallback.
        assert lattice_coords("triangular", 12) is None
        assert lattice_coords("kagome", 12) is None

    def test_remap_none_coords_is_identity(self):
        from qmbp_simulation.analysis.warmstart import remap_edges_by_coords

        edges = [(0, 1), (2, 3)]
        # Missing either coord map → edges returned unchanged (back-compat).
        assert remap_edges_by_coords(edges, None, {0: (0, 0)}) == edges
        assert remap_edges_by_coords(edges, {0: (0, 0)}, None) == edges

    def test_remap_aligns_by_cell_across_widths(self):
        from qmbp_simulation.analysis.warmstart import lattice_coords, remap_edges_by_coords

        # Same physical bond gets DIFFERENT raw indices at N10 (4 cols) vs N18
        # (5 cols). Remapping by cell must translate the donor edge to the target
        # index for the SAME two grid cells.
        dc = lattice_coords("square", 10)  # 4 cols
        tc = lattice_coords("square", 18)  # 5 cols
        # donor edge (0,1): cells (0,0)-(0,1). In the target those cells are
        # qubits 0 and 1 as well (both top-left) → maps to (0,1).
        remapped = remap_edges_by_coords([(0, 1)], dc, tc)
        assert remapped == [(0, 1)]
        # donor edge (0,4): cells (0,0)-(1,0) [col width 4]. Target cell (1,0) is
        # qubit 5 (col width 5) → donor (0,4) must remap to (0,5), NOT stay (0,4).
        remapped2 = remap_edges_by_coords([(0, 4)], dc, tc)
        assert remapped2 == [(0, 5)]

    def test_remap_absent_cell_sentinel(self):
        from qmbp_simulation.analysis.warmstart import lattice_coords, remap_edges_by_coords

        # A donor cell that doesn't exist in a SMALLER target → sentinel (-1,-1).
        dc = lattice_coords("square", 18)
        tc = lattice_coords("square", 10)
        # qubit 17 in N18 is cell (3,2); N10 has no row 3 → sentinel.
        out = remap_edges_by_coords([(0, 17)], dc, tc)
        assert out[0] == (-1, -1)

    def test_geometric_transfer_matches_more_than_raw(self):
        import numpy as np

        from qmbp_simulation.analysis.warmstart import (
            lattice_coords,
            transfer_theta_for_blocks,
        )
        from qmbp_simulation.models import make_lattice
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        # Donor N10 → target N18, single [nn,nnn,x] block. The geometric path must
        # fill MORE target nn entries (non-fill) than the raw-index path, because
        # grid widths differ (4 vs 5 cols).
        ld = make_lattice("square", 10, J=1.0, h=0.5)
        lt = make_lattice("square", 18, J=1.0, h=0.5)
        nn_d, nn_t = list(ld.edges), list(lt.edges)
        nnn_d = HamiltonianBuilder._generate_nnn_edges(ld)
        nnn_t = HamiltonianBuilder._generate_nnn_edges(lt)
        donor = np.linspace(0.1, 1.0, len(nn_d) + len(nnn_d) + 10)
        common = dict(
            donor_nnn_edges=nnn_d,
            target_nnn_edges=nnn_t,
            n_nn=len(nn_t),
            n_qubits=18,
            donor_n_nn=len(nn_d),
            donor_n_qubits=10,
            donor_nn_edges=nn_d,
            target_nn_edges=nn_t,
        )
        raw = transfer_theta_for_blocks(donor, ["nn", "nnn", "x"], **common)
        geo = transfer_theta_for_blocks(
            donor,
            ["nn", "nnn", "x"],
            **common,
            donor_coords=lattice_coords("square", 10),
            target_coords=lattice_coords("square", 18),
        )
        assert raw is not None and geo is not None
        # nn block is the first len(nn_t) entries; count non-zero (transferred).
        raw_nn_filled = int(np.sum(np.abs(raw[: len(nn_t)]) > 1e-12))
        geo_nn_filled = int(np.sum(np.abs(geo[: len(nn_t)]) > 1e-12))
        assert geo_nn_filled > raw_nn_filled

    def test_no_coords_is_backcompat(self):
        import numpy as np

        from qmbp_simulation.analysis.warmstart import transfer_theta_for_blocks

        # Without coords the result must be byte-identical to the pre-migration
        # behavior (same-N verbatim transfer).
        n_nn, nq = 2, 2
        donor_nnn = [(0, 2), (1, 3)]
        blocks = ["nn", "nnn", "x"]
        donor = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        out = transfer_theta_for_blocks(
            donor, blocks, donor_nnn_edges=donor_nnn, target_nnn_edges=donor_nnn, n_nn=n_nn, n_qubits=nq
        )
        np.testing.assert_allclose(out, donor)


class TestTransferDonorBlocks:
    """transfer_theta donor_blocks: selective per-block transfer (divide & conquer)."""

    def _donor(self):
        # layout n_nn=2, n_nnn=2, n_q=4, p=1; distinguishable blocks
        return np.array([-0.9, -0.8, 0.11, 0.12, 1.40, 1.41, 1.42, 1.43])

    def test_default_transfers_all_blocks(self):
        from qmbp_simulation.analysis.warmstart import transfer_theta

        donor = self._donor()
        out = transfer_theta(
            donor, donor_n_nn=2, donor_n_nnn=2, donor_p=1, target_n_nn=2, target_n_nnn=2, target_p=1, n_qubits=4
        )
        np.testing.assert_allclose(out, donor)  # all blocks from donor

    def test_exclude_nnn_uses_fill(self):
        from qmbp_simulation.analysis.warmstart import transfer_theta

        donor = self._donor()
        fill = np.array([-0.11, -0.11, -0.33, -0.33, 0.9, 0.9, 0.9, 0.9])
        out = transfer_theta(
            donor,
            donor_n_nn=2,
            donor_n_nnn=2,
            donor_p=1,
            target_n_nn=2,
            target_n_nnn=2,
            target_p=1,
            n_qubits=4,
            fill_theta=fill,
            donor_blocks=("nn", "x"),
        )
        # nn + x from donor, nnn from fill
        assert out[0] == pytest.approx(-0.9) and out[1] == pytest.approx(-0.8)
        assert out[2] == pytest.approx(-0.33) and out[3] == pytest.approx(-0.33)  # nnn = fill
        np.testing.assert_allclose(out[4:8], donor[4:8])  # x from donor

    def test_exclude_nnn_without_fill_uses_fill_value(self):
        from qmbp_simulation.analysis.warmstart import transfer_theta

        donor = self._donor()
        out = transfer_theta(
            donor,
            donor_n_nn=2,
            donor_n_nnn=2,
            donor_p=1,
            target_n_nn=2,
            target_n_nnn=2,
            target_p=1,
            n_qubits=4,
            donor_blocks=("nn", "x"),
            fill_value=0.0,
        )
        assert out[2] == pytest.approx(0.0) and out[3] == pytest.approx(0.0)


class TestStructuralWarmstart:
    """structural_warmstart_theta: θ_x const + sparse (suppressed) nnn."""

    def test_length_and_finite(self):
        from qmbp_simulation.analysis.warmstart import structural_warmstart_theta

        s = structural_warmstart_theta(13, 15, 10, 2, 0.5, J2=0.5)
        assert s.shape[0] == (13 + 15 + 10) * 2
        assert np.all(np.isfinite(s)) and np.all(np.abs(s) <= np.pi)

    def test_nnn_suppressed_vs_calibrated(self):
        from qmbp_simulation.analysis.warmstart import (
            calibrated_warmstart_theta,
            structural_warmstart_theta,
        )

        n_nn, n_nnn, n_q, p = 13, 15, 10, 2
        s = structural_warmstart_theta(n_nn, n_nnn, n_q, p, 0.5, J2=0.5)
        c = calibrated_warmstart_theta(n_nn, n_nnn, n_q, p, 0.5, J2=0.5)
        per = n_nn + n_nnn + n_q
        s_nnn = np.abs(s[n_nn : n_nn + n_nnn]).mean()
        c_nnn = np.abs(c[n_nn : n_nn + n_nnn]).mean()
        assert s_nnn < c_nnn  # structural suppresses the nnn block

    def test_theta_x_matches_calibrated(self):
        from qmbp_simulation.analysis.warmstart import (
            calibrated_warmstart_theta,
            structural_warmstart_theta,
        )

        n_nn, n_nnn, n_q, p = 13, 15, 10, 2
        s = structural_warmstart_theta(n_nn, n_nnn, n_q, p, 0.5, J2=0.5)
        c = calibrated_warmstart_theta(n_nn, n_nnn, n_q, p, 0.5, J2=0.5)
        # θ_x block (same calibrated x_scale) must match
        np.testing.assert_allclose(
            s[n_nn + n_nnn : n_nn + n_nnn + n_q], c[n_nn + n_nnn : n_nn + n_nnn + n_q], atol=1e-9
        )

    def test_appears_in_combined_cascade(self):
        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        r = best_combined_warmstart(n_nn=13, n_nnn=15, n_qubits=10, p_layers=2, h=0.5, J2=0.5, fid_fn=lambda t: 0.5)
        assert "structural" in [x["label"] for x in r["report"]]

    def test_toggle_off_excludes_structural(self):
        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        r = best_combined_warmstart(
            n_nn=13, n_nnn=15, n_qubits=10, p_layers=2, h=0.5, J2=0.5, fid_fn=lambda t: 0.5, include_structural=False
        )
        assert "structural" not in [x["label"] for x in r["report"]]


class TestEnsembleDonorSeed:
    """ensemble_donor_seed: circular mean of same-phase donors; include_ensemble."""

    def _donor(self, h, *, n_nn=N_NN, n_nnn=N_NNN, n_q=N_QUBITS, p=P, fill=0.1):
        from qmbp_simulation.analysis.warmstart import select_regime_seed

        base, _ = select_regime_seed(n_nn, n_nnn, n_q, p, h, J=J, J2=J2)
        return {"theta": base + fill, "n_nn": n_nn, "n_nnn": n_nnn, "p": p, "n_qubits": n_q, "h": h}

    def test_none_when_too_few_same_phase(self):
        from qmbp_simulation.analysis.warmstart import ensemble_donor_seed

        # Only one near-h_c donor → below min_donors=2 → None.
        out = ensemble_donor_seed(
            [self._donor(0.5)], n_nn=N_NN, n_nnn=N_NNN, n_qubits=N_QUBITS, p_layers=P, h=0.5, J=J, J2=J2
        )
        assert out is None

    def test_pools_only_same_phase(self):
        from qmbp_simulation.analysis.warmstart import ensemble_donor_seed

        # Two near-h_c donors + one paramagnet donor; target near-h_c pools 2.
        donors = [self._donor(0.5), self._donor(0.6), self._donor(1.4)]
        out = ensemble_donor_seed(donors, n_nn=N_NN, n_nnn=N_NNN, n_qubits=N_QUBITS, p_layers=P, h=0.5, J=J, J2=J2)
        assert out is not None
        seed, k = out
        assert k == 2
        assert seed.shape[0] == (N_NN + N_NNN + N_QUBITS) * P
        assert np.all(np.isfinite(seed)) and np.all(np.abs(seed) <= np.pi + 1e-9)

    def test_circular_mean_handles_wrap(self):
        from qmbp_simulation.analysis.warmstart import _circular_mean

        # +π and −π average to ±π (not 0) under circular mean.
        m = _circular_mean([np.array([np.pi - 1e-6]), np.array([-np.pi + 1e-6])])
        assert abs(abs(m[0]) - np.pi) < 1e-3

    def test_mean_of_identical_donors_is_identity(self):
        from qmbp_simulation.analysis.warmstart import ensemble_donor_seed

        d = self._donor(0.5)
        out = ensemble_donor_seed(
            [d, dict(d)], n_nn=N_NN, n_nnn=N_NNN, n_qubits=N_QUBITS, p_layers=P, h=0.5, J=J, J2=J2
        )
        assert out is not None
        seed, k = out
        assert k == 2
        # Mean of two identical transferred seeds equals that seed (wrapped).
        from qmbp_simulation.analysis.warmstart import transfer_theta

        single = transfer_theta(
            d["theta"],
            donor_n_nn=d["n_nn"],
            donor_n_nnn=d["n_nnn"],
            donor_p=d["p"],
            target_n_nn=N_NN,
            target_n_nnn=N_NNN,
            target_p=P,
            n_qubits=N_QUBITS,
            canonicalize=True,
            donor_n_qubits=d["n_qubits"],
        )
        np.testing.assert_allclose(np.sin(seed), np.sin(single), atol=1e-6)
        np.testing.assert_allclose(np.cos(seed), np.cos(single), atol=1e-6)

    def test_include_ensemble_toggle_in_cascade(self):
        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        donors = [self._donor(0.5), self._donor(0.6)]
        on = best_combined_warmstart(
            n_nn=N_NN,
            n_nnn=N_NNN,
            n_qubits=N_QUBITS,
            p_layers=P,
            h=0.5,
            J=J,
            J2=J2,
            donors=donors,
            fid_fn=lambda t: 0.5,
            include_ensemble=True,
        )
        off = best_combined_warmstart(
            n_nn=N_NN,
            n_nnn=N_NNN,
            n_qubits=N_QUBITS,
            p_layers=P,
            h=0.5,
            J=J,
            J2=J2,
            donors=donors,
            fid_fn=lambda t: 0.5,
            include_ensemble=False,
        )
        on_labels = [x["label"] for x in on["report"]]
        off_labels = [x["label"] for x in off["report"]]
        assert any(l.startswith("ensemble<") for l in on_labels)
        assert not any(l.startswith("ensemble<") for l in off_labels)


class TestBlockMixWarmstart:
    """block_source_distances + block_mix_warmstart + per-phase policy."""

    def _sources(self):
        from qmbp_simulation.analysis.warmstart import (
            calibrated_warmstart_theta,
            select_regime_seed,
            structural_warmstart_theta,
        )

        cal = calibrated_warmstart_theta(N_NN, N_NNN, N_QUBITS, P, 0.3, J2=J2)
        struct = structural_warmstart_theta(N_NN, N_NNN, N_QUBITS, P, 0.3, J2=J2)
        reg, _ = select_regime_seed(N_NN, N_NNN, N_QUBITS, P, 0.3, J=J, J2=J2)
        return {"calibrated": cal, "structural": struct, "regime": reg}

    def test_policy_is_phase_gated(self):
        from qmbp_simulation.analysis.warmstart import block_mix_policy_for

        assert block_mix_policy_for(0.3)["nnn"] == "structural"  # ordered
        assert block_mix_policy_for(0.5)["x"] == "regime"  # near_hc
        assert block_mix_policy_for(1.3)["nnn"] == "regime"  # paramag
        # nn is calibrated in every phase
        for h in (0.3, 0.5, 1.3):
            assert block_mix_policy_for(h)["nn"] == "calibrated"

    def test_mix_takes_each_block_from_its_source(self):
        from qmbp_simulation.analysis.warmstart import block_mix_warmstart

        src = self._sources()
        policy = {"nn": "calibrated", "nnn": "structural", "x": "regime"}
        mix = block_mix_warmstart(src, policy, n_nn=N_NN, n_nnn=N_NNN, n_qubits=N_QUBITS, p_layers=P)
        assert mix is not None and mix.size == (N_NN + N_NNN + N_QUBITS) * P
        for layer in range(P):
            s_nn, s_nnn, s_x = _slices(layer)
            np.testing.assert_allclose(mix[s_nn], src["calibrated"][s_nn], atol=1e-9)
            np.testing.assert_allclose(mix[s_nnn], src["structural"][s_nnn], atol=1e-9)
            np.testing.assert_allclose(mix[s_x], src["regime"][s_x], atol=1e-9)

    def test_missing_source_falls_back_to_default(self):
        from qmbp_simulation.analysis.warmstart import block_mix_warmstart

        src = self._sources()  # no "donor"
        policy = {"nn": "calibrated", "nnn": "structural", "x": "donor"}
        mix = block_mix_warmstart(
            src, policy, n_nn=N_NN, n_nnn=N_NNN, n_qubits=N_QUBITS, p_layers=P, default_source="calibrated"
        )
        assert mix is not None
        s_nn, s_nnn, s_x = _slices(0)
        # x block had missing "donor" → falls back to calibrated
        np.testing.assert_allclose(mix[s_x], src["calibrated"][s_x], atol=1e-9)

    def test_mix_returns_none_without_usable_sources(self):
        from qmbp_simulation.analysis.warmstart import block_mix_warmstart

        bad = {"calibrated": np.zeros(3)}  # wrong length
        mix = block_mix_warmstart(bad, {"nn": "calibrated"}, n_nn=N_NN, n_nnn=N_NNN, n_qubits=N_QUBITS, p_layers=P)
        assert mix is None

    def test_oracle_identifies_closest_source_per_block(self):
        from qmbp_simulation.analysis.warmstart import block_source_distances

        src = self._sources()
        # Build a synthetic θ_opt that equals structural in nnn and calibrated
        # elsewhere → oracle must pick structural for nnn, calibrated for nn/x.
        opt = np.array(src["calibrated"], float, copy=True)
        s_nn, s_nnn, s_x = _slices(0)
        for layer in range(P):
            a, b, _c = _slices(layer)
            opt[b] = src["structural"][b]
        rep = block_source_distances(opt, src, n_nn=N_NN, n_nnn=N_NNN, n_qubits=N_QUBITS, p_layers=P)
        assert rep["per_block"][(0, "nnn")]["best"] == "structural"
        assert rep["per_block"][(0, "nn")]["best"] == "calibrated"

    def test_block_mix_off_by_default(self):
        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        # Validated negative result → off by default (fidelity not block-separable).
        r = best_combined_warmstart(
            n_nn=N_NN, n_nnn=N_NNN, n_qubits=N_QUBITS, p_layers=P, h=0.3, J=J, J2=J2, fid_fn=lambda t: 0.5
        )
        assert not any(x["label"].startswith("block_mix<") for x in r["report"])

    def test_block_mix_toggle_on(self):
        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        r = best_combined_warmstart(
            n_nn=N_NN,
            n_nnn=N_NNN,
            n_qubits=N_QUBITS,
            p_layers=P,
            h=0.3,
            J=J,
            J2=J2,
            fid_fn=lambda t: 0.5,
            include_block_mix=True,
        )
        assert any(x["label"].startswith("block_mix<") for x in r["report"])


class TestBlockCoordinateDescent:
    """block_indices + block_coordinate_descent (coupled per-block refinement)."""

    def test_block_indices_partition_and_cover(self):
        from qmbp_simulation.analysis.warmstart import block_indices

        idx = block_indices(N_NN, N_NNN, N_QUBITS, P)
        allidx = np.concatenate([idx["nn"], idx["nnn"], idx["x"]])
        # cover every parameter exactly once
        assert sorted(allidx.tolist()) == list(range((N_NN + N_NNN + N_QUBITS) * P))
        assert idx["nn"].size == N_NN * P
        assert idx["nnn"].size == N_NNN * P
        assert idx["x"].size == N_QUBITS * P

    def test_descent_moves_only_active_block_per_step(self):
        # Quadratic objective toward a fixed target; a 1-iter perfect minimizer
        # over the active block must set that block to target, leaving the rest.
        from qmbp_simulation.analysis.warmstart import block_coordinate_descent

        npar = (N_NN + N_NNN + N_QUBITS) * P
        target = np.linspace(-0.5, 0.5, npar)

        def cost(x):
            return float(np.sum((x - target) ** 2))

        def grad(x):
            return 2.0 * (np.asarray(x, float) - target)

        def exact_lbfgsb(c, x0, *, maxiter, grad=None):
            # exact 1D-per-coord minimizer of the reduced quadratic: x = target_sub
            # recovered by a Newton step since Hessian is 2I.
            g = grad(x0)
            x = np.asarray(x0, float) - 0.5 * g  # x0 - H^{-1} g, H=2I
            return x, c(x), 1

        theta0 = np.zeros(npar)
        out = block_coordinate_descent(
            theta0,
            cost,
            grad,
            n_nn=N_NN,
            n_nnn=N_NNN,
            n_qubits=N_QUBITS,
            p_layers=P,
            lbfgsb=exact_lbfgsb,
            order=("nnn", "nn", "x"),
            sweeps=1,
        )
        # after one sweep over all three blocks, every coord hit its target
        np.testing.assert_allclose(out, target, atol=1e-9)

    def test_does_not_mutate_input(self):
        from qmbp_simulation.analysis.warmstart import block_coordinate_descent

        npar = (N_NN + N_NNN + N_QUBITS) * P

        def cost(x):
            return float(np.sum(x**2))

        def grad(x):
            return 2.0 * np.asarray(x, float)

        def lbfgsb(c, x0, *, maxiter, grad=None):
            return np.asarray(x0, float) * 0.5, c(x0), 1

        theta0 = np.ones(npar)
        snapshot = theta0.copy()
        _ = block_coordinate_descent(
            theta0, cost, grad, n_nn=N_NN, n_nnn=N_NNN, n_qubits=N_QUBITS, p_layers=P, lbfgsb=lbfgsb
        )
        np.testing.assert_array_equal(theta0, snapshot)  # input untouched

    def test_monotone_nonincreasing_cost(self):
        # Each block step can only lower (or hold) the full-vector cost.
        from qmbp_simulation.analysis.warmstart import block_coordinate_descent

        npar = (N_NN + N_NNN + N_QUBITS) * P
        rng = np.random.default_rng(0)
        A = rng.normal(size=(npar,)) ** 2 + 0.1  # positive curvature per coord
        target = rng.normal(size=npar) * 0.3

        def cost(x):
            return float(np.sum(A * (x - target) ** 2))

        def grad(x):
            return 2.0 * A * (np.asarray(x, float) - target)

        def lbfgsb(c, x0, *, maxiter, grad=None):
            # Backtracking gradient descent on the reduced objective — guaranteed
            # non-increasing (true line search), independent of curvature scale.
            x = np.asarray(x0, float)
            for _ in range(maxiter):
                g = grad(x)
                f0 = c(x)
                step = 1.0
                while step > 1e-6:
                    xn = x - step * g
                    if c(xn) <= f0:
                        x = xn
                        break
                    step *= 0.5
            return x, c(x), maxiter

        theta0 = rng.normal(size=npar)
        c0 = cost(theta0)
        out = block_coordinate_descent(
            theta0,
            cost,
            grad,
            n_nn=N_NN,
            n_nnn=N_NNN,
            n_qubits=N_QUBITS,
            p_layers=P,
            lbfgsb=lbfgsb,
            sweeps=2,
            maxiter_per_block=5,
        )
        assert cost(out) <= c0 + 1e-9


class TestMicroDescentBudgetScaling:
    """micro_descent_budget: N-scaling in the transition, floor elsewhere."""

    def test_small_N_at_floor_every_phase(self):
        from qmbp_simulation.analysis.warmstart import (
            MICRO_DESCENT_MIN,
            micro_descent_budget,
        )

        # N <= N0 stays at the floor at every h (seed already suffices)
        for h in (0.3, 0.5, 1.0, 1.3, 1.8):
            assert micro_descent_budget(8, h) == MICRO_DESCENT_MIN
            assert micro_descent_budget(10, h) == MICRO_DESCENT_MIN

    def test_floor_grows_with_N_even_off_transition(self):
        from qmbp_simulation.analysis.warmstart import micro_descent_budget

        # h=1.0 / paramagnet: phase_proximity≈0, but the N floor still grows so
        # large N is not starved (the old bug). Monotone non-decreasing in N.
        for h in (1.0, 1.3, 1.8):
            b10 = micro_descent_budget(10, h)
            b14 = micro_descent_budget(14, h)
            b18 = micro_descent_budget(18, h)
            assert b10 <= b14 <= b18
            assert b14 > b10  # N14 gets more than the bare floor

    def test_scales_with_N_in_transition(self):
        from qmbp_simulation.analysis.warmstart import micro_descent_budget

        # at h_c the budget grows strictly with N (super-linear)
        b10 = micro_descent_budget(10, 0.5)
        b12 = micro_descent_budget(12, 0.5)
        b14 = micro_descent_budget(14, 0.5)
        assert b10 < b12 < b14
        # super-linear: the N12→N14 jump exceeds the N10→N12 jump
        assert (b14 - b12) > (b12 - b10)

    def test_clamped_to_max(self):
        from qmbp_simulation.analysis.warmstart import (
            MICRO_DESCENT_MAX,
            micro_descent_budget,
        )

        # very large N at h_c saturates at the ceiling (never unbounded)
        assert micro_descent_budget(40, 0.5) == MICRO_DESCENT_MAX
        assert micro_descent_budget(100, 0.5) == MICRO_DESCENT_MAX

    def test_within_bounds_always(self):
        from qmbp_simulation.analysis.warmstart import (
            MICRO_DESCENT_MAX,
            MICRO_DESCENT_MIN,
            micro_descent_budget,
        )

        for n in (4, 8, 10, 12, 14, 18, 24, 40):
            for h in (0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.8, 2.5):
                b = micro_descent_budget(n, h)
                assert MICRO_DESCENT_MIN <= b <= MICRO_DESCENT_MAX


class TestCascadeEfficiencyImprovements:
    """M2 (two-pass early-exit), M3 (dedup), M4 (per-degree θ_x fill)."""

    def _descent_stub(self, target):
        """A fake descent_fn: returns (theta, fidelity) where fidelity is the
        negative L2 distance to a fixed `target` (closer seed → higher 'fid')."""
        import numpy as np

        def _fn(theta):
            th = np.asarray(theta, float)
            return th, -float(np.linalg.norm(th - target))

        return _fn

    def test_m3_dedup_shares_descent_keeps_report(self):
        import numpy as np

        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        # Two identical extra candidates: BOTH stay in the report (semantics
        # unchanged), but the duplicate reuses the representative's descent — the
        # descent_fn is called once for the shared vector, not twice.
        dup = np.full(9, 0.3)
        calls = {"n": 0}

        def desc(theta):
            calls["n"] += 1
            return np.asarray(theta, float), -float(np.linalg.norm(theta))

        res = best_combined_warmstart(
            n_nn=2,
            n_nnn=1,
            n_qubits=3,
            p_layers=1,
            h=0.5,
            include_calibrated=False,
            include_structural=False,
            include_regime=True,
            extra_candidates=[(dup, "dupA"), (dup, "dupB")],
            descent_fn=desc,
            target_len=9,
        )
        labels = [r["label"] for r in res["report"]]
        assert "dupA" in labels and "dupB" in labels  # both kept in report
        # regime + dupA + dupB = 3 candidates, but dupA==dupB share one descent,
        # so descent_fn runs at most twice (regime + the shared dup vector).
        assert calls["n"] <= 2

    def test_m2_two_pass_equals_full_when_topk_covers_all(self):
        import numpy as np

        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        desc = self._descent_stub(target)
        extras = [(np.full(9, v), f"c{v}") for v in (0.1, 0.4, 0.7)]
        common = dict(
            n_nn=2,
            n_nnn=1,
            n_qubits=3,
            p_layers=1,
            h=0.5,
            include_calibrated=False,
            include_structural=False,
            include_regime=True,
            extra_candidates=extras,
            descent_fn=desc,
            target_len=9,
        )
        full = best_combined_warmstart(**common)
        # two-pass with top_k >= n_candidates must pick the SAME winner.
        tp = best_combined_warmstart(**common, descent_fn_short=desc, two_pass_top_k=10)
        assert tp["provenance"] == full["provenance"]

    def test_m2_two_pass_off_by_default(self):
        import numpy as np

        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        # Without descent_fn_short the two-pass path is inert (back-compat).
        target = np.zeros(9)
        res = best_combined_warmstart(
            n_nn=2,
            n_nnn=1,
            n_qubits=3,
            p_layers=1,
            h=0.5,
            include_calibrated=False,
            include_structural=False,
            include_regime=True,
            extra_candidates=[(np.full(9, 0.2), "a"), (np.full(9, 0.8), "b")],
            descent_fn=self._descent_stub(target),
            target_len=9,
        )
        assert res["provenance"].endswith("+descent")

    def test_m4_grid_degree_corner_edge_bulk(self):
        from qmbp_simulation.analysis.warmstart import _grid_degree, lattice_coords

        # 3x3 grid (N=9): center z=4, edges z=3, corners z=2.
        deg = _grid_degree(lattice_coords("square", 9))
        counts = sorted(deg.values())
        assert counts.count(4) == 1  # one bulk (center)
        assert counts.count(2) == 4  # four corners
        assert counts.count(3) == 4  # four edges

    def test_m4_grid_degree_none_without_coords(self):
        from qmbp_simulation.analysis.warmstart import _grid_degree

        assert _grid_degree(None) is None

    def test_prerank_dedups_short_descent(self):
        # fix 1: the short pre-rank runs ONCE per distinct candidate; duplicates
        # inherit the representative's score (no redundant short descent).
        import numpy as np

        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        dup = np.full(9, 0.3)
        short_calls = {"n": 0}

        def short(theta):
            short_calls["n"] += 1
            return np.asarray(theta, float), float(np.mean(theta))

        def full(theta):
            return np.asarray(theta, float), -float(np.linalg.norm(theta))

        best_combined_warmstart(
            n_nn=2,
            n_nnn=1,
            n_qubits=3,
            p_layers=1,
            h=0.5,
            include_calibrated=False,
            include_structural=False,
            include_regime=False,
            extra_candidates=[(dup, "A"), (dup, "B"), (np.full(9, -0.5), "C")],
            descent_fn=full,
            descent_fn_short=short,
            two_pass_top_k=2,
            target_len=9,
        )
        # 3 candidates but A==B share one vector → 2 distinct short descents.
        assert short_calls["n"] == 2

    def test_warm_restart_full_off_by_default_equals_exhaustive(self):
        # fix 2 default OFF: two-pass with top_k >= n_candidates picks the SAME
        # winner as the exhaustive full-descent-from-raw-seed path.
        import numpy as np

        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        def mk(iters, lr=0.3):
            def fn(theta):
                x = np.asarray(theta, float).copy()
                attr = np.sign(x) * 0.8
                for _ in range(iters):
                    x = x - lr * 2 * (x - attr)
                return x, float(1.0 / (1.0 + np.linalg.norm(x - attr)) + 0.01 * np.mean(x))

            return fn

        full, short = mk(400), mk(100)
        extras = [(np.full(9, v), f"c{v}") for v in (0.1, 0.9, -0.7, 0.5, 0.3)]
        common = dict(
            n_nn=2,
            n_nnn=1,
            n_qubits=3,
            p_layers=1,
            h=0.5,
            include_calibrated=False,
            include_structural=False,
            include_regime=False,
            extra_candidates=extras,
            descent_fn=full,
            target_len=9,
        )
        exhaustive = best_combined_warmstart(**common)
        two_pass = best_combined_warmstart(**common, descent_fn_short=short, two_pass_top_k=5)  # top_k=all
        assert two_pass["provenance"] == exhaustive["provenance"]

    def test_warm_restart_full_opt_in_reuses_short(self):
        # fix 2 ON: the full descent starts from the short-refined θ. With a
        # convergent descent this never lowers the reached fidelity.
        import numpy as np

        from qmbp_simulation.analysis.warmstart import best_combined_warmstart

        target = np.linspace(-0.5, 0.9, 9)

        def mk(iters, lr=0.3):
            def fn(theta):
                x = np.asarray(theta, float).copy()
                for _ in range(iters):
                    x = x - lr * 2 * (x - target)
                return x, -float(np.linalg.norm(x - target))

            return fn

        full, short = mk(400), mk(100)
        extras = [(np.full(9, v), f"c{v}") for v in (0.1, 0.5, -0.2)]
        common = dict(
            n_nn=2,
            n_nnn=1,
            n_qubits=3,
            p_layers=1,
            h=0.5,
            include_calibrated=False,
            include_structural=False,
            include_regime=False,
            extra_candidates=extras,
            descent_fn=full,
            descent_fn_short=short,
            two_pass_top_k=2,
            target_len=9,
        )
        off = best_combined_warmstart(**common)
        on = best_combined_warmstart(**common, warm_restart_full=True)
        assert on["init_fidelity"] >= off["init_fidelity"] - 1e-9


class TestShortDescentBudget:
    """short_descent_budget: the two-pass (M1+M2) pre-rank budget."""

    def test_fraction_of_full(self):
        from qmbp_simulation.analysis.warmstart import short_descent_budget

        # Default 0.25 of a large full budget, well above the floor.
        assert short_descent_budget(400) == 100
        assert short_descent_budget(400, frac=0.5) == 200

    def test_floored_at_min(self):
        from qmbp_simulation.analysis.warmstart import (
            SHORT_DESCENT_MIN,
            short_descent_budget,
        )

        # A small full budget would give frac·full < min; the floor kicks in.
        assert short_descent_budget(40) == SHORT_DESCENT_MIN
        assert short_descent_budget(10) == min(10, SHORT_DESCENT_MIN)

    def test_never_exceeds_full(self):
        from qmbp_simulation.analysis.warmstart import short_descent_budget

        # Even a huge frac is clamped to the full budget (short <= full).
        for full in (24, 60, 200, 400):
            assert short_descent_budget(full, frac=5.0) <= full

    def test_cheaper_than_full_at_transition(self):
        from qmbp_simulation.analysis.warmstart import (
            micro_descent_budget,
            short_descent_budget,
        )

        # At the transition the full budget is large (~400); the short pre-rank
        # must be strictly cheaper so the two-pass actually saves work.
        full = micro_descent_budget(18, 0.5)
        assert short_descent_budget(full) < full


class TestAnalyticSeedCache:
    """M5: cached analytic seeds — same values, no cross-config contamination."""

    @pytest.fixture(autouse=True)
    def _clean_caches(self):
        # Module-level caches: start each test from empty for determinism.
        from qmbp_simulation.analysis.warmstart import clear_analytic_seed_caches

        clear_analytic_seed_caches()
        yield
        clear_analytic_seed_caches()

    def _layout(self):
        return dict(n_nn=7, n_nnn=6, n_qubits=6, p_layers=2)

    def test_cached_matches_pure_calibrated(self):
        import numpy as np

        from qmbp_simulation.analysis.warmstart import (
            cached_calibrated_warmstart_theta,
            calibrated_warmstart_theta,
        )

        lo = self._layout()
        pure = calibrated_warmstart_theta(**lo, h=0.5, J2=0.5)
        cached = cached_calibrated_warmstart_theta(**lo, h=0.5, J2=0.5)
        np.testing.assert_array_equal(pure, cached)

    def test_cached_matches_pure_structural(self):
        import numpy as np

        from qmbp_simulation.analysis.warmstart import (
            cached_structural_warmstart_theta,
            structural_warmstart_theta,
        )

        lo = self._layout()
        pure = structural_warmstart_theta(**lo, h=0.5, J2=0.5)
        cached = cached_structural_warmstart_theta(**lo, h=0.5, J2=0.5)
        np.testing.assert_array_equal(pure, cached)

    def test_cached_matches_pure_regime(self):
        import numpy as np

        from qmbp_simulation.analysis.warmstart import (
            cached_select_regime_seed,
            select_regime_seed,
        )

        lo = self._layout()
        ps, pn = select_regime_seed(**lo, h=0.5, J2=0.5)
        cs, cn = cached_select_regime_seed(**lo, h=0.5, J2=0.5)
        np.testing.assert_array_equal(ps, cs)
        assert pn == cn

    def test_no_collision_across_h(self):
        import numpy as np

        from qmbp_simulation.analysis.warmstart import cached_calibrated_warmstart_theta

        lo = self._layout()
        a = cached_calibrated_warmstart_theta(**lo, h=0.5, J2=0.5)
        b = cached_calibrated_warmstart_theta(**lo, h=0.7, J2=0.5)
        assert not np.array_equal(a, b)

    def test_no_collision_across_close_h_exact_bits(self):
        # float.hex() keying: 0.501 and 0.502 must NOT alias (a :.2f key would).
        import numpy as np

        from qmbp_simulation.analysis.warmstart import cached_calibrated_warmstart_theta

        lo = self._layout()
        a = cached_calibrated_warmstart_theta(**lo, h=0.501, J2=0.5)
        b = cached_calibrated_warmstart_theta(**lo, h=0.502, J2=0.5)
        assert not np.array_equal(a, b)

    def test_no_collision_across_j2(self):
        import numpy as np

        from qmbp_simulation.analysis.warmstart import cached_calibrated_warmstart_theta

        lo = self._layout()
        a = cached_calibrated_warmstart_theta(**lo, h=0.5, J2=0.5)
        b = cached_calibrated_warmstart_theta(**lo, h=0.5, J2=0.0)
        assert not np.array_equal(a, b)

    def test_no_collision_across_nnn_suppress(self):
        import numpy as np

        from qmbp_simulation.analysis.warmstart import cached_structural_warmstart_theta

        lo = self._layout()
        a = cached_structural_warmstart_theta(**lo, h=0.5, J2=0.5, nnn_suppress=0.35)
        b = cached_structural_warmstart_theta(**lo, h=0.5, J2=0.5, nnn_suppress=0.1)
        assert not np.array_equal(a, b)

    def test_no_collision_across_use_calibrated(self):
        import numpy as np

        from qmbp_simulation.analysis.warmstart import cached_select_regime_seed

        lo = self._layout()
        a, _ = cached_select_regime_seed(**lo, h=0.5, J2=0.5, use_calibrated=False)
        b, _ = cached_select_regime_seed(**lo, h=0.5, J2=0.5, use_calibrated=True)
        assert not np.array_equal(a, b)

    def test_returned_array_is_independent_copy(self):
        # Mutating the returned seed must NOT corrupt the cached entry.
        import numpy as np

        from qmbp_simulation.analysis.warmstart import (
            cached_calibrated_warmstart_theta,
            clear_analytic_seed_caches,
        )

        clear_analytic_seed_caches()
        lo = self._layout()
        v1 = cached_calibrated_warmstart_theta(**lo, h=0.5, J2=0.5)
        reference = v1.copy()
        v1[:] = 999.0
        v2 = cached_calibrated_warmstart_theta(**lo, h=0.5, J2=0.5)
        np.testing.assert_array_equal(v2, reference)
        assert not np.any(v2 == 999.0)

    def test_hit_does_not_duplicate_entry(self):
        from qmbp_simulation.analysis.warmstart import (
            _CALIBRATED_SEED_CACHE,
            cached_calibrated_warmstart_theta,
            clear_analytic_seed_caches,
        )

        clear_analytic_seed_caches()
        lo = self._layout()
        cached_calibrated_warmstart_theta(**lo, h=0.5, J2=0.5)
        cached_calibrated_warmstart_theta(**lo, h=0.5, J2=0.5)
        assert len(_CALIBRATED_SEED_CACHE) == 1

    def test_fifo_bound_respected(self):
        from qmbp_simulation.analysis.warmstart import (
            _ANALYTIC_SEED_CACHE_MAX,
            _CALIBRATED_SEED_CACHE,
            cached_calibrated_warmstart_theta,
            clear_analytic_seed_caches,
        )

        clear_analytic_seed_caches()
        lo = self._layout()
        for i in range(_ANALYTIC_SEED_CACHE_MAX + 10):
            cached_calibrated_warmstart_theta(**lo, h=0.1 + 0.001 * i, J2=0.5)
        assert len(_CALIBRATED_SEED_CACHE) == _ANALYTIC_SEED_CACHE_MAX

    def test_cached_matches_pure_variant(self):
        import numpy as np

        from qmbp_simulation.analysis.warmstart import (
            cached_variant_warmstart_theta,
            variant_warmstart_theta,
        )

        blocks = ["nn", "nnn", "x", "nn", "x"]
        pure = variant_warmstart_theta(blocks, 7, 6, 6, 0.5, J2=0.5, rx_final=True)
        cached = cached_variant_warmstart_theta(blocks, 7, 6, 6, 0.5, J2=0.5, rx_final=True)
        np.testing.assert_array_equal(pure, cached)

    def test_variant_no_collision_across_blocks(self):
        from qmbp_simulation.analysis.warmstart import cached_variant_warmstart_theta

        a = cached_variant_warmstart_theta(["nn", "nnn", "x"], 7, 6, 6, 0.5, J2=0.5)
        b = cached_variant_warmstart_theta(["nn", "x"], 7, 6, 6, 0.5, J2=0.5)
        # Different block sequence → different length → never aliased.
        assert a.size != b.size

    def test_variant_no_collision_across_rx_final(self):
        from qmbp_simulation.analysis.warmstart import cached_variant_warmstart_theta

        blocks = ["nn", "nnn", "x"]
        a = cached_variant_warmstart_theta(blocks, 7, 6, 6, 0.5, J2=0.5, rx_final=True)
        b = cached_variant_warmstart_theta(blocks, 7, 6, 6, 0.5, J2=0.5, rx_final=False)
        assert a.size != b.size

    def test_variant_returned_is_independent_copy(self):
        import numpy as np

        from qmbp_simulation.analysis.warmstart import (
            cached_variant_warmstart_theta,
            clear_analytic_seed_caches,
        )

        clear_analytic_seed_caches()
        blocks = ["nn", "nnn", "x"]
        v1 = cached_variant_warmstart_theta(blocks, 7, 6, 6, 0.5, J2=0.5)
        ref = v1.copy()
        v1[:] = 999.0
        v2 = cached_variant_warmstart_theta(blocks, 7, 6, 6, 0.5, J2=0.5)
        np.testing.assert_array_equal(v2, ref)
