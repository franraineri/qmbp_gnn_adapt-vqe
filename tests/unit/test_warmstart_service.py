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
