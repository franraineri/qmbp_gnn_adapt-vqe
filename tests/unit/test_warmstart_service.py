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
    return (slice(o, o + N_NN),
            slice(o + N_NN, o + N_NN + N_NNN),
            slice(o + N_NN + N_NNN, o + PER))


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
        np.testing.assert_allclose(th[:PER], th[PER:2 * PER])

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
            N_NN, N_NNN, N_QUBITS, P, h, J=J, J2=J2,
            shrink_coef=DEFAULT_SHRINK_COEF, curv_coef=DEFAULT_CURV_COEF,
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
        second = second_order_warmstart_theta(
            N_NN, N_NNN, N_QUBITS, P, h, J=J, J2=J2, shrink_coef=0.0, curv_coef=0.0
        )
        np.testing.assert_allclose(second, first)

    def test_shrink_reduces_zz_magnitude_near_transition(self):
        h = 0.5
        first = first_order_warmstart_theta(N_NN, N_NNN, N_QUBITS, P, h, J=J, J2=J2)
        second = second_order_warmstart_theta(N_NN, N_NNN, N_QUBITS, P, h, J=J, J2=J2)
        nn, _, _ = _slices(0)
        # ZZ angles are negative; shrink pulls them toward zero => smaller |.|.
        assert np.abs(second[nn]).max() < np.abs(first[nn]).max()

    def test_correction_magnitude_grows_as_h_falls(self):
        assert (second_order_correction_magnitude(0.3)
                > second_order_correction_magnitude(0.5)
                > second_order_correction_magnitude(0.7))


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
