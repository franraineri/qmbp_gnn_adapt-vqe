"""Unit tests for affine_correct_energy — comprehensive coverage.

Covers the bug fix from 2026-06-22 where the "soft correction" formula
moved energy FURTHER from bounds instead of toward them (614× amplification).

Test categories:
1. Sub-ground-state inputs (the buggy branch)
2. Above-upper-bound inputs (the symmetric buggy branch)
3. In-bounds inputs (no correction)
4. Boundary conditions
5. Monotonicity property (correction NEVER worsens result)
6. Regression test for the exact FAIL run values
7. Edge cases (equal bounds, None parameters, etc.)

References:
- Bug report: scripts/verify_affine_bug.py
- Affected runs: run_20260616_154738, run_20260616_155721, run_20260617_141440
"""

import numpy as np
import pytest

from qmbp_simulation.execution import affine_correct_energy


class TestAffineCorrectionSubGroundState:
    """Tests for energy below e_ground (the fixed buggy branch)."""

    def test_barely_below_ground_clips_to_ground(self):
        """Energy 0.003 below ground → clips to e_ground (not amplifies)."""
        result = affine_correct_energy(
            mitigated_energy=-10.003,
            e_ground=-10.0,
            e_upper=5.0,
        )
        assert result.corrected_energy == -10.0
        assert result.correction_applied is True
        assert result.correction_magnitude == pytest.approx(0.003, abs=1e-10)

    def test_far_below_ground_clips_to_ground(self):
        """Energy far below ground → clips to e_ground."""
        result = affine_correct_energy(
            mitigated_energy=-20.0,
            e_ground=-10.0,
            e_upper=5.0,
        )
        assert result.corrected_energy == -10.0
        assert result.correction_applied is True

    def test_epsilon_below_ground_clips(self):
        """Tiny violation (1e-10) still clips to ground."""
        eps = 1e-10
        result = affine_correct_energy(
            mitigated_energy=-10.0 - eps,
            e_ground=-10.0,
            e_upper=5.0,
        )
        # Should clip — corrected >= e_ground always
        assert result.corrected_energy >= -10.0

    def test_sub_ground_never_moves_further_below(self):
        """REGRESSION: old bug moved energy further below. Verify fix."""
        # This is the exact scenario from the 614× amplification bug
        for violation in [0.001, 0.003, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
            e_ground = -33.198
            result = affine_correct_energy(
                mitigated_energy=e_ground - violation,
                e_ground=e_ground,
                e_upper=41.5,
            )
            # The corrected energy must be AT or ABOVE e_ground
            assert result.corrected_energy >= e_ground, (
                f"violation={violation}: corrected={result.corrected_energy} < "
                f"e_ground={e_ground} — energy moved further below!"
            )


class TestAffineCorrectionAboveUpperBound:
    """Tests for energy above e_upper (the symmetric fixed branch)."""

    def test_barely_above_upper_clips_to_upper(self):
        """Energy 0.003 above upper bound → clips to e_upper."""
        result = affine_correct_energy(
            mitigated_energy=5.003,
            e_ground=-10.0,
            e_upper=5.0,
        )
        assert result.corrected_energy == 5.0
        assert result.correction_applied is True

    def test_far_above_upper_clips_to_upper(self):
        """Energy far above upper → clips to e_upper."""
        result = affine_correct_energy(
            mitigated_energy=50.0,
            e_ground=-10.0,
            e_upper=5.0,
        )
        assert result.corrected_energy == 5.0
        assert result.correction_applied is True

    def test_above_upper_never_moves_further_above(self):
        """REGRESSION: symmetric bug — verify correction goes DOWN not UP."""
        for violation in [0.001, 0.01, 0.1, 1.0, 5.0]:
            e_upper = 5.0
            result = affine_correct_energy(
                mitigated_energy=e_upper + violation,
                e_ground=-10.0,
                e_upper=e_upper,
            )
            assert result.corrected_energy <= e_upper, (
                f"violation={violation}: corrected={result.corrected_energy} > "
                f"e_upper={e_upper} — energy moved further above!"
            )


class TestAffineCorrectionInBounds:
    """Tests for energy within [e_ground, e_upper] — no correction."""

    def test_in_bounds_no_correction(self):
        """Energy within bounds → no modification."""
        result = affine_correct_energy(
            mitigated_energy=-7.5,
            e_ground=-10.0,
            e_upper=5.0,
        )
        assert result.corrected_energy == -7.5
        assert result.correction_applied is False
        assert result.correction_magnitude == 0.0

    def test_at_exact_ground_no_correction(self):
        """Energy exactly at e_ground → no correction (not below)."""
        result = affine_correct_energy(
            mitigated_energy=-10.0,
            e_ground=-10.0,
            e_upper=5.0,
        )
        assert result.corrected_energy == -10.0
        assert result.correction_applied is False

    def test_at_exact_upper_no_correction(self):
        """Energy exactly at e_upper → no correction (not above)."""
        result = affine_correct_energy(
            mitigated_energy=5.0,
            e_ground=-10.0,
            e_upper=5.0,
        )
        assert result.corrected_energy == 5.0
        assert result.correction_applied is False


class TestAffineCorrectionMonotonicity:
    """Property: correction NEVER increases |E - E_closest_bound|."""

    @pytest.mark.parametrize("energy", np.linspace(-40, 50, 50).tolist())
    def test_monotonicity_sweep(self, energy: float):
        """For any input, correction moves energy toward or to a bound."""
        e_ground = -33.198
        e_upper = 41.5

        result = affine_correct_energy(energy, e_ground, e_upper=e_upper)

        if e_ground <= energy <= e_upper:
            # In bounds: should not change
            assert result.corrected_energy == energy
        elif energy < e_ground:
            # Below ground: corrected should be at e_ground (clipped up)
            assert result.corrected_energy == e_ground
        else:
            # Above upper: corrected should be at e_upper (clipped down)
            assert result.corrected_energy == e_upper

    def test_never_worsens_error_vs_exact(self):
        """H8 property: |corrected - e_exact| <= |input - e_exact| always."""
        e_exact = -10.0
        e_upper = 5.0

        # Test both sides of e_exact and within bounds
        test_energies = [-15, -12, -10.5, -10.001, -10.0, -9.5, -5, 0, 5, 7, 10]
        for e_input in test_energies:
            result = affine_correct_energy(e_input, e_exact, e_upper=e_upper)
            error_before = abs(e_input - e_exact)
            error_after = abs(result.corrected_energy - e_exact)
            assert error_after <= error_before + 1e-10, (
                f"e_input={e_input}: error increased from {error_before:.6f} to {error_after:.6f}"
            )


class TestAffineCorrectionRegressionRun141440:
    """Exact regression test for the FAIL run that exposed the bug."""

    def test_run_141440_values(self):
        """Reproduce exact run_20260617_141440 scenario."""
        # Values from the actual FAIL run
        e_zne = -33.20130666097003
        e_exact = -33.198273652500575
        gap = 4.429858148786295

        result = affine_correct_energy(
            mitigated_energy=e_zne,
            e_ground=e_exact,
            n_qubits=10,
            h_value=3.25,
        )

        # Must clip to e_ground (not produce -35.064)
        assert result.corrected_energy == e_exact
        assert result.correction_applied is True

        # Verify verdict would be PASS
        delta_e_gap = abs(result.corrected_energy - e_exact) / gap
        assert delta_e_gap < 0.05, f"Expected PASS (dE/gap < 5%), got {delta_e_gap * 100:.4f}%"

    def test_run_155721_values(self):
        """Reproduce run_20260616_155721 scenario (another affected run)."""
        # e_zne was barely below e_exact → old code amplified to 23%
        e_exact = -33.198273652500575
        gap = 4.429858148786295
        # This run had dE/gap = 0.0427% pre-affine
        violation = 0.0427 / 100 * gap  # reconstruct e_zne
        e_zne = e_exact - violation

        result = affine_correct_energy(
            mitigated_energy=e_zne,
            e_ground=e_exact,
            n_qubits=10,
            h_value=3.25,
        )

        assert result.corrected_energy == e_exact
        delta = abs(result.corrected_energy - e_exact) / gap
        assert delta < 0.05


class TestAffineCorrectionEdgeCases:
    """Edge cases and parameter estimation."""

    def test_none_e_upper_estimates_from_params(self):
        """When e_upper=None, estimates from n_qubits and h_value."""
        result = affine_correct_energy(
            mitigated_energy=-5.0,
            e_ground=-10.0,
            n_qubits=6,
            h_value=2.0,
        )
        # e_upper should be estimated as: |J|*(N-1) + |h|*N = 5 + 12 = 17
        assert result.upper_bound == pytest.approx(17.0)
        assert result.correction_applied is False  # -5 is within [-10, 17]

    def test_none_e_upper_no_params_uses_zero(self):
        """When e_upper=None and no params, falls back to 0."""
        result = affine_correct_energy(
            mitigated_energy=-5.0,
            e_ground=-10.0,
        )
        assert result.upper_bound == 0.0
        assert result.correction_applied is False  # -5 is within [-10, 0]

    def test_e_ground_equals_e_upper_skips(self):
        """When e_ground >= e_upper, no correction applied."""
        result = affine_correct_energy(
            mitigated_energy=-5.0,
            e_ground=0.0,
            e_upper=0.0,
        )
        assert result.correction_applied is False
        assert result.corrected_energy == -5.0

    def test_output_bounds_metadata(self):
        """Result contains correct bounds metadata."""
        result = affine_correct_energy(
            mitigated_energy=-12.0,
            e_ground=-10.0,
            e_upper=5.0,
        )
        assert result.lower_bound == -10.0
        assert result.upper_bound == 5.0
        assert result.original_energy == -12.0

    def test_large_negative_energy_clips(self):
        """Very large negative energy clips to ground."""
        result = affine_correct_energy(
            mitigated_energy=-1000.0,
            e_ground=-10.0,
            e_upper=5.0,
        )
        assert result.corrected_energy == -10.0

    def test_large_positive_energy_clips(self):
        """Very large positive energy clips to upper."""
        result = affine_correct_energy(
            mitigated_energy=1000.0,
            e_ground=-10.0,
            e_upper=5.0,
        )
        assert result.corrected_energy == 5.0
