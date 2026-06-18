"""Unit tests for apply_affine_on_raw in the mitigation benchmark runner.

Validates:
  - Affine correction applied when config.affine_enabled=True AND zne_method=None
  - Result unchanged when config.affine_enabled=False
  - Result unchanged when config.zne_method is not None
  - e_mitigated is set in ResultEnvelope (execution_result dict)
  - H8: affine never worsens — corrected energy is at least as close to e_exact
"""

from scripts.experiment_runners.hardware.benchmark_configs import (
    BenchmarkConfig,
)
from scripts.experiment_runners.hardware.run_mitigation_benchmark import (
    apply_affine_on_raw,
)


class TestApplyAffineOnRaw:
    """Tests for apply_affine_on_raw helper."""

    def test_applies_when_affine_enabled_and_no_zne(self):
        """Affine correction applied when affine_enabled=True, zne_method=None."""
        config = BenchmarkConfig(
            config_id="C1_dd_only",
            dd_enabled=True,
            dd_sequence="XpXm",
            affine_enabled=True,
            priority=1,
        )
        # e_raw is below e_exact (overshoot below ground state)
        execution_result = {"e_raw": -15.0, "e_mitigated": None}
        e_exact = -12.0
        e_upper = 5.0

        result = apply_affine_on_raw(config, execution_result, e_exact, e_upper)

        # e_mitigated should be set (not None anymore)
        assert result["e_mitigated"] is not None

    def test_noop_when_affine_disabled(self):
        """No correction when affine_enabled=False."""
        config = BenchmarkConfig(
            config_id="C0_raw",
            affine_enabled=False,
            priority=0,
            n_layouts=1,
        )
        execution_result = {"e_raw": -15.0, "e_mitigated": None}
        e_exact = -12.0
        e_upper = 5.0

        result = apply_affine_on_raw(config, execution_result, e_exact, e_upper)

        # e_mitigated should remain unchanged (None)
        assert result["e_mitigated"] is None

    def test_noop_when_zne_method_set(self):
        """No correction when zne_method is not None (ZNE handles it)."""
        config = BenchmarkConfig(
            config_id="C5_full_pea_balanced",
            dd_enabled=True,
            dd_sequence="XpXm",
            twirling_num_randomizations=48,
            trex_enabled=True,
            zne_method="pea",
            pea_num_randomizations=48,
            pea_shots_per_randomization=192,
            affine_enabled=True,
            priority=0,
        )
        execution_result = {"e_raw": -15.0, "e_mitigated": -11.5}
        e_exact = -12.0
        e_upper = 5.0

        result = apply_affine_on_raw(config, execution_result, e_exact, e_upper)

        # e_mitigated should remain at the ZNE value, not be overwritten
        assert result["e_mitigated"] == -11.5

    def test_e_mitigated_within_bounds(self):
        """H8: affine-corrected energy stays within [e_exact, e_upper]."""
        config = BenchmarkConfig(
            config_id="C2_dd_tw",
            dd_enabled=True,
            dd_sequence="XpXm",
            twirling_num_randomizations=32,
            trex_enabled=True,
            affine_enabled=True,
            priority=1,
        )
        e_exact = -12.0
        e_upper = 5.0

        # Test with e_raw within bounds
        execution_result = {"e_raw": -8.0, "e_mitigated": None}
        result = apply_affine_on_raw(config, execution_result, e_exact, e_upper)
        assert e_exact <= result["e_mitigated"] <= e_upper

    def test_affine_never_worsens_h8(self):
        """H8: affine correction never increases |e - e_exact| when in bounds."""
        config = BenchmarkConfig(
            config_id="C1_dd_only",
            dd_enabled=True,
            dd_sequence="XpXm",
            affine_enabled=True,
            priority=1,
        )
        e_exact = -12.0
        e_upper = 5.0

        # e_raw already in bounds — correction should not worsen it
        e_raw = -10.5
        execution_result = {"e_raw": e_raw, "e_mitigated": None}
        result = apply_affine_on_raw(config, execution_result, e_exact, e_upper)

        error_raw = abs(e_raw - e_exact)
        error_mitigated = abs(result["e_mitigated"] - e_exact)
        assert error_mitigated <= error_raw

    def test_returns_same_dict_reference(self):
        """Function modifies and returns the same dict (in-place update)."""
        config = BenchmarkConfig(
            config_id="C1_dd_only",
            dd_enabled=True,
            dd_sequence="XpXm",
            affine_enabled=True,
            priority=1,
        )
        execution_result = {"e_raw": -10.0, "e_mitigated": None}
        e_exact = -12.0
        e_upper = 5.0

        result = apply_affine_on_raw(config, execution_result, e_exact, e_upper)
        assert result is execution_result
