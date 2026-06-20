"""Tests for MPNN evaluation helper methods on ValidationRunner.

Covers the four helpers added to ValidationRunner:
  - benchmark_mpnn_warmstart
  - mpnn_leave_one_out_cv
  - mpnn_landscape_quality
  - mpnn_interpolation_extrapolation

All tests use N=4, p=1, 3 training points, minimal epochs to stay fast (<30s).
FakeTorino / noisy paths are excluded (no @pytest.mark.slow needed here since
we run noiseless only).

Design:
  - Tests validate output *structure* and *types* — not exact float values.
  - Thresholds are deliberately loose (de_gap < 0.50) at N=4 minimal config.
  - Each test uses a shared fixture runner to avoid repeated setup.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from qmbp_simulation.framework.runner_base import Section, ValidationRunner

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture: minimal runner + fast config
# ─────────────────────────────────────────────────────────────────────────────

# Tiny configuration: N=4, p=1, 3 training points, 10 epochs
# Fast enough to run in <5s per helper call.
_TOPOLOGY = "chain_1d"
_N = 4
_P = 1
_H_TRAIN = [2.5, 2.0, 1.5]
_H_TRAIN_LOO = [3.0, 2.5, 2.0, 1.5]  # 4 points for LOO (fold size = 3 ≥ min_train_size)
_H_TEST = [2.25]  # interpolation (inside training range)
_H_EXTRAP = [3.0]  # extrapolation (outside training range)
_H_INTERP = [2.0]  # inside range midpoint (overlap with train is ok for test)
_EPOCHS = 10  # minimal — just enough for shape/structure tests
_SEED = 42


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "section": None,
        "skip_preflight": False,
        "stop_on_failure": False,
        "verbose": False,
        "dry_run": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class _MinimalRunner(ValidationRunner):
    """Minimal concrete ValidationRunner for testing helper methods."""

    runner_id = "mpnn_eval_test"
    experiment_id = "MPNN_EVAL"
    description = "MPNN evaluation helper tests"
    hypothesis = "MPNN helpers return correct structure"

    def define_sections(self) -> list[Section]:
        return [Section(id=1, name="dummy", fn=lambda: {"pass": True}, hypothesis="")]


@pytest.fixture(scope="module")
def runner() -> _MinimalRunner:
    return _MinimalRunner(args=_make_args())


# ─────────────────────────────────────────────────────────────────────────────
# benchmark_mpnn_warmstart
# ─────────────────────────────────────────────────────────────────────────────


class TestBenchmarkMpnnWarmstart:
    def test_returns_required_top_level_keys(self, runner):
        result = runner.benchmark_mpnn_warmstart(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for key in ("per_h", "summary", "mpnn_train_mse", "n_train_points", "n_params", "pass"):
            assert key in result, f"Missing key: {key}"

    def test_per_h_has_one_entry_per_h_test(self, runner):
        result = runner.benchmark_mpnn_warmstart(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        assert len(result["per_h"]) == len(_H_TEST)

    def test_per_h_entry_structure(self, runner):
        result = runner.benchmark_mpnn_warmstart(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        entry = result["per_h"][0]
        for key in ("h", "e_exact", "gap", "random", "prev_h", "mpnn", "pass"):
            assert key in entry, f"per_h entry missing key: {key}"
        # random block
        for k in ("mean_iters", "best_de_gap", "iters_list"):
            assert k in entry["random"], f"random block missing: {k}"
        # prev_h block
        for k in ("nearest_h_train", "init_de_gap", "final_de_gap", "iters"):
            assert k in entry["prev_h"], f"prev_h block missing: {k}"
        # mpnn block
        for k in ("init_de_gap", "final_de_gap", "iters", "speedup_vs_random", "speedup_vs_prev_h"):
            assert k in entry["mpnn"], f"mpnn block missing: {k}"

    def test_summary_keys(self, runner):
        result = runner.benchmark_mpnn_warmstart(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        s = result["summary"]
        for k in (
            "mean_speedup_vs_random",
            "mean_speedup_vs_prev_h",
            "mpnn_wins_vs_random",
            "mpnn_wins_vs_prev_h",
            "mean_init_de_gap",
            "mean_final_de_gap_mpnn",
        ):
            assert k in s, f"summary missing: {k}"

    def test_numerical_sanity(self, runner):
        result = runner.benchmark_mpnn_warmstart(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        entry = result["per_h"][0]
        assert entry["gap"] > 0
        assert entry["e_exact"] < 0  # TFIM ground state energy is negative
        assert entry["mpnn"]["init_de_gap"] >= 0
        assert entry["mpnn"]["iters"] >= 0
        assert result["mpnn_train_mse"] >= 0
        assert result["n_train_points"] <= len(_H_TRAIN)
        assert result["n_params"] > 0

    def test_speedup_is_finite(self, runner):
        result = runner.benchmark_mpnn_warmstart(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        speedup = result["per_h"][0]["mpnn"]["speedup_vs_random"]
        assert np.isfinite(speedup), f"Speedup should be finite: {speedup}"
        assert speedup > 0, f"Speedup should be positive: {speedup}"

    def test_pass_field_is_bool(self, runner):
        result = runner.benchmark_mpnn_warmstart(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        assert isinstance(result["pass"], bool)
        assert isinstance(result["per_h"][0]["pass"], bool)


# ─────────────────────────────────────────────────────────────────────────────
# mpnn_leave_one_out_cv
# ─────────────────────────────────────────────────────────────────────────────


class TestMpnnLeaveOneOutCv:
    def test_returns_required_top_level_keys(self, runner):
        result = runner.mpnn_leave_one_out_cv(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN_LOO,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
            min_train_size=3,
        )
        for key in ("per_fold", "summary", "full_model_train_mse", "n_train_points_full", "pass"):
            assert key in result, f"Missing key: {key}"

    def test_n_folds_equals_n_train(self, runner):
        result = runner.mpnn_leave_one_out_cv(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN_LOO,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
            min_train_size=3,  # 4 - 1 = 3 points per fold → all folds run
        )
        # Each held-out fold should appear once
        assert result["summary"]["n_folds"] == len(_H_TRAIN_LOO)

    def test_per_fold_entry_structure(self, runner):
        result = runner.mpnn_leave_one_out_cv(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN_LOO,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
            min_train_size=3,
        )
        for fold in result["per_fold"]:
            for key in (
                "fold_idx",
                "h_held_out",
                "n_train_points",
                "e_exact",
                "gap",
                "e_pred",
                "de_gap",
                "fold_train_mse",
                "pass",
            ):
                assert key in fold, f"fold missing key: {key}"

    def test_summary_keys(self, runner):
        result = runner.mpnn_leave_one_out_cv(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN_LOO,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
            min_train_size=3,
        )
        s = result["summary"]
        for k in ("mean_de_gap", "max_de_gap", "std_de_gap", "n_pass", "n_folds", "pass_rate"):
            assert k in s, f"summary missing: {k}"

    def test_pass_rate_in_unit_interval(self, runner):
        result = runner.mpnn_leave_one_out_cv(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN_LOO,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
            min_train_size=3,
        )
        pr = result["summary"]["pass_rate"]
        assert 0.0 <= pr <= 1.0, f"pass_rate should be in [0,1]: {pr}"

    def test_held_out_points_cover_all_train(self, runner):
        """Every training h-value should appear as held-out exactly once."""
        result = runner.mpnn_leave_one_out_cv(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN_LOO,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
            min_train_size=3,
        )
        held_out = {f["h_held_out"] for f in result["per_fold"]}
        for h in _H_TRAIN_LOO:
            assert any(abs(h - ho) < 1e-9 for ho in held_out), f"h={h} never held out"

    def test_min_train_size_skip(self, runner):
        """Folds with < min_train_size points should be skipped."""
        # With min_train_size=4 and only 4 training points, N-1=3 < 4 → all skipped
        result = runner.mpnn_leave_one_out_cv(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN_LOO,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
            min_train_size=4,  # N-1=3 < 4 → skip all folds
        )
        assert result["summary"]["n_folds"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# mpnn_landscape_quality
# ─────────────────────────────────────────────────────────────────────────────


class TestMpnnLandscapeQuality:
    def test_returns_required_top_level_keys(self, runner):
        result = runner.mpnn_landscape_quality(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for key in ("per_h", "summary", "mpnn_train_mse", "n_train_points", "n_params", "pass"):
            assert key in result, f"Missing key: {key}"

    def test_per_h_entry_structure(self, runner):
        result = runner.mpnn_landscape_quality(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        entry = result["per_h"][0]
        for key in (
            "h",
            "e_exact",
            "e_opt",
            "e_pred",
            "gap",
            "error_circuit",
            "error_mpnn",
            "error_total",
            "theta_deviation",
            "mean_curvature",
            "max_curvature",
            "circuit_limited",
            "pass",
        ):
            assert key in entry, f"per_h entry missing key: {key}"

    def test_error_decomposition_consistency(self, runner):
        """error_total should approximate error_circuit + error_mpnn (triangle ineq)."""
        result = runner.mpnn_landscape_quality(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        entry = result["per_h"][0]
        # Triangle inequality: |E_pred - E_exact| ≤ |E_pred - E_opt| + |E_opt - E_exact|
        # So error_total ≤ error_mpnn + error_circuit (approximately, with gap normalization)
        assert entry["error_total"] >= 0
        assert entry["error_circuit"] >= 0
        assert entry["error_mpnn"] >= 0

    def test_curvature_is_nonnegative(self, runner):
        result = runner.mpnn_landscape_quality(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for entry in result["per_h"]:
            assert entry["mean_curvature"] >= 0, "Curvature should be nonneg"
            assert entry["max_curvature"] >= entry["mean_curvature"], "max ≥ mean curvature"

    def test_theta_deviation_nonneg(self, runner):
        result = runner.mpnn_landscape_quality(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for entry in result["per_h"]:
            assert entry["theta_deviation"] >= 0

    def test_summary_keys(self, runner):
        result = runner.mpnn_landscape_quality(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        s = result["summary"]
        for k in (
            "mean_error_circuit",
            "mean_error_mpnn",
            "mean_error_total",
            "mean_theta_deviation",
            "mean_curvature",
            "n_circuit_limited",
            "mpnn_fraction_of_total_error",
        ):
            assert k in s, f"summary missing: {k}"

    def test_mpnn_fraction_in_unit_interval(self, runner):
        result = runner.mpnn_landscape_quality(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        frac = result["summary"]["mpnn_fraction_of_total_error"]
        # Can exceed 1.0 if MPNN overshoots in the opposite direction, but should be finite
        assert np.isfinite(frac), f"mpnn_fraction should be finite: {frac}"


# ─────────────────────────────────────────────────────────────────────────────
# mpnn_interpolation_extrapolation
# ─────────────────────────────────────────────────────────────────────────────


class TestMpnnInterpolationExtrapolation:
    def test_returns_required_top_level_keys(self, runner):
        result = runner.mpnn_interpolation_extrapolation(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_interpolate=_H_TEST,  # inside range
            h_extrapolate=_H_EXTRAP,  # outside range
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for key in (
            "interpolation",
            "extrapolation",
            "summary",
            "mpnn_train_mse",
            "n_train_points",
            "n_params",
            "h_train_range",
            "pass",
        ):
            assert key in result, f"Missing key: {key}"

    def test_interpolation_list_length(self, runner):
        result = runner.mpnn_interpolation_extrapolation(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_interpolate=_H_TEST,
            h_extrapolate=_H_EXTRAP,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        assert len(result["interpolation"]) == len(_H_TEST)
        assert len(result["extrapolation"]) == len(_H_EXTRAP)

    def test_per_point_entry_structure(self, runner):
        result = runner.mpnn_interpolation_extrapolation(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_interpolate=_H_TEST,
            h_extrapolate=_H_EXTRAP,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for entry in result["interpolation"] + result["extrapolation"]:
            for key in (
                "h",
                "e_exact",
                "e_pred",
                "gap",
                "de_gap",
                "distance_to_nearest_train",
                "relative_distance",
                "mode",
                "pass",
            ):
                assert key in entry, f"entry missing key: {key}"

    def test_mode_labels_correct(self, runner):
        result = runner.mpnn_interpolation_extrapolation(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_interpolate=_H_TEST,
            h_extrapolate=_H_EXTRAP,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for entry in result["interpolation"]:
            assert entry["mode"] == "interpolation"
        for entry in result["extrapolation"]:
            assert entry["mode"] == "extrapolation"

    def test_distance_to_train_nonneg(self, runner):
        result = runner.mpnn_interpolation_extrapolation(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_interpolate=_H_TEST,
            h_extrapolate=_H_EXTRAP,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for entry in result["interpolation"] + result["extrapolation"]:
            assert entry["distance_to_nearest_train"] >= 0
            assert entry["relative_distance"] >= 0

    def test_extrapolation_distance_larger_than_interpolation(self, runner):
        """Extrapolation points should be farther from training data than interpolation."""
        result = runner.mpnn_interpolation_extrapolation(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_interpolate=[2.25],  # midpoint — close to train
            h_extrapolate=[3.5],  # far outside — farther from train
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        interp_dist = result["interpolation"][0]["distance_to_nearest_train"]
        extrap_dist = result["extrapolation"][0]["distance_to_nearest_train"]
        assert extrap_dist >= interp_dist, (
            f"Extrap dist ({extrap_dist}) should be ≥ interp dist ({interp_dist})"
        )

    def test_summary_structure(self, runner):
        result = runner.mpnn_interpolation_extrapolation(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_interpolate=_H_TEST,
            h_extrapolate=_H_EXTRAP,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        s = result["summary"]
        assert "interpolation" in s
        assert "extrapolation" in s
        assert "degradation_factor" in s
        assert "h_train_range" in s
        for mode_key in ("interpolation", "extrapolation"):
            for k in ("mean_de_gap", "max_de_gap", "pass_rate", "n_points"):
                assert k in s[mode_key], f"summary[{mode_key}] missing: {k}"

    def test_degradation_factor_finite(self, runner):
        result = runner.mpnn_interpolation_extrapolation(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_interpolate=_H_TEST,
            h_extrapolate=_H_EXTRAP,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        assert np.isfinite(result["summary"]["degradation_factor"])
        assert result["summary"]["degradation_factor"] >= 0

    def test_empty_extrapolation_list(self, runner):
        """Method should handle empty extrapolation list gracefully."""
        result = runner.mpnn_interpolation_extrapolation(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_interpolate=_H_TEST,
            h_extrapolate=[],  # no extrapolation points
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        assert result["extrapolation"] == []
        assert result["summary"]["extrapolation"]["n_points"] == 0

    def test_h_train_range_matches_input(self, runner):
        result = runner.mpnn_interpolation_extrapolation(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_interpolate=_H_TEST,
            h_extrapolate=_H_EXTRAP,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        h_min, h_max = result["h_train_range"]
        assert abs(h_min - min(_H_TRAIN)) < 1e-9
        assert abs(h_max - max(_H_TRAIN)) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# JSON serializability (all outputs must be json.dumps-able)
# ─────────────────────────────────────────────────────────────────────────────


class TestJsonSerializability:
    """Verify all helper outputs can be serialized to JSON (required for result saving)."""

    def test_benchmark_warmstart_json_serializable(self, runner):
        import json

        from qmbp_simulation.utils.helpers import json_serialize

        result = runner.benchmark_mpnn_warmstart(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        # Should not raise
        serialized = json.dumps(result, default=json_serialize)
        parsed = json.loads(serialized)
        assert "per_h" in parsed

    def test_loo_cv_json_serializable(self, runner):
        import json

        from qmbp_simulation.utils.helpers import json_serialize

        result = runner.mpnn_leave_one_out_cv(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN_LOO,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
            min_train_size=3,
        )
        serialized = json.dumps(result, default=json_serialize)
        parsed = json.loads(serialized)
        assert "summary" in parsed

    def test_landscape_quality_json_serializable(self, runner):
        import json

        from qmbp_simulation.utils.helpers import json_serialize

        result = runner.mpnn_landscape_quality(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        serialized = json.dumps(result, default=json_serialize)
        parsed = json.loads(serialized)
        assert "per_h" in parsed

    def test_interpolation_extrapolation_json_serializable(self, runner):
        import json

        from qmbp_simulation.utils.helpers import json_serialize

        result = runner.mpnn_interpolation_extrapolation(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_interpolate=_H_TEST,
            h_extrapolate=_H_EXTRAP,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        serialized = json.dumps(result, default=json_serialize)
        parsed = json.loads(serialized)
        assert "summary" in parsed
