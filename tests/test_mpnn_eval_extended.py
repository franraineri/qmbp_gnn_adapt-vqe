"""Tests for extended MPNN evaluation helpers (sections 15-19).

Covers:
  - mpnn_scaling_with_system_size
  - mpnn_learning_curve
  - mpnn_topology_transfer
  - mpnn_data_efficiency_vs_loo
  - mpnn_curvature_noise_correlation

All tests use N=4, minimal epochs (10), minimal restarts.
Module scope fixture reuses one runner instance for all classes.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from qmbp_simulation.framework.runner_base import Section, ValidationRunner
from qmbp_simulation.utils.helpers import json_serialize

# ─────────────────────────────────────────────────────────────────────────────
# Shared minimal config
# ─────────────────────────────────────────────────────────────────────────────

_TOPOLOGY = "chain_1d"
_N = 4
_P = 1
_H_TRAIN = [2.5, 2.0, 1.5]
_H_TRAIN_4 = [3.0, 2.5, 2.0, 1.5]  # 4 pts for LOO (fold size ≥ 3)
_H_TEST = [2.25]
_EPOCHS = 10
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
    runner_id = "mpnn_ext_test"
    experiment_id = "MPNN_EXT"
    description = "Extended MPNN eval tests"
    hypothesis = "All helpers return correct structure"

    def define_sections(self):
        return [Section(id=1, name="dummy", fn=lambda: {"pass": True}, hypothesis="")]


@pytest.fixture(scope="module")
def runner() -> _MinimalRunner:
    return _MinimalRunner(args=_make_args())


# ─────────────────────────────────────────────────────────────────────────────
# Section 15: mpnn_scaling_with_system_size
# ─────────────────────────────────────────────────────────────────────────────


class TestMpnnScalingWithN:
    def test_top_level_keys(self, runner):
        result = runner.mpnn_scaling_with_system_size(
            topology=_TOPOLOGY,
            system_sizes=[4],
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for k in ("per_n", "summary", "scaling_trend", "pass"):
            assert k in result, f"Missing: {k}"

    def test_per_n_has_one_entry_per_size(self, runner):
        result = runner.mpnn_scaling_with_system_size(
            topology=_TOPOLOGY,
            system_sizes=[4],
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        assert len(result["per_n"]) == 1
        entry = result["per_n"][0]
        for k in (
            "n_qubits",
            "n_params",
            "speedup_vs_random",
            "init_de_gap",
            "final_de_gap",
            "train_mse",
            "pass",
        ):
            assert k in entry

    def test_scaling_trend_valid_value(self, runner):
        result = runner.mpnn_scaling_with_system_size(
            topology=_TOPOLOGY,
            system_sizes=[4],
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        assert result["scaling_trend"] in ("increasing", "flat", "decreasing")

    def test_summary_keys(self, runner):
        result = runner.mpnn_scaling_with_system_size(
            topology=_TOPOLOGY,
            system_sizes=[4],
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for k in ("mean_speedup", "min_speedup", "max_speedup", "speedup_slope_per_N"):
            assert k in result["summary"]

    def test_json_serializable(self, runner):
        result = runner.mpnn_scaling_with_system_size(
            topology=_TOPOLOGY,
            system_sizes=[4],
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        json.dumps(result, default=json_serialize)


# ─────────────────────────────────────────────────────────────────────────────
# Section 16: mpnn_learning_curve
# ─────────────────────────────────────────────────────────────────────────────


class TestMpnnLearningCurve:
    def test_top_level_keys(self, runner):
        result = runner.mpnn_learning_curve(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_pool=_H_TRAIN_4,
            h_test=[2.25],
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for k in ("per_size", "train_sizes_tested", "h_pool_size", "summary", "pass"):
            assert k in result

    def test_per_size_structure(self, runner):
        result = runner.mpnn_learning_curve(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_pool=_H_TRAIN_4,
            h_test=[2.25],
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for e in result["per_size"]:
            for k in ("train_size", "mean_de_gap", "max_de_gap", "pass_rate", "train_mse", "pass"):
                assert k in e

    def test_train_sizes_ascending(self, runner):
        result = runner.mpnn_learning_curve(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_pool=_H_TRAIN_4,
            h_test=[2.25],
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        sizes = result["train_sizes_tested"]
        assert sizes == sorted(sizes), "Train sizes should be ascending"

    def test_de_gap_nonneg(self, runner):
        result = runner.mpnn_learning_curve(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_pool=_H_TRAIN_4,
            h_test=[2.25],
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for e in result["per_size"]:
            assert e["mean_de_gap"] >= 0
            assert 0.0 <= e["pass_rate"] <= 1.0

    def test_summary_structure(self, runner):
        result = runner.mpnn_learning_curve(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_pool=_H_TRAIN_4,
            h_test=[2.25],
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        s = result["summary"]
        # critical_size can be None (if no k achieves 80% pass rate with 10 epochs)
        assert "critical_size" in s
        assert np.isfinite(s["sample_efficiency_slope"])

    def test_json_serializable(self, runner):
        result = runner.mpnn_learning_curve(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_pool=_H_TRAIN_4,
            h_test=[2.25],
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        json.dumps(result, default=json_serialize)


# ─────────────────────────────────────────────────────────────────────────────
# Section 17: mpnn_topology_transfer
# ─────────────────────────────────────────────────────────────────────────────


class TestMpnnTopologyTransfer:
    def test_top_level_keys(self, runner):
        result = runner.mpnn_topology_transfer(
            source_topology=_TOPOLOGY,
            target_topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for k in ("zero_shot", "in_distribution", "summary", "pass"):
            assert k in result

    def test_summary_keys(self, runner):
        result = runner.mpnn_topology_transfer(
            source_topology=_TOPOLOGY,
            target_topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for k in (
            "mean_de_gap_zero_shot",
            "mean_de_gap_in_distribution",
            "mean_de_gap_random",
            "transfer_ratio",
            "zero_shot_pass_rate",
            "in_dist_pass_rate",
        ):
            assert k in result["summary"]

    def test_transfer_ratio_positive(self, runner):
        result = runner.mpnn_topology_transfer(
            source_topology=_TOPOLOGY,
            target_topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        ratio = result["summary"]["transfer_ratio"]
        assert ratio > 0 and np.isfinite(ratio)

    def test_per_h_result_structure(self, runner):
        result = runner.mpnn_topology_transfer(
            source_topology=_TOPOLOGY,
            target_topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for r in result["zero_shot"] + result["in_distribution"]:
            for k in ("h", "e_exact", "e_pred", "gap", "de_gap", "de_gap_random", "pass"):
                assert k in r

    def test_json_serializable(self, runner):
        result = runner.mpnn_topology_transfer(
            source_topology=_TOPOLOGY,
            target_topology=_TOPOLOGY,
            n_qubits=_N,
            h_train=_H_TRAIN,
            h_test=_H_TEST,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        json.dumps(result, default=json_serialize)


# ─────────────────────────────────────────────────────────────────────────────
# Section 18: mpnn_data_efficiency_vs_loo
# ─────────────────────────────────────────────────────────────────────────────


class TestMpnnDataEfficiencyVsLoo:
    def test_top_level_keys(self, runner):
        result = runner.mpnn_data_efficiency_vs_loo(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_pool=_H_TRAIN_4,
            n_seeds=2,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for k in ("per_seed", "per_fold_stats", "summary", "robust", "pass"):
            assert k in result

    def test_n_seeds_matches_per_seed(self, runner):
        result = runner.mpnn_data_efficiency_vs_loo(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_pool=_H_TRAIN_4,
            n_seeds=2,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        assert len(result["per_seed"]) == 2

    def test_summary_keys(self, runner):
        result = runner.mpnn_data_efficiency_vs_loo(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_pool=_H_TRAIN_4,
            n_seeds=2,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for k in (
            "mean_pass_rate",
            "std_pass_rate",
            "cv_pass_rate",
            "mean_de_gap",
            "std_de_gap",
            "n_seeds",
        ):
            assert k in result["summary"]

    def test_pass_rates_in_unit_interval(self, runner):
        result = runner.mpnn_data_efficiency_vs_loo(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_pool=_H_TRAIN_4,
            n_seeds=2,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        s = result["summary"]
        assert 0.0 <= s["mean_pass_rate"] <= 1.0
        assert s["std_pass_rate"] >= 0.0

    def test_per_fold_stats_have_h_values(self, runner):
        result = runner.mpnn_data_efficiency_vs_loo(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_pool=_H_TRAIN_4,
            n_seeds=2,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        for f in result["per_fold_stats"]:
            assert "h" in f
            assert "mean_de_gap" in f
            assert "std_de_gap" in f
            assert "cv" in f
            assert f["std_de_gap"] >= 0

    def test_json_serializable(self, runner):
        result = runner.mpnn_data_efficiency_vs_loo(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_pool=_H_TRAIN_4,
            n_seeds=2,
            p_layers=_P,
            seed=_SEED,
            mpnn_epochs=_EPOCHS,
            de_gap_threshold=0.50,
        )
        json.dumps(result, default=json_serialize)


# ─────────────────────────────────────────────────────────────────────────────
# Section 19: mpnn_curvature_noise_correlation
# ─────────────────────────────────────────────────────────────────────────────


class TestMpnnCurvatureNoiseCorrelation:
    def test_top_level_keys(self, runner):
        result = runner.mpnn_curvature_noise_correlation(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_grid=_H_TRAIN,
            p_layers=_P,
            seed=_SEED,
            noise_levels=[0.01, 0.10],
            de_gap_threshold=0.50,
        )
        for k in ("per_h", "correlations", "noise_levels", "summary", "pass"):
            assert k in result

    def test_per_h_structure(self, runner):
        result = runner.mpnn_curvature_noise_correlation(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_grid=_H_TRAIN,
            p_layers=_P,
            seed=_SEED,
            noise_levels=[0.01],
            de_gap_threshold=0.50,
        )
        for entry in result["per_h"]:
            for k in ("h", "kappa", "e_opt", "e_exact", "gap", "de_gap_opt", "noise_sensitivity"):
                assert k in entry
            assert entry["kappa"] >= 0
            assert "0.01" in entry["noise_sensitivity"]

    def test_correlations_per_sigma(self, runner):
        sigmas = [0.01, 0.10]
        result = runner.mpnn_curvature_noise_correlation(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_grid=_H_TRAIN,
            p_layers=_P,
            seed=_SEED,
            noise_levels=sigmas,
            de_gap_threshold=0.50,
        )
        for s in sigmas:
            assert str(s) in result["correlations"]

    def test_pearson_r_in_valid_range(self, runner):
        result = runner.mpnn_curvature_noise_correlation(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_grid=_H_TRAIN,
            p_layers=_P,
            seed=_SEED,
            noise_levels=[0.01, 0.10],
            de_gap_threshold=0.50,
        )
        for sigma, r_val in result["correlations"].items():
            if not np.isnan(r_val):
                assert -1.0 <= r_val <= 1.0, f"r={r_val} out of [-1,1] for σ={sigma}"

    def test_summary_keys(self, runner):
        result = runner.mpnn_curvature_noise_correlation(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_grid=_H_TRAIN,
            p_layers=_P,
            seed=_SEED,
            noise_levels=[0.01],
            de_gap_threshold=0.50,
        )
        for k in ("mean_kappa", "max_kappa", "mean_pearson_r", "kappa_is_reliable_predictor"):
            assert k in result["summary"]

    def test_pass_uses_absolute_value(self, runner):
        """Pass criterion should use |r| ≥ 0.70, not r ≥ 0.70."""
        result = runner.mpnn_curvature_noise_correlation(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_grid=_H_TRAIN,
            p_layers=_P,
            seed=_SEED,
            noise_levels=[0.01, 0.10],
            de_gap_threshold=0.50,
        )
        # pass should be True iff |mean_r| >= 0.70
        mean_r = result["summary"]["mean_pearson_r"]
        expected_pass = abs(mean_r) >= 0.70
        assert result["pass"] == expected_pass, (
            f"pass={result['pass']} but |mean_r|={abs(mean_r):.4f} "
            f"(threshold=0.70) → expected pass={expected_pass}"
        )

    def test_json_serializable(self, runner):
        result = runner.mpnn_curvature_noise_correlation(
            topology=_TOPOLOGY,
            n_qubits=_N,
            h_grid=_H_TRAIN,
            p_layers=_P,
            seed=_SEED,
            noise_levels=[0.01],
            de_gap_threshold=0.50,
        )
        json.dumps(result, default=json_serialize)


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer parse tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyzerParsers:
    """Test that analyzer parsers handle section data correctly."""

    def test_parse_scaling_with_n(self):
        from project_health.analysis.mpnn_eval_analyzer import parse_scaling_with_n

        fake = {
            "data": {
                "per_n": [
                    {
                        "n_qubits": 4,
                        "speedup_vs_random": 2.5,
                        "n_params": 2,
                        "init_de_gap": 0.01,
                        "final_de_gap": 0.005,
                        "train_mse": 1e-4,
                        "pass": True,
                    }
                ],
                "summary": {
                    "mean_speedup": 2.5,
                    "min_speedup": 2.5,
                    "max_speedup": 2.5,
                    "speedup_slope_per_N": 0.0,
                },
                "scaling_trend": "flat",
                "pass": True,
            }
        }
        result = parse_scaling_with_n(fake)
        assert result is not None
        assert result.mean_speedup == pytest.approx(2.5)
        assert result.scaling_trend == "flat"

    def test_parse_learning_curve(self):
        from project_health.analysis.mpnn_eval_analyzer import parse_learning_curve

        fake = {
            "data": {
                "per_size": [
                    {
                        "train_size": 3,
                        "mean_de_gap": 0.08,
                        "max_de_gap": 0.12,
                        "pass_rate": 0.5,
                        "train_mse": 1e-3,
                        "pass": False,
                    },
                    {
                        "train_size": 5,
                        "mean_de_gap": 0.03,
                        "max_de_gap": 0.04,
                        "pass_rate": 1.0,
                        "train_mse": 5e-4,
                        "pass": True,
                    },
                ],
                "train_sizes_tested": [3, 5],
                "h_pool_size": 7,
                "summary": {
                    "critical_size": 5,
                    "sample_efficiency_slope": -0.025,
                    "best_mean_de_gap": 0.03,
                    "full_dataset_de_gap": 0.03,
                },
                "pass": True,
            }
        }
        result = parse_learning_curve(fake)
        assert result is not None
        assert result.critical_size == 5
        assert result.sample_efficient  # critical_size=5 ≤ 10

    def test_parse_curvature_noise(self):
        from project_health.analysis.mpnn_eval_analyzer import parse_curvature_noise

        fake = {
            "data": {
                "summary": {
                    "mean_kappa": 45.0,
                    "max_kappa": 55.0,
                    "mean_pearson_r": -0.85,
                    "kappa_is_reliable_predictor": True,
                },
                "correlations": {"0.01": -0.82, "0.1": -0.88},
                "pass": True,
            }
        }
        result = parse_curvature_noise(fake)
        assert result is not None
        assert result.mean_pearson_r == pytest.approx(-0.85)
        assert result.kappa_is_reliable
        assert "negative_strong" in result.correlation_interpretation

    def test_parse_topology_transfer_skipped(self):
        from project_health.analysis.mpnn_eval_analyzer import parse_topology_transfer

        fake = {"data": {"skipped": True, "reason": "same topology", "pass": True}}
        result = parse_topology_transfer(fake)
        assert result is not None
        assert result.skipped
        assert result.pass_
        assert result.transfer_quality == "skipped"
