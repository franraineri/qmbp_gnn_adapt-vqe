"""Tests for p=2 pipeline functionality.

Covers: tile_theta, load_theta_from_npz, MultiNAggregator p_layers,
AcceleratedConfig.resolve_for_p, EvalCache partitioning, validation functions,
and anti-regression guard.
"""
from __future__ import annotations

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# tile_theta_for_higher_p
# ═══════════════════════════════════════════════════════════════════════════════


class TestTileTheta:
    """Tests for tile_theta_for_higher_p utility."""

    def test_basic_p1_to_p2(self):
        from qmbp_simulation.utils.helpers import tile_theta_for_higher_p

        theta_p1 = np.array([0.1, -0.2, 0.3, 0.05, -0.1])
        result = tile_theta_for_higher_p(theta_p1, p_target=2, noise_std=0.0)
        assert result.shape == (10,)
        np.testing.assert_array_equal(result[:5], theta_p1)
        np.testing.assert_array_equal(result[5:], theta_p1)

    def test_p1_to_p3(self):
        from qmbp_simulation.utils.helpers import tile_theta_for_higher_p

        theta = np.ones(19)
        result = tile_theta_for_higher_p(theta, p_target=3, noise_std=0.0)
        assert result.shape == (57,)
        np.testing.assert_array_equal(result[:19], theta)
        np.testing.assert_array_equal(result[19:38], theta)
        np.testing.assert_array_equal(result[38:], theta)

    def test_p2_to_p4(self):
        from qmbp_simulation.utils.helpers import tile_theta_for_higher_p

        theta_p2 = np.random.randn(38)
        result = tile_theta_for_higher_p(theta_p2, p_target=4, p_source=2, noise_std=0.0)
        assert result.shape == (76,)
        np.testing.assert_array_equal(result[:38], theta_p2)
        np.testing.assert_array_equal(result[38:], theta_p2)

    def test_noise_extra_only(self):
        from qmbp_simulation.utils.helpers import tile_theta_for_higher_p

        theta = np.ones(19)
        result = tile_theta_for_higher_p(theta, p_target=2, noise_std=0.1, noise_layers="extra", seed=42)
        # First layer unchanged
        np.testing.assert_array_equal(result[:19], theta)
        # Second layer has noise
        assert not np.allclose(result[19:], theta)

    def test_noise_all(self):
        from qmbp_simulation.utils.helpers import tile_theta_for_higher_p

        theta = np.ones(19)
        result = tile_theta_for_higher_p(theta, p_target=2, noise_std=0.1, noise_layers="all", seed=42)
        # Both layers have noise
        assert not np.allclose(result[:19], theta)

    def test_deterministic_with_seed(self):
        from qmbp_simulation.utils.helpers import tile_theta_for_higher_p

        theta = np.random.randn(19)
        r1 = tile_theta_for_higher_p(theta, p_target=2, seed=123)
        r2 = tile_theta_for_higher_p(theta, p_target=2, seed=123)
        np.testing.assert_array_equal(r1, r2)

    def test_expected_n_params_pass(self):
        from qmbp_simulation.utils.helpers import tile_theta_for_higher_p

        theta = np.ones(19)
        result = tile_theta_for_higher_p(theta, p_target=2, expected_n_params=38, noise_std=0.0)
        assert result.shape == (38,)

    def test_expected_n_params_fail(self):
        from qmbp_simulation.utils.helpers import tile_theta_for_higher_p

        theta = np.ones(19)
        with pytest.raises(ValueError, match="circuit expects"):
            tile_theta_for_higher_p(theta, p_target=2, expected_n_params=40, noise_std=0.0)

    def test_p_target_lte_p_source_raises(self):
        from qmbp_simulation.utils.helpers import tile_theta_for_higher_p

        with pytest.raises(ValueError, match="must be >"):
            tile_theta_for_higher_p(np.ones(19), p_target=1, p_source=1)

    def test_empty_theta_raises(self):
        from qmbp_simulation.utils.helpers import tile_theta_for_higher_p

        with pytest.raises(ValueError, match="empty"):
            tile_theta_for_higher_p(np.array([]), p_target=2)

    def test_non_divisible_raises(self):
        from qmbp_simulation.utils.helpers import tile_theta_for_higher_p

        with pytest.raises(ValueError, match="not divisible"):
            tile_theta_for_higher_p(np.ones(19), p_target=4, p_source=3)

    def test_invalid_noise_layers_raises(self):
        from qmbp_simulation.utils.helpers import tile_theta_for_higher_p

        with pytest.raises(ValueError, match="noise_layers"):
            tile_theta_for_higher_p(np.ones(19), p_target=2, noise_layers="invalid")


# ═══════════════════════════════════════════════════════════════════════════════
# load_theta_from_npz
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadThetaFromNpz:
    """Tests for load_theta_from_npz and load_p1_theta_for_warmstart."""

    def test_load_existing_p1(self):
        from qmbp_simulation.utils.helpers import load_theta_from_npz

        result = load_theta_from_npz("chain_1d", 10, p_layers=1)
        if result is not None:  # Only if data exists
            assert isinstance(result, dict)
            assert all(isinstance(k, float) for k in result.keys())
            assert all(isinstance(v, np.ndarray) for v in result.values())
            # All thetas should be finite
            for v in result.values():
                assert np.all(np.isfinite(v))

    def test_load_nonexistent_returns_none(self):
        from qmbp_simulation.utils.helpers import load_theta_from_npz

        result = load_theta_from_npz("nonexistent_topo", 999, p_layers=1)
        assert result is None

    def test_load_p2_returns_none_when_no_data(self):
        from qmbp_simulation.utils.helpers import load_theta_from_npz

        # p=2 data shouldn't exist yet
        result = load_theta_from_npz("chain_1d", 10, p_layers=2)
        # May be None or have data depending on whether p2 runs have happened
        if result is not None:
            assert isinstance(result, dict)

    def test_h_values_filter(self):
        from qmbp_simulation.utils.helpers import load_theta_from_npz

        # Request specific h-values
        result = load_theta_from_npz("chain_1d", 10, p_layers=1, h_values=np.array([99.9]))
        assert result is None  # No match at h=99.9

    def test_load_p1_wrapper(self):
        from qmbp_simulation.utils.helpers import load_p1_theta_for_warmstart, load_theta_from_npz

        r1 = load_p1_theta_for_warmstart("chain_1d", 10)
        r2 = load_theta_from_npz("chain_1d", 10, p_layers=1)
        # Should return same data
        if r1 is not None and r2 is not None:
            assert set(r1.keys()) == set(r2.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# MultiNAggregator p_layers
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiNAggregatorPLayers:
    """Tests for MultiNAggregator p_layers parameterization."""

    def test_default_p_layers_is_1(self):
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        agg = MultiNAggregator(topology="chain_1d")
        assert agg.p_layers == 1

    def test_p_layers_2_accepted(self):
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        agg = MultiNAggregator(topology="chain_1d", p_layers=2)
        assert agg.p_layers == 2

    def test_p_layers_0_raises(self):
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        with pytest.raises(ValueError, match="p_layers must be >= 1"):
            MultiNAggregator(topology="chain_1d", p_layers=0)

    def test_p_layers_negative_raises(self):
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        with pytest.raises(ValueError, match="p_layers must be >= 1"):
            MultiNAggregator(topology="chain_1d", p_layers=-1)

    def test_multi_topology_aggregator_p_layers(self):
        from qmbp_simulation.predictors.multi_n_aggregator import MultiTopologyAggregator

        mt = MultiTopologyAggregator(p_layers=2)
        assert mt.p_layers == 2

    def test_multi_topology_aggregator_p_layers_invalid(self):
        from qmbp_simulation.predictors.multi_n_aggregator import MultiTopologyAggregator

        with pytest.raises(ValueError):
            MultiTopologyAggregator(p_layers=0)


# ═══════════════════════════════════════════════════════════════════════════════
# AcceleratedConfig.resolve_for_p
# ═══════════════════════════════════════════════════════════════════════════════


class TestAcceleratedConfigResolveForP:
    """Tests for AcceleratedConfig auto-optimization for p≥2."""

    def test_p1_no_changes(self):
        from qmbp_simulation.pipeline.accelerated import AcceleratedConfig

        cfg = AcceleratedConfig(n_anchors=10, maxiter=500)
        resolved = cfg.resolve_for_p(1, 19)
        assert resolved is cfg  # Same object, no copy

    def test_p2_enables_lbfgsb(self):
        from qmbp_simulation.pipeline.accelerated import AcceleratedConfig

        cfg = AcceleratedConfig()
        resolved = cfg.resolve_for_p(2, 38)
        assert resolved.force_method == "L-BFGS-B"

    def test_p2_reduces_maxiter(self):
        from qmbp_simulation.pipeline.accelerated import AcceleratedConfig

        cfg = AcceleratedConfig(maxiter=1500)
        resolved = cfg.resolve_for_p(2, 38)
        assert resolved.maxiter == 300

    def test_p2_reduces_anchors(self):
        from qmbp_simulation.pipeline.accelerated import AcceleratedConfig

        cfg = AcceleratedConfig(n_anchors=10)
        resolved = cfg.resolve_for_p(2, 38)
        assert resolved.n_anchors == 7  # 70% of 10

    def test_p2_reduces_restarts(self):
        from qmbp_simulation.pipeline.accelerated import AcceleratedConfig

        cfg = AcceleratedConfig(n_restarts=10)
        resolved = cfg.resolve_for_p(2, 38)
        assert resolved.n_restarts == 3  # 10 // 3

    def test_p2_sets_p1_informed_strategy(self):
        from qmbp_simulation.pipeline.accelerated import AcceleratedConfig

        cfg = AcceleratedConfig()
        resolved = cfg.resolve_for_p(2, 38)
        assert resolved.anchor_strategy == "p1_informed"

    def test_user_override_respected(self):
        from qmbp_simulation.pipeline.accelerated import AcceleratedConfig

        cfg = AcceleratedConfig(force_method="COBYLA", anchor_strategy="endpoints_plus_center")
        resolved = cfg.resolve_for_p(2, 38)
        assert resolved.force_method == "COBYLA"
        assert resolved.anchor_strategy == "endpoints_plus_center"

    def test_small_n_params_no_lbfgsb(self):
        from qmbp_simulation.pipeline.accelerated import AcceleratedConfig

        cfg = AcceleratedConfig()
        resolved = cfg.resolve_for_p(2, 15)  # Small circuit
        assert resolved.force_method is None

    def test_auto_optimize_disabled(self):
        from qmbp_simulation.pipeline.accelerated import AcceleratedConfig

        cfg = AcceleratedConfig(auto_optimize_for_higher_p=False)
        resolved = cfg.resolve_for_p(2, 38)
        assert resolved is cfg

    def test_immutability(self):
        from qmbp_simulation.pipeline.accelerated import AcceleratedConfig

        cfg = AcceleratedConfig(n_anchors=10, maxiter=1000)
        resolved = cfg.resolve_for_p(2, 38)
        assert cfg.n_anchors == 10  # Original unchanged
        assert cfg.maxiter == 1000
        assert resolved.n_anchors != cfg.n_anchors

    def test_min_anchors_3(self):
        from qmbp_simulation.pipeline.accelerated import AcceleratedConfig

        cfg = AcceleratedConfig(n_anchors=3)
        resolved = cfg.resolve_for_p(2, 38)
        assert resolved.n_anchors >= 3


# ═══════════════════════════════════════════════════════════════════════════════
# EvalCache partitioning
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvalCachePartitioning:
    """Tests for EvalCache p_layers-based file partitioning."""

    def test_p1_uses_default_path(self):
        from qmbp_simulation.execution.eval_cache import EvalCache

        cache = EvalCache(p_layers=1, enabled=False)
        assert cache._path.name == "eval_cache.json"

    def test_p2_uses_separate_file(self):
        from qmbp_simulation.execution.eval_cache import EvalCache

        cache = EvalCache(p_layers=2, enabled=False)
        assert cache._path.name == "eval_cache_p2.json"

    def test_p3_uses_separate_file(self):
        from qmbp_simulation.execution.eval_cache import EvalCache

        cache = EvalCache(p_layers=3, enabled=False)
        assert cache._path.name == "eval_cache_p3.json"

    def test_no_p_layers_uses_default(self):
        from qmbp_simulation.execution.eval_cache import EvalCache

        cache = EvalCache(enabled=False)
        assert cache._path.name == "eval_cache.json"

    def test_explicit_path_overrides_p_layers(self):
        from pathlib import Path

        from qmbp_simulation.execution.eval_cache import EvalCache

        custom = Path("/tmp/my_cache.json")
        cache = EvalCache(path=custom, p_layers=2, enabled=False)
        assert cache._path == custom


# ═══════════════════════════════════════════════════════════════════════════════
# Validation functions
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidationFunctions:
    """Tests for validate_npz_integrity and validate_p2_vs_p1_energy_monotonicity."""

    def test_validate_npz_integrity_returns_dict(self):
        from qmbp_simulation.analysis.metrics import validate_npz_integrity

        result = validate_npz_integrity(p_layers=1)
        assert isinstance(result, dict)
        assert "n_files" in result
        assert "n_issues" in result
        assert "by_p_layers" in result

    def test_validate_npz_integrity_p2_empty(self):
        from qmbp_simulation.analysis.metrics import validate_npz_integrity

        result = validate_npz_integrity(p_layers=2)
        # p=2 may have 0 files (no data yet)
        assert result["n_files"] >= 0
        assert result["n_issues"] >= 0

    def test_validate_npz_integrity_topology_filter(self):
        from qmbp_simulation.analysis.metrics import validate_npz_integrity

        result = validate_npz_integrity(topology="chain_1d", p_layers=1)
        # Should only have chain_1d files
        for iss in result.get("issues", []):
            if "topology" in iss:
                assert iss["topology"] == "chain_1d"

    def test_monotonicity_check_no_p2_data(self):
        from qmbp_simulation.analysis.metrics import validate_p2_vs_p1_energy_monotonicity

        result = validate_p2_vs_p1_energy_monotonicity()
        assert isinstance(result, dict)
        assert "n_violations" in result
        # If no p=2 data, should report 0 common points
        if result["n_common_points"] == 0:
            assert result["n_violations"] == 0

    def test_anti_regression_guard_no_baseline(self):
        from qmbp_simulation.analysis.metrics import check_p2_regression_vs_p1

        # Non-existent topology → no baseline → passes unconditionally
        result = check_p2_regression_vs_p1("nonexistent_topo_xyz", p2_pass_rate=0.5)
        assert result["passed"] is True
        assert result["p1_pass_rate"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# NPZ fingerprinting
# ═══════════════════════════════════════════════════════════════════════════════


class TestNpzFingerprinting:
    """Tests for compute_npz_fingerprint."""

    def test_deterministic(self):
        from qmbp_simulation.utils.helpers import compute_npz_fingerprint

        fp1 = compute_npz_fingerprint("data/multi_n_training/chain_1d_N10_p1.npz")
        fp2 = compute_npz_fingerprint("data/multi_n_training/chain_1d_N10_p1.npz")
        if fp1 != "missing":
            assert fp1 == fp2

    def test_missing_file(self):
        from qmbp_simulation.utils.helpers import compute_npz_fingerprint

        assert compute_npz_fingerprint("nonexistent.npz") == "missing"

    def test_length(self):
        from qmbp_simulation.utils.helpers import compute_npz_fingerprint

        fp = compute_npz_fingerprint("data/multi_n_training/chain_1d_N10_p1.npz")
        if fp != "missing":
            assert len(fp) == 12


# ═══════════════════════════════════════════════════════════════════════════════
# Model provenance in runner_base
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelProvenance:
    """Tests for model provenance tracking in ValidationRunner."""

    def test_model_provenance_attribute_exists(self):
        import inspect

        from qmbp_simulation.framework.runner_base import ValidationRunner

        src = inspect.getsource(ValidationRunner.__init__)
        assert "_model_provenance" in src

    def test_provenance_in_envelope_code(self):
        import inspect

        from qmbp_simulation.framework.runner_base import ValidationRunner

        src = inspect.getsource(ValidationRunner._build_envelope)
        assert "model_provenance" in src


# ═══════════════════════════════════════════════════════════════════════════════
# transfer_model_to_higher_p
# ═══════════════════════════════════════════════════════════════════════════════


class TestTransferModelToHigherP:
    """Tests for transfer_model_to_higher_p function."""

    def test_missing_checkpoint_raises(self):
        from qmbp_simulation.predictors.unified_mpnn import transfer_model_to_higher_p

        with pytest.raises(FileNotFoundError):
            transfer_model_to_higher_p(
                "nonexistent_checkpoint.pt", p_target=2, topology="chain_1d"
            )

    def test_no_topology_no_dataset_raises(self):
        from qmbp_simulation.predictors.unified_mpnn import transfer_model_to_higher_p

        with pytest.raises((FileNotFoundError, ValueError)):
            transfer_model_to_higher_p(
                "data/model_zoo/checkpoints/unified_tfim_br_MT_residual+film_p1.pt",
                p_target=2,
            )
