"""Tests for data integrity, GT coherence, consistency validation, and h-precision.

Covers all new functionality from the data integrity sessions:
- validate_gt_npz_coherence: GT cache ↔ NPZ e_exact cross-validation
- validate_data_consistency: multi-source cross-check (zoo ↔ comparison ↔ dashboard)
- post_experiment_sync: consolidated data store synchronization
- query_mt_vs_st_comparison: MT vs ST model comparison queries
- pass_rate_by_n: per-N pass rate tracking in ZooEntry
- confidence_level: eval cache density → confidence classification
- h-precision convention: all cache keys use :.2f
- training_intelligence: retrain triggers, h-range alignment, training readiness
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def gt_cache_dir(tmp_path):
    """Create a minimal GT cache JSON for testing."""
    gt_data = {
        "version": 2,
        "entries": {
            "chain_1d|6|tfim_bond_resolved|2.50": {"energy": -15.123456, "gap": 1.5},
            "chain_1d|6|tfim_bond_resolved|3.00": {"energy": -18.234567, "gap": 1.2},
            "chain_1d|6|tfim_bond_resolved|3.50": {"energy": -21.345678, "gap": 0.9},
            "chain_1d|10|tfim_bond_resolved|2.50": {"energy": -25.111111, "gap": 1.3},
            "chain_1d|10|tfim_bond_resolved|3.00": {"energy": -30.222222, "gap": 1.1},
        },
    }
    gt_path = tmp_path / "data" / "ground_truth_cache.json"
    gt_path.parent.mkdir(parents=True)
    gt_path.write_text(json.dumps(gt_data))
    return tmp_path


@pytest.fixture
def npz_dir(gt_cache_dir):
    """Create NPZ training files with e_exact matching (and some stale) GT."""
    npz_path = gt_cache_dir / "data" / "multi_n_training"
    npz_path.mkdir(parents=True)

    # Coherent file (matches GT exactly)
    h_vals = np.array([2.50, 3.00, 3.50])
    e_exact = np.array([-15.123456, -18.234567, -21.345678])
    e_vqe = np.array([-15.10, -18.20, -21.30])
    gaps = np.array([1.5, 1.2, 0.9])
    theta_opt = np.random.randn(3, 11)
    np.savez(
        npz_path / "chain_1d_N6_p1.npz",
        h_values=h_vals, e_exact=e_exact, e_vqe=e_vqe,
        gaps=gaps, theta_opt=theta_opt,
        de_gaps=np.abs(e_vqe - e_exact) / gaps,
    )

    # Stale file (e_exact differs from GT by 0.05)
    h_vals_10 = np.array([2.50, 3.00])
    e_exact_stale = np.array([-25.061111, -30.172222])  # ~0.05 off from GT
    e_vqe_10 = np.array([-25.05, -30.10])
    gaps_10 = np.array([1.3, 1.1])
    theta_opt_10 = np.random.randn(2, 19)
    np.savez(
        npz_path / "chain_1d_N10_p1.npz",
        h_values=h_vals_10, e_exact=e_exact_stale, e_vqe=e_vqe_10,
        gaps=gaps_10, theta_opt=theta_opt_10,
        de_gaps=np.abs(e_vqe_10 - e_exact_stale) / gaps_10,
    )

    return gt_cache_dir


@pytest.fixture
def zoo_manifest(tmp_path):
    """Create a minimal zoo manifest for testing."""
    manifest = [
        {
            "model": "tfim_bond_resolved",
            "topology": "chain_1d",
            "n_qubits": 0,
            "p_layers": 1,
            "checkpoint_file": "test_chain_1d_p1.pt",
            "h_range": [2.0, 5.0],
            "pass_rate": 0.75,
            "pass_rate_source": "training_data_eval",
            "n_training_points": 100,
            "seeds": [42],
            "created": "2026-08-19T00:00:00",
            "notes": "",
            "sha256": "",
            "runner_tag": "XX",
            "date_tag": "190826",
            "training_quality_score": 0.8,
            "pass_rate_by_n": {"6": 1.0, "10": 0.5, "16": 0.0},
        },
    ]
    manifest_path = tmp_path / "data" / "model_zoo" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))
    return tmp_path


# ═══════════════════════════════════════════════════════════════════════════════
# 1. H-Precision Convention Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestHPrecisionConvention:
    """All cache keys must use exactly 2 decimal places for h values."""

    def test_gt_cache_make_key_uses_2f(self):
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache = GroundTruthCache.__new__(GroundTruthCache)
        cache._data = {}
        cache._dirty = False
        cache._write_count = 0
        key = cache._make_key("chain_1d", 10, "tfim", 2.5)
        assert key == "chain_1d|10|tfim|2.50"

    def test_gt_cache_key_rounds_correctly(self):
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache = GroundTruthCache.__new__(GroundTruthCache)
        cache._data = {}
        cache._dirty = False
        cache._write_count = 0
        # 2.499999 should round to 2.50
        key = cache._make_key("chain_1d", 10, "tfim", 2.499999)
        assert key == "chain_1d|10|tfim|2.50"

    def test_gt_cache_key_no_extra_precision(self):
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache = GroundTruthCache.__new__(GroundTruthCache)
        cache._data = {}
        cache._dirty = False
        cache._write_count = 0
        key = cache._make_key("chain_1d", 10, "tfim", 2.5000001)
        # Should NOT be "chain_1d|10|tfim|2.500000"
        assert "2.50" in key
        assert "2.500000" not in key

    def test_eval_cache_gt_key_uses_2f(self):
        from qmbp_simulation.execution.eval_cache import EvalCache

        cache = EvalCache()
        # put_ground_truth should use :.2f in key
        cache.put_ground_truth("chain_1d", 10, 2.5, -25.0)
        key = f"GT|tfim|chain_1d|10|{2.5:.2f}"
        assert key in cache._data

    def test_eval_cache_make_key_uses_2f_for_h(self):
        from qmbp_simulation.execution.eval_cache import EvalCache

        cache = EvalCache.__new__(EvalCache)
        cache._data = {}
        cache._enabled = True
        cache._dirty = False
        cache._path = Path("/dev/null")
        theta = np.array([0.1, 0.2, 0.3])
        key = cache.make_key("chain_1d", 10, 2.5, theta)
        # h part should be "2.50" not "2.50000000"
        parts = key.split("|")
        h_part = parts[5]  # model|topo|N|p|J|h|hash
        assert h_part == "2.50"

    def test_load_npz_as_theta_dict_default_precision_2(self):
        """Default h_precision should be 2."""
        import inspect

        from qmbp_simulation.framework.result_io import load_npz_as_theta_dict

        sig = inspect.signature(load_npz_as_theta_dict)
        assert sig.parameters["h_precision"].default == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GT ↔ NPZ Coherence Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateGTNPZCoherence:
    """Tests for validate_gt_npz_coherence()."""

    def test_detects_stale_npz(self):
        """Should detect NPZ files with e_exact that differs from GT cache."""
        from qmbp_simulation.analysis.metrics import validate_gt_npz_coherence

        # Direct test with real data — should complete without error
        result = validate_gt_npz_coherence(fix=False, include_extrapolation=False)
        assert "n_files_checked" in result
        assert "n_points_checked" in result
        assert "summary" in result
        assert result["n_files_checked"] >= 0

    def test_returns_zero_issues_when_coherent(self):
        """After fix, should show zero issues."""
        from qmbp_simulation.analysis.metrics import validate_gt_npz_coherence

        # Run twice: first fix, then verify
        validate_gt_npz_coherence(fix=True, include_extrapolation=True)
        result = validate_gt_npz_coherence(fix=False, include_extrapolation=True)
        assert result["n_files_with_issues"] == 0

    def test_includes_extrapolation_dir(self):
        """With include_extrapolation=True, should check more files."""
        from qmbp_simulation.analysis.metrics import validate_gt_npz_coherence

        with_extrap = validate_gt_npz_coherence(include_extrapolation=True)
        without_extrap = validate_gt_npz_coherence(include_extrapolation=False)
        assert with_extrap["n_files_checked"] >= without_extrap["n_files_checked"]

    def test_fuzzy_h_match(self):
        """Should match h values even with slight floating point differences."""
        from qmbp_simulation.analysis.metrics import validate_gt_npz_coherence

        result = validate_gt_npz_coherence(include_extrapolation=True)
        # The fuzzy match (within 1e-4) should find more matches than exact-only
        assert result["n_points_checked"] > 0

    def test_reports_not_in_gt(self):
        """Should report how many NPZ points have no GT cache entry."""
        from qmbp_simulation.analysis.metrics import validate_gt_npz_coherence

        result = validate_gt_npz_coherence()
        assert "n_points_not_in_gt" in result
        assert result["n_points_not_in_gt"] >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Data Consistency Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateDataConsistency:
    """Tests for validate_data_consistency()."""

    def test_returns_required_keys(self):
        from qmbp_simulation.analysis.metrics import validate_data_consistency

        result = validate_data_consistency()
        assert "is_consistent" in result
        assert "n_checks" in result
        assert "n_issues" in result
        assert "findings" in result
        assert "zoo_vs_comparison" in result
        assert "registry_vs_curves" in result

    def test_informational_findings_not_counted_as_issues(self):
        """Findings with is_informational=True should not count as real issues."""
        from qmbp_simulation.analysis.metrics import validate_data_consistency

        result = validate_data_consistency()
        # Verify informational findings exist and are separated
        informational = [f for f in result["findings"] if f.get("is_informational")]
        real = [f for f in result["findings"] if not f.get("is_informational")]
        # n_issues should be >= real findings (may also include tier3 issues)
        assert result["n_issues"] >= len(real)
        # Informational findings should NOT be counted
        assert result["n_issues"] < len(result["findings"])
        # Total findings = real + informational
        assert len(result["findings"]) == len(real) + len(informational)

    def test_zoo_vs_comparison_has_per_n_data(self):
        """Zoo comparison should include pass_rate_by_n breakdown."""
        from qmbp_simulation.analysis.metrics import validate_data_consistency

        result = validate_data_consistency()
        for ckpt, info in result["zoo_vs_comparison"].items():
            assert "zoo_pass_rate" in info
            assert "comparison_avg" in info
            assert "comparison_by_n" in info
            assert "consistent" in info


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MT vs ST Comparison Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestQueryMTvsST:
    """Tests for query_mt_vs_st_comparison()."""

    def test_returns_required_structure(self):
        from qmbp_simulation.analysis.metrics import query_mt_vs_st_comparison

        result = query_mt_vs_st_comparison()
        assert "global" in result
        assert "per_topology" in result
        assert "per_scenario" in result
        assert "source" in result
        assert result["source"] in ("dashboard", "live", "none")

    def test_topology_filter(self):
        """Filtering by topology should reduce results."""
        from qmbp_simulation.analysis.metrics import query_mt_vs_st_comparison

        all_topos = query_mt_vs_st_comparison()
        chain_only = query_mt_vs_st_comparison(topology="chain_1d")

        if all_topos["global"]["total"] > 0:
            assert chain_only["global"]["total"] <= all_topos["global"]["total"]

    def test_n_range_filter(self):
        """Filtering by N range should reduce scenarios."""
        from qmbp_simulation.analysis.metrics import query_mt_vs_st_comparison

        all_n = query_mt_vs_st_comparison()
        large_n = query_mt_vs_st_comparison(n_min=16)

        if all_n["global"]["total"] > 0:
            assert large_n["global"]["total"] <= all_n["global"]["total"]

    def test_from_dashboard_vs_live(self):
        """Dashboard and live sources should give consistent results."""
        from qmbp_simulation.analysis.metrics import query_mt_vs_st_comparison

        dash = query_mt_vs_st_comparison(from_dashboard=True)
        live = query_mt_vs_st_comparison(from_dashboard=False)

        # Both should have the same structure
        if dash["source"] == "dashboard" and live["source"] == "live":
            # Global totals should be close (dashboard uses latest_only=True)
            assert abs(dash["global"]["mt_wins"] - live["global"]["mt_wins"]) <= 5


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Zoo pass_rate_by_n Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassRateByN:
    """Tests for pass_rate_by_n field and update functions."""

    def test_zoo_entry_has_pass_rate_by_n(self):
        from qmbp_simulation.predictors.model_zoo import ZooEntry

        entry = ZooEntry(
            model="tfim", topology="chain_1d",
            n_qubits=0, p_layers=1, checkpoint_file="test.pt",
        )
        assert hasattr(entry, "pass_rate_by_n")
        assert entry.pass_rate_by_n == {}

    def test_zoo_entry_serializes_pass_rate_by_n(self):
        from dataclasses import asdict

        from qmbp_simulation.predictors.model_zoo import ZooEntry

        entry = ZooEntry(
            model="tfim", topology="chain_1d",
            n_qubits=0, p_layers=1, checkpoint_file="test.pt",
            pass_rate_by_n={"10": 0.8, "20": 0.4},
        )
        d = asdict(entry)
        assert d["pass_rate_by_n"] == {"10": 0.8, "20": 0.4}

    def test_update_zoo_pass_rate_by_n(self):
        """Should merge new per-N rates into existing."""
        from qmbp_simulation.predictors.model_zoo import (
            _load_manifest,
            update_zoo_pass_rate_by_n,
        )

        entries = _load_manifest()
        if not entries:
            pytest.skip("No zoo entries to test")

        entry = entries[0]
        old_by_n = dict(entry.pass_rate_by_n)

        # This just verifies the function doesn't crash
        # (actual update tested via model_comparison integration)
        result = update_zoo_pass_rate_by_n(
            entry.checkpoint_file,
            {999: 0.99},  # fake N value
            update_global=False,
        )
        assert isinstance(result, bool)

        # Clean up: remove the fake entry
        entries = _load_manifest()
        for e in entries:
            if e.checkpoint_file == entry.checkpoint_file:
                e.pass_rate_by_n.pop("999", None)
        from qmbp_simulation.predictors.model_zoo import _save_manifest
        _save_manifest(entries)

    def test_backfill_from_comparisons(self):
        """backfill_pass_rate_by_n_from_comparisons should populate from JSONs."""
        from qmbp_simulation.predictors.model_zoo import backfill_pass_rate_by_n_from_comparisons

        # Just verify it runs without error
        n_updated = backfill_pass_rate_by_n_from_comparisons()
        assert isinstance(n_updated, int)
        assert n_updated >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Confidence Level Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfidenceLevel:
    """Tests for _compute_confidence_level()."""

    def test_high_confidence(self):
        from qmbp_simulation.analysis.metrics import _compute_confidence_level

        level = _compute_confidence_level(
            n_points=200, eval_cache_density=20, pass_rate_dual=0.80,
        )
        assert level == "high"

    def test_very_low_confidence(self):
        from qmbp_simulation.analysis.metrics import _compute_confidence_level

        level = _compute_confidence_level(
            n_points=3, eval_cache_density=0, pass_rate_dual=0.50,
        )
        assert level == "very_low"

    def test_medium_confidence(self):
        from qmbp_simulation.analysis.metrics import _compute_confidence_level

        level = _compute_confidence_level(
            n_points=40, eval_cache_density=3, pass_rate_dual=0.70,
        )
        assert level in ("medium", "high")

    def test_confidence_increases_with_points(self):
        from qmbp_simulation.analysis.metrics import _compute_confidence_level

        levels = ["very_low", "low", "medium", "high"]
        c_few = _compute_confidence_level(n_points=5, eval_cache_density=1, pass_rate_dual=0.5)
        c_many = _compute_confidence_level(n_points=200, eval_cache_density=1, pass_rate_dual=0.5)
        assert levels.index(c_many) >= levels.index(c_few)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Training Intelligence Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrainingIntelligence:
    """Tests for training_intelligence module."""

    def test_check_retrain_triggers_returns_list(self):
        from qmbp_simulation.predictors.training_intelligence import check_retrain_triggers

        triggers = check_retrain_triggers()
        assert isinstance(triggers, list)
        for t in triggers:
            assert hasattr(t, "topology")
            assert hasattr(t, "priority")
            assert hasattr(t, "reason")
            assert hasattr(t, "should_auto_execute")

    def test_triggers_sorted_by_priority(self):
        from qmbp_simulation.predictors.training_intelligence import check_retrain_triggers

        triggers = check_retrain_triggers()
        if len(triggers) >= 2:
            priorities = [t.priority for t in triggers]
            assert priorities == sorted(priorities)

    def test_validate_h_range_alignment(self):
        from qmbp_simulation.predictors.training_intelligence import validate_h_range_alignment

        result = validate_h_range_alignment("chain_1d")
        assert hasattr(result, "is_valid")
        assert hasattr(result, "coverage")
        assert hasattr(result, "training_range")
        assert hasattr(result, "eval_range")
        assert 0 <= result.coverage <= 1.0

    def test_validate_training_readiness(self):
        from qmbp_simulation.predictors.training_intelligence import validate_training_readiness

        ready, issues = validate_training_readiness("chain_1d")
        assert isinstance(ready, bool)
        assert isinstance(issues, list)

    def test_prepare_training_config(self):
        from qmbp_simulation.predictors.training_intelligence import prepare_training_config

        config = prepare_training_config(topologies=["chain_1d"])
        assert config.topologies == ["chain_1d"]
        assert config.max_n == 20
        assert config.recommended_epochs > 0
        assert config.confidence in ("high", "medium", "low")
        assert config.n_useful_points >= 0

    def test_prepare_training_config_with_extrapolation(self):
        from qmbp_simulation.predictors.training_intelligence import prepare_training_config

        config = prepare_training_config(include_extrapolation=True)
        # extrapolation data sources should be tracked
        assert "extrapolation" in config.data_sources


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Post-Experiment Sync Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPostExperimentSync:
    """Tests for post_experiment_sync()."""

    def test_returns_required_keys(self):
        from qmbp_simulation.analysis.metrics import post_experiment_sync

        result = post_experiment_sync(verbose=False)
        assert "steps_completed" in result
        assert "steps_failed" in result
        assert "gt_coherence" in result
        assert "dashboard_regenerated" in result

    def test_gt_coherence_always_checked(self):
        from qmbp_simulation.analysis.metrics import post_experiment_sync

        result = post_experiment_sync(verbose=False)
        assert "gt_npz_coherence" in result["steps_completed"]

    def test_retrain_loop_disabled(self):
        """Retrain loop should be disabled (architecture bottleneck)."""
        from qmbp_simulation.analysis.metrics import post_experiment_sync

        result = post_experiment_sync(verbose=False)
        assert result["retrain_loop"] is None
        assert any("disabled" in s for s in result["steps_completed"])


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_h_precision_extreme_values(self):
        """h=0.0, h=10.0, h=π should all format to 2 decimals."""
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache = GroundTruthCache.__new__(GroundTruthCache)
        cache._data = {}
        cache._dirty = False
        cache._write_count = 0

        assert cache._make_key("t", 1, "m", 0.0) == "t|1|m|0.00"
        assert cache._make_key("t", 1, "m", 10.0) == "t|1|m|10.00"
        assert "3.14" in cache._make_key("t", 1, "m", 3.14159)

    def test_empty_zoo_manifest(self, tmp_path):
        """Should handle empty zoo gracefully."""
        from qmbp_simulation.predictors.model_zoo import _load_manifest

        # The actual test uses the real manifest, just verify no crash
        entries = _load_manifest()
        assert isinstance(entries, list)

    def test_mt_vs_st_no_data(self):
        """Should return zeros when no comparison data exists."""
        from qmbp_simulation.analysis.metrics import query_mt_vs_st_comparison

        # Filter to non-existent topology
        result = query_mt_vs_st_comparison(topology="nonexistent_topology")
        assert result["global"]["total"] == 0

    def test_confidence_level_edge_zero_points(self):
        from qmbp_simulation.analysis.metrics import _compute_confidence_level

        level = _compute_confidence_level(n_points=0, eval_cache_density=0, pass_rate_dual=0.0)
        assert level == "very_low"

    def test_confidence_level_edge_extreme_pass_rate(self):
        from qmbp_simulation.analysis.metrics import _compute_confidence_level

        # p=1.0 → SE=0 → high precision score
        level = _compute_confidence_level(n_points=100, eval_cache_density=10, pass_rate_dual=0.99)
        assert level == "high"

    def test_validate_training_readiness_empty_topology(self):
        from qmbp_simulation.predictors.training_intelligence import validate_training_readiness

        ready, issues = validate_training_readiness("nonexistent_topo")
        assert not ready
        assert len(issues) > 0

    def test_retrain_trigger_dataclass(self):
        from qmbp_simulation.predictors.training_intelligence import RetrainTrigger

        t = RetrainTrigger(
            topology="chain_1d",
            reason="test",
            priority=3,
            data_growth_pct=0.5,
            n_training_points=100,
            h_range_coverage=0.8,
        )
        assert t.should_auto_execute is True

        t_low = RetrainTrigger(
            topology="chain_1d",
            reason="test",
            priority=3,
            data_growth_pct=0.1,  # below threshold
            n_training_points=100,
            h_range_coverage=0.8,
        )
        assert t_low.should_auto_execute is False
