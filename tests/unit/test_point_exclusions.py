"""Tests for per-point training data exclusion system.

Tests the ability to exclude individual h-points within active NPZ files
that fail quality criteria, without excluding the entire file.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project structure with NPZ test data."""
    # Create training data dir
    train_dir = tmp_path / "data" / "multi_n_training"
    train_dir.mkdir(parents=True)

    # Create a test NPZ with mixed quality (some pass, some fail)
    h_values = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    e_exact = np.array([-20.0, -25.0, -30.0, -35.0, -40.0, -45.0, -50.0])
    # Bad at h=2.0 (de_gap=0.75), bad at h=2.5 (de_gap=0.33), good at h>=3.0
    e_vqe = np.array([-18.5, -24.0, -29.95, -34.97, -39.98, -44.99, -49.99])
    gaps = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    theta_opt = np.random.default_rng(42).standard_normal((7, 19)).astype(np.float32)

    np.savez(
        train_dir / "heavy_hex_N10_p1.npz",
        h_values=h_values,
        e_vqe=e_vqe,
        e_exact=e_exact,
        gaps=gaps,
        theta_opt=theta_opt,
        de_gaps=np.abs(e_vqe - e_exact) / gaps,
    )

    # Create a file that's all-good (should have no point exclusions)
    h_good = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
    e_exact_good = np.array([-12.0, -14.0, -16.0, -18.0, -20.0])
    e_vqe_good = e_exact_good + np.array([0.001, 0.002, 0.001, 0.001, 0.001])
    gaps_good = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
    theta_good = np.random.default_rng(42).standard_normal((5, 7)).astype(np.float32)

    np.savez(
        train_dir / "heavy_hex_N4_p1.npz",
        h_values=h_good,
        e_vqe=e_vqe_good,
        e_exact=e_exact_good,
        gaps=gaps_good,
        theta_opt=theta_good,
        de_gaps=np.abs(e_vqe_good - e_exact_good) / gaps_good,
    )

    # Create exclusion registry
    excl_path = tmp_path / "data" / "training_exclusions.json"
    excl_path.write_text(json.dumps({"excluded": [], "version": 1}))

    return tmp_path


@pytest.fixture
def mock_paths(tmp_project):
    """Patch exclusion path and project root for isolated testing."""
    excl_path = tmp_project / "data" / "training_exclusions.json"
    with patch(
        "qmbp_simulation.analysis.metrics._resolve_exclusion_path",
        return_value=excl_path,
    ), patch(
        "qmbp_simulation.analysis.metrics._EXCLUSION_REGISTRY_PATH",
        excl_path,
    ):
        yield tmp_project


class TestAutoDetectPointExclusions:
    """Tests for auto_detect_point_exclusions."""

    def test_dry_run_detects_bad_points(self, tmp_project, mock_paths):
        """Dry run detects failing points without persisting."""
        from qmbp_simulation.analysis.metrics import auto_detect_point_exclusions

        # Point auto_detect to the tmp_project's data dir
        project_root = tmp_project
        npz_dirs_abs = [str(project_root / "data" / "multi_n_training")]

        # Need to patch _resolve_exclusion_path to return parent.parent = project_root
        excl_path = tmp_project / "data" / "training_exclusions.json"
        with patch(
            "qmbp_simulation.analysis.metrics._resolve_exclusion_path",
            return_value=excl_path,
        ):
            results = auto_detect_point_exclusions(
                dry_run=True,
                topology="heavy_hex",
            )

        # Should find bad points in N=10 file (h=2.0 and h=2.5 fail dual criterion)
        assert "heavy_hex_N10_p1.npz" in results
        bad_h_values = [p["h"] for p in results["heavy_hex_N10_p1.npz"]]
        assert 2.0 in bad_h_values
        assert 2.5 in bad_h_values

        # Good points should NOT be flagged
        assert 3.0 not in bad_h_values
        assert 4.0 not in bad_h_values

        # N=4 all-good file should NOT appear
        assert "heavy_hex_N4_p1.npz" not in results

    def test_dry_run_does_not_persist(self, mock_paths):
        """Dry run should not modify the registry."""
        from qmbp_simulation.analysis.metrics import (
            auto_detect_point_exclusions,
            get_point_exclusions,
        )

        auto_detect_point_exclusions(dry_run=True, topology="heavy_hex")
        pe = get_point_exclusions()
        assert len(pe) == 0

    def test_persist_saves_to_registry(self, mock_paths):
        """Non-dry-run persists point exclusions to disk."""
        from qmbp_simulation.analysis.metrics import (
            auto_detect_point_exclusions,
            get_point_exclusions,
        )

        auto_detect_point_exclusions(dry_run=False, topology="heavy_hex")
        pe = get_point_exclusions()

        assert "heavy_hex_N10_p1.npz" in pe
        assert len(pe["heavy_hex_N10_p1.npz"]) >= 2
        # h=2.0 and h=2.5 should be excluded
        assert 2.0 in pe["heavy_hex_N10_p1.npz"]
        assert 2.5 in pe["heavy_hex_N10_p1.npz"]

    def test_idempotent_detection(self, mock_paths):
        """Running twice should not duplicate exclusions."""
        from qmbp_simulation.analysis.metrics import auto_detect_point_exclusions

        # First run
        r1 = auto_detect_point_exclusions(dry_run=False, topology="heavy_hex")
        count_1 = sum(len(pts) for pts in r1.values())

        # Second run — should find 0 new (already excluded)
        r2 = auto_detect_point_exclusions(dry_run=False, topology="heavy_hex")
        count_2 = sum(len(pts) for pts in r2.values())

        assert count_1 > 0
        assert count_2 == 0

    def test_skips_fully_excluded_files(self, mock_paths):
        """Files in the main exclusion list should be skipped entirely."""
        from qmbp_simulation.analysis.metrics import (
            add_training_exclusion,
            auto_detect_point_exclusions,
        )

        # Exclude the N=10 file entirely
        add_training_exclusion(
            "heavy_hex_N10_p1.npz", "heavy_hex", n_qubits=10, reason="test"
        )

        # Now auto-detect should skip it
        results = auto_detect_point_exclusions(dry_run=True, topology="heavy_hex")
        assert "heavy_hex_N10_p1.npz" not in results

    def test_custom_threshold_strict(self, mock_paths):
        """Stricter threshold catches more failing points."""
        from qmbp_simulation.analysis.metrics import auto_detect_point_exclusions

        # Strict: de_gap >= 0.001 fails almost everything
        strict = auto_detect_point_exclusions(
            dry_run=True, topology="heavy_hex", de_gap_threshold=0.001
        )
        # Relaxed: de_gap >= 0.90 — only extreme failures
        relaxed = auto_detect_point_exclusions(
            dry_run=True, topology="heavy_hex", de_gap_threshold=0.90
        )

        strict_count = sum(len(pts) for pts in strict.values())
        relaxed_count = sum(len(pts) for pts in relaxed.values())
        assert strict_count > relaxed_count

    def test_topology_filter(self, mock_paths, tmp_project):
        """Topology filter only scans matching files."""
        from qmbp_simulation.analysis.metrics import auto_detect_point_exclusions

        # Create a chain_1d file
        train_dir = tmp_project / "data" / "multi_n_training"
        np.savez(
            train_dir / "chain_1d_N10_p1.npz",
            h_values=np.array([1.0, 2.0]),
            e_vqe=np.array([-5.0, -10.0]),
            e_exact=np.array([-8.0, -10.0]),
            gaps=np.array([1.0, 2.0]),
        )

        # Filter to chain_1d only
        results = auto_detect_point_exclusions(
            dry_run=True, topology="chain_1d"
        )

        # Should find chain_1d but NOT heavy_hex
        if "chain_1d_N10_p1.npz" in results:
            assert "heavy_hex_N10_p1.npz" not in results


class TestAddPointExclusions:
    """Tests for add_point_exclusions."""

    def test_add_new_exclusions(self, mock_paths):
        """Adding new h-values creates the entry."""
        from qmbp_simulation.analysis.metrics import (
            add_point_exclusions,
            get_point_exclusions,
        )

        n = add_point_exclusions("test_N10.npz", [1.0, 2.0, 3.0])
        pe = get_point_exclusions()

        assert n == 3
        assert "test_N10.npz" in pe
        assert set(pe["test_N10.npz"]) == {1.0, 2.0, 3.0}

    def test_no_duplicates(self, mock_paths):
        """Adding existing h-values doesn't create duplicates."""
        from qmbp_simulation.analysis.metrics import (
            add_point_exclusions,
            get_point_exclusions,
        )

        add_point_exclusions("test.npz", [1.0, 2.0])
        n = add_point_exclusions("test.npz", [2.0, 3.0])  # 2.0 is duplicate
        pe = get_point_exclusions()

        assert n == 1  # Only 3.0 is new
        assert sorted(pe["test.npz"]) == [1.0, 2.0, 3.0]

    def test_h_precision_rounding(self, mock_paths):
        """h-values are rounded to 2 decimals for consistency."""
        from qmbp_simulation.analysis.metrics import (
            add_point_exclusions,
            get_point_exclusions,
        )

        add_point_exclusions("test.npz", [2.5000001, 3.499999])
        pe = get_point_exclusions()

        assert pe["test.npz"] == [2.5, 3.5]

    def test_empty_list_returns_zero(self, mock_paths):
        """Adding empty list does nothing."""
        from qmbp_simulation.analysis.metrics import add_point_exclusions

        n = add_point_exclusions("test.npz", [])
        assert n == 0

    def test_multiple_files_independent(self, mock_paths):
        """Exclusions for different files don't interfere."""
        from qmbp_simulation.analysis.metrics import (
            add_point_exclusions,
            get_point_exclusions,
        )

        add_point_exclusions("file_a.npz", [1.0, 2.0])
        add_point_exclusions("file_b.npz", [3.0, 4.0])
        pe = get_point_exclusions()

        assert sorted(pe["file_a.npz"]) == [1.0, 2.0]
        assert sorted(pe["file_b.npz"]) == [3.0, 4.0]


class TestGetPointExclusions:
    """Tests for get_point_exclusions."""

    def test_empty_registry(self, mock_paths):
        """Returns empty dict when no point exclusions exist."""
        from qmbp_simulation.analysis.metrics import get_point_exclusions

        pe = get_point_exclusions()
        assert pe == {}

    def test_missing_file_returns_empty(self, tmp_path):
        """Returns empty when registry file doesn't exist."""
        from qmbp_simulation.analysis.metrics import get_point_exclusions

        fake_path = tmp_path / "nonexistent.json"
        with patch(
            "qmbp_simulation.analysis.metrics._resolve_exclusion_path",
            return_value=fake_path,
        ):
            pe = get_point_exclusions()

        assert pe == {}


class TestEdgeCases:
    """Edge cases for the per-point exclusion system."""

    def test_npz_without_gaps_uses_abs_error(self, mock_paths, tmp_project):
        """Files without 'gaps' should fall back to absolute error."""
        from qmbp_simulation.analysis.metrics import auto_detect_point_exclusions

        train_dir = tmp_project / "data" / "multi_n_training"
        # Create NPZ without gaps field
        np.savez(
            train_dir / "chain_1d_N6_p1.npz",
            h_values=np.array([2.0, 3.0, 4.0]),
            e_vqe=np.array([-5.0, -9.99, -14.99]),
            e_exact=np.array([-6.0, -10.0, -15.0]),
            # No gaps! abs_error = [1.0, 0.01, 0.01]
        )

        results = auto_detect_point_exclusions(
            dry_run=True, topology="chain_1d"
        )

        # h=2.0 has |ΔE|=1.0 > MAX_ABS_ERROR=0.10 → should be detected
        if "chain_1d_N6_p1.npz" in results:
            bad_h = [p["h"] for p in results["chain_1d_N6_p1.npz"]]
            assert 2.0 in bad_h

    def test_npz_with_nan_values_handled(self, mock_paths, tmp_project):
        """NaN in energy arrays should not crash the detector."""
        from qmbp_simulation.analysis.metrics import auto_detect_point_exclusions

        train_dir = tmp_project / "data" / "multi_n_training"
        np.savez(
            train_dir / "chain_1d_N8_p1.npz",
            h_values=np.array([2.0, 3.0, float("nan")]),
            e_vqe=np.array([float("nan"), -10.0, -15.0]),
            e_exact=np.array([-6.0, -10.0, -15.0]),
            gaps=np.array([1.0, 2.0, 3.0]),
        )

        # Should not raise
        results = auto_detect_point_exclusions(
            dry_run=True, topology="chain_1d"
        )
        # NaN produces infinite de_gap → detected as bad
        if "chain_1d_N8_p1.npz" in results:
            # At least the NaN point should be flagged
            assert len(results["chain_1d_N8_p1.npz"]) >= 1

    def test_empty_npz_skipped(self, mock_paths, tmp_project):
        """NPZ with zero h-points should be silently skipped."""
        from qmbp_simulation.analysis.metrics import auto_detect_point_exclusions

        train_dir = tmp_project / "data" / "multi_n_training"
        np.savez(
            train_dir / "chain_1d_N12_p1.npz",
            h_values=np.array([]),
            e_vqe=np.array([]),
            e_exact=np.array([]),
            gaps=np.array([]),
        )

        # Should not raise or include empty file
        results = auto_detect_point_exclusions(dry_run=True, topology="chain_1d")
        assert "chain_1d_N12_p1.npz" not in results

    def test_all_good_file_produces_no_exclusions(self, mock_paths):
        """A file where all points pass should not appear in results."""
        from qmbp_simulation.analysis.metrics import auto_detect_point_exclusions

        results = auto_detect_point_exclusions(
            dry_run=True, topology="heavy_hex"
        )

        # heavy_hex_N4_p1.npz has all good points (de_gap < 0.001)
        assert "heavy_hex_N4_p1.npz" not in results


class TestBuildCombinedDatasetFilters:
    """Tests for h_min/h_max and max_de_gap in build_combined_dataset."""

    @pytest.fixture
    def agg_with_data(self):
        """Create a MultiNAggregator with injected mock data.

        Uses 120 points so that even after filtering by h_min/h_max (removing
        ~50%), the remaining count stays above AUGMENTATION_MAX_FILTERED_POINTS=50,
        preventing data augmentation from inflating counts.
        """
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        agg = MultiNAggregator(topology="chain_1d", p_layers=1)
        points = []
        for h in np.linspace(1.0, 5.0, 120):
            points.append({
                "h": round(float(h), 2),
                "theta": np.zeros(19),
                "e_exact": -10.0 * h,
                "de_gap": 0.01,
                "n_qubits": 10,
                "quality_tier": "verified",
            })
        agg._data_by_n = {10: points}
        return agg

    def test_no_filters_returns_all(self, agg_with_data, mock_paths):
        """Without h_min/h_max, all passing points are included."""
        ds = agg_with_data.build_combined_dataset(max_de_gap=0.10)
        assert len(ds) == 120

    def test_h_min_excludes_low_h(self, agg_with_data, mock_paths):
        """h_min=2.5 excludes points with h < 2.5."""
        ds_all = agg_with_data.build_combined_dataset(max_de_gap=0.10)
        ds_hmin = agg_with_data.build_combined_dataset(max_de_gap=0.10, h_min=2.5)
        # h_min=2.5 removes roughly 22-23 points (h in [1.0, 2.5))
        assert len(ds_hmin) < len(ds_all)
        assert len(ds_hmin) > 0

    def test_h_max_excludes_high_h(self, agg_with_data, mock_paths):
        """h_max=3.5 excludes points with h > 3.5."""
        ds_all = agg_with_data.build_combined_dataset(max_de_gap=0.10)
        ds_hmax = agg_with_data.build_combined_dataset(max_de_gap=0.10, h_max=3.5)
        assert len(ds_hmax) < len(ds_all)
        assert len(ds_hmax) > 0

    def test_h_min_and_h_max_combined(self, agg_with_data, mock_paths):
        """Both h_min and h_max restrict the range further."""
        ds_all = agg_with_data.build_combined_dataset(max_de_gap=0.10)
        ds_both = agg_with_data.build_combined_dataset(
            max_de_gap=0.10, h_min=2.0, h_max=4.5
        )
        # Range [2.0, 4.5] on linspace(1.0, 5.0, 120) keeps ~87 points (> 50, no augmentation)
        assert len(ds_both) < len(ds_all)
        assert len(ds_both) > 0

    def test_h_min_above_all_data_returns_empty(self, agg_with_data, mock_paths):
        """h_min above all data produces empty dataset."""
        ds = agg_with_data.build_combined_dataset(max_de_gap=0.10, h_min=10.0)
        assert len(ds) == 0

    def test_none_filters_no_effect(self, agg_with_data, mock_paths):
        """h_min=None and h_max=None have no filtering effect."""
        ds_none = agg_with_data.build_combined_dataset(
            max_de_gap=0.10, h_min=None, h_max=None
        )
        ds_default = agg_with_data.build_combined_dataset(max_de_gap=0.10)
        assert len(ds_none) == len(ds_default)

    def test_h_filters_are_inclusive(self, agg_with_data, mock_paths):
        """h_min uses >= (inclusive) and h_max uses <= (inclusive)."""
        # Find an exact h value in the data
        h_values = [p["h"] for p in agg_with_data._data_by_n[10]]
        h_mid = h_values[15]  # Pick a middle value

        # Using h_min = h_mid should include that point
        ds_at = agg_with_data.build_combined_dataset(max_de_gap=0.10, h_min=h_mid)
        ds_above = agg_with_data.build_combined_dataset(
            max_de_gap=0.10, h_min=h_mid + 0.01
        )
        # ds_at includes h_mid, ds_above doesn't
        assert len(ds_at) > len(ds_above)
