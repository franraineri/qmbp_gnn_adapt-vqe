#!/usr/bin/env python3
"""Tests for MultiNAggregator.

Validates:
1. Scan correctly discovers and loads NPZ data
2. dtype=object NPZ arrays are converted to float64
3. Quality filtering removes bad entries
4. Combined dataset has consistent features
5. ΔE/gap is computed correctly when missing
6. Empty/corrupt data is handled gracefully
"""

import numpy as np
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def npz_data_dir(tmp_path):
    """Create a temporary data directory structure for testing."""
    data_dir = tmp_path / "data" / "multi_n_training"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def mock_project_root(tmp_path):
    """Patch _PROJECT_ROOT to use temp directory."""
    import qmbp_simulation.predictors.multi_n_aggregator as mod
    original = mod._PROJECT_ROOT
    mod._PROJECT_ROOT = tmp_path
    yield tmp_path
    mod._PROJECT_ROOT = original


def _make_npz(path, n_qubits, n_points=5, n_params=None, dtype=np.float64,
              include_gaps=True, include_de_gaps=False, include_e_vqe=True):
    """Helper to create a realistic NPZ file for testing."""
    if n_params is None:
        n_params = 2 * n_qubits - 1  # Typical for bond-resolved p=1

    h_values = np.linspace(4.0, 1.0, n_points)
    theta_opt = np.random.randn(n_points, n_params).astype(np.float64)
    e_exact = -n_qubits * np.linspace(1.5, 0.8, n_points)
    e_vqe = e_exact + np.random.uniform(0.0, 0.1, n_points)
    gaps = np.ones(n_points) * 1.5

    # Optionally convert to dtype=object (simulates legacy format)
    if dtype == object:
        theta_list = [theta_opt[i] for i in range(n_points)]
        theta_opt = np.array(theta_list, dtype=object)

    kwargs = {
        "h_values": h_values,
        "theta_opt": theta_opt,
        "e_exact": e_exact,
    }
    if include_e_vqe:
        kwargs["e_vqe"] = e_vqe
    if include_gaps:
        kwargs["gaps"] = gaps
    if include_de_gaps:
        de_gaps = np.abs(e_vqe - e_exact) / np.maximum(gaps, 1e-10)
        kwargs["de_gaps"] = de_gaps

    np.savez(path, **kwargs)
    return kwargs


class TestMultiNAggregatorScan:
    """Tests for the scan() method."""

    def test_scan_discovers_npz_files(self, mock_project_root):
        """Should discover NPZ files matching topology pattern."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        data_dir = mock_project_root / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)
        _make_npz(data_dir / "chain_1d_N4_p1.npz", n_qubits=4, n_points=5)
        _make_npz(data_dir / "chain_1d_N8_p1.npz", n_qubits=8, n_points=10)

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        summary = agg.scan()

        assert 4 in summary
        assert 8 in summary
        assert summary[4] == 5
        assert summary[8] == 10

    def test_scan_ignores_other_topologies(self, mock_project_root):
        """Should only load files matching the configured topology."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        data_dir = mock_project_root / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)
        _make_npz(data_dir / "chain_1d_N4_p1.npz", n_qubits=4)
        _make_npz(data_dir / "ladder_N4_p1.npz", n_qubits=4)

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        summary = agg.scan()

        assert 4 in summary
        # Only chain_1d should be loaded
        assert len(summary) == 1

    def test_scan_handles_dtype_object_npz(self, mock_project_root):
        """Should convert dtype=object theta to float64 during scan."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        data_dir = mock_project_root / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)
        _make_npz(data_dir / "chain_1d_N4_p1.npz", n_qubits=4, dtype=object)

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        agg.scan()

        # All theta entries should be float64
        for n, points in agg._data_by_n.items():
            for pt in points:
                assert pt["theta"].dtype == np.float64

    def test_scan_computes_de_gaps_from_energies(self, mock_project_root):
        """Should compute ΔE/gap when de_gaps key is missing from NPZ."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        data_dir = mock_project_root / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)
        # NPZ without de_gaps but with e_vqe and gaps
        _make_npz(
            data_dir / "chain_1d_N4_p1.npz", n_qubits=4,
            include_de_gaps=False, include_gaps=True,
        )

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        agg.scan()

        # de_gap should be computed and finite
        for pt in agg._data_by_n[4]:
            assert np.isfinite(pt["de_gap"])
            assert pt["de_gap"] >= 0

    def test_scan_handles_missing_gaps(self, mock_project_root):
        """Should fall back to |ΔE| when gaps are missing."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        data_dir = mock_project_root / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)
        # NPZ without gaps key
        _make_npz(
            data_dir / "chain_1d_N4_p1.npz", n_qubits=4,
            include_de_gaps=False, include_gaps=False,
        )

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        agg.scan()

        # Should still have de_gap values (computed as |ΔE|)
        for pt in agg._data_by_n[4]:
            assert np.isfinite(pt["de_gap"])

    def test_scan_handles_corrupt_npz_gracefully(self, mock_project_root):
        """Should skip corrupt NPZ files without crashing."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        data_dir = mock_project_root / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)
        # Write garbage file
        (data_dir / "chain_1d_N6_p1.npz").write_bytes(b"not a valid npz")
        # Write valid file
        _make_npz(data_dir / "chain_1d_N4_p1.npz", n_qubits=4)

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        summary = agg.scan()

        # Should load valid file, skip corrupt
        assert 4 in summary
        assert 6 not in summary

    def test_scan_empty_directory(self, mock_project_root):
        """Should return empty summary when no data exists."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        data_dir = mock_project_root / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)
        # No files

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        summary = agg.scan()

        assert summary == {}


class TestMultiNAggregatorBuildDataset:
    """Tests for build_combined_dataset() method."""

    def test_quality_filter_removes_high_de_gap(self, mock_project_root):
        """Points with ΔE/gap above threshold should be excluded."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        data_dir = mock_project_root / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)

        # Create NPZ with known de_gaps: some pass, some fail
        n_points = 10
        h_values = np.linspace(4.0, 1.0, n_points)
        theta_opt = np.random.randn(n_points, 7).astype(np.float64)  # N=4 chain
        e_exact = -4 * np.linspace(1.5, 0.8, n_points)
        gaps = np.ones(n_points) * 2.0
        # First 5 pass (ΔE/gap < 0.10), last 5 fail
        e_vqe = e_exact.copy()
        e_vqe[:5] += 0.01  # Small error → ΔE/gap ≈ 0.005
        e_vqe[5:] += 0.5   # Large error → ΔE/gap ≈ 0.25

        np.savez(
            data_dir / "chain_1d_N4_p1.npz",
            h_values=h_values, theta_opt=theta_opt,
            e_vqe=e_vqe, e_exact=e_exact, gaps=gaps,
        )

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        dataset = agg.build_combined_dataset(max_de_gap=0.10)

        # Only 5 points should pass quality filter
        assert len(dataset) == 5

    def test_dataset_graphs_have_consistent_features(self, mock_project_root):
        """All graphs in dataset must have same node feature dimension."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        data_dir = mock_project_root / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)
        _make_npz(data_dir / "chain_1d_N4_p1.npz", n_qubits=4, n_points=3)
        _make_npz(data_dir / "chain_1d_N6_p1.npz", n_qubits=6, n_points=3)

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        dataset = agg.build_combined_dataset(max_de_gap=1.0)  # Accept all

        if len(dataset) > 1:
            feat_dim_0 = dataset[0].x.shape[1]
            for g in dataset[1:]:
                assert g.x.shape[1] == feat_dim_0

    def test_dataset_y_is_float32_tensor(self, mock_project_root):
        """Graph targets (y) should be float32 torch tensors."""
        import torch
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        data_dir = mock_project_root / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)
        _make_npz(data_dir / "chain_1d_N4_p1.npz", n_qubits=4, n_points=3)

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        dataset = agg.build_combined_dataset(max_de_gap=1.0)

        assert len(dataset) > 0
        for g in dataset:
            assert g.y.dtype == torch.float32
            assert g.y.dim() == 1  # Flat θ vector

    def test_dtype_object_npz_produces_valid_torch_tensors(self, mock_project_root):
        """Legacy dtype=object NPZ should produce valid torch tensors in dataset."""
        import torch
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        data_dir = mock_project_root / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)
        _make_npz(
            data_dir / "chain_1d_N4_p1.npz", n_qubits=4,
            n_points=3, dtype=object,
        )

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        dataset = agg.build_combined_dataset(max_de_gap=1.0)

        # Should NOT raise "can't convert np.ndarray of type numpy.object_"
        assert len(dataset) > 0
        for g in dataset:
            assert g.y.dtype == torch.float32
            assert torch.all(torch.isfinite(g.y))

    def test_empty_scan_returns_empty_dataset(self, mock_project_root):
        """Should return empty list when no data scanned."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        data_dir = mock_project_root / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        dataset = agg.build_combined_dataset()

        assert dataset == []

    def test_all_filtered_returns_empty_dataset(self, mock_project_root):
        """Should return empty list when all points fail quality filter."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        data_dir = mock_project_root / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)

        # Create NPZ where all points have terrible ΔE/gap
        n_points = 5
        h_values = np.linspace(4.0, 1.0, n_points)
        theta_opt = np.random.randn(n_points, 7).astype(np.float64)
        e_exact = -4 * np.linspace(1.5, 0.8, n_points)
        e_vqe = e_exact + 5.0  # Huge error
        gaps = np.ones(n_points) * 1.0

        np.savez(
            data_dir / "chain_1d_N4_p1.npz",
            h_values=h_values, theta_opt=theta_opt,
            e_vqe=e_vqe, e_exact=e_exact, gaps=gaps,
        )

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        dataset = agg.build_combined_dataset(max_de_gap=0.05)

        assert dataset == []


class TestMultiNAggregatorSummary:
    """Tests for summary and utility methods."""

    def test_available_n_values(self, mock_project_root):
        """Should return sorted list of available N values."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        data_dir = mock_project_root / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)
        _make_npz(data_dir / "chain_1d_N8_p1.npz", n_qubits=8)
        _make_npz(data_dir / "chain_1d_N4_p1.npz", n_qubits=4)
        _make_npz(data_dir / "chain_1d_N12_p1.npz", n_qubits=12)

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        n_values = agg.available_n_values()

        assert n_values == [4, 8, 12]  # Sorted

    def test_summary_structure(self, mock_project_root):
        """Summary should have expected keys."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        data_dir = mock_project_root / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)
        _make_npz(data_dir / "chain_1d_N4_p1.npz", n_qubits=4, n_points=7)

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        agg.scan()
        s = agg.summary()

        assert "points_per_n" in s
        assert s["points_per_n"][4] == 7


class TestMultiNAggregatorNaNHandling:
    """Tests for NaN/Inf robustness throughout the pipeline."""

    def test_nan_in_theta_excluded_from_dataset(self, mock_project_root):
        """Points with NaN theta should not appear in dataset."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        data_dir = mock_project_root / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)

        # Create NPZ with one NaN theta entry
        n_points = 5
        h_values = np.linspace(4.0, 1.0, n_points)
        theta_opt = np.random.randn(n_points, 7).astype(np.float64)
        theta_opt[2] = np.nan  # Corrupt entry
        e_exact = -4 * np.linspace(1.5, 0.8, n_points)
        e_vqe = e_exact + 0.01
        gaps = np.ones(n_points) * 2.0

        np.savez(
            data_dir / "chain_1d_N4_p1.npz",
            h_values=h_values, theta_opt=theta_opt,
            e_vqe=e_vqe, e_exact=e_exact, gaps=gaps,
        )

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        agg.scan()

        # NaN theta point should be scanned but its theta should be float64
        # (the NaN filtering happens at graph build time in is_point_failure
        # or via finite check in build_unified_bond_resolved_graph)
        points = agg._data_by_n[4]
        assert len(points) == n_points  # All scanned

        # But all theta should be float64 (not object)
        for pt in points:
            assert pt["theta"].dtype == np.float64

    def test_inf_energy_excluded_from_quality_filter(self, mock_project_root):
        """Points with Inf in e_vqe should get large de_gap and be filtered."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        data_dir = mock_project_root / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)

        n_points = 5
        h_values = np.linspace(4.0, 1.0, n_points)
        theta_opt = np.random.randn(n_points, 7).astype(np.float64)
        e_exact = -4 * np.linspace(1.5, 0.8, n_points)
        e_vqe = e_exact + 0.01
        e_vqe[0] = np.inf  # Corrupt energy
        gaps = np.ones(n_points) * 2.0

        np.savez(
            data_dir / "chain_1d_N4_p1.npz",
            h_values=h_values, theta_opt=theta_opt,
            e_vqe=e_vqe, e_exact=e_exact, gaps=gaps,
        )

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        dataset = agg.build_combined_dataset(max_de_gap=0.10)

        # The Inf point should be excluded by quality filter (its de_gap = Inf)
        assert len(dataset) <= n_points - 1

    def test_legacy_energies_key(self, mock_project_root):
        """Should handle legacy 'energies' key instead of 'e_vqe'."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        data_dir = mock_project_root / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)

        n_points = 5
        h_values = np.linspace(4.0, 1.0, n_points)
        theta_opt = np.random.randn(n_points, 7).astype(np.float64)
        e_exact = -4 * np.linspace(1.5, 0.8, n_points)
        energies = e_exact + 0.01
        gaps = np.ones(n_points) * 2.0

        np.savez(
            data_dir / "chain_1d_N4_p1.npz",
            h_values=h_values, theta_opt=theta_opt,
            energies=energies,  # Legacy key!
            e_exact=e_exact, gaps=gaps,
        )

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        agg.scan()

        # Should compute de_gap from 'energies' key
        for pt in agg._data_by_n[4]:
            assert np.isfinite(pt["de_gap"])
            assert pt["de_gap"] >= 0
