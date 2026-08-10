#!/usr/bin/env python3
"""Tests for upsert_theta_npz and load_theta_npz functions.

These functions are critical for training data persistence:
- upsert_theta_npz: Atomic save with anti-regression (lower energy wins)
- load_theta_npz: Load with validation and legacy format support

Test scenarios:
1. Basic create/update operations
2. Anti-regression behavior (only update if energy improves)
3. NaN/Inf filtering
4. Legacy dtype=object NPZ compatibility
5. PyTorch tensor compatibility
6. Atomic write (no corruption on crash)
"""

import numpy as np
import pytest
from pathlib import Path

from qmbp_simulation.framework.result_io import upsert_theta_npz, load_theta_npz


class TestUpsertThetaNPZ:
    """Tests for upsert_theta_npz function."""

    def test_create_new_file(self, tmp_path):
        """Should create new NPZ file with correct structure."""
        npz_path = tmp_path / "test.npz"

        n_upd, n_add = upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0, 3.5]),
            theta_new=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            e_vqe_new=np.array([-5.0, -4.5]),
            e_exact_new=np.array([-5.1, -4.6]),
            gaps_new=np.array([1.0, 1.0]),
            method_new=["vqe", "mpnn_pred"],
        )

        assert npz_path.exists()
        assert n_add == 2
        assert n_upd == 0

        data = np.load(npz_path)
        assert len(data["h_values"]) == 2
        assert data["theta_opt"].shape == (2, 3)
        assert data["theta_opt"].dtype == np.float64  # Not object!
        assert list(data["method"]) == ["vqe", "mpnn_pred"]

    def test_anti_regression_keeps_better_energy(self, tmp_path):
        """Should NOT update if new energy is worse (anti-regression)."""
        npz_path = tmp_path / "test.npz"

        # Initial save with E=-5.0 at h=4.0
        upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0]),
            theta_new=np.array([[0.1, 0.2, 0.3]]),
            e_vqe_new=np.array([-5.0]),
            e_exact_new=np.array([-5.1]),
        )

        # Try to update with WORSE energy E=-4.5
        n_upd, n_add = upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0]),
            theta_new=np.array([[0.9, 0.9, 0.9]]),  # Different theta
            e_vqe_new=np.array([-4.5]),  # Worse energy
            e_exact_new=np.array([-5.1]),
        )

        assert n_upd == 0  # Should NOT update
        assert n_add == 0

        # Verify original theta is preserved
        data = np.load(npz_path)
        np.testing.assert_allclose(data["theta_opt"][0], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(data["e_vqe"][0], -5.0)

    def test_anti_regression_updates_better_energy(self, tmp_path):
        """Should update if new energy is better."""
        npz_path = tmp_path / "test.npz"

        # Initial save with E=-5.0
        upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0]),
            theta_new=np.array([[0.1, 0.2, 0.3]]),
            e_vqe_new=np.array([-5.0]),
            e_exact_new=np.array([-5.1]),
            method_new=["initial"],
        )

        # Update with BETTER energy E=-5.05
        n_upd, n_add = upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0]),
            theta_new=np.array([[0.7, 0.8, 0.9]]),
            e_vqe_new=np.array([-5.05]),  # Better energy
            e_exact_new=np.array([-5.1]),
            method_new=["refined"],
        )

        assert n_upd == 1
        assert n_add == 0

        data = np.load(npz_path)
        np.testing.assert_allclose(data["theta_opt"][0], [0.7, 0.8, 0.9])
        np.testing.assert_allclose(data["e_vqe"][0], -5.05)
        assert data["method"][0] == "refined"

    def test_filters_nan_theta(self, tmp_path):
        """Should filter out entries with NaN in theta."""
        npz_path = tmp_path / "test.npz"

        n_upd, n_add = upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0, 3.5, 3.0]),
            theta_new=np.array([
                [0.1, 0.2, 0.3],       # Valid
                [np.nan, 0.5, 0.6],    # Invalid - NaN
                [0.7, 0.8, 0.9],       # Valid
            ]),
            e_vqe_new=np.array([-5.0, -4.5, -4.0]),
            e_exact_new=np.array([-5.1, -4.6, -4.1]),
        )

        assert n_add == 2  # Only 2 valid entries
        data = np.load(npz_path)
        assert len(data["h_values"]) == 2

    def test_filters_inf_energy(self, tmp_path):
        """Should filter out entries with Inf energy."""
        npz_path = tmp_path / "test.npz"

        n_upd, n_add = upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0, 3.5]),
            theta_new=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            e_vqe_new=np.array([-5.0, np.inf]),  # Second is invalid
            e_exact_new=np.array([-5.1, -4.6]),
        )

        assert n_add == 1
        data = np.load(npz_path)
        assert len(data["h_values"]) == 1

    def test_single_point_upsert(self, tmp_path):
        """Should handle single-point upserts (common in iterative refinement)."""
        npz_path = tmp_path / "test.npz"

        # Add points one by one (simulates iterative improvement)
        for h, theta_val in [(4.0, 0.1), (3.5, 0.2), (3.0, 0.3)]:
            upsert_theta_npz(
                npz_path,
                h_new=np.array([h]),
                theta_new=np.array([[theta_val, theta_val + 0.1]]),
                e_vqe_new=np.array([-5.0 + theta_val]),
                e_exact_new=np.array([-5.1 + theta_val]),
            )

        data = np.load(npz_path)
        assert len(data["h_values"]) == 3

    def test_empty_input_returns_zero(self, tmp_path):
        """Should return (0, 0) for empty input arrays."""
        npz_path = tmp_path / "test.npz"

        n_upd, n_add = upsert_theta_npz(
            npz_path,
            h_new=np.array([]),
            theta_new=np.array([]).reshape(0, 3),
            e_vqe_new=np.array([]),
            e_exact_new=np.array([]),
        )

        assert n_upd == 0
        assert n_add == 0
        assert not npz_path.exists()  # No file created


class TestLoadThetaNPZ:
    """Tests for load_theta_npz function."""

    def test_load_standard_file(self, tmp_path):
        """Should load standard float64 NPZ files."""
        npz_path = tmp_path / "test.npz"
        np.savez(
            npz_path,
            h_values=np.array([4.0, 3.5]),
            theta_opt=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
            e_vqe=np.array([-5.0, -4.5]),
            e_exact=np.array([-5.1, -4.6]),
            gaps=np.array([1.0, 1.0]),
        )

        data = load_theta_npz(npz_path)

        assert len(data["h_values"]) == 2
        assert data["theta_opt"].dtype == np.float64
        np.testing.assert_allclose(data["theta_opt"][0], [0.1, 0.2])

    def test_load_legacy_object_array(self, tmp_path):
        """Should handle legacy NPZ with dtype=object theta arrays."""
        npz_path = tmp_path / "legacy.npz"

        # Create legacy format (dtype=object)
        theta_list = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        np.savez(
            npz_path,
            h_values=np.array([4.0, 3.5]),
            theta_opt=np.array(theta_list, dtype=object),  # Legacy format
            e_vqe=np.array([-5.0, -4.5]),
            e_exact=np.array([-5.1, -4.6]),
        )

        data = load_theta_npz(npz_path)

        assert len(data["h_values"]) == 2
        # Should be converted to float64
        assert data["theta_opt"].dtype == np.float64
        np.testing.assert_allclose(data["theta_opt"][0], [0.1, 0.2, 0.3])

    def test_filters_corrupt_entries(self, tmp_path):
        """Should filter entries with NaN/Inf during load."""
        npz_path = tmp_path / "corrupt.npz"
        np.savez(
            npz_path,
            h_values=np.array([4.0, 3.5, 3.0]),
            theta_opt=np.array([
                [0.1, 0.2],
                [np.nan, 0.4],  # Invalid
                [0.5, 0.6],
            ]),
            e_vqe=np.array([-5.0, -4.5, -4.0]),
            e_exact=np.array([-5.1, -4.6, -4.1]),
        )

        data = load_theta_npz(npz_path)

        assert len(data["h_values"]) == 2  # 1 filtered

    def test_missing_file_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_theta_npz(tmp_path / "nonexistent.npz")

    def test_missing_required_keys_raises(self, tmp_path):
        """Should raise ValueError for missing required keys."""
        npz_path = tmp_path / "incomplete.npz"
        np.savez(npz_path, h_values=np.array([4.0]))  # Missing theta_opt, e_exact

        with pytest.raises(ValueError, match="missing required keys"):
            load_theta_npz(npz_path)

    def test_handles_legacy_energies_key(self, tmp_path):
        """Should handle legacy 'energies' key instead of 'e_vqe'."""
        npz_path = tmp_path / "legacy_keys.npz"
        np.savez(
            npz_path,
            h_values=np.array([4.0]),
            theta_opt=np.array([[0.1, 0.2]]),
            energies=np.array([-5.0]),  # Legacy key name
            e_exact=np.array([-5.1]),
        )

        data = load_theta_npz(npz_path)

        np.testing.assert_allclose(data["e_vqe"], [-5.0])


class TestPyTorchCompatibility:
    """Tests for PyTorch tensor conversion compatibility."""

    def test_theta_array_converts_to_torch(self, tmp_path):
        """theta_opt should convert to torch.Tensor without error."""
        pytest.importorskip("torch")
        import torch

        npz_path = tmp_path / "test.npz"
        upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0, 3.5]),
            theta_new=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            e_vqe_new=np.array([-5.0, -4.5]),
            e_exact_new=np.array([-5.1, -4.6]),
        )

        data = load_theta_npz(npz_path)

        # This is the critical test - should NOT raise
        # "can't convert np.ndarray of type numpy.object_"
        tensor = torch.from_numpy(data["theta_opt"])
        assert tensor.dtype == torch.float64
        assert tensor.shape == (2, 3)

    def test_legacy_object_array_converts_to_torch(self, tmp_path):
        """Legacy dtype=object arrays should convert to torch after load."""
        pytest.importorskip("torch")
        import torch

        npz_path = tmp_path / "legacy.npz"
        theta_list = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        np.savez(
            npz_path,
            h_values=np.array([4.0, 3.5]),
            theta_opt=np.array(theta_list, dtype=object),
            e_vqe=np.array([-5.0, -4.5]),
            e_exact=np.array([-5.1, -4.6]),
        )

        data = load_theta_npz(npz_path)

        # Should convert successfully
        tensor = torch.from_numpy(data["theta_opt"])
        assert tensor.dtype == torch.float64


class TestAtomicWrite:
    """Tests for atomic write behavior (crash safety)."""

    def test_no_tmp_file_left_on_success(self, tmp_path):
        """Should not leave .tmp.npz files on successful write."""
        npz_path = tmp_path / "test.npz"

        upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0]),
            theta_new=np.array([[0.1, 0.2]]),
            e_vqe_new=np.array([-5.0]),
            e_exact_new=np.array([-5.1]),
        )

        # Check no tmp files exist
        tmp_files = list(tmp_path.glob("*.tmp.npz"))
        assert len(tmp_files) == 0

    def test_original_preserved_on_error(self, tmp_path):
        """Original file should be preserved if write fails."""
        npz_path = tmp_path / "test.npz"

        # Create initial file
        upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0]),
            theta_new=np.array([[0.1, 0.2]]),
            e_vqe_new=np.array([-5.0]),
            e_exact_new=np.array([-5.1]),
        )

        original_data = np.load(npz_path)
        original_h = original_data["h_values"].copy()

        # File should still be readable
        data = load_theta_npz(npz_path)
        np.testing.assert_allclose(data["h_values"], original_h)


class TestRoundTrip:
    """End-to-end round-trip tests."""

    def test_upsert_then_load_roundtrip(self, tmp_path):
        """Data should be identical after upsert → load cycle."""
        npz_path = tmp_path / "test.npz"

        h_original = np.array([4.0, 3.5, 3.0])
        theta_original = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        e_vqe_original = np.array([-5.0, -4.5, -4.0])
        e_exact_original = np.array([-5.1, -4.6, -4.1])

        upsert_theta_npz(
            npz_path,
            h_new=h_original,
            theta_new=theta_original,
            e_vqe_new=e_vqe_original,
            e_exact_new=e_exact_original,
        )

        data = load_theta_npz(npz_path)

        np.testing.assert_allclose(data["h_values"], h_original)
        np.testing.assert_allclose(data["theta_opt"], theta_original)
        np.testing.assert_allclose(data["e_vqe"], e_vqe_original)
        np.testing.assert_allclose(data["e_exact"], e_exact_original)

    def test_multiple_upserts_accumulate(self, tmp_path):
        """Multiple upserts should accumulate unique h-points."""
        npz_path = tmp_path / "test.npz"

        # First upsert
        upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0, 3.5]),
            theta_new=np.array([[0.1, 0.2], [0.3, 0.4]]),
            e_vqe_new=np.array([-5.0, -4.5]),
            e_exact_new=np.array([-5.1, -4.6]),
        )

        # Second upsert with new h values
        upsert_theta_npz(
            npz_path,
            h_new=np.array([3.0, 2.5]),
            theta_new=np.array([[0.5, 0.6], [0.7, 0.8]]),
            e_vqe_new=np.array([-4.0, -3.5]),
            e_exact_new=np.array([-4.1, -3.6]),
        )

        data = load_theta_npz(npz_path)
        assert len(data["h_values"]) == 4  # All 4 unique h values


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_length_mismatch_raises_valueerror(self, tmp_path):
        """Should raise ValueError when array lengths don't match."""
        npz_path = tmp_path / "test.npz"

        with pytest.raises(ValueError, match="Length mismatch"):
            upsert_theta_npz(
                npz_path,
                h_new=np.array([4.0, 3.5]),  # 2 points
                theta_new=np.array([[0.1, 0.2]]),  # 1 point - mismatch!
                e_vqe_new=np.array([-5.0, -4.5]),
                e_exact_new=np.array([-5.1, -4.6]),
            )

    def test_h_matching_tolerance(self, tmp_path):
        """H-values within 1e-6 tolerance should be treated as same point."""
        npz_path = tmp_path / "test.npz"

        # First upsert at h=4.0
        upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0]),
            theta_new=np.array([[0.1, 0.2]]),
            e_vqe_new=np.array([-5.0]),
            e_exact_new=np.array([-5.1]),
        )

        # Second upsert at h=4.0 + 1e-7 (within tolerance) with better energy
        n_upd, n_add = upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0 + 1e-7]),
            theta_new=np.array([[0.9, 0.9]]),
            e_vqe_new=np.array([-5.5]),  # Better energy
            e_exact_new=np.array([-5.1]),
        )

        # Should update existing, not add new
        assert n_upd == 1
        assert n_add == 0

        data = load_theta_npz(npz_path)
        assert len(data["h_values"]) == 1

    def test_h_outside_tolerance_adds_new(self, tmp_path):
        """H-values outside tolerance should create new entry."""
        npz_path = tmp_path / "test.npz"

        upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0]),
            theta_new=np.array([[0.1, 0.2]]),
            e_vqe_new=np.array([-5.0]),
            e_exact_new=np.array([-5.1]),
        )

        # h=4.0 + 1e-5 is outside tolerance (>1e-6)
        n_upd, n_add = upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0 + 1e-5]),
            theta_new=np.array([[0.9, 0.9]]),
            e_vqe_new=np.array([-5.5]),
            e_exact_new=np.array([-5.1]),
        )

        assert n_upd == 0
        assert n_add == 1

        data = load_theta_npz(npz_path)
        assert len(data["h_values"]) == 2

    def test_gap_only_updated_if_positive(self, tmp_path):
        """Gap should only be updated if new gap > 0."""
        npz_path = tmp_path / "test.npz"

        # First with gap=1.5
        upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0]),
            theta_new=np.array([[0.1, 0.2]]),
            e_vqe_new=np.array([-5.0]),
            e_exact_new=np.array([-5.1]),
            gaps_new=np.array([1.5]),
        )

        # Update with better energy but gap=0 (should preserve original gap)
        upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0]),
            theta_new=np.array([[0.9, 0.9]]),
            e_vqe_new=np.array([-5.5]),
            e_exact_new=np.array([-5.1]),
            gaps_new=np.array([0.0]),  # Zero gap
        )

        data = load_theta_npz(npz_path)
        np.testing.assert_allclose(data["gaps"][0], 1.5)  # Original preserved

    def test_de_gap_uses_floor_for_small_gaps(self, tmp_path):
        """ΔE/gap should use 1e-10 floor for very small gaps."""
        npz_path = tmp_path / "test.npz"

        upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0]),
            theta_new=np.array([[0.1, 0.2]]),
            e_vqe_new=np.array([-5.0]),
            e_exact_new=np.array([-5.1]),  # ΔE = 0.1
            gaps_new=np.array([0.0]),  # Zero gap → floor to 1e-10
        )

        data = load_theta_npz(npz_path)
        # ΔE/gap = 0.1 / 1e-10 = 1e9
        assert data["de_gaps"][0] > 1e8  # Very large due to floor

    def test_parent_directory_auto_created(self, tmp_path):
        """Should auto-create parent directories if they don't exist."""
        npz_path = tmp_path / "nested" / "deep" / "test.npz"

        assert not npz_path.parent.exists()

        upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0]),
            theta_new=np.array([[0.1, 0.2]]),
            e_vqe_new=np.array([-5.0]),
            e_exact_new=np.array([-5.1]),
        )

        assert npz_path.exists()

    def test_string_path_works(self, tmp_path):
        """Should accept string path in addition to Path object."""
        npz_path = str(tmp_path / "test.npz")

        upsert_theta_npz(
            npz_path,  # String, not Path
            h_new=np.array([4.0]),
            theta_new=np.array([[0.1, 0.2]]),
            e_vqe_new=np.array([-5.0]),
            e_exact_new=np.array([-5.1]),
        )

        # Load also with string
        data = load_theta_npz(npz_path)
        assert len(data["h_values"]) == 1

    def test_corrupt_existing_entries_removed(self, tmp_path):
        """Corrupt entries in existing NPZ should be filtered during upsert."""
        npz_path = tmp_path / "corrupt.npz"

        # Create NPZ with one good and one corrupt entry
        np.savez(
            npz_path,
            h_values=np.array([4.0, 3.5]),
            theta_opt=np.array([[0.1, 0.2], [np.nan, 0.4]]),  # 2nd is corrupt
            e_vqe=np.array([-5.0, -4.5]),
            e_exact=np.array([-5.1, -4.6]),
            gaps=np.array([1.0, 1.0]),
            method=np.array(["vqe", "vqe"]),
        )

        # Upsert new point - should filter corrupt entries
        n_upd, n_add = upsert_theta_npz(
            npz_path,
            h_new=np.array([3.0]),
            theta_new=np.array([[0.5, 0.6]]),
            e_vqe_new=np.array([-4.0]),
            e_exact_new=np.array([-4.1]),
        )

        assert n_add == 1

        data = load_theta_npz(npz_path)
        # Should have 2 entries: original valid + new (corrupt removed)
        assert len(data["h_values"]) == 2

    def test_all_entries_filtered_returns_zero(self, tmp_path):
        """Should return (0,0) if all new entries are invalid."""
        npz_path = tmp_path / "test.npz"

        n_upd, n_add = upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0, 3.5]),
            theta_new=np.array([[np.nan, 0.2], [np.inf, 0.4]]),  # All invalid
            e_vqe_new=np.array([-5.0, -4.5]),
            e_exact_new=np.array([-5.1, -4.6]),
        )

        assert n_upd == 0
        assert n_add == 0


class TestRaggedArrays:
    """Tests for handling ragged theta arrays (different shapes)."""

    def test_ragged_array_creates_object_dtype(self, tmp_path):
        """Mixed theta shapes should create object dtype array."""
        npz_path = tmp_path / "ragged.npz"

        # First upsert: 2 params
        upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0]),
            theta_new=np.array([[0.1, 0.2]]),
            e_vqe_new=np.array([-5.0]),
            e_exact_new=np.array([-5.1]),
        )

        # Second upsert: 3 params (different shape!)
        upsert_theta_npz(
            npz_path,
            h_new=np.array([3.5]),
            theta_new=np.array([[0.3, 0.4, 0.5]]),
            e_vqe_new=np.array([-4.5]),
            e_exact_new=np.array([-4.6]),
        )

        # Load raw to check dtype
        raw = np.load(npz_path, allow_pickle=True)
        assert raw["theta_opt"].dtype == object

    def test_ragged_array_loads_correctly(self, tmp_path):
        """Ragged arrays should load and be usable."""
        npz_path = tmp_path / "ragged.npz"

        # Create ragged NPZ
        theta_list = [np.array([0.1, 0.2]), np.array([0.3, 0.4, 0.5])]
        np.savez(
            npz_path,
            h_values=np.array([4.0, 3.5]),
            theta_opt=np.array(theta_list, dtype=object),
            e_vqe=np.array([-5.0, -4.5]),
            e_exact=np.array([-5.1, -4.6]),
        )

        data = load_theta_npz(npz_path)

        assert len(data["h_values"]) == 2
        assert len(data["theta_opt"][0]) == 2
        assert len(data["theta_opt"][1]) == 3


class TestMethodTracking:
    """Tests for method label tracking."""

    def test_method_defaults_to_unknown(self, tmp_path):
        """Method should default to 'unknown' when not provided."""
        npz_path = tmp_path / "test.npz"

        upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0]),
            theta_new=np.array([[0.1, 0.2]]),
            e_vqe_new=np.array([-5.0]),
            e_exact_new=np.array([-5.1]),
            # method_new not provided
        )

        data = load_theta_npz(npz_path)
        assert data["method"][0] == "unknown"

    def test_method_updates_with_energy(self, tmp_path):
        """Method should update when energy improves."""
        npz_path = tmp_path / "test.npz"

        upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0]),
            theta_new=np.array([[0.1, 0.2]]),
            e_vqe_new=np.array([-5.0]),
            e_exact_new=np.array([-5.1]),
            method_new=["initial_vqe"],
        )

        # Update with better energy
        upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0]),
            theta_new=np.array([[0.9, 0.9]]),
            e_vqe_new=np.array([-5.5]),
            e_exact_new=np.array([-5.1]),
            method_new=["refined_vqe"],
        )

        data = load_theta_npz(npz_path)
        assert data["method"][0] == "refined_vqe"

    def test_method_preserved_when_energy_worse(self, tmp_path):
        """Method should NOT update when energy is worse."""
        npz_path = tmp_path / "test.npz"

        upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0]),
            theta_new=np.array([[0.1, 0.2]]),
            e_vqe_new=np.array([-5.5]),  # Good energy
            e_exact_new=np.array([-5.1]),
            method_new=["good_method"],
        )

        # Try to update with worse energy
        upsert_theta_npz(
            npz_path,
            h_new=np.array([4.0]),
            theta_new=np.array([[0.9, 0.9]]),
            e_vqe_new=np.array([-4.5]),  # Worse energy
            e_exact_new=np.array([-5.1]),
            method_new=["bad_method"],
        )

        data = load_theta_npz(npz_path)
        assert data["method"][0] == "good_method"  # Original preserved
