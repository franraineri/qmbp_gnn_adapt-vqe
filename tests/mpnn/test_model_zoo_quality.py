"""Tests for model_zoo quality tier integration functions."""
import tempfile
from pathlib import Path
import numpy as np
import pytest


class TestGetTrainingDataQuality:
    """Tests for get_training_data_quality function."""

    def test_returns_not_found_for_missing_npz(self, tmp_path, monkeypatch):
        """When NPZ doesn't exist, should return found=False."""
        from qmbp_simulation.predictors import model_zoo
        monkeypatch.setattr(model_zoo, "_PROJECT_ROOT", tmp_path)
        quality = model_zoo.get_training_data_quality("chain_1d", 6, "tfim_bond_resolved", 1)
        assert quality["found"] is False
        assert quality["n_points"] == 0
        assert len(quality["warnings"]) > 0

    def test_reads_quality_tier_from_npz(self, tmp_path, monkeypatch):
        """When NPZ has quality_tier, should read and count them."""
        from qmbp_simulation.predictors import model_zoo
        data_dir = tmp_path / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)
        monkeypatch.setattr(model_zoo, "_PROJECT_ROOT", tmp_path)
        n_pts = 10
        np.savez(
            data_dir / "chain_1d_N6_p1.npz",
            h_values=np.linspace(2.0, 4.0, n_pts),
            theta_opt=np.random.randn(n_pts, 7),
            e_vqe=np.linspace(-8, -6, n_pts),
            e_exact=np.linspace(-8.1, -6.1, n_pts),
            gaps=np.ones(n_pts) * 2.0,
            quality_tier=np.array(["verified"] * 4 + ["approximate"] * 3 + ["unverified"] * 3),
        )
        quality = model_zoo.get_training_data_quality("chain_1d", 6, "tfim_bond_resolved", 1)
        assert quality["found"] is True
        assert quality["n_points"] == n_pts
        assert quality["n_verified"] == 4
        assert quality["n_approximate"] == 3
        assert quality["n_unverified"] == 3
        assert 0.6 < quality["quality_score"] < 0.9


class TestValidationRunnerQualityHelpers:
    """Tests for ValidationRunner quality tier helpers."""

    def test_get_npz_quality_tiers_exists(self):
        """ValidationRunner should have get_npz_quality_tiers static method."""
        from qmbp_simulation.framework.runner_base import ValidationRunner
        assert hasattr(ValidationRunner, "get_npz_quality_tiers")

    def test_get_npz_quality_tiers_with_tiers(self, tmp_path):
        """get_npz_quality_tiers should read tier distribution."""
        from qmbp_simulation.framework.runner_base import ValidationRunner
        npz_path = tmp_path / "test.npz"
        n_pts = 10
        np.savez(
            npz_path,
            h_values=np.linspace(2.0, 4.0, n_pts),
            theta_opt=np.random.randn(n_pts, 7),
            quality_tier=np.array(["verified"] * 5 + ["approximate"] * 3 + ["unverified"] * 2),
        )
        result = ValidationRunner.get_npz_quality_tiers(npz_path)
        assert result["n_verified"] == 5
        assert result["n_approximate"] == 3
        assert result["n_unverified"] == 2
        assert result["n_total"] == 10
        assert result["has_quality_tier"] is True
        assert result["verified_ratio"] == pytest.approx(0.5, abs=0.01)

    def test_get_npz_quality_tiers_without_tiers(self, tmp_path):
        """Legacy NPZ should return all as unverified."""
        from qmbp_simulation.framework.runner_base import ValidationRunner
        npz_path = tmp_path / "legacy.npz"
        n_pts = 6
        np.savez(npz_path, h_values=np.linspace(2.0, 4.0, n_pts), theta_opt=np.random.randn(n_pts, 7))
        result = ValidationRunner.get_npz_quality_tiers(npz_path)
        assert result["n_unverified"] == n_pts
        assert result["n_verified"] == 0
        assert result["has_quality_tier"] is False

    def test_get_npz_quality_tiers_missing_file(self, tmp_path):
        """Missing file should return zeros."""
        from qmbp_simulation.framework.runner_base import ValidationRunner
        result = ValidationRunner.get_npz_quality_tiers(tmp_path / "missing.npz")
        assert result["n_total"] == 0
        assert result["quality_score"] == 0.0
