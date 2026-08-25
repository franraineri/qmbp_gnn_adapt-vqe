"""Tests for new utilities: quality_score, refresh_npz_ground_truth, select_best_theta_init."""

import numpy as np


class TestComputeTrainingQualityScore:
    """Test compute_training_quality_score formula correctness."""

    def test_all_verified_high_coverage(self, tmp_path, monkeypatch):
        """100% verified + 100 pts → score near 1.0."""
        from qmbp_simulation.predictors.model_zoo import compute_training_quality_score

        npz_dir = tmp_path / "data" / "multi_n_training"
        npz_dir.mkdir(parents=True)
        # Create NPZ with 100 verified points, all passing dual criterion
        h = np.linspace(2.0, 4.0, 100)
        theta = np.random.randn(100, 10)
        e_vqe = -np.arange(100, dtype=float) - 10  # arbitrary
        e_exact = e_vqe - 0.01  # |ΔE| = 0.01 < 0.10 → passes
        gaps = np.ones(100) * 2.0  # ΔE/gap = 0.005 < 0.05 → passes
        tiers = np.array(["verified"] * 100)
        np.savez(
            npz_dir / "test_N10_p1.npz",
            h_values=h,
            theta_opt=theta,
            e_vqe=e_vqe,
            e_exact=e_exact,
            gaps=gaps,
            quality_tier=tiers,
            de_gaps=np.abs(e_vqe - e_exact) / gaps,
        )

        monkeypatch.chdir(tmp_path)
        score = compute_training_quality_score("test", n_qubits=10, p_layers=1)
        # 0.50*1.0 + 0.20*1.0 + 0.30*1.0 = 1.0
        assert score >= 0.95

    def test_all_unverified_low_coverage(self, tmp_path, monkeypatch):
        """0% verified + 5 pts → low score."""
        from qmbp_simulation.predictors.model_zoo import compute_training_quality_score

        npz_dir = tmp_path / "data" / "multi_n_training"
        npz_dir.mkdir(parents=True)
        h = np.linspace(2.0, 3.0, 5)
        theta = np.random.randn(5, 10)
        e_vqe = np.zeros(5)
        e_exact = -np.ones(5) * 10  # |ΔE| = 10 → fails everything
        gaps = np.ones(5)
        tiers = np.array(["unverified"] * 5)
        np.savez(
            npz_dir / "test_N10_p1.npz",
            h_values=h,
            theta_opt=theta,
            e_vqe=e_vqe,
            e_exact=e_exact,
            gaps=gaps,
            quality_tier=tiers,
            de_gaps=np.abs(e_vqe - e_exact) / gaps,
        )

        monkeypatch.chdir(tmp_path)
        score = compute_training_quality_score("test", n_qubits=10, p_layers=1)
        # 0.50*0.0 + 0.20*0.05 + 0.30*0.0 = 0.01
        assert score < 0.05

    def test_no_data_returns_zero(self, tmp_path, monkeypatch):
        """No NPZ files → 0.0."""
        from qmbp_simulation.predictors.model_zoo import compute_training_quality_score

        monkeypatch.chdir(tmp_path)
        score = compute_training_quality_score("nonexistent", n_qubits=0, p_layers=1)
        assert score == 0.0


class TestRefreshNpzGroundTruth:
    """Test refresh_npz_ground_truth updates stale e_exact."""

    def test_updates_stale_energy(self, tmp_path):
        """If GT cache has lower energy, NPZ e_exact should be updated."""
        from unittest.mock import MagicMock, patch

        from qmbp_simulation.framework.result_io import refresh_npz_ground_truth

        npz_path = tmp_path / "chain_1d_N6_p1.npz"
        h_values = np.array([2.0, 3.0, 4.0])
        e_exact_stale = np.array([-10.0, -8.0, -5.0])  # Stale (higher)
        gaps = np.array([1.0, 1.5, 2.0])
        theta = np.random.randn(3, 10)
        e_vqe = np.array([-9.9, -7.9, -4.9])
        np.savez(
            npz_path,
            h_values=h_values,
            theta_opt=theta,
            e_vqe=e_vqe,
            e_exact=e_exact_stale,
            gaps=gaps,
        )

        # Mock GroundTruthCache to return lower (more accurate) energy at h=2.0
        mock_cache_instance = MagicMock()

        def _mock_get(topo, n, model, h):
            if abs(h - 2.0) < 0.01:
                return {"energy": -10.5, "gap": 1.2}
            return None

        mock_cache_instance.get = _mock_get

        mock_cache_class = MagicMock(return_value=mock_cache_instance)

        with patch("qmbp_simulation.solvers.ground_truth_cache.GroundTruthCache", mock_cache_class):
            n_refreshed = refresh_npz_ground_truth(
                npz_path,
                topology="chain_1d",
                n_qubits=6,
            )

        # Verify: h=2.0 should have been updated
        assert n_refreshed == 1
        data = np.load(npz_path)
        np.testing.assert_allclose(data["e_exact"][0], -10.5, atol=1e-10)

    def test_no_file_returns_zero(self, tmp_path):
        """Non-existent file → 0."""
        from qmbp_simulation.framework.result_io import refresh_npz_ground_truth

        assert refresh_npz_ground_truth(tmp_path / "nope.npz", "x", 6) == 0


class TestSelectBestThetaInit:
    """Test select_best_theta_init picks the lower-energy candidate."""

    def test_pred_better(self):
        """When prediction has lower energy, returns prediction."""
        from qmbp_simulation.framework.result_io import select_best_theta_init

        theta_pred = np.array([0.1, 0.2, 0.3])
        theta_prev = np.array([0.4, 0.5, 0.6])
        best_theta, best_e = select_best_theta_init(
            theta_pred,
            e_pred=-10.5,
            theta_prev=theta_prev,
            e_prev=-10.0,
        )
        np.testing.assert_array_equal(best_theta, theta_pred)
        assert best_e == -10.5

    def test_prev_better(self):
        """When previous has lower energy, returns previous."""
        from qmbp_simulation.framework.result_io import select_best_theta_init

        theta_pred = np.array([0.1, 0.2, 0.3])
        theta_prev = np.array([0.4, 0.5, 0.6])
        best_theta, best_e = select_best_theta_init(
            theta_pred,
            e_pred=-9.0,
            theta_prev=theta_prev,
            e_prev=-10.0,
        )
        np.testing.assert_array_equal(best_theta, theta_prev)
        assert best_e == -10.0

    def test_no_prev_returns_pred(self):
        """When theta_prev is None, returns prediction."""
        from qmbp_simulation.framework.result_io import select_best_theta_init

        theta_pred = np.array([0.1, 0.2])
        best_theta, best_e = select_best_theta_init(
            theta_pred,
            e_pred=-5.0,
            theta_prev=None,
            e_prev=None,
        )
        np.testing.assert_array_equal(best_theta, theta_pred)
        assert best_e == -5.0

    def test_prev_no_energy_with_eval_fn(self):
        """When e_prev is None but eval_fn provided, evaluates θ_prev."""
        from qmbp_simulation.framework.result_io import select_best_theta_init

        theta_pred = np.array([0.1, 0.2])
        theta_prev = np.array([0.3, 0.4])

        # eval_fn says θ_prev gives -11.0 (better than pred's -10.0)
        best_theta, best_e = select_best_theta_init(
            theta_pred,
            e_pred=-10.0,
            theta_prev=theta_prev,
            e_prev=None,
            eval_fn=lambda t: -11.0,
        )
        np.testing.assert_array_equal(best_theta, theta_prev)
        assert best_e == -11.0

    def test_prev_no_energy_no_eval_returns_pred(self):
        """When e_prev is None and no eval_fn, returns prediction."""
        from qmbp_simulation.framework.result_io import select_best_theta_init

        theta_pred = np.array([0.1, 0.2])
        theta_prev = np.array([0.3, 0.4])
        best_theta, best_e = select_best_theta_init(
            theta_pred,
            e_pred=-5.0,
            theta_prev=theta_prev,
            e_prev=None,
            eval_fn=None,
        )
        np.testing.assert_array_equal(best_theta, theta_pred)
        assert best_e == -5.0
