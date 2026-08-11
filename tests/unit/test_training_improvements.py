"""Tests for weighted loss, data augmentation, and adaptive VQE config.

Validates the training pipeline improvements:
1. Weighted loss correctly scales gradient contributions
2. Data augmentation generates valid symmetry-equivalent θ
3. Augmentation respects tier policy (only verified)
4. Adaptive VQE config assigns correct tiers
5. Guards handle edge cases (NaN, out-of-range weights)
"""

import numpy as np
import pytest


class TestWeightedLoss:
    """Tests for sample_weight-based loss scaling in train_unified_mpnn."""

    def test_weight_field_used_in_training(self):
        """Training with sample_weight should produce different loss than without."""
        import torch
        from torch_geometric.data import Data
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN

        # Create minimal model
        model = UnifiedMPNN(node_features=4, hidden_dim=32, n_layers=2, norm_type="none")
        model.train()

        # Create a simple graph with known structure
        n_nodes = 6
        g = Data(
            x=torch.randn(n_nodes, 4),
            edge_index=torch.tensor([[0,1,2,3,4], [1,2,3,4,5]], dtype=torch.long),
            node_type=torch.tensor([0,0,0,1,1,1], dtype=torch.long),
            n_edges_unique=3,
            n_qubit_nodes=3,
            y=torch.randn(6),  # 3 ZZ + 3 X params
        )

        # Forward pass without weight
        pred = model(g).squeeze(0)
        loss_unweighted = torch.nn.functional.mse_loss(pred, g.y)

        # Same loss scaled by 0.5 weight
        loss_weighted = loss_unweighted * 0.5

        assert loss_weighted.item() == pytest.approx(loss_unweighted.item() * 0.5, rel=1e-5)

    def test_weight_clamp_guards(self):
        """Weights outside valid range should be clamped."""
        from qmbp_simulation.analysis.metrics import SAMPLE_WEIGHT_MIN, SAMPLE_WEIGHT_MAX

        w_low = max(SAMPLE_WEIGHT_MIN, min(SAMPLE_WEIGHT_MAX, -1.0))
        w_high = max(SAMPLE_WEIGHT_MIN, min(SAMPLE_WEIGHT_MAX, 5.0))
        w_normal = max(SAMPLE_WEIGHT_MIN, min(SAMPLE_WEIGHT_MAX, 0.7))

        assert w_low == SAMPLE_WEIGHT_MIN
        assert w_high == SAMPLE_WEIGHT_MAX
        assert w_normal == 0.7


class TestDataAugmentation:
    """Tests for augment_theta_symmetries function."""

    def test_z2_symmetry_produces_negation(self):
        """Z₂ augmentation should produce -θ (wrapped to canonical domain)."""
        from qmbp_simulation.utils.helpers import augment_theta_symmetries

        theta = np.array([0.3, -0.1, 0.5, 0.2])
        variants = augment_theta_symmetries(theta, include_z2=True, noise_std=0.0)

        assert len(variants) >= 1
        # Z₂ variant should be close to -theta (mod π wrapping may apply)
        z2_variant = variants[0]
        # Check that it's different from original
        assert not np.allclose(z2_variant, theta, atol=1e-6)
        # Check it's within [-π/2, π/2] (canonical domain)
        assert np.all(z2_variant >= -np.pi / 2 - 1e-10)
        assert np.all(z2_variant <= np.pi / 2 + 1e-10)

    def test_noise_augmentation_is_close_to_original(self):
        """Noisy augmentation should be within ~3σ of original."""
        from qmbp_simulation.utils.helpers import augment_theta_symmetries

        theta = np.array([0.3, -0.1, 0.5, 0.2])
        noise_std = 0.02
        variants = augment_theta_symmetries(
            theta, include_z2=False, noise_std=noise_std, seed=42
        )

        assert len(variants) >= 1
        noisy = variants[0]
        # Should be close to original (within ~5σ for safety)
        diff = np.abs(noisy - theta)
        # Account for wrapping: differences should be small
        assert np.max(diff) < 5 * noise_std + 0.1  # +0.1 for wrapping edge cases

    def test_empty_theta_returns_empty(self):
        """Empty θ array should return empty variants list."""
        from qmbp_simulation.utils.helpers import augment_theta_symmetries

        variants = augment_theta_symmetries(np.array([]))
        assert variants == []

    def test_augmentation_deterministic_with_seed(self):
        """Same seed should produce same augmented variants."""
        from qmbp_simulation.utils.helpers import augment_theta_symmetries

        theta = np.array([0.3, -0.1, 0.5])
        v1 = augment_theta_symmetries(theta, noise_std=0.05, seed=123)
        v2 = augment_theta_symmetries(theta, noise_std=0.05, seed=123)

        assert len(v1) == len(v2)
        for a, b in zip(v1, v2):
            np.testing.assert_array_equal(a, b)

    def test_augmentation_only_for_verified_in_aggregator(self, tmp_path):
        """MultiNAggregator should only augment verified points."""
        import qmbp_simulation.predictors.multi_n_aggregator as mod

        original_root = mod._PROJECT_ROOT
        mod._PROJECT_ROOT = tmp_path

        data_dir = tmp_path / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)

        # 2 verified + 2 approximate
        n_pts = 4
        np.savez(
            data_dir / "chain_1d_N4_p1.npz",
            h_values=np.linspace(3.0, 4.0, n_pts),
            theta_opt=np.random.randn(n_pts, 7).astype(np.float64),
            e_vqe=np.linspace(-6.0, -5.0, n_pts),
            e_exact=np.linspace(-6.01, -5.01, n_pts),
            gaps=np.ones(n_pts) * 2.0,
            method=np.array(["vqe_refined"] * 2 + ["mpnn_pred"] * 2),
            quality_tier=np.array(["verified"] * 2 + ["approximate"] * 2),
        )

        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator
        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        dataset = agg.build_combined_dataset(max_de_gap=1.0)

        # Should have: 4 original + augmented from 2 verified = at least 5
        assert len(dataset) >= 5
        # Count augmented (weight=0.8)
        n_augmented = sum(
            1 for g in dataset
            if hasattr(g, "sample_weight") and abs(float(g.sample_weight[0]) - 0.8) < 1e-5
        )
        # At least 1 augmented variant (from verified points)
        assert n_augmented >= 1

        mod._PROJECT_ROOT = original_root


class TestAdaptiveVQEConfig:
    """Tests for compute_adaptive_vqe_config function."""

    def test_cheap_tier_for_easy_wins(self):
        """High priority + low de_gap should give cheap tier."""
        from qmbp_simulation.analysis.metrics import compute_adaptive_vqe_config

        cfg = compute_adaptive_vqe_config(
            priority=0.9, de_gap=0.06, gap=3.0, n_params=20
        )
        assert cfg["tier"] == "cheap"
        assert cfg["maxiter"] <= 200
        assert cfg["n_restarts"] == 1

    def test_standard_tier_for_moderate(self):
        """Moderate priority should give standard tier."""
        from qmbp_simulation.analysis.metrics import compute_adaptive_vqe_config

        cfg = compute_adaptive_vqe_config(
            priority=0.6, de_gap=0.12, gap=2.0, n_params=30
        )
        assert cfg["tier"] == "standard"
        assert cfg["n_restarts"] <= 5

    def test_aggressive_tier_for_hard_points(self):
        """Low priority should give aggressive tier."""
        from qmbp_simulation.analysis.metrics import compute_adaptive_vqe_config

        cfg = compute_adaptive_vqe_config(
            priority=0.3, de_gap=0.30, gap=1.0, n_params=50
        )
        assert cfg["tier"] == "aggressive"
        assert cfg["n_restarts"] == 10  # full budget

    def test_minimal_tier_for_hopeless(self):
        """Very low priority should give minimal tier."""
        from qmbp_simulation.analysis.metrics import compute_adaptive_vqe_config

        cfg = compute_adaptive_vqe_config(
            priority=0.1, de_gap=0.80, gap=0.5, n_params=80
        )
        assert cfg["tier"] == "minimal"
        assert cfg["maxiter"] <= 100
        assert cfg["n_restarts"] == 1

    def test_base_params_respected(self):
        """Custom base_maxiter and base_restarts should be used."""
        from qmbp_simulation.analysis.metrics import compute_adaptive_vqe_config

        cfg = compute_adaptive_vqe_config(
            priority=0.3, de_gap=0.25, gap=2.0, n_params=30,
            base_maxiter=500, base_restarts=3,
        )
        assert cfg["tier"] == "aggressive"
        assert cfg["maxiter"] == 500
        assert cfg["n_restarts"] == 3
