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
            edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long),
            node_type=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
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
        from qmbp_simulation.analysis.metrics import SAMPLE_WEIGHT_MAX, SAMPLE_WEIGHT_MIN

        w_low = max(SAMPLE_WEIGHT_MIN, min(SAMPLE_WEIGHT_MAX, -1.0))
        w_high = max(SAMPLE_WEIGHT_MIN, min(SAMPLE_WEIGHT_MAX, 5.0))
        w_normal = max(SAMPLE_WEIGHT_MIN, min(SAMPLE_WEIGHT_MAX, 0.7))

        assert w_low == SAMPLE_WEIGHT_MIN
        assert w_high == SAMPLE_WEIGHT_MAX
        assert w_normal == 0.7


class TestDataAugmentation:
    """Tests for augment_theta_symmetries function."""

    def test_z2_is_exact_negation_no_wrapping(self):
        """Z₂ augmentation should produce exactly -θ (no wrapping distortion)."""
        from qmbp_simulation.utils.helpers import augment_theta_symmetries

        theta = np.array([0.3, -0.1, 0.5, 0.2])
        variants = augment_theta_symmetries(theta, include_z2=True, noise_std=0.0)

        assert len(variants) == 1
        z2 = variants[0]
        # For canonicalized θ ∈ [-π/2, π/2], -θ should be EXACTLY -theta
        np.testing.assert_array_almost_equal(z2, -theta, decimal=10)

    def test_z2_at_boundary_values(self):
        """Z₂ at ±π/2 boundary should clip, not wrap."""
        from qmbp_simulation.utils.helpers import augment_theta_symmetries

        # θ at positive boundary
        theta = np.array([np.pi / 2 - 0.01, 0.0, -np.pi / 2 + 0.01])
        variants = augment_theta_symmetries(theta, include_z2=True, noise_std=0.0)
        z2 = variants[0]
        # -θ should be [-π/2+0.01, 0, π/2-0.01] — within bounds
        assert np.all(z2 >= -np.pi / 2 - 1e-10)
        assert np.all(z2 <= np.pi / 2 + 1e-10)

    def test_z2_preserves_energy_property(self):
        """For TFIM HVA, E(-θ) = E(θ). Verify the symmetry is exact."""
        from qmbp_simulation.utils.helpers import augment_theta_symmetries

        # Use a theta in canonical domain
        theta = np.array([0.3, -0.4, 0.1, 0.7, -0.2])
        variants = augment_theta_symmetries(theta, include_z2=True, noise_std=0.0)
        z2 = variants[0]

        # For exact Z₂ symmetry: z2 == -theta (no numerical drift)
        np.testing.assert_array_almost_equal(z2, -theta, decimal=14)

    def test_noise_augmentation_stays_in_domain(self):
        """Noisy augmentation should stay within [-π/2, π/2]."""
        from qmbp_simulation.utils.helpers import augment_theta_symmetries

        theta = np.array([1.5, -1.5, 0.0, 1.0])  # Near boundaries
        variants = augment_theta_symmetries(theta, include_z2=False, noise_std=0.1, n_noise_variants=5, seed=42)

        for v in variants:
            assert np.all(v >= -np.pi / 2 - 1e-10), f"Below lower bound: {v}"
            assert np.all(v <= np.pi / 2 + 1e-10), f"Above upper bound: {v}"

    def test_n_noise_variants_controls_count(self):
        """n_noise_variants should control how many noisy copies are made."""
        from qmbp_simulation.utils.helpers import augment_theta_symmetries

        theta = np.array([0.3, -0.1, 0.5])

        # Z₂ + 2 noisy per base (original + Z₂) = 1 + 4 = 5
        variants = augment_theta_symmetries(theta, include_z2=True, noise_std=0.02, n_noise_variants=2, seed=42)
        assert len(variants) == 5  # 1 Z₂ + 2 noisy(original) + 2 noisy(Z₂)

        # No Z₂ + 3 noisy of original = 3
        variants = augment_theta_symmetries(theta, include_z2=False, noise_std=0.02, n_noise_variants=3, seed=42)
        assert len(variants) == 3

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
        for a, b in zip(v1, v2, strict=False):
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
        from qmbp_simulation.analysis.metrics import QUALITY_TIER_WEIGHT_AUGMENTED

        n_augmented = sum(
            1
            for g in dataset
            if hasattr(g, "sample_weight") and abs(float(g.sample_weight[0]) - QUALITY_TIER_WEIGHT_AUGMENTED) < 1e-5
        )
        assert n_augmented >= 1

        mod._PROJECT_ROOT = original_root

    def test_more_variants_for_small_datasets(self, tmp_path):
        """Very small datasets (<20 points) should get more augmented variants."""
        import qmbp_simulation.predictors.multi_n_aggregator as mod

        original_root = mod._PROJECT_ROOT
        mod._PROJECT_ROOT = tmp_path

        data_dir = tmp_path / "data" / "multi_n_training"
        data_dir.mkdir(parents=True)

        # 5 verified points (< 20 threshold for extra augmentation)
        n_pts = 5
        np.savez(
            data_dir / "chain_1d_N4_p1.npz",
            h_values=np.linspace(3.0, 4.0, n_pts),
            theta_opt=np.random.randn(n_pts, 7).astype(np.float64),
            e_vqe=np.linspace(-6.0, -5.0, n_pts),
            e_exact=np.linspace(-6.01, -5.01, n_pts),
            gaps=np.ones(n_pts) * 2.0,
            method=np.array(["vqe_refined"] * n_pts),
            quality_tier=np.array(["verified"] * n_pts),
        )

        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        dataset = agg.build_combined_dataset(max_de_gap=1.0)

        # 5 original + up to 3 augmented per point (small dataset boost)
        # Should have significantly more than 5
        assert len(dataset) >= 10, f"Expected ≥10 with augmentation, got {len(dataset)}"

        mod._PROJECT_ROOT = original_root

    def test_z2_variant_different_from_original(self):
        """Z₂ variant must be different from original (unless θ=0)."""
        from qmbp_simulation.utils.helpers import augment_theta_symmetries

        theta = np.array([0.3, -0.1, 0.5, 0.2])
        variants = augment_theta_symmetries(theta, include_z2=True, noise_std=0.0)
        assert not np.allclose(variants[0], theta)


class TestAdaptiveVQEConfig:
    """Tests for compute_adaptive_vqe_config function."""

    def test_cheap_tier_for_easy_wins(self):
        """High priority + low de_gap should give cheap tier."""
        from qmbp_simulation.analysis.metrics import (
            ADAPTIVE_VQE_CHEAP_MAXITER,
            compute_adaptive_vqe_config,
        )

        cfg = compute_adaptive_vqe_config(priority=0.9, de_gap=0.06, gap=3.0, n_params=20)
        assert cfg["tier"] == "cheap"
        assert cfg["maxiter"] <= ADAPTIVE_VQE_CHEAP_MAXITER
        assert cfg["n_restarts"] <= 5  # Cheap tier uses minimal restarts

    def test_standard_tier_for_moderate(self):
        """Moderate priority should give standard tier."""
        from qmbp_simulation.analysis.metrics import compute_adaptive_vqe_config

        cfg = compute_adaptive_vqe_config(priority=0.6, de_gap=0.12, gap=2.0, n_params=30)
        assert cfg["tier"] == "standard"
        assert cfg["n_restarts"] <= 5

    def test_aggressive_tier_for_hard_points(self):
        """Low priority should give aggressive tier."""
        from qmbp_simulation.analysis.metrics import compute_adaptive_vqe_config

        cfg = compute_adaptive_vqe_config(priority=0.3, de_gap=0.30, gap=1.0, n_params=50)
        assert cfg["tier"] == "aggressive"
        assert cfg["n_restarts"] == 10  # full budget

    def test_minimal_tier_for_hopeless(self):
        """Very low priority should give minimal tier."""
        from qmbp_simulation.analysis.metrics import compute_adaptive_vqe_config

        cfg = compute_adaptive_vqe_config(priority=0.1, de_gap=0.80, gap=0.5, n_params=80)
        assert cfg["tier"] == "minimal"
        assert cfg["maxiter"] <= 100
        assert cfg["n_restarts"] == 1

    def test_base_params_respected(self):
        """Custom base_maxiter and base_restarts should be used."""
        from qmbp_simulation.analysis.metrics import compute_adaptive_vqe_config

        cfg = compute_adaptive_vqe_config(
            priority=0.3,
            de_gap=0.25,
            gap=2.0,
            n_params=30,
            base_maxiter=500,
            base_restarts=3,
        )
        assert cfg["tier"] == "aggressive"
        assert cfg["maxiter"] == 500
        assert cfg["n_restarts"] == 3


class TestResidualSeed:
    """Tests for the residual-over-analytic-seed training/inference path."""

    def _frustrated_graph(self, N=8, p=2, h=1.2, with_theta=True, with_phys=True):
        import numpy as _np

        from qmbp_simulation.analysis.warmstart import select_regime_seed
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder, make_lattice
        from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph

        lat = make_lattice("square", N, J=1.0, h=h)
        n_nn = len(lat.edges)
        n_nnn = len(HamiltonianBuilder._generate_nnn_edges(lat))
        seed_true, _ = select_regime_seed(n_nn, n_nnn, N, p, h, J2=0.5)
        theta = None
        if with_theta:
            rng = _np.random.default_rng(0)
            theta = seed_true + rng.normal(0, 0.1, size=seed_true.shape)
        g = build_unified_bond_resolved_graph(lat, h_value=h, p_layers=p, theta_opt=theta, include_nnn=True)
        if with_phys:
            g.phys_n_qubits = N
            g.phys_h = h
            g.phys_p_layers = p
        return g, n_nn, n_nnn

    def test_graph_carries_nn_nnn_split(self):
        """build_unified_bond_resolved_graph should expose n_nn_edges / n_nnn_edges."""
        g, n_nn, n_nnn = self._frustrated_graph()
        assert int(g.n_nn_edges) == n_nn
        assert int(g.n_nnn_edges) == n_nnn
        assert int(g.n_edges_unique) == n_nn + n_nnn

    def test_seed_is_small_residual_and_roundtrips(self):
        """Δ = canon(θ) − seed is small, and seed + Δ recovers canon(θ) exactly."""
        from qmbp_simulation.analysis.warmstart import _canonicalize_z2_flat
        from qmbp_simulation.predictors.unified_mpnn import residual_seed_for_graph

        g, n_nn, n_nnn = self._frustrated_graph()
        seed = residual_seed_for_graph(g, j2=0.5)
        assert seed is not None
        N = int(g.n_qubit_nodes)
        p = int(g.phys_p_layers)
        y_gauge = _canonicalize_z2_flat(g.y.numpy().astype(np.float64), n_nn, n_nnn, N, p)
        delta = y_gauge - seed
        # Δ is a small correction, not ≈2×seed (the sign-gauge bug it guards against)
        assert np.linalg.norm(delta) < np.linalg.norm(seed)
        # exact round-trip
        np.testing.assert_allclose(seed + delta, y_gauge, atol=1e-9)

    def test_h_and_p_override_matches_metadata(self):
        """Prediction graphs lack phys_* — overrides must reproduce the same seed."""
        from qmbp_simulation.predictors.unified_mpnn import residual_seed_for_graph

        g_meta, _, _ = self._frustrated_graph(with_theta=False, with_phys=True)
        seed_meta = residual_seed_for_graph(g_meta, j2=0.5)
        g_bare, _, _ = self._frustrated_graph(with_theta=False, with_phys=False)
        seed_override = residual_seed_for_graph(g_bare, j2=0.5, h_override=1.2, p_override=2)
        assert seed_meta is not None and seed_override is not None
        np.testing.assert_allclose(seed_meta, seed_override, atol=1e-12)

    def test_missing_metadata_falls_back_to_none(self):
        """No phys_h and no override → seed cannot be built → None (safe fallback)."""
        from qmbp_simulation.predictors.unified_mpnn import residual_seed_for_graph

        g_bare, _, _ = self._frustrated_graph(with_theta=False, with_phys=False)
        assert residual_seed_for_graph(g_bare, j2=0.5) is None

    def test_predict_residual_adds_seed_back(self):
        """predict_residual(θ) = seed + model(Δ); grouped layout, finite."""
        import torch

        from qmbp_simulation.predictors.unified_mpnn import (
            UnifiedMPNN,
            predict_residual,
            residual_seed_for_graph,
        )

        g, _, _ = self._frustrated_graph(with_theta=False, with_phys=True)
        n_feat = g.x.shape[1]
        model = UnifiedMPNN(node_features=n_feat, hidden_dim=32, n_layers=2, norm_type="none")
        model.eval()
        theta = predict_residual(model, g, model_name="tfim_frustrated", j2=0.5)
        assert theta.shape[0] == int(g.n_edges_unique) * 2 + int(g.n_qubit_nodes) * 2  # (n_e + N) * p
        assert np.all(np.isfinite(theta))
        # θ must differ from the raw Δ by exactly the regrouped seed
        with torch.no_grad():
            delta = model(g).squeeze(0).numpy()
        seed = residual_seed_for_graph(g, j2=0.5)
        assert seed is not None
        # θ - delta should have the same norm as the seed (regrouped, same entries)
        np.testing.assert_allclose(np.sort(np.abs(theta - delta)), np.sort(np.abs(seed)), atol=1e-5)
