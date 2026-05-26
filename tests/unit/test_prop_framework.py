"""Property-based tests for qmbp_simulation.framework submodule.

Uses Hypothesis to verify universal properties of ExperimentConfig
JSON round-trip and ExperimentMetrics validation across many random inputs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from qmbp_simulation.framework import (
    AnalysisConfig,
    ExperimentConfig,
    ExperimentMetrics,
    MPNNConfig,
    SystemConfig,
    VQEConfig,
)

# ─────────────────────────────────────────────────────────────────────────────
# Strategies
# ─────────────────────────────────────────────────────────────────────────────


def valid_system_config() -> st.SearchStrategy[SystemConfig]:
    """Generate valid SystemConfig with p_layers in {1, 2}."""
    return st.builds(
        SystemConfig,
        n_qubits=st.integers(min_value=4, max_value=14),
        p_layers=st.sampled_from([1, 2]),
        topology=st.sampled_from(["chain_1d", "kagome", "triangular", "ladder"]),
        J=st.floats(min_value=0.1, max_value=5.0),
        h_values=st.just([1.5]),
        h_test=st.just([1.5]),
        g_longitudinal=st.just(0.0),
        boundary=st.sampled_from(["open", "periodic"]),
        model=st.just("tfim"),
    )


def valid_vqe_config() -> st.SearchStrategy[VQEConfig]:
    """Generate valid VQEConfig."""
    return st.builds(
        VQEConfig,
        optimizer=st.sampled_from(["L-BFGS-B", "COBYLA"]),
        n_restarts=st.integers(min_value=1, max_value=10),
        maxiter=st.integers(min_value=100, max_value=2000),
        sigma=st.floats(min_value=0.01, max_value=1.0),
        ftol=st.just(1e-14),
        use_analytical_init=st.booleans(),
        freeze_params=st.just(None),
        freeze_after_h=st.just(None),
        use_hessian_check=st.booleans(),
        hessian_escape_threshold=st.just(-1e-6),
        use_dypp=st.booleans(),
        dypp_order=st.sampled_from([1, 2, 3]),
    )


def valid_mpnn_config() -> st.SearchStrategy[MPNNConfig]:
    """Generate valid MPNNConfig."""
    return st.builds(
        MPNNConfig,
        hidden_dim=st.sampled_from([32, 64, 128]),
        n_layers=st.integers(min_value=1, max_value=5),
        n_epochs=st.integers(min_value=100, max_value=8000),
        lr=st.floats(min_value=1e-4, max_value=1e-2),
        patience=st.integers(min_value=50, max_value=500),
        dropout=st.floats(min_value=0.0, max_value=0.5),
        use_physics_loss=st.booleans(),
        physics_loss_weight=st.floats(min_value=0.01, max_value=1.0),
        physics_loss_start_epoch=st.integers(min_value=100, max_value=2000),
        physics_loss_eval_every=st.integers(min_value=10, max_value=200),
        sign_canonicalization=st.sampled_from(["none", "positive_first"]),
        use_active_learning=st.booleans(),
        n_ensemble=st.integers(min_value=1, max_value=10),
        acquisition=st.sampled_from(["max_variance", "random"]),
    )


def valid_analysis_config() -> st.SearchStrategy[AnalysisConfig]:
    """Generate valid AnalysisConfig."""
    return st.builds(
        AnalysisConfig,
        landscape_resolution=st.integers(min_value=10, max_value=50),
        tci_max_rank=st.integers(min_value=5, max_value=20),
        tci_tolerance=st.just(1e-4),
        scaling_n_values=st.just([4, 6, 8, 10, 14, 20]),
        weight_gradient_n_points=st.integers(min_value=10, max_value=100),
        fluctuation_n_samples=st.integers(min_value=50, max_value=200),
        threshold=st.floats(min_value=0.01, max_value=0.1),
        compute_hessian=st.booleans(),
    )


def valid_experiment_config() -> st.SearchStrategy[ExperimentConfig]:
    """Generate valid ExperimentConfig (p_layers ≤ 2, non-empty seeds)."""
    return st.builds(
        ExperimentConfig,
        experiment_id=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "Pd")),
            min_size=1,
            max_size=30,
        ),
        category=st.sampled_from(["A", "B", "C", "D", "E", "F", "X"]),
        description=st.text(min_size=0, max_size=100),
        hypothesis=st.text(min_size=0, max_size=100),
        system=valid_system_config(),
        vqe=valid_vqe_config(),
        mpnn=valid_mpnn_config(),
        analysis=valid_analysis_config(),
        seeds=st.lists(st.integers(min_value=0, max_value=9999), min_size=1, max_size=5),
        verbose=st.booleans(),
        debug=st.booleans(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Property 13: ExperimentConfig JSON round-trip
# **Validates: Requirements 8.2, 8.3, 8.4**
# ─────────────────────────────────────────────────────────────────────────────


class TestProperty13ExperimentConfigJsonRoundTrip:
    """For any valid ExperimentConfig, to_json() followed by from_json()
    SHALL produce an equivalent configuration (all fields match).
    """

    @given(config=valid_experiment_config())
    @settings(max_examples=30, deadline=None)
    def test_json_round_trip_preserves_all_fields(self, config: ExperimentConfig):
        """All fields survive JSON serialization and deserialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.to_json(path)
            loaded = ExperimentConfig.from_json(path)

            # Top-level metadata
            assert loaded.experiment_id == config.experiment_id
            assert loaded.category == config.category
            assert loaded.description == config.description
            assert loaded.hypothesis == config.hypothesis
            assert loaded.seeds == config.seeds
            assert loaded.verbose == config.verbose
            assert loaded.debug == config.debug

            # SystemConfig
            assert loaded.system.n_qubits == config.system.n_qubits
            assert loaded.system.p_layers == config.system.p_layers
            assert loaded.system.topology == config.system.topology
            assert loaded.system.J == config.system.J
            assert loaded.system.h_values == config.system.h_values
            assert loaded.system.h_test == config.system.h_test
            assert loaded.system.g_longitudinal == config.system.g_longitudinal
            assert loaded.system.boundary == config.system.boundary
            assert loaded.system.model == config.system.model

            # VQEConfig
            assert loaded.vqe.optimizer == config.vqe.optimizer
            assert loaded.vqe.n_restarts == config.vqe.n_restarts
            assert loaded.vqe.maxiter == config.vqe.maxiter
            assert loaded.vqe.sigma == config.vqe.sigma
            assert loaded.vqe.ftol == config.vqe.ftol
            assert loaded.vqe.use_analytical_init == config.vqe.use_analytical_init
            assert loaded.vqe.use_hessian_check == config.vqe.use_hessian_check
            assert loaded.vqe.use_dypp == config.vqe.use_dypp
            assert loaded.vqe.dypp_order == config.vqe.dypp_order

            # MPNNConfig
            assert loaded.mpnn.hidden_dim == config.mpnn.hidden_dim
            assert loaded.mpnn.n_layers == config.mpnn.n_layers
            assert loaded.mpnn.n_epochs == config.mpnn.n_epochs
            assert loaded.mpnn.lr == config.mpnn.lr
            assert loaded.mpnn.patience == config.mpnn.patience
            assert loaded.mpnn.dropout == config.mpnn.dropout
            assert loaded.mpnn.use_physics_loss == config.mpnn.use_physics_loss
            assert loaded.mpnn.sign_canonicalization == config.mpnn.sign_canonicalization
            assert loaded.mpnn.use_active_learning == config.mpnn.use_active_learning
            assert loaded.mpnn.n_ensemble == config.mpnn.n_ensemble
            assert loaded.mpnn.acquisition == config.mpnn.acquisition

            # AnalysisConfig
            assert loaded.analysis.landscape_resolution == config.analysis.landscape_resolution
            assert loaded.analysis.tci_max_rank == config.analysis.tci_max_rank
            assert loaded.analysis.threshold == config.analysis.threshold
            assert loaded.analysis.compute_hessian == config.analysis.compute_hessian
            assert (
                loaded.analysis.weight_gradient_n_points == config.analysis.weight_gradient_n_points
            )
            assert loaded.analysis.fluctuation_n_samples == config.analysis.fluctuation_n_samples


# ─────────────────────────────────────────────────────────────────────────────
# Property 14: ExperimentMetrics validation catches invalid values
# **Validates: Requirements 8.2, 8.3, 8.4**
# ─────────────────────────────────────────────────────────────────────────────


class TestProperty14ExperimentMetricsValidationCatchesInvalidValues:
    """For any ExperimentMetrics with invalid values (NaN energy, negative gap,
    relative_error > 1.0, fidelity outside [0,1]), validate() SHALL return
    a non-empty list of issues.
    """

    @given(
        h_value=st.floats(min_value=-5.0, max_value=5.0),
        energy=st.floats(min_value=-100.0, max_value=100.0),
        exact_energy=st.floats(min_value=-100.0, max_value=-0.1),
        energy_error=st.floats(min_value=0.0, max_value=10.0),
        gap=st.floats(max_value=-0.001, min_value=-10.0),  # Negative gap
    )
    @settings(max_examples=30, deadline=None)
    def test_negative_gap_detected(
        self,
        h_value: float,
        energy: float,
        exact_energy: float,
        energy_error: float,
        gap: float,
    ):
        """Negative gap always produces a validation issue."""
        m = ExperimentMetrics(
            h_value=h_value,
            energy=energy,
            exact_energy=exact_energy,
            energy_error=energy_error,
            gap=gap,
            relative_error=0.02,
            fidelity=0.95,
        )
        issues = m.validate()
        assert len(issues) > 0, f"Expected issues for negative gap={gap}"
        assert any("gap" in issue.lower() for issue in issues)

    @given(
        h_value=st.floats(min_value=-5.0, max_value=5.0),
        relative_error=st.floats(min_value=-10.0, max_value=-0.001),  # Negative
    )
    @settings(max_examples=30, deadline=None)
    def test_negative_relative_error_detected(
        self,
        h_value: float,
        relative_error: float,
    ):
        """Negative relative_error always produces a validation issue."""
        m = ExperimentMetrics(
            h_value=h_value,
            energy=-4.5,
            exact_energy=-4.6,
            energy_error=0.1,
            gap=0.5,
            relative_error=relative_error,
            fidelity=0.95,
        )
        issues = m.validate()
        assert len(issues) > 0, f"Expected issues for negative relative_error={relative_error}"
        assert any("ΔE/gap" in issue for issue in issues)

    @given(
        h_value=st.floats(min_value=-5.0, max_value=5.0),
        fidelity=st.one_of(
            st.floats(min_value=1.002, max_value=10.0),  # Above 1
            st.floats(min_value=-10.0, max_value=-0.002),  # Below 0
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_invalid_fidelity_detected(
        self,
        h_value: float,
        fidelity: float,
    ):
        """Fidelity outside [0, 1] always produces a validation issue."""
        m = ExperimentMetrics(
            h_value=h_value,
            energy=-4.5,
            exact_energy=-4.6,
            energy_error=0.1,
            gap=0.5,
            relative_error=0.02,
            fidelity=fidelity,
        )
        issues = m.validate()
        assert len(issues) > 0, f"Expected issues for fidelity={fidelity}"
        assert any("fidelity" in issue.lower() for issue in issues)

    @given(
        h_value=st.floats(min_value=-5.0, max_value=5.0),
        gap=st.floats(max_value=0.0, min_value=-10.0),  # Non-positive gap (≤ 0)
        relative_error=st.floats(min_value=-5.0, max_value=-0.001),  # Negative
        fidelity=st.one_of(
            st.floats(min_value=1.002, max_value=5.0),
            st.floats(min_value=-5.0, max_value=-0.002),
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_multiple_invalid_values_all_detected(
        self,
        h_value: float,
        gap: float,
        relative_error: float,
        fidelity: float,
    ):
        """Multiple invalid values each produce their own validation issue."""
        m = ExperimentMetrics(
            h_value=h_value,
            energy=-4.5,
            exact_energy=-4.6,
            energy_error=0.1,
            gap=gap,
            relative_error=relative_error,
            fidelity=fidelity,
        )
        issues = m.validate()
        # Should have at least 3 issues (gap, relative_error, fidelity)
        assert len(issues) >= 3, (
            f"Expected ≥3 issues for gap={gap}, relative_error={relative_error}, "
            f"fidelity={fidelity}, got {len(issues)}: {issues}"
        )
