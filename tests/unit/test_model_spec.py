"""Unit tests for ModelSpec and ModelRegistry."""

from __future__ import annotations

import pytest

from qmbp_simulation.models.model_registry import get_model_spec, list_models


class TestModelSpec:
    """Tests for the ModelSpec frozen dataclass."""

    def test_model_spec_frozen(self):
        """ModelSpec is immutable (frozen dataclass)."""
        spec = get_model_spec("tfim")
        with pytest.raises(Exception):  # FrozenInstanceError
            spec.name = "modified"

    def test_model_spec_total_params(self):
        """total_params = params_per_layer × 2."""
        spec = get_model_spec("heisenberg")
        assert spec.total_params == spec.params_per_layer * 2

    def test_model_spec_total_params_for_p(self):
        """total_params_for_p(1) = params_per_layer, total_params_for_p(2) = params_per_layer × 2."""
        spec = get_model_spec("heisenberg")
        assert spec.total_params_for_p(1) == spec.params_per_layer
        assert spec.total_params_for_p(2) == spec.params_per_layer * 2

    def test_model_spec_with_delta(self):
        """with_delta returns new spec with updated hamiltonian_kwargs."""
        spec = get_model_spec("heisenberg")
        new_spec = spec.with_delta(0.5)
        assert new_spec.hamiltonian_kwargs["delta"] == 0.5
        # Original unchanged
        assert spec.hamiltonian_kwargs["delta"] == 1.0

    def test_model_spec_with_delta_preserves_other_fields(self):
        """with_delta preserves all fields except hamiltonian_kwargs."""
        spec = get_model_spec("heisenberg")
        new_spec = spec.with_delta(0.5)
        assert new_spec.name == spec.name
        assert new_spec.params_per_layer == spec.params_per_layer
        assert new_spec.initial_state == spec.initial_state
        assert new_spec.vqe_defaults == spec.vqe_defaults
        assert new_spec.fidelity_threshold == spec.fidelity_threshold
        assert new_spec.mpnn_hidden_dim == spec.mpnn_hidden_dim

    def test_model_spec_get_vqe_config_overrides(self):
        """get_vqe_config_overrides returns a copy of vqe_defaults."""
        spec = get_model_spec("heisenberg")
        overrides = spec.get_vqe_config_overrides()
        assert overrides == spec.vqe_defaults
        # Verify it's a copy (mutation doesn't affect original)
        overrides["n_restarts"] = 999
        assert spec.vqe_defaults["n_restarts"] != 999


class TestModelRegistry:
    """Tests for the model registry."""

    def test_registry_lists_builtin_models(self):
        """Registry contains tfim, heisenberg, and xy."""
        models = list_models()
        assert "tfim" in models
        assert "heisenberg" in models
        assert "xy" in models

    def test_registry_get_tfim(self):
        """TFIM spec has params_per_layer=2, initial_state='plus'."""
        spec = get_model_spec("tfim")
        assert spec.params_per_layer == 2
        assert spec.initial_state == "plus"

    def test_registry_get_heisenberg(self):
        """Heisenberg spec has params_per_layer=4, initial_state='neel', delta=1.0."""
        spec = get_model_spec("heisenberg")
        assert spec.params_per_layer == 4
        assert spec.initial_state == "neel"
        assert spec.hamiltonian_kwargs["delta"] == 1.0

    def test_registry_get_xy(self):
        """XY spec has params_per_layer=4, delta=0.0."""
        spec = get_model_spec("xy")
        assert spec.params_per_layer == 4
        assert spec.hamiltonian_kwargs["delta"] == 0.0

    def test_registry_unknown_model_raises(self):
        """ValueError raised for unknown model type."""
        with pytest.raises(ValueError, match="not registered"):
            get_model_spec("nonexistent_model")

    def test_registry_heisenberg_vqe_defaults(self):
        """Heisenberg VQE defaults: n_restarts=10, restart_sigma=0.5, maxiter=1500."""
        spec = get_model_spec("heisenberg")
        assert spec.vqe_defaults["n_restarts"] == 10
        assert spec.vqe_defaults["restart_sigma"] == 0.5
        assert spec.vqe_defaults["maxiter"] == 1500

    def test_registry_heisenberg_fidelity_threshold(self):
        """Heisenberg fidelity_threshold=0.60 (relaxed)."""
        spec = get_model_spec("heisenberg")
        assert spec.fidelity_threshold == 0.60

    def test_registry_heisenberg_mpnn_hidden_dim(self):
        """Heisenberg mpnn_hidden_dim=128."""
        spec = get_model_spec("heisenberg")
        assert spec.mpnn_hidden_dim == 128

    def test_registry_tfim_fidelity_threshold(self):
        """TFIM fidelity_threshold=0.93 (default)."""
        spec = get_model_spec("tfim")
        assert spec.fidelity_threshold == 0.93
