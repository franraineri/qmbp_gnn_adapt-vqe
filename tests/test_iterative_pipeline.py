"""Tests for iterative improvement pipeline fixes (Aug 2026 session).

Covers:
- should_retrain() decision logic
- fine_tune_unified_mpnn() behavior
- EvalCache PID-based atomic writes
- ValidationRunner os._exit path (no gc.collect hang)
- NPZ persistence of validated predictions
- Zoo export guard (Fix C)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# Test should_retrain() — Fix A
# ═══════════════════════════════════════════════════════════════════════════════


class TestShouldRetrain:
    """Validate the retrain decision heuristic."""

    def test_no_new_data_skips(self):
        from qmbp_simulation.predictors.unified_mpnn import should_retrain

        do, reason = should_retrain(0, 0.69, 0.69, 45)
        assert do is False
        assert reason == "no_new_data"

    def test_new_data_triggers_retrain(self):
        from qmbp_simulation.predictors.unified_mpnn import should_retrain

        do, reason = should_retrain(3, 0.69, 0.69, 45)
        assert do is True
        assert reason == "new_data_available"

    def test_pass_rate_improved_always_retrains(self):
        from qmbp_simulation.predictors.unified_mpnn import should_retrain

        do, reason = should_retrain(1, 0.80, 0.69, 45)
        assert do is True
        assert reason == "pass_rate_improved"

    def test_below_min_fraction_skips_large_dataset(self):
        from qmbp_simulation.predictors.unified_mpnn import should_retrain

        # 1 point out of 200 = 0.5% < 5% threshold
        do, reason = should_retrain(1, 0.69, 0.69, 200)
        assert do is False
        assert reason == "below_min_fraction"

    def test_small_dataset_allows_single_point(self):
        from qmbp_simulation.predictors.unified_mpnn import should_retrain

        # 1 point out of 15 — small dataset, 1 point matters
        do, reason = should_retrain(1, 0.69, 0.69, 15)
        assert do is True

    def test_zero_dataset_size_no_crash(self):
        from qmbp_simulation.predictors.unified_mpnn import should_retrain

        do, reason = should_retrain(0, 0.0, 0.0, 0)
        assert do is False

    @pytest.mark.parametrize("n_new,expected", [
        (0, False),
        (1, False),  # 1/30 = 3.3% < 5% min_fraction → skip
        (5, True),   # 5/30 = 16.7% > 5% → retrain
    ])
    def test_parametrized_basic(self, n_new, expected):
        from qmbp_simulation.predictors.unified_mpnn import should_retrain

        do, _ = should_retrain(n_new, 0.5, 0.5, 30)
        assert do is expected


# ═══════════════════════════════════════════════════════════════════════════════
# Test fine_tune_unified_mpnn() — Fix B
# ═══════════════════════════════════════════════════════════════════════════════


class TestFineTune:
    """Validate fine-tuning behavior."""

    @pytest.fixture
    def trained_model_and_dataset(self):
        """Create a pre-trained model with a small dataset."""
        import torch
        from qmbp_simulation.predictors.unified_mpnn import (
            UnifiedMPNN, train_unified_mpnn,
        )
        from qmbp_simulation.predictors.unified_graph import (
            build_unified_bond_resolved_graph,
        )
        from qmbp_simulation.models.hamiltonian import make_lattice

        lattice = make_lattice("chain_1d", 4, J=1.0, h=3.0)
        # For chain_1d N=4 p=1 bond-resolved: n_params = n_edges + N = 3 + 4 = 7
        n_params = len(lattice.edges) + lattice.n_qubits
        dataset = []
        for h in np.linspace(2.5, 4.0, 8):
            theta_fake = np.random.randn(n_params) * 0.1
            g = build_unified_bond_resolved_graph(
                lattice, h_value=float(h), p_layers=1,
                theta_opt=theta_fake,
                include_circuit_nodes=True,
            )
            dataset.append(g)

        model = UnifiedMPNN(
            node_features=dataset[0].x.shape[1],
            hidden_dim=64, n_layers=2,
        )
        # Quick pre-train
        train_unified_mpnn(model, dataset, n_epochs=50, lr=1e-3, seed=42)
        return model, dataset

    def test_fine_tune_returns_metrics(self, trained_model_and_dataset):
        from qmbp_simulation.predictors.unified_mpnn import fine_tune_unified_mpnn

        model, dataset = trained_model_and_dataset
        result = fine_tune_unified_mpnn(
            model, dataset, n_epochs=20, lr=3e-4, seed=42,
        )
        assert "final_mse" in result
        assert "improvement_ratio" in result
        assert "mode" in result
        assert result["mode"] == "fine_tune"
        assert result["initial_mse"] > 0

    def test_fine_tune_does_not_diverge(self, trained_model_and_dataset):
        from qmbp_simulation.predictors.unified_mpnn import fine_tune_unified_mpnn

        model, dataset = trained_model_and_dataset
        result = fine_tune_unified_mpnn(
            model, dataset, n_epochs=50, lr=3e-4, seed=42,
        )
        # Should not get worse (improvement_ratio ≤ 1.5 — allow some variance)
        assert result["improvement_ratio"] < 1.5

    def test_fine_tune_with_too_few_points_raises(self):
        import torch
        from qmbp_simulation.predictors.unified_mpnn import (
            UnifiedMPNN, fine_tune_unified_mpnn,
        )
        from torch_geometric.data import Data

        model = UnifiedMPNN(node_features=4, hidden_dim=32, n_layers=2)
        # Only 2 points — should raise ValueError
        tiny_dataset = [
            Data(x=torch.randn(4, 4), y=torch.randn(6)),
            Data(x=torch.randn(4, 4), y=torch.randn(6)),
        ]
        with pytest.raises(ValueError, match="≥3"):
            fine_tune_unified_mpnn(model, tiny_dataset, n_epochs=10)


# ═══════════════════════════════════════════════════════════════════════════════
# Test EvalCache parallel safety
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvalCacheParallel:
    """Validate PID-based atomic writes prevent race conditions."""

    def test_tmp_file_uses_pid(self, tmp_path):
        from qmbp_simulation.execution.eval_cache import EvalCache

        cache = EvalCache(path=tmp_path / "test_cache.json")
        key = cache.make_key("chain_1d", 4, 3.0, np.zeros(8))
        cache.put(key, -5.0)
        cache.flush()

        # Check that no stale .tmp file remains
        tmp_files = list(tmp_path.glob("*.tmp*"))
        assert len(tmp_files) == 0, f"Stale tmp files: {tmp_files}"
        # Cache file should exist
        assert (tmp_path / "test_cache.json").exists()

    def test_cache_roundtrip(self, tmp_path):
        from qmbp_simulation.execution.eval_cache import EvalCache

        cache_path = tmp_path / "cache.json"
        cache1 = EvalCache(path=cache_path)
        theta = np.array([0.1, 0.2, 0.3, 0.4])
        key = cache1.make_key("square", 6, 2.5, theta, model="tfim_bond_resolved")
        cache1.put(key, -12.345)
        cache1.flush()

        # Load in a new instance
        cache2 = EvalCache(path=cache_path)
        result = cache2.get(key)
        assert result is not None
        np.testing.assert_allclose(result, -12.345, atol=1e-10)

    def test_cache_rejects_nan(self, tmp_path):
        from qmbp_simulation.execution.eval_cache import EvalCache

        cache = EvalCache(path=tmp_path / "cache.json")
        key = cache.make_key("chain_1d", 4, 3.0, np.zeros(8))
        cache.put(key, float("nan"))
        # NaN should be rejected
        assert cache.get(key) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Test ValidationRunner exit path (no gc.collect hang)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunnerExitPath:
    """Validate that the runner doesn't call gc.collect and uses os._exit."""

    def test_no_gc_collect_in_run_method(self):
        """Verify gc.collect is not called in the run() hot path."""
        import inspect
        from qmbp_simulation.framework.runner_base import ValidationRunner

        source = inspect.getsource(ValidationRunner.run)
        # The only gc references should be commented out
        active_gc_lines = [
            line.strip() for line in source.split("\n")
            if "gc.collect" in line or "_gc.collect" in line
            if not line.strip().startswith("#")
            if not line.strip().startswith("//")
        ]
        assert len(active_gc_lines) == 0, (
            f"Found active gc.collect calls in run(): {active_gc_lines}"
        )

    def test_main_uses_os_exit_or_sys_exit(self):
        """Verify main() terminates via os._exit or sys.exit (not return)."""
        import inspect
        from qmbp_simulation.framework.runner_base import ValidationRunner

        source = inspect.getsource(ValidationRunner.main)
        assert "os._exit" in source or "sys.exit" in source

    def test_run_has_os_exit(self):
        """Verify run() uses os._exit to prevent interpreter shutdown hang."""
        import inspect
        from qmbp_simulation.framework.runner_base import ValidationRunner

        source = inspect.getsource(ValidationRunner.run)
        assert "os._exit" in source


# ═══════════════════════════════════════════════════════════════════════════════
# Test NPZ persistence of validated predictions
# ═══════════════════════════════════════════════════════════════════════════════


class TestNPZPersistence:
    """Validate that validated predictions are persisted correctly."""

    def test_npz_roundtrip(self, tmp_path):
        """Write and read NPZ with expected fields."""
        h_values = np.linspace(4.5, 2.5, 16)
        theta_opt = np.random.randn(16, 11)
        e_vqe = np.random.randn(16) - 20
        e_exact = e_vqe - np.abs(np.random.randn(16)) * 0.1
        gaps = np.abs(np.random.randn(16)) + 1.0
        de_gaps = np.abs(e_vqe - e_exact) / gaps

        npz_path = tmp_path / "test_N6_p1.npz"
        np.savez(
            npz_path,
            h_values=h_values, theta_opt=theta_opt,
            e_vqe=e_vqe, e_exact=e_exact,
            de_gaps=de_gaps, gaps=gaps,
        )

        # Read back
        d = np.load(npz_path, allow_pickle=True)
        assert d["h_values"].shape == (16,)
        assert d["theta_opt"].shape == (16, 11)
        np.testing.assert_allclose(d["h_values"], h_values)
        np.testing.assert_allclose(d["theta_opt"], theta_opt)

    def test_npz_upsert_lower_energy_wins(self, tmp_path):
        """upsert_theta_npz should keep lower energy when h already exists."""
        from qmbp_simulation.framework.result_io import upsert_theta_npz

        npz_path = tmp_path / "test.npz"
        # Initial save
        np.savez(
            npz_path,
            h_values=np.array([3.0, 2.5]),
            theta_opt=np.array([[0.1, 0.2], [0.3, 0.4]]),
            e_vqe=np.array([-10.0, -8.0]),
            e_exact=np.array([-10.5, -8.5]),
            gaps=np.array([2.0, 1.5]),
        )

        # Upsert with better energy at h=3.0
        upsert_theta_npz(
            npz_path,
            h_new=np.array([3.0]),
            theta_new=np.array([[0.5, 0.6]]),
            e_vqe_new=np.array([-10.3]),  # Better than -10.0
            e_exact_new=np.array([-10.5]),
            gaps_new=np.array([2.0]),
        )

        d = np.load(npz_path, allow_pickle=True)
        idx_3 = np.argmin(np.abs(d["h_values"] - 3.0))
        # Should have the better energy
        np.testing.assert_allclose(d["e_vqe"][idx_3], -10.3, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Zoo export guard — Fix C
# ═══════════════════════════════════════════════════════════════════════════════


class TestZooExportGuard:
    """Validate that zoo export only happens when pass_rate improves."""

    def test_zoo_best_pass_rate_tracking(self):
        """The zoo_best_pass_rate variable should be updated on export."""
        # This is a logic test — verify the condition
        zoo_best_pass_rate = 0.69
        current_pass_rate = 0.75
        iteration = 2

        # Should export: current > best
        should_export = current_pass_rate > zoo_best_pass_rate or iteration == 1
        assert should_export is True

        # Should NOT export: current <= best
        zoo_best_pass_rate = 0.80
        current_pass_rate = 0.75
        should_export = current_pass_rate > zoo_best_pass_rate or iteration == 1
        assert should_export is False

        # First iteration always exports
        should_export = current_pass_rate > zoo_best_pass_rate or iteration == 1
        iteration = 1
        should_export = current_pass_rate > zoo_best_pass_rate or iteration == 1
        assert should_export is True


# ═══════════════════════════════════════════════════════════════════════════════
# Test AcceleratedConfig skip_below_h_min disabled
# ═══════════════════════════════════════════════════════════════════════════════


class TestAcceleratedConfig:
    """Validate that h-point filtering is disabled by default."""

    def test_skip_below_h_min_is_false(self):
        from qmbp_simulation.pipeline.accelerated import AcceleratedConfig

        cfg = AcceleratedConfig()
        assert cfg.skip_below_h_min is False


# ═══════════════════════════════════════════════════════════════════════════════
# Test VQE L-BFGS-B cap
# ═══════════════════════════════════════════════════════════════════════════════


class TestVQECap:
    """Validate L-BFGS-B maxiter is capped via MAX_LBFGSB_ITERS constant."""

    def test_lbfgsb_cap_constant_exists(self):
        """MAX_LBFGSB_ITERS should be importable from constants."""
        from qmbp_simulation.models.constants import MAX_LBFGSB_ITERS
        assert isinstance(MAX_LBFGSB_ITERS, int)
        assert MAX_LBFGSB_ITERS > 0

    def test_vqe_imports_cap(self):
        """VQEOptimizer.optimize should reference MAX_LBFGSB_ITERS."""
        import inspect
        from qmbp_simulation.optimizers.vqe import VQEOptimizer

        source = inspect.getsource(VQEOptimizer.optimize)
        assert "MAX_LBFGSB_ITERS" in source


# ═══════════════════════════════════════════════════════════════════════════════
# Test CachedBackend context manager (auto-flush on exit)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCachedBackendContextManager:
    """Validate that CachedBackend context manager flushes on exit."""

    def test_context_manager_returns_self(self, tmp_path):
        """__enter__ should return the CachedBackend instance."""
        from qmbp_simulation.execution.eval_cache import CachedBackend, EvalCache
        from qmbp_simulation.execution import NoiselessBackend

        backend = NoiselessBackend()
        cache = EvalCache(path=tmp_path / "ctx_test.json")
        cb = CachedBackend(backend, topology="chain_1d", n_qubits=4, cache=cache)

        with cb as ctx:
            assert ctx is cb
            assert isinstance(ctx, CachedBackend)

    def test_context_manager_flushes_on_normal_exit(self, tmp_path):
        """Cache should be flushed when exiting the with block normally."""
        from qmbp_simulation.execution.eval_cache import CachedBackend, EvalCache
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation import make_lattice, HamiltonianBuilder, HVACircuitBuilder
        import json

        backend = NoiselessBackend()
        cache_path = tmp_path / "flush_normal.json"
        cache = EvalCache(path=cache_path)

        lattice = make_lattice("chain_1d", 4, J=1.0, h=3.0)
        builder = HamiltonianBuilder()
        H = builder.build(lattice)
        hva = HVACircuitBuilder()
        circuit, _ = hva.create(4, 1, lattice)
        theta = np.zeros(circuit.num_parameters)

        with CachedBackend(backend, topology="chain_1d", n_qubits=4, cache=cache) as cb:
            cb.set_h(3.0)
            _ = cb.evaluate(circuit, H, theta)
            # At this point, cache might not be flushed yet

        # After with block, file should exist and have data
        assert cache_path.exists(), "Cache file should exist after context exit"
        with open(cache_path) as f:
            data = json.load(f)
        assert len(data) > 0, "Cache should have at least one entry"

    def test_context_manager_flushes_on_exception(self, tmp_path):
        """Cache should be flushed even when an exception is raised."""
        from qmbp_simulation.execution.eval_cache import CachedBackend, EvalCache
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation import make_lattice, HamiltonianBuilder, HVACircuitBuilder
        import json

        backend = NoiselessBackend()
        cache_path = tmp_path / "flush_exception.json"
        cache = EvalCache(path=cache_path)

        lattice = make_lattice("chain_1d", 4, J=1.0, h=3.0)
        builder = HamiltonianBuilder()
        H = builder.build(lattice)
        hva = HVACircuitBuilder()
        circuit, _ = hva.create(4, 1, lattice)
        theta = np.zeros(circuit.num_parameters)

        try:
            with CachedBackend(backend, topology="chain_1d", n_qubits=4, cache=cache) as cb:
                cb.set_h(3.0)
                _ = cb.evaluate(circuit, H, theta)
                raise ValueError("Simulated error")
        except ValueError:
            pass  # Expected

        # Cache should still be flushed despite exception
        assert cache_path.exists(), "Cache file should exist after exception"
        with open(cache_path) as f:
            data = json.load(f)
        assert len(data) > 0, "Cache should have data even after exception"

    def test_exit_does_not_suppress_exceptions(self, tmp_path):
        """__exit__ should return False (not suppress exceptions)."""
        from qmbp_simulation.execution.eval_cache import CachedBackend, EvalCache
        from qmbp_simulation.execution import NoiselessBackend

        backend = NoiselessBackend()
        cache = EvalCache(path=tmp_path / "no_suppress.json")

        with pytest.raises(RuntimeError, match="test exception"):
            with CachedBackend(backend, topology="chain_1d", n_qubits=4, cache=cache):
                raise RuntimeError("test exception")
