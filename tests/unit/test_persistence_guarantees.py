"""Persistence guarantees: verify all data paths save and load correctly.

Tests the full roundtrip for:
- NPZ upsert (theta, energy, metrics, anti-regression)
- GroundTruthCache (put/get/flush/reload)
- EvalCache (key uniqueness, put/get)
- atomic_savez (crash-safe write)
- Manifest heal (detect missing checkpoints)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qmbp_simulation.framework.result_io import upsert_theta_npz
from qmbp_simulation.utils.helpers import atomic_savez


class TestNPZUpsertRoundtrip:
    """Verify theta NPZ persistence with anti-regression."""

    def test_fresh_upsert_adds_all_points(self, tmp_path):
        npz = tmp_path / "test.npz"
        h = np.array([1.0, 1.5, 2.0])
        theta = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        e_vqe = np.array([-5.0, -4.5, -4.0])
        e_exact = np.array([-5.1, -4.6, -4.1])
        gaps = np.array([0.5, 0.4, 0.3])

        n_upd, n_add = upsert_theta_npz(npz, h, theta, e_vqe, e_exact, gaps)
        assert n_add == 3
        assert n_upd == 0

    def test_load_roundtrip_preserves_fields(self, tmp_path):
        npz = tmp_path / "test.npz"
        h = np.array([1.0, 2.0])
        theta = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        e_vqe = np.array([-5.0, -4.0])
        e_exact = np.array([-5.1, -4.1])
        gaps = np.array([0.5, 0.3])

        upsert_theta_npz(npz, h, theta, e_vqe, e_exact, gaps)
        data = np.load(npz, allow_pickle=True)

        assert np.allclose(data["h_values"], h)
        assert np.allclose(data["theta_opt"], theta)
        assert np.allclose(data["e_exact"], e_exact)
        assert np.allclose(data["gaps"], gaps)

    def test_anti_regression_lower_energy_wins(self, tmp_path):
        npz = tmp_path / "test.npz"
        h = np.array([1.5])
        theta = np.array([[0.1, 0.2]])
        e_vqe = np.array([-4.5])
        e_exact = np.array([-4.6])
        gaps = np.array([0.4])

        upsert_theta_npz(npz, h, theta, e_vqe, e_exact, gaps)

        # Try to overwrite with worse energy
        theta_worse = np.array([[0.9, 0.9]])
        e_worse = np.array([-4.0])  # Worse
        n_upd, n_add = upsert_theta_npz(npz, h, theta_worse, e_worse, e_exact, gaps)
        assert n_upd == 0 and n_add == 0

        # Verify original is preserved
        data = np.load(npz, allow_pickle=True)
        e_key = "e_vqe" if "e_vqe" in data else "e_pred"
        assert data[e_key][0] < -4.4

    def test_anti_regression_better_energy_updates(self, tmp_path):
        npz = tmp_path / "test.npz"
        h = np.array([1.5])
        theta = np.array([[0.1, 0.2]])
        e_vqe = np.array([-4.5])
        e_exact = np.array([-4.6])
        gaps = np.array([0.4])

        upsert_theta_npz(npz, h, theta, e_vqe, e_exact, gaps)

        # Overwrite with better energy
        theta_better = np.array([[0.11, 0.21]])
        e_better = np.array([-4.8])  # Better
        n_upd, n_add = upsert_theta_npz(npz, h, theta_better, e_better, e_exact, gaps)
        assert n_upd == 1

        data = np.load(npz, allow_pickle=True)
        e_key = "e_vqe" if "e_vqe" in data else "e_pred"
        assert data[e_key][0] < -4.7


class TestGroundTruthCache:
    """Verify GT cache put/get/flush roundtrip."""

    def test_put_get_roundtrip(self):
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        gt = GroundTruthCache()
        gt.put("chain_1d", 99, "test_model", 2.5, energy=-10.123, gap=0.456)
        cached = gt.get("chain_1d", 99, "test_model", 2.5)
        assert cached is not None
        assert abs(cached["energy"] - (-10.123)) < 1e-10
        assert abs(cached["gap"] - 0.456) < 1e-10

    def test_cache_miss_returns_none(self):
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        gt = GroundTruthCache()
        miss = gt.get("nonexistent", 999, "fake", 99.9)
        assert miss is None

    def test_flush_persists_to_disk(self):
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        gt = GroundTruthCache()
        gt.put("chain_1d", 98, "test_flush", 3.0, energy=-8.0, gap=0.3)
        gt.flush()

        gt2 = GroundTruthCache()
        cached = gt2.get("chain_1d", 98, "test_flush", 3.0)
        assert cached is not None
        assert abs(cached["energy"] - (-8.0)) < 1e-10


class TestEvalCache:
    """Verify EvalCache key uniqueness and roundtrip."""

    def test_make_key_deterministic(self):
        from qmbp_simulation.execution.eval_cache import EvalCache

        cache = EvalCache(enabled=True)
        theta = np.array([0.1, 0.2, 0.3])
        k1 = cache.make_key("chain_1d", 10, 2.5, theta, model="tfim", p_layers=1)
        k2 = cache.make_key("chain_1d", 10, 2.5, theta, model="tfim", p_layers=1)
        assert k1 == k2

    def test_different_theta_different_key(self):
        from qmbp_simulation.execution.eval_cache import EvalCache

        cache = EvalCache(enabled=True)
        t1 = np.array([0.1, 0.2, 0.3])
        t2 = np.array([0.1, 0.2, 0.4])
        k1 = cache.make_key("chain_1d", 10, 2.5, t1, model="tfim", p_layers=1)
        k2 = cache.make_key("chain_1d", 10, 2.5, t2, model="tfim", p_layers=1)
        assert k1 != k2

    def test_put_get_roundtrip(self):
        from qmbp_simulation.execution.eval_cache import EvalCache

        cache = EvalCache(enabled=True)
        theta = np.array([0.1, 0.2, 0.3])
        key = cache.make_key("chain_1d", 10, 2.5, theta, model="tfim", p_layers=1)
        cache.put(key, -5.123)
        result = cache.get(key)
        assert result is not None
        assert abs(result - (-5.123)) < 1e-10

    def test_cache_miss(self):
        from qmbp_simulation.execution.eval_cache import EvalCache

        cache = EvalCache(enabled=True)
        result = cache.get("nonexistent_key_xyz")
        assert result is None


class TestAtomicSavez:
    """Verify atomic_savez crash-safe write."""

    def test_roundtrip(self, tmp_path):
        npz = tmp_path / "atomic.npz"
        h = np.array([1.0, 2.0])
        theta = np.random.randn(2, 4)

        atomic_savez(npz, h_values=h, theta_opt=theta)
        assert npz.exists()

        loaded = np.load(npz)
        assert np.allclose(loaded["h_values"], h)
        assert np.allclose(loaded["theta_opt"], theta)

    def test_no_temp_file_left(self, tmp_path):
        npz = tmp_path / "atomic.npz"
        atomic_savez(npz, data=np.array([1, 2, 3]))

        tmp_file = npz.with_suffix(".tmp.npz")
        assert not tmp_file.exists()

    def test_creates_parent_dirs(self, tmp_path):
        npz = tmp_path / "deep" / "nested" / "file.npz"
        atomic_savez(npz, x=np.array([1.0]))
        assert npz.exists()


class TestManifestIntegrity:
    """Verify manifest ↔ disk coherence."""

    def test_heal_manifest_detects_missing(self):
        from qmbp_simulation.predictors.model_zoo import heal_manifest

        result = heal_manifest(dry_run=True)
        assert "missing_checkpoints" in result
        assert "orphan_files" in result
        assert "duplicates_removed" in result

    def test_all_manifest_entries_have_required_fields(self):
        from qmbp_simulation.predictors.model_zoo import _load_manifest

        entries = _load_manifest()
        for e in entries:
            assert e.model, f"Empty model field: {e.checkpoint_file}"
            assert e.topology, f"Empty topology: {e.checkpoint_file}"
            assert e.p_layers >= 1, f"Invalid p_layers: {e.checkpoint_file}"
            assert e.checkpoint_file, "Empty checkpoint_file"
