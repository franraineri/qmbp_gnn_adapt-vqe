"""Tests for the fidelity cache + critical-window ranking features.

Covers the principal, bug-prone behaviors of the recently added functionality:

1. EvalCache fidelity get/put round-trip (persistent, range-validated).
2. compute_exact_fidelity with cache_ctx returns a cache hit (no recompute)
   and stores the computed value.
3. update_zoo_critical_ranking upsert + only_if_better semantics.
4. get_critical_metrics_at_h reports per-N |ΔE|/fidelity at h≈1.0.
"""

from __future__ import annotations

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1. EvalCache fidelity get/put
# ─────────────────────────────────────────────────────────────────────────────


class TestEvalCacheFidelity:
    """EvalCache stores and retrieves ground-state fidelities."""

    def _cache(self, tmp_path):
        from qmbp_simulation.execution.eval_cache import EvalCache

        return EvalCache(path=tmp_path / "eval_cache.json")

    def test_put_then_get_roundtrip(self, tmp_path):
        cache = self._cache(tmp_path)
        theta = np.array([0.1, 0.2, 0.3])
        cache.put_fidelity("chain_1d", 10, 1.0, theta, 0.95,
                           model="tfim_bond_resolved", p_layers=1)
        got = cache.get_fidelity("chain_1d", 10, 1.0, theta,
                                 model="tfim_bond_resolved", p_layers=1)
        assert got == pytest.approx(0.95)

    def test_miss_returns_none(self, tmp_path):
        cache = self._cache(tmp_path)
        got = cache.get_fidelity("chain_1d", 10, 1.0, np.array([0.0]),
                                 model="tfim_bond_resolved", p_layers=1)
        assert got is None

    def test_out_of_range_is_rejected(self, tmp_path):
        cache = self._cache(tmp_path)
        theta = np.array([0.5])
        cache.put_fidelity("chain_1d", 10, 1.0, theta, 1.5, p_layers=1)  # invalid > 1
        assert cache.get_fidelity("chain_1d", 10, 1.0, theta, p_layers=1) is None

    def test_key_depends_on_theta(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.put_fidelity("chain_1d", 10, 1.0, np.array([0.1]), 0.9, p_layers=1)
        # Different theta → different key → miss
        assert cache.get_fidelity("chain_1d", 10, 1.0, np.array([0.2]), p_layers=1) is None

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "eval_cache.json"
        from qmbp_simulation.execution.eval_cache import EvalCache

        theta = np.array([0.7, 0.8])
        c1 = EvalCache(path=path)
        c1.put_fidelity("chain_1d", 8, 1.2, theta, 0.88, p_layers=1)
        c1.flush()
        c2 = EvalCache(path=path)  # fresh instance reads from disk
        assert c2.get_fidelity("chain_1d", 8, 1.2, theta, p_layers=1) == pytest.approx(0.88)


# ─────────────────────────────────────────────────────────────────────────────
# 2. compute_exact_fidelity with cache_ctx
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeExactFidelityCache:
    """compute_exact_fidelity caches via cache_ctx and reuses on hit."""

    def _tiny_case(self):
        """A trivial 1-qubit circuit + exact state so the overlap is deterministic."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.rx(0.0, 0)  # identity-ish, parameterless bound below
        theta = np.array([])  # no parameters
        exact = np.array([1.0, 0.0], dtype=complex)  # |0>
        return qc, theta, exact

    def test_no_ctx_still_computes(self):
        from qmbp_simulation.analysis.fidelity import compute_exact_fidelity

        qc, theta, exact = self._tiny_case()
        fid = compute_exact_fidelity(qc, theta, exact)
        assert fid is not None and 0.0 <= fid <= 1.0

    def test_ctx_stores_and_hits(self, tmp_path, monkeypatch):
        # Point the shared EvalCache default path at tmp so the test is isolated.
        import qmbp_simulation.execution.eval_cache as ec

        monkeypatch.setattr(ec, "_DEFAULT_CACHE_PATH", tmp_path / "eval_cache.json")

        from qmbp_simulation.analysis.fidelity import compute_exact_fidelity

        qc, theta, exact = self._tiny_case()
        ctx = {"topology": "chain_1d", "n_qubits": 1, "h": 1.0,
               "model": "tfim_bond_resolved", "p_layers": 1}
        f1 = compute_exact_fidelity(qc, theta, exact, cache_ctx=ctx)

        # Stored under a FID| key
        cache = ec.EvalCache(path=tmp_path / "eval_cache.json")
        assert any(k.startswith("FID|") for k in cache._data)

        # Second call returns the same value (from cache)
        f2 = compute_exact_fidelity(qc, theta, exact, cache_ctx=ctx)
        assert f2 == pytest.approx(f1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. update_zoo_critical_ranking upsert
# ─────────────────────────────────────────────────────────────────────────────


class TestUpdateCriticalRanking:
    """update_zoo_critical_ranking writes and respects only_if_better."""

    def _isolate_manifest(self, tmp_path, monkeypatch):
        import qmbp_simulation.predictors.model_zoo as mz
        from qmbp_simulation.predictors.model_zoo import ZooEntry, _save_manifest

        manifest = tmp_path / "manifest.json"
        monkeypatch.setattr(mz, "_MANIFEST_PATH", manifest)
        monkeypatch.setattr(mz, "_REPO_MANIFEST_PATH", manifest)
        entry = ZooEntry(
            model="tfim_bond_resolved", topology="chain_1d", n_qubits=0, p_layers=1,
            checkpoint_file="model_a.pt", h_range=(0.5, 5.0), pass_rate=0.7,
        )
        _save_manifest([entry])
        return mz

    def test_upsert_writes_record(self, tmp_path, monkeypatch):
        mz = self._isolate_manifest(tmp_path, monkeypatch)
        ok = mz.update_zoo_critical_ranking(
            "model_a.pt", {"abs_error_mean": 0.2, "fidelity_mean": 0.9, "grade": "C"}
        )
        assert ok
        entry = next(e for e in mz._load_manifest() if e.checkpoint_file == "model_a.pt")
        rec = entry.critical_ranking[mz._critical_window_key()]
        assert rec["abs_error_mean"] == pytest.approx(0.2)
        assert rec["fidelity_mean"] == pytest.approx(0.9)

    def test_only_if_better_keeps_lower_abs_error(self, tmp_path, monkeypatch):
        mz = self._isolate_manifest(tmp_path, monkeypatch)
        mz.update_zoo_critical_ranking("model_a.pt", {"abs_error_mean": 0.10})
        # Worse |ΔE| should NOT overwrite when only_if_better=True
        changed = mz.update_zoo_critical_ranking(
            "model_a.pt", {"abs_error_mean": 0.50}, only_if_better=True
        )
        assert changed is False
        entry = next(e for e in mz._load_manifest() if e.checkpoint_file == "model_a.pt")
        rec = entry.critical_ranking[mz._critical_window_key()]
        assert rec["abs_error_mean"] == pytest.approx(0.10)

    def test_unknown_checkpoint_returns_false(self, tmp_path, monkeypatch):
        mz = self._isolate_manifest(tmp_path, monkeypatch)
        assert mz.update_zoo_critical_ranking("nope.pt", {"abs_error_mean": 0.1}) is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. get_critical_metrics_at_h
# ─────────────────────────────────────────────────────────────────────────────


class TestCriticalMetricsAtH:
    """get_critical_metrics_at_h returns per-N |ΔE|/fidelity at h≈1.0."""

    def test_reports_per_n_at_h1(self, tmp_path, monkeypatch):
        import qmbp_simulation.predictors.model_zoo as mz
        from qmbp_simulation.predictors.model_zoo import ZooEntry, _save_manifest

        manifest = tmp_path / "manifest.json"
        monkeypatch.setattr(mz, "_MANIFEST_PATH", manifest)
        monkeypatch.setattr(mz, "_REPO_MANIFEST_PATH", manifest)

        win = mz._critical_window_key()
        entry = ZooEntry(
            model="tfim_bond_resolved", topology="chain_1d", n_qubits=0, p_layers=1,
            checkpoint_file="model_a.pt",
            critical_ranking={
                win: {
                    "abs_error_mean": 0.3,
                    "per_n": {
                        "10": {"abs_error_mean": 0.2,
                               "at_h1": {"h": 1.0, "abs_error": 0.15, "fidelity": 0.92}},
                        "20": {"abs_error_mean": 0.5,
                               "at_h1": {"h": 1.05, "abs_error": 0.40, "fidelity": 0.80}},
                        "30": {"abs_error_mean": 0.9, "at_h1": None},
                    },
                }
            },
        )
        _save_manifest([entry])

        out = mz.get_critical_metrics_at_h("model_a.pt", h=1.0, p_layers=1,
                                           n_values=[10, 20, 30])
        assert "model_a.pt" in out
        per_n = out["model_a.pt"]
        assert per_n[10]["abs_error"] == pytest.approx(0.15)
        assert per_n[10]["fidelity"] == pytest.approx(0.92)
        assert per_n[20]["abs_error"] == pytest.approx(0.40)
        # N=30 had at_h1=None → excluded
        assert 30 not in per_n
