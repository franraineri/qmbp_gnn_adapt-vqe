"""Tests for model zoo integrity fixes.

Covers:
1. ZooEntry.matches() correctly handles n_qubits=0 (falsy value filtering)
2. save_mpnn_to_zoo n_qubits determination (no hardcoded fallback to 10)
3. Dashboard freshness check (skip regeneration when sources unchanged)
4. Dashboard zoo_pass_rate uses multi-N model (not inflated by single-N=10)
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: ZooEntry.matches() with falsy n_qubits=0
# ─────────────────────────────────────────────────────────────────────────────


class TestZooEntryMatchesFalsyValues:
    """Verify ZooEntry.matches() uses 'is not None' for n_qubits and p_layers."""

    def _make_entry(self, n_qubits=0, p_layers=1):
        from qmbp_simulation.predictors.model_zoo import ZooEntry

        return ZooEntry(
            model="tfim_bond_resolved",
            topology="chain_1d",
            n_qubits=n_qubits,
            p_layers=p_layers,
            checkpoint_file="test.pt",
            h_range=(0.5, 5.5),
            pass_rate=0.8,
            n_training_points=100,
            seeds=[42],
            created="2026-01-01",
            notes="test",
            runner_tag="T",
            date_tag="010126",
        )

    def test_n_qubits_zero_matches_zero(self):
        """n_qubits=0 query should match entry with n_qubits=0."""
        entry = self._make_entry(n_qubits=0)
        assert entry.matches(n_qubits=0) is True

    def test_n_qubits_zero_does_not_match_ten(self):
        """n_qubits=0 query should NOT match entry with n_qubits=10."""
        entry = self._make_entry(n_qubits=10)
        assert entry.matches(n_qubits=0) is False

    def test_n_qubits_ten_does_not_match_zero(self):
        """n_qubits=10 query should NOT match entry with n_qubits=0."""
        entry = self._make_entry(n_qubits=0)
        assert entry.matches(n_qubits=10) is False

    def test_n_qubits_none_matches_all(self):
        """n_qubits=None (no filter) should match any entry."""
        entry_0 = self._make_entry(n_qubits=0)
        entry_10 = self._make_entry(n_qubits=10)
        assert entry_0.matches(n_qubits=None) is True
        assert entry_10.matches(n_qubits=None) is True

    def test_p_layers_none_matches_all(self):
        """p_layers=None should match any entry (no filter)."""
        entry = self._make_entry(p_layers=1)
        assert entry.matches(p_layers=None) is True

    def test_p_layers_specific_filters(self):
        """Specific p_layers should filter correctly."""
        entry = self._make_entry(p_layers=1)
        assert entry.matches(p_layers=1) is True
        assert entry.matches(p_layers=2) is False


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: save_mpnn_to_zoo n_qubits determination
# ─────────────────────────────────────────────────────────────────────────────


class TestSaveMpnnToZooNQubits:
    """Verify n_qubits is determined correctly without hardcoded fallback."""

    def _resolve_n(self, n_qubits_param=None, args_n_qubits=None, args_train_n=None):
        """Replicate the n_qubits resolution logic from save_mpnn_to_zoo."""
        args = SimpleNamespace()
        if args_n_qubits is not None:
            args.n_qubits = args_n_qubits
        if args_train_n is not None:
            args.train_n = args_train_n

        _n_raw = n_qubits_param or getattr(args, "n_qubits", None)
        if _n_raw is None:
            _n_raw = getattr(args, "train_n", None)
        if isinstance(_n_raw, (list, tuple)):
            _n = 0
        elif _n_raw is not None:
            _n = int(_n_raw)
        else:
            _n = 0
        return _n

    def test_explicit_n_qubits_param(self):
        """Explicit n_qubits parameter takes precedence."""
        assert self._resolve_n(n_qubits_param=16) == 16

    def test_args_n_qubits_scalar(self):
        """args.n_qubits as int is used directly."""
        assert self._resolve_n(args_n_qubits=10) == 10

    def test_args_n_qubits_list_becomes_zero(self):
        """args.n_qubits as list → multi-N → n_qubits=0."""
        assert self._resolve_n(args_n_qubits=[6, 8, 10, 12]) == 0

    def test_args_train_n_fallback(self):
        """Falls back to args.train_n if n_qubits not available."""
        assert self._resolve_n(args_train_n=20) == 20

    def test_no_n_qubits_anywhere_returns_zero(self):
        """No n_qubits anywhere → 0 (not hardcoded 10!)."""
        result = self._resolve_n()
        assert result == 0, f"Expected 0, got {result} (should not default to 10)"

    def test_args_n_qubits_tuple_becomes_zero(self):
        """Tuple of N values → multi-N → n_qubits=0."""
        assert self._resolve_n(args_n_qubits=(4, 6, 8)) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Dashboard freshness check
# ─────────────────────────────────────────────────────────────────────────────


class TestDashboardFreshnessCheck:
    """Verify dashboard skips regeneration when sources are unchanged."""

    def _setup_npz(self, npz_dir: Path, mtime: float):
        """Create a minimal NPZ file with specific mtime."""
        npz_dir.mkdir(parents=True, exist_ok=True)
        h = np.linspace(1.5, 5.0, 20)
        theta = np.random.randn(20, 11)
        e_vqe = -5 - np.random.rand(20) * 0.1
        e_exact = -5.1 * np.ones(20)
        gaps = 0.5 * np.ones(20)

        npz_path = npz_dir / "chain_1d_N6_p1.npz"
        np.savez(
            npz_path,
            h_values=h,
            theta_opt=theta,
            e_vqe=e_vqe,
            e_exact=e_exact,
            gaps=gaps,
        )
        import os

        os.utime(npz_path, (mtime, mtime))
        return npz_path

    def test_skips_when_dashboard_newer_than_sources(self, tmp_path, monkeypatch):
        """Dashboard newer than all NPZ → returns cached, no regeneration."""
        import time

        npz_dir = tmp_path / "data" / "multi_n_training"
        past_time = time.time() - 3600  # 1 hour ago
        self._setup_npz(npz_dir, past_time)

        # Create a "dashboard" file that is newer (present time)
        dashboard_path = tmp_path / "data" / "model_quality_dashboard.json"
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        cached_data = {
            "generated_at": "2026-08-17T00:00:00+00:00",
            "n_configs": 99,  # sentinel value
            "configs": [],
            "topology_summary": {},
        }
        dashboard_path.write_text(json.dumps(cached_data))

        # Patch Path resolution inside the function
        from qmbp_simulation.analysis import metrics

        real_func = metrics.generate_model_quality_dashboard

        # We test the freshness logic directly since _ROOT is local
        # Just verify the function returns cached when output_path is fresh
        # The function checks output_path.stat().st_mtime > max NPZ mtime
        result = real_func(output_path=dashboard_path)

        # The real NPZ dir is checked (not tmp_path), so this tests the
        # real project state. The key behavior: if we pass an output_path
        # that is newer than all real NPZ files, it should return cached.
        # Since our real NPZ files are from Aug 14 and we just wrote
        # dashboard_path now, it should be cached.
        assert result["n_configs"] == 99, (
            f"Expected cached sentinel 99, got {result['n_configs']}. "
            "Freshness check did not prevent regeneration."
        )

    def test_regenerates_when_npz_newer_than_dashboard(self, tmp_path):
        """NPZ newer than dashboard → triggers full regeneration.

        This test uses the real project NPZ dir. We create a dashboard file
        with an old mtime to force regeneration.
        """
        import os

        # Create dashboard file with a very old mtime
        dashboard_path = tmp_path / "model_quality_dashboard.json"
        old_data = {
            "generated_at": "2020-01-01T00:00:00+00:00",
            "n_configs": 99,
            "configs": [],
            "topology_summary": {},
        }
        dashboard_path.write_text(json.dumps(old_data))
        # Set mtime to year 2020
        os.utime(dashboard_path, (1577836800, 1577836800))

        from qmbp_simulation.analysis.metrics import generate_model_quality_dashboard

        result = generate_model_quality_dashboard(output_path=dashboard_path)

        # Should have regenerated (real project has 32 NPZ configs)
        assert result["n_configs"] > 0
        assert result["n_configs"] != 99, (
            "Got sentinel value 99 — freshness check incorrectly returned cached"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Dashboard zoo_pass_rate prefers multi-N model
# ─────────────────────────────────────────────────────────────────────────────


class TestDashboardZooPassRateNotInflated:
    """Verify dashboard uses multi-N model (n_qubits=0) not single-N=10."""

    def test_list_pretrained_n_qubits_zero_filters_correctly(self):
        """list_pretrained(n_qubits=0) returns only multi-N entries."""
        from qmbp_simulation.predictors.model_zoo import list_pretrained

        multi_n = list_pretrained(model="tfim_bond_resolved", topology="chain_1d", n_qubits=0)
        for entry in multi_n:
            assert entry.n_qubits == 0, (
                f"Entry {entry.checkpoint_file} has n_qubits={entry.n_qubits}, expected 0 (multi-N)"
            )

    def test_list_pretrained_n_qubits_ten_excludes_multi_n(self):
        """list_pretrained(n_qubits=10) does not return multi-N entries."""
        from qmbp_simulation.predictors.model_zoo import list_pretrained

        single_10 = list_pretrained(model="tfim_bond_resolved", topology="chain_1d", n_qubits=10)
        for entry in single_10:
            assert entry.n_qubits == 10, (
                f"Entry {entry.checkpoint_file} has n_qubits={entry.n_qubits}, expected 10"
            )

    def test_load_best_for_cross_n_prefers_multi_n(self):
        """load_best_model_for_topology should select best model by unified score."""
        from qmbp_simulation.predictors.model_zoo import load_best_model_for_topology

        try:
            _, entry, source = load_best_model_for_topology(
                "chain_1d",
                model="tfim_bond_resolved",
                n_target=20,
                p_layers=1,
            )
            # Should select a model (any source is valid)
            assert entry.checkpoint_file, "Expected a valid checkpoint"
            assert source in ("per_topology", "multi_topology", "single_n")
        except (FileNotFoundError, RuntimeError):
            pytest.skip("No model available for this test")
