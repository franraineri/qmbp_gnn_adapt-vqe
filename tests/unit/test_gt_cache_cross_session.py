"""Test ground truth cache cross-session persistence.

Verifies the scenario the large_n_extrapolation runner relies on:
1. Runner A computes a GT point via exact_ground_state() → flushes to disk
2. Runner B (new instance, empty in-memory cache) calls exact_ground_state()
   with the same (topology, N, model, h) → finds it from disk, no recompute

Also tests h-precision alignment between:
- runner_base.exact_ground_state (round(h, 2) in-memory key)
- GroundTruthCache._make_key (h:.2f disk key)
- load_extrapolation_npz (round(h, 6) NPZ key)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from qmbp_simulation.framework.runner_base import Section, ValidationRunner


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


class _GTTestRunner(ValidationRunner):
    """Minimal runner for GT cache testing."""

    runner_id = "gt_cache_test"
    experiment_id = "GT_TEST"
    description = "GT cache persistence test"
    hypothesis = "GT cache survives across sessions"

    def define_sections(self):
        return [Section(id=1, name="Dummy", fn=lambda: {"pass": True}, hypothesis="")]


def _make_args(**kwargs):
    defaults = {
        "section": None,
        "skip_preflight": False,
        "stop_on_failure": False,
        "verbose": False,
        "dry_run": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Cross-session GT cache persistence via exact_ground_state
# ═══════════════════════════════════════════════════════════════════════════════


class TestGTCacheCrossSession:
    """Verify GT computed in one runner session is found in subsequent sessions."""

    def test_exact_ground_state_persists_and_reloads(self, tmp_path):
        """Core test: compute GT in runner A, reload from disk in runner B.

        Uses tmp_path for an isolated cache file so this test doesn't
        pollute or depend on the production ground_truth_cache.json.
        """
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache_path = tmp_path / "gt_cache.json"

        # ── Session A: compute fresh GT ──────────────────────────────────
        runner_a = _GTTestRunner(args=_make_args())

        # Inject isolated disk cache (avoids production file)
        runner_a._disk_gt_cache = GroundTruthCache(path=cache_path)

        # Compute GT for N=4 (small, fast)
        e_exact_a, gap_a = runner_a.exact_ground_state(
            "chain_1d", 4, 2.5, model="tfim"
        )

        # Verify results are physically reasonable
        assert e_exact_a < 0, f"Expected negative energy, got {e_exact_a}"
        assert gap_a > 0, f"Expected positive gap, got {gap_a}"

        # Verify the disk cache file was created and has content
        assert cache_path.exists(), "GT cache file not created after exact_ground_state()"
        with open(cache_path) as f:
            data = json.load(f)
        assert "entries" in data
        assert len(data["entries"]) > 0

        # Verify the key format uses 2-decimal h
        expected_key = "chain_1d|4|tfim|2.50"
        assert expected_key in data["entries"], (
            f"Expected key '{expected_key}' not in cache. "
            f"Keys found: {list(data['entries'].keys())}"
        )

        # ── Session B: new runner, empty in-memory cache ─────────────────
        runner_b = _GTTestRunner(args=_make_args())
        # Fresh disk cache pointing to same file (simulates new process)
        runner_b._disk_gt_cache = GroundTruthCache(path=cache_path)

        # Verify in-memory cache is empty (new runner, fresh session)
        assert len(runner_b._gt_cache) == 0

        # Call exact_ground_state — should hit disk cache, not recompute
        t0 = time.perf_counter()
        e_exact_b, gap_b = runner_b.exact_ground_state(
            "chain_1d", 4, 2.5, model="tfim"
        )
        elapsed = time.perf_counter() - t0

        # Values must match exactly (same source: disk cache)
        assert e_exact_b == e_exact_a, (
            f"Energy mismatch: session A={e_exact_a}, session B={e_exact_b}"
        )
        assert gap_b == gap_a, (
            f"Gap mismatch: session A={gap_a}, session B={gap_b}"
        )

        # Should be fast (cache hit, no DMRG/eigsh)
        assert elapsed < 0.5, (
            f"Cache lookup took {elapsed:.2f}s — expected <0.5s for a cache hit. "
            "GT cache may not be loading correctly from disk."
        )

        # In-memory cache should now be populated (promoted from disk)
        cache_key = ("tfim", "chain_1d", 4, 2.5)
        assert cache_key in runner_b._gt_cache

    def test_multiple_h_values_persist_correctly(self, tmp_path):
        """Multiple h-values stored and retrieved correctly."""
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache_path = tmp_path / "gt_cache.json"

        # Session A: compute multiple h
        runner_a = _GTTestRunner(args=_make_args())
        runner_a._disk_gt_cache = GroundTruthCache(path=cache_path)

        h_values = [2.0, 2.5, 3.0, 3.5, 4.0]
        results_a = {}
        for h in h_values:
            e, gap = runner_a.exact_ground_state("chain_1d", 4, h, model="tfim")
            results_a[h] = (e, gap)

        # Session B: verify all are found
        runner_b = _GTTestRunner(args=_make_args())
        runner_b._disk_gt_cache = GroundTruthCache(path=cache_path)

        for h in h_values:
            e_b, gap_b = runner_b.exact_ground_state("chain_1d", 4, h, model="tfim")
            assert e_b == results_a[h][0], f"Energy mismatch at h={h}"
            assert gap_b == results_a[h][1], f"Gap mismatch at h={h}"

    def test_h_precision_linspace_values_roundtrip(self, tmp_path):
        """h-values from np.linspace (potential floating-point noise) work correctly.

        This tests the exact scenario in run_large_n_extrapolation:
        h_values = [round(h, 2) for h in np.linspace(2.5, 5.0, 6)]
        """
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache_path = tmp_path / "gt_cache.json"

        # Generate h-values the same way the runner does
        h_values = [round(h, 2) for h in np.linspace(2.5, 5.0, 6)]

        # Session A
        runner_a = _GTTestRunner(args=_make_args())
        runner_a._disk_gt_cache = GroundTruthCache(path=cache_path)

        results_a = {}
        for h in h_values:
            e, gap = runner_a.exact_ground_state("chain_1d", 4, float(h), model="tfim")
            results_a[h] = (e, gap)

        # Session B: same linspace generation
        h_values_b = [round(h, 2) for h in np.linspace(2.5, 5.0, 6)]
        runner_b = _GTTestRunner(args=_make_args())
        runner_b._disk_gt_cache = GroundTruthCache(path=cache_path)

        for h in h_values_b:
            e_b, gap_b = runner_b.exact_ground_state("chain_1d", 4, float(h), model="tfim")
            assert e_b == results_a[h][0], (
                f"h={h}: energy mismatch after linspace roundtrip"
            )

    def test_different_topology_n_model_dont_collide(self, tmp_path):
        """Cache keys distinguish topology, N, and model."""
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache_path = tmp_path / "gt_cache.json"
        runner = _GTTestRunner(args=_make_args())
        runner._disk_gt_cache = GroundTruthCache(path=cache_path)

        # Same h, different topology
        e_chain, _ = runner.exact_ground_state("chain_1d", 4, 3.0, model="tfim")
        e_ladder, _ = runner.exact_ground_state("ladder", 4, 3.0, model="tfim")

        # Different topologies should give different energies
        assert e_chain != e_ladder, "chain_1d and ladder should have different GS energies"

        # Reload and verify both are correctly stored
        runner2 = _GTTestRunner(args=_make_args())
        runner2._disk_gt_cache = GroundTruthCache(path=cache_path)

        e_chain_2, _ = runner2.exact_ground_state("chain_1d", 4, 3.0, model="tfim")
        e_ladder_2, _ = runner2.exact_ground_state("ladder", 4, 3.0, model="tfim")

        assert e_chain_2 == e_chain
        assert e_ladder_2 == e_ladder


class TestGTCacheHPrecision:
    """Verify h-precision alignment across the full stack."""

    def test_gt_cache_key_2f_format(self):
        """GroundTruthCache._make_key uses :.2f (e.g., 2.50 not 2.5)."""
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache = GroundTruthCache.__new__(GroundTruthCache)
        cache._data = {}
        cache._dirty = False

        # All these should produce the same key
        key_a = cache._make_key("chain_1d", 10, "tfim", 2.5)
        key_b = cache._make_key("chain_1d", 10, "tfim", 2.50)
        key_c = cache._make_key("chain_1d", 10, "tfim", 2.500000001)
        key_d = cache._make_key("chain_1d", 10, "tfim", 2.4999999999)

        assert key_a == "chain_1d|10|tfim|2.50"
        assert key_b == "chain_1d|10|tfim|2.50"
        assert key_c == "chain_1d|10|tfim|2.50"
        # Note: 2.4999999999 rounds to 2.50 at 2 decimals
        assert key_d == "chain_1d|10|tfim|2.50"

    def test_runner_inmemory_key_matches_disk_key(self, tmp_path):
        """In-memory cache key round(h, 2) aligns with disk key h:.2f."""
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache_path = tmp_path / "gt_cache.json"
        runner = _GTTestRunner(args=_make_args())
        runner._disk_gt_cache = GroundTruthCache(path=cache_path)

        # Use a value that could have precision issues
        h = 2.5000000001  # Tiny floating-point noise

        e, gap = runner.exact_ground_state("chain_1d", 4, h, model="tfim")

        # In-memory key should use round(h, 2) = 2.5
        inmem_key = ("tfim", "chain_1d", 4, round(h, 2))
        assert inmem_key in runner._gt_cache

        # Disk key should be "chain_1d|4|tfim|2.50"
        with open(cache_path) as f:
            data = json.load(f)
        assert "chain_1d|4|tfim|2.50" in data["entries"]

        # Subsequent lookup with clean h=2.5 should find it
        runner2 = _GTTestRunner(args=_make_args())
        runner2._disk_gt_cache = GroundTruthCache(path=cache_path)

        e2, gap2 = runner2.exact_ground_state("chain_1d", 4, 2.5, model="tfim")
        assert e2 == e
        assert gap2 == gap

    def test_npz_h_key_round6_matches_linspace_round2(self):
        """load_extrapolation_npz uses round(h, 6) but values come from round(h, 2).

        Verifies that round(round(h, 2), 6) == round(h, 2) for all relevant
        h-values (no precision drift).
        """
        # Simulate the h-values generated by the runner
        h_values_from_runner = [round(h, 2) for h in np.linspace(2.5, 5.0, 6)]

        for h in h_values_from_runner:
            # The NPZ loader does: round(float(h_from_npz), 6)
            # h_from_npz comes from np.array storage of the runner's h_values
            h_stored = np.float64(h)  # As stored in NPZ
            h_key_npz = round(float(h_stored), 6)

            # The runner lookup uses the same h (from self._h_values which is round(h, 2))
            h_key_lookup = round(float(h), 6)

            assert h_key_npz == h_key_lookup, (
                f"h={h}: NPZ stored key {h_key_npz} != lookup key {h_key_lookup}"
            )

    def test_npz_h_roundtrip_with_linspace_noise(self):
        """Edge case: np.linspace may produce values like 3.0000000000000004.

        After round(h, 2) → store in NPZ → reload → round(h, 6), the key
        must still match.
        """
        # np.linspace can produce floating-point artifacts
        raw_h_values = np.linspace(2.5, 5.0, 6)

        for h_raw in raw_h_values:
            h_rounded = round(float(h_raw), 2)  # What the runner does
            # Store in NPZ (as float64)
            h_stored = np.float64(h_rounded)
            # Reload from NPZ
            h_reloaded = float(h_stored)
            # NPZ key generation
            h_key = round(h_reloaded, 6)

            # Must equal the original rounded value
            assert h_key == h_rounded, (
                f"h_raw={h_raw}: after NPZ roundtrip, "
                f"key={h_key} != original={h_rounded}"
            )


class TestGTCacheFlushBehavior:
    """Verify flush is called at the right times."""

    def test_exact_ground_state_flushes_immediately(self, tmp_path):
        """Each new GT computation triggers an immediate disk flush."""
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache_path = tmp_path / "gt_cache.json"
        runner = _GTTestRunner(args=_make_args())
        runner._disk_gt_cache = GroundTruthCache(path=cache_path)

        # Compute first point
        runner.exact_ground_state("chain_1d", 4, 2.0, model="tfim")

        # File should exist immediately (not deferred)
        assert cache_path.exists(), "Cache file not created after first GT compute"

        # Check that the entry is actually in the file
        with open(cache_path) as f:
            data = json.load(f)
        assert "chain_1d|4|tfim|2.00" in data["entries"]

        # Compute second point
        runner.exact_ground_state("chain_1d", 4, 3.0, model="tfim")

        # Second point should also be persisted immediately
        with open(cache_path) as f:
            data = json.load(f)
        assert "chain_1d|4|tfim|3.00" in data["entries"]
        assert len(data["entries"]) >= 2

    def test_cache_not_dirty_after_flush(self, tmp_path):
        """After flush, cache reports not dirty."""
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache_path = tmp_path / "gt_cache.json"
        runner = _GTTestRunner(args=_make_args())
        runner._disk_gt_cache = GroundTruthCache(path=cache_path)

        runner.exact_ground_state("chain_1d", 4, 2.0, model="tfim")

        # After exact_ground_state, cache should have been flushed
        assert not runner._disk_gt_cache._dirty, (
            "Cache still dirty after exact_ground_state — flush not called?"
        )

    def test_disk_cache_hit_does_not_trigger_flush(self, tmp_path):
        """Reading from disk cache should not mark dirty or trigger write."""
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache_path = tmp_path / "gt_cache.json"

        # Pre-populate cache
        gt = GroundTruthCache(path=cache_path)
        gt.put("chain_1d", 4, "tfim", 2.5, energy=-5.0, gap=0.3)
        gt.flush()

        # Record file modification time
        mtime_before = cache_path.stat().st_mtime

        # Small sleep to ensure mtime would differ
        import time
        time.sleep(0.05)

        # New runner reads from disk cache
        runner = _GTTestRunner(args=_make_args())
        runner._disk_gt_cache = GroundTruthCache(path=cache_path)

        e, gap = runner.exact_ground_state("chain_1d", 4, 2.5, model="tfim")

        # Values should come from cache
        assert abs(e - (-5.0)) < 1e-10
        assert abs(gap - 0.3) < 1e-10

        # File should NOT have been rewritten (cache hit, no new data)
        mtime_after = cache_path.stat().st_mtime
        assert mtime_after == mtime_before, (
            "Cache file was rewritten on a read-only hit — unnecessary I/O"
        )


class TestGTCacheValidation:
    """Verify rejection of invalid values."""

    def test_rejects_nan_energy(self, tmp_path):
        """NaN energy should not be stored in cache."""
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache_path = tmp_path / "gt_cache.json"
        gt = GroundTruthCache(path=cache_path)
        gt.put("chain_1d", 10, "tfim", 2.5, energy=float("nan"), gap=0.3)
        gt.flush()

        # Should not be retrievable
        assert gt.get("chain_1d", 10, "tfim", 2.5) is None

    def test_rejects_negative_gap(self, tmp_path):
        """Negative gap should not be stored."""
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache_path = tmp_path / "gt_cache.json"
        gt = GroundTruthCache(path=cache_path)
        gt.put("chain_1d", 10, "tfim", 2.5, energy=-5.0, gap=-0.1)

        assert gt.get("chain_1d", 10, "tfim", 2.5) is None

    def test_rejects_inf_energy(self, tmp_path):
        """Inf energy should not be stored."""
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache_path = tmp_path / "gt_cache.json"
        gt = GroundTruthCache(path=cache_path)
        gt.put("chain_1d", 10, "tfim", 2.5, energy=float("inf"), gap=0.3)

        assert gt.get("chain_1d", 10, "tfim", 2.5) is None



class TestGetOrCompute:
    """Verify GroundTruthCache.get_or_compute() convenience method."""

    def test_computes_and_caches_on_miss(self, tmp_path):
        """First call computes, second call returns from cache."""
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache_path = tmp_path / "gt_cache.json"
        gt = GroundTruthCache(path=cache_path)

        # First call: computes fresh
        e, gap = gt.get_or_compute("chain_1d", 4, "tfim", 2.5)
        assert e < 0
        assert gap > 0

        # File should exist (flushed immediately)
        assert cache_path.exists()

        # Second call: cache hit (same instance)
        e2, gap2 = gt.get_or_compute("chain_1d", 4, "tfim", 2.5)
        assert e2 == e
        assert gap2 == gap

    def test_cross_instance_persistence(self, tmp_path):
        """Value computed by instance A is found by instance B."""
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache_path = tmp_path / "gt_cache.json"

        # Instance A: compute
        gt_a = GroundTruthCache(path=cache_path)
        e_a, gap_a = gt_a.get_or_compute("chain_1d", 4, "tfim", 3.0)

        # Instance B: should find from disk
        gt_b = GroundTruthCache(path=cache_path)
        e_b, gap_b = gt_b.get_or_compute("chain_1d", 4, "tfim", 3.0)

        assert e_b == e_a
        assert gap_b == gap_a

    def test_flush_false_defers_write(self, tmp_path):
        """flush=False skips immediate disk write."""
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache_path = tmp_path / "gt_cache.json"
        gt = GroundTruthCache(path=cache_path)

        gt.get_or_compute("chain_1d", 4, "tfim", 2.0, flush=False)

        # Data is in memory but may not be on disk yet
        # (unless the batch threshold triggered — which it won't for 1 put)
        assert gt._dirty  # Still dirty because flush=False

        # Manual flush persists
        gt.flush()
        assert not gt._dirty

        # Now a new instance finds it
        gt2 = GroundTruthCache(path=cache_path)
        cached = gt2.get("chain_1d", 4, "tfim", 2.0)
        assert cached is not None

    def test_stale_floor_gap_recomputed(self, tmp_path):
        """A stale floor gap (gap ≈ 2π/N for N>18) triggers recompute."""
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        cache_path = tmp_path / "gt_cache.json"
        gt = GroundTruthCache(path=cache_path)

        # Manually inject a stale entry: N=20, gap = 2π/20 = 0.31415...
        n = 20
        stale_gap = 2 * np.pi / n
        gt.put("chain_1d", n, "tfim", 3.0, energy=-25.0, gap=stale_gap)
        gt.flush()

        # get_or_compute should detect this as stale and recompute
        e, gap = gt.get_or_compute("chain_1d", n, "tfim", 3.0)

        # The recomputed gap should be different from the stale floor
        assert abs(gap - stale_gap) > 1e-4, (
            f"Gap should have been recomputed, but still equals stale floor: {gap}"
        )
        # Energy should be physically reasonable
        assert e < 0

    def test_matches_runner_base_exact_ground_state(self, tmp_path):
        """get_or_compute produces same result as runner_base.exact_ground_state."""
        import argparse

        from qmbp_simulation.framework.runner_base import Section, ValidationRunner
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        class _Runner(ValidationRunner):
            runner_id = "cmp_test"
            experiment_id = "CMP"
            description = "Comparison test"
            hypothesis = ""

            def define_sections(self):
                return [Section(id=1, name="X", fn=lambda: {}, hypothesis="")]

        args = argparse.Namespace(
            section=None, skip_preflight=False, stop_on_failure=False,
            verbose=False, dry_run=False,
        )
        runner = _Runner(args=args)
        runner._disk_gt_cache = GroundTruthCache(path=tmp_path / "runner_gt.json")

        # Via runner_base
        e_runner, gap_runner = runner.exact_ground_state("chain_1d", 4, 2.5, model="tfim")

        # Via get_or_compute
        gt = GroundTruthCache(path=tmp_path / "standalone_gt.json")
        e_standalone, gap_standalone = gt.get_or_compute("chain_1d", 4, "tfim", 2.5)

        assert abs(e_runner - e_standalone) < 1e-10, (
            f"Energy mismatch: runner={e_runner}, standalone={e_standalone}"
        )
        assert abs(gap_runner - gap_standalone) < 1e-10, (
            f"Gap mismatch: runner={gap_runner}, standalone={gap_standalone}"
        )
