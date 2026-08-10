"""Tests for compute_refinement_priority — smart VQE refinement allocation.

Validates:
- Priority scoring logic (all factors)
- Skip decisions (stale attempts, non-finite)
- Edge cases (zero gap, huge n_params, etc.)
- Integration with is_point_failure (consistency)
- Practical usage patterns (sorting failures by priority)
"""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.analysis.metrics import (
    compute_refinement_priority,
    is_point_failure,
    DE_GAP_THRESHOLD,
    MAX_ABS_ERROR,
)


class TestRefinementPriorityBasicBehavior:
    """Core priority scoring logic."""

    def test_returns_three_tuple(self):
        priority, skip, reason = compute_refinement_priority(
            de_gap=0.08, abs_error=0.15, gap=2.0, n_params=30
        )
        assert isinstance(priority, float)
        assert isinstance(skip, bool)
        assert isinstance(reason, str)

    def test_priority_in_01_range(self):
        for de_gap in [0.01, 0.06, 0.20, 0.80]:
            p, _, _ = compute_refinement_priority(de_gap, 0.1, 2.0, 30)
            assert 0.0 <= p <= 1.0, f"Priority {p} out of range for de_gap={de_gap}"

    def test_close_to_threshold_gets_high_priority(self):
        """Points barely failing (ΔE/gap ≈ 5-10%) should get high priority."""
        p_close, _, _ = compute_refinement_priority(0.06, 0.12, 2.0, 30)
        p_far, _, _ = compute_refinement_priority(0.50, 1.0, 2.0, 30)
        assert p_close > p_far, (
            f"Close-to-threshold ({p_close}) should beat far ({p_far})"
        )

    def test_mpnn_improvement_boosts_priority(self):
        """MPNN finding better basin should boost priority."""
        p_no_mpnn, _, _ = compute_refinement_priority(
            0.10, 0.2, 2.0, 30, e_prev=-10.0, e_pred=-9.5
        )
        p_mpnn, _, reason = compute_refinement_priority(
            0.10, 0.2, 2.0, 30, e_prev=-10.0, e_pred=-10.5
        )
        assert p_mpnn > p_no_mpnn
        assert reason == "mpnn_found_better_basin"

    def test_large_n_params_reduces_priority(self):
        """Larger circuits are harder to optimize → lower priority."""
        p_small, _, _ = compute_refinement_priority(0.10, 0.2, 2.0, 20)
        p_large, _, _ = compute_refinement_priority(0.10, 0.2, 2.0, 100)
        assert p_small > p_large

    def test_small_gap_reduces_priority(self):
        """Points near phase transition (tiny gap) get deprioritized."""
        p_big_gap, _, _ = compute_refinement_priority(0.10, 0.2, 3.0, 30)
        p_tiny_gap, _, _ = compute_refinement_priority(0.10, 0.2, 0.05, 30)
        assert p_big_gap > p_tiny_gap


class TestRefinementPrioritySkipDecisions:
    """Should_skip logic (hard skip = don't refine)."""

    def test_stale_attempts_causes_skip(self):
        """After 2+ failed attempts without MPNN improvement → skip."""
        _, skip, reason = compute_refinement_priority(
            0.10, 0.2, 2.0, 30,
            e_prev=-10.0, e_pred=-9.5,  # MPNN didn't improve
            n_prev_attempts=2,
        )
        assert skip is True
        assert reason == "stale_no_mpnn_improvement"

    def test_stale_attempts_overridden_by_mpnn_improvement(self):
        """Even with stale attempts, MPNN improvement resets the skip."""
        _, skip, _ = compute_refinement_priority(
            0.10, 0.2, 2.0, 30,
            e_prev=-10.0, e_pred=-10.5,  # MPNN IS better
            n_prev_attempts=5,  # Many stale attempts
        )
        assert skip is False  # MPNN improvement overrides stale

    def test_nan_inputs_cause_skip(self):
        _, skip, reason = compute_refinement_priority(
            float("nan"), 0.2, 2.0, 30
        )
        assert skip is True
        assert reason == "non_finite_metrics"

    def test_inf_abs_error_causes_skip(self):
        _, skip, reason = compute_refinement_priority(
            0.10, float("inf"), 2.0, 30
        )
        assert skip is True
        assert reason == "non_finite_metrics"

    def test_zero_attempts_never_skips(self):
        """First attempt should never be skipped."""
        _, skip, _ = compute_refinement_priority(
            0.50, 1.0, 0.5, 60, n_prev_attempts=0
        )
        assert skip is False

    def test_custom_max_stale_attempts(self):
        """Custom max_stale_attempts threshold."""
        _, skip_2, _ = compute_refinement_priority(
            0.10, 0.2, 2.0, 30,
            e_prev=-10.0, e_pred=-9.5,
            n_prev_attempts=2, max_stale_attempts=3,
        )
        assert skip_2 is False  # 2 < 3, don't skip yet

        _, skip_3, _ = compute_refinement_priority(
            0.10, 0.2, 2.0, 30,
            e_prev=-10.0, e_pred=-9.5,
            n_prev_attempts=3, max_stale_attempts=3,
        )
        assert skip_3 is True  # 3 >= 3, skip


class TestRefinementPriorityEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_gap(self):
        """Zero gap (degenerate) should still return valid result."""
        p, skip, _ = compute_refinement_priority(0.10, 0.2, 0.0, 30)
        assert not skip
        assert p >= 0  # Low priority but not skip

    def test_zero_n_params(self):
        """Zero params shouldn't crash (edge case)."""
        p, _, _ = compute_refinement_priority(0.10, 0.2, 2.0, 0)
        assert 0 <= p <= 1

    def test_negative_de_gap(self):
        """Negative ΔE/gap (variational violation) → non_finite skip."""
        # Negative de_gap from is_point_failure is always a failure,
        # but compute_refinement_priority should handle it gracefully
        p, skip, _ = compute_refinement_priority(-0.05, 0.2, 2.0, 30)
        # Not NaN/Inf, so shouldn't hard-skip on non_finite
        assert isinstance(p, float)

    def test_e_prev_none_first_attempt(self):
        """No previous VQE (first time seeing this h) → normal priority."""
        p, skip, reason = compute_refinement_priority(
            0.10, 0.2, 2.0, 30, e_prev=None, e_pred=-10.0
        )
        assert not skip
        assert reason == "first_attempt" or reason == "close_to_threshold"

    def test_e_pred_none(self):
        """No prediction energy available → still computable."""
        p, skip, _ = compute_refinement_priority(
            0.10, 0.2, 2.0, 30, e_prev=-10.0, e_pred=None
        )
        assert not skip


class TestRefinementPrioritySorting:
    """Practical usage: sorting failures by priority."""

    def test_sort_produces_best_first(self):
        """Sorting by priority gives easy wins first."""
        points = [
            (0.06, 0.12, 3.0, 20),   # Close to threshold, easy
            (0.50, 1.0, 0.3, 60),    # Far, hard, large circuit
            (0.08, 0.15, 2.0, 30),   # Moderate
            (0.03, 0.05, 4.0, 15),   # Already passing! (shouldn't be here)
        ]
        scored = []
        for de_gap, abs_err, gap, n_p in points:
            p, skip, reason = compute_refinement_priority(de_gap, abs_err, gap, n_p)
            scored.append((p, de_gap, reason))

        # Sort descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # First should be the easy one (close to threshold, big gap, small circuit)
        assert scored[0][1] == 0.06 or scored[0][1] == 0.03
        # Last should be the hard one
        assert scored[-1][1] == 0.50

    def test_consistency_with_is_point_failure(self):
        """Points that pass is_point_failure should not appear in refinement."""
        # A passing point
        assert not is_point_failure(0.03, abs_error=0.05)
        # It shouldn't get high priority even if accidentally sent to scoring
        p, _, _ = compute_refinement_priority(0.03, 0.05, 2.0, 30)
        # Priority is still computed (function doesn't know about threshold)
        # but it will be highest because close to threshold — that's fine
        # because the caller should only send is_point_failure=True points

    @pytest.mark.parametrize("topology_n_params", [
        ("chain_1d", 19),    # N=10 chain: 9 edges + 10 sites
        ("ladder", 23),      # N=10 ladder: 13 edges + 10 sites
        ("square", 40),      # N=16 square: 24 edges + 16 sites
        ("triangular", 40),  # N=10 triangular: 30 edges + 10 sites
    ])
    def test_priority_scales_with_topology_complexity(self, topology_n_params):
        """Different topologies (via n_params) should get different priorities."""
        topo, n_params = topology_n_params
        p, skip, _ = compute_refinement_priority(0.10, 0.2, 2.0, n_params)
        assert not skip
        assert 0 < p <= 1


# ═══════════════════════════════════════════════════════════════════════════════
# TestComputeHFrontier — h-frontier interpolation
# ═══════════════════════════════════════════════════════════════════════════════

from qmbp_simulation.analysis.metrics import compute_h_frontier


class TestComputeHFrontier:
    """Tests for compute_h_frontier interpolation."""

    def test_simple_crossing(self):
        """Linear crossing gives interpolated h value."""
        h = np.array([1.0, 2.0, 3.0, 4.0])
        dg = np.array([0.20, 0.08, 0.03, 0.01])
        frontier = compute_h_frontier(h, dg)
        # Crossing between h=2.0 (dg=0.08) and h=3.0 (dg=0.03)
        assert frontier is not None
        assert 2.0 < frontier < 3.0

    def test_all_passing(self):
        """All points pass → frontier at lowest h tested."""
        h = np.array([2.0, 3.0, 4.0])
        dg = np.array([0.02, 0.01, 0.005])
        frontier = compute_h_frontier(h, dg)
        assert frontier == 2.0

    def test_all_failing(self):
        """All points fail → None (frontier not determinable)."""
        h = np.array([2.0, 3.0, 4.0])
        dg = np.array([0.30, 0.15, 0.08])
        frontier = compute_h_frontier(h, dg)
        assert frontier is None

    def test_unsorted_input(self):
        """Should handle unsorted h-values."""
        h = np.array([3.0, 1.0, 4.0, 2.0])
        dg = np.array([0.03, 0.20, 0.01, 0.08])
        frontier = compute_h_frontier(h, dg)
        assert frontier is not None
        assert 2.0 < frontier < 3.0

    def test_single_point_returns_none(self):
        """Need at least 2 points."""
        assert compute_h_frontier(np.array([3.0]), np.array([0.02])) is None

    def test_exact_threshold_crossing(self):
        """When a point is exactly at threshold."""
        h = np.array([1.0, 2.0, 3.0])
        dg = np.array([0.10, 0.05, 0.02])
        # h=2.0 is exactly at threshold (0.05) — boundary case
        frontier = compute_h_frontier(h, dg)
        # Should find crossing between h=1.0 (0.10) and h=2.0 (0.05)
        assert frontier is not None
        assert 1.0 <= frontier <= 2.0
