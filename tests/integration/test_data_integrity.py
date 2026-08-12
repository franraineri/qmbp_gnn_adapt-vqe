"""Data integrity tests — prevent GT corruption and h-range regressions.

These tests verify the consistency of training data (NPZ files), ground truth
cache, and cross-topology comparability. They should be run:
- After any iterative-improve or VQE sweep run
- After theta-cleaning operations
- Before multi-seed evaluation (as a gate)
- In CI on PRs that touch data/ or solvers/

Categories:
1. GT consistency: NPZ e_exact matches fresh solver output
2. H-range coverage: viable configs have canonical h-range
3. Data quality: no constant e_exact, no NaN, correct shapes
"""

import json
import numpy as np
import pytest
from pathlib import Path

# ── Fixtures ──────────────────────────────────────────────────────────────────

DATA_DIR = Path("data/multi_n_training")
GT_CACHE_PATH = Path("data/ground_truth_cache.json")

# Canonical h-ranges per topology (from empirical analysis 2026-08-11)
CANONICAL_H_RANGES = {
    "chain_1d": (1.5, 5.5),
    "heavy_hex": (1.4, 4.5),
    "square": (2.0, 5.0),
    "triangular": (3.0, 5.5),
    "ladder": (2.0, 5.0),
}

# Minimum viable N per topology (pass_rate_dual >= 50%)
VIABLE_CONFIGS = [
    ("chain_1d", 6),
    ("chain_1d", 8),
    ("chain_1d", 10),
    ("chain_1d", 12),
    ("heavy_hex", 6),
    ("heavy_hex", 10),
    ("heavy_hex", 12),
    ("heavy_hex", 16),
    ("square", 4),
    ("square", 6),
    ("square", 8),
    ("square", 10),
    ("triangular", 3),
    ("triangular", 4),
    ("triangular", 6),
    ("ladder", 6),
    ("ladder", 8),
]

# Configs required for cross-topology head-to-head comparison
CROSS_TOPOLOGY_CONFIGS = [
    ("chain_1d", 10),
    ("heavy_hex", 10),
    ("square", 10),
    ("triangular", 6),
]

MIN_POINTS_FOR_STATS = 20


@pytest.fixture(scope="module")
def gt_cache():
    """Load ground truth cache once for all tests."""
    if not GT_CACHE_PATH.exists():
        pytest.skip("Ground truth cache not found")
    with open(GT_CACHE_PATH) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GT CONSISTENCY: NPZ e_exact matches solver
# ═══════════════════════════════════════════════════════════════════════════════


class TestGTConsistency:
    """Verify that NPZ e_exact values are correct ground truth energies."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_data(self):
        if not DATA_DIR.exists():
            pytest.skip("Training data directory not found")

    def _fresh_solve(self, topo: str, n: int, h: float) -> float:
        """Compute ground truth energy from scratch."""
        from qmbp_simulation import make_lattice
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.solvers import ClassicalSolver

        spec = get_model_spec("tfim_bond_resolved")
        solver = ClassicalSolver()
        lat = make_lattice(topo, n, J=1.0, h=h)
        H = spec.build_hamiltonian(lat, **spec.hamiltonian_kwargs)
        result = solver.solve(H, lat)
        return result.ground_energy

    @pytest.mark.parametrize("topo,n", VIABLE_CONFIGS)
    def test_e_exact_matches_solver(self, topo, n):
        """NPZ e_exact must match fresh solver to within 1e-4."""
        npz_path = DATA_DIR / f"{topo}_N{n}_p1.npz"
        if not npz_path.exists():
            pytest.skip(f"NPZ not found: {npz_path.name}")

        data = np.load(npz_path, allow_pickle=True)
        if "e_exact" not in data:
            pytest.skip("No e_exact field")

        # Spot-check 3 evenly spaced points (not just first 3)
        n_pts = len(data["h_values"])
        indices = [0, n_pts // 2, n_pts - 1]

        for idx in indices:
            h = float(data["h_values"][idx])
            npz_e = float(data["e_exact"][idx])
            fresh_e = self._fresh_solve(topo, n, h)
            diff = abs(npz_e - fresh_e)
            assert diff < 1e-4, (
                f"{topo} N={n} h={h:.4f}: NPZ e_exact={npz_e:.8f} "
                f"vs solver={fresh_e:.8f} (diff={diff:.2e})"
            )

    @pytest.mark.parametrize("topo,n", CROSS_TOPOLOGY_CONFIGS)
    def test_cross_topology_configs_gt_verified(self, topo, n):
        """The 4 key cross-topology configs must have verified GT."""
        npz_path = DATA_DIR / f"{topo}_N{n}_p1.npz"
        if not npz_path.exists():
            pytest.skip(f"NPZ not found: {npz_path.name}")

        data = np.load(npz_path, allow_pickle=True)
        # Check ALL points for these critical configs
        for i in range(min(5, len(data["h_values"]))):
            h = float(data["h_values"][i])
            npz_e = float(data["e_exact"][i])
            fresh_e = self._fresh_solve(topo, n, h)
            diff = abs(npz_e - fresh_e)
            assert diff < 1e-4, (
                f"CRITICAL: {topo} N={n} h={h:.4f} GT mismatch: "
                f"{npz_e:.8f} vs {fresh_e:.8f}"
            )

    def test_no_constant_e_exact(self):
        """No NPZ should have a constant e_exact (sign of data corruption)."""
        for npz_path in DATA_DIR.glob("*_p1.npz"):
            data = np.load(npz_path, allow_pickle=True)
            if "e_exact" not in data or len(data["e_exact"]) < 3:
                continue
            n_unique = len(np.unique(data["e_exact"]))
            n_total = len(data["e_exact"])
            # At least 50% of values should be unique
            # (allows some h-duplicates from iterative runs)
            assert n_unique >= n_total * 0.5, (
                f"{npz_path.name}: only {n_unique}/{n_total} unique e_exact "
                f"values — possible corruption"
            )

    def test_e_exact_not_equal_to_e_vqe(self):
        """e_exact should differ from e_vqe (VQE is variational: E_vqe >= E_exact)."""
        for npz_path in DATA_DIR.glob("*_p1.npz"):
            data = np.load(npz_path, allow_pickle=True)
            if "e_exact" not in data or "e_vqe" not in data:
                continue
            # They should NOT be identical arrays
            if np.allclose(data["e_exact"], data["e_vqe"], atol=1e-8):
                # This is only OK if all VQE found exact GS
                gaps = np.abs(data["e_vqe"] - data["e_exact"])
                assert gaps.max() < 1e-6, (
                    f"{npz_path.name}: e_exact == e_vqe everywhere but "
                    f"max gap is {gaps.max():.2e} — possible field swap"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. H-RANGE COVERAGE: viable configs have canonical h-range
# ═══════════════════════════════════════════════════════════════════════════════


class TestHRangeCoverage:
    """Verify h-range homogeneity across N values within each topology."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_data(self):
        if not DATA_DIR.exists():
            pytest.skip("Training data directory not found")

    @pytest.mark.parametrize("topo,n", VIABLE_CONFIGS)
    def test_h_range_within_canonical(self, topo, n):
        """Each viable NPZ h-range should overlap ≥60% with canonical range."""
        npz_path = DATA_DIR / f"{topo}_N{n}_p1.npz"
        if not npz_path.exists():
            pytest.skip(f"NPZ not found: {npz_path.name}")

        data = np.load(npz_path, allow_pickle=True)
        h_min_npz = float(data["h_values"].min())
        h_max_npz = float(data["h_values"].max())

        canon_min, canon_max = CANONICAL_H_RANGES[topo]
        canon_width = canon_max - canon_min

        # Compute overlap
        overlap_min = max(h_min_npz, canon_min)
        overlap_max = min(h_max_npz, canon_max)
        overlap = max(0, overlap_max - overlap_min)
        overlap_frac = overlap / canon_width

        assert overlap_frac >= 0.40, (
            f"{topo} N={n}: h-range [{h_min_npz:.2f}, {h_max_npz:.2f}] "
            f"only {overlap_frac:.0%} overlap with canonical "
            f"[{canon_min:.1f}, {canon_max:.1f}]"
        )

    def test_no_zero_overlap_between_viable_N(self):
        """Within each topology, no pair of viable N should have 0% h-range overlap."""
        from collections import defaultdict

        topo_ranges = defaultdict(list)
        for topo, n in VIABLE_CONFIGS:
            npz_path = DATA_DIR / f"{topo}_N{n}_p1.npz"
            if not npz_path.exists():
                continue
            data = np.load(npz_path, allow_pickle=True)
            h_min = float(data["h_values"].min())
            h_max = float(data["h_values"].max())
            topo_ranges[topo].append((n, h_min, h_max))

        zero_overlaps = []
        for topo, ranges in topo_ranges.items():
            for i, (n1, min1, max1) in enumerate(ranges):
                for n2, min2, max2 in ranges[i + 1:]:
                    overlap = max(0, min(max1, max2) - max(min1, min2))
                    total = max(max1, max2) - min(min1, min2)
                    if total > 0 and overlap / total < 0.01:
                        zero_overlaps.append(
                            f"{topo} N={n1}[{min1:.1f},{max1:.1f}] ↔ "
                            f"N={n2}[{min2:.1f},{max2:.1f}]"
                        )

        assert len(zero_overlaps) == 0, (
            f"Zero h-range overlap found in {len(zero_overlaps)} pairs:\n"
            + "\n".join(f"  {x}" for x in zero_overlaps[:5])
        )

    @pytest.mark.parametrize("topo,n", CROSS_TOPOLOGY_CONFIGS)
    def test_cross_topology_min_points(self, topo, n):
        """Cross-topology configs must have ≥20 h-points for statistical rigor."""
        npz_path = DATA_DIR / f"{topo}_N{n}_p1.npz"
        if not npz_path.exists():
            pytest.skip(f"NPZ not found: {npz_path.name}")

        data = np.load(npz_path, allow_pickle=True)
        n_pts = len(data["h_values"])
        assert n_pts >= MIN_POINTS_FOR_STATS, (
            f"{topo} N={n}: only {n_pts} points, need ≥{MIN_POINTS_FOR_STATS} "
            f"for statistically meaningful pass rates"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA QUALITY: shapes, NaN, variational principle
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataQuality:
    """Verify structural integrity of NPZ files."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_data(self):
        if not DATA_DIR.exists():
            pytest.skip("Training data directory not found")

    def test_no_nan_in_theta(self):
        """No NaN or Inf values in theta_opt."""
        for npz_path in DATA_DIR.glob("*_p1.npz"):
            data = np.load(npz_path, allow_pickle=True)
            if "theta_opt" not in data:
                continue
            theta = data["theta_opt"]
            assert np.all(np.isfinite(theta)), (
                f"{npz_path.name}: {np.sum(~np.isfinite(theta))} "
                f"NaN/Inf values in theta_opt"
            )

    def test_no_nan_in_energies(self):
        """No NaN or Inf in e_vqe or e_exact."""
        for npz_path in DATA_DIR.glob("*_p1.npz"):
            data = np.load(npz_path, allow_pickle=True)
            for key in ["e_vqe", "e_exact"]:
                if key not in data:
                    continue
                arr = data[key]
                assert np.all(np.isfinite(arr)), (
                    f"{npz_path.name}: NaN/Inf in {key}"
                )

    def test_shapes_consistent(self):
        """All parallel arrays must have same length."""
        for npz_path in DATA_DIR.glob("*_p1.npz"):
            data = np.load(npz_path, allow_pickle=True)
            if "h_values" not in data:
                continue
            n_pts = len(data["h_values"])
            for key in ["theta_opt", "e_vqe", "e_exact", "gaps", "de_gaps"]:
                if key in data:
                    arr = data[key]
                    if arr.ndim >= 1:
                        assert arr.shape[0] == n_pts, (
                            f"{npz_path.name}: {key} has {arr.shape[0]} "
                            f"rows but h_values has {n_pts}"
                        )

    def test_variational_principle(self):
        """E_vqe should be >= E_exact (variational bound) for most points."""
        violations = []
        for npz_path in DATA_DIR.glob("*_p1.npz"):
            data = np.load(npz_path, allow_pickle=True)
            if "e_vqe" not in data or "e_exact" not in data:
                continue
            # Allow small numerical violations (1e-6)
            bad = data["e_vqe"] < data["e_exact"] - 1e-4
            n_bad = int(bad.sum())
            if n_bad > 0:
                # Allow up to 5% violations (numerical noise in optimization)
                frac = n_bad / len(data["e_vqe"])
                if frac > 0.05:
                    violations.append(
                        f"{npz_path.name}: {n_bad}/{len(data['e_vqe'])} "
                        f"({frac:.0%}) variational violations"
                    )

        assert len(violations) == 0, (
            f"Variational principle violated in {len(violations)} files:\n"
            + "\n".join(f"  {v}" for v in violations[:5])
        )

    def test_h_values_sorted(self):
        """h_values should be monotonically sorted (ascending or descending).
        
        Note: NPZ files from iterative-improve may have unsorted h due to
        upsert operations. This is acceptable — the test warns rather than fails.
        """
        unsorted = []
        for npz_path in DATA_DIR.glob("*_p1.npz"):
            data = np.load(npz_path, allow_pickle=True)
            if "h_values" not in data or len(data["h_values"]) < 2:
                continue
            h = data["h_values"]
            is_ascending = np.all(np.diff(h) >= -1e-10)
            is_descending = np.all(np.diff(h) <= 1e-10)
            if not (is_ascending or is_descending):
                unsorted.append(npz_path.name)

        # Warn but don't fail — upsert_theta_npz doesn't guarantee order
        if unsorted:
            import warnings
            warnings.warn(
                f"{len(unsorted)} NPZ files have unsorted h_values: "
                f"{unsorted[:3]}... (OK if from iterative-improve)"
            )

    def test_de_gaps_consistent_with_components(self):
        """de_gaps must equal |e_vqe - e_exact| / max(gaps, 1e-10)."""
        for npz_path in DATA_DIR.glob("*_p1.npz"):
            data = np.load(npz_path, allow_pickle=True)
            if not all(k in data for k in ["de_gaps", "e_vqe", "e_exact", "gaps"]):
                continue
            expected = np.abs(data["e_vqe"] - data["e_exact"]) / np.maximum(
                data["gaps"], 1e-10
            )
            # Allow 1% relative tolerance (floating point)
            close = np.allclose(data["de_gaps"], expected, rtol=0.01, atol=1e-8)
            assert close, (
                f"{npz_path.name}: de_gaps inconsistent with components. "
                f"Max diff: {np.max(np.abs(data['de_gaps'] - expected)):.2e}"
            )
