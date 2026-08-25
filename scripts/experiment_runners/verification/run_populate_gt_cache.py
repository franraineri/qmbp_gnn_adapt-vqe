#!/usr/bin/env python3
"""Populate Ground Truth Cache for Grade F configurations.

Computes exact diagonalization (N≤22) or DMRG (N>22) ground truth for
configurations where the GT cache is currently empty. This is a prerequisite
for the QPU verification experiment (Phase 5).

The script uses ValidationRunner.exact_ground_state() which automatically:
- Selects exact diag (N≤22) or DMRG (N>22)
- Caches results at 2 levels (in-memory + disk-persistent GroundTruthCache)
- Validates gaps and energies

Configurations targeted (Grade F from project-status.md):
- heavy_hex N=20,30 (TFIM)
- ladder N=20,26,30 (TFIM)
- square N=20,30 (TFIM — excluded: gap≈0 at h_c makes metric meaningless)
- triangular N=12,16,24 (TFIM — gap≈0 at h_c for 2D frustrated)

Usage:
    # All Grade F configs with dense h-grid
    .venv/bin/python scripts/experiment_runners/verification/run_populate_gt_cache.py

    # Specific topology
    .venv/bin/python scripts/experiment_runners/verification/run_populate_gt_cache.py \
        --topologies heavy_hex --n-values 20 30

    # Quick test (fewer h-points)
    .venv/bin/python scripts/experiment_runners/verification/run_populate_gt_cache.py \
        --topologies heavy_hex --n-values 20 --h-density sparse

    # Dry run (show what would be computed)
    .venv/bin/python scripts/experiment_runners/verification/run_populate_gt_cache.py --dry-run

    # Include square/triangular (warning: gap≈0 near h=1.0)
    .venv/bin/python scripts/experiment_runners/verification/run_populate_gt_cache.py \
        --topologies heavy_hex triangular square ladder --n-values 16 20 24 30
"""
from __future__ import annotations

import logging
import sys
import time

import numpy as np

from qmbp_simulation.framework.runner_base import (
    Section,
    ValidationRunner,
    resolve_project_root,
)

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Defaults — Grade F configurations from project-status.md
# ═══════════════════════════════════════════════════════════════════════════════

# Topologies where we have Grade F at large N
DEFAULT_TOPOLOGIES = ["heavy_hex", "ladder"]

# N values that are Grade F (or needed for verification)
DEFAULT_N_VALUES = [16, 20, 26, 30]

# Maximum N for exact diagonalization (above this → DMRG)
EXACT_DIAG_MAX_N = 22


def _generate_h_grid(density: str = "dense") -> np.ndarray:
    """Generate h-grid for GT population.

    Parameters
    ----------
    density : str
        "dense" — 27 points, non-uniform (standard project grid)
        "sparse" — 8 points for quick testing
        "verification" — 12 points focused on paramagnetic + critical
    """
    if density == "dense":
        # Standard non-uniform grid (dense near h_c=1.0)
        h_outer = np.arange(0.0, 0.8, 0.1)
        h_critical = np.arange(0.8, 1.45, 0.05)
        h_deep = np.arange(1.5, 2.1, 0.1)
        return np.sort(np.concatenate([h_outer, h_critical, h_deep]))
    elif density == "sparse":
        return np.array([0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0])
    elif density == "verification":
        # Focus on paramagnetic (h>1) where HVA works + a few critical points
        return np.array([0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0])
    else:
        raise ValueError(f"Unknown density: {density!r}")


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


class PopulateGTCacheRunner(ValidationRunner):
    """Populate GroundTruthCache for Grade F configurations.

    Computes exact/DMRG ground truth and persists to disk cache for
    subsequent use by MPSPrecisionStudy, HardwareRehearsal, and
    cross-N validation runners.
    """

    runner_id = "populate_gt_cache_v1"
    experiment_id = "verification/populate_gt_cache"
    description = "Populate GT Cache — exact/DMRG ground truth for Grade F configs"
    hypothesis = (
        "All target configurations produce valid ground truth with gap > 0 "
        "in the paramagnetic regime (h > 1.2), establishing reliable reference "
        "energies for QPU verification."
    )

    @classmethod
    def _add_custom_args(cls, parser):
        parser.add_argument(
            "--topologies",
            type=str,
            nargs="+",
            default=DEFAULT_TOPOLOGIES,
            help="Topologies to populate GT for",
        )
        parser.add_argument(
            "--n-values",
            type=int,
            nargs="+",
            default=DEFAULT_N_VALUES,
            help="System sizes to compute GT for",
        )
        parser.add_argument(
            "--h-density",
            type=str,
            choices=["dense", "sparse", "verification"],
            default="verification",
            help="H-grid density: dense (27pts), sparse (8pts), verification (12pts)",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="tfim",
            help="Model name (default: tfim)",
        )
        parser.add_argument(
            "--chi-max",
            type=int,
            default=256,
            help="Max bond dimension for DMRG (N > 22). Default: 256.",
        )

    def run_preflight(self) -> bool:
        """Validate configurations before computing."""
        from qmbp_simulation.models.hamiltonian import make_lattice

        errors = []
        for topology in self._args.topologies:
            for N in self._args.n_values:
                try:
                    make_lattice(topology, N)
                except Exception as e:
                    errors.append(f"  Cannot build {topology} N={N}: {e}")

        if errors:
            for err in errors:
                logger.error(err)
            return False
        return True

    def define_sections(self) -> list[Section]:
        """Define sections: one per topology."""
        sections = []
        for i, topology in enumerate(self._args.topologies, start=1):
            sections.append(
                Section(
                    id=i,
                    name=f"GT Population: {topology}",
                    hypothesis=(
                        f"Exact/DMRG ground truth for {topology} at "
                        f"N={self._args.n_values} produces valid E₀ and gap > 0 "
                        f"in paramagnetic regime."
                    ),
                    fn=lambda topo=topology: self._section_populate_topology(topo),
                )
            )
        return sections

    def _section_populate_topology(self, topology: str) -> dict:
        """Compute and cache GT for all (N, h) in a given topology."""
        h_values = _generate_h_grid(self._args.h_density)
        model = self._args.model
        results = []
        n_cached = 0
        n_computed = 0
        n_gap_zero = 0

        for N in self._args.n_values:
            method = "exact" if N <= EXACT_DIAG_MAX_N else "dmrg"
            logger.info(f"  {topology} N={N} ({method}): {len(h_values)} h-points...")
            t0_n = time.perf_counter()

            for h in h_values:
                t0 = time.perf_counter()

                # Check if already in disk cache
                from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

                disk_cache = getattr(self, "_disk_gt_cache", None)
                if disk_cache is None:
                    disk_cache = GroundTruthCache()
                    self._disk_gt_cache = disk_cache

                cached = disk_cache.get(topology, N, model, h)
                if cached is not None:
                    n_cached += 1
                    continue

                # Compute ground truth (auto-persists via exact_ground_state)
                e_exact, gap = self.exact_ground_state(
                    topology, N, h, model=model
                )
                elapsed = time.perf_counter() - t0
                n_computed += 1

                if gap <= 1e-6:
                    n_gap_zero += 1
                    logger.warning(
                        f"    ⚠ {topology} N={N} h={h:.2f}: gap={gap:.2e} "
                        f"(near-degenerate, ΔE/gap will be unstable)"
                    )

                results.append({
                    "topology": topology,
                    "n_qubits": N,
                    "h": round(float(h), 4),
                    "energy": e_exact,
                    "gap": gap,
                    "method": method,
                    "time_s": round(elapsed, 2),
                })

            elapsed_n = time.perf_counter() - t0_n
            logger.info(
                f"    Done N={N}: {n_computed} computed, "
                f"{n_cached} cached, {elapsed_n:.1f}s total"
            )

        # Ensure disk cache is flushed
        disk_cache = getattr(self, "_disk_gt_cache", None)
        if disk_cache is not None:
            disk_cache.flush()

        # Pass criteria: at least 80% of points in paramagnetic regime have gap > 0
        paramagnetic_results = [r for r in results if r["h"] > 1.2]
        n_valid_gap = sum(1 for r in paramagnetic_results if r["gap"] > 1e-6)
        pass_rate = n_valid_gap / len(paramagnetic_results) if paramagnetic_results else 0

        return {
            "pass": pass_rate >= 0.8,
            "topology": topology,
            "n_computed": n_computed,
            "n_cached": n_cached,
            "n_gap_zero": n_gap_zero,
            "n_total_points": n_computed + n_cached,
            "paramagnetic_valid_gap_rate": round(pass_rate, 3),
            "per_point": results[:20],  # First 20 for report (avoid huge JSON)
            "summary": (
                f"{topology}: {n_computed} computed, {n_cached} already cached, "
                f"{n_gap_zero} near-zero gap. "
                f"Paramagnetic gap validity: {pass_rate:.0%}"
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    PopulateGTCacheRunner.main()
