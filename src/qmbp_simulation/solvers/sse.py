"""SSE QMC Solver — Stochastic Series Expansion for TFIM ground state energy.

Implements Sandvik's SSE algorithm (PRE 68, 056701, 2003) using Numba-JIT compiled
kernels for performance. Sign-problem-free for ferromagnetic TFIM on any topology.

The inner loops (diagonal update, off-diagonal Poisson resampling, measurement)
are compiled to machine code via Numba, giving ~100-500x speedup over pure Python.

References
----------
- Sandvik, PRE 68, 056701 (2003): SSE for quantum Ising with arbitrary interactions.
- Sandvik, AIP Conf. Proc. 1297, 135 (2010): Computational studies review.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from qmbp_simulation.models import GroundTruthResult, LatticeConfig

logger = logging.getLogger(__name__)


@dataclass
class SSEConfig:
    """Configuration for SSE QMC simulation.

    Attributes
    ----------
    beta : float | None
        Inverse temperature. Auto-scaled if None: β = max(4*N, 100).
    n_thermalize : int
        MC sweeps for thermalization.
    n_measure : int
        MC sweeps for measurements.
    n_bins : int
        Bins for jackknife error estimation.
    seed : int | None
        Random seed for reproducibility.
    target_precision : float
        Target relative error (reserved for adaptive mode).
    max_expansion_order : int | None
        Max operator string length M. Auto-scaled if None.
    """

    beta: float | None = None
    n_thermalize: int = 5000
    n_measure: int = 20000
    n_bins: int = 20
    seed: int | None = 42
    target_precision: float = 1e-6
    max_expansion_order: int | None = None


class SSESolver:
    """SSE QMC solver for TFIM ground state energy (Numba-accelerated).

    Uses compiled kernels from ``_sse_kernels.py`` for the hot loops.
    Provides GroundTruthResult compatible with ClassicalSolver.

    Handles:
    - Arbitrary topologies via explicit edge lists
    - Uniform or per-bond J, per-site h
    - Sign-problem validation before simulation
    - Auto-scaling of β and M

    Limitations:
    - No ground state vector (ground_state=None)
    - Gap estimation uses analytical formula
    - Only TFIM-class Hamiltonians (ZZ + X)
    """

    def __init__(self, config: SSEConfig | None = None) -> None:
        self.config = config or SSEConfig()

    def solve(
        self,
        lattice: LatticeConfig,
        *,
        compute_gap: bool = True,
        compute_observables: bool = True,
    ) -> GroundTruthResult:
        """Compute ground state energy via SSE QMC.

        Parameters
        ----------
        lattice : LatticeConfig
            Lattice with topology, edges, J, h.
        compute_gap : bool
            Whether to estimate spectral gap.
        compute_observables : bool
            Whether to measure ⟨Z_iZ_j⟩.

        Returns
        -------
        GroundTruthResult
        """
        cfg = self.config
        n = lattice.n_qubits
        edges = lattice.edges
        n_bonds = len(edges)

        # Expand couplings/fields
        J_vals = self._expand_couplings(lattice.J, n_bonds)
        h_vals = self._expand_fields(lattice.h, n)
        h_scalar = float(np.mean(h_vals))

        # Validate sign-problem-free
        self._validate_sign_free(J_vals, lattice.topology, edges, n)

        # Auto-scale β for ground state projection
        beta = cfg.beta if cfg.beta is not None else max(4.0 * n, 100.0)

        # Auto-scale M
        energy_scale = float(np.sum(np.abs(J_vals)) + np.sum(np.abs(h_vals)))
        M = cfg.max_expansion_order
        if M is None:
            M = max(int(3.0 * beta * energy_scale), 4 * n)

        logger.info(
            "SSE QMC (Numba): %s N=%d, %d bonds, β=%.1f, M=%d, "
            "thermalize=%d, measure=%d",
            lattice.topology, n, n_bonds, beta, M,
            cfg.n_thermalize, cfg.n_measure,
        )

        # Prepare arrays for Numba kernels
        bond_i = np.array([e[0] for e in edges], dtype=np.int64)
        bond_j = np.array([e[1] for e in edges], dtype=np.int64)
        seed = cfg.seed if cfg.seed is not None else 42

        t0 = time.perf_counter()

        # Run compiled simulation
        from qmbp_simulation.solvers._sse_kernels import run_sse_simulation

        energy, energy_err, mean_zz = run_sse_simulation(
            n=n,
            n_bonds=n_bonds,
            bond_i=bond_i,
            bond_j=bond_j,
            J_vals=J_vals.astype(np.float64),
            h_vals=h_vals.astype(np.float64),
            beta=beta,
            M=M,
            n_thermalize=cfg.n_thermalize,
            n_measure=cfg.n_measure,
            n_bins=cfg.n_bins,
            seed=seed,
        )

        elapsed = time.perf_counter() - t0

        # Gap estimation
        gap = 0.0
        gap_method = "sse_not_computed"
        if compute_gap:
            gap = self._estimate_gap(n, edges, J_vals, h_vals)
            gap_method = "sse_analytical"

        if gap <= 0:
            gap = 2 * np.pi / n
            gap_method = "sse_floor_2pi_n"
            logger.warning(
                "SSE gap ≤ 0 for %s N=%d h=%.4f. Using floor=%.4f.",
                lattice.topology, n, h_scalar, gap,
            )

        logger.info(
            "SSE QMC done: E₀=%.8f ± %.2e, gap=%.6f, %.2fs",
            energy, energy_err, gap, elapsed,
        )

        # Observables
        per_bond_zz = mean_zz if compute_observables else np.zeros(n_bonds)
        per_site_mx = np.zeros(n)  # X magnetization not measured in this version

        return GroundTruthResult(
            h_value=h_scalar,
            ground_energy=float(energy),
            gap=float(gap),
            ground_state=None,
            mag_x=0.0,
            corr_zz=float(np.mean(per_bond_zz)) if n_bonds > 0 else 0.0,
            per_site_mag_x=per_site_mx,
            per_bond_corr_zz=per_bond_zz,
            gap_method=gap_method,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _estimate_gap(
        self,
        n: int,
        edges: list[tuple[int, int]],
        J_vals: np.ndarray,
        h_vals: np.ndarray,
    ) -> float:
        """Estimate spectral gap analytically."""
        h_mean = float(np.mean(h_vals))
        J_mean = float(np.mean(J_vals))
        n_bonds = len(edges)
        z_mean = 2 * n_bonds / n if n > 0 else 2.0

        gap_pert = 2.0 * (h_mean - J_mean * z_mean)
        gap_floor = 2 * np.pi / n

        if gap_pert > gap_floor:
            fs_correction = np.pi**2 / (n**2 * max(h_mean, 1.0))
            return gap_pert + fs_correction
        return gap_floor

    @staticmethod
    def _expand_couplings(J: float | np.ndarray, n_bonds: int) -> np.ndarray:
        """Expand scalar J to per-bond array."""
        if isinstance(J, np.ndarray):
            return J.astype(np.float64)
        return np.full(n_bonds, float(J), dtype=np.float64)

    @staticmethod
    def _expand_fields(h: float | np.ndarray, n: int) -> np.ndarray:
        """Expand scalar h to per-site array."""
        if isinstance(h, np.ndarray):
            return h.astype(np.float64)
        return np.full(n, float(h), dtype=np.float64)

    @staticmethod
    def _validate_sign_free(
        J_vals: np.ndarray,
        topology: str,
        edges: list[tuple[int, int]],
        n: int,
    ) -> None:
        """Validate sign-problem-free condition."""
        if np.all(J_vals > 0):
            return

        if np.all(J_vals < 0):
            _BIPARTITE = ("chain_1d", "ladder", "square", "heavy_hex")
            if topology in _BIPARTITE:
                return
            raise ValueError(
                f"SSE sign problem: AF coupling on non-bipartite '{topology}'. "
                f"Use DMRG or exact diag instead."
            )

        if np.any(J_vals < 0) and np.any(J_vals > 0):
            raise ValueError(
                "SSE sign problem: mixed ferro/antiferro couplings. "
                "Use DMRG or exact diag instead."
            )
