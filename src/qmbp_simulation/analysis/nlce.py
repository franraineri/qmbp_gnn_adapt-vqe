"""NLCE — Numerical Linked-Cluster Expansion for 1D spin systems.

Provides a modular NLCE framework that decomposes bulk (thermodynamic-limit)
properties as sums over finite clusters. For 1D translationally-invariant
systems, clusters are connected intervals of length L.

Architecture:
    NLCERunner
    ├── Cluster enumeration (intervals [1..L] for 1D)
    ├── Cluster solver interface (pluggable: exact diag, VQE, GNN-VQE)
    ├── Euler subtraction (inclusion-exclusion for 1D)
    └── Convergence analysis (Cauchy criterion, weight decay)

The GNN-HVA pipeline serves as the cluster solver for larger clusters
or hardware deployment scenarios.

Usage:
    from qmbp_simulation.analysis.nlce import NLCERunner, NLCEConfig

    config = NLCEConfig(l_max=10, model="tfim", topology="chain_1d")
    runner = NLCERunner(config)
    result = runner.compute(h=2.0)
    print(f"E/N = {result.energy_per_site:.8f}")
    print(f"Converged: {result.converged}")

References:
    - Rigol et al., PRL 97, 187202 (2006): Original NLCE for quantum lattice models
    - Tang et al., PRB 92, 125145 (2015): NLCE for frustrated magnets
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class NLCEConfig:
    """Configuration for an NLCE computation.

    Parameters
    ----------
    l_max : int
        Maximum cluster size (number of sites in longest interval).
    model : str
        Hamiltonian model identifier ("tfim", "tfim_frustrated", etc.).
    topology : str
        Lattice topology for clusters (default "chain_1d").
    J : float
        Nearest-neighbor coupling constant.
    J2 : float
        Next-nearest-neighbor coupling (only for frustrated models).
    solver_method : str
        Cluster solver method: "exact" (default) or "vqe" or "gnn".
    convergence_threshold : float
        Relative Cauchy convergence threshold (|ΔE/E| < threshold).
    periodic : bool
        Whether clusters use periodic boundary conditions.
        For NLCE correctness with open-boundary subtraction, use False.
    """

    l_max: int = 10
    model: str = "tfim"
    topology: str = "chain_1d"
    J: float = 1.0
    J2: float = 0.0
    solver_method: str = "exact"
    convergence_threshold: float = 0.01
    periodic: bool = False


@dataclass
class ClusterResult:
    """Result for a single cluster of size L."""

    L: int
    ground_energy: float
    energy_per_site: float
    gap: float
    solver_time_s: float
    solver_method: str = "exact"


@dataclass
class NLCEResult:
    """Result of an NLCE computation at a single parameter point.

    Contains the thermodynamic-limit estimate and convergence diagnostics.
    """

    h: float
    energy_per_site: float
    l_max: int
    converged: bool
    cauchy_delta: float
    weights: dict[int, float] = field(default_factory=dict)
    partial_sums: dict[int, float] = field(default_factory=dict)
    cluster_results: list[ClusterResult] = field(default_factory=list)
    total_time_s: float = 0.0
    model: str = "tfim"
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "h": self.h,
            "energy_per_site": self.energy_per_site,
            "l_max": self.l_max,
            "converged": self.converged,
            "cauchy_delta": self.cauchy_delta,
            "weights": {str(k): v for k, v in self.weights.items()},
            "partial_sums": {str(k): v for k, v in self.partial_sums.items()},
            "cluster_results": [
                {
                    "L": cr.L,
                    "ground_energy": cr.ground_energy,
                    "energy_per_site": cr.energy_per_site,
                    "gap": cr.gap,
                    "solver_time_s": cr.solver_time_s,
                    "solver_method": cr.solver_method,
                }
                for cr in self.cluster_results
            ],
            "total_time_s": self.total_time_s,
            "model": self.model,
            "extra_params": self.extra_params,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Cluster Solver Interface
# ═══════════════════════════════════════════════════════════════════════════════


class ClusterSolver:
    """Pluggable cluster solver for NLCE.

    Default implementation uses exact diagonalization via ClassicalSolver.
    Override `solve_cluster()` for VQE or GNN-based cluster solving.

    Parameters
    ----------
    model : str
        Hamiltonian model identifier.
    J : float
        NN coupling.
    J2 : float
        NNN coupling (only for frustrated models).
    solver_method : str
        Backend method ("exact" or "dmrg").
    """

    def __init__(
        self,
        model: str = "tfim",
        J: float = 1.0,
        J2: float = 0.0,
        solver_method: str = "exact",
    ) -> None:
        self._model = model
        self._J = J
        self._J2 = J2
        self._solver_method = solver_method

        # Lazy imports to respect module DAG
        from qmbp_simulation.models import HamiltonianBuilder, make_lattice
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.solvers import ClassicalSolver

        self._builder = HamiltonianBuilder()
        self._solver = ClassicalSolver()
        self._make_lattice = make_lattice
        self._get_model_spec = get_model_spec

    def solve_cluster(self, L: int, h: float) -> ClusterResult:
        """Solve ground state for a cluster of size L at field h.

        Parameters
        ----------
        L : int
            Number of sites in the cluster.
        h : float
            Transverse field strength.

        Returns
        -------
        ClusterResult
            Ground state energy, gap, and timing.
        """
        import time

        t0 = time.time()

        # Special case: L=1 (single spin in transverse field)
        # H = -h·X → eigenvalues ±h → E₀ = -h, gap = 2h
        if L == 1:
            elapsed = time.time() - t0
            return ClusterResult(
                L=1,
                ground_energy=-h,
                energy_per_site=-h,
                gap=2 * h,
                solver_time_s=elapsed,
                solver_method="analytical",
            )

        lattice = self._make_lattice("chain_1d", L, J=self._J, h=h, periodic=False)

        # Build Hamiltonian based on model type
        if self._model == "tfim_frustrated" and L >= 3:
            spec = self._get_model_spec("tfim_frustrated")
            H = spec.build_hamiltonian(lattice, J2=self._J2)
        elif self._model == "tfim_frustrated" and L < 3:
            # L<3: no NNN bonds exist → standard TFIM
            H = self._builder.build(lattice)
        elif self._model == "tfim_longitudinal":
            spec = self._get_model_spec("tfim_longitudinal")
            g = self._J2  # Reuse J2 field for longitudinal coupling
            H = spec.build_hamiltonian(lattice, g=g)
        else:
            # Standard TFIM
            H = self._builder.build(lattice)

        # Solve
        method = self._solver_method if L <= 15 else "dmrg"
        gt = self._solver.solve(H, lattice, method=method)

        elapsed = time.time() - t0

        return ClusterResult(
            L=L,
            ground_energy=gt.ground_energy,
            energy_per_site=gt.ground_energy / L,
            gap=gt.gap,
            solver_time_s=elapsed,
            solver_method=method,
        )


class VQEClusterSolver(ClusterSolver):
    """Cluster solver using VQE with MPS backend for larger clusters.

    Extends ClusterSolver for L > 15 where exact diag is infeasible
    but VQE with MPS backend provides accurate results.

    Parameters
    ----------
    model : str
        Hamiltonian model identifier.
    J : float
        NN coupling.
    J2 : float
        NNN coupling.
    chi_max : int
        MPS bond dimension.
    precision : float
        MPS precision parameter.
    seed : int
        Random seed for VQE.
    """

    def __init__(
        self,
        model: str = "tfim",
        J: float = 1.0,
        J2: float = 0.0,
        chi_max: int = 64,
        precision: float = 0.005,
        seed: int = 42,
    ) -> None:
        super().__init__(model=model, J=J, J2=J2, solver_method="exact")
        self._chi_max = chi_max
        self._precision = precision
        self._seed = seed

    def solve_cluster(self, L: int, h: float) -> ClusterResult:
        """Solve cluster — exact for L≤15, MPS-VQE for L>15."""
        if L <= 15:
            return super().solve_cluster(L, h)

        # For larger clusters, use DMRG ground truth (faster than VQE for NLCE)
        import time

        t0 = time.time()
        lattice = self._make_lattice("chain_1d", L, J=self._J, h=h, periodic=False)

        if self._model == "tfim_frustrated" and L >= 3:
            spec = self._get_model_spec("tfim_frustrated")
            H = spec.build_hamiltonian(lattice, J2=self._J2)
        else:
            H = self._builder.build(lattice)

        gt = self._solver.solve(H, lattice, method="dmrg")
        elapsed = time.time() - t0

        return ClusterResult(
            L=L,
            ground_energy=gt.ground_energy,
            energy_per_site=gt.ground_energy / L,
            gap=gt.gap,
            solver_time_s=elapsed,
            solver_method="dmrg",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NLCE Runner
# ═══════════════════════════════════════════════════════════════════════════════


class NLCERunner:
    """Numerical Linked-Cluster Expansion runner for 1D systems.

    Computes thermodynamic-limit properties by summing cluster contributions
    with Euler subtraction. For 1D translationally-invariant systems:

        P(∞) = Σ_{L=1}^{L_max} W(L)

    where W(L) = P(L) - Σ_{l<L} W(l) and P(L) = property per site for
    a cluster of L sites with open boundaries.

    Parameters
    ----------
    config : NLCEConfig
        NLCE configuration.
    cluster_solver : ClusterSolver | None
        Custom cluster solver. If None, uses default exact diag solver.

    Examples
    --------
    >>> config = NLCEConfig(l_max=10, model="tfim")
    >>> runner = NLCERunner(config)
    >>> result = runner.compute(h=2.0)
    >>> print(f"E/N = {result.energy_per_site:.8f}")
    """

    def __init__(
        self,
        config: NLCEConfig,
        cluster_solver: ClusterSolver | None = None,
    ) -> None:
        self._config = config
        self._solver = cluster_solver or ClusterSolver(
            model=config.model,
            J=config.J,
            J2=config.J2,
            solver_method="exact",
        )

    @property
    def config(self) -> NLCEConfig:
        """Return current configuration."""
        return self._config

    def compute(self, h: float, **extra_params) -> NLCEResult:
        """Run NLCE computation at a single field value h.

        Parameters
        ----------
        h : float
            Transverse field strength.
        **extra_params
            Additional parameters stored in result metadata.

        Returns
        -------
        NLCEResult
            Thermodynamic-limit energy estimate with convergence info.
        """
        import time

        t_total_start = time.time()
        L_max = self._config.l_max

        # Step 1: Solve all clusters
        cluster_results: list[ClusterResult] = []
        cluster_energies: dict[int, float] = {}

        for L in range(1, L_max + 1):
            cr = self._solver.solve_cluster(L, h)
            cluster_results.append(cr)
            cluster_energies[L] = cr.energy_per_site

        # Step 2: Euler subtraction
        weights = self._euler_subtraction(cluster_energies, L_max)

        # Step 3: Compute partial sums (for convergence tracking)
        partial_sums: dict[int, float] = {}
        running = 0.0
        for L in range(1, L_max + 1):
            running += weights.get(L, 0.0)
            partial_sums[L] = running

        # Step 4: Final energy per site
        energy_per_site = partial_sums[L_max]

        # Step 5: Convergence analysis
        cauchy_delta = abs(partial_sums[L_max] - partial_sums.get(L_max - 1, 0.0))
        relative_cauchy = cauchy_delta / max(abs(energy_per_site), 1e-15)
        converged = relative_cauchy < self._config.convergence_threshold

        total_time = time.time() - t_total_start

        return NLCEResult(
            h=h,
            energy_per_site=energy_per_site,
            l_max=L_max,
            converged=converged,
            cauchy_delta=cauchy_delta,
            weights=weights,
            partial_sums=partial_sums,
            cluster_results=cluster_results,
            total_time_s=total_time,
            model=self._config.model,
            extra_params=extra_params,
        )

    def compute_sweep(self, h_values: list[float] | np.ndarray) -> list[NLCEResult]:
        """Run NLCE at multiple h-values.

        Parameters
        ----------
        h_values : array-like
            List of transverse field values.

        Returns
        -------
        list[NLCEResult]
            One result per h-value.
        """
        results = []
        for h in h_values:
            result = self.compute(h)
            logger.info(
                f"  NLCE h={h:.3f}: E/N={result.energy_per_site:.8f}, "
                f"converged={result.converged}, Δ={result.cauchy_delta:.2e}, "
                f"t={result.total_time_s:.1f}s"
            )
            results.append(result)
        return results

    @staticmethod
    def _euler_subtraction(cluster_energies: dict[int, float], L_max: int) -> dict[int, float]:
        """Compute NLCE weights via Euler inclusion-exclusion.

        For 1D translationally-invariant systems with OBC clusters:
            W(1) = E(1)/1
            W(L) = E(L)/L - Σ_{l=1}^{L-1} W(l)   for L > 1

        This follows from the fact that a length-L cluster embeds exactly
        one copy of each shorter subcluster in the 1D infinite chain.

        Parameters
        ----------
        cluster_energies : dict[int, float]
            Mapping L → E₀(L)/L.
        L_max : int
            Maximum cluster size.

        Returns
        -------
        dict[int, float]
            Mapping L → W(L).
        """
        weights: dict[int, float] = {}
        for L in range(1, L_max + 1):
            e_per_site = cluster_energies.get(L, 0.0)
            sub_sum = sum(weights.get(l, 0.0) for l in range(1, L))
            weights[L] = e_per_site - sub_sum
        return weights


# ═══════════════════════════════════════════════════════════════════════════════
# Analytical References
# ═══════════════════════════════════════════════════════════════════════════════


def tfim_analytical_energy_per_site(h: float, J: float = 1.0) -> float:
    """Analytical ground state energy per site for infinite 1D TFIM.

    Uses the Jordan-Wigner exact solution:
        E₀/N = -(1/π) ∫₀^π dk √(J² + h² - 2Jh·cos(k))

    Valid for the Hamiltonian H = -J Σ ZᵢZᵢ₊₁ - h Σ Xᵢ with PBC
    in the thermodynamic limit.

    Parameters
    ----------
    h : float
        Transverse field strength.
    J : float
        NN coupling (default 1.0).

    Returns
    -------
    float
        E₀/N in the thermodynamic limit.

    References
    ----------
    Pfeuty, P. (1970). Ann. Phys. 57, 79-90.
    Sachdev, S. (2011). Quantum Phase Transitions, Ch. 5.
    """
    from scipy.integrate import quad

    def integrand(k: float) -> float:
        return -np.sqrt(J**2 + h**2 - 2 * J * h * np.cos(k))

    result, _ = quad(integrand, 0, np.pi, limit=200)
    return result / np.pi


def nlce_convergence_analysis(results: list[NLCEResult]) -> dict[str, Any]:
    """Analyze NLCE convergence across multiple h-values.

    Parameters
    ----------
    results : list[NLCEResult]
        NLCE results from compute_sweep().

    Returns
    -------
    dict
        Convergence summary with per-h diagnostics.
    """
    if not results:
        return {"error": "empty results list"}

    n_converged = sum(1 for r in results if r.converged)
    cauchy_deltas = [r.cauchy_delta for r in results]

    # Weight decay analysis: do |W(L)| decrease with L?
    weight_decays = []
    for r in results:
        if len(r.weights) >= 3:
            last_3 = [abs(r.weights.get(r.l_max - i, 0.0)) for i in range(3)]
            if last_3[2] > 0:
                decay_rate = last_3[0] / last_3[2]
                weight_decays.append(decay_rate)

    return {
        "n_results": len(results),
        "n_converged": n_converged,
        "convergence_rate": n_converged / len(results),
        "mean_cauchy_delta": float(np.mean(cauchy_deltas)),
        "max_cauchy_delta": float(np.max(cauchy_deltas)),
        "mean_weight_decay": float(np.mean(weight_decays)) if weight_decays else None,
        "h_values": [r.h for r in results],
        "energies_per_site": [r.energy_per_site for r in results],
    }
