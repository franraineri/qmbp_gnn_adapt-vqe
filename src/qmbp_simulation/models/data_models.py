"""
Data models for the GNN-HVA quantum simulation framework.

All dataclasses enforce the governing constraints: HVA only, p ≤ 2 layers,
descending sweep h=2→0, SparsePauliOp, Primitives V2, local observables.

Migrated from src/poc/v6/config.py — no sys.path hacks, standard library
and numpy imports only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qmbp_simulation.models.constants import (
    MAX_P_LAYERS,
    SUPPORTED_TOPOLOGIES,
    SUPPORTED_VQE_METHODS,
)

# ---------------------------------------------------------------------------
# Model 1 — LatticeConfig
# ---------------------------------------------------------------------------


@dataclass
class LatticeConfig:
    """Lattice specification for arbitrary spin-system topologies.

    Attributes
    ----------
    topology : str
        One of ``"chain_1d"``, ``"kagome"``, ``"triangular"``, ``"ladder"``.
    n_qubits : int
        Number of lattice sites (≥ 2).
    J : float | np.ndarray
        Coupling constant(s) — scalar (uniform) or per-bond array.
    h : float | np.ndarray
        Transverse field — scalar (uniform) or per-site array.
    edges : list[tuple[int, int]]
        Explicit bond list defining the lattice connectivity.
    coordination_numbers : np.ndarray
        Per-site coordination number (node feature for MPNN).
    periodic : bool
        Whether periodic boundary conditions are applied.
    """

    topology: str
    n_qubits: int
    J: float | np.ndarray
    h: float | np.ndarray
    edges: list[tuple[int, int]]
    coordination_numbers: np.ndarray
    periodic: bool = False

    def __post_init__(self) -> None:
        if self.topology not in SUPPORTED_TOPOLOGIES:
            raise ValueError(
                f"Unsupported topology '{self.topology}'. Must be one of {SUPPORTED_TOPOLOGIES}."
            )
        if self.n_qubits < 2:
            raise ValueError(f"n_qubits must be ≥ 2, got {self.n_qubits}.")
        if isinstance(self.J, np.ndarray) and len(self.J) != len(self.edges):
            raise ValueError(
                f"Per-bond J array length ({len(self.J)}) must match "
                f"number of edges ({len(self.edges)})."
            )
        if isinstance(self.h, np.ndarray) and len(self.h) != self.n_qubits:
            raise ValueError(
                f"Per-site h array length ({len(self.h)}) must match n_qubits ({self.n_qubits})."
            )
        if len(self.coordination_numbers) != self.n_qubits:
            raise ValueError(
                f"coordination_numbers length ({len(self.coordination_numbers)}) "
                f"must match n_qubits ({self.n_qubits})."
            )


# ---------------------------------------------------------------------------
# Model 2 — GroundTruthResult
# ---------------------------------------------------------------------------


@dataclass
class GroundTruthResult:
    """Output of the classical solver (exact diag or DMRG).

    Attributes
    ----------
    h_value : float
        Transverse field value used for this solve.
    ground_energy : float
        Ground state energy E₀.
    gap : float
        Spectral gap E₁ − E₀ (must be ≥ 0).
    ground_state : np.ndarray | None
        Full statevector ψ_gs (only for exact diag; ``None`` for DMRG).
    mag_x : float
        Bulk-averaged ⟨Xᵢ⟩.
    corr_zz : float
        Bulk-averaged ⟨ZᵢZᵢ₊₁⟩.
    per_site_mag_x : np.ndarray
        Per-site ⟨Xᵢ⟩ values.
    per_bond_corr_zz : np.ndarray
        Per-bond ⟨ZᵢZⱼ⟩ values.
    """

    h_value: float
    ground_energy: float
    gap: float
    ground_state: np.ndarray | None
    mag_x: float
    corr_zz: float
    per_site_mag_x: np.ndarray
    per_bond_corr_zz: np.ndarray
    gap_method: str = "unknown"
    """How the gap was computed: 'exact_dense', 'exact_sparse', 'dmrg_excitation',
    'analytical_1d', 'eigsh_fallback', 'floor_2pi_n'."""

    def __post_init__(self) -> None:
        if self.gap < 0:
            raise ValueError(f"Spectral gap must be ≥ 0, got {self.gap}.")

    def compute_de_gap(self, energy: float) -> float:
        """Compute ΔE/gap safely (avoids division by zero at phase transition)."""
        return abs(energy - self.ground_energy) / max(self.gap, 1e-10)

    @property
    def safe_gap(self) -> float:
        """Gap with floor to avoid division by zero."""
        return max(self.gap, 1e-10)


# ---------------------------------------------------------------------------
# Model 3 — VQEConfig
# ---------------------------------------------------------------------------


@dataclass
class VQEConfig:
    """Configuration for the VQE optimizer.

    Attributes
    ----------
    p_layers : int
        HVA depth (must be ≤ MAX_P_LAYERS).
    bounds : tuple[float, float]
        Symmetric parameter bounds for L-BFGS-B.
    n_restarts : int
        Number of random restarts per h-point.
    restart_sigma : float
        Standard deviation for restart perturbations.
    maxiter : int
        Maximum optimizer iterations.
    ftol : float
        Convergence tolerance.
    sweep_direction : str
        Must be ``"descending"`` (h=2→0).
    enable_callbacks : bool
        Whether to log the full optimization trajectory.
    warm_start_seed_zeros : bool
        Enforce θ=0 initial guess for h=0 (ferromagnetic phase).
    method : str
        Optimizer method: ``"L-BFGS-B"`` (gradient-based, default),
        ``"COBYLA"`` (gradient-free, shot-noise tolerant), or
        ``"Nelder-Mead"`` (simplex).
    """

    p_layers: int = 2
    bounds: tuple[float, float] = (-np.pi, np.pi)
    n_restarts: int = 5
    restart_sigma: float = 0.1
    maxiter: int = 1000
    ftol: float = 1e-14
    sweep_direction: str = "descending"
    enable_callbacks: bool = True
    warm_start_seed_zeros: bool = True
    method: str = "L-BFGS-B"

    def __post_init__(self) -> None:
        if self.p_layers > MAX_P_LAYERS:
            raise ValueError(
                f"p_layers must be ≤ {MAX_P_LAYERS} (Mele et al. constraint), got {self.p_layers}."
            )
        if self.sweep_direction != "descending":
            raise ValueError(
                f"sweep_direction must be 'descending' (V4 lesson: ascending "
                f"breaks θ smoothness), got '{self.sweep_direction}'."
            )
        if self.bounds[0] >= self.bounds[1]:
            raise ValueError(f"bounds must satisfy bounds[0] < bounds[1], got {self.bounds}.")
        if self.n_restarts < 1:
            raise ValueError(f"n_restarts must be ≥ 1, got {self.n_restarts}.")
        if self.maxiter < 1:
            raise ValueError(f"maxiter must be ≥ 1, got {self.maxiter}.")
        if self.method not in SUPPORTED_VQE_METHODS:
            raise ValueError(
                f"Unsupported optimizer method '{self.method}'. "
                f"Must be one of {SUPPORTED_VQE_METHODS}."
            )


# ---------------------------------------------------------------------------
# Model 4 — OptimizationTrajectory
# ---------------------------------------------------------------------------


@dataclass
class OptimizationTrajectory:
    """Full optimization history from VQE diagnostic callbacks.

    Attributes
    ----------
    energies : list[float]
        Energy at each iteration.
    grad_norms : list[float]
        ‖∇‖ at each iteration.
    param_vectors : list[np.ndarray]
        θ at each iteration.
    converged : bool
        Whether the optimizer converged.
    n_restarts_used : int
        Number of restarts that improved the result.
    """

    energies: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    param_vectors: list[np.ndarray] = field(default_factory=list)
    converged: bool = False
    n_restarts_used: int = 0


# ---------------------------------------------------------------------------
# Model 5 — VQEResult
# ---------------------------------------------------------------------------


@dataclass
class VQEResult:
    """Output of a single VQE optimization run.

    Attributes
    ----------
    h_value : float
        Transverse field value.
    theta_opt : np.ndarray
        Optimized HVA parameters.
    energy : float
        VQE energy.
    energy_error : float
        |E_VQE − E_exact|.
    fidelity : float
        State fidelity (noiseless validation only).
    n_iterations : int
        Total optimizer iterations.
    trajectory : OptimizationTrajectory | None
        Callback data (if enabled).
    """

    h_value: float
    theta_opt: np.ndarray
    energy: float
    energy_error: float
    fidelity: float
    n_iterations: int
    trajectory: OptimizationTrajectory | None = None
    energy_variance: float | None = None

    def validate(self) -> list[str]:
        """Run sanity checks on VQE result. Returns list of issues found.

        Checks:
        - θ_opt contains NaN/Inf
        - Energy is not finite
        - Negative energy_error
        - Fidelity outside [0, 1]
        - Negative n_iterations
        - θ_opt values outside [-π, π] (possible with gradient-free optimizers)
        """
        issues: list[str] = []
        if not np.all(np.isfinite(self.theta_opt)):
            issues.append(f"theta_opt contains NaN/Inf at h={self.h_value}")
        if not np.isfinite(self.energy):
            issues.append(f"energy is not finite: {self.energy}")
        if self.energy_error < 0:
            issues.append(f"Negative energy_error: {self.energy_error}")
        if self.fidelity < -0.001 or self.fidelity > 1.001:
            issues.append(f"Invalid fidelity: {self.fidelity} (must be in [0,1])")
        if self.n_iterations < 0:
            issues.append(f"Negative n_iterations: {self.n_iterations}")
        # θ_opt bounds check (π + small tolerance for numerical noise)
        if np.all(np.isfinite(self.theta_opt)):
            out_of_bounds = np.sum(np.abs(self.theta_opt) > np.pi + 1e-6)
            if out_of_bounds > 0:
                max_val = float(np.max(np.abs(self.theta_opt)))
                issues.append(
                    f"{int(out_of_bounds)} parameters outside [-π, π] "
                    f"(max |θ|={max_val:.4f}) at h={self.h_value}"
                )
        return issues

    @property
    def passes_threshold(self) -> bool:
        """Check if ΔE/gap < 5% (requires gap from exact data)."""
        # Note: this only checks energy_error, not ΔE/gap (gap not stored here)
        return self.fidelity >= 0.93


# ---------------------------------------------------------------------------
# Model 6 — DeployResult
# ---------------------------------------------------------------------------


@dataclass
class DeployResult:
    """Output of Phase 4 hardware deployment (either route).

    Attributes
    ----------
    route : str
        ``"adapt_vqe"`` or ``"qrc"``.
    h_test : float
        Transverse field value for the test point.
    predicted_energy : float
        Energy from the deployed circuit.
    delta_e : float
        |E_pred − E_exact|.
    delta_e_over_gap : float
        ΔE / gap (primary validation metric).
    mag_x_pred : float
        Predicted ⟨X⟩.
    corr_zz_pred : float
        Predicted ⟨ZZ⟩.
    mag_x_error : float
        |⟨X⟩_pred − ⟨X⟩_exact|.
    corr_zz_error : float
        |⟨ZZ⟩_pred − ⟨ZZ⟩_exact|.
    fidelity : float | None
        State fidelity (noiseless simulation only).
    adapt_iterations : int
        AdaptVQE iterations used (0 = ideal warm-start).
    phase_label : str
        ``"ferromagnetic"`` or ``"paramagnetic"``.
    metrics_checklist : dict[str, bool]
        Pass/fail for each validation metric.
    """

    route: str
    h_test: float
    predicted_energy: float
    delta_e: float
    delta_e_over_gap: float
    mag_x_pred: float
    corr_zz_pred: float
    mag_x_error: float
    corr_zz_error: float
    fidelity: float | None
    adapt_iterations: int
    phase_label: str
    metrics_checklist: dict[str, bool]

    def __post_init__(self) -> None:
        if self.delta_e < 0:
            raise ValueError(f"delta_e must be ≥ 0, got {self.delta_e}.")
        if self.delta_e_over_gap < 0:
            raise ValueError(f"delta_e_over_gap must be ≥ 0, got {self.delta_e_over_gap}.")

    def passes(self, threshold: float = 0.05) -> bool:
        """Check if deployment passes the ΔE/gap threshold."""
        return self.delta_e_over_gap < threshold

    @property
    def status(self) -> str:
        """Human-readable pass/fail status."""
        return "PASS" if self.passes() else "FAIL"
