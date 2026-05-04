"""
Shared data models for the GNN-HVA v6.0 pipeline.

All dataclasses follow the design document specifications and enforce
the governing constraints: HVA only, p ≤ 2 layers, SparsePauliOp,
Primitives V2, local observables, descending sweep h=2→0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    import torch_geometric.data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_TOPOLOGIES = ("chain_1d", "kagome", "triangular", "ladder")
MAX_P_LAYERS = 2
EXACT_DIAG_QUBIT_LIMIT = 15
DMRG_QUBIT_LIMIT = 40


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

    def __post_init__(self) -> None:
        if self.gap < 0:
            raise ValueError(f"Spectral gap must be ≥ 0, got {self.gap}.")


# ---------------------------------------------------------------------------
# Model 3 — VQEConfig
# ---------------------------------------------------------------------------


@dataclass
class VQEConfig:
    """Configuration for the VQE optimizer.

    Attributes
    ----------
    p_layers : int
        HVA depth (MUST be ≤ 2 per Mele et al. constraint).
    bounds : tuple[float, float]
        Symmetric parameter bounds for L-BFGS-B.
    n_restarts : int
        Number of random restarts per h-point.
    restart_sigma : float
        Standard deviation for restart perturbations.
    maxiter : int
        Maximum L-BFGS-B iterations.
    ftol : float
        Convergence tolerance.
    sweep_direction : str
        Must be ``"descending"`` (h=2→0).
    enable_callbacks : bool
        Whether to log the full optimization trajectory.
    warm_start_seed_zeros : bool
        Enforce θ=0 initial guess for h=0 (ferromagnetic phase).
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


# ---------------------------------------------------------------------------
# Model 5 — OptimizationTrajectory (new in V6)
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
# Model 4 — VQEResult
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


# ---------------------------------------------------------------------------
# Model 6 — GraphData
#
# NOTE: This is a *factory helper* that produces torch_geometric.data.Data
# objects.  We define a lightweight builder rather than subclassing Data so
# that the config module stays import-safe when torch_geometric is not
# installed (e.g. during Phase 1/2 work).
# ---------------------------------------------------------------------------


def make_graph_data(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    e_exact: float,
) -> torch_geometric.data.Data:
    """Create a ``torch_geometric.data.Data`` instance for MPNN training.

    Parameters
    ----------
    x : Tensor [n_qubits, 2]
        Node features: (h_i, coordination_number_i).
    edge_index : Tensor [2, n_edges]
        Bond connectivity in COO format (must be symmetric / undirected).
    y : Tensor [2p]
        Target θ_opt.
    e_exact : float
        Exact ground state energy (for energy-driven validation).

    Returns
    -------
    torch_geometric.data.Data
    """
    import torch_geometric.data as tgd  # noqa: F811 — lazy import

    data = tgd.Data(x=x, edge_index=edge_index, y=y)
    data.e_exact = e_exact
    return data


# ---------------------------------------------------------------------------
# Model 7 — DeployResult
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
