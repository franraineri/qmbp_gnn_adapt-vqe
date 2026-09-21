"""C1: Physics-Informed MPNN Loss.

Adds an energy-validation term to the MPNN training loss that evaluates
E(θ_pred) on the actual Hamiltonian every K epochs. This prevents the MPNN
from learning parameters with low MSE but high energy error.

Optionally adds a FIDELITY-validation term F = |<psi_exact|psi(theta_pred)>|^2
for system sizes where the exact ground-state vector is computable (N <= 22).
Fidelity directly targets state overlap — the quantity that matters for
quench dynamics state preparation — where low energy alone does not
guarantee high overlap (relevant near small gaps / degeneracies).

IMPORTANT: This does NOT change the VQE cost function (V5.x lesson).
The theta targets remain pure-energy VQE optima. Both the energy and
fidelity terms are MPNN training regularizers only.

References:
    - Miao et al. (2024) PRApplied 21, 014053 — energy-aware NN training
    - Zhang et al. (2025) arXiv:2505.01236 (Qracle) — GNN with energy feedback
    - Lee et al. (2026) arXiv:2602.19752 — energy-based loss for VQE params
    - arXiv:2606.15061 — low energy does not imply high ground-state overlap
"""

from __future__ import annotations

import numpy as np
import torch


class PhysicsInformedLoss(torch.nn.Module):
    """Combined MSE + energy (+ optional fidelity) validation loss for MPNN.

    loss = MSE(theta_pred, theta_target)
         + weight       * mean(|E(theta_pred) - E_exact|)
         + fidelity_weight * mean(1 - F(theta_pred))       [if enabled, N<=22]

    Both physics terms are only evaluated every ``eval_every`` epochs and on a
    random subset of training points (for efficiency). The fidelity term is
    optional and gracefully degrades to energy-only for N>22 (where the exact
    ground-state vector is not available).
    """

    def __init__(
        self,
        weight: float = 0.1,
        start_epoch: int = 1000,
        eval_every: int = 100,
        n_eval_points: int = 5,
        fidelity_weight: float = 0.0,
    ):
        """Initialize physics-informed loss.

        Parameters
        ----------
        weight : float
            Weight for the energy term.
        start_epoch : int
            Epoch at which to start adding physics terms.
        eval_every : int
            Evaluate physics terms every N epochs.
        n_eval_points : int
            Number of random training points to evaluate on.
        fidelity_weight : float
            Weight for the fidelity term (1 - F). Default 0.0 (disabled).
            When > 0, adds a fidelity penalty for N<=22 training points.
        """
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.weight = weight
        self.start_epoch = start_epoch
        self.eval_every = eval_every
        self.n_eval_points = n_eval_points
        self.fidelity_weight = fidelity_weight
        self._current_epoch = 0
        self._energy_history: list[float] = []
        self._fidelity_history: list[float] = []

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch (called by training loop)."""
        self._current_epoch = epoch

    def should_eval_energy(self) -> bool:
        """Check if physics terms should be evaluated this epoch."""
        if self._current_epoch < self.start_epoch:
            return False
        return (self._current_epoch - self.start_epoch) % self.eval_every == 0

    # Alias for semantic clarity — same schedule governs both terms
    should_eval_physics = should_eval_energy

    @property
    def fidelity_enabled(self) -> bool:
        """Whether the fidelity term is active."""
        return self.fidelity_weight > 0.0

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        energy_errors: torch.Tensor | None = None,
        infidelities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute combined loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted theta, shape (batch, n_params).
        target : torch.Tensor
            Target theta, shape (batch, n_params).
        energy_errors : torch.Tensor | None
            |E(theta_pred) - E_exact| for selected points.
        infidelities : torch.Tensor | None
            (1 - F(theta_pred)) for selected points. Only used if
            fidelity_weight > 0. Points where fidelity is unavailable
            (N>22) should be omitted from this tensor by the caller.

        Returns
        -------
        torch.Tensor
            Combined loss value.
        """
        loss = self.mse(pred, target)

        if self._current_epoch < self.start_epoch:
            return loss

        if energy_errors is not None:
            energy_loss = torch.mean(energy_errors)
            self._energy_history.append(float(energy_loss.item()))
            loss = loss + self.weight * energy_loss

        if self.fidelity_enabled and infidelities is not None and len(infidelities) > 0:
            fidelity_loss = torch.mean(infidelities)
            self._fidelity_history.append(float(fidelity_loss.item()))
            loss = loss + self.fidelity_weight * fidelity_loss

        return loss

    @property
    def energy_history(self) -> list[float]:
        """History of energy loss values."""
        return self._energy_history

    @property
    def fidelity_history(self) -> list[float]:
        """History of infidelity (1 - F) loss values."""
        return self._fidelity_history


def evaluate_energy_batch(
    theta_batch: np.ndarray,
    hamiltonians: list,
    circuit,
    exact_energies: np.ndarray,
) -> np.ndarray:
    """Evaluate energy errors for a batch of predicted parameters.

    Parameters
    ----------
    theta_batch : np.ndarray
        Predicted parameters, shape (batch, n_params).
    hamiltonians : list
        List of Hamiltonians (one per batch element).
    circuit : QuantumCircuit
        Parameterized circuit.
    exact_energies : np.ndarray
        Exact ground state energies, shape (batch,).

    Returns
    -------
    np.ndarray
        |E(θ_pred) - E_exact| for each point, shape (batch,).
    """
    from qiskit.primitives import StatevectorEstimator

    estimator = StatevectorEstimator()
    errors = np.zeros(len(theta_batch))

    for i, (theta, H, e_exact) in enumerate(zip(theta_batch, hamiltonians, exact_energies, strict=False)):
        bound = circuit.assign_parameters(theta)
        job = estimator.run([(bound, H)])
        e_pred = float(job.result()[0].data.evs)
        errors[i] = abs(e_pred - e_exact)

    return errors


def evaluate_infidelity_batch(
    theta_batch: np.ndarray,
    ground_state_vectors: list[np.ndarray | None],
    circuit,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate infidelities (1 - F) for a batch of predicted parameters.

    F = |<psi_exact | psi(theta_pred)>|^2, where psi(theta_pred) is the
    statevector produced by the parametrized circuit bound to theta_pred.

    Points whose ground_state_vector is None (e.g. N>22, unavailable) are
    skipped and reported via the returned mask.

    Parameters
    ----------
    theta_batch : np.ndarray
        Predicted parameters, shape (batch, n_params).
    ground_state_vectors : list[np.ndarray | None]
        Exact ground-state vectors (one per batch element), or None where
        unavailable. Each vector has shape (2^N,).
    circuit : QuantumCircuit
        Parameterized circuit (same for all if fixed N, or per-point).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (infidelities, valid_mask) — infidelities for valid points only
        (shape (n_valid,)), and a boolean mask (shape (batch,)) indicating
        which batch points had a computable fidelity.
    """
    from qiskit.quantum_info import Statevector, state_fidelity

    infidelities: list[float] = []
    valid_mask = np.zeros(len(theta_batch), dtype=bool)

    for i, (theta, gs_vec) in enumerate(zip(theta_batch, ground_state_vectors, strict=False)):
        if gs_vec is None:
            continue
        bound = circuit.assign_parameters(theta)
        psi_pred = Statevector(bound)
        fidelity = float(state_fidelity(psi_pred, Statevector(gs_vec)))
        infidelities.append(1.0 - fidelity)
        valid_mask[i] = True

    return np.asarray(infidelities, dtype=float), valid_mask


def compute_ground_state_vectors(
    hamiltonians: list,
    n_qubits_list: list[int],
    *,
    max_n: int = 22,
) -> list[np.ndarray | None]:
    """Compute exact ground-state vectors for a batch (for fidelity loss).

    Reuses ClassicalSolver.ground_state_vector() which auto-selects dense
    eigh (N<=12) or sparse eigsh (13<=N<=22). Returns None for N>max_n where
    the 2^N statevector is infeasible.

    Parameters
    ----------
    hamiltonians : list
        SparsePauliOp Hamiltonians, one per training point.
    n_qubits_list : list[int]
        System size for each point.
    max_n : int
        Maximum N for which to compute a vector (default 22).

    Returns
    -------
    list[np.ndarray | None]
        Ground-state vectors, or None where N>max_n or computation failed.
    """
    from qmbp_simulation.solvers import ClassicalSolver

    solver = ClassicalSolver()
    vectors: list[np.ndarray | None] = []
    for H, n in zip(hamiltonians, n_qubits_list, strict=False):
        if n > max_n:
            vectors.append(None)
            continue
        try:
            vectors.append(solver.ground_state_vector(H, n_qubits=n))
        except (ValueError, MemoryError):
            vectors.append(None)
    return vectors


def select_eval_subset(
    n_total: int,
    n_eval: int,
    seed: int | None = None,
) -> list[int]:
    """Select random subset of training points for energy evaluation."""
    rng = np.random.default_rng(seed)
    n_eval = min(n_eval, n_total)
    return rng.choice(n_total, size=n_eval, replace=False).tolist()
