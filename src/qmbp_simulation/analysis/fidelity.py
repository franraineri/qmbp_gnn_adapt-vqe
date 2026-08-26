"""State fidelity estimation at any system size.

Single source of truth for ground-state fidelity used by runners and
standalone experiment scripts. Two regimes:

- N ≤ STATEVECTOR_MAX_N: exact fidelity F = |⟨E₀|ψ⟩|² via dense statevector.
- N > STATEVECTOR_MAX_N: rigorous variance-based lower bound (Eckart /
  Weinstein–Temple inequality):

      F  ≥  1 − Var(H) / gap²          Var(H) = ⟨H²⟩ − ⟨H⟩²

  The bound uses only expectation values of the variational state, so it is
  independent of basis/qubit-ordering conventions and computable at any N via
  the MPS backend's ``compute_energy_variance`` (``save_expectation_value``).

Unlike ``analysis.metrics`` (pure, no heavy deps), this module imports Qiskit
and the MPS backend lazily inside functions, so the top-level import stays
cheap and the "no heavy imports" contract of metrics.py is preserved.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_exact_fidelity(circuit, theta: np.ndarray, exact_state: np.ndarray) -> float | None:
    """Exact fidelity |⟨exact|ψ(θ)⟩|² via statevector. None on error."""
    try:
        from qiskit.quantum_info import Statevector, state_fidelity

        bound = circuit.assign_parameters(theta)
        fid = float(state_fidelity(Statevector(bound), Statevector(exact_state)))
        if not np.isfinite(fid):
            return None
        return float(np.clip(fid, 0.0, 1.0))
    except (MemoryError, ValueError, AttributeError) as exc:
        logger.debug("compute_exact_fidelity failed: %s", exc)
        return None


def compute_variance_fidelity_bound(
    circuit,
    theta: np.ndarray,
    hamiltonian,
    gap: float,
    *,
    chi_max: int = 64,
) -> dict | None:
    """Eckart lower bound F ≥ 1 − Var(H)/gap² for large N.

    Computes Var(H) via a deterministic MPS backend, which works at any N
    (``save_expectation_value`` on H and H²). Returns None if the variance is
    not computable or the gap is non-positive.

    Returns
    -------
    dict | None
        Keys: ``fidelity`` (lower bound in [0,1]), ``method`` ("variance_bound"),
        ``is_lower_bound`` (True), ``energy_variance`` (Var(H)).
    """
    if gap is None or gap <= 1e-9:
        return None
    try:
        from qmbp_simulation.execution import MPSBackend

        backend = MPSBackend(strategy="aer_mps", chi_max=chi_max, deterministic=True)
        var_h = backend.compute_energy_variance(circuit, hamiltonian, np.asarray(theta))
        if var_h is None or not np.isfinite(var_h):
            return None
        fid_lb = float(np.clip(1.0 - var_h / (gap * gap), 0.0, 1.0))
        return {
            "fidelity": fid_lb,
            "method": "variance_bound",
            "is_lower_bound": True,
            "energy_variance": float(var_h),
        }
    except (MemoryError, ValueError, AttributeError, KeyError) as exc:
        logger.debug("compute_variance_fidelity_bound failed: %s", exc)
        return None


def estimate_fidelity_from_primitives(
    circuit,
    theta: np.ndarray,
    hamiltonian,
    gap: float,
    n_qubits: int,
    *,
    exact_state: np.ndarray | None = None,
    chi_max: int = 64,
) -> dict:
    """Best-available fidelity from already-built primitives, at any N.

    Dispatches to exact statevector fidelity (N ≤ STATEVECTOR_MAX_N, requires
    ``exact_state``) or the variance-based lower bound (larger N). Always
    returns a dict (never raises) with provenance metadata so callers can
    persist it directly.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized circuit (unbound).
    theta : np.ndarray
        Parameter values.
    hamiltonian : SparsePauliOp
        Hamiltonian H (for the variance bound).
    gap : float
        Spectral gap E₁ − E₀ (for the variance bound).
    n_qubits : int
        System size (selects exact vs bound).
    exact_state : np.ndarray | None
        Exact ground-state vector. Required for the exact path; if None at
        small N, falls through to the variance bound.
    chi_max : int
        MPS bond dimension for the variance computation.

    Returns
    -------
    dict
        ``fidelity`` : float | None
        ``method`` : "exact" | "variance_bound" | "unavailable"
        ``is_lower_bound`` : bool
        ``energy_variance`` : float | None
    """
    from qmbp_simulation.models.constants import STATEVECTOR_MAX_N

    if n_qubits <= STATEVECTOR_MAX_N and exact_state is not None:
        fid = compute_exact_fidelity(circuit, theta, exact_state)
        if fid is not None:
            return {
                "fidelity": fid,
                "method": "exact",
                "is_lower_bound": False,
                "energy_variance": None,
            }

    bound = compute_variance_fidelity_bound(
        circuit, theta, hamiltonian, gap, chi_max=chi_max
    )
    if bound is not None:
        return bound
    return {
        "fidelity": None,
        "method": "unavailable",
        "is_lower_bound": False,
        "energy_variance": None,
    }
