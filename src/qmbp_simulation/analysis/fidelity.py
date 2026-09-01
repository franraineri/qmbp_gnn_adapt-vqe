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

Caveat — gap accuracy near criticality (read before trusting large-N bounds).
  Both the variance bound (∝ 1/gap²) and the energy-gap bound (∝ 1/gap) are
  only as reliable as the spectral gap fed to them. For N above the exact-
  diagonalization limit the gap comes from DMRG, and when the excited-state
  solve does not converge the solver falls back to an analytical estimate with
  a finite-size FLOOR of 2π/N (see ``solvers.classical``). Near h_c the true
  gap collapses below that floor, so the floor OVERESTIMATES the gap → Var/gap²
  is underestimated → the fidelity bound reads optimistically HIGH in exactly
  the critical region where it matters most. Treat large-N fidelity near h_c
  as an upper-optimistic estimate, and prefer the exact path (N ≤
  STATEVECTOR_MAX_N) or a converged DMRG gap for definitive statements.

Unlike ``analysis.metrics`` (pure, no heavy deps), this module imports Qiskit
and the MPS backend lazily inside functions, so the top-level import stays
cheap and the "no heavy imports" contract of metrics.py is preserved.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_exact_fidelity(
    circuit,
    theta: np.ndarray,
    exact_state: np.ndarray,
    *,
    cache_ctx: dict | None = None,
) -> float | None:
    """Exact fidelity |⟨exact|ψ(θ)⟩|² via statevector. None on error.

    Persistent caching (opt-in). When ``cache_ctx`` is provided it must carry
    ``topology``, ``n_qubits`` and ``h`` (optionally ``model``, ``p_layers``),
    and the result is looked up / stored in the shared ``EvalCache`` keyed by
    (topology, N, h, model, p_layers, θ-hash). This lets any caller — runners
    via ``ValidationRunner.safe_compute_fidelity`` and offline backfillers —
    reuse a fidelity already computed for the same (model, N, h, θ) instead of
    re-diagonalizing. Without ``cache_ctx`` the behavior is unchanged (no cache).
    """
    cache = None
    if cache_ctx is not None and all(k in cache_ctx for k in ("topology", "n_qubits", "h")):
        try:
            from qmbp_simulation.execution.eval_cache import EvalCache

            cache = EvalCache()
            hit = cache.get_fidelity(
                cache_ctx["topology"], int(cache_ctx["n_qubits"]), float(cache_ctx["h"]),
                np.asarray(theta),
                model=cache_ctx.get("model", "tfim"),
                p_layers=int(cache_ctx.get("p_layers", 0)),
            )
            if hit is not None:
                return hit
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            logger.debug("compute_exact_fidelity: cache lookup skipped: %s", exc)
            cache = None

    try:
        from qiskit.quantum_info import Statevector, state_fidelity

        bound = circuit.assign_parameters(theta)
        fid = float(state_fidelity(Statevector(bound), Statevector(exact_state)))
        if not np.isfinite(fid):
            return None
        fid = float(np.clip(fid, 0.0, 1.0))
    except (MemoryError, ValueError, AttributeError) as exc:
        logger.debug("compute_exact_fidelity failed: %s", exc)
        return None

    if cache is not None:
        try:
            cache.put_fidelity(
                cache_ctx["topology"], int(cache_ctx["n_qubits"]), float(cache_ctx["h"]),
                np.asarray(theta), fid,
                model=cache_ctx.get("model", "tfim"),
                p_layers=int(cache_ctx.get("p_layers", 0)),
            )
            cache.flush()
        except Exception as exc:  # noqa: BLE001 - never fail on cache write
            logger.debug("compute_exact_fidelity: cache store skipped: %s", exc)

    return fid


def energy_gap_fidelity_bound(
    e_pred: float,
    e_exact: float,
    gap: float,
) -> dict | None:
    de = float(e_pred) - float(e_exact)
    if de < 0:
        de = 0.0
    if not (np.isfinite(de) and np.isfinite(gap)) or gap <= 1e-10:
        return None
    if de >= 0.5 * gap:
        return {
            "fidelity": None,
            "method": "energy_gap_bound_invalid",
            "de_gap": de / gap,
        }
    f_bound = float(np.clip(1.0 - de / gap, 0.0, 1.0))
    return {
        "fidelity": f_bound,
        "method": "energy_gap_bound",
        "de_gap": de / gap,
    }


def compute_variance_fidelity_bound(
    circuit,
    theta: np.ndarray,
    hamiltonian,
    gap: float,
    *,
    chi_max: int = 64,
    e_pred: float | None = None,
    e0: float | None = None,
) -> dict | None:
    """Eckart / Weinstein–Temple lower bound F ≥ 1 − Var(H)/gap² for large N.

    Computes Var(H) via a deterministic MPS backend, which works at any N
    (``save_expectation_value`` on H and H²). Returns None if the variance is
    not computable or the gap is non-positive.

    Validity condition (important). The inequality only bounds the overlap with
    the GROUND state when the variational state lies closer to E₀ than to the
    first excited level E₁ — the standard Temple/Eckart requirement
    E_pred < (E₀ + E₁)/2. Outside that window a small Var(H) can reflect
    proximity to an EXCITED eigenstate, so ``1 − Var/gap²`` would falsely
    certify high ground-state fidelity. When ``e_pred`` and ``e0`` are supplied
    and this condition fails, the bound is flagged invalid: ``fidelity`` is None
    and ``method`` is ``"variance_bound_invalid"`` (Var(H) still returned for
    diagnostics). If either is None the check is skipped (backward compatible).

    Returns
    -------
    dict | None
        Keys: ``fidelity`` (lower bound in [0,1] | None), ``method``
        ("variance_bound" | "variance_bound_invalid"), ``is_lower_bound`` (True),
        ``energy_variance`` (Var(H)).
    """
    if gap is None or gap <= 1e-9:
        return None
    try:
        from qmbp_simulation.execution import MPSBackend

        backend = MPSBackend(strategy="aer_mps", chi_max=chi_max, deterministic=True)
        var_h = backend.compute_energy_variance(circuit, hamiltonian, np.asarray(theta))
        if var_h is None or not np.isfinite(var_h):
            return None

        # Temple/Eckart validity: the state must sit below the midpoint between
        # E₀ and E₁, else the bound does not certify GROUND-state fidelity.
        if e_pred is not None and e0 is not None:
            e1 = e0 + gap
            midpoint = 0.5 * (e0 + e1)
            if e_pred >= midpoint:
                return {
                    "fidelity": None,
                    "method": "variance_bound_invalid",
                    "is_lower_bound": True,
                    "energy_variance": float(var_h),
                }

        raw_lb = 1.0 - var_h / (gap * gap)
        # When Var(H)/gap² ≥ 1 the Eckart bound is non-positive: it collapses to
        # the trivial statement F ≥ 0, which is true for any state and carries no
        # information. Report fidelity=None (→ "N/A") instead of a noisy "≥0.0000";
        # Var(H) is still returned so the dirty_state diagnostic stays available.
        fidelity = float(np.clip(raw_lb, 0.0, 1.0)) if raw_lb > 0.0 else None
        return {
            "fidelity": fidelity,
            "method": "variance_bound",
            "is_lower_bound": True,
            "energy_variance": float(var_h),
        }
    except (MemoryError, ValueError, AttributeError, KeyError) as exc:
        logger.debug("compute_variance_fidelity_bound failed: %s", exc)
        return None


def classify_infidelity_factor(
    energy_variance: float | None,
    gap: float | None,
) -> dict:
    """Decompose ground-state infidelity into its dominant physical factor.

    Uses the Eckart inequality F ≥ 1 − Var(H)/gap² to attribute the infidelity
    of a variational state to one of two causes, from Var(H) and the gap alone
    (no state reconstruction needed — cheap enough to run on every point):

    - ``dirty_state``: Var(H) is large ⇒ |ψ⟩ is far from an eigenstate. The
      infidelity is a preparation/optimization problem — ATTACKABLE (more
      restarts, warm-start, or a variance-penalized objective).
    - ``small_gap``: Var(H) is small but the gap is small (near criticality) ⇒
      the Eckart term Var/gap² is inflated by the vanishing denominator. This
      is a PHYSICS ceiling, not an optimization failure.
    - ``clean``: both fine ⇒ no meaningful infidelity from this diagnostic.
    - ``unknown``: inputs unavailable.

    This is a DIAGNOSTIC only. It never gates pass/fail (see
    ``metrics.is_point_failure``, which uses the dual energy criterion).

    Parameters
    ----------
    energy_variance : float | None
        Var(H) = ⟨H²⟩ − ⟨H⟩² of the variational state.
    gap : float | None
        Spectral gap Δ = E₁ − E₀.

    Returns
    -------
    dict
        Keys:
        - ``infidelity_dominant_factor`` : str
          ("dirty_state" | "small_gap" | "clean" | "unknown")
        - ``variance_over_gap2`` : float | None  (Var(H)/gap², the Eckart term)
        - ``energy_variance`` : float | None  (echoed for convenience)
        - ``gap`` : float | None  (echoed for convenience)
    """
    from qmbp_simulation.analysis.constants import (
        DIRTY_STATE_VARIANCE_THRESHOLD,
        SMALL_GAP_THRESHOLD,
    )

    out = {
        "infidelity_dominant_factor": "unknown",
        "variance_over_gap2": None,
        "energy_variance": (float(energy_variance) if energy_variance is not None else None),
        "gap": (float(gap) if gap is not None else None),
    }
    if energy_variance is None or gap is None:
        return out
    if not np.isfinite(energy_variance) or not np.isfinite(gap) or gap <= 1e-9:
        return out

    var_over_gap2 = float(energy_variance) / (float(gap) ** 2)
    out["variance_over_gap2"] = var_over_gap2

    dirty = energy_variance > DIRTY_STATE_VARIANCE_THRESHOLD
    small_gap = gap < SMALL_GAP_THRESHOLD

    if dirty:
        # A dirty state is attackable regardless of gap; report it as the
        # actionable factor. If the gap is also small, the Eckart term is
        # doubly inflated but the state is still the thing we can fix.
        out["infidelity_dominant_factor"] = "dirty_state"
    elif small_gap:
        out["infidelity_dominant_factor"] = "small_gap"
    else:
        out["infidelity_dominant_factor"] = "clean"
    return out


def compute_fidelity_decomposition(
    circuit,
    theta: np.ndarray,
    hamiltonian,
    gap: float,
    *,
    chi_max: int = 64,
) -> dict:
    """Full infidelity decomposition from primitives (computes Var(H) itself).

    Convenience wrapper that computes Var(H) via ``compute_variance_fidelity_bound``
    (single source of truth for the variance) and then classifies the dominant
    infidelity factor with ``classify_infidelity_factor``. Use this when Var(H)
    has NOT been computed yet; if you already have Var(H) and gap, call
    ``classify_infidelity_factor`` directly (no recomputation).

    Returns
    -------
    dict
        Keys from ``classify_infidelity_factor`` plus ``fidelity_bound`` (the
        Eckart lower bound F ≥ 1 − Var(H)/gap², clipped to [0,1]) when
        computable, else None.
    """
    bound = compute_variance_fidelity_bound(circuit, theta, hamiltonian, gap, chi_max=chi_max)
    if bound is None:
        result = classify_infidelity_factor(None, gap)
        result["fidelity_bound"] = None
        return result
    result = classify_infidelity_factor(bound["energy_variance"], gap)
    result["fidelity_bound"] = bound["fidelity"]
    return result


def estimate_fidelity_from_primitives(
    circuit,
    theta: np.ndarray,
    hamiltonian,
    gap: float,
    n_qubits: int,
    *,
    exact_state: np.ndarray | None = None,
    chi_max: int = 64,
    cache_ctx: dict | None = None,
    e_pred: float | None = None,
    e0: float | None = None,
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
    cache_ctx : dict | None
        Optional cache context ({topology, n_qubits, h, model, p_layers}) passed
        through to ``compute_exact_fidelity`` so the exact-path result is stored
        in / reused from the shared EvalCache. No effect on the bound path.
    e_pred, e0 : float | None
        Variational energy ⟨H⟩ and exact ground energy E₀. When both are given,
        the variance-bound path enforces the Temple/Eckart validity condition
        (E_pred < (E₀+E₁)/2); otherwise the check is skipped. No effect on the
        exact path.

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
        fid = compute_exact_fidelity(circuit, theta, exact_state, cache_ctx=cache_ctx)
        if fid is not None:
            return {
                "fidelity": fid,
                "method": "exact",
                "is_lower_bound": False,
                "energy_variance": None,
            }

    bound = compute_variance_fidelity_bound(
        circuit, theta, hamiltonian, gap, chi_max=chi_max, e_pred=e_pred, e0=e0
    )
    if bound is not None:
        return bound
    return {
        "fidelity": None,
        "method": "unavailable",
        "is_lower_bound": False,
        "energy_variance": None,
    }
