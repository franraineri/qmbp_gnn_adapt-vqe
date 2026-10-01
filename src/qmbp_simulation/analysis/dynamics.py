"""Shared quench-dynamics helpers: MPS truncation, classical crossover, t_noise.

Single source of truth for the dynamics primitives that the quench runners
(``run_prune03_dynamics``, ``run_kitaev_heavyhex_noise``, ``run_prep_evolution_noise``)
each re-implemented verbatim:

- ``truncate_mps`` — statevector → χ-bounded MPS → statevector (sequential SVD).
  Identical to ``ValidationRunner.truncate_statevector_mps``; kept here so the
  dynamics scripts do not import the 7k-line runner_base god-node.
- ``classical_crossover`` — Trotter step t* where a χ-MPS first loses the exact
  evolved state (infidelity > threshold). The classical-simulability frontier.
- ``evolve_states`` — exact unitary time series under a dense Hamiltonian.
- ``correlator_zz`` — ⟨Z_i Z_j⟩ in the computational basis (vectorized).
- ``tnoise_curve`` — transpile prep + k Trotter steps per step to a (fake)
  backend, score with ``compute_error_budget`` (layout-aware), return the
  fidelity curve and the step where it drops below the 1/e floor.

The t_noise helpers are layout-aware: the chip-wide error mean on fake backends
is poisoned by dead edges (error=1.0), so the actual transpiled layout is passed
to ``compute_error_budget``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit

# 1/e fidelity floor: the Trotter step where the predicted circuit fidelity
# on real hardware first drops below this is "t_noise".
NOISE_FLOOR = float(np.exp(-1.0))

# Median two-qubit (CZ) gate error of IBM Nighthawk r2 ("Heron-class" fidelity).
# Derived from the hardware-optimization benchmark (arXiv:2607.11637): the onset
# of noise-dominated execution at ~770 2q gates for F≈0.1 implies
# ε₂q = −ln(0.1)/770 ≈ 3e-3. Cross-validated by the Nighthawk r2 RCS paper
# (arXiv:2609.28657, F_XEB≈2.3e-3). See
# internal/documentation/hardware/nighthawk_r2_gate_limits_and_noise.md.
NIGHTHAWK_CZ_ERROR = 3e-3


def fidelity_from_2q(n_2q: int, eps_2q: float = NIGHTHAWK_CZ_ERROR) -> float:
    """Exponential-decay circuit fidelity from a 2q-gate count: exp(−N₂q·ε₂q).

    The transparent, reproducible alternative to a Qiskit fake backend: a single
    tunable knob ``eps_2q`` (default = Nighthawk r2 median CZ error). A future
    noise-suppression model enters by lowering ``eps_2q`` (or the 2q count).
    """
    return float(np.exp(-max(0, int(n_2q)) * eps_2q))


def tnoise_from_2q_curve(
    cumulative_2q: Sequence[int],
    *,
    eps_2q: float = NIGHTHAWK_CZ_ERROR,
    floor: float | None = None,
) -> dict:
    """t_noise from a cumulative 2q-gate count per Trotter step (analytic model).

    ``cumulative_2q[k]`` is the total 2q-gate count of prep + k Trotter steps.
    Returns the fidelity curve ``exp(−N₂q·ε₂q)`` and the first step where it drops
    below ``floor`` (default 1/e) — the realistic ``t_noise`` without any fake
    backend. ``eps_2q`` is the median CZ error (lower it to model mitigation).
    """
    if floor is None:
        floor = NOISE_FLOOR
    fidelity_curve = [fidelity_from_2q(n, eps_2q) for n in cumulative_2q]
    t_noise = next((s for s, f in enumerate(fidelity_curve) if f < floor), None)
    return {
        "fidelity_curve": fidelity_curve,
        "cz_cumulative": [int(n) for n in cumulative_2q],
        "t_noise_step": t_noise,
        "eps_2q": eps_2q,
        "floor": floor,
        "source": "analytic_exp_decay",
    }


def gates_to_floor(eps_2q: float = NIGHTHAWK_CZ_ERROR, *, floor: float | None = None) -> int:
    """Number of 2q gates at which fidelity reaches ``floor`` (default 1/e).

    For the 1/e floor this is simply ``round(1/eps_2q)`` (333 at the median CZ
    error). Useful as a quick budget: a circuit with fewer 2q gates stays above
    the floor.
    """
    if floor is None:
        floor = NOISE_FLOOR
    return int(round(-np.log(floor) / eps_2q))


def truncate_mps(psi: np.ndarray, n_qubits: int, chi_max: int) -> np.ndarray:
    """Truncate a statevector to bond dimension ``chi_max`` via sequential SVD.

    Converts |ψ⟩ to an MPS, caps each bond to ``chi_max`` singular values, and
    reconstructs the (normalized) truncated statevector. Low χ mimics hardware
    decoherence: both destroy long-range correlations.
    """
    state = np.asarray(psi).reshape(-1)
    remaining = state.reshape(1, -1)
    tensors = []
    for _ in range(n_qubits - 1):
        chi_left = remaining.shape[0]
        mat = remaining.reshape(chi_left * 2, -1)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        chi_new = min(len(s), chi_max)
        u, s, vh = u[:, :chi_new], s[:chi_new], vh[:chi_new, :]
        norm = np.linalg.norm(s)
        if norm > 1e-15:
            s = s / norm
        tensors.append(u.reshape(chi_left, 2, chi_new))
        remaining = np.diag(s) @ vh
    tensors.append(remaining.reshape(remaining.shape[0], 2, 1))

    out = tensors[0]
    for t in tensors[1:]:
        out = np.einsum("ijk,klm->ijlm", out, t)
        out = out.reshape(out.shape[0], out.shape[1] * t.shape[1], t.shape[2])
    out = out.reshape(-1)
    norm = np.linalg.norm(out)
    return out / norm if norm > 1e-15 else out


def evolve_states(
    psi0: np.ndarray,
    n_steps: int,
    *,
    propagator: np.ndarray | None = None,
    evolver: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Return the exact time series ``[ψ(0), ψ(dt), …, ψ(n·dt)]``.

    Provide exactly one evolution rule:

    - ``propagator`` — a dense single-step unitary ``U = exp(-i H dt)``; applied
      as ``U @ ψ`` (good for N ≤ ~16 where the dense matrix fits).
    - ``evolver`` — a callable ``ψ → ψ_next`` (e.g. a ``scipy.sparse`` action via
      ``expm_multiply``); use for N up to ~22 where the dense unitary is too big.

    Each state is renormalized to suppress accumulated floating-point drift.
    """
    if (propagator is None) == (evolver is None):
        raise ValueError("Provide exactly one of 'propagator' or 'evolver'.")
    step_fn = evolver if evolver is not None else (lambda p: propagator @ p)
    psi = np.asarray(psi0, dtype=complex).copy()
    states = [psi.copy()]
    for _ in range(n_steps):
        psi = step_fn(psi)
        psi /= np.linalg.norm(psi)
        states.append(psi.copy())
    return states


def _crossover_from_states(
    states: Sequence[np.ndarray],
    n_qubits: int,
    chi_values: Sequence[int],
    infidelity_threshold: float,
) -> dict[int, int | None]:
    """First step where the χ-truncation infidelity exceeds the threshold."""
    out: dict[int, int | None] = {}
    for chi in chi_values:
        t_star = None
        for step, exact in enumerate(states):
            trunc = truncate_mps(exact, n_qubits, chi)
            if 1.0 - abs(np.vdot(exact, trunc)) ** 2 > infidelity_threshold:
                t_star = step
                break
        out[int(chi)] = t_star
    return out


def classical_crossover(
    psi0: np.ndarray,
    hamiltonian: np.ndarray,
    dt: float,
    n_steps: int,
    n_qubits: int,
    chi_values: Sequence[int],
    *,
    infidelity_threshold: float = 0.01,
    propagator: np.ndarray | None = None,
) -> dict[int, int | None]:
    """Trotter step t* where a χ-MPS first loses the DENSE exact evolved state.

    Evolves ``psi0`` exactly under ``exp(-i H dt)`` (dense) for ``n_steps``; at
    each step truncates the exact state to each χ in ``chi_values`` and records
    the first step whose infidelity ``1 − |⟨exact|χ⟩|²`` exceeds the threshold.

    For N beyond the dense reach, use :func:`classical_crossover_sparse`.
    Returns ``{chi: t*}`` with ``t* = None`` when χ tracks the whole window.
    """
    from scipy.linalg import expm

    if propagator is None:
        propagator = expm(-1j * np.asarray(hamiltonian) * dt)
    states = evolve_states(psi0, n_steps, propagator=propagator)
    return _crossover_from_states(states, n_qubits, chi_values, infidelity_threshold)


def classical_crossover_sparse(
    psi0: np.ndarray,
    hamiltonian_sparse,
    dt: float,
    n_steps: int,
    n_qubits: int,
    chi_values: Sequence[int],
    *,
    infidelity_threshold: float = 0.01,
) -> dict[int, int | None]:
    """Like :func:`classical_crossover` but evolves via sparse ``expm_multiply``.

    ``hamiltonian_sparse`` is a scipy sparse matrix (``H.to_matrix(sparse=True)``).
    Avoids materializing the 2ᴺ×2ᴺ dense unitary, so it reaches N ≈ 18–22 where
    the dense path is infeasible.
    """
    from scipy.sparse.linalg import expm_multiply

    generator = -1j * hamiltonian_sparse * dt
    states = evolve_states(psi0, n_steps, evolver=lambda p: expm_multiply(generator, p))
    return _crossover_from_states(states, n_qubits, chi_values, infidelity_threshold)


def count_2q_gates(qc: QuantumCircuit) -> int:
    """Logical 2-qubit gate count of a circuit (pre-transpile, as authored).

    Distinct from ``cx_and_params`` (which transpiles) and ``BondSelection.n_2q``
    (a theoretical 2·bonds estimate): this counts the 2q operations actually
    present in the circuit object.
    """
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 2)


def correlator_zz(psi: np.ndarray, i: int, j: int, n_qubits: int) -> float:
    """⟨Z_i Z_j⟩ in the computational basis (vectorized over basis states)."""
    dim = 2**n_qubits
    probs = np.abs(psi) ** 2
    k = np.arange(dim, dtype=np.int64)
    zi = 1.0 - 2.0 * ((k >> i) & 1)
    zj = 1.0 - 2.0 * ((k >> j) & 1)
    return float(np.dot(probs, zi * zj))


def tnoise_curve(
    prep_circuit: QuantumCircuit,
    trotter_step: QuantumCircuit,
    n_steps: int,
    backend,
    *,
    floor: float | None = None,
    eps_2q: float | None = None,
) -> dict:
    """Fidelity curve of prep + k Trotter steps transpiled per step to ``backend``.

    Transpiles ``prep + k·trotter_step`` at each k and reports the step where the
    estimated fidelity first drops below ``floor`` (default 1/e). Two noise models:

    - ``eps_2q is None`` (default): score with ``compute_error_budget`` using the
      real transpiled layout (fake-backend calibration; dead-edge rates handled
      via the layout).
    - ``eps_2q`` given: the ANALYTIC model ``exp(−N₂q·ε₂q)`` on the transpiled 2q
      count. ``backend`` is still used only to transpile (realistic routing/2q
      count); the per-gate error comes from ``eps_2q`` (e.g. NIGHTHAWK_CZ_ERROR).
      Transparent and reproducible; lower ``eps_2q`` to model noise suppression.

    When ``backend`` is ``None`` the budget falls back to typical gate rates.
    Stops early once the floor is crossed (the curve is monotone in k).
    """
    from qiskit import transpile

    from qmbp_simulation.analysis.circuit_visualizer import compute_error_budget

    if floor is None:
        floor = NOISE_FLOOR
    fidelity_curve: list[float] = []
    cz_cumulative: list[int] = []
    budget_curve: list[float] = []
    source = "typical_fallback"
    full = prep_circuit.copy()
    early = False
    for step in range(n_steps + 1):
        if backend is not None:
            try:
                tqc = transpile(full, backend=backend, optimization_level=3, seed_transpiler=42)
                layout = tqc.layout.final_index_layout() if tqc.layout else None
                budget = compute_error_budget(tqc, backend=backend, layout=layout)
            except Exception:  # noqa: BLE001
                budget = compute_error_budget(full, backend=None)
        else:
            budget = compute_error_budget(full, backend=None)
        n_2q = int(budget["n_2q_gates"])
        if eps_2q is not None:
            # Analytic model on the transpiled 2q count (realistic routing kept).
            fidelity = fidelity_from_2q(n_2q, eps_2q)
            source = "analytic_exp_decay"
            error_budget = n_2q * eps_2q
        else:
            fidelity = float(budget["fidelity_estimate"])
            source = budget["source"]
            error_budget = float(budget["error_budget"])
        fidelity_curve.append(fidelity)
        cz_cumulative.append(n_2q)
        budget_curve.append(error_budget)
        if fidelity < floor:
            early = True
            break
        if step < n_steps:
            full = full.compose(trotter_step)
    t_noise = next((s for s, f in enumerate(fidelity_curve) if f < floor), None)
    return {
        "fidelity_curve": fidelity_curve,
        "cz_cumulative": cz_cumulative,
        "error_budget_curve": budget_curve,
        "t_noise_step": t_noise,
        "early_stopped": early,
        "source": source,
        "eps_2q": eps_2q,
        "prep_2q": int(cz_cumulative[0]) if cz_cumulative else None,
    }


def resolve_fake_backend(name: str):
    """Return a named IBM fake backend instance, or ``None`` if unavailable.

    Known names: ``"nighthawk"`` (FakeNighthawk, square degree-4) and
    ``"heron"`` (FakeTorino, heavy-hex). Unknown names return ``None``.
    """
    try:
        if name == "nighthawk":
            from qiskit_ibm_runtime.fake_provider import FakeNighthawk

            return FakeNighthawk()
        if name == "heron":
            from qiskit_ibm_runtime.fake_provider import FakeTorino

            return FakeTorino()
    except Exception:  # noqa: BLE001
        return None
    return None


def tnoise_on_backend(
    prep_circuit: QuantumCircuit,
    trotter_step: QuantumCircuit,
    n_steps: int,
    backend_name: str,
    *,
    floor: float | None = None,
    eps_2q: float | None = None,
) -> dict | None:
    """``tnoise_curve`` against a named fake backend; ``None`` if unresolvable.

    ``backend_name`` selects the backend used for transpilation (realistic 2q
    count/routing). With ``eps_2q`` set, the fidelity uses the analytic model
    instead of the fake-backend calibration (see :func:`tnoise_curve`).
    """
    backend = resolve_fake_backend(backend_name)
    if backend is None:
        return None
    curve = tnoise_curve(prep_circuit, trotter_step, n_steps, backend,
                         floor=floor, eps_2q=eps_2q)
    return {
        "backend": backend_name,
        "noise_model": "analytic" if eps_2q is not None else "fake_calibration",
        "eps_2q": eps_2q,
        "fidelity_curve": curve["fidelity_curve"],
        "cz_cumulative": curve["cz_cumulative"],
        "t_noise_step": curve["t_noise_step"],
    }
