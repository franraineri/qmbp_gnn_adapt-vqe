"""Shared building blocks for the frustrated-HVA study runners.

The ``vl_vs_hva`` study grew several standalone argparse runners
(``run_basin_count``, ``run_n18_fair_convergence``, ``run_ansatz_variants``)
that each re-implemented the same three pieces:

- build the frustrated ground state + bond-resolved HVA circuit,
- run a two-segment L-BFGS-B "starvation probe" (half budget, then the rest) to
  tell a genuinely-converged minimum apart from a budget-starved one,
- run a multi-seed best-of-by-energy optimization.

This module factors those into pure, testable ``src`` functions so the runners
declare *what* to run, not *how*. The optimizer primitives (``_lbfgsb``) and the
block mask live in :mod:`qmbp_simulation.framework.study_runner`; this module
reuses them rather than duplicating the loop.

The functions intentionally take a pre-built ``x0`` (rather than seeding
internally) so each runner keeps its own RNG call pattern byte-for-byte — the
consolidation changes structure, not numerics.
"""

from __future__ import annotations

import numpy as np

from qmbp_simulation.framework.study_runner import _lbfgsb


def ground_state(topology, n, h, j2, p):
    """Frustrated ground state + bond-resolved HVA circuit for one point.

    Returns ``(lat, qc, H, psi, e0, gap, n_nn, n_nnn)`` where ``psi`` is the
    exact ground eigenvector, ``gap`` the spectral gap, and ``n_nn``/``n_nnn``
    the nearest/next-nearest bond counts of the lattice.
    """
    from scipy.sparse.linalg import eigsh

    from qmbp_simulation.circuits import HVACircuitBuilder
    from qmbp_simulation.models import make_lattice
    from qmbp_simulation.models.hamiltonian import HamiltonianBuilder
    from qmbp_simulation.models.model_registry import get_model_spec

    spec = get_model_spec("tfim_frustrated")
    hk = dict(getattr(spec, "hamiltonian_kwargs", {}))
    hk["J2"] = j2
    lat = make_lattice(topology, n, J=1.0, h=h)
    H = spec.build_hamiltonian(lat, **hk)
    ev, evec = eigsh(H.to_matrix(sparse=True), k=2, which="SA")
    order = np.argsort(ev)
    psi = evec[:, order[0]].astype(complex)
    e0 = float(ev[order[0]])
    gap = float(ev[order[1]] - ev[order[0]])
    qc, _ = HVACircuitBuilder().create_bond_resolved_frustrated(n, p, lat)
    n_nn = len(lat.edges)
    n_nnn = len(HamiltonianBuilder._generate_nnn_edges(lat))
    return lat, qc, H, psi, e0, gap, n_nn, n_nnn


def make_cost_fid(qc, H, psi, backend=None):
    """Return ``(cost, fid, grad, backend)`` for a circuit against target ``psi``.

    ``cost(theta)`` is ⟨H⟩ via the cached-Hamiltonian NoiselessBackend, ``fid``
    is the exact ground-state overlap, ``grad`` the adjoint gradient (jac).
    """
    from qiskit.quantum_info import Statevector

    from qmbp_simulation.execution import NoiselessBackend

    if backend is None:
        backend = NoiselessBackend()
    grad = backend.gradient(qc, H)

    def cost(x):
        return backend.evaluate(qc, H, x)

    def fid(x):
        ps = np.asarray(Statevector(qc.assign_parameters(x)).data)
        return float(abs(np.vdot(psi, ps)) ** 2)

    return cost, fid, grad, backend


def diagnose_starvation(
    converged: bool, late_gain: float, *, starved_label="starved", local_min_label="local_min", threshold=1e-3
) -> str:
    """Classify a capped optimization: converged / starved / genuine local min.

    ``late_gain`` is the energy still gained in the second budget segment. A
    capped run that kept descending (``late_gain > threshold``) is starved (needs
    more iterations), not a worse basin; a flat one is a genuine local minimum.
    """
    if converged:
        return "converged"
    if late_gain > threshold:
        return starved_label
    return local_min_label


def starvation_probe(cost, grad, x0, *, maxiter):
    """Two-segment L-BFGS-B: half the budget, then the remainder from that point.

    Returns ``(x_fin, e_fin, total_nit, e_half, late_gain, converged)``.
    ``converged`` means the second segment settled before its cap; ``late_gain``
    is the energy dropped in the second half (the starvation signal).
    """
    x0 = np.asarray(x0, float)
    half = max(maxiter // 2, 1)
    x1, e_half, nit1 = _lbfgsb(cost, x0, maxiter=half, grad=grad)
    rest = maxiter - half
    x_fin, e_fin, nit2 = _lbfgsb(cost, x1, maxiter=rest, grad=grad)
    total_nit = int(nit1) + int(nit2)
    converged = int(nit2) < rest
    late_gain = float(e_half) - float(e_fin)
    return x_fin, float(e_fin), total_nit, float(e_half), late_gain, converged


def optimize_bestof(cost, fid, grad, npar, *, restarts, maxiter, seed0, warm_theta=None, on_restart=None):
    """Multi-seed best-of to genuine convergence. Returns ``(best_fid, best_e, runs)``.

    Restart 0 uses ``warm_theta`` (clipped to the box) when it matches ``npar``,
    else the θ=0 seed; the rest are random in-box draws from
    ``np.random.default_rng(seed0)``. Selection is by ENERGY (validated monotone
    with fidelity for this ansatz family). Mirrors the per-restart record shape
    the runners persist.

    ``on_restart(runs, best)`` — optional callback fired AFTER every restart with
    the accumulated ``runs`` list and the current best dict
    ``{"fidelity", "energy", "restart"}``. This is the base per-restart
    persistence hook: any runner can pass a persister so a crash/interrupt at
    large N never loses completed restarts (the expensive part). Exceptions in
    the callback are swallowed (persistence must never break the optimization).
    """
    if restarts < 1:
        raise ValueError(f"restarts must be >= 1, got {restarts}")

    warm0 = None
    if warm_theta is not None and len(warm_theta) == npar:
        warm0 = np.clip(np.asarray(warm_theta, float), -np.pi, np.pi)

    rng = np.random.default_rng(seed0)
    best_e, best_fid, best_r, runs = None, None, None, []
    for r in range(restarts):
        if r == 0:
            x0 = warm0 if warm0 is not None else np.zeros(npar)
        else:
            x0 = rng.uniform(-np.pi, np.pi, npar)
        x, e, nit = _lbfgsb(cost, x0, maxiter=maxiter, grad=grad)
        f = fid(x)  # computed once per restart; reused as the return value
        runs.append(
            {
                "restart": r,
                "energy": e,
                "fidelity": f,
                "nit": nit,
                "converged": nit < maxiter,
                "theta_init": np.asarray(x0, float).tolist(),
                "theta_final": np.asarray(x, float).tolist(),
            }
        )
        if best_e is None or e < best_e:
            best_e, best_fid, best_r = e, f, r
        if on_restart is not None:
            try:
                on_restart(runs, {"fidelity": best_fid, "energy": best_e, "restart": best_r})
            except Exception:
                pass  # persistence must never break the optimization
    return best_fid, best_e, runs


def cx_and_params(qc):
    """Real transpiled 2q count + logical params, via the canonical summarizer.

    Binds NON-ZERO angles before transpiling: zero angles make every RZZ the
    identity, which the transpiler cancels (yielding a spurious ``n_2q=0``).
    """
    from qmbp_simulation.analysis.circuit_visualizer import circuit_summary

    logical = circuit_summary(qc)
    try:
        from qiskit import transpile

        angles = np.full(qc.num_parameters, 0.37)
        t = transpile(qc.assign_parameters(angles), basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
        n_2q = circuit_summary(t)["n_2q_gates"]
    except Exception:
        n_2q = logical["n_2q_gates"]
    return n_2q, int(logical["n_parameters"])
