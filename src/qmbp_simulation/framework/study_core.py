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

    # Single-slot statevector reuse: cost(x) and fid(x) at the SAME x share one
    # Statevector build (the expensive part). L-BFGS-B and the candidate ranking
    # evaluate cost then fid at identical θ; caching the last state halves that
    # work with no call-site change. Keyed by params bytes (exact match only).
    _cache: dict = {"key": None, "psi": None}

    def _state(x):
        xb = np.asarray(x, float)
        key = xb.tobytes()
        if _cache["key"] != key:
            _cache["psi"] = np.asarray(Statevector(qc.assign_parameters(xb)).data)
            _cache["key"] = key
        return _cache["psi"]

    hmat = getattr(backend, "_hmat", None)

    def cost(x):
        if hmat is not None:
            ps = _state(x)
            return float(np.real(np.vdot(ps, hmat(H) @ ps)))
        return backend.evaluate(qc, H, x)

    def fid(x):
        ps = _state(x)
        return float(abs(np.vdot(psi, ps)) ** 2)

    def cost_fid(x):
        """(⟨H⟩, fidelity) sharing one statevector build — for candidate ranking."""
        ps = _state(x)
        if hmat is not None:
            e = float(np.real(np.vdot(ps, hmat(H) @ ps)))
        else:
            e = backend.evaluate(qc, H, x)
        return e, float(abs(np.vdot(psi, ps)) ** 2)

    cost.cost_fid = cost_fid  # attach without changing the 4-tuple signature
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


def optimize_xspace_bestof(cost, fid, grad, seed, x_indices, *, restarts, maxiter,
                           seed0, sigma_x=0.3, on_restart=None):
    """Best-of that spends its exploration budget on the θ_x subspace only.

    Every restart runs a FULL L-BFGS-B optimization (all parameters free — the
    ZZ angles still relax), but the restart perturbation is applied ONLY to the
    ``x_indices`` coordinates, keeping the ZZ angles anchored at their
    renormalized seed value. This matches the regime-aware strategy: the analytic
    seed pins the ZZ angles well, so exploration is best spent on the soft θ_x
    direction (the one the seeds least determine) rather than isotropically.

    Restart 0 starts exactly at ``seed``; restarts 1.. perturb ``seed`` on the
    θ_x coordinates by ``N(0, sigma_x)`` (a θ_x basin-hop). Selection is by
    energy. Returns ``(best_fid, best_e, runs)`` like :func:`optimize_bestof`.
    """
    if restarts < 1:
        raise ValueError(f"restarts must be >= 1, got {restarts}")
    seed = np.clip(np.asarray(seed, float), -np.pi, np.pi)
    npar = seed.shape[0]
    x_idx = np.asarray(sorted(set(int(i) for i in x_indices)), dtype=int)
    if x_idx.size and (x_idx.min() < 0 or x_idx.max() >= npar):
        raise ValueError("x_indices out of range for seed length")

    rng = np.random.default_rng(seed0)
    best_e, best_fid, best_r, runs = None, None, None, []
    for r in range(restarts):
        x0 = seed.copy()
        if r > 0 and x_idx.size:
            x0[x_idx] = np.clip(seed[x_idx] + rng.normal(0.0, sigma_x, x_idx.size),
                                -np.pi, np.pi)
        x, e, nit = _lbfgsb(cost, x0, maxiter=maxiter, grad=grad)
        f = fid(x)
        runs.append({
            "restart": r, "energy": e, "fidelity": f, "nit": nit,
            "converged": nit < maxiter,
            "theta_init": np.asarray(x0, float).tolist(),
            "theta_final": np.asarray(x, float).tolist(),
        })
        if best_e is None or e < best_e:
            best_e, best_fid, best_r = e, f, r
        if on_restart is not None:
            try:
                on_restart(runs, {"fidelity": best_fid, "energy": best_e, "restart": best_r})
            except Exception:
                pass
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


def converge_circuit(qc, H, psi, *, restarts, maxiter, seed0=0, warm_theta=None,
                     on_restart=None, backend=None):
    """Build the cost/fid pair and run the multi-seed best-of in one call.

    The three-line ``make_cost_fid`` + ``optimize_bestof`` wrapper every study
    runner re-implemented. Pass a shared ``backend`` to reuse the cached dense
    Hamiltonian across the variants of one sweep (set up once, not per circuit).
    Returns ``(best_fid, best_e, runs)`` — the same tuple as ``optimize_bestof``.
    """
    cost, fid, grad, _ = make_cost_fid(qc, H, psi, backend=backend)
    return optimize_bestof(cost, fid, grad, qc.num_parameters, restarts=restarts,
                           maxiter=maxiter, seed0=seed0, warm_theta=warm_theta,
                           on_restart=on_restart)


def build_variant_row(name, fid, e_best, runs, *, n_2q, n_params, e0, gap,
                      blocks=None, rx_final=None, bond_selection=None,
                      seed_kind=None, seconds=None, **extra):
    """Canonical per-variant result row — the single source of this dict shape.

    Every bond/variant study runner (``run_bond_ablation``, ``run_variant_topk``,
    ``run_ansatz_variants``, ``run_bond_topk_regime``) built the same result dict
    by hand with slightly different fields (so ``de_gap`` existed in some and not
    others). This centralizes it: energy/fidelity metrics, the transpiled 2q
    count, the derived ``de_gap`` and ``fidelity_per_cx``, the selected best θ,
    and the bond-selection bookkeeping. ``**extra`` is merged last for
    runner-specific columns. Pure (no I/O).

    ``runs`` is the per-restart list from :func:`optimize_bestof`; the lowest-energy
    restart supplies ``best_theta_final``.
    """
    best = min(runs, key=lambda r: r["energy"]) if runs else {}
    sel = bond_selection
    row = {
        "variant": name,
        "best_fidelity": fid,
        "e_best": e_best,
        "e0": e0,
        "gap": gap,
        "abs_error": abs(e_best - e0),
        "de_gap": abs(e_best - e0) / gap if gap and gap > 0 else None,
        "n_2q_transpiled": n_2q,
        "n_params": n_params,
        "fidelity_per_cx": (fid / n_2q) if n_2q else None,
        "n_nn_bonds": len(sel.nn_edges) if sel is not None else None,
        "n_nnn_bonds": len(sel.nnn_edges) if sel is not None else None,
        "selection_provenance": getattr(sel, "provenance", "full") if sel is not None else "full",
        "best_theta_final": best.get("theta_final"),
    }
    if blocks is not None:
        row["blocks"] = list(blocks)
    if rx_final is not None:
        row["rx_final"] = rx_final
    if seed_kind is not None:
        row["seed_kind"] = seed_kind
    if seconds is not None:
        row["seconds"] = round(float(seconds), 1)
    row.update(extra)
    return row


def prepare_warmstart(qc, H, psi, *, n_nn, n_nnn, n_qubits, p_layers, h,
                      J=1.0, J2=0.0, donors=None, target_nnn_edges=None,
                      extra_candidates=None, strategy="combined",
                      topology=None, model=None, gap=None,
                      micro_descent=None, cost=None, fid=None, grad=None,
                      backend=None, target_len=None):
    """One-call warm-start for any runner — the canonical integration point.

    Wraps the full combined cascade so a runner gets the BEST available seed in a
    single line, chosen ad-hoc from the problem structure (N, h, topology, model,
    J, J2, bonds), without re-implementing the fid/grad/micro-descent plumbing. It:

    1. builds (or reuses) ``cost``/``fid``/``grad`` for ``qc`` against ``psi``
       via :func:`make_cost_fid`;
    2. picks the technique set for this structure via
       :func:`qmbp_simulation.analysis.warmstart.warmstart_profile` (``topology``,
       ``model``, ``J2``) — the Ising-family analytic seeds (calibrated/structural)
       are enabled only where they are calibrated; the general regime seed and
       data-driven donors are always on; the validated-negative techniques
       (ensemble, block_mix, coupled coordinate-descent) are always off;
    3. constructs an L-BFGS-B micro-descent closure so candidates are ranked by
       the basin they fall into, not raw init-fid. ``micro_descent=None`` (the
       new default) sets the iteration budget ad-hoc from the (N, h) difficulty
       via :func:`qmbp_simulation.analysis.warmstart.micro_descent_budget` — the
       budget-fair study showed a single full descent is the best use of compute
       and the old flat 12 iters under-converged the hard phases (e.g. 0.65→0.99
       at h=0.3 going 12→48 iters). An explicit int is honored verbatim;
    4. delegates to :func:`qmbp_simulation.analysis.warmstart.best_combined_warmstart`
       (profile-selected analytic seeds + regime + transferred/cross-N donors +
       any extra candidates).

    ``strategy``:
      - ``"combined"`` (default): the full structure-aware cascade. THIS IS THE
        BEST TECHNIQUE and the default every runner now gets automatically.
      - ``"regime"``: regime seed only (analytic/donors off) — the pre-cascade
        behavior, kept for A/B comparisons.

    ``topology`` / ``model`` are optional; when omitted the TFIM-frustrated family
    (what the runners were built on) is assumed, so existing callers are
    unchanged. Passing them lets the gate disable the Ising-only analytic seeds on
    non-Ising models (Heisenberg/XY/Kitaev) instead of applying them blindly.

    Returns the ``best_combined_warmstart`` dict ``{seed, provenance,
    init_fidelity, report}`` plus the ``fid``/``grad``/``cost`` it built (keys
    ``_fid`` / ``_grad`` / ``_cost``) and the resolved ``_profile`` for provenance.

    Backend-agnostic and N-agnostic: it only needs an evaluable ``qc``/``H``/``psi``
    (the caller owns the backend), so every runner integrates the same way.
    """
    from qmbp_simulation.analysis.warmstart import (
        best_combined_warmstart,
        micro_descent_budget,
        warmstart_profile,
    )

    if cost is None or fid is None or grad is None:
        cost, fid, grad, backend = make_cost_fid(qc, H, psi, backend=backend)

    # Micro-descent budget: AUTO (difficulty-adaptive) when the caller does not
    # pin it. The budget-fair study found a single full L-BFGS-B is the best use
    # of the budget and the old flat 12 iters under-converged the hard phases, so
    # the default now scales iters with the (N, h) difficulty. An explicit int is
    # still honored verbatim (back-compat / A-B).
    if micro_descent is None:
        micro_descent = micro_descent_budget(n_qubits, h, gap, J2=J2)

    descent_fn = None
    if micro_descent and micro_descent > 0:
        def descent_fn(theta, _c=cost, _f=fid, _g=grad, _m=micro_descent):
            xr, _e, _nit = _lbfgsb(_c, np.asarray(theta, float), maxiter=_m, grad=_g)
            return xr, _f(xr)

    # Structure-aware technique selection (ad-hoc from model/topology/J2).
    profile = warmstart_profile(model=model, topology=topology, J2=J2)
    combined = strategy == "combined"
    use_donors = donors if combined else None

    res = best_combined_warmstart(
        n_nn=n_nn, n_nnn=n_nnn, n_qubits=n_qubits, p_layers=p_layers, h=h, J=J, J2=J2,
        fid_fn=fid, descent_fn=descent_fn, donors=use_donors,
        target_nnn_edges=target_nnn_edges, extra_candidates=extra_candidates,
        # In "combined" mode the profile decides the analytic candidates;
        # "regime" forces the pre-cascade regime-only behavior for A/B.
        include_calibrated=combined and profile["include_calibrated"],
        include_structural=combined and profile["include_structural"],
        include_regime=True,
        include_ensemble=profile["include_ensemble"],      # always False
        include_block_mix=profile["include_block_mix"],    # always False
        target_len=target_len,
    )
    res["_fid"] = fid
    res["_grad"] = grad
    res["_cost"] = cost
    res["_profile"] = profile
    res["_micro_descent"] = int(micro_descent)
    return res
