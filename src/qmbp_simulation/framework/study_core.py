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

import time

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


def optimize_xspace_bestof(cost, fid, grad, seed, x_indices, *, restarts, maxiter, seed0, sigma_x=0.3, on_restart=None):
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
            x0[x_idx] = np.clip(seed[x_idx] + rng.normal(0.0, sigma_x, x_idx.size), -np.pi, np.pi)
        x, e, nit = _lbfgsb(cost, x0, maxiter=maxiter, grad=grad)
        f = fid(x)
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


def converge_circuit(qc, H, psi, *, restarts, maxiter, seed0=0, warm_theta=None, on_restart=None, backend=None):
    """Build the cost/fid pair and run the multi-seed best-of in one call.

    The three-line ``make_cost_fid`` + ``optimize_bestof`` wrapper every study
    runner re-implemented. Pass a shared ``backend`` to reuse the cached dense
    Hamiltonian across the variants of one sweep (set up once, not per circuit).
    Returns ``(best_fid, best_e, runs)`` — the same tuple as ``optimize_bestof``.
    """
    cost, fid, grad, _ = make_cost_fid(qc, H, psi, backend=backend)
    return optimize_bestof(
        cost,
        fid,
        grad,
        qc.num_parameters,
        restarts=restarts,
        maxiter=maxiter,
        seed0=seed0,
        warm_theta=warm_theta,
        on_restart=on_restart,
    )


def build_variant_row(
    name,
    fid,
    e_best,
    runs,
    *,
    n_2q,
    n_params,
    e0,
    gap,
    blocks=None,
    rx_final=None,
    bond_selection=None,
    seed_kind=None,
    seconds=None,
    **extra,
):
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


def prepare_warmstart(
    qc,
    H,
    psi,
    *,
    n_nn,
    n_nnn,
    n_qubits,
    p_layers,
    h,
    J=1.0,
    J2=0.0,
    donors=None,
    target_nnn_edges=None,
    extra_candidates=None,
    strategy="combined",
    topology=None,
    model=None,
    gap=None,
    micro_descent=None,
    cost=None,
    fid=None,
    grad=None,
    backend=None,
    target_len=None,
    two_pass=False,
    short_descent=None,
    two_pass_top_k=2,
    warm_restart_full=False,
):
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

    ``two_pass`` (default False): enable the M1+M2 two-pass selector. A cheap
    SHORT descent (:func:`short_descent_budget` of the full budget, or an explicit
    ``short_descent``) pre-ranks every candidate, then the full difficulty-adaptive
    descent runs only on the top ``two_pass_top_k`` survivors. Grounded in the
    selector_budget study: the full budget at the transition already reaches the
    K>=400 regime where the ranking is correct, so the short pass only prunes
    obvious losers and never decides the winner. Off by default → ``descent_fn_short``
    is None and the exhaustive full-descent-on-all-candidates behaviour (and its
    tests) is unchanged. The short pre-rank runs once per DISTINCT candidate
    (duplicates inherit the score) — a pure dedup.

    ``warm_restart_full`` (default False): continue each survivor's full descent
    from its short-refined θ (reuse the pre-rank work). NOT decision-neutral on a
    multi-basin landscape — opt-in, A/B-validate before enabling. The default
    starts the full descent from the raw seed (exhaustive behaviour preserved).

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
        short_descent_budget,
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

    # Two-pass selector (M1+M2): a cheap SHORT descent pre-ranks all candidates,
    # the full (difficulty-adaptive) descent settles the winner on the top-k
    # survivors. The selector_budget study showed the full budget already reaches
    # the K>=400 regime where the ranking is correct at the transition, so the
    # short pass only prunes — it never decides. Off by default (two_pass=False →
    # descent_fn_short=None) so the exhaustive full-descent-on-all behaviour and
    # its tests are byte-for-byte unchanged.
    descent_fn_short = None
    if two_pass and micro_descent and micro_descent > 0:
        if short_descent is None:
            short_descent = short_descent_budget(micro_descent)
        if short_descent and short_descent > 0:

            def descent_fn_short(theta, _c=cost, _f=fid, _g=grad, _m=short_descent):
                xr, _e, _nit = _lbfgsb(_c, np.asarray(theta, float), maxiter=_m, grad=_g)
                return xr, _f(xr)

    # Structure-aware technique selection (ad-hoc from model/topology/J2).
    profile = warmstart_profile(model=model, topology=topology, J2=J2)
    combined = strategy == "combined"
    use_donors = donors if combined else None

    res = best_combined_warmstart(
        n_nn=n_nn,
        n_nnn=n_nnn,
        n_qubits=n_qubits,
        p_layers=p_layers,
        h=h,
        J=J,
        J2=J2,
        fid_fn=fid,
        descent_fn=descent_fn,
        donors=use_donors,
        target_nnn_edges=target_nnn_edges,
        extra_candidates=extra_candidates,
        topology=topology,
        # In "combined" mode the profile decides the analytic candidates;
        # "regime" forces the pre-cascade regime-only behavior for A/B.
        include_calibrated=combined and profile["include_calibrated"],
        include_structural=combined and profile["include_structural"],
        include_regime=True,
        include_ensemble=profile["include_ensemble"],  # always False
        include_block_mix=profile["include_block_mix"],  # always False
        target_len=target_len,
        descent_fn_short=descent_fn_short,
        two_pass_top_k=two_pass_top_k,
        warm_restart_full=warm_restart_full,
    )
    res["_fid"] = fid
    res["_grad"] = grad
    res["_cost"] = cost
    res["_profile"] = profile
    res["_micro_descent"] = int(micro_descent)
    res["_short_descent"] = int(short_descent) if descent_fn_short is not None else None
    res["_two_pass"] = bool(descent_fn_short is not None)
    return res


def bond_energy_gradients(builder, n, lat, selection, candidate_nn, candidate_nnn, H, psi, theta_current):
    """|∂E/∂θ| for each CANDIDATE bond, evaluated at the CURRENT OPTIMIZED state.

    The ADAPT growth signal, shared by the ADAPT runner and the Gate-0 validator.
    Builds a probe circuit = ``selection`` + the candidate bonds appended, then
    takes the adjoint gradient of ⟨H⟩ at the working point: the kept bonds and X
    rotations carry their optimized values (from ``theta_current``), the candidate
    θ start at 0. Evaluating at the OPTIMIZED state (not θ=0) is essential — at the
    symmetric θ=0 point the RZZ gradient vanishes by symmetry (the classic ADAPT
    "zero initial gradient" trap), which would stall growth.

    ``selection`` is the current :class:`BondSelection` (blocks ``[nn, nnn, x]``);
    ``candidate_nn`` / ``candidate_nnn`` are edge lists to probe. ``theta_current``
    is the optimized θ for the current selection in that same block order.

    Returns ``(grad_nn, grad_nnn)`` aligned with ``candidate_nn`` / ``candidate_nnn``.
    Centralizes the ``_bond_gradients`` helper that run_adapt_bonds defined inline,
    so Gate 0 (ranking agreement) and Gate 2 (growth) consume one implementation.
    """
    from qmbp_simulation.circuits.bond_mask import BondSelection
    from qmbp_simulation.circuits.hva_variants import build_variant, make_masked_variant

    probe_sel = BondSelection(
        nn_edges=list(selection.nn_edges) + list(candidate_nn),
        nnn_edges=list(selection.nnn_edges) + list(candidate_nnn),
    )
    v = make_masked_variant("probe", "adapt gradient probe", ["nn", "nnn", "x"], probe_sel)
    qc, _ = build_variant(builder, n, lat, v)
    _cost, _fid, grad_fn, _ = make_cost_fid(qc, H, psi)

    # Probe θ layout: [nn_kept, nn_cand, nnn_kept, nnn_cand, x].
    # Current θ layout (masked, blocks nn,nnn,x): [nn_kept, nnn_kept, x].
    n_nn_kept = len(selection.nn_edges)
    n_nn_cand = len(candidate_nn)
    n_nnn_kept = len(selection.nnn_edges)
    n_nnn_cand = len(candidate_nnn)
    tc = np.asarray(theta_current, float)
    theta_probe = np.zeros(qc.num_parameters)
    theta_probe[:n_nn_kept] = tc[:n_nn_kept]
    p_off = n_nn_kept + n_nn_cand
    theta_probe[p_off : p_off + n_nnn_kept] = tc[n_nn_kept : n_nn_kept + n_nnn_kept]
    theta_probe[-n:] = tc[-n:]

    g = np.asarray(grad_fn(theta_probe), float)
    grad_nn = np.abs(g[n_nn_kept : n_nn_kept + n_nn_cand]) if n_nn_cand else np.array([])
    off = n_nn_kept + n_nn_cand + n_nnn_kept
    grad_nnn = np.abs(g[off : off + n_nnn_cand]) if n_nnn_cand else np.array([])
    return grad_nn, grad_nnn


def _remap_theta_grow_nnn(theta_prev, n_nn, n_nnn_prev, n_added, n_qubits):
    """Map a converged θ to the layout AFTER appending ``n_added`` nnn bonds.

    The masked ``[nn, nnn, x]`` layout is ``[nn (n_nn) | nnn (n_nnn) | x (n)]``.
    Growing the nnn block from ``n_nnn_prev`` to ``n_nnn_prev + n_added`` inserts
    ``n_added`` fresh angles (initialized to 0 — a near-identity RZZ) between the
    old nnn angles and the x block, so the warm-start continues from the previous
    optimum instead of restarting cold. Returns the remapped θ (length
    ``n_nn + n_nnn_prev + n_added + n_qubits``).
    """
    tp = np.asarray(theta_prev, float)
    out = np.zeros(n_nn + n_nnn_prev + n_added + n_qubits)
    out[:n_nn] = tp[:n_nn]
    out[n_nn : n_nn + n_nnn_prev] = tp[n_nn : n_nn + n_nnn_prev]
    out[-n_qubits:] = tp[-n_qubits:]
    return out


def grow_bonds_adapt(
    builder,
    n,
    lat,
    H,
    psi,
    e0,
    gap,
    *,
    nn_all,
    nnn_all,
    n_nn,
    n_nnn,
    restarts,
    maxiter,
    seed0,
    target_fid,
    grow_step=2,
    grad_tol=1e-6,
    warm_per_step=True,
    efficiency_stop=False,
    efficiency_patience=1,
    seed_selection=None,
    warm_theta0=None,
    backend=None,
    on_step=None,
):
    """ADAPT bond-growth loop → the fidelity-vs-2q growth curve (list of step dicts).

    Grows the nnn block one ``grow_step`` at a time, each new bond chosen by
    ``|∂E/∂θ|`` (via :func:`bond_energy_gradients`) at the current optimized
    state — the ADAPT signal that avoids the θ=0 zero-gradient trap. Factored out
    of the ``run_adapt_bonds`` script so the Gate-1/Gate-2 runners consume one
    implementation instead of re-growing bonds by hand.

    Enhancements over the original inline loop:

    - ``warm_per_step`` (default True): seed each growth step's restart-0 from the
      PREVIOUS step's optimized θ (remapped to the grown layout via
      :func:`_remap_theta_grow_nnn`), the natural ADAPT continuation. The random
      restarts (1..) still explore, so a new bond opening a different basin is not
      missed — keep ``restarts >= 2`` for that safety. Step 0 uses ``warm_theta0``
      (e.g. an analytic/donor seed) when given.
    - ``grow_step`` may be an ``int`` (fixed) or a ``callable(step_index,
      n_nnn_selected, n_nnn_total) -> int`` (adaptive: coarse early, fine near the
      target). Clamped to the remaining-bond count.
    - ``efficiency_stop`` (default False): also stop when fidelity-per-CX stops
      improving for ``efficiency_patience`` consecutive steps, so the returned
      curve ends at the Pareto-efficient point, not merely the first fidelity
      crossing.
    - ``seed_selection``: an initial :class:`BondSelection` to start FROM (e.g. a
      known top-k selection) instead of the nn-only backbone — turns ADAPT into a
      refiner of a known good structure.

    ``on_step(step_dict)`` is called after each optimized step (for crash-safe
    persistence). Returns the list of per-step dicts (same schema the ADAPT runner
    persisted). Pure w.r.t. I/O (no file writes); the caller owns persistence.
    """
    from qmbp_simulation.circuits.bond_mask import BondSelection, rank_by_gradient
    from qmbp_simulation.circuits.hva_variants import build_variant, make_masked_variant

    if seed_selection is not None:
        selection = BondSelection(
            nn_edges=list(seed_selection.nn_edges),
            nnn_edges=list(seed_selection.nnn_edges),
            provenance="adapt_seed(from_selection)",
        )
    else:
        selection = BondSelection(nn_edges=list(nn_all), nnn_edges=[], provenance="adapt_seed(nn_only)")

    def _grow_count(step_index, n_sel, n_tot):
        if callable(grow_step):
            return max(1, int(grow_step(step_index, n_sel, n_tot)))
        return max(1, int(grow_step))

    steps: list[dict] = []
    warm_prev = warm_theta0
    n_nnn_prev = len(selection.nnn_edges)
    best_fpc = -1.0
    stale = 0
    reached = False

    for it in range(n_nnn + 1):
        v = make_masked_variant(
            f"adapt_step{it}",
            f"adapt step {it} ({selection.n_bonds} bonds)",
            ["nn", "nnn", "x"],
            selection,
            tags=("adapt",),
        )
        qc, _ = build_variant(builder, n, lat, v)
        n_2q, npar = cx_and_params(qc)

        warm = warm_prev if (warm_per_step and warm_prev is not None and len(warm_prev) == npar) else None
        t0 = time.time()
        fid, e_best, runs = converge_circuit(
            qc, H, psi, restarts=restarts, maxiter=maxiter, seed0=seed0, warm_theta=warm, backend=backend
        )
        best_run = min(runs, key=lambda r: r["energy"])
        fpc = (fid / n_2q) if n_2q > 0 else None
        step = {
            "step": it,
            "n_nn_bonds": len(selection.nn_edges),
            "n_nnn_bonds": len(selection.nnn_edges),
            "n_bonds": selection.n_bonds,
            "best_fidelity": fid,
            "e_best": e_best,
            "e0": e0,
            "gap": gap,
            "abs_error": abs(e_best - e0),
            "de_gap": abs(e_best - e0) / gap if gap > 0 else None,
            "n_2q_transpiled": n_2q,
            "n_params": npar,
            "fidelity_per_cx": fpc,
            "selection_provenance": selection.provenance,
            "best_theta_final": best_run["theta_final"],
            "seconds": round(time.time() - t0, 1),
        }
        steps.append(step)
        if on_step is not None:
            try:
                on_step(step)
            except Exception:
                pass  # persistence must never break the growth loop

        if fid >= target_fid:
            reached = True
            step["stop_reason"] = "target_fid"
            break

        # Efficiency stop: fid/CX stopped improving.
        if efficiency_stop and fpc is not None:
            if fpc > best_fpc + 1e-9:
                best_fpc = fpc
                stale = 0
            else:
                stale += 1
                if stale >= efficiency_patience:
                    step["stop_reason"] = "efficiency_plateau"
                    break

        remaining = [e for e in nnn_all if e not in set(selection.nnn_edges)]
        if not remaining:
            step["stop_reason"] = "full_layer"
            break

        theta_current = np.asarray(best_run["theta_final"], float)
        _gnn, g_nnn = bond_energy_gradients(builder, n, lat, selection, [], remaining, H, psi, theta_current)
        ranked = rank_by_gradient([], remaining, np.array([]), g_nnn)
        g = _grow_count(it, len(selection.nnn_edges), n_nnn)
        top = [edge for _kind, edge, score in ranked if score > grad_tol][:g]
        if not top:
            step["stop_reason"] = "grad_below_tol"
            break

        # Warm-start for the NEXT step: remap this step's θ to the grown layout.
        n_nnn_prev = len(selection.nnn_edges)
        warm_prev = _remap_theta_grow_nnn(theta_current, n_nn, n_nnn_prev, len(top), n)
        selection = BondSelection(
            nn_edges=list(selection.nn_edges),
            nnn_edges=list(selection.nnn_edges) + top,
            provenance=f"adapt(step{it + 1}, +{len(top)} nnn by grad)",
        )

    if steps and "stop_reason" not in steps[-1]:
        steps[-1]["stop_reason"] = "max_steps"
    return steps, reached


def _block_param_size(block, n_nn_sel, n_nnn_sel, n_qubits):
    """θ-count a single block contributes for a given masked selection."""
    return {"nn": n_nn_sel, "nnn": n_nnn_sel, "x": n_qubits, "z": n_qubits}[block]


def _append_blocks_warm(theta_prev, append_blocks, n_nn_sel, n_nnn_sel, n_qubits):
    """Warm-start θ after APPENDING blocks at the end (new params start at 0).

    The masked builder lays params out in ``blocks`` order; appending blocks
    (the 'repeat a layer' action) adds their angles at the TAIL. Continuing from
    the previous optimum means keeping the old θ and zero-filling the new tail —
    the repeated entangling layer starts as near-identity and the optimizer bends
    it only as far as it helps. Returns the extended θ.
    """
    tp = np.asarray(theta_prev, float)
    extra = sum(_block_param_size(b, n_nn_sel, n_nnn_sel, n_qubits) for b in append_blocks)
    return np.concatenate([tp, np.zeros(extra)])


def grow_adapt_pool(
    builder,
    n,
    lat,
    H,
    psi,
    e0,
    gap,
    *,
    nn_all,
    nnn_all,
    n_nn,
    n_nnn,
    restarts,
    maxiter,
    seed0,
    target_fid,
    base_blocks=("nn", "nnn", "x"),
    repeat_block=("nn", "nnn", "x"),
    grow_step=2,
    grad_tol=1e-6,
    allow_repeat_layer=True,
    max_layers=6,
    efficiency_patience=2,
    min_dfid_keep_growing=0.01,
    layer_growth_penalty=0.0,
    fast_rank=False,
    analytic_seed_new=False,
    select_by_fidelity=False,
    block_precondition=False,
    seed_selection=None,
    warm_theta0=None,
    backend=None,
    on_step=None,
):
    """ADAPT with NO fixed p — grow by the action that buys most fidelity per 2q.

    Option A of "unrestricted-p ADAPT": instead of only appending nnn bonds to a
    single fixed layer, each step picks the best action from a POOL and applies
    it, where "best" is the highest Δfidelity / Δ(2q-gates) — the project's actual
    objective (more fidelity for fewer two-qubit gates). The pool per step:

    - ``add_bonds``: append the top-``grow_step`` not-yet-selected nnn bonds by
      ``|∂E/∂θ|`` (the ADAPT bond signal), same as :func:`grow_bonds_adapt`.
    - ``repeat_layer``: append one more entangling layer (``repeat_block``) to the
      block sequence — a LOCAL, gradient-justified version of raising p, applied
      only when it beats adding a bond on the Δfid/Δ2q metric.

    Each candidate is actually converged (the pool is tiny, so Δfid/Δ2q is
    measured, not estimated) warm-started from the current optimum; the winner
    becomes the new state. Growth stops at ``target_fid``, when no action improves
    Δfid/Δ2q for ``efficiency_patience`` steps, when the nnn set is full and the
    layer cap is hit, or when no candidate exists.

    The action is an explicit record (``kind``/``blocks``/``selection``/``warm``/
    ``delta_2q``), so Option B (a pool of individual Pauli operators, no layers)
    plugs into the SAME loop by extending :func:`_adapt_pool_actions` — the loop
    never assumes layers or bonds, only "apply the best action".

    Speed / quality levers (all opt-in; defaults reproduce the validated run):

    - ``fast_rank`` (default False): score candidates by energy gradient per 2q
      and converge ONLY the top one. NOTE (measured): this is reliable only when
      the pool is LARGE and HOMOGENEOUS (many same-type candidates, as in the
      operator pool of :func:`grow_adapt_operators`). For THIS pool — two
      heterogeneous actions (add_bonds vs repeat_layer) — the gradient proxy does
      NOT compare the two action types well and can pick the wrong one, collapsing
      fidelity (N10 h=0.7: 0.97→0.61). Keep it False for grow_adapt_pool; it exists
      for the operator-list loop. Prefer ``warm_theta0`` + ``layer_growth_penalty``
      to speed up this pool (measured 2.1× faster at equal fidelity).
    - ``warm_theta0``: seed step 0 from an analytic/donor warm-start
      (:func:`prepare_warmstart`) instead of the raw backbone, cutting iterations
      on the (otherwise cold) first and most expensive step.
    - ``layer_growth_penalty`` (default 0): raise the Δfid needed to keep growing
      as layers accumulate — ``min_dfid_eff = min_dfid * (1 + penalty*(n_layers-1))``.
      Early layers (cheap fidelity) grow freely; late layers must earn their 2q
      cost, so the loop stops over-growing at the critical point (N14 h=0.7 grew to
      5 layers) while still rescuing the paramagnetic phase (growth there is early).

    Returns ``(steps, reached)`` with the same per-step schema as
    :func:`grow_bonds_adapt` plus ``blocks`` and ``action_kind`` per step.
    """
    from qmbp_simulation.circuits.bond_mask import BondSelection, rank_by_gradient
    from qmbp_simulation.circuits.hva_variants import build_variant, make_masked_variant

    base_blocks = list(base_blocks)
    repeat_block = list(repeat_block)

    if seed_selection is not None:
        selection = BondSelection(
            nn_edges=list(seed_selection.nn_edges),
            nnn_edges=list(seed_selection.nnn_edges),
            provenance="adapt_pool_seed(from_selection)",
        )
    else:
        selection = BondSelection(nn_edges=list(nn_all), nnn_edges=[], provenance="adapt_pool_seed(nn_only)")
    blocks = list(base_blocks)

    def _converge(bl, sel, warm):
        v = make_masked_variant(
            "adapt_pool", f"adapt pool ({sel.n_bonds} bonds, {len(bl)} blocks)", bl, sel, tags=("adapt_pool",)
        )
        qc, _ = build_variant(builder, n, lat, v)
        n_2q, npar = cx_and_params(qc)
        warm_ok = warm if (warm is not None and len(warm) == npar) else None
        # A non-finite warm-start would bind NaN into the circuit; drop it.
        if warm_ok is not None and not np.all(np.isfinite(warm_ok)):
            warm_ok = None

        cost_f = fid_f = grad_f = None
        if (analytic_seed_new or block_precondition) and warm_ok is not None:
            cost_f, fid_f, grad_f, _ = make_cost_fid(qc, H, psi, backend=backend)

        # 2c: replace the 0-initialized new params with a 1D Newton estimate.
        if analytic_seed_new and warm_ok is not None:
            new_idx = [i for i, val in enumerate(warm_ok) if val == 0.0]
            if new_idx:
                warm_ok = analytic_seed_new_params(cost_f, warm_ok, new_idx)

        # 2b: per-block preconditioner D from the gradient at the warm point.
        D = None
        if block_precondition and warm_ok is not None:
            try:
                g0 = np.asarray(grad_f(warm_ok), float)
                n_nn_sel = len(sel.nn_edges)
                n_nnn_sel = len(sel.nnn_edges)
                D = block_precondition_scales(bl, n_nn_sel, n_nnn_sel, n, g0, rx_final=False, rz_final=False)
            except Exception:
                D = None

        try:
            if D is not None:
                fid, e_best, runs = converge_circuit_preconditioned(
                    qc,
                    H,
                    psi,
                    restarts=restarts,
                    maxiter=maxiter,
                    seed0=seed0,
                    warm_theta=warm_ok,
                    D=D,
                    backend=backend,
                )
            else:
                fid, e_best, runs = converge_circuit(
                    qc, H, psi, restarts=restarts, maxiter=maxiter, seed0=seed0, warm_theta=warm_ok, backend=backend
                )
        except Exception:
            # A candidate whose optimization diverges (e.g. L-BFGS-B hitting a
            # NaN gradient) must not kill the whole point — discard it so the
            # pool simply never picks it (fid=-inf loses every Δfid/Δ2q compare).
            return float("-inf"), float("inf"), n_2q, npar, np.zeros(npar)
        theta = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)
        if not np.all(np.isfinite(theta)):
            return float("-inf"), float("inf"), n_2q, npar, np.zeros(npar)
        return fid, e_best, n_2q, npar, theta

    def _layer_count(bl):
        # One entangling layer == one nn block (every standard/repeat layer has one).
        return sum(1 for b in bl if b == "nn")

    # Step 0: converge the seed state (no action chosen yet).
    t0 = time.time()
    fid, e_best, n_2q, npar, theta = _converge(blocks, selection, warm_theta0)
    steps: list[dict] = []
    best_fpc = (fid / n_2q) if n_2q > 0 else 0.0
    stale = 0
    reached = False

    def _record(it, action_kind, fid, e_best, n_2q, npar, theta, seconds):
        fpc = (fid / n_2q) if n_2q > 0 else None
        s = {
            "step": it,
            "action_kind": action_kind,
            "n_nn_bonds": len(selection.nn_edges),
            "n_nnn_bonds": len(selection.nnn_edges),
            "n_bonds": selection.n_bonds,
            "n_layers": _layer_count(blocks),
            "blocks": list(blocks),
            "best_fidelity": fid,
            "e_best": e_best,
            "e0": e0,
            "gap": gap,
            "abs_error": abs(e_best - e0),
            "de_gap": abs(e_best - e0) / gap if gap > 0 else None,
            "n_2q_transpiled": n_2q,
            "n_params": npar,
            "fidelity_per_cx": fpc,
            "best_theta_final": theta.tolist(),
            "seconds": round(seconds, 1),
        }
        steps.append(s)
        if on_step is not None:
            try:
                on_step(s)
            except Exception:
                pass
        return s

    _record(0, "seed", fid, e_best, n_2q, npar, theta, time.time() - t0)
    if not np.isfinite(fid):
        steps[-1]["stop_reason"] = "seed_diverged"
        return steps, False
    if fid >= target_fid:
        steps[-1]["stop_reason"] = "target_fid"
        return steps, True

    max_steps = n_nnn + max_layers + 2
    for it in range(1, max_steps + 1):
        candidates = _adapt_pool_actions(
            builder,
            n,
            lat,
            H,
            psi,
            blocks,
            selection,
            theta,
            nnn_all,
            n_nn,
            n,
            grow_step=grow_step,
            grad_tol=grad_tol,
            repeat_block=repeat_block,
            allow_repeat_layer=allow_repeat_layer,
            max_layers=max_layers,
            rank_by_gradient=rank_by_gradient,
            _BondSelection=BondSelection,
            select_by_fidelity=select_by_fidelity,
        )
        if not candidates:
            steps[-1]["stop_reason"] = "no_action"
            break

        tcand = time.time()
        if fast_rank and len(candidates) > 1:
            # Cheap pre-rank: score each candidate by |energy gradient on its NEW
            # params| / Δ2q at the current optimum (no optimization), then converge
            # ONLY the top one. Collapses len(pool) full optimizations/step → ~1.
            scored = []
            for act in candidates:
                g = _candidate_grad_score(
                    builder, n, lat, H, psi, act, theta, n_2q, make_cost_fid, build_variant, make_masked_variant
                )
                scored.append((g, act))
            scored.sort(key=lambda t: -t[0])
            candidates = [scored[0][1]]  # keep only the gradient-best action

        evaluated = []
        for act in candidates:
            c_fid, c_e, c_2q, c_npar, c_theta = _converge(act["blocks"], act["selection"], act["warm"])
            d2q = max(c_2q - n_2q, 1)  # avoid divide-by-zero; actions add ≥1 CX pair
            dfid = c_fid - fid
            evaluated.append(
                {
                    "act": act,
                    "fid": c_fid,
                    "e": c_e,
                    "n_2q": c_2q,
                    "npar": c_npar,
                    "theta": c_theta,
                    "dfid_per_2q": dfid / d2q,
                    "dfid": dfid,
                }
            )
        best = max(evaluated, key=lambda r: r["dfid_per_2q"])
        if not np.isfinite(best["fid"]):
            # Every candidate diverged (all fid=-inf) — keep the last good state.
            steps[-1]["stop_reason"] = "diverged"
            break

        blocks = list(best["act"]["blocks"])
        selection = best["act"]["selection"]
        fid, e_best, n_2q, npar, theta = (best["fid"], best["e"], best["n_2q"], best["npar"], best["theta"])
        _record(it, best["act"]["kind"], fid, e_best, n_2q, npar, theta, time.time() - tcand)

        if fid >= target_fid:
            reached = True
            steps[-1]["stop_reason"] = "target_fid"
            break

        # Progress = fid/CX improved, OR absolute fidelity still rising hard.
        # The second clause is the fix: a step that adds an expensive layer drops
        # fid/CX yet may lift fidelity a lot (the paramagnetic phase needed more
        # layers but efficiency alone stopped growth at 1). Keep growing while the
        # absolute Δfid clears min_dfid_keep_growing, so the plateau rule governs
        # only once fidelity has genuinely flattened.
        # The Δfid required to "keep growing" RISES with the number of layers
        # already present: early layers buy cheap fidelity and grow freely, late
        # layers must clear a higher bar to justify their 2q cost. penalty=0 keeps
        # the flat threshold (validated behavior); penalty>0 stops over-growth at
        # the critical point without penalizing the paramagnetic rescue (whose
        # growth happens in early layers).
        cur_layers = _layer_count(blocks)
        min_dfid_eff = min_dfid_keep_growing * (1.0 + layer_growth_penalty * max(cur_layers - 1, 0))
        fpc = (fid / n_2q) if n_2q > 0 else 0.0
        improved_fpc = fpc > best_fpc + 1e-9 and best["dfid"] > 1e-6
        strong_dfid = best["dfid"] >= min_dfid_eff
        if improved_fpc or strong_dfid:
            best_fpc = max(best_fpc, fpc)
            stale = 0
        else:
            stale += 1
            if stale >= efficiency_patience:
                steps[-1]["stop_reason"] = "efficiency_plateau"
                break

    if steps and "stop_reason" not in steps[-1]:
        steps[-1]["stop_reason"] = "max_steps"
    return steps, reached


def _candidate_grad_score(
    builder, n, lat, H, psi, act, theta_current, n_2q_prev, make_cost_fid, build_variant, make_masked_variant
):
    """Cheap pre-rank score for a pool action: |∂E/∂θ on NEW params| / Δ2q.

    Builds the candidate circuit, places the inherited θ on the shared prefix and
    0 on the action's new params, and takes the adjoint gradient at that point —
    one gradient, no optimization. The score is the largest new-param gradient
    magnitude divided by the action's 2q cost, i.e. "steepest energy descent per
    two-qubit gate" — the fast_rank proxy for the measured Δfid/Δ2q. Evaluating at
    the inherited (optimized-prefix) point keeps RZZ gradients non-vanishing.
    """
    from qmbp_simulation.circuits.hva_variants import AnsatzVariant

    sel = act["selection"]
    bl = act["blocks"]
    v = (
        make_masked_variant("probe", "fast-rank probe", bl, sel)
        if sel is not None
        else AnsatzVariant(name="probe", description="fast-rank probe", blocks=list(bl))
    )
    qc, _ = build_variant(builder, n, lat, v)
    npar = qc.num_parameters
    warm = np.asarray(act["warm"], float)
    if warm.size != npar:
        return -np.inf
    # New params = those the inherited prefix did not fill (warm==0 at the tail /
    # inserted slots). Use the action's own warm layout: the gradient on the full
    # vector, then take the max over the entries the action introduced.
    _cost, _fid, grad_fn, _ = make_cost_fid(qc, H, psi)
    try:
        g = np.abs(np.asarray(grad_fn(warm), float))
    except Exception:
        return -np.inf
    d2q = max(act.get("delta_2q_hint", 2), 1)
    # The introduced params are the ones where the inherited θ is exactly 0
    # (added bonds / fresh layer start at 0). If none are zero, fall back to the
    # global max gradient (still a valid steepness proxy).
    new_mask = warm == 0.0
    gnew = g[new_mask] if np.any(new_mask) else g
    return float(np.max(gnew)) / d2q if gnew.size else -np.inf


def _adapt_pool_actions(
    builder,
    n,
    lat,
    H,
    psi,
    blocks,
    selection,
    theta,
    nnn_all,
    n_nn,
    n_qubits,
    *,
    grow_step,
    grad_tol,
    repeat_block,
    allow_repeat_layer,
    max_layers,
    rank_by_gradient,
    _BondSelection,
    select_by_fidelity=False,
):
    """Build the candidate action pool for one grow_adapt_pool step.

    Each action is a dict ``{kind, blocks, selection, warm, delta_2q_hint}``.
    This is the Option-B seam: adding an individual-operator pool means returning
    more actions here, with no change to the grow_adapt_pool loop. Returns the
    (possibly empty) list of applicable actions.
    """
    actions = []
    n_nn_sel = len(selection.nn_edges)
    n_nnn_sel = len(selection.nnn_edges)

    # Action: add the top-grow_step nnn bonds. Rank by |∂E/∂θ| (default), or by
    # |∂Fidelity/∂θ| when select_by_fidelity (1a) — the latter aligns the choice
    # with the actual objective where energy is flat but fidelity is not (small gap).
    remaining = [e for e in nnn_all if e not in set(selection.nnn_edges)]
    if remaining:
        if select_by_fidelity:
            _g_nn, g_nnn = fidelity_bond_gradients(builder, n, lat, selection, [], remaining, H, psi, theta)
        else:
            _g_nn, g_nnn = bond_energy_gradients(builder, n, lat, selection, [], remaining, H, psi, theta)
        # Symmetry fallback: if the current optimum sits at a θ=0-like point (e.g.
        # right after a repeat_layer whose new angles stayed at 0), every nnn
        # gradient vanishes by symmetry and ranking is meaningless. Break the
        # symmetry with a tiny deterministic perturbation so add_bonds stays in
        # the pool instead of starving the whole loop (the ADAPT zero-gradient
        # trap, re-entering through the repeat action).
        if float(np.max(np.abs(g_nnn))) <= grad_tol:
            rng = np.random.default_rng(len(selection.nnn_edges) + 12345)
            theta_pert = np.asarray(theta, float) + rng.normal(0.0, 0.05, len(theta))
            _g_nn, g_nnn = bond_energy_gradients(builder, n, lat, selection, [], remaining, H, psi, theta_pert)
        ranked = rank_by_gradient([], remaining, np.array([]), g_nnn)
        top = [edge for _k, edge, score in ranked if score > grad_tol][:grow_step]
        if not top and remaining:
            top = remaining[:grow_step]  # last resort: add in lattice order
        if top:
            new_sel = _BondSelection(
                nn_edges=list(selection.nn_edges),
                nnn_edges=list(selection.nnn_edges) + top,
                provenance=f"+{len(top)} nnn by grad",
            )
            # Warm: new nnn angles inserted at 0 (layout: existing blocks keep θ).
            warm = _warm_for_added_bonds(theta, blocks, selection, len(top), n_nn_sel, n_nnn_sel, n_qubits)
            actions.append(
                {
                    "kind": "add_bonds",
                    "blocks": list(blocks),
                    "selection": new_sel,
                    "warm": warm,
                    "delta_2q_hint": 2 * len(top),
                }
            )

    # Action: repeat an entangling layer (local, gradient-free raise of p).
    n_layers = sum(1 for b in blocks if b == "nn")
    if allow_repeat_layer and n_layers < max_layers:
        new_blocks = list(blocks) + list(repeat_block)
        warm = _append_blocks_warm(theta, repeat_block, n_nn_sel, n_nnn_sel, n_qubits)
        # Break the θ=0 symmetry of the fresh layer: a near-identity repeated
        # layer sits at the vanishing-gradient point, so with few restarts the
        # optimizer leaves it untouched (observed: restarts=1 → layer never
        # activates). A tiny deterministic perturbation on ONLY the new tail
        # angles seeds a descent direction while keeping the inherited θ verbatim.
        n_prev = len(np.asarray(theta, float))
        rng = np.random.default_rng(n_prev + len(new_blocks) + 777)
        warm = np.asarray(warm, float).copy()
        warm[n_prev:] = rng.normal(0.0, 0.05, warm.size - n_prev)
        extra = 2 * (n_nn_sel * repeat_block.count("nn") + n_nnn_sel * repeat_block.count("nnn"))
        actions.append(
            {"kind": "repeat_layer", "blocks": new_blocks, "selection": selection, "warm": warm, "delta_2q_hint": extra}
        )
    return actions


def _warm_for_added_bonds(theta, blocks, selection, n_added, n_nn_sel, n_nnn_sel, n_qubits):
    """Warm-start θ after growing the nnn selection, for an arbitrary block seq.

    Every ``nnn`` block in ``blocks`` widens by ``n_added`` angles (the new bonds,
    inserted at 0); every other block keeps its θ verbatim. Generalizes
    :func:`_remap_theta_grow_nnn` (single ``[nn,nnn,x]`` layer) to the multi-layer
    sequences grow_adapt_pool produces.
    """
    tp = np.asarray(theta, float)
    out = []
    off = 0
    size = {"nn": n_nn_sel, "nnn": n_nnn_sel, "x": n_qubits, "z": n_qubits}
    for b in blocks:
        sz = size[b]
        out.append(tp[off : off + sz])
        if b == "nnn":
            out.append(np.zeros(n_added))  # new bonds appended within each nnn block
        off += sz
    return np.concatenate(out) if out else tp


def _default_operator_pool(nn_all, nnn_all, n_qubits, *, include_rotations=True):
    """Candidate operator pool for ADAPT-B: every RZZ edge + single-qubit rots.

    Returns a list of ``(kind, target)`` operators the growth loop can append:
    one ``("rzz", edge)`` per nn and nnn edge, and (optionally) one ``("rx", q)``
    and ``("rz", q)`` per qubit. This is the full canonical pool — the loop ranks
    it by energy gradient each step and adds the best, with NO layer structure.
    """
    pool = [("rzz", (int(i), int(j))) for i, j in nn_all]
    pool += [("rzz", (int(i), int(j))) for i, j in nnn_all]
    if include_rotations:
        pool += [("rx", q) for q in range(n_qubits)]
        pool += [("rz", q) for q in range(n_qubits)]
    return pool


def operator_pool_gradients(builder, n, operators, candidates, H, psi, theta_current):
    """|∂E/∂θ| for each CANDIDATE operator, at the current optimized state.

    The ADAPT-B growth signal, mirroring :func:`bond_energy_gradients` for the
    operator-list ansatz. Builds a probe circuit = ``operators`` (carrying the
    optimized ``theta_current``) followed by every candidate appended with θ=0,
    then takes the adjoint gradient of ⟨H⟩. Evaluating at the OPTIMIZED state (not
    θ=0) is what makes the RZZ gradients non-vanishing — the same zero-gradient
    trap avoidance as the bond growth. Returns ``|grad|`` aligned with ``candidates``.
    """
    probe_ops = list(operators) + list(candidates)
    qc, _ = builder.create_operator_list_circuit(n, probe_ops)
    _cost, _fid, grad_fn, _ = make_cost_fid(qc, H, psi)
    n_cur = len(operators)
    tc = np.asarray(theta_current, float)
    theta_probe = np.zeros(qc.num_parameters)
    theta_probe[:n_cur] = tc[:n_cur]  # candidates stay at 0 (symmetric → use state)
    g = np.asarray(grad_fn(theta_probe), float)
    return np.abs(g[n_cur : n_cur + len(candidates)])


def grow_adapt_operators(
    builder,
    n,
    lat,
    H,
    psi,
    e0,
    gap,
    *,
    nn_all,
    nnn_all,
    restarts,
    maxiter,
    seed0,
    target_fid,
    pool=None,
    include_rotations=True,
    grow_step=1,
    grad_tol=1e-6,
    max_ops=400,
    efficiency_patience=3,
    min_dfid_keep_growing=0.01,
    warm_theta0=None,
    backend=None,
    on_step=None,
):
    """Canonical ADAPT-VQE (Option B) — grow an ORDERED operator list, no layers.

    The literal "add whatever gate helps most, wherever" ansatz. Each step ranks
    the whole candidate ``pool`` (every RZZ edge + single-qubit rotations, via
    :func:`_default_operator_pool`) by ``|∂E/∂θ|`` at the current optimum
    (:func:`operator_pool_gradients`), appends the top ``grow_step`` operators to
    the running list, and re-converges warm from the previous optimum. There is NO
    block grammar and NO fixed p — the circuit is exactly the operators chosen.

    Objective-aligned stop (same contract as :func:`grow_adapt_pool`): stop at
    ``target_fid``; otherwise keep going while a step's absolute Δfidelity clears
    ``min_dfid_keep_growing`` OR fid-per-CX still improves, and stop on an
    ``efficiency_patience`` plateau. ``max_ops`` caps total operators.

    ``on_step(step_dict)`` fires after every converged step for crash-safe
    persistence. Returns ``(steps, reached)``; each step dict carries the same
    objective columns as the pool runner plus ``n_ops`` and ``last_op``.

    Reuses the project's optimizer (``converge_circuit``), gate convention
    (``create_operator_list_circuit``), and gradient trick — no new physics.
    """
    if pool is None:
        pool = _default_operator_pool(nn_all, nnn_all, n, include_rotations=include_rotations)

    def _converge(ops, warm):
        qc, _ = builder.create_operator_list_circuit(n, ops)
        n_2q, npar = cx_and_params(qc)
        warm_ok = warm if (warm is not None and len(warm) == npar) else None
        if warm_ok is not None and not np.all(np.isfinite(warm_ok)):
            warm_ok = None
        try:
            fid, e_best, runs = converge_circuit(
                qc, H, psi, restarts=restarts, maxiter=maxiter, seed0=seed0, warm_theta=warm_ok, backend=backend
            )
        except Exception:
            return float("-inf"), float("inf"), n_2q, npar, np.zeros(npar)
        theta = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)
        if not np.all(np.isfinite(theta)):
            return float("-inf"), float("inf"), n_2q, npar, np.zeros(npar)
        return fid, e_best, n_2q, npar, theta

    # Seed: start from the cheapest sensible ansatz — all nn RZZ once + a global
    # rx (a 1-layer-ish backbone), so the gradient signal is non-trivial at step 0.
    operators = [("rzz", (int(i), int(j))) for i, j in nn_all]
    if include_rotations:
        operators += [("rx", q) for q in range(n)]

    steps: list[dict] = []
    best_fpc = 0.0
    stale = 0
    reached = False

    def _record(it, fid, e_best, n_2q, npar, theta, last_op, seconds):
        fpc = (fid / n_2q) if n_2q > 0 else None
        s = {
            "step": it,
            "n_ops": len(operators),
            "last_op": last_op,
            "best_fidelity": fid,
            "e_best": e_best,
            "e0": e0,
            "gap": gap,
            "abs_error": abs(e_best - e0),
            "de_gap": abs(e_best - e0) / gap if gap > 0 else None,
            "n_2q_transpiled": n_2q,
            "n_params": npar,
            "fidelity_per_cx": fpc,
            "best_theta_final": theta.tolist(),
            "seconds": round(seconds, 1),
        }
        steps.append(s)
        if on_step is not None:
            try:
                on_step(s)
            except Exception:
                pass
        return s

    t0 = time.time()
    fid, e_best, n_2q, npar, theta = _converge(operators, warm_theta0)
    _record(0, fid, e_best, n_2q, npar, theta, "seed", time.time() - t0)
    if not np.isfinite(fid):
        steps[-1]["stop_reason"] = "seed_diverged"
        return steps, False
    if fid >= target_fid:
        steps[-1]["stop_reason"] = "target_fid"
        return steps, True

    for it in range(1, max_ops + 1):
        if len(operators) >= max_ops:
            steps[-1]["stop_reason"] = "max_ops"
            break
        # Candidate = pool minus operators already present (avoid exact dupes).
        present = set(operators)
        candidates = [op for op in pool if op not in present]
        if not candidates:
            steps[-1]["stop_reason"] = "pool_exhausted"
            break

        grads = operator_pool_gradients(builder, n, operators, candidates, H, psi, theta)
        if float(np.max(grads)) <= grad_tol:
            # Symmetry/degenerate point: perturb to break it, else stop.
            rng = np.random.default_rng(len(operators) + 24680)
            theta_pert = np.asarray(theta, float) + rng.normal(0.0, 0.05, len(theta))
            grads = operator_pool_gradients(builder, n, operators, candidates, H, psi, theta_pert)
            if float(np.max(grads)) <= grad_tol:
                steps[-1]["stop_reason"] = "grad_below_tol"
                break
        order = np.argsort(-grads)
        chosen = [candidates[int(idx)] for idx in order[:grow_step]]

        new_ops = list(operators) + chosen
        warm = np.concatenate([np.asarray(theta, float), np.zeros(len(chosen))])
        tcand = time.time()
        c_fid, c_e, c_2q, c_npar, c_theta = _converge(new_ops, warm)
        if not np.isfinite(c_fid):
            steps[-1]["stop_reason"] = "diverged"
            break
        dfid = c_fid - fid
        operators = new_ops
        fid, e_best, n_2q, npar, theta = c_fid, c_e, c_2q, c_npar, c_theta
        _record(it, fid, e_best, n_2q, npar, theta, f"{chosen[0][0]}{chosen[0][1]}", time.time() - tcand)

        if fid >= target_fid:
            reached = True
            steps[-1]["stop_reason"] = "target_fid"
            break
        fpc = (fid / n_2q) if n_2q > 0 else 0.0
        if (fpc > best_fpc + 1e-9 and dfid > 1e-6) or dfid >= min_dfid_keep_growing:
            best_fpc = max(best_fpc, fpc)
            stale = 0
        else:
            stale += 1
            if stale >= efficiency_patience:
                steps[-1]["stop_reason"] = "efficiency_plateau"
                break

    if steps and "stop_reason" not in steps[-1]:
        steps[-1]["stop_reason"] = "max_ops"
    return steps, reached


def analytic_seed_new_params(cost, warm, new_idx, *, eps=0.15, clip=np.pi):
    """Replace the 0-initialized NEW params with a 1D Newton estimate of their
    optimum, given the current state (improvement 2c).

    A freshly added RZZ/rotation starts at θ=0 (near-identity). Rather than let the
    optimizer discover its angle from scratch, estimate it by one Newton step along
    each new coordinate independently: with g = ∂E/∂θ and c = ∂²E/∂θ² (central
    finite differences on ``cost`` around the warm point), the energy-minimizing
    angle to first order is ``θ* = -g / c`` when ``c > 0`` (a convex direction),
    clipped to the box. A non-convex direction (``c <= 0``) keeps a small
    symmetry-breaking value so the optimizer still has a descent direction.

    ``warm`` is the warm-start vector (new params at 0); ``new_idx`` are the indices
    of the newly added params. Pure w.r.t. state (only evaluates ``cost``); returns
    a new warm vector. Costs ~3 cost() calls per new param — cheap (few new params).
    """
    w = np.asarray(warm, float).copy()
    for p in new_idx:
        base = w[p]
        wp = w.copy()
        wp[p] = base + eps
        wm = w.copy()
        wm[p] = base - eps
        e0 = cost(w)
        ep = cost(wp)
        em = cost(wm)
        g = (ep - em) / (2.0 * eps)
        c = (ep - 2.0 * e0 + em) / (eps * eps)
        if c > 1e-9:
            w[p] = float(np.clip(base - g / c, -clip, clip))
        else:
            # Non-convex / flat: nudge opposite the gradient to seed a descent.
            w[p] = float(np.clip(base - np.sign(g) * eps, -clip, clip))
    return w


def fidelity_bond_gradients(builder, n, lat, selection, candidate_nn, candidate_nnn, H, psi, theta_current, *, eps=0.1):
    """|∂Fidelity/∂θ| for each CANDIDATE bond, at the current optimized state (1a).

    The energy-gradient signal (:func:`bond_energy_gradients`) ranks candidates by
    how much they lower ⟨H⟩ — but near a small gap the energy is nearly flat while
    the fidelity is not, so energy is a poor proxy for the actual objective there.
    This ranks by the FIDELITY sensitivity instead: for each candidate bond, the
    central finite-difference of the exact ground-state overlap w.r.t. its (so far
    zero) angle, evaluated at the current optimum. Returns ``(grad_nn, grad_nnn)``
    aligned with ``candidate_nn`` / ``candidate_nnn``.

    Uses the same probe-circuit layout as :func:`bond_energy_gradients` (kept bonds
    carry their optimized θ, candidates start at 0). Costs ~2 fid() evals per
    candidate — cheap. Falls back to the shared cost/fid builder, no new physics.
    """
    from qmbp_simulation.circuits.bond_mask import BondSelection
    from qmbp_simulation.circuits.hva_variants import build_variant, make_masked_variant

    probe_sel = BondSelection(
        nn_edges=list(selection.nn_edges) + list(candidate_nn),
        nnn_edges=list(selection.nnn_edges) + list(candidate_nnn),
    )
    v = make_masked_variant("probe_fid", "fidelity gradient probe", ["nn", "nnn", "x"], probe_sel)
    qc, _ = build_variant(builder, n, lat, v)
    _cost, fid_fn, _grad, _ = make_cost_fid(qc, H, psi)

    n_nn_kept = len(selection.nn_edges)
    n_nn_cand = len(candidate_nn)
    n_nnn_kept = len(selection.nnn_edges)
    n_nnn_cand = len(candidate_nnn)
    tc = np.asarray(theta_current, float)
    theta0 = np.zeros(qc.num_parameters)
    theta0[:n_nn_kept] = tc[:n_nn_kept]
    p_off = n_nn_kept + n_nn_cand
    theta0[p_off : p_off + n_nnn_kept] = tc[n_nn_kept : n_nn_kept + n_nnn_kept]
    theta0[-n:] = tc[-n:]

    def _fid_grad(idx):
        tp = theta0.copy()
        tp[idx] += eps
        tm = theta0.copy()
        tm[idx] -= eps
        return abs(fid_fn(tp) - fid_fn(tm)) / (2.0 * eps)

    grad_nn = np.array([_fid_grad(n_nn_kept + k) for k in range(n_nn_cand)]) if n_nn_cand else np.array([])
    off = n_nn_kept + n_nn_cand + n_nnn_kept
    grad_nnn = np.array([_fid_grad(off + k) for k in range(n_nnn_cand)]) if n_nnn_cand else np.array([])
    return grad_nn, grad_nnn


def block_precondition_scales(
    blocks, n_nn_sel, n_nnn_sel, n_qubits, grad_at_warm, *, rx_final=False, rz_final=False, floor=0.1, cap=10.0
):
    """Per-parameter rescaling D so θ = D·φ makes the landscape more isotropic (2b).

    L-BFGS-B works in the Euclidean θ metric and treats every angle as equally
    scaled, but the blocks have very different sensitivities (θ_x soft, θ_zz
    rigid). We rescale each BLOCK by ``1/sqrt(mean|grad| over that block)`` so a
    stiff block (large gradient) takes smaller φ-steps and a soft block larger
    ones — a cheap diagonal, block-constant preconditioner derived from the
    gradient already computed at the warm point (no extra Hessian).

    Returns a length-``n_params`` vector ``D`` (same block layout the masked
    builder uses). Scales are normalized to geometric-mean 1 (so the overall step
    size is unchanged, only the per-block balance), then clipped to
    ``[floor, cap]`` for numerical safety. Pure.
    """
    g = np.abs(np.asarray(grad_at_warm, float))
    size = {"nn": n_nn_sel, "nnn": n_nnn_sel, "x": n_qubits, "z": n_qubits}
    spans = []
    off = 0
    for b in blocks:
        spans.append((b, off, off + size[b]))
        off += size[b]
    if rx_final:
        spans.append(("x", off, off + n_qubits))
        off += n_qubits
    if rz_final:
        spans.append(("z", off, off + n_qubits))
        off += n_qubits

    D = np.ones(len(g))
    block_scale = {}
    for b, a, z in spans:
        gm = float(np.mean(g[a:z])) if z > a else 0.0
        s = 1.0 / np.sqrt(gm) if gm > 1e-9 else 1.0
        block_scale.setdefault(b, []).append(s)
    # One scale per block TYPE (average across its instances), geo-mean-normalized.
    type_scale = {b: float(np.exp(np.mean(np.log(np.clip(v, 1e-9, None))))) for b, v in block_scale.items()}
    vals = np.array(list(type_scale.values()), float)
    gnorm = float(np.exp(np.mean(np.log(np.clip(vals, 1e-9, None))))) if vals.size else 1.0
    for b, a, z in spans:
        D[a:z] = np.clip(type_scale[b] / gnorm, floor, cap)
    return D


def converge_circuit_preconditioned(
    qc, H, psi, *, restarts, maxiter, seed0=0, warm_theta=None, D=None, on_restart=None, backend=None
):
    """``converge_circuit`` in rescaled coordinates θ = D·φ (block preconditioner).

    Optimizes in φ-space where the per-block preconditioner ``D`` has balanced the
    landscape, then maps the result back to θ. When ``D`` is None (or all ones)
    this is byte-for-byte :func:`converge_circuit`. The optimizer sees
    ``cost(D·φ)`` with jac ``D·grad(D·φ)`` (chain rule), and the warm-start / box
    are mapped through ``D`` consistently. Returns ``(best_fid, best_e, runs)``
    with θ-space θ_final in each run (so downstream code is unchanged).
    """
    cost, fid, grad, _ = make_cost_fid(qc, H, psi, backend=backend)
    npar = qc.num_parameters
    if D is None:
        D = np.ones(npar)
    D = np.asarray(D, float)
    if D.shape[0] != npar or np.allclose(D, 1.0):
        return optimize_bestof(
            cost,
            fid,
            grad,
            npar,
            restarts=restarts,
            maxiter=maxiter,
            seed0=seed0,
            warm_theta=warm_theta,
            on_restart=on_restart,
        )

    def cost_phi(phi):
        return cost(D * np.asarray(phi, float))

    def grad_phi(phi):
        return D * np.asarray(grad(D * np.asarray(phi, float)), float)

    def fid_phi(phi):
        return fid(D * np.asarray(phi, float))

    cost_phi.cost_fid = None  # not needed in φ-space
    warm_phi = (np.asarray(warm_theta, float) / D) if warm_theta is not None else None
    best_fid, best_e, runs_phi = optimize_bestof(
        cost_phi,
        fid_phi,
        grad_phi,
        npar,
        restarts=restarts,
        maxiter=maxiter,
        seed0=seed0,
        warm_theta=warm_phi,
        on_restart=on_restart,
    )
    # Map every run's θ back to θ-space so callers see standard θ_final.
    for r in runs_phi:
        r["theta_final"] = (D * np.asarray(r["theta_final"], float)).tolist()
        r["theta_init"] = (D * np.asarray(r["theta_init"], float)).tolist()
    return best_fid, best_e, runs_phi
