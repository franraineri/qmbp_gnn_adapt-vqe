#!/usr/bin/env python
"""N=20 small-ansatz fidelity/2q sweep — the viable subset at dim 2^20.

The full p2/p3 ansatze (265+ params, ~11s/grad) are computationally out of reach
at N=20 on a dense statevector. This runner converges only SMALL ansatze (few
params, few 2q gates) that converge in minutes, to answer "what fidelity, at what
2q cost, can we reach for the N=20 h=1.0 energy" with a real measurement.

Each ansatz is one warm-started best-of (restarts configurable, default 1 — the
h=1.0 regime has a large gap so the analytic seed converges without random
restarts). Reuses ground_state, prepare_warmstart, converge_circuit, cx_and_params
and the masked-variant builder — no new physics. Crash-safe partial persist.

Ansatze swept (increasing cost):
  nn1     : 1 entangling layer, nn bonds only            (backbone floor)
  nn2     : 2 nn layers                                   (depth, nn only)
  nnn1    : 1 layer nn+nnn                                (frustration bonds)
  nnn1_half : 1 layer nn+nnn + extra half nn + rx         (half_nn_rx structure)

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/run_n20_small_ansatze.py \
        --n 20 --h 1.0 --restarts 1 --maxiter 400
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import StudyPersister, save_ansatz_spec  # noqa: E402

from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.bond_mask import BondSelection  # noqa: E402
from qmbp_simulation.circuits.hva_variants import build_variant, make_masked_variant  # noqa: E402
from qmbp_simulation.framework.study_core import (  # noqa: E402
    converge_circuit,
    cx_and_params,
    ground_state,
    prepare_warmstart,
)
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402

SUBDIR = "bond_ablation"
MODEL = "tfim_frustrated"
SPEC_MIN_FID = 0.85  # only persist a converged ansatz as a reusable donor if decent


def _ansatze(nn_edges, nnn_edges):
    """(label, blocks, BondSelection) for each small ansatz, cheap first."""
    nn_only = BondSelection(nn_edges=nn_edges, nnn_edges=[])
    nn_full = BondSelection(nn_edges=nn_edges, nnn_edges=nnn_edges)
    small = [
        ("nn1", ["nn", "x"], nn_only),
        ("nn2", ["nn", "x", "nn", "x"], nn_only),
        ("nnn1", ["nn", "nnn", "x"], nn_full),
        ("nnn1_half", ["nn", "nnn", "x", "nn", "x"], nn_full),
    ]
    # Extended: push nnn1_half (best small, 0.714) deeper to see if fidelity rises
    # past ~0.71 at the cost of more 2q. Two more entangling layers on the
    # frustration bonds.
    extended = [
        ("nnn2", ["nn", "nnn", "x", "nn", "nnn", "x"], nn_full),
        ("nnn2_half", ["nn", "nnn", "x", "nn", "nnn", "x", "nn", "x"], nn_full),
    ]
    return small, extended


def _degenerate_ground_space(H, k=4):
    """Lowest-k eigenpairs; detect a near-degenerate ground doublet/manifold.

    Returns ``(evecs_manifold, e0, phys_gap, manifold_dim)`` where the manifold is
    the set of states within ``DOUBLET_TOL`` of E0 (a near-degenerate ground space
    at a first-order point), and ``phys_gap`` is the gap to the first state ABOVE
    the manifold — the physically meaningful gap for ΔE/gap when E0 is degenerate.
    Reuses the same sparse eigsh as study_core.ground_state, just with more levels.
    """
    from scipy.sparse.linalg import eigsh

    DOUBLET_TOL = 1e-3
    ev, evec = eigsh(H.to_matrix(sparse=True), k=k, which="SA")
    order = np.argsort(ev)
    ev = ev[order]
    evec = evec[:, order]
    e0 = float(ev[0])
    in_manifold = [i for i in range(len(ev)) if ev[i] - e0 <= DOUBLET_TOL]
    m = len(in_manifold)
    above = [i for i in range(len(ev)) if i not in in_manifold]
    phys_gap = float(ev[above[0]] - e0) if above else float("nan")
    manifold = evec[:, in_manifold].astype(complex)
    return manifold, e0, phys_gap, m


def _subspace_fidelity(circuit, theta, manifold):
    """Σ_k |⟨E_k|ψ(θ)⟩|² over the degenerate ground manifold — the right fidelity
    target when the ground state is a near-degenerate doublet (a single-vector
    fidelity to an arbitrary eigsh combination is meaningless there)."""
    from qiskit.quantum_info import Statevector

    psi = Statevector(circuit.assign_parameters(theta)).data
    overlaps = manifold.conj().T @ psi  # (m,)
    return float(np.clip(np.sum(np.abs(overlaps) ** 2), 0.0, 1.0))


def _point(args, builder, backend, persister, rows):
    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(
        args.topology, args.n, args.h, args.j2, 1)
    nn_edges = list(lat.edges)
    nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
    # Degenerate-manifold handling: when the ground state is a near-degenerate
    # doublet (small-gap, first-order point like N=20 h=0.5, gap~7e-5), measure
    # fidelity to the WHOLE ground space and report the physical gap (to the first
    # state above the manifold). For a well-separated GS this is a no-op (m=1).
    manifold, e0_m, phys_gap, m_dim = _degenerate_ground_space(H)
    degenerate = m_dim > 1
    print(f"[n20_small] N={args.n} h={args.h} e0={e0:.5f} gap={gap:.5f} "
          f"n_nn={n_nn} n_nnn={n_nnn} restarts={args.restarts} maxiter={args.maxiter}",
          flush=True)

    small, extended = _ansatze(nn_edges, nnn_edges)
    chosen = {"small": small, "extended": extended, "all": small + extended}[args.set]
    for label, blocks, sel in chosen:
        v = make_masked_variant(label, f"N20 {label}", blocks, sel)
        qc, _ = build_variant(builder, args.n, lat, v)
        n_2q, npar = cx_and_params(qc)
        ws = prepare_warmstart(qc, H, psi, n_nn=len(sel.nn_edges),
                               n_nnn=len(sel.nnn_edges), n_qubits=args.n,
                               p_layers=1, h=args.h, J2=args.j2, backend=backend,
                               topology=args.topology, model=MODEL, gap=gap,
                               target_len=qc.num_parameters, auto_donors=True)
        seed = np.asarray(ws["seed"], float)
        if seed.size != npar:
            fit = np.zeros(npar); fit[:min(seed.size, npar)] = seed[:min(seed.size, npar)]
            seed = fit
        t0 = time.time()
        c0 = time.process_time()  # CPU-time: robust to wall-clock inflation under CPU contention
        fid, e_best, runs = converge_circuit(
            qc, H, psi, restarts=args.restarts, maxiter=args.maxiter,
            seed0=args.seed0, warm_theta=seed, backend=backend)
        # θ of the energy-best restart, for the subspace fidelity.
        theta_best = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)
        sub_fid = _subspace_fidelity(qc, theta_best, manifold) if degenerate else None
        # Report fid/CX on the physically meaningful fidelity: subspace fid in the
        # degenerate case, single-vector fid otherwise.
        report_fid = sub_fid if degenerate else fid
        fpc = (report_fid / n_2q) if n_2q > 0 else None
        de_gap = abs(float(e_best) - e0) / phys_gap if (phys_gap and phys_gap > 0) else None
        row = {
            "label": label, "h": round(float(args.h), 2), "n_qubits": args.n,
            "topology": args.topology, "blocks": blocks,
            "e0": e0, "gap": gap, "phys_gap": phys_gap, "manifold_dim": m_dim,
            "degenerate": degenerate, "e_best": float(e_best),
            "abs_error": abs(float(e_best) - e0),
            "de_gap": de_gap,
            "best_fidelity": float(fid), "subspace_fidelity": sub_fid,
            "report_fidelity": float(report_fid),
            "n_2q_transpiled": n_2q, "n_params": npar,
            "fidelity_per_cx": fpc, "seed_kind": ws.get("provenance"),
            "seconds": round(time.time() - t0, 1),
            "cpu_seconds": round(time.process_time() - c0, 1),
        }
        rows.append(row)
        # Persist a reusable spec of this converged ansatz so future warm-starts
        # (auto_donors) can transfer from it — closes the self-reinforcing loop.
        # Only save decent fits (≥ SPEC_MIN_FID): a low-fidelity θ would be noise
        # as a donor and discover_donors filters it out anyway.
        if float(report_fid) >= SPEC_MIN_FID:
            try:
                save_ansatz_spec(v, topology=args.topology, n_qubits=args.n, h=args.h,
                                 j2=args.j2, theta=theta_best.tolist(),
                                 nn_edges=sel.nn_edges, nnn_edges=sel.nnn_edges,
                                 fidelity=float(report_fid), n_2q=n_2q, n_params=npar,
                                 p_layers=1)
            except Exception as exc:  # spec persistence is best-effort
                print(f"    [spec save skipped: {exc}]", flush=True)
        persister.persist(rows)
        sub_s = f" sub_fid={sub_fid:.4f}" if sub_fid is not None else ""
        deg_s = f" ΔE/gap={de_gap:.3f}" if de_gap is not None else " ΔE/gap=n/a"
        print(f"  {label:10s} fid={fid:.4f}{sub_s} 2q={n_2q:3d} params={npar:3d} "
              f"fid/cx={fpc:.3e}{deg_s} ({row['cpu_seconds']}s cpu / {row['seconds']}s wall)",
              flush=True)


def run(args) -> int:
    from qmbp_simulation.execution import NoiselessBackend
    from qmbp_simulation.execution.eval_cache import CachedBackend, EvalCache

    builder = HVACircuitBuilder()
    # Persistent evaluation cache: wrap the dense backend so repeated circuit
    # energy evaluations (across restarts, across re-runs of the same N/h/model)
    # are reused instead of recomputed — the big win at N=20 (~2-11s/eval).
    backend = CachedBackend(NoiselessBackend(), topology=args.topology, n_qubits=args.n,
                            model=MODEL, p_layers=1, cache=EvalCache())
    backend.set_h(float(args.h))
    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR, out_file=f"n20_{args.set}_ansatze_{args.topology}_N{args.n}_h{args.h:.2f}.json",
        fingerprint={"n_qubits": args.n, "p_layers": None, "model": MODEL,
                     "topology": args.topology, "h": args.h},
        extra={"topology": args.topology, "N": args.n, "h": args.h, "J2": args.j2,
               "schema": "n20_small_ansatze_v1"},
        params={"experiment": "n20_small_ansatze", "N": args.n, "h": args.h},
        description="N=20 small-ansatz fidelity/2q sweep (viable subset at dim 2^20)",
        rows_ref=rows)
    _point(args, builder, backend, persister, rows)
    persister.persist(rows, status="final")
    backend.flush()  # persist the evaluation cache to disk for future re-runs
    print("\n=== N20 ANSATZE (report_fid | 2q | fid/cx) ===", flush=True)
    for r in sorted(rows, key=lambda x: x["n_2q_transpiled"]):
        rf = r.get("report_fidelity", r["best_fidelity"])
        print(f"  {r['label']:10s} fid={rf:.4f} 2q={r['n_2q_transpiled']:3d} "
              f"fid/cx={(r['fidelity_per_cx'] or 0):.3e}", flush=True)
    print("DONE", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="N=20 small-ansatz fidelity/2q sweep")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--h", type=float, default=1.0)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--maxiter", type=int, default=400)
    p.add_argument("--seed0", type=int, default=70000)
    p.add_argument("--set", choices=["small", "extended", "all"], default="small",
                   help="Which ansatz set: small (nn1..nnn1_half), extended "
                        "(deeper nnn2/nnn2_half), or all.")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
