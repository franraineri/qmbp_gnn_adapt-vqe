#!/usr/bin/env python
"""ADAPT-style bond growth: build the cheapest masked HVA that hits a fidelity target.

Instead of fixing the ansatz structure a priori, GROW the set of RZZ bonds one
(or a few) at a time, guided by the energy gradient — the ADAPT-VQE idea applied
to bond selection. Each iteration:

  1. Optimize the current masked ansatz (``study_core.optimize_bestof``).
  2. Rank the NOT-yet-included bonds by ``|∂E/∂θ_bond|`` at the current state
     (:func:`bond_mask.rank_by_gradient`) — the operator that most lowers the
     energy is added next.
  3. Add the top ``--grow-step`` bond(s) to the :class:`BondSelection`, rebuild
     via ``create_bond_resolved_masked``, and repeat.

Stops when fidelity ≥ ``--target-fid``, no candidate bond has gradient above
``--grad-tol``, or all bonds are included (== the full layer). The result is a
fidelity-vs-2q growth curve showing the cheapest circuit reaching the target.

Reuses (no new physics/optimizer): ``study_core`` (ground_state, make_cost_fid,
optimize_bestof, cx_and_params), ``bond_mask`` (BondSelection, rank_by_gradient),
``hva_variants`` (make_masked_variant, build_variant), and ``save_json`` into the
dedicated ``bond_ablation`` subfolder (schema ``adapt_bonds_v1``).

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/run_adapt_bonds.py \
        --n 10 --h 1.0 --target-fid 0.90 --grow-step 2 --restarts 2 --maxiter 1000
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

from hva_vl_study_common import save_json  # noqa: E402

from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.bond_mask import BondSelection, rank_by_gradient  # noqa: E402
from qmbp_simulation.circuits.hva_variants import build_variant, make_masked_variant  # noqa: E402
from qmbp_simulation.framework.study_checkpoint import build_resumable_payload  # noqa: E402
from qmbp_simulation.framework.study_core import (  # noqa: E402
    cx_and_params,
    ground_state,
    make_cost_fid,
    optimize_bestof,
)
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402

SUBDIR = "bond_ablation"


def _bond_gradients(builder, n, lat, selection, candidate_nn, candidate_nnn, H, psi,
                    theta_current):
    """|∂E/∂θ| for each CANDIDATE bond, evaluated at the CURRENT OPTIMIZED state.

    Builds a probe circuit = current selection + candidate bonds appended, then
    takes the adjoint gradient of ⟨H⟩ at the working point: the kept bonds and X
    rotations carry their optimized values (from ``theta_current``), the
    candidate θ start at 0. Evaluating at the OPTIMIZED state (not θ=0) is
    essential — at the symmetric θ=0 point the RZZ gradient vanishes by symmetry
    (the classic ADAPT "zero initial gradient" trap), which would stall growth.

    Returns ``(grad_nn, grad_nnn)`` aligned with ``candidate_nn`` / ``candidate_nnn``.
    """
    probe_sel = BondSelection(
        nn_edges=list(selection.nn_edges) + list(candidate_nn),
        nnn_edges=list(selection.nnn_edges) + list(candidate_nnn),
    )
    v = make_masked_variant("probe", "adapt gradient probe", ["nn", "nnn", "x"],
                            probe_sel)
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
    # nn_kept: copy optimized nn angles
    theta_probe[:n_nn_kept] = tc[:n_nn_kept]
    # nnn_kept: sits after nn_kept+nn_cand in the probe, after nn_kept in current
    p_off = n_nn_kept + n_nn_cand
    theta_probe[p_off:p_off + n_nnn_kept] = tc[n_nn_kept:n_nn_kept + n_nnn_kept]
    # x block: last n qubits in both layouts
    theta_probe[-n:] = tc[-n:]

    g = np.asarray(grad_fn(theta_probe), float)

    grad_nn = np.abs(g[n_nn_kept:n_nn_kept + n_nn_cand]) if n_nn_cand else np.array([])
    off = n_nn_kept + n_nn_cand + n_nnn_kept
    grad_nnn = np.abs(g[off:off + n_nnn_cand]) if n_nnn_cand else np.array([])
    return grad_nn, grad_nnn


def run(args) -> int:
    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(
        args.topology, args.n, args.h, args.j2, 1)
    nn_all = list(lat.edges)
    nnn_all = HamiltonianBuilder._generate_nnn_edges(lat)
    builder = HVACircuitBuilder()
    out_file = f"adapt_bonds_{args.topology}_N{args.n}_h{args.h:.2f}.json"

    print(f"[adapt_bonds] N={args.n} h={args.h} e0={e0:.5f} gap={gap:.5f} "
          f"n_nn={n_nn} n_nnn={n_nnn} target_fid={args.target_fid} "
          f"grow_step={args.grow_step}", flush=True)

    # Seed selection: all nn bonds (cheap backbone), no nnn yet. ADAPT grows nnn.
    selection = BondSelection(nn_edges=list(nn_all), nnn_edges=[],
                              provenance="adapt_seed(nn_only)")
    steps: list[dict] = []

    def _persist(status="partial"):
        payload = build_resumable_payload(
            rows=steps, theta=(steps[-1]["best_theta_final"] if steps else None),
            fingerprint={"n_qubits": args.n, "p_layers": 1,
                         "model": "tfim_frustrated", "topology": args.topology,
                         "h": args.h},
            extra={"topology": args.topology, "N": args.n, "h": args.h,
                   "p_layers": 1, "J2": args.j2, "e0": e0, "gap": gap,
                   "target_fid": args.target_fid, "grow_step": args.grow_step,
                   "grad_tol": args.grad_tol, "n_nnn_total": n_nnn,
                   "schema": "adapt_bonds_v1", "status": status},
        )
        save_json(payload, SUBDIR, out_file,
                  params={"experiment": "adapt_bonds", "N": args.n, "h": args.h},
                  description="ADAPT bond growth — cheapest masked HVA reaching a "
                              "fidelity target (gradient-guided nnn insertion)")

    reached = False
    for it in range(n_nnn + 1):  # at most: add every nnn bond
        # Optimize current selection.
        v = make_masked_variant(
            f"adapt_step{it}", f"adapt step {it} ({selection.n_bonds} bonds)",
            ["nn", "nnn", "x"], selection, tags=("adapt",))
        qc, _ = build_variant(builder, args.n, lat, v)
        n_2q, npar = cx_and_params(qc)
        cost, fid_fn, grad_fn, _ = make_cost_fid(qc, H, psi)
        t0 = time.time()
        fid, e_best, runs = optimize_bestof(
            cost, fid_fn, grad_fn, npar, restarts=args.restarts,
            maxiter=args.maxiter, seed0=args.seed0)
        best_run = min(runs, key=lambda r: r["energy"])
        step = {
            "step": it, "n_nn_bonds": len(selection.nn_edges),
            "n_nnn_bonds": len(selection.nnn_edges), "n_bonds": selection.n_bonds,
            "best_fidelity": fid, "e_best": e_best, "e0": e0, "gap": gap,
            "abs_error": abs(e_best - e0),
            "de_gap": abs(e_best - e0) / gap if gap > 0 else None,
            "n_2q_transpiled": n_2q, "n_params": npar,
            "fidelity_per_cx": (fid / n_2q) if n_2q > 0 else None,
            "best_theta_final": best_run["theta_final"],
            "seconds": round(time.time() - t0, 1),
        }
        steps.append(step)
        _persist()
        print(f"  step{it:>2} bonds={selection.n_bonds:>3} (nnn={len(selection.nnn_edges)}) "
              f"fid={fid:.4f} 2q={n_2q} fid/cx={step['fidelity_per_cx']:.2e} "
              f"({step['seconds']}s)", flush=True)

        if fid >= args.target_fid:
            reached = True
            print(f"  ✓ target fidelity {args.target_fid} reached at "
                  f"{selection.n_bonds} bonds ({n_2q} CX)", flush=True)
            break

        # Candidate nnn bonds not yet selected.
        remaining = [e for e in nnn_all if e not in set(selection.nnn_edges)]
        if not remaining:
            print("  all nnn bonds included — reached the full layer", flush=True)
            break

        # Rank remaining bonds by |∂E/∂θ| at the current OPTIMIZED state.
        theta_current = np.asarray(best_run["theta_final"], float)
        _gnn, g_nnn = _bond_gradients(builder, args.n, lat, selection, [], remaining,
                                      H, psi, theta_current)
        ranked = rank_by_gradient([], remaining, np.array([]), g_nnn)
        top = [edge for _kind, edge, score in ranked if score > args.grad_tol][:args.grow_step]
        if not top:
            print(f"  no candidate bond exceeds grad_tol={args.grad_tol} — stop", flush=True)
            break
        selection = BondSelection(
            nn_edges=list(selection.nn_edges),
            nnn_edges=list(selection.nnn_edges) + top,
            provenance=f"adapt(step{it + 1}, +{len(top)} nnn by grad)")

    _persist(status="final")

    print("\n=== ADAPT growth curve (fid | 2q | fid/cx) ===", flush=True)
    for s in steps:
        print(f"  step{s['step']:>2} bonds={s['n_bonds']:>3} fid={s['best_fidelity']:.4f} "
              f"2q={s['n_2q_transpiled']:>4} fid/cx={s['fidelity_per_cx']:.2e}", flush=True)
    tag = "TARGET REACHED" if reached else "target not reached"
    print(f"→ {tag}. DONE", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ADAPT bond growth (masked HVA)")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--h", type=float, default=1.0)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--maxiter", type=int, default=1000)
    p.add_argument("--seed0", type=int, default=70000)
    p.add_argument("--target-fid", type=float, default=0.90,
                   help="Stop once fidelity reaches this.")
    p.add_argument("--grow-step", type=int, default=2,
                   help="How many top-gradient bonds to add per iteration.")
    p.add_argument("--grad-tol", type=float, default=1e-6,
                   help="Ignore candidate bonds with |grad| below this.")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
