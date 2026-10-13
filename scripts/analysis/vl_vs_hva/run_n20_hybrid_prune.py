#!/usr/bin/env python
"""N=20 aggressive re-prune of the diag_axial50 hybrid, with ad-hoc warm-start.

diag_axial50 reached fid 0.979 @ 326 2q (35 nnn). Its converged θ shows HALF the
nnn are near-dead (|θ|<0.1 on 22/35). So two things are possible and tested here:

Q1 (ad-hoc warm-start): instead of the generic analytic cascade, seed each pruned
   variant by TRANSFERRING diag_axial50's own converged θ onto the shared bonds
   (transfer_theta_for_blocks). That is the most informed seed available — same
   structure/N/h, just fewer bonds — so convergence starts in the right basin.

Q2 (fewer gates > 0.9 fid): keep only the top-k nnn by |θ| of diag_axial50 and
   converge. Sweep k so we find the smallest gate count still clearing fid 0.9.

Reuses: AnsatzSpec, bond_weights_for_blocks, transfer_theta_for_blocks (the core
cross-selection warm-start), converge_circuit, the subspace-fidelity + phys_gap
degenerate handling, CachedBackend, spec persistence.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/run_n20_hybrid_prune.py \
        --n 20 --h 0.5 --keep 20 13 10 8 --restarts 1 --maxiter 400
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

from qmbp_simulation.analysis.warmstart import transfer_theta_for_blocks  # noqa: E402
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.ansatz_spec import AnsatzSpec  # noqa: E402
from qmbp_simulation.circuits.bond_mask import BondSelection, bond_weights_for_blocks  # noqa: E402
from qmbp_simulation.circuits.hva_variants import build_variant, make_masked_variant  # noqa: E402
from qmbp_simulation.framework.study_core import (  # noqa: E402
    converge_circuit,
    cx_and_params,
    ground_state,
    prepare_warmstart,
)

SUBDIR = "bond_ablation"
MODEL = "tfim_frustrated"
SPEC_MIN_FID = 0.85
DONOR_NAME = "diag_axial50"


# Degenerate-manifold detection + subspace fidelity now live in the src helper
# (qmbp_simulation.analysis.fidelity) — shared by every study runner. Aliased to
# the original local names so the call sites below are unchanged.
from qmbp_simulation.analysis.fidelity import (  # noqa: E402
    degenerate_ground_manifold as _degenerate_ground_space,
    subspace_fidelity as _subspace_fidelity,
)


def run(args) -> int:
    from qmbp_simulation.execution import NoiselessBackend
    from qmbp_simulation.execution.eval_cache import CachedBackend, EvalCache

    donor = AnsatzSpec.load(
        str(Path("results/hva_vl_study/ansatz_specs") /
            f"{DONOR_NAME}_{args.topology}_N{args.n}_h{args.h:.2f}.spec.json"))
    theta_d = np.asarray(donor.theta, float)
    blocks = donor.blocks
    nn_edges = [tuple(e) for e in donor.nn_edges]
    nnn_edges_d = [tuple(e) for e in donor.nnn_edges]
    w_nn, w_nnn = bond_weights_for_blocks(theta_d, blocks, len(nn_edges), len(nnn_edges_d),
                                          args.n, rx_final=donor.rx_final, rz_final=donor.rz_final)
    order = np.argsort(-np.asarray(w_nnn))  # strongest nnn first

    builder = HVACircuitBuilder()
    backend = CachedBackend(NoiselessBackend(), topology=args.topology, n_qubits=args.n,
                            model=MODEL, p_layers=1, cache=EvalCache())
    backend.set_h(float(args.h))
    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(args.topology, args.n, args.h, args.j2, 1)
    manifold, e0_m, phys_gap, m_dim = _degenerate_ground_space(H)
    degenerate = m_dim > 1
    print(f"[hybrid_prune] N={args.n} h={args.h} donor={DONOR_NAME}(fid {donor.fidelity:.4f}, "
          f"35 nnn, 326 2q) phys_gap={phys_gap:.4f} manifold={m_dim} | keep={args.keep}", flush=True)

    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR, out_file=f"n20_hybrid_prune_{args.topology}_N{args.n}_h{args.h:.2f}.json",
        fingerprint={"n_qubits": args.n, "p_layers": None, "model": MODEL,
                     "topology": args.topology, "h": args.h},
        extra={"topology": args.topology, "N": args.n, "h": args.h, "J2": args.j2,
               "donor": DONOR_NAME, "schema": "n20_hybrid_prune_v1"},
        params={"experiment": "n20_hybrid_prune", "N": args.n, "h": args.h},
        description="N=20 aggressive re-prune of diag_axial50 with ad-hoc θ-transfer warm-start",
        rows_ref=rows)

    for keep in args.keep:
        keep = min(keep, len(nnn_edges_d))
        sel_nnn = [nnn_edges_d[i] for i in order[:keep]]
        sel = BondSelection(nn_edges=nn_edges, nnn_edges=sel_nnn)
        v = make_masked_variant(f"{DONOR_NAME}_top{keep}", f"re-prune top-{keep} nnn", blocks, sel)
        qc, _ = build_variant(builder, args.n, lat, v)
        n_2q, npar = cx_and_params(qc)

        # Ad-hoc warm-start: transfer the donor's converged θ onto this pruned
        # layout (shared bonds keep their angle; the rest regime-filled). This is
        # Q1 — the most informed seed. If the transfer length mismatches, fall back
        # to the analytic cascade with auto_donors (which now sees the donor spec).
        seed = transfer_theta_for_blocks(
            theta_d, list(blocks), donor_nnn_edges=nnn_edges_d, target_nnn_edges=sel_nnn,
            n_nn=len(nn_edges), n_qubits=args.n, rx_final=donor.rx_final, rz_final=donor.rz_final)
        seed_kind = "transfer<diag_axial50>"
        if seed is None or np.asarray(seed).size != npar:
            ws = prepare_warmstart(qc, H, psi, n_nn=len(nn_edges), n_nnn=keep, n_qubits=args.n,
                                   p_layers=1, h=args.h, J2=args.j2, backend=backend,
                                   topology=args.topology, model=MODEL, gap=phys_gap,
                                   target_len=npar, auto_donors=True)
            seed, seed_kind = np.asarray(ws["seed"], float), ws.get("provenance")

        # Report the seed's pre-optimization subspace fidelity: this is the direct
        # evidence of whether the transfer warm-start lands in the right basin.
        init_sub = _subspace_fidelity(qc, np.asarray(seed, float), manifold) if degenerate else None

        t0, c0 = time.time(), time.process_time()
        fid, e_best, runs = converge_circuit(qc, H, psi, restarts=args.restarts,
                                             maxiter=args.maxiter, seed0=args.seed0,
                                             warm_theta=seed, backend=backend,
                                             target_fid=args.target_fid, ftol=args.ftol)
        theta_best = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)
        sub = _subspace_fidelity(qc, theta_best, manifold) if degenerate else None
        report = sub if degenerate else fid
        row = {
            "label": f"{DONOR_NAME}_top{keep}", "h": round(float(args.h), 2), "n_qubits": args.n,
            "topology": args.topology, "blocks": blocks, "n_nn": len(nn_edges), "n_nnn": keep,
            "e0": e0, "phys_gap": phys_gap, "degenerate": degenerate, "e_best": float(e_best),
            "de_gap": abs(float(e_best) - e0) / phys_gap if phys_gap > 0 else None,
            "best_fidelity": float(fid), "subspace_fidelity": sub, "report_fidelity": float(report),
            "n_2q_transpiled": n_2q, "n_params": npar,
            "fidelity_per_cx": (report / n_2q) if n_2q > 0 else None,
            "seed_kind": seed_kind, "init_subspace_fidelity": init_sub,
            "seconds": round(time.time() - t0, 1), "cpu_seconds": round(time.process_time() - c0, 1),
        }
        rows.append(row)
        if float(report) >= SPEC_MIN_FID:
            try:
                save_ansatz_spec(v, topology=args.topology, n_qubits=args.n, h=args.h, j2=args.j2,
                                 theta=theta_best.tolist(), nn_edges=nn_edges, nnn_edges=sel_nnn,
                                 fidelity=float(report), n_2q=n_2q, n_params=npar, p_layers=1)
            except Exception as exc:
                print(f"    [spec save skipped: {exc}]", flush=True)
        persister.persist(rows)
        flag = "✓>0.9" if report > 0.9 else ""
        init_str = f"init={init_sub:.3f}→" if init_sub is not None else ""
        print(f"  top{keep:2d} nnn: {init_str}fid={report:.4f} 2q={n_2q} nnn={keep} "
              f"fid/cx={row['fidelity_per_cx']:.3e} seed={seed_kind[:22]} {flag} ({row['cpu_seconds']}s)",
              flush=True)

    persister.persist(rows, status="final")
    backend.flush()
    print("\n=== N20 HYBRID RE-PRUNE (report_fid | 2q | fid/cx) ===", flush=True)
    for r in sorted(rows, key=lambda x: x["n_2q_transpiled"]):
        over = "✓" if r["report_fidelity"] > 0.9 else " "
        print(f"  {over} {r['label']:22s} fid={r['report_fidelity']:.4f} 2q={r['n_2q_transpiled']:3d} "
              f"fid/cx={(r['fidelity_per_cx'] or 0):.3e}", flush=True)
    print("DONE", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="N=20 aggressive re-prune of diag_axial50 with ad-hoc warm-start")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--keep", type=int, nargs="+", default=[20, 13, 10, 8],
                   help="nnn keep-counts to sweep (top-k by donor |θ|).")
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--maxiter", type=int, default=400)
    p.add_argument("--seed0", type=int, default=70000)
    p.add_argument("--target-fid", type=float, default=None,
                   help="early-stop: skip remaining restarts once a restart clears this fidelity.")
    p.add_argument("--ftol", type=float, default=1e-9,
                   help="L-BFGS-B relative tolerance. Looser than the 1e-12 default so a good "
                        "transfer warm-start converges in fewer iterations at N=20.")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
