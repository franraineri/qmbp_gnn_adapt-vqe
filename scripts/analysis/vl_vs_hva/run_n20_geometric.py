#!/usr/bin/env python
"""N=20 geometry-aware ansatz sweep — exploit the commensurate 4×5 grid.

N=20 forms a PERFECT 4×5 square grid (unlike N=18's incomplete grid). On that
regular lattice the J2 frustration is carried by the DIAGONAL next-nearest bonds
(grid displacement (1,±1)); the AXIAL nnn (displacement (0,2)/(2,0)) are a
different, less frustration-relevant coupling. This runner tests whether keeping
ONLY the frustration diagonals — ~half the nnn — reaches the full-nnn fidelity at
far fewer 2q gates (the fid-per-CX win), plus a hybrid that adds back the strongest
axial nnn by |θ|.

Reuses everything: study_core.ground_state, the masked-variant builder, the
subspace-fidelity + physical-gap handling for the degenerate regime, the
auto_donors warm-start cascade, CachedBackend, and spec persistence.

Ansatze (structure = p2_half_nn: [nn,nnn,x]×2 + nn,x), differing only in which nnn:
  full       : all 46 nnn               (the 0.965 @ 370 2q reference)
  diag_only  : 24 frustration diagonals (1,±1)        → ~282 2q
  diag_axial50: diagonals + strongest 50% axial by |θ| of the full run

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/run_n20_geometric.py \
        --n 20 --h 0.5 --restarts 1 --maxiter 400
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
from qmbp_simulation.circuits.bond_mask import BondSelection, bond_weights_for_blocks  # noqa: E402
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
SPEC_MIN_FID = 0.85
BLOCKS = ["nn", "nnn", "x", "nn", "nnn", "x", "nn", "x"]  # p2_half_nn structure


def _grid_cols(n):
    return int(np.ceil(np.sqrt(n)))


def _classify_nnn(nnn_edges, cols):
    """Split nnn into (diagonals (1,±1), axials (0,2)/(2,0)) by grid displacement."""
    def rc(s):
        return (s // cols, s % cols)

    diag, axial = [], []
    for a, b in nnn_edges:
        lo, hi = min(a, b), max(a, b)
        dr = rc(hi)[0] - rc(lo)[0]
        dc = rc(hi)[1] - rc(lo)[1]
        (diag if abs(dr) == 1 and abs(dc) == 1 else axial).append((a, b))
    return diag, axial


# Shared src helpers (dedup of the manifold detection + subspace fidelity that
# were copied across the study runners). Aliased to the local names used below.
from qmbp_simulation.analysis.fidelity import (  # noqa: E402
    degenerate_ground_manifold as _degenerate_ground_space,
    subspace_fidelity as _subspace_fidelity,
)


def _converge_selection(args, builder, backend, lat, H, psi, manifold, degenerate,
                        e0, phys_gap, label, sel):
    v = make_masked_variant(label, f"N20 geom {label}", BLOCKS, sel)
    qc, _ = build_variant(builder, args.n, lat, v)
    n_2q, npar = cx_and_params(qc)
    ws = prepare_warmstart(qc, H, psi, n_nn=len(sel.nn_edges), n_nnn=len(sel.nnn_edges),
                           n_qubits=args.n, p_layers=1, h=args.h, J2=args.j2,
                           backend=backend, topology=args.topology, model=MODEL,
                           gap=phys_gap, target_len=qc.num_parameters, auto_donors=True)
    seed = np.asarray(ws["seed"], float)
    if seed.size != npar:
        fit = np.zeros(npar); fit[:min(seed.size, npar)] = seed[:min(seed.size, npar)]
        seed = fit
    t0, c0 = time.time(), time.process_time()
    fid, e_best, runs = converge_circuit(qc, H, psi, restarts=args.restarts,
                                         maxiter=args.maxiter, seed0=args.seed0,
                                         warm_theta=seed, backend=backend)
    theta_best = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)
    sub = _subspace_fidelity(qc, theta_best, manifold) if degenerate else None
    report = sub if degenerate else fid
    row = {
        "label": label, "h": round(float(args.h), 2), "n_qubits": args.n,
        "topology": args.topology, "blocks": BLOCKS,
        "n_nn": len(sel.nn_edges), "n_nnn": len(sel.nnn_edges),
        "e0": e0, "phys_gap": phys_gap, "degenerate": degenerate,
        "e_best": float(e_best), "de_gap": abs(float(e_best) - e0) / phys_gap if phys_gap > 0 else None,
        "best_fidelity": float(fid), "subspace_fidelity": sub, "report_fidelity": float(report),
        "n_2q_transpiled": n_2q, "n_params": npar,
        "fidelity_per_cx": (report / n_2q) if n_2q > 0 else None,
        "seed_kind": ws.get("provenance"),
        "seconds": round(time.time() - t0, 1), "cpu_seconds": round(time.process_time() - c0, 1),
    }
    if float(report) >= SPEC_MIN_FID:
        try:
            save_ansatz_spec(v, topology=args.topology, n_qubits=args.n, h=args.h,
                             j2=args.j2, theta=theta_best.tolist(), nn_edges=sel.nn_edges,
                             nnn_edges=sel.nnn_edges, fidelity=float(report),
                             n_2q=n_2q, n_params=npar, p_layers=1)
        except Exception as exc:
            print(f"    [spec save skipped: {exc}]", flush=True)
    return row, theta_best


def run(args) -> int:
    from qmbp_simulation.execution import NoiselessBackend
    from qmbp_simulation.execution.eval_cache import CachedBackend, EvalCache

    builder = HVACircuitBuilder()
    backend = CachedBackend(NoiselessBackend(), topology=args.topology, n_qubits=args.n,
                            model=MODEL, p_layers=1, cache=EvalCache())
    backend.set_h(float(args.h))

    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(args.topology, args.n, args.h, args.j2, 1)
    nn_edges = list(lat.edges)
    nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
    cols = _grid_cols(args.n)
    diag, axial = _classify_nnn(nnn_edges, cols)
    manifold, e0_m, phys_gap, m_dim = _degenerate_ground_space(H)
    degenerate = m_dim > 1
    print(f"[n20_geom] N={args.n} h={args.h} grid {args.n // cols}x{cols} e0={e0:.5f} "
          f"phys_gap={phys_gap:.4f} manifold={m_dim} | nnn={len(nnn_edges)} "
          f"diag={len(diag)} axial={len(axial)}", flush=True)

    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR, out_file=f"n20_geometric_{args.topology}_N{args.n}_h{args.h:.2f}.json",
        fingerprint={"n_qubits": args.n, "p_layers": None, "model": MODEL,
                     "topology": args.topology, "h": args.h},
        extra={"topology": args.topology, "N": args.n, "h": args.h, "J2": args.j2,
               "grid_cols": cols, "n_diag": len(diag), "n_axial": len(axial),
               "schema": "n20_geometric_v1"},
        params={"experiment": "n20_geometric", "N": args.n, "h": args.h},
        description="N=20 geometry-aware nnn selection (frustration diagonals) on the 4x5 grid",
        rows_ref=rows)

    # 1) full reference (also gives per-bond |θ| to rank axials for the hybrid).
    sel_full = BondSelection(nn_edges=nn_edges, nnn_edges=nnn_edges)
    row, theta_full = _converge_selection(args, builder, backend, lat, H, psi, manifold,
                                          degenerate, e0, phys_gap, "full", sel_full)
    rows.append(row); persister.persist(rows)
    print(f"  {'full':14s} fid={row['report_fidelity']:.4f} 2q={row['n_2q_transpiled']} "
          f"fid/cx={row['fidelity_per_cx']:.3e} ({row['cpu_seconds']}s cpu)", flush=True)

    # 2) diagonals only — the frustration-carrying nnn, ~half the gates.
    sel_diag = BondSelection(nn_edges=nn_edges, nnn_edges=diag)
    row, _ = _converge_selection(args, builder, backend, lat, H, psi, manifold,
                                 degenerate, e0, phys_gap, "diag_only", sel_diag)
    rows.append(row); persister.persist(rows)
    print(f"  {'diag_only':14s} fid={row['report_fidelity']:.4f} 2q={row['n_2q_transpiled']} "
          f"fid/cx={row['fidelity_per_cx']:.3e} ({row['cpu_seconds']}s cpu)", flush=True)

    # 3) hybrid: diagonals + strongest 50% axials by |θ| from the full run.
    w_nn, w_nnn = bond_weights_for_blocks(theta_full, BLOCKS, len(nn_edges), len(nnn_edges), args.n)
    axial_set = {tuple(sorted(e)) for e in axial}
    axial_w = [(w, e) for w, e in zip(w_nnn, nnn_edges) if tuple(sorted(e)) in axial_set]
    axial_w.sort(key=lambda t: -t[0])
    keep_axial = [e for _w, e in axial_w[: max(1, len(axial) // 2)]]
    sel_hyb = BondSelection(nn_edges=nn_edges, nnn_edges=diag + keep_axial)
    row, _ = _converge_selection(args, builder, backend, lat, H, psi, manifold,
                                 degenerate, e0, phys_gap, "diag_axial50", sel_hyb)
    rows.append(row); persister.persist(rows)
    print(f"  {'diag_axial50':14s} fid={row['report_fidelity']:.4f} 2q={row['n_2q_transpiled']} "
          f"fid/cx={row['fidelity_per_cx']:.3e} ({row['cpu_seconds']}s cpu)", flush=True)

    persister.persist(rows, status="final")
    backend.flush()
    print("\n=== N20 GEOMETRIC (report_fid | 2q | fid/cx) ===", flush=True)
    for r in sorted(rows, key=lambda x: x["n_2q_transpiled"]):
        print(f"  {r['label']:14s} fid={r['report_fidelity']:.4f} 2q={r['n_2q_transpiled']:3d} "
              f"fid/cx={(r['fidelity_per_cx'] or 0):.3e}", flush=True)
    print("DONE", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="N=20 geometry-aware nnn selection sweep")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--maxiter", type=int, default=400)
    p.add_argument("--seed0", type=int, default=70000)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
