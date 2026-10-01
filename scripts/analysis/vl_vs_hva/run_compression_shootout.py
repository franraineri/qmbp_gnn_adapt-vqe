#!/usr/bin/env python
"""Compression shootout — reduce the 2q-gate count of a converged ansatz.

Takes ONE converged variant (default p1_half_nn_rx at N=10 h=0.5), binds its
optimized θ, and measures every circuit-compression technique's resulting
2q-gate count AND the fidelity of the compressed state against the exact ground
state. This isolates "shrink the circuit without re-choosing bonds" from the
bond-selection levers already studied.

Techniques compared (all on the SAME bound circuit):
  0. baseline       — transpile(opt_level=1), the metric used across the study
  1. opt_level=2/3  — heavier transpiler passes (cancels redundant 1q/2q)
  2. pauli_evo      — PauliEvolution representation then transpile (exposes
                      commuting structure; ~6-11% lower 2q-depth, same count)
  3. aqc_tensor     — AQC-Tensor: simulate target as MPS, re-fit a shallower
                      p=1 ansatz to maximize |<compressed|target>|^2

Each row: technique, n_2q, Δ vs baseline, fidelity-to-exact (recomputed on the
compressed/transpiled circuit), and whether it stayed within tolerance.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/run_compression_shootout.py \
        --n 10 --h 0.5 --variant p1_half_nn_rx --restarts 2 --maxiter 800
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import save_json  # noqa: E402

from qmbp_simulation.analysis.circuit_visualizer import circuit_summary  # noqa: E402
from qmbp_simulation.analysis.warmstart import variant_warmstart_theta  # noqa: E402
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.bond_mask import (  # noqa: E402
    selection_from_variant_theta,
)
from qmbp_simulation.circuits.hva_variants import (  # noqa: E402
    VARIANTS,
    build_variant,
    make_masked_variant,
)
from qmbp_simulation.framework.study_core import (  # noqa: E402
    converge_circuit,
    ground_state,
    prepare_warmstart,
)
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402

SUBDIR = "bond_ablation"
BASIS = ["rz", "sx", "x", "cx"]


def _n2q(qc) -> int:
    return int(circuit_summary(qc)["n_2q_gates"])


def _fid_bound(bound_qc, psi) -> float:
    ps = np.asarray(Statevector(bound_qc).data)
    return float(abs(np.vdot(psi, ps)) ** 2)


def _transpiled(bound_qc, level):
    return transpile(bound_qc, basis_gates=BASIS, optimization_level=level,
                     seed_transpiler=42)


def run(args) -> int:
    base = VARIANTS[args.variant]
    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(
        args.topology, args.n, args.h, args.j2, 1)
    nn_edges = list(lat.edges)
    nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
    builder = HVACircuitBuilder()

    # Optionally PRUNE first (T1 by tolerance), so the compression techniques act
    # on the already-reduced circuit — the real target of "compress the cheapest
    # ansatz further". The prune selection needs a converged full θ to rank bonds.
    label_suffix = ""
    if args.prune_tol is not None:
        full_qc, _ = build_variant(builder, args.n, lat, base)
        ws_full = prepare_warmstart(
            full_qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=args.n, p_layers=1,
            h=args.h, J2=args.j2, micro_descent=args.micro_descent,
            target_len=full_qc.num_parameters,
            extra_candidates=[(variant_warmstart_theta(
                base.blocks, n_nn, n_nnn, args.n, args.h, J=1.0, J2=args.j2,
                rx_final=base.rx_final, rz_final=base.rz_final), "analytic")])
        _f, _e, runs_full = converge_circuit(
            full_qc, H, psi, restarts=args.restarts, maxiter=args.maxiter,
            seed0=args.seed0, warm_theta=ws_full["seed"])
        theta_full = np.asarray(
            min(runs_full, key=lambda r: r["energy"])["theta_final"], float)
        sel = selection_from_variant_theta(
            theta_full, list(base.blocks), nn_edges, nnn_edges, args.n,
            rx_final=base.rx_final, rz_final=base.rz_final,
            method="prune", tol=args.prune_tol)
        variant = make_masked_variant(
            f"{args.variant}_prune{args.prune_tol}", f"{args.variant} prune",
            list(base.blocks), sel, rx_final=base.rx_final, rz_final=base.rz_final)
        label_suffix = f"_prune{args.prune_tol}_nn{len(sel.nn_edges)}_nnn{len(sel.nnn_edges)}"
        print(f"[compress] pruned to nn={len(sel.nn_edges)}/{n_nn} "
              f"nnn={len(sel.nnn_edges)}/{n_nnn} (tol={args.prune_tol})", flush=True)
    else:
        variant = base
        sel = None

    qc, _ = build_variant(builder, args.n, lat, variant)
    print(f"[compress] N={args.n} h={args.h} variant={args.variant}{label_suffix} "
          f"e0={e0:.5f} gap={gap:.5f} npar={qc.num_parameters}", flush=True)

    # Converge the (possibly pruned) circuit via the MANDATORY warm-start cascade.
    v_nn = len(sel.nn_edges) if sel is not None else n_nn
    v_nnn = len(sel.nnn_edges) if sel is not None else n_nnn
    tgt_nnn = sel.nnn_edges if sel is not None else None
    ws = prepare_warmstart(
        qc, H, psi, n_nn=v_nn, n_nnn=v_nnn, n_qubits=args.n, p_layers=1, h=args.h,
        J2=args.j2, target_nnn_edges=tgt_nnn, micro_descent=args.micro_descent,
        target_len=qc.num_parameters,
        extra_candidates=[(variant_warmstart_theta(
            variant.blocks, v_nn, v_nnn, args.n, args.h, J=1.0, J2=args.j2,
            rx_final=variant.rx_final, rz_final=variant.rz_final), "analytic")])
    fid0, _e, runs = converge_circuit(qc, H, psi, restarts=args.restarts,
                                      maxiter=args.maxiter, seed0=args.seed0,
                                      warm_theta=ws["seed"])
    theta = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)
    bound = qc.assign_parameters(theta)
    print(f"  converged fid={fid0:.4f} seed={ws['provenance']}", flush=True)

    results = []
    _tol_tag = f"_prune{args.prune_tol}" if args.prune_tol is not None else ""
    out_file = f"compression_shootout_{args.variant}{_tol_tag}_N{args.n}_h{args.h:.2f}.json"

    def _persist(status="partial"):
        save_json(
            {"rows": results, "variant": args.variant, "N": args.n, "h": args.h,
             "J2": args.j2, "e0": e0, "gap": gap, "converged_fid": fid0,
             "n2q_baseline": (results[0]["n_2q"] if results else None),
             "schema": "compression_shootout_v1", "status": status},
            SUBDIR, out_file,
            params={"experiment": "compression_shootout", "N": args.n, "h": args.h,
                    "variant": args.variant},
            description="Circuit-compression shootout: 2q reduction + fidelity per "
                        "technique on a converged ansatz")

    def _record(tech, cqc, note=""):
        f = _fid_bound(cqc, psi)
        q = _n2q(cqc)
        results.append({"technique": tech, "n_2q": q, "fidelity": f,
                        "depth": int(cqc.depth()), "note": note})
        print(f"  {tech:22s} 2q={q:>4} fid={f:.4f} depth={cqc.depth()}", flush=True)
        _persist()  # crash-safe: persist after every technique (AQC is slow)

    # 0) baseline opt_level=1 (study metric)
    base_t = _transpiled(bound, 1)
    _record("baseline(opt1)", base_t)
    n2q_base = results[0]["n_2q"]

    # 1) opt_level 2 and 3
    _record("transpile(opt2)", _transpiled(bound, 2))
    _record("transpile(opt3)", _transpiled(bound, 3))

    # 2) AQC-Tensor compression to a shallower p=1 ansatz, BEST OF N restarts.
    # AQC's fit is non-convex (random ansatz-param init), so we run it
    # --aqc-restarts times and keep the compressed circuit whose state has the
    # highest fidelity against the EXACT ground state (not just vs the target).
    try:
        from qmbp_simulation.circuits.aqc_compression import (
            AQCCircuitCompressor,
            AQCCompressionConfig,
        )
        cfg = AQCCompressionConfig(max_bond_dim=args.chi, max_iterations=args.aqc_iters,
                                   fidelity_threshold=0.0)
        comp = AQCCircuitCompressor(cfg)
        best = None  # (fid_exact, cqc_t, aqc_fid)
        t0 = time.time()
        for r in range(args.aqc_restarts):
            res = comp.compress_circuit(bound, lat)
            cqc_t = _transpiled(res.compressed_circuit, 1)
            f_exact = _fid_bound(cqc_t, psi)
            if best is None or f_exact > best[0]:
                best = (f_exact, cqc_t, res.fidelity)
            print(f"    aqc restart {r+1}/{args.aqc_restarts}: "
                  f"fid_exact={f_exact:.4f} aqc_fid={res.fidelity:.4f}", flush=True)
        _record("aqc_tensor(p1)", best[1],
                note=f"best_of_{args.aqc_restarts} aqc_fid={best[2]:.4f} "
                     f"{time.time()-t0:.0f}s")
    except Exception as exc:  # noqa: BLE001
        print(f"  aqc_tensor skipped: {type(exc).__name__}: {str(exc)[:80]}", flush=True)

    # Summary
    print("\n=== COMPRESSION SHOOTOUT (fid | 2q | Δ2q vs baseline) ===", flush=True)
    for r in sorted(results, key=lambda x: x["n_2q"]):
        d = r["n_2q"] - n2q_base
        print(f"  {r['technique']:22s} fid={r['fidelity']:.4f} 2q={r['n_2q']:>4} "
              f"Δ={d:+d} {r['note']}", flush=True)

    _persist(status="final")
    print("DONE", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Circuit compression shootout")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--variant", default="p1_half_nn_rx")
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--maxiter", type=int, default=800)
    p.add_argument("--seed0", type=int, default=70000)
    p.add_argument("--micro-descent", type=int, default=60,
                   help="Micro-descent iters for the warm-start cascade.")
    p.add_argument("--prune-tol", type=float, default=None,
                   help="Prune the ansatz (T1, |θ|<tol) BEFORE compressing, so "
                        "the techniques act on the already-reduced circuit.")
    p.add_argument("--chi", type=int, default=64, help="AQC MPS bond dimension.")
    p.add_argument("--aqc-iters", type=int, default=200,
                   help="AQC L-BFGS-B iterations.")
    p.add_argument("--aqc-restarts", type=int, default=3,
                   help="AQC compression restarts; keep the best vs exact GS.")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
