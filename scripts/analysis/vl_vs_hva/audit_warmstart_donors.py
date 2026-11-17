#!/usr/bin/env python
"""Audit the cross-N DONOR layer of the warm-start cascade on a structure variant.

Three conditions on the SAME target (variant, N, h), same circuit and budget:

  A) analytic-only  : prepare_warmstart without donors
  B) cascade+donors : same cascade + smaller-N converged specs as extra candidates
  C) random best-of : R uniform starts

Donors come from results/hva_vl_study/ansatz_specs and are transferred with the
core geometric transfer (transfer_theta_for_blocks + lattice_coords), exactly as
run_variant_topk does.

--selector-budgets reproduces the selector diagnostic: each candidate (analytic +
every donor) is relaxed for K iterations and the fidelity reported per K. It
shows at which budget the micro-descent ranking becomes trustworthy (measured
for p2_half_nn_rx N18 h0.5: wrong winner at K=100, correct from K>=400).

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/audit_warmstart_donors.py \
        --n-list 14 --maxiter 800 --random-starts 3
    .venv/bin/python scripts/analysis/vl_vs_hva/audit_warmstart_donors.py \
        --n-list 18 --selector-budgets 100 400 800 --skip-conditions
"""

from __future__ import annotations

import argparse
import glob
import json
import time

import numpy as np

from qmbp_simulation.analysis.warmstart import (
    cached_variant_warmstart_theta,
    lattice_coords,
    transfer_theta_for_blocks,
)
from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.circuits.hva_variants import VARIANTS, build_variant
from qmbp_simulation.framework.study_core import ground_state, make_cost_fid, prepare_warmstart
from qmbp_simulation.framework.study_runner import _lbfgsb
from qmbp_simulation.models import make_lattice
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

SPECS = "results/hva_vl_study/ansatz_specs"


def _converge(cost, grad, fid, x0, maxiter):
    x, e, nit = _lbfgsb(cost, np.asarray(x0, float), maxiter=maxiter, grad=grad)
    return float(fid(x)), float(e), int(nit)


def load_donors(args, n, base, nn_edges, nnn_edges):
    t_coords = lattice_coords(args.topology, n)
    out = []
    pattern = f"{SPECS}/{args.variant}_{args.topology}_N*_h{args.h:.2f}.spec.json"
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            spec = json.load(fh)
        dn, theta = spec.get("n_qubits"), spec.get("theta")
        if dn is None or dn >= n or not theta:
            continue
        lat_d = make_lattice(args.topology, dn, J=1.0, h=args.h)
        nn_d = list(lat_d.edges)
        nnn_d = HamiltonianBuilder._generate_nnn_edges(lat_d)
        seed = transfer_theta_for_blocks(
            np.asarray(theta, float),
            list(base.blocks),
            donor_nnn_edges=nnn_d,
            target_nnn_edges=nnn_edges,
            n_nn=len(nn_edges),
            n_qubits=n,
            rx_final=base.rx_final,
            rz_final=base.rz_final,
            donor_n_nn=len(nn_d),
            donor_n_qubits=dn,
            donor_nn_edges=nn_d,
            target_nn_edges=nn_edges,
            donor_coords=lattice_coords(args.topology, dn),
            target_coords=t_coords,
        )
        if seed is not None:
            out.append((seed, f"donorN{dn}@{spec.get('fidelity', 0):.3f}"))
    return out


def run_one(n, args):
    lat, _qc, H, psi, _e0, gap, n_nn, n_nnn = ground_state(args.topology, n, args.h, args.j2, 1)
    base = VARIANTS[args.variant]
    nn_edges = list(lat.edges)
    nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
    qc, _ = build_variant(HVACircuitBuilder(), n, lat, base)
    npar = qc.num_parameters
    cost, fid, grad, backend = make_cost_fid(qc, H, psi)
    print(
        f"\n=== N={n} h={args.h} gap={gap:.5f} variant={args.variant} npar={npar} maxiter={args.maxiter} ===",
        flush=True,
    )

    analytic = cached_variant_warmstart_theta(
        base.blocks, n_nn, n_nnn, n, args.h, J2=args.j2, rx_final=base.rx_final, rz_final=base.rz_final
    )
    donors = load_donors(args, n, base, nn_edges, nnn_edges)
    print(f"  donors: {[lbl for _, lbl in donors]}", flush=True)

    def cascade(extra):
        return prepare_warmstart(
            qc,
            H,
            psi,
            n_nn=n_nn,
            n_nnn=n_nnn,
            n_qubits=n,
            p_layers=1,
            h=args.h,
            J2=args.j2,
            micro_descent=args.select_descent,
            extra_candidates=extra,
            cost=cost,
            fid=fid,
            grad=grad,
            backend=backend,
            target_len=npar,
        )

    result = {"n": n}
    if not args.skip_conditions:
        t0 = time.time()
        ws_a = cascade([(analytic, "analytic")])
        result["analytic"], _, nit_a = _converge(cost, grad, fid, ws_a["seed"], args.maxiter)
        print(
            f"  A) analytic-only : fid={result['analytic']:.4f} nit={nit_a} "
            f"prov={ws_a['provenance']} ({time.time() - t0:.0f}s)",
            flush=True,
        )

        t0 = time.time()
        ws_b = cascade([(analytic, "analytic")] + donors)
        result["cascade"], _, nit_b = _converge(cost, grad, fid, ws_b["seed"], args.maxiter)
        print(
            f"  B) cascade+donors: fid={result['cascade']:.4f} nit={nit_b} "
            f"prov={ws_b['provenance']} ({time.time() - t0:.0f}s)",
            flush=True,
        )

        t0 = time.time()
        rng = np.random.default_rng(args.seed + n)
        best_e, best_f = None, -1.0
        for _ in range(args.random_starts):
            f, e, _ = _converge(cost, grad, fid, rng.uniform(-np.pi, np.pi, npar), args.maxiter)
            if best_e is None or e < best_e:
                best_e, best_f = e, f
        result["random"] = best_f
        print(f"  C) random bo-{args.random_starts}  : fid={best_f:.4f} ({time.time() - t0:.0f}s)", flush=True)
        print(
            f"  -> donor rescue over analytic: {result['cascade'] - result['analytic']:+.4f} | "
            f"cascade over random: {result['cascade'] - best_f:+.4f}",
            flush=True,
        )

    if args.selector_budgets:
        ks = sorted(args.selector_budgets)
        print("\n  selector ranking vs micro-descent budget (fidelity after K iters)", flush=True)
        print(f"  {'candidate':14s}" + "".join(f"{'K=' + str(k):>9s}" for k in ks), flush=True)
        winners = {k: ("", -1.0) for k in ks}
        for seed, label in [(analytic, "analytic")] + donors:
            fids = [_converge(cost, grad, fid, seed, k)[0] for k in ks]
            for k, f in zip(ks, fids, strict=True):
                if f > winners[k][1]:
                    winners[k] = (label, f)
            print(f"  {label:14s}" + "".join(f"{f:>9.4f}" for f in fids), flush=True)
        print("  winner per budget: " + ", ".join(f"K={k}:{winners[k][0]}" for k in ks), flush=True)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cross-N donor layer audit")
    p.add_argument("--topology", default="square")
    p.add_argument("--variant", default="p2_half_nn_rx")
    p.add_argument("--n-list", type=int, nargs="+", default=[14])
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--maxiter", type=int, default=800, help="Final convergence budget (800 = production N18).")
    p.add_argument(
        "--select-descent", type=int, default=120, help="Micro-descent iters used to rank cascade candidates."
    )
    p.add_argument("--random-starts", type=int, default=3)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument(
        "--selector-budgets",
        type=int,
        nargs="+",
        default=None,
        help="Also print the candidate-ranking table for these K values.",
    )
    p.add_argument(
        "--skip-conditions",
        action="store_true",
        help="Skip A/B/C (use with --selector-budgets for the diagnostic alone).",
    )
    args = p.parse_args(argv)

    rows = [run_one(n, args) for n in args.n_list]
    if not args.skip_conditions:
        print("\n=== SUMMARY ===", flush=True)
        for r in rows:
            print(
                f"  N={r['n']}: analytic={r['analytic']:.4f} cascade+donors={r['cascade']:.4f} "
                f"random={r['random']:.4f} | donor rescue {r['cascade'] - r['analytic']:+.4f}",
                flush=True,
            )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
