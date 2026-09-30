#!/usr/bin/env python
"""E2 (cheap part): fidelity AT the seed for the nn-shrink vs standard seed.

Evaluates the ANALYTIC seed's fidelity (no VQE / no optimization — just a
statevector overlap) for the standard second-order seed and for the
nn-shrink variant across a scan of ``nn_extra_shrink`` factors, at several N and
both h regimes. This is cheap (one statevector per point) and non-competing for
CPU, and it answers: does shrinking θ_nn start the optimizer CLOSER to the ground
state? If the best factor is consistently < 1, E1's trend is confirmed and gives
the calibrated default.

This measures the SEED quality (starting point), not the final optimized fidelity
— a good seed reduces restart dependence but the VQE evaluation (E2 full) is
separate and heavier.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/eval_nn_shrink_seed.py \
        --n-list 6 8 10 12 --h 0.5 1.0 --p 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import save_json  # noqa: E402


def main(argv=None) -> int:
    from qiskit.quantum_info import Statevector

    from qmbp_simulation.analysis.warmstart import (
        second_order_nn_shrink_theta,
        second_order_warmstart_theta,
    )
    from qmbp_simulation.framework.study_core import ground_state
    from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

    ap = argparse.ArgumentParser(description="Fidelity at seed: nn-shrink vs standard")
    ap.add_argument("--topology", default="square")
    ap.add_argument("--n-list", type=int, nargs="+", default=[6, 8, 10, 12])
    ap.add_argument("--h", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--j2", type=float, default=0.5)
    ap.add_argument("--shrink-scan", type=float, nargs="+", default=[1.0, 0.8, 0.6, 0.4, 0.2])
    args = ap.parse_args(argv)

    def fid_at(qc, psi, theta):
        sv = np.asarray(Statevector(qc.assign_parameters(theta)).data)
        return float(abs(np.vdot(psi, sv)) ** 2)

    results = {
        "schema": "nn_shrink_seed_eval_v1",
        "topology": args.topology,
        "J2": args.j2,
        "p_layers": args.p,
        "shrink_scan": args.shrink_scan,
        "rows": [],
    }

    print(f"{'N':>3} {'h':>5}  {'std':>7}  " + "  ".join(f"s={s:<4}" for s in args.shrink_scan) + "   best_s  best_F")
    print("-" * 78)
    for n in args.n_list:
        lat_probe = None
        try:
            from qmbp_simulation.models import make_lattice

            lat_probe = make_lattice(args.topology, n, J=1.0, h=args.h[0])
        except Exception:
            continue
        n_nn = len(lat_probe.edges)
        n_nnn = len(HamiltonianBuilder._generate_nnn_edges(lat_probe))
        for h in args.h:
            hf = round(float(h), 2)
            lat, qc, H, psi, e0, gap, _, _ = ground_state(args.topology, n, hf, args.j2, args.p)
            std_theta = second_order_warmstart_theta(n_nn, n_nnn, n, args.p, hf, J=1.0, J2=args.j2)
            f_std = fid_at(qc, psi, std_theta)
            scan = {}
            for s in args.shrink_scan:
                th = second_order_nn_shrink_theta(n_nn, n_nnn, n, args.p, hf, J=1.0, J2=args.j2, nn_extra_shrink=s)
                scan[s] = fid_at(qc, psi, th)
            best_s = max(scan, key=lambda k: scan[k])
            results["rows"].append(
                {
                    "N": n,
                    "h": hf,
                    "gap": gap,
                    "fid_std_seed": f_std,
                    "scan": {str(k): v for k, v in scan.items()},
                    "best_shrink": best_s,
                    "best_fid": scan[best_s],
                }
            )
            scan_str = "  ".join(f"{scan[s]:.4f}" for s in args.shrink_scan)
            print(f"{n:>3} {hf:>5.2f}  {f_std:>7.4f}  {scan_str}   {best_s:>5}  {scan[best_s]:.4f}")

    save_json(
        results,
        "hva_nnn_sweep",
        "nn_shrink_seed_eval_square.json",
        params={"experiment": "nn_shrink_seed_eval", "p_layers": args.p},
        description="E2 cheap: fidelity AT the analytic seed for nn-shrink scan vs standard second-order seed (no VQE)",
    )
    print(f"\n[E2-cheap] {len(results['rows'])} points → nn_shrink_seed_eval_square.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
