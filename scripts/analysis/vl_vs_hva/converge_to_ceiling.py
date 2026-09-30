#!/usr/bin/env python
"""Continue an ansatz-variant optimization from its saved θ until it CONVERGES.

The partial-p3 runs (`p2_half_nn`, `p2_half_nn_rx` at N=18 h=0.5) stopped as
FLOORS (converged 0/1 at maxiter=1500). This driver RESUMES each variant from its
best saved θ and runs iterated L-BFGS-B segments (each `--maxiter` iters) until a
segment settles before its cap (`nit < maxiter`) — i.e. the genuine ceiling — or
`--max-segments` is hit. It never restarts from scratch, so no prior compute is
lost, and it persists after every segment (crash-safe).

Reads the best θ from the variant artifact
(`ansatz_variants_<topo>_N<n>_h<h>.json`), writes progress to a resume artifact
(`converge_ceiling_<topo>_N<n>_h<h>.json`), and updates the scoreboard-friendly
fields so the consolidated report/scoreboard pick up the improved fidelity.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/converge_to_ceiling.py \
        --n 18 --h 0.5 --p 2 --variants p2_half_nn p2_half_nn_rx \
        --maxiter 3000 --max-segments 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import save_json  # noqa: E402


def _load_saved_theta(study: Path, topology, n, h, variant):
    """Best saved θ for a variant from its ansatz_variants artifact."""
    f = study / "hva_nnn_sweep" / f"ansatz_variants_{topology}_N{n}_h{h:.2f}.json"
    if not f.exists():
        return None, None
    d = json.loads(f.read_text())
    for r in d.get("rows", []):
        if r.get("variant") == variant:
            th = r.get("best_theta_final")
            if th:
                return np.asarray(th, float), float(r.get("best_fidelity", 0.0))
    return None, None


def main(argv=None) -> int:
    from qmbp_simulation.analysis.circuit_visualizer import circuit_summary
    from qmbp_simulation.circuits import HVACircuitBuilder
    from qmbp_simulation.circuits.hva_variants import VARIANTS, build_variant
    from qmbp_simulation.framework.runner_base import resolve_project_root
    from qmbp_simulation.framework.study_core import ground_state, make_cost_fid
    from qmbp_simulation.framework.study_runner import _lbfgsb

    ap = argparse.ArgumentParser(description="Resume variant θ to genuine convergence")
    ap.add_argument("--topology", default="square")
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--j2", type=float, default=0.5)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--max-segments", type=int, default=8)
    ap.add_argument(
        "--ftol-gain", type=float, default=1e-4, help="Stop early if a segment improves fidelity by less than this."
    )
    args = ap.parse_args(argv)

    root = resolve_project_root(Path(__file__))
    study = root / "results" / "hva_vl_study"
    hk = round(float(args.h), 2)
    builder = HVACircuitBuilder()

    lat, _qc_def, H, psi, e0, gap, n_nn, n_nnn = ground_state(args.topology, args.n, hk, args.j2, args.p)

    out_file = f"converge_ceiling_{args.topology}_N{args.n}_h{hk:.2f}.json"
    results = {
        "schema": "converge_ceiling_v1",
        "topology": args.topology,
        "N": args.n,
        "h": hk,
        "p_layers": args.p,
        "J2": args.j2,
        "e0": e0,
        "gap": gap,
        "maxiter": args.maxiter,
        "rows": [],
    }

    def _2q(qc):
        try:
            from qiskit import transpile

            t = transpile(
                qc.assign_parameters(np.full(qc.num_parameters, 0.37)),
                basis_gates=["rz", "sx", "x", "cx"],
                optimization_level=1,
            )
            return circuit_summary(t)["n_2q_gates"]
        except Exception:
            return None

    for name in args.variants:
        if name not in VARIANTS:
            print(f"  skip {name}: unknown variant", flush=True)
            continue
        qc, _ = build_variant(builder, args.n, lat, VARIANTS[name])
        theta, f0 = _load_saved_theta(study, args.topology, args.n, hk, name)
        if theta is None or len(theta) != qc.num_parameters:
            print(f"  skip {name}: no saved θ of matching length", flush=True)
            continue
        cost, fid, grad, _ = make_cost_fid(qc, H, psi)
        n2q = _2q(qc)

        print(
            f"[ceiling] {name}: resume from F={f0:.4f} ({len(theta)} params, "
            f"{n2q} CX), maxiter={args.maxiter} up to {args.max_segments} segments",
            flush=True,
        )
        x = np.asarray(theta, float)
        f_prev = fid(x)
        segments = []
        converged = False
        t0 = time.time()
        for seg in range(args.max_segments):
            x, e, nit = _lbfgsb(cost, x, maxiter=args.maxiter, grad=grad)
            f_now = fid(x)
            seg_converged = nit < args.maxiter
            gain = f_now - f_prev
            segments.append(
                {
                    "segment": seg,
                    "nit": int(nit),
                    "fidelity": f_now,
                    "energy": float(e),
                    "converged": seg_converged,
                    "gain": float(gain),
                }
            )
            print(f"  seg{seg}: nit={nit:>5} F={f_now:.4f} (Δ{gain:+.4f}) converged={seg_converged}", flush=True)
            # Persist after every segment (crash-safe).
            row = {
                "variant": name,
                "n_2q_transpiled": n2q,
                "n_params": qc.num_parameters,
                "start_fidelity": f0,
                "best_fidelity": f_now,
                "e_best": float(e),
                "e0": e0,
                "gap": gap,
                "de_gap": abs(e - e0) / gap if gap > 0 else None,
                "converged_at_ceiling": seg_converged,
                "segments": segments,
                "best_theta_final": x.tolist(),
                "seconds": round(time.time() - t0, 1),
            }
            results["rows"] = [r for r in results["rows"] if r["variant"] != name] + [row]
            save_json(
                results,
                "hva_nnn_sweep",
                out_file,
                params={"experiment": "converge_to_ceiling", "N": args.n, "h": hk, "p_layers": args.p},
                description="Resume ansatz-variant θ to genuine convergence "
                "(ceiling) from saved floors; per-segment crash-safe",
            )
            f_prev = f_now
            if seg_converged:
                converged = True
                break
            if 0 <= gain < args.ftol_gain:
                print(f"  early stop: gain {gain:.2e} < {args.ftol_gain}", flush=True)
                break
        status = "CEILING (converged)" if converged else "floor (capped)"
        print(f"  → {name}: F={f_prev:.4f} @ {n2q} CX — {status}\n", flush=True)

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
