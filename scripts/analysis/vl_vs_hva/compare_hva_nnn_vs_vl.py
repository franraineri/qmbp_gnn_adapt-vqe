#!/usr/bin/env python
"""Homogeneous HVA-nnn vs VL circuit comparison at fixed (N, p, h, J2).

Answers "which method serves which scenario" under an IDENTICAL setting — same N,
same p, same h, same model — reporting the FULL gate breakdown (total / 2q / 1q /
depth / 2q-depth) for both, plus fidelity. Hardware executability and θ-search
cost are intentionally out of scope here.

- HVA (nnn = create_bond_resolved_frustrated) is reproduced deterministically via
  study_core (best-of-by-energy), so θ are captured and persisted — the sweep
  artifact did not store them. Both the logical and transpiled (rz/sx/x/cx)
  breakdowns are counted with the canonical circuit_summary.
- VL numbers are read from the VL sweep artifact (Haiqu-measured; not recomputable
  locally without the Haiqu cloud). Missing VL rows are reported as such.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/compare_hva_nnn_vs_vl.py \
        --n 9 --p 2 --h 0.5 1.0 --restarts 4 --maxiter 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import save_json, study_dir  # noqa: E402

_VL_SWEEP = "vl_h_sweep/vl_h_sweep_frustrated.json"
_VL_QUALITY = "vl_quality_sweep/vl_quality_sweep_frustrated.json"


def _vl_rows_by_h(root: Path, n: int, topology: str) -> dict[float, dict]:
    """Load VL sweep rows keyed by rounded h (2-decimal), filtered to (N, topology).

    The sweep artifact mixes system sizes (e.g. N=9 and N=10 both at h=1.0), so
    filtering by N is required to compare under the SAME scenario.
    """
    path = root / "results" / "hva_vl_study" / _VL_SWEEP
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {
        round(float(r["h"]), 2): r
        for r in data.get("rows", [])
        if int(r.get("N", -1)) == n and r.get("topology", topology) == topology
    }


def _vl_quality_by_h(root: Path, n: int, topology: str) -> dict[float, list[dict]]:
    """Load VL quality-sweep configs per h (N/topology filtered), sorted by fidelity."""
    path = root / "results" / "hva_vl_study" / _VL_QUALITY
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    by_h: dict[float, list[dict]] = {}
    for r in data.get("rows", []):
        if int(r.get("N", -1)) != n or r.get("topology", topology) != topology:
            continue
        by_h.setdefault(round(float(r["h"]), 2), []).append(
            {
                "num_layers": r.get("num_layers"),
                "fine_tuning_iterations": r.get("fine_tuning_iterations"),
                "fidelity": r.get("vl_fidelity"),
                "n_2q_gates": r.get("vl_2q"),
                "abs_error": r.get("vl_abs_error"),
            }
        )
    for h in by_h:
        by_h[h].sort(key=lambda c: (c["fidelity"] if c["fidelity"] is not None else -1))
    return by_h


def _transpile_counts(qc):
    """Structural gate breakdown of the HVA circuit in the shared basis.

    Binds NON-ZERO angles (0.37) before transpiling so the count reflects the
    ansatz's structural 2q cost, not an accidental reduction from optimized
    θ that happen to make some RZZ ≈ identity (which the transpiler cancels).
    This matches how the study reports ``n_2q_transpiled``.
    """
    from qiskit import transpile

    from qmbp_simulation.analysis.circuit_visualizer import circuit_summary

    logical = circuit_summary(qc)
    angles = np.full(qc.num_parameters, 0.37)
    t = transpile(qc.assign_parameters(angles), basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
    trans = circuit_summary(t)
    return logical, trans, t.depth()


def _best_warmstart_ceiling(
    cost, fid, grad, npar, *, n_nn, n_nnn, n_qubits, p_layers, h, j2, restarts, maxiter, seed0, strategies
):
    """Best HVA fidelity over the study's warm-start methods + random best-of.

    Runs each requested analytic warm-start strategy (``run_strategy``, e.g. the
    ``mixed`` multi-seed method the study found most robust) AND a random
    best-of, then keeps the LOWEST-ENERGY result across all of them (best-of by
    energy — validated monotone with fidelity for this ansatz family). Returns
    ``(best_fid, best_e, best_theta, method, per_method)``.
    """
    from qmbp_simulation.framework.study_core import optimize_bestof
    from qmbp_simulation.framework.study_runner import run_strategy

    candidates = []  # (energy, fidelity, theta, method)
    per_method: dict[str, dict] = {}

    for strat in strategies:
        res = run_strategy(
            strat,
            cost=cost,
            fidelity=fid,
            n_nn=n_nn,
            n_nnn=n_nnn,
            n_qubits=n_qubits,
            p_layers=p_layers,
            h=h,
            J=1.0,
            J2=j2,
            maxiter=maxiter,
            seed0=seed0,
            grad=grad,
        )
        label = f"warmstart:{res.strategy}"
        candidates.append((res.energy, res.fidelity, np.asarray(res.best_theta), label))
        per_method[label] = {"fidelity": res.fidelity, "energy": res.energy, "n_runs": len(res.runs)}

    rb_fid, rb_e, rb_runs = optimize_bestof(cost, fid, grad, npar, restarts=restarts, maxiter=maxiter, seed0=seed0)
    rb_best = min(rb_runs, key=lambda r: r["energy"])
    candidates.append((rb_e, rb_fid, np.asarray(rb_best["theta_final"]), "random_bestof"))
    per_method["random_bestof"] = {"fidelity": rb_fid, "energy": rb_e, "n_runs": len(rb_runs)}

    best_e, best_fid, best_theta, method = min(candidates, key=lambda c: c[0])
    return float(best_fid), float(best_e), best_theta, method, per_method


def main(argv=None) -> int:
    from qmbp_simulation.framework.runner_base import resolve_project_root
    from qmbp_simulation.framework.study_core import ground_state, make_cost_fid

    ap = argparse.ArgumentParser(description="Homogeneous HVA-nnn vs VL comparison")
    ap.add_argument("--topology", default="square")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--j2", type=float, default=0.5)
    ap.add_argument("--h", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--restarts", type=int, default=8, help="Random best-of restarts (extra basin coverage).")
    ap.add_argument("--maxiter", type=int, default=2000)
    ap.add_argument("--seed0", type=int, default=70000)
    ap.add_argument(
        "--strategies",
        nargs="+",
        default=["mixed", "second_order", "metropolis"],
        help="Warm-start methods to try; best-of-by-energy across all "
        "(+ random best-of). Default: the study's strongest set.",
    )
    args = ap.parse_args(argv)

    root = resolve_project_root(Path(__file__))
    vl_by_h = _vl_rows_by_h(root, args.n, args.topology)
    vl_quality_by_h = _vl_quality_by_h(root, args.n, args.topology)

    rows = []
    print(f"[compare] HVA-nnn vs VL  N={args.n} p={args.p} J2={args.j2} topology={args.topology}", flush=True)
    for h in args.h:
        hk = round(float(h), 2)
        # HVA nnn: reproduce deterministically, capture θ + full gate breakdown.
        lat, qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(args.topology, args.n, h, args.j2, args.p)
        cost, fid, grad, _ = make_cost_fid(qc, H, psi)
        best_fid, best_e, theta, method, per_method = _best_warmstart_ceiling(
            cost,
            fid,
            grad,
            qc.num_parameters,
            n_nn=n_nn,
            n_nnn=n_nnn,
            n_qubits=args.n,
            p_layers=args.p,
            h=h,
            j2=args.j2,
            restarts=args.restarts,
            maxiter=args.maxiter,
            seed0=args.seed0,
            strategies=args.strategies,
        )
        theta = np.asarray(theta, float).tolist()
        logical, trans, tdepth = _transpile_counts(qc)

        vl = vl_by_h.get(hk)
        # Scenario key stamped identically on both sides so the report generator
        # can assert the two methods were compared under the SAME conditions.
        scenario = {
            "topology": args.topology,
            "N": args.n,
            "p_layers": args.p,
            "h": hk,
            "J2": args.j2,
            "model": "tfim_frustrated",
        }
        row = {
            **scenario,
            "scenario": scenario,
            "E0": e0,
            "gap": gap,
            "n_nn": n_nn,
            "n_nnn": n_nnn,
            "vl_present": vl is not None,
            "hva_nnn": {
                "ansatz": "nnn (create_bond_resolved_frustrated)",
                "fidelity": best_fid,
                "energy": best_e,
                "abs_error": abs(best_e - e0),
                "de_gap": abs(best_e - e0) / gap if gap > 0 else None,
                "logical": {
                    "total_gates": logical["n_gates_total"],
                    "n_2q_gates": logical["n_2q_gates"],
                    "n_1q_gates": logical["n_1q_gates"],
                    "depth": logical["depth"],
                    "gate_counts": logical["gate_counts"],
                    "n_parameters": logical["n_parameters"],
                },
                "transpiled": {
                    "total_gates": trans["n_gates_total"],
                    "n_2q_gates": trans["n_2q_gates"],
                    "n_1q_gates": trans["n_1q_gates"],
                    "depth": tdepth,
                    "gate_counts": trans["gate_counts"],
                },
                "theta_final": theta,
                "best_method": method,
                "per_method": per_method,
            },
            "vl": None
            if vl is None
            else {
                "fidelity": vl.get("vl_fidelity"),
                "energy": vl.get("vl_energy"),
                "abs_error": vl.get("vl_abs_error"),
                "de_gap": (abs(vl["vl_abs_error"]) / gap if vl.get("vl_abs_error") is not None and gap > 0 else None),
                "total_gates": vl.get("vl_total_gates"),
                "n_2q_gates": vl.get("vl_2q"),
                "n_1q_gates": vl.get("vl_1q"),
                "depth": vl.get("vl_depth"),
                "gate_counts": vl.get("vl_gate_counts"),
                "settings": "default L2/F20 (Haiqu-measured)",
            },
            # Full VL quality curve at this scenario (all num_layers/fine_tuning
            # configs), so the report can show default AND best VL without
            # hand-copying numbers. Empty list when the sweep has no data here.
            "vl_quality_configs": vl_quality_by_h.get(hk, []),
        }
        rows.append(row)

        vl_fid = f"{vl['vl_fidelity']:.4f}" if vl else "n/a"
        vl_2q = vl.get("vl_2q") if vl else "n/a"
        vl_tot = vl.get("vl_total_gates") if vl else "n/a"
        print(
            f"  h={hk:>4}: HVA-nnn F={best_fid:.4f} ({method}) "
            f"2q={trans['n_2q_gates']:>3} total={trans['n_gates_total']:>3} "
            f"depth={tdepth:>3} | VL F={vl_fid} 2q={vl_2q} total={vl_tot}",
            flush=True,
        )

    save_json(
        {
            "rows": rows,
            "topology": args.topology,
            "N": args.n,
            "p_layers": args.p,
            "J2": args.j2,
            "restarts": args.restarts,
            "maxiter": args.maxiter,
            "strategies": args.strategies,
            "schema": "hva_nnn_vs_vl_full_v1",
        },
        "hva_nnn_sweep",
        f"compare_hva_nnn_vs_vl_{args.topology}_N{args.n}_p{args.p}.json",
        params={"experiment": "hva_nnn_vs_vl_full", "N": args.n, "p_layers": args.p},
        description="Homogeneous HVA-nnn vs VL: full gate breakdown + fidelity "
        "at fixed (N,p,h,J2); HVA θ captured, VL Haiqu-measured",
    )
    print(
        f"[compare] saved -> {study_dir('hva_nnn_sweep')}/"
        f"compare_hva_nnn_vs_vl_{args.topology}_N{args.n}_p{args.p}.json",
        flush=True,
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
