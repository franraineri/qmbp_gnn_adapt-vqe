"""Cheap go/no-go checks for the adaptive warm-start plan — NO full convergence.

Two read-only probes over the EXISTING converged artifacts
(``bond_topk_regime_square_N*_p2_h*.json``), each costing at most one exact
diagonalization per target point:

1. **Cross-N init-fidelity matrix (at fixed h)** — the continuation-in-N probe
   (Fase B). Transfer each N's converged full-ref θ onto every other N's layout
   at the SAME h and evaluate the init-fidelity (no reoptimization) against the
   target ground state. If a smaller-N donor gives a high init-fid at a larger N
   (especially at h=0.5, the regime that collapses), continuation-in-N is
   validated before investing in full runs.

2. **θ_x ~ arctan(J/h) tracking (per N)** — the difficulty-index feature probe
   (Fase A). For each converged θ, compare its θ_x block mean to the analytic
   ``arctan(J/h)``. A small bounded deviation means θ_x is usable as an analytic
   phase feature without measurement.

Reuses only existing pieces: study_core (ground_state, make_cost_fid),
warmstart (warmstart_init_fidelity, theta_x_arctan_deviation), hamiltonian
(nnn edges). Read-only on artifacts; writes nothing.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/check_warmstart_adaptive.py \
        --topology square --p 2 --j2 0.5 --ns 8 10 12 --hs 0.3 0.5 1.3
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

from qmbp_simulation.analysis.theta_patterns import extract_theta_metrics  # noqa: E402
from qmbp_simulation.analysis.warmstart import (  # noqa: E402
    difficulty_index,
    select_regime_seed,
    theta_x_arctan_deviation,
    warmstart_init_fidelity,
)
from qmbp_simulation.framework.study_core import ground_state, make_cost_fid  # noqa: E402
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402

SUBDIR = "results/hva_vl_study/bond_ablation"


def metrics_table(args):
    """Tidy per-(N,h) θ-metrics table — automates the ad-hoc extraction loop."""
    print("\n" + "=" * 72)
    print("PROBE 3: θ metrics table (automated extraction, replaces ad-hoc scripts)")
    print("=" * 72)
    print(f"  {'N':>3} {'h':>4} {'phase':>8} {'fid':>7} {'gap':>8} | "
          f"{'nn_abs':>6} {'nnn_abs':>7} {'x_abs':>6} {'L2/L1':>6} {'d_seed':>7} {'D_idx':>7}")
    for h in args.hs:
        for N in args.ns:
            r = _load_full_ref(args.topology, N, h, args.p)
            if r is None:
                continue
            seed, _ = select_regime_seed(r["n_nn"], r["n_nnn"], N, args.p, h, J2=args.j2)
            m = extract_theta_metrics(r["theta"], r["n_nn"], r["n_nnn"], N, args.p, h,
                                      J2=args.j2, seed_theta=seed)
            D = difficulty_index(N, h, r["gap"], J2=args.j2)
            print(f"  {N:>3} {h:>4} {m['phase']:>8} {r['fid']:>7.4f} {r['gap']:>8.5f} | "
                  f"{m['nn_absmean']:>6.3f} {m['nnn_absmean']:>7.3f} {m['x_absmean']:>6.3f} "
                  f"{m['l2_l1_nn_ratio']:>6.2f} {(m['d_theta_to_seed'] or 0):>7.3f} {D:>7.2f}")


def _load_full_ref(topology, N, h, p):
    """Converged p2 full-ref θ + layout for one (N, h), or None if missing."""
    f = Path(SUBDIR) / f"bond_topk_regime_{topology}_N{N}_p{p}_h{h:.2f}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    row = next((r for r in d.get("rows", []) if r.get("variant") == "p2_full_ref"), None)
    if row is None:
        return None
    th = row.get("best_theta_final") or row.get("theta_final")
    if not th:
        return None
    return {
        "N": N, "h": h, "p": p, "gap": d.get("gap"), "fid": row.get("best_fidelity"),
        "n_nn": row.get("n_nn_bonds"), "n_nnn": row.get("n_nnn_bonds"),
        "theta": np.asarray(th, float),
    }


def _lattice_edges(topology, N, h, j2, p):
    """(nn_edges, nnn_edges, fid_fn-builder inputs) for the target (N, h)."""
    lat, qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(topology, N, h, j2, p)
    _cost, fid, _grad, _ = make_cost_fid(qc, H, psi)
    nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
    return lat.edges, nnn_edges, fid, n_nn, n_nnn, gap


def cross_n_matrix(args):
    """Print, per h, the init-fid matrix donor-N → target-N (transfer, no reopt)."""
    print("=" * 72)
    print("PROBE 1: cross-N init-fidelity (donor θ transferred to target N, same h)")
    print("=" * 72)
    for h in args.hs:
        donors = {}
        for N in args.ns:
            r = _load_full_ref(args.topology, N, h, args.p)
            if r is not None:
                # donor nnn_edges = full lattice nnn at donor N
                _nn, dnnn, _fid, _nnn_cnt, _, _ = _lattice_edges(
                    args.topology, N, h, args.j2, args.p)
                r["nnn_edges"] = dnnn
                donors[N] = r
        if not donors:
            print(f"\n--- h={h}: no artifacts ---")
            continue
        print(f"\n--- h={h} ---")
        print(f"  {'donor\\target':>12} " + " ".join(f"N{N:>2}" for N in args.ns))
        # build target fid fns + edges once per target
        tgt = {}
        for N in args.ns:
            nn_e, nnn_e, fid, n_nn, n_nnn, gap = _lattice_edges(
                args.topology, N, h, args.j2, args.p)
            tgt[N] = dict(nnn_edges=nnn_e, fid=fid, n_nn=n_nn, n_nnn=n_nnn, gap=gap)
        for dN in args.ns:
            if dN not in donors:
                continue
            d = donors[dN]
            cells = []
            for tN in args.ns:
                t = tgt[tN]
                fi, _ = warmstart_init_fidelity(
                    d["theta"], donor_n_nn=d["n_nn"], donor_n_nnn=d["n_nnn"],
                    donor_p=d["p"], target_n_nn=t["n_nn"], target_n_nnn=t["n_nnn"],
                    target_p=args.p, n_qubits=tN, fid_fn=t["fid"],
                    donor_nnn_edges=d["nnn_edges"], target_nnn_edges=t["nnn_edges"],
                    donor_n_qubits=dN,
                )
                mark = "*" if dN == tN else " "
                cells.append(f"{(fi if fi is not None else float('nan')):.3f}{mark}")
            print(f"  {('N'+str(dN)):>12} " + " ".join(f"{c:>5}" for c in cells))
        print(f"  gaps: " + ", ".join(f"N{N}:{tgt[N]['gap']:.5f}" for N in args.ns))
    print("\n  (* = self; off-diagonal below self in the same column = N-transfer quality)")


def theta_x_table(args):
    """Print θ_x vs arctan(J/h) tracking for every converged (N, h)."""
    print("\n" + "=" * 72)
    print("PROBE 2: θ_x block mean vs arctan(J/h)  (phase-feature check)")
    print("=" * 72)
    print(f"  {'N':>3} {'h':>4} {'arctan':>7} {'theta_x':>8} {'rel_dev':>8} {'tracks?':>7}")
    for h in args.hs:
        for N in args.ns:
            r = _load_full_ref(args.topology, N, h, args.p)
            if r is None:
                continue
            dev = theta_x_arctan_deviation(
                r["theta"], r["n_nn"], r["n_nnn"], N, args.p, h, J=1.0)
            print(f"  {N:>3} {h:>4} {dev['arctan_pred']:>7.4f} "
                  f"{dev['theta_x_mean']:>8.4f} {dev['rel_deviation']:>8.4f} "
                  f"{str(dev['tracks_arctan']):>7}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cheap go/no-go checks for adaptive warm-start")
    p.add_argument("--topology", default="square")
    p.add_argument("--p", type=int, default=2)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--ns", type=int, nargs="+", default=[8, 10, 12])
    p.add_argument("--hs", type=float, nargs="+", default=[0.3, 0.5, 1.3])
    args = p.parse_args(argv)
    cross_n_matrix(args)
    theta_x_table(args)
    metrics_table(args)
    print("\nCHECK_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
