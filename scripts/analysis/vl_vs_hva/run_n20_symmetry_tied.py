#!/usr/bin/env python
"""N=20 symmetry-tied vs independent ansatz — does orbit-tying rescue fidelity?

Hypothesis (from the whole study): the low-2q wall is a CONVERGENCE problem, not
an expressivity one — the shallow/pruned structures under 170 2q optimize poorly
in the near-degenerate manifold. The square N=20 ground state is exactly D2-
invariant (symmetry study: |⟨ψ|P|ψ⟩|=1.0), so tying bonds/sites by symmetry orbit
shrinks the parameter space ×3.3 WITHOUT changing the 2q count. Fewer parameters
→ an easier landscape → potentially higher fidelity at the same gate budget.

This runner builds, for each structure, BOTH circuits on the SAME bond selection
(identical 2q) — the independent (masked) and the symmetry-tied — converges each
from the same analytic warm-start, and reports the fidelity gap. A tied circuit
that reaches ≥ the independent one at fewer parameters confirms the lever.

Reuses: ground_state, converge_circuit, cx_and_params (study_core),
degenerate_ground_manifold + subspace_fidelity (analysis.fidelity — the correct
metric in the doublet), detect_symmetry_group + compress_seed_to_orbits
(circuits.symmetry),
HVACircuitBuilder.create_bond_resolved_masked / _symmetry_tied, StudyPersister +
save_ansatz_spec (persistence). No new physics.

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_n20_symmetry_tied.py --h 0.5 \
        --restarts 4 --maxiter 400
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

from hva_vl_study_common import StudyPersister  # noqa: E402

from qmbp_simulation.analysis.fidelity import (  # noqa: E402
    cached_degenerate_ground_manifold,
    subspace_fidelity,
)
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.bond_mask import BondSelection, full_selection  # noqa: E402
from qmbp_simulation.circuits.symmetry import (  # noqa: E402
    analyze_symmetry,
    compress_seed_to_orbits,
    detect_symmetry_group,
)
from qmbp_simulation.framework.study_core import (  # noqa: E402
    converge_circuit,
    cx_and_params,
    ground_state,
    prepare_warmstart,
)
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402

SUBDIR = "bond_ablation"
MODEL = "tfim_frustrated"


def _structures(nn_edges, nnn_edges):
    """(label, blocks, BondSelection) shallow/mid structures under ~170-300 2q.

    These are the structures where tying should matter most: few layers, few
    params, degenerate regime. nn kept complete (backbone irreducible); nnn
    subset controls the 2q budget. Each RZZ = 2 CX.
    """
    nn_full = list(nn_edges)

    # keep the first k nnn (deterministic subset; ranking is orthogonal here —
    # we compare tied vs independent on the SAME selection, so any fixed subset
    # isolates the tying effect).
    def sel(k):
        return BondSelection(nn_edges=nn_full, nnn_edges=list(nnn_edges)[:k])

    return [
        ("p1_half_nn_rx", ["nn", "nnn", "x", "nn", "x"], full_selection(nn_full, [])),  # nnn=0
        ("p1_nnn8", ["nn", "nnn", "x"], sel(8)),
        ("p1_nnn14", ["nn", "nnn", "x"], sel(14)),
        ("p2_nnn8", ["nn", "nnn", "x", "nn", "nnn", "x"], sel(8)),
    ]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="N=20 symmetry-tied vs independent ansatz")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--maxiter", type=int, default=400)
    p.add_argument("--seed0", type=int, default=90000)
    p.add_argument("--rx-final", action="store_true", help="append a trailing RX block (0 CX)")
    p.add_argument(
        "--target-fid",
        type=float,
        default=0.98,
        help="Early-stop: skip remaining restarts once one clears this fidelity. "
        "Essential in the near-degenerate manifold (gap~1e-5) where L-BFGS-B "
        "otherwise burns every maxiter chasing a flat landscape. Set 1.0 to disable.",
    )
    p.add_argument(
        "--ftol",
        type=float,
        default=1e-9,
        help="L-BFGS-B relative tolerance (default 1e-9, loosened for the flat "
        "degenerate regime so it stops chasing infinitesimal energy gains).",
    )
    p.add_argument(
        "--micro-descent",
        type=int,
        default=40,
        help="Bounded L-BFGS-B iters to score the warm-start candidate (default "
        "40). At N=20 the adaptive (None) budget over 2^20 costs more than the "
        "convergence itself; a small fixed budget just ranks the analytic seed.",
    )
    args = p.parse_args(argv)

    from qmbp_simulation.execution import NoiselessBackend
    from qmbp_simulation.execution.eval_cache import CachedBackend, EvalCache

    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(args.topology, args.n, args.h, args.j2, 1)
    nn_edges = list(lat.edges)
    nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
    manifold, _e0m, phys_gap, m_dim = cached_degenerate_ground_manifold(
        H, topology=args.topology, n_qubits=args.n, model=MODEL, h=args.h, j2=args.j2
    )
    degenerate = m_dim > 1

    # Symmetry group + reduction (persist the report alongside the run).
    group = detect_symmetry_group(nn_edges, nnn_edges, args.n)
    _g, sym_report = analyze_symmetry(nn_edges, nnn_edges, args.n, topology=args.topology, h=args.h, j2=args.j2)
    print(
        f"[tied] N={args.n} h={args.h} gap={gap:.5f} manifold_dim={m_dim} "
        f"symmetry_order={group.order} reduction=×{sym_report.reduction_factor:.2f} "
        f"restarts={args.restarts}",
        flush=True,
    )

    builder = HVACircuitBuilder()
    backend = CachedBackend(
        NoiselessBackend(), topology=args.topology, n_qubits=args.n, model=MODEL, p_layers=1, cache=EvalCache()
    )
    backend.set_h(float(args.h))

    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR,
        out_file=f"symmetry_tied_{args.topology}_N{args.n}_h{args.h:.2f}.json",
        fingerprint={"n_qubits": args.n, "p_layers": None, "model": MODEL, "topology": args.topology, "h": args.h},
        extra={
            "topology": args.topology,
            "N": args.n,
            "h": args.h,
            "J2": args.j2,
            "schema": "symmetry_tied_v1",
            "symmetry": sym_report.to_dict(),
        },
        params={"experiment": "symmetry_tied", "N": args.n, "h": args.h},
        description="N=20 symmetry-tied vs independent ansatz (same 2q, fewer params)",
        rows_ref=rows,
    )

    def _subfid(qc, theta):
        return subspace_fidelity(qc, np.asarray(theta, float), manifold) if degenerate else None

    def _tied_seed(indep_seed, blocks, sel):
        """Compress an independent per-bond warm-start to the tied orbit layout.

        Thin wrapper over the shared orbit-compression helper so study and circuit
        use the SAME contract (one value per orbit per block, orbit mean of the
        independent values); rx_final ties over site orbits. Returns None when the
        independent seed is absent (cold start) — the tied run then cold-starts too.
        """
        if indep_seed is None:
            return None
        return compress_seed_to_orbits(
            indep_seed, blocks, sel.nn_edges, sel.nnn_edges, args.n, group, rx_final=args.rx_final
        )

    for label, blocks, sel in _structures(nn_edges, nnn_edges):
        # One analytic warm-start per structure (independent layout); the tied run
        # reuses it compressed to orbits — SAME physical seed, fair comparison.
        qc_ind, _ = builder.create_bond_resolved_masked(
            args.n, lat, blocks=blocks, bond_selection=sel, rx_final=args.rx_final
        )
        # Warm-start: the structure-aware analytic seed (blocks/rx_final) is the
        # candidate that matters here. At N=20 an unbounded adaptive micro-descent
        # over 2^20 and an auto-donor scan cost more than the convergence itself
        # (no same-N shallow donor exists), so bound the micro-descent and skip
        # the donor scan — the seed quality comes from converge_circuit, not from
        # an expensive seed-selection pass.
        ws = prepare_warmstart(
            qc_ind,
            H,
            psi,
            n_nn=len(sel.nn_edges),
            n_nnn=len(sel.nnn_edges),
            n_qubits=args.n,
            p_layers=1,
            h=args.h,
            J2=args.j2,
            backend=backend,
            topology=args.topology,
            model=MODEL,
            gap=gap,
            target_len=qc_ind.num_parameters,
            auto_donors=False,
            micro_descent=args.micro_descent,
            blocks=blocks,
            rx_final=args.rx_final,
        )
        indep_seed = np.asarray(ws["seed"], float)
        # The structure-aware seed now matches the layout; guard anyway so a
        # length mismatch degrades to a cold start instead of crashing the run.
        if indep_seed.size != qc_ind.num_parameters:
            print(
                f"  [warn] seed len {indep_seed.size} != {qc_ind.num_parameters} ({label}); falling back to cold start",
                flush=True,
            )
            indep_seed = None

        for mode in ("independent", "tied"):
            if mode == "independent":
                qc = qc_ind
                seed = indep_seed
            else:
                qc, _ = builder.create_bond_resolved_symmetry_tied(
                    args.n, lat, blocks=blocks, bond_selection=sel, symmetry_group=group, rx_final=args.rx_final
                )
                seed = _tied_seed(indep_seed, blocks, sel)
            n2q, npar = cx_and_params(qc)
            if seed is not None and seed.size != npar:  # layout mismatch → cold start
                seed = None
            t0 = time.time()
            c0 = time.process_time()
            fid, e_best, runs = converge_circuit(
                qc,
                H,
                psi,
                restarts=args.restarts,
                maxiter=args.maxiter,
                seed0=args.seed0,
                warm_theta=seed,
                backend=backend,
                target_fid=args.target_fid,
                ftol=args.ftol,
            )
            theta_best = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)
            sub = _subfid(qc, theta_best)
            report_fid = sub if sub is not None else fid
            row = {
                "label": f"{label}_{mode}",
                "mode": mode,
                "structure": label,
                "h": round(float(args.h), 2),
                "n_qubits": args.n,
                "topology": args.topology,
                "blocks": blocks,
                "e0": e0,
                "gap": gap,
                "phys_gap": phys_gap,
                "manifold_dim": m_dim,
                "degenerate": degenerate,
                "e_best": float(e_best),
                "best_fidelity": float(fid),
                "subspace_fidelity": sub,
                "report_fidelity": float(report_fid),
                "n_2q_transpiled": n2q,
                "n_params": npar,
                "fidelity_per_cx": (report_fid / n2q) if n2q else None,
                "seconds": round(time.time() - t0, 1),
                "cpu_seconds": round(time.process_time() - c0, 1),
            }
            rows.append(row)
            persister.persist(rows)
            print(
                f"  {label:16s} {mode:11s} fid={report_fid:.4f} 2q={n2q:3d} "
                f"params={npar:3d} ({row['cpu_seconds']}s cpu)",
                flush=True,
            )

    persister.persist(rows, status="final")
    backend.flush()

    # Pairwise summary: tied vs independent at identical 2q.
    print("\n=== TIED vs INDEPENDENT (same 2q) ===", flush=True)
    by_struct: dict[str, dict] = {}
    for r in rows:
        by_struct.setdefault(r["structure"], {})[r["mode"]] = r
    for s, pair in by_struct.items():
        ind, tie = pair.get("independent"), pair.get("tied")
        if ind and tie:
            d = tie["report_fidelity"] - ind["report_fidelity"]
            winner = "TIED" if d > 0 else "indep"
            print(
                f"  {s:16s} 2q={ind['n_2q_transpiled']:3d}  indep={ind['report_fidelity']:.4f} "
                f"({ind['n_params']}p)  tied={tie['report_fidelity']:.4f} ({tie['n_params']}p)  "
                f"Δ={d:+.4f} → {winner}",
                flush=True,
            )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
