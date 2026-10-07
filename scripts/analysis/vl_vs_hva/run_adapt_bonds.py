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
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import save_json  # noqa: E402

from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.framework.study_checkpoint import build_resumable_payload  # noqa: E402
from qmbp_simulation.framework.study_core import (  # noqa: E402
    ground_state,
    grow_bonds_adapt,
)
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402

SUBDIR = "bond_ablation"


def run(args) -> int:
    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(args.topology, args.n, args.h, args.j2, 1)
    nn_all = list(lat.edges)
    nnn_all = HamiltonianBuilder._generate_nnn_edges(lat)
    builder = HVACircuitBuilder()
    out_file = f"adapt_bonds_{args.topology}_N{args.n}_h{args.h:.2f}.json"

    print(
        f"[adapt_bonds] N={args.n} h={args.h} e0={e0:.5f} gap={gap:.5f} "
        f"n_nn={n_nn} n_nnn={n_nnn} target_fid={args.target_fid} "
        f"grow_step={args.grow_step} warm_per_step={not args.no_warm_per_step} "
        f"efficiency_stop={args.efficiency_stop}",
        flush=True,
    )

    def _persist(steps, status="partial"):
        payload = build_resumable_payload(
            rows=steps,
            theta=(steps[-1]["best_theta_final"] if steps else None),
            fingerprint={
                "n_qubits": args.n,
                "p_layers": 1,
                "model": "tfim_frustrated",
                "topology": args.topology,
                "h": args.h,
            },
            extra={
                "topology": args.topology,
                "N": args.n,
                "h": args.h,
                "p_layers": 1,
                "J2": args.j2,
                "e0": e0,
                "gap": gap,
                "target_fid": args.target_fid,
                "grow_step": args.grow_step,
                "grad_tol": args.grad_tol,
                "n_nnn_total": n_nnn,
                "warm_per_step": not args.no_warm_per_step,
                "efficiency_stop": args.efficiency_stop,
                "schema": "adapt_bonds_v1",
                "status": status,
            },
        )
        save_json(
            payload,
            SUBDIR,
            out_file,
            params={"experiment": "adapt_bonds", "N": args.n, "h": args.h},
            description="ADAPT bond growth — cheapest masked HVA reaching a "
            "fidelity target (gradient-guided nnn insertion)",
        )

    steps_ref: list[dict] = []

    def _on_step(step):
        # grow_bonds_adapt appends to its own list; mirror here for crash-safe persist.
        steps_ref.append(step)
        _persist(steps_ref)
        print(
            f"  step{step['step']:>2} bonds={step['n_bonds']:>3} "
            f"(nnn={step['n_nnn_bonds']}) fid={step['best_fidelity']:.4f} "
            f"2q={step['n_2q_transpiled']} fid/cx={step['fidelity_per_cx']:.2e} "
            f"({step['seconds']}s)",
            flush=True,
        )

    steps, reached = grow_bonds_adapt(
        builder,
        args.n,
        lat,
        H,
        psi,
        e0,
        gap,
        nn_all=nn_all,
        nnn_all=nnn_all,
        n_nn=n_nn,
        n_nnn=n_nnn,
        restarts=args.restarts,
        maxiter=args.maxiter,
        seed0=args.seed0,
        target_fid=args.target_fid,
        grow_step=args.grow_step,
        grad_tol=args.grad_tol,
        warm_per_step=not args.no_warm_per_step,
        efficiency_stop=args.efficiency_stop,
        efficiency_patience=args.efficiency_patience,
        on_step=_on_step,
    )

    _persist(steps, status="final")

    print("\n=== ADAPT growth curve (fid | 2q | fid/cx) ===", flush=True)
    for s in steps:
        print(
            f"  step{s['step']:>2} bonds={s['n_bonds']:>3} fid={s['best_fidelity']:.4f} "
            f"2q={s['n_2q_transpiled']:>4} fid/cx={s['fidelity_per_cx']:.2e}",
            flush=True,
        )
    stop = steps[-1].get("stop_reason") if steps else "no_steps"
    tag = "TARGET REACHED" if reached else f"target not reached (stop: {stop})"
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
    p.add_argument("--target-fid", type=float, default=0.90, help="Stop once fidelity reaches this.")
    p.add_argument("--grow-step", type=int, default=2, help="How many top-gradient bonds to add per iteration.")
    p.add_argument("--grad-tol", type=float, default=1e-6, help="Ignore candidate bonds with |grad| below this.")
    p.add_argument(
        "--no-warm-per-step",
        action="store_true",
        help="Disable warm-starting each growth step from the previous step's θ (restart cold instead).",
    )
    p.add_argument(
        "--efficiency-stop",
        action="store_true",
        help="Also stop when fidelity-per-CX stops improving (Pareto point), not only at the target-fid crossing.",
    )
    p.add_argument(
        "--efficiency-patience", type=int, default=1, help="Consecutive non-improving fid/CX steps before stopping."
    )
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
