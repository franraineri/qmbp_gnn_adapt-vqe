#!/usr/bin/env python
"""Dense h-sweep of optimized θ for the bond-resolved frustrated HVA.

Generates a high-fidelity θ dataset across the frustrated transition on a small,
exact-tractable system (default N=8 square, J2=0.5) so the angle patterns and
symmetries can be studied densely. For each h it optimizes θ via the shared
``StudyRunner`` (regime-gated strategy: second-order seed inside the transition
window, Metropolis otherwise) and stores the full θ vector plus per-h metrics.

Saves a single traceable JSON (per-h θ + fidelity + gap) under the organized
tree via the study service, and per-h θ NPZ artifacts. Crash-safe + resumable
via ``StudyCheckpoint``.

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/theta_pattern_sweep.py \
        --n 8 --p 1 --strategy bestof --hmin 0.2 --hmax 2.5 --nh 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import save_json, study_dir  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dense h-sweep of optimized θ (pattern study)")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--p", type=int, default=1)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--model", default="tfim_frustrated")
    p.add_argument("--strategy", default="bestof",
                   choices=["first_order", "second_order", "bestof", "metropolis"])
    p.add_argument("--hmin", type=float, default=0.2)
    p.add_argument("--hmax", type=float, default=2.5)
    p.add_argument("--nh", type=int, default=20, help="number of h points")
    p.add_argument("--maxiter", type=int, default=500)
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--seeds", type=int, default=1,
                   help="number of independent optimization seeds per h "
                        "(>1 stores all θ so the analysis can report θ(h) ± σ)")
    return p


def _multiseed_point(runner, h, *, strategy, seeds, maxiter, sigma, j2):
    """Optimize one h from ``seeds`` independent seeds; return a per-h record.

    Reuses the runner's ground state + circuit + backend, then calls
    ``run_strategy`` once per seed with a distinct ``seed0`` (so the perturbed
    restarts explore different noise draws — genuine seed variation). Stores
    every seed's optimized θ + fidelity plus the best, so the analysis can report
    θ(h) ± σ across seeds.
    """
    from qmbp_simulation.framework.study_runner import run_strategy
    from qmbp_simulation.models import make_lattice

    psi, e0, gap, H, gs_method = runner.ground_state(h)
    lat = make_lattice(runner.topology, runner.n_qubits, J=runner.J, h=h)
    qc, n_nn, n_nnn = runner.build_circuit(lat)
    grad = runner._make_adjoint_grad(qc, H)

    def cost(x):
        return runner.backend.evaluate(qc, H, x)

    def fidelity(x):
        from qmbp_simulation.framework.study_runner import _state_fidelity_exact
        return _state_fidelity_exact(qc, x, psi)

    seed_thetas, seed_fids, seed_energies = [], [], []
    best = None
    for s in range(seeds):
        res = run_strategy(
            strategy, cost=cost, fidelity=fidelity, n_nn=n_nn, n_nnn=n_nnn,
            n_qubits=runner.n_qubits, p_layers=runner.p_layers, h=h,
            J=runner.J, J2=(j2 if runner.ansatz == "nnn" else 0.0),
            maxiter=maxiter, sigma=sigma, temperature=gap,
            # distinct base seed per seed index → different perturbation draws
            seed0=100_000 + 1000 * s, grad=grad,
        )
        seed_thetas.append(np.asarray(res.best_theta).tolist())
        seed_fids.append(float(res.fidelity))
        seed_energies.append(float(res.energy))
        if best is None or res.energy < best["e_vqe"]:
            best = {"e_vqe": float(res.energy), "fidelity": float(res.fidelity),
                    "best_theta": np.asarray(res.best_theta).tolist(),
                    "strategy_resolved": res.strategy}

    return {
        "topology": runner.topology, "h": h, "n_qubits": runner.n_qubits,
        "p_layers": runner.p_layers, "J2": j2, "model": runner.model,
        "ansatz": runner.ansatz, "gs_method": gs_method,
        "e0_exact": float(e0), "gap": float(gap),
        "e_vqe": best["e_vqe"], "fidelity": best["fidelity"],
        "abs_error": float(abs(best["e_vqe"] - e0)),
        "de_gap": float(abs(best["e_vqe"] - e0) / gap) if gap > 0 else None,
        "n_params": int(qc.num_parameters), "n_nn": n_nn, "n_nnn": n_nnn,
        "strategy_requested": strategy, "strategy_resolved": best["strategy_resolved"],
        "n_seeds": seeds,
        "best_theta": best["best_theta"],
        "seed_thetas": seed_thetas,        # all seeds' optimized θ (for ±σ)
        "seed_fidelities": seed_fids,
        "seed_energies": seed_energies,
    }


def main(argv=None) -> int:
    from qmbp_simulation.framework.study_checkpoint import StudyCheckpoint
    from qmbp_simulation.framework.study_runner import StudyRunner

    args = build_parser().parse_args(argv)
    h_values = [round(h, 2) for h in np.linspace(args.hmin, args.hmax, args.nh)]

    subdir = "theta_patterns"
    seed_tag = "" if args.seeds <= 1 else f"_seeds{args.seeds}"
    tag = f"{args.topology}_N{args.n}_p{args.p}_J2{args.j2:.2f}_{args.strategy}{seed_tag}"
    out_file = f"theta_sweep_{tag}.json"
    out_path = study_dir(subdir) / out_file
    art_dir = study_dir(subdir) / f"theta_{tag}"

    runner = StudyRunner(
        topology=args.topology, n_qubits=args.n, p_layers=args.p, j2=args.j2,
        model=args.model, strategy=args.strategy, maxiter=args.maxiter,
        sigma=args.sigma, ansatz="nnn",
    )

    def writer(payload):
        return save_json(payload, subdir, out_file,
                         params={"N": args.n, "p_layers": args.p, "J2": args.j2,
                                 "model": args.model, "strategy": args.strategy,
                                 "maxiter": args.maxiter, "sigma": args.sigma,
                                 "seeds": args.seeds, "h_values": h_values},
                         description=f"Dense θ h-sweep for pattern/symmetry study: "
                                     f"{args.topology} N={args.n} p={args.p} nnn "
                                     f"J2={args.j2}, strategy={args.strategy}, "
                                     f"seeds={args.seeds}")

    ckpt = StudyCheckpoint("per_h", writer=writer, path=out_path)
    n_resumed = ckpt.resume()
    if n_resumed:
        print(f"[resume] {n_resumed} h already done", flush=True)

    print(f"[theta-sweep] {args.topology} N={args.n} p={args.p} nnn J2={args.j2} "
          f"strategy={args.strategy} seeds={args.seeds} "
          f"h={args.hmin}..{args.hmax} ({args.nh} pts)", flush=True)

    def on_point(pt):
        extra = ""
        if pt.get("n_seeds", 1) > 1:
            fids = pt.get("seed_fidelities", [])
            extra = f" seed_fid[min={min(fids):.3f} max={max(fids):.3f}]"
        print(f"[h={pt['h']:.2f}] strat={pt['strategy_resolved']:12s} "
              f"fid={pt['fidelity']:.4f} dE/gap="
              f"{pt['de_gap'] if pt['de_gap'] is None else round(pt['de_gap'], 3)} "
              f"gap={pt['gap']:.4f}{extra}", flush=True)

    if args.seeds <= 1:
        runner.run_sweep(
            h_values, ckpt, artifact_dir=art_dir,
            extra={"schema": "theta_pattern_sweep_v1"},
            on_point=on_point,
        )
    else:
        # Multi-seed path: run_sweep drives one point per h; we compute the
        # multi-seed record and persist it through the same checkpoint (resume-safe).
        ckpt.resume()
        for h in h_values:
            key = f"{h:.2f}"
            if ckpt.is_done(key):
                continue
            rec = _multiseed_point(runner, h, strategy=args.strategy,
                                   seeds=args.seeds, maxiter=args.maxiter,
                                   sigma=args.sigma, j2=args.j2)
            on_point(rec)
            ckpt.record(key, rec)
            ckpt.persist(extra={"schema": "theta_pattern_sweep_multiseed_v1"})

    print(f"\n→ θ dataset: {out_path}")
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
