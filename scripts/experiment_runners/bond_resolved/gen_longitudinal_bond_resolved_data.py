"""Generate VQE training NPZ for the bond-resolved longitudinal TFIM.

Runs a descending-h VQE sweep with the bond-resolved longitudinal ansatz
(RZZ per-bond + RX + RZ per-site) against H = -J·ZZ - h·X - g·Z, and writes one
NPZ per N in the canonical training layout consumed by MultiNAggregator
(data/multi_n_training/<model>/<topology>_N<n>_p<p>.npz).

Only data generation — no MPNN training. Warm-start chains θ from the previous
(higher) h to the next, mirroring the pipeline's Fase 2 strategy.

Usage:
    python scripts/experiment_runners/bond_resolved/gen_longitudinal_bond_resolved_data.py \
        --topology heavy_hex --n-values 6 8 10 12 16 --g 0.3 --p-layers 1
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation.execution.backends import NoiselessBackend
from qmbp_simulation.framework.result_io import training_npz_path
from qmbp_simulation.models import make_lattice
from qmbp_simulation.models.model_registry import get_model_spec
from qmbp_simulation.optimizers.vqe import VQEOptimizer
from qmbp_simulation.solvers.classical import ClassicalSolver
from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

MODEL = "tfim_bond_resolved_longitudinal"


def solve_ground_truth(n_qubits, topology, j, h_values, solver, spec, gt):
    """Resolve ground truth once per (N, h): energy, gap and state vector.

    The ground truth is independent of the VQE seed, so it is computed once
    and reused across seeds. Energy/gap go through GroundTruthCache; the state
    vector (needed for fidelity) is kept in-memory for this run only.
    Returns a dict h -> {e_exact, gap, gs}.
    """
    model_kwargs = {"g": spec.hamiltonian_kwargs.get("g", 0.0)}
    cache = {}
    for h in h_values:
        lat_h = make_lattice(topology, n_qubits, J=j, h=h)
        H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)
        e_exact, gap = gt.get_or_compute(
            topology, n_qubits, MODEL, round(float(h), 2),
            flush=False, model_kwargs=model_kwargs, solver=solver,
        )
        res_gt = solver.solve(H, lat_h)
        cache[round(float(h), 2)] = {
            "H": H,
            "e_exact": float(e_exact),
            "gap": gap if gap > 1e-10 else 0.1,
            "gs": getattr(res_gt, "ground_state", None),
        }
    gt.flush()
    return cache


def generate_for_n(n_qubits, topology, p_layers, j, h_values, backend, spec, seed, gt_cache):
    """Run a descending-h VQE sweep for one (N, seed), reusing precomputed GT."""
    lattice = make_lattice(topology, n_qubits, J=j, h=2.0)
    circuit, theta_param = spec.create_circuit(n_qubits, p_layers, lattice)
    n_params = len(theta_param)
    rng = np.random.default_rng(seed)
    vqe = VQEOptimizer(config=_vqe_config(p_layers, seed), backend=backend, seed=seed)

    rows_h, rows_theta, rows_e_vqe, rows_e_exact, rows_gap, rows_fid, rows_degap = (
        [], [], [], [], [], [], []
    )
    prev_theta = None
    for h in h_values:
        gtp = gt_cache[round(float(h), 2)]
        H, e_exact, gap, gs = gtp["H"], gtp["e_exact"], gtp["gap"], gtp["gs"]
        # Warm-start from previous h (descending sweep), else random
        x0 = prev_theta.copy() if prev_theta is not None else rng.uniform(-np.pi, np.pi, n_params)
        result = vqe.optimize(H, circuit, x0, exact_energy=e_exact, exact_state=gs)
        theta_opt = np.asarray(result.theta_opt, dtype=np.float64)
        e_vqe = float(result.energy)
        prev_theta = theta_opt
        fid = float(result.fidelity) if result.fidelity else float("nan")
        de_gap = abs(e_vqe - e_exact) / max(gap, 1e-10)
        rows_h.append(round(float(h), 2))
        rows_theta.append(theta_opt)
        rows_e_vqe.append(e_vqe)
        rows_e_exact.append(e_exact)
        rows_gap.append(gap)
        rows_fid.append(fid)
        rows_degap.append(de_gap)
        print(
            f"  N={n_qubits} seed={seed} h={h:.2f} | de/gap={de_gap:.4f} F={fid:.3f} "
            f"E_vqe={e_vqe:.5f} E_exact={e_exact:.5f} k={result.n_iterations}",
            flush=True,
        )

    tier = ["verified" if f >= 0.93 else "unverified" for f in rows_fid]
    return {
        "h_values": np.array(rows_h, dtype=np.float64),
        "theta_opt": np.array(rows_theta, dtype=object),
        "e_vqe": np.array(rows_e_vqe, dtype=np.float64),
        "e_exact": np.array(rows_e_exact, dtype=np.float64),
        "gaps": np.array(rows_gap, dtype=np.float64),
        "fidelities": np.array(rows_fid, dtype=np.float64),
        "de_gaps": np.array(rows_degap, dtype=np.float64),
        "quality_tier": np.array(tier, dtype=object),
        "method": np.array(["vqe_warm"] * len(rows_h), dtype=object),
        "g": float(spec.hamiltonian_kwargs.get("g", 0.0)),
        "n_params": int(n_params),
        "seed": int(seed),
    }


def _vqe_config(p_layers, seed):
    from qmbp_simulation.models.data_models import VQEConfig

    return VQEConfig(p_layers=p_layers, method="L-BFGS-B", n_restarts=5, maxiter=1500)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=str, default="heavy_hex")
    parser.add_argument("--n-values", type=int, nargs="+", default=[6, 8, 10, 12, 16])
    parser.add_argument("--g", type=float, default=0.3)
    parser.add_argument("--j", type=float, default=1.0)
    parser.add_argument("--p-layers", type=int, default=1)
    parser.add_argument("--h-min", type=float, default=1.3)
    parser.add_argument("--h-max", type=float, default=5.0)
    parser.add_argument("--h-points", type=int, default=40)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="VQE seed(s). One → canonical training NPZ. Several → per-seed NPZs "
        "under a _multiseed/ subdir (for dispersion analysis, not the main "
        "training corpus). Ground truth is solved once per (N,h) and reused.",
    )
    args = parser.parse_args()

    spec = get_model_spec(MODEL).with_params(g=args.g)
    backend = NoiselessBackend()
    solver = ClassicalSolver()
    gt = GroundTruthCache()
    # Descending sweep (high h → low h) for warm-start continuity.
    h_values = [round(h, 2) for h in np.linspace(args.h_max, args.h_min, args.h_points)]
    multi_seed = len(args.seeds) > 1

    print(
        f"Generating {MODEL} data | {args.topology} g={args.g} p={args.p_layers} "
        f"N={args.n_values} seeds={args.seeds} | h∈[{args.h_min},{args.h_max}] "
        f"({args.h_points} pts)",
        file=sys.stderr,
    )
    for n in args.n_values:
        print(f"\n── N={n} ──", flush=True)
        # Ground truth once per (N, h) — reused across all seeds.
        gt_cache = solve_ground_truth(n, args.topology, args.j, h_values, solver, spec, gt)
        for seed in args.seeds:
            payload = generate_for_n(
                n, args.topology, args.p_layers, args.j, h_values,
                backend, spec, seed, gt_cache,
            )
            if multi_seed:
                # Per-seed NPZ in a sibling _multiseed/ dir so the aggregator's
                # {topo}_N*_p{p} scan (which parses N from the name) is not
                # confused by a seed suffix.
                base = training_npz_path(
                    args.topology, n, args.p_layers, model=MODEL, for_write=True
                )
                out = base.parent / "_multiseed" / (
                    f"{args.topology}_N{n}_p{args.p_layers}_seed{seed}.npz"
                )
            else:
                out = training_npz_path(
                    args.topology, n, args.p_layers, model=MODEL, for_write=True
                )
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, timestamp=datetime.now(UTC).isoformat(), **payload)
            print(f"  saved → {out}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
