#!/usr/bin/env python
"""Compare standard bond-resolved HVA vs bond-resolved longitudinal HVA (per-site RZ).

Both ansätze are optimized against the SAME pure TFIM ground state
(H = -J·ZZ - h·X, built with HamiltonianBuilder.build → g=0). The per-site RZ
block of the longitudinal ansatz is treated purely as an ANSATZ degree of freedom,
NOT as a term added to the target Hamiltonian. This isolates the single question:
which ansatz approximates the same TFIM ground state better at fixed depth/p?

Everything is ideal statevector simulation (NoiselessBackend). No noise, no hardware.

Controlled comparison — for each (ansatz, h) the following are held identical:
topology, N, p, h, J, backend, seeds, and VQE optimizer config (method, maxiter,
n_restarts). The only variable is the ansatz (create_bond_resolved vs
create_bond_resolved_longitudinal).

Reuses production modules only: HVACircuitBuilder, HamiltonianBuilder, make_lattice,
ClassicalSolver, VQEOptimizer + NoiselessBackend, circuit_summary.

Usage:
    .venv/bin/python scripts/analysis/compare_hva_std_vs_longitudinal.py --anchor-only
    .venv/bin/python scripts/analysis/compare_hva_std_vs_longitudinal.py \
        --topologies heavy_hex --n-qubits 10 --seeds 3
    .venv/bin/python scripts/analysis/compare_hva_std_vs_longitudinal.py \
        --topologies heavy_hex chain_1d --n-qubits 6 8 10 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from qmbp_simulation import (
    ClassicalSolver,
    HamiltonianBuilder,
    VQEConfig,
    VQEOptimizer,
    make_lattice,
)
from qmbp_simulation.analysis.circuit_visualizer import circuit_summary
from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.execution.backends import NoiselessBackend

# Default sweep crossing the transition; h=1.0 = h_c (hardest point).
DEFAULT_H_VALUES = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

# Anchor: verified in a prior session (heavy_hex N=10 p=1 h=1.0 std bond-resolved).
ANCHOR = {
    "topology": "heavy_hex",
    "n_qubits": 10,
    "p_layers": 1,
    "h": 1.0,
    "ansatz_min": -12.0806,
    "e0": -12.4722,
    "gap": 0.2201,
    "fidelity": 0.796,
    "tol": 0.01,
}

ANSATZE = {
    "std": "create_bond_resolved",
    "longitudinal": "create_bond_resolved_longitudinal",
}


@dataclass
class PointResult:
    """One (ansatz, topology, N, p, h) optimization outcome, aggregated over seeds."""

    ansatz: str
    topology: str
    n_qubits: int
    p_layers: int
    h: float
    e0: float
    gap: float
    # aggregated over seeds (best restart per seed, then mean±std across seeds)
    e_ansatz_min_mean: float
    e_ansatz_min_std: float
    e_ansatz_min_best: float
    abs_de_mean: float
    abs_de_std: float
    de_over_gap_mean: float
    fidelity_mean: float
    fidelity_std: float
    fidelity_best: float
    n_params: int
    n_2q_gates: int
    depth: int
    n_iterations_mean: float
    theta_best: list[float]


def _build_circuit(hva: HVACircuitBuilder, ansatz: str, n_qubits: int, p_layers: int, lattice):
    method = getattr(hva, ANSATZE[ansatz])
    return method(n_qubits, p_layers, lattice)


def _optimize_ansatz(
    ansatz: str,
    topology: str,
    n_qubits: int,
    p_layers: int,
    h: float,
    seeds: list[int],
    maxiter: int,
    n_restarts: int,
) -> PointResult:
    """Optimize one ansatz for one h across multiple seeds; keep best restart per seed."""
    hva = HVACircuitBuilder()
    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    backend = NoiselessBackend()

    lattice = make_lattice(topology, n_qubits, J=1.0, h=h)
    # Target Hamiltonian = PURE TFIM (g=0) for BOTH ansätze.
    hamiltonian = builder.build(lattice)
    ground_truth = solver.solve(hamiltonian, lattice)
    e0 = float(ground_truth.ground_energy)
    gap = float(ground_truth.gap)
    exact_state = ground_truth.ground_state

    circuit, theta = _build_circuit(hva, ansatz, n_qubits, p_layers, lattice)
    n_params = circuit.num_parameters

    summary = circuit_summary(circuit)
    n_2q_gates = int(summary["n_2q_gates"])
    depth = int(summary["depth"])

    energies: list[float] = []
    fidelities: list[float] = []
    n_iters: list[int] = []
    thetas: list[np.ndarray] = []

    for seed in seeds:
        cfg = VQEConfig(
            p_layers=p_layers,
            n_restarts=n_restarts,
            maxiter=maxiter,
            method="L-BFGS-B",
            enable_callbacks=False,
        )
        optimizer = VQEOptimizer(config=cfg, backend=backend, seed=seed)
        # Small warm-start near |+>^N (θ≈0) for reproducibility; restarts explore.
        rng = np.random.default_rng(seed)
        init = rng.uniform(-0.05, 0.05, n_params)
        result = optimizer.optimize(
            hamiltonian, circuit, init, exact_energy=e0, exact_state=exact_state
        )
        energies.append(float(result.energy))
        fidelities.append(float(result.fidelity))
        n_iters.append(int(result.n_iterations))
        thetas.append(np.asarray(result.theta_opt, dtype=float))

    energies_arr = np.array(energies)
    fidelities_arr = np.array(fidelities)
    best_idx = int(np.argmin(energies_arr))

    abs_de = np.abs(energies_arr - e0)
    de_over_gap = abs_de / max(gap, 1e-10)

    return PointResult(
        ansatz=ansatz,
        topology=topology,
        n_qubits=n_qubits,
        p_layers=p_layers,
        h=round(h, 2),
        e0=e0,
        gap=gap,
        e_ansatz_min_mean=float(energies_arr.mean()),
        e_ansatz_min_std=float(energies_arr.std()),
        e_ansatz_min_best=float(energies_arr.min()),
        abs_de_mean=float(abs_de.mean()),
        abs_de_std=float(abs_de.std()),
        de_over_gap_mean=float(de_over_gap.mean()),
        fidelity_mean=float(fidelities_arr.mean()),
        fidelity_std=float(fidelities_arr.std()),
        fidelity_best=float(fidelities_arr.max()),
        n_params=int(n_params),
        n_2q_gates=n_2q_gates,
        depth=depth,
        n_iterations_mean=float(np.mean(n_iters)),
        theta_best=thetas[best_idx].tolist(),
    )


def run_anchor_check(maxiter: int, n_restarts: int) -> bool:
    """Reproduce the verified anchor number before sweeping. Returns True if within tol."""
    print("=" * 72)
    print("ANCHOR CHECK — heavy_hex N=10 p=1 h=1.0, standard bond-resolved HVA (ideal)")
    print("=" * 72)
    res = _optimize_ansatz(
        ansatz="std",
        topology=ANCHOR["topology"],
        n_qubits=ANCHOR["n_qubits"],
        p_layers=ANCHOR["p_layers"],
        h=ANCHOR["h"],
        seeds=[0, 1, 2],
        maxiter=maxiter,
        n_restarts=n_restarts,
    )
    got_min = res.e_ansatz_min_best
    got_e0 = res.e0
    got_fid = res.fidelity_best
    d_min = abs(got_min - ANCHOR["ansatz_min"])
    d_e0 = abs(got_e0 - ANCHOR["e0"])
    print(f"  ansatz_min: got {got_min:.4f}  expected {ANCHOR['ansatz_min']:.4f}  |Δ|={d_min:.4f}")
    print(f"  E0        : got {got_e0:.4f}  expected {ANCHOR['e0']:.4f}  |Δ|={d_e0:.4f}")
    print(f"  gap       : got {res.gap:.4f}  expected ~{ANCHOR['gap']:.4f}")
    print(f"  fidelity  : got {got_fid:.4f}  expected ~{ANCHOR['fidelity']:.4f}")
    ok = d_min <= ANCHOR["tol"] and d_e0 <= ANCHOR["tol"]
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} (tol ±{ANCHOR['tol']})")
    print("=" * 72)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies", nargs="+", default=["heavy_hex"])
    parser.add_argument("--n-qubits", nargs="+", type=int, default=[10])
    parser.add_argument("--p-layers", type=int, default=1)
    parser.add_argument("--h-values", nargs="+", type=float, default=DEFAULT_H_VALUES)
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds (0..seeds-1)")
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--n-restarts", type=int, default=8)
    parser.add_argument("--anchor-only", action="store_true")
    parser.add_argument("--skip-anchor", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_project_root / "results" / "analysis" / "hva_std_vs_longitudinal"),
    )
    args = parser.parse_args()

    seeds = list(range(args.seeds))

    if not args.skip_anchor:
        ok = run_anchor_check(args.maxiter, args.n_restarts)
        if not ok:
            print("Anchor check FAILED — stopping before sweep. Revise the setup.")
            return 1
    if args.anchor_only:
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[PointResult] = []
    t0 = time.time()
    for topology in args.topologies:
        for n_qubits in args.n_qubits:
            for h in args.h_values:
                for ansatz in ("std", "longitudinal"):
                    tag = f"{ansatz:12s} {topology:10s} N={n_qubits:>2} p={args.p_layers} h={h:.2f}"
                    print(f"[{time.time()-t0:6.1f}s] optimizing {tag} ...", flush=True)
                    res = _optimize_ansatz(
                        ansatz=ansatz,
                        topology=topology,
                        n_qubits=n_qubits,
                        p_layers=args.p_layers,
                        h=h,
                        seeds=seeds,
                        maxiter=args.maxiter,
                        n_restarts=args.n_restarts,
                    )
                    all_results.append(res)
                    print(
                        f"    E_min(best)={res.e_ansatz_min_best:.4f} "
                        f"|ΔE|(mean)={res.abs_de_mean:.4f} fid(best)={res.fidelity_best:.4f} "
                        f"nparams={res.n_params} 2q={res.n_2q_gates}",
                        flush=True,
                    )

    payload = {
        "meta": {
            "description": "Standard bond-resolved HVA vs bond-resolved longitudinal "
            "HVA, same pure TFIM target (g=0), ideal statevector.",
            "topologies": args.topologies,
            "n_qubits": args.n_qubits,
            "p_layers": args.p_layers,
            "h_values": args.h_values,
            "seeds": seeds,
            "maxiter": args.maxiter,
            "n_restarts": args.n_restarts,
            "backend": "noiseless_statevector",
            "target_hamiltonian": "pure TFIM H=-J*ZZ-h*X (g=0), builder.build",
            "elapsed_s": round(time.time() - t0, 1),
        },
        "results": [asdict(r) for r in all_results],
    }
    results_path = out_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results: {results_path}")

    # Save optimal thetas separately (reproducible artifact).
    thetas_payload = {
        f"{r.ansatz}|{r.topology}|N{r.n_qubits}|p{r.p_layers}|h{r.h:.2f}": r.theta_best
        for r in all_results
    }
    thetas_path = out_dir / "optimal_thetas.json"
    with open(thetas_path, "w") as f:
        json.dump(thetas_payload, f, indent=2)
    print(f"Saved thetas : {thetas_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
