"""Compare MPS deterministic vs stochastic mode at N=40, 50, 80.

Validates that the new deterministic mode produces results consistent with
the old stochastic mode (which produced all existing scaling results).

For each N, runs VQE at 1 h-point in both modes and compares:
- Energy difference between modes
- ΔE/gap in each mode
- Time per evaluation

Expected outcome: deterministic gives slightly better ΔE/gap (no shot noise)
and is ~100-375× faster per evaluation.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)
logging.getLogger("tenpy").setLevel(logging.WARNING)

from qmbp_simulation import ClassicalSolver, HamiltonianBuilder, make_lattice
from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.execution import MPSBackend
from qmbp_simulation.models import VQEConfig
from qmbp_simulation.optimizers import VQEOptimizer

# Test configurations matching existing scaling results
TEST_CASES = [
    {"N": 40, "h": 5.5, "label": "N=40 (existing result)"},
    {"N": 50, "h": 6.5, "label": "N=50 (existing result)"},
    {"N": 80, "h": 9.5, "label": "N=80 (existing result)"},
]

builder = HamiltonianBuilder()
solver = ClassicalSolver()
hva = HVACircuitBuilder()

print("=" * 70)
print("  MPS MODE COMPARISON: Deterministic vs Stochastic")
print("=" * 70)
print()

results = []

for case in TEST_CASES:
    N = case["N"]
    h = case["h"]
    label = case["label"]
    print(f"--- {label} ---")

    lattice = make_lattice("chain_1d", N, J=1.0, h=h)
    H = builder.build(lattice)
    gt = solver.solve(H, lattice, method="dmrg")
    circuit, _ = hva.create(N, 1, lattice)

    e_exact = gt.ground_energy
    gap = gt.gap
    print(f"  DMRG: E0={e_exact:.6f}, gap={gap:.4f}")

    # ── Deterministic mode (new default) ──────────────────────────────
    backend_det = MPSBackend(strategy="aer_mps", chi_max=64, deterministic=True, seed=42)
    config = VQEConfig(
        method="COBYLA",
        p_layers=1,
        n_restarts=1,
        maxiter=100,
        enable_callbacks=False,
    )
    opt_det = VQEOptimizer(config=config, backend=backend_det, seed=42)

    rng = np.random.default_rng(42)
    theta_init = rng.uniform(-0.01, 0.01, 2)

    t0 = time.time()
    res_det = opt_det.optimize(H, circuit, theta_init.copy(), exact_energy=e_exact)
    t_det = time.time() - t0
    de_det = abs(res_det.energy - e_exact) / max(gap, 1e-10)

    print(
        f"  Deterministic: E={res_det.energy:.8f}, dE/gap={de_det:.6f}, "
        f"iters={res_det.n_iterations}, time={t_det:.1f}s"
    )

    # ── Stochastic mode (old behavior) ────────────────────────────────
    backend_sto = MPSBackend(
        strategy="aer_mps",
        chi_max=64,
        precision=0.005,
        deterministic=False,
        seed=42,
    )
    opt_sto = VQEOptimizer(config=config, backend=backend_sto, seed=42)

    t0 = time.time()
    res_sto = opt_sto.optimize(H, circuit, theta_init.copy(), exact_energy=e_exact)
    t_sto = time.time() - t0
    de_sto = abs(res_sto.energy - e_exact) / max(gap, 1e-10)

    print(
        f"  Stochastic:    E={res_sto.energy:.8f}, dE/gap={de_sto:.6f}, "
        f"iters={res_sto.n_iterations}, time={t_sto:.1f}s"
    )

    # ── Comparison ────────────────────────────────────────────────────
    energy_diff = abs(res_det.energy - res_sto.energy)
    speedup = t_sto / max(t_det, 0.001)
    theta_diff = np.linalg.norm(res_det.theta_opt - res_sto.theta_opt)

    print(f"  Energy diff: {energy_diff:.6f}")
    print(f"  Theta L2 diff: {theta_diff:.6f}")
    print(f"  Speedup: {speedup:.1f}x")
    print(
        f"  Both pass 5%: det={'PASS' if de_det < 0.05 else 'FAIL'}, "
        f"sto={'PASS' if de_sto < 0.05 else 'FAIL'}"
    )
    print()

    results.append(
        {
            "N": N,
            "h": h,
            "e_exact": float(e_exact),
            "gap": float(gap),
            "deterministic": {
                "energy": float(res_det.energy),
                "de_gap": float(de_det),
                "n_iterations": res_det.n_iterations,
                "theta_opt": res_det.theta_opt.tolist(),
                "time_s": round(t_det, 2),
                "passed": de_det < 0.05,
            },
            "stochastic": {
                "energy": float(res_sto.energy),
                "de_gap": float(de_sto),
                "n_iterations": res_sto.n_iterations,
                "theta_opt": res_sto.theta_opt.tolist(),
                "time_s": round(t_sto, 2),
                "passed": de_sto < 0.05,
            },
            "comparison": {
                "energy_diff": float(energy_diff),
                "theta_l2_diff": float(theta_diff),
                "speedup_factor": round(speedup, 1),
                "de_gap_improvement": float(de_sto - de_det),
            },
        }
    )

# ── Summary ───────────────────────────────────────────────────────────
print("=" * 70)
print("  SUMMARY")
print("=" * 70)
all_det_pass = all(r["deterministic"]["passed"] for r in results)
all_sto_pass = all(r["stochastic"]["passed"] for r in results)
mean_speedup = np.mean([r["comparison"]["speedup_factor"] for r in results])
mean_energy_diff = np.mean([r["comparison"]["energy_diff"] for r in results])
# Use relative metric: energy_diff / gap (what matters for thesis)
mean_relative_diff = np.mean(
    [r["comparison"]["energy_diff"] / max(r["gap"], 1e-10) for r in results]
)

print(f"  All deterministic pass: {all_det_pass}")
print(f"  All stochastic pass: {all_sto_pass}")
print(f"  Mean speedup: {mean_speedup:.1f}x")
print(f"  Mean energy diff: {mean_energy_diff:.6f}")
print(f"  Mean relative diff (ΔE/gap): {mean_relative_diff * 100:.2f}%")
print(f"  Conclusion: Modes are {'CONSISTENT' if mean_relative_diff < 0.05 else 'DIVERGENT'}")
print()

# Save
output = {
    "experiment": "mps_mode_comparison",
    "description": "Deterministic vs stochastic MPS evaluation at N=40/50/80",
    "metadata": {
        "mps_evaluation_modes_compared": ["deterministic", "stochastic"],
        "deterministic_precision": "exact (machine epsilon)",
        "stochastic_precision": 0.005,
        "date": "2026-06-10",
        "note": "Results before 2026-06-10 used stochastic mode by default.",
    },
    "results": results,
    "summary": {
        "all_det_pass": all_det_pass,
        "all_sto_pass": all_sto_pass,
        "mean_speedup": round(mean_speedup, 1),
        "mean_energy_diff": float(mean_energy_diff),
        "mean_relative_diff_de_gap": float(mean_relative_diff),
        "modes_consistent": mean_relative_diff < 0.05,
        "note": "Modes are consistent if relative diff < 5% of gap",
    },
}
out_path = Path("results/scaling/mps_mode_comparison.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(
        output, f, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)
    )
print(f"  Saved to {out_path}")
