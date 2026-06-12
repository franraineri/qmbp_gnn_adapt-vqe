"""N=120 full VQE sweep — rigorous validation near h_min boundary.

Tests 5 h-points starting from near h_min_safe, with 3 seeds.
Validates the scaling law AT the boundary (not just deep paramagnetic).

Optimization: DMRG is deterministic, so we cache ground-truth per h-value
and only vary VQE seeds. This reduces 15 DMRG calls to 5.

Expected time: ~10-15 min.
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

# ═══════════════════════════════════════════════════════════════════════════════
N = 120
SEEDS = [42, 43, 44]
H_MIN_SAFE = 1.5 + 0.020 * N**1.31  # ~12.09

# 5 h-points: from h_min_safe+0.5 to h_min_safe+3.0
H_VALUES = [
    round(H_MIN_SAFE + 0.5, 2),  # ~12.59 — closest to boundary
    round(H_MIN_SAFE + 1.0, 2),  # ~13.09
    round(H_MIN_SAFE + 1.5, 2),  # ~13.59
    round(H_MIN_SAFE + 2.0, 2),  # ~14.09
    round(H_MIN_SAFE + 3.0, 2),  # ~15.09 — deep paramagnetic
]

print("N=120 Full VQE Sweep")
print(f"  h_min_safe = {H_MIN_SAFE:.4f}")
print(f"  h_values = {H_VALUES}")
print(f"  seeds = {SEEDS}")
print()

builder = HamiltonianBuilder()
solver = ClassicalSolver()
hva = HVACircuitBuilder()

# Step 1: DMRG ground truth (ONCE per h-value, cached)
print("--- Phase 1: DMRG ground truth (5 h-points) ---")
ground_truths = {}
for h in H_VALUES:
    t0 = time.time()
    lattice = make_lattice("chain_1d", N, J=1.0, h=h)
    H = builder.build(lattice)
    gt = solver.solve(H, lattice, method="dmrg")
    elapsed = time.time() - t0
    ground_truths[h] = {"e_exact": gt.ground_energy, "gap": gt.gap, "time_s": elapsed}
    print(f"  h={h:.2f}: E0={gt.ground_energy:.6f}, gap={gt.gap:.4f}, time={elapsed:.1f}s")

print()

# Step 2: VQE sweep (3 seeds × 5 h-points, descending)
print("--- Phase 2: VQE sweep (3 seeds x 5 h-points) ---")
results_all = []
t_start = time.time()

for seed in SEEDS:
    print(f"\n  Seed {seed}:")
    rng = np.random.default_rng(seed)
    prev_theta = rng.uniform(-0.01, 0.01, 2)

    for h in sorted(H_VALUES, reverse=True):  # Descending sweep
        lattice = make_lattice("chain_1d", N, J=1.0, h=h)
        H = builder.build(lattice)
        circuit, _ = hva.create(N, 1, lattice)

        e_exact = ground_truths[h]["e_exact"]
        gap = ground_truths[h]["gap"]

        backend = MPSBackend(strategy="aer_mps", chi_max=64, precision=0.005, seed=seed)
        config = VQEConfig(
            method="COBYLA",
            p_layers=1,
            n_restarts=3,
            maxiter=500,
            enable_callbacks=False,
        )
        optimizer = VQEOptimizer(config=config, backend=backend, seed=seed)

        t0 = time.time()
        result = optimizer.optimize(H, circuit, prev_theta.copy(), exact_energy=e_exact)
        t_vqe = time.time() - t0

        de_gap = abs(result.energy - e_exact) / max(gap, 1e-10)
        passed = de_gap < 0.05
        prev_theta = result.theta_opt.copy()

        results_all.append(
            {
                "seed": seed,
                "h": h,
                "e_exact": float(e_exact),
                "gap": float(gap),
                "e_vqe": float(result.energy),
                "de_gap": float(de_gap),
                "n_iterations": result.n_iterations,
                "theta_opt": result.theta_opt.tolist(),
                "time_s": round(t_vqe, 2),
                "passed": passed,
            }
        )

        status = "PASS" if passed else "FAIL"
        print(
            f"    h={h:.2f}: dE/gap={de_gap:.6f}, iters={result.n_iterations}, "
            f"{t_vqe:.1f}s [{status}]"
        )

total_time = time.time() - t_start


# Step 3: Summary
print(f"\nTotal VQE time: {total_time:.1f}s ({total_time / 60:.1f} min)")

n_total = len(results_all)
n_pass = sum(1 for r in results_all if r["passed"])
de_gaps = [r["de_gap"] for r in results_all]

print(f"\n{'=' * 60}")
print("  N=120 SWEEP SUMMARY")
print(f"{'=' * 60}")
print(f"  Total points: {n_total} ({len(SEEDS)} seeds x {len(H_VALUES)} h-points)")
print(f"  Passed: {n_pass}/{n_total} (threshold: dE/gap < 5%)")
print(f"  Mean dE/gap: {np.mean(de_gaps):.6f}")
print(f"  Max dE/gap: {np.max(de_gaps):.6f}")
print(f"  Std dE/gap: {np.std(de_gaps):.6f}")

# Per-h summary
print("\n  Per-h breakdown:")
for h in sorted(H_VALUES):
    h_results = [r for r in results_all if r["h"] == h]
    h_de = [r["de_gap"] for r in h_results]
    h_pass = sum(1 for r in h_results if r["passed"])
    print(
        f"    h={h:.2f}: mean={np.mean(h_de):.6f}, "
        f"std={np.std(h_de):.6f}, pass={h_pass}/{len(h_results)}"
    )

# Bootstrap CI on mean dE/gap
boot_rng = np.random.default_rng(99)
boot_means = [
    float(np.mean(boot_rng.choice(de_gaps, size=len(de_gaps), replace=True))) for _ in range(1000)
]
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
print(f"\n  Bootstrap 95% CI on mean dE/gap: [{ci_lo:.6f}, {ci_hi:.6f}]")

# Scaling law validation
print("\n  Scaling law check:")
print(f"    Formula: h_min = 1.5 + 0.020 * N^1.31 = {H_MIN_SAFE:.4f}")
print(f"    Lowest h tested: {min(H_VALUES):.2f} (margin: +{min(H_VALUES) - H_MIN_SAFE:.2f})")
lowest_h_results = [r for r in results_all if r["h"] == min(H_VALUES)]
if lowest_h_results:
    lowest_pass = all(r["passed"] for r in lowest_h_results)
    lowest_max = max(r["de_gap"] for r in lowest_h_results)
    print(f"    All pass at lowest h: {lowest_pass} (max dE/gap={lowest_max:.6f})")

# Save
output = {
    "experiment": "N120_full_sweep",
    "description": "Rigorous N=120 VQE sweep with 3 seeds near h_min boundary",
    "n_qubits": N,
    "h_min_safe": H_MIN_SAFE,
    "h_values": H_VALUES,
    "seeds": SEEDS,
    "total_time_s": total_time,
    "dmrg_cache": {str(h): gt for h, gt in ground_truths.items()},
    "metadata": {
        "n": N,
        "topology": "chain_1d",
        "p_layers": 1,
        "chi_max": 64,
        "strategy": "aer_mps",
        "mps_evaluation_mode": "deterministic",
        "h_values": H_VALUES,
        "seeds": SEEDS,
        "model": "tfim",
        "vqe_method": "COBYLA",
        "vqe_n_restarts": 3,
        "vqe_maxiter": 500,
    },
    "summary": {
        "n_total": n_total,
        "n_pass": n_pass,
        "pass_rate": n_pass / n_total,
        "mean_de_gap": float(np.mean(de_gaps)),
        "max_de_gap": float(np.max(de_gaps)),
        "std_de_gap": float(np.std(de_gaps)),
        "bootstrap_ci_95_mean_de_gap": [ci_lo, ci_hi],
    },
    "scaling_law": {
        "formula": "h_min = 1.5 + 0.020 * N^1.31",
        "h_min_predicted": H_MIN_SAFE,
        "validated": n_pass == n_total,
    },
    "per_point": results_all,
}

out_dir = Path("results/scaling")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "scaling_N120_full_sweep.json"
with open(out_path, "w") as f:
    json.dump(
        output, f, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)
    )
print(f"\n  Saved to {out_path}")
