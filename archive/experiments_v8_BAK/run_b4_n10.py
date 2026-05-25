#!/usr/bin/env python
"""B4 at N=10: Verify Hessian landscape properties scale with system size.

Hypothesis: The saddle-free property and condition number growth pattern
observed at N=6 also holds at N=10.

Expected: ~5 min execution.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize

from scripts.experiments_v8.core.landscape import compute_hessian
from src.poc.v6 import ClassicalSolver, HamiltonianBuilder, HVACircuitBuilder, make_lattice

builder, solver, hva = HamiltonianBuilder(), ClassicalSolver(), HVACircuitBuilder()
estimator = StatevectorEstimator()
N, p = 10, 2
qc, _ = hva.create(N, p, make_lattice("chain_1d", N, J=1.0, h=1.0))
n_params = qc.num_parameters

print(f"B4 at N=10: Hessian at VQE Minima (N={N}, p={p}, n_params={n_params})")
print(f"{'h':<6}{'DE/gap':<9}{'Type':<12}{'Eigenvalues':<40}{'Cond':<8}")
print("-" * 72)

np.random.seed(42)
prev = np.random.uniform(-0.01, 0.01, n_params)

results = []
for h in [2.0, 1.75, 1.5, 1.25, 1.0]:
    lat = make_lattice("chain_1d", N, J=1.0, h=h)
    H = builder.build(lat)
    exact = solver.solve(H, lat)

    def cost(params, _H=H):
        return float(estimator.run([(qc.assign_parameters(params), _H)]).result()[0].data.evs)

    best_e, best_t = float("inf"), prev.copy()
    for r in range(5):
        x0 = prev + np.random.normal(0, 0.1, n_params) if r > 0 else prev
        x0 = np.clip(x0, -np.pi, np.pi)
        res = minimize(
            cost,
            x0,
            method="L-BFGS-B",
            bounds=[(-np.pi, np.pi)] * n_params,
            options={"maxiter": 500, "ftol": 1e-14},
        )
        if res.fun < best_e:
            best_e, best_t = res.fun, res.x.copy()

    gap = exact.gap if exact.gap > 1e-10 else max(2 * abs(1 - h), 2 * np.pi / N)
    de = abs(best_e - exact.ground_energy) / gap

    H_mat = compute_hessian(cost, best_t, epsilon=5e-3)
    eigs = np.linalg.eigvalsh(H_mat)
    n_neg = int(np.sum(eigs < -0.01))
    tp = "minimum" if n_neg == 0 else f"saddle({n_neg})"
    pos = eigs[eigs > 0.01]
    cond = pos[-1] / pos[0] if len(pos) >= 2 else 1.0
    eig_s = ",".join(f"{e:.1f}" for e in eigs)
    print(f"{h:<6.2f}{de:<9.4f}{tp:<12}[{eig_s}]{cond:>8.1f}")
    results.append({"h": h, "de_gap": de, "type": tp, "eigs": eigs.tolist(), "cond": cond})
    prev = best_t.copy()

print("\n" + "=" * 72)
print("COMPARISON: N=6 vs N=10")
print("=" * 72)
print(f"{'h':<6}{'N=6 Cond':<12}{'N=10 Cond':<12}{'N=6 Type':<12}{'N=10 Type':<12}")
print("-" * 54)
# N=6 known results from B4-lite
n6_data = {
    2.0: (1399, "minimum"),
    1.5: (36, "minimum"),
    1.25: (23, "minimum"),
    1.0: (14, "minimum"),
}
for r in results:
    h = r["h"]
    if h in n6_data:
        c6, t6 = n6_data[h]
        print(f"{h:<6.2f}{c6:<12.1f}{r['cond']:<12.1f}{t6:<12}{r['type']:<12}")

print("\nConclusion:")
all_minima = all(r["type"] == "minimum" for r in results)
if all_minima:
    print("  ✅ ALL minima are genuine at N=10 (no saddle points) — confirms N=6 result")
else:
    saddles = [r for r in results if r["type"] != "minimum"]
    print(f"  ⚠️ Found {len(saddles)} saddle point(s) at N=10")
    for s in saddles:
        print(f"     h={s['h']}: {s['type']}")
