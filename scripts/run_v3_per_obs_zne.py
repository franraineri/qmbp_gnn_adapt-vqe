#!/usr/bin/env python3
"""Variant 3: Per-Observable ZNE at N=10 (p=2).

Hypothesis: Sites 2 and 9 at N=10 lose 62-87% of signal (layout-dependent
bad qubits). Per-site ZNE extrapolation on "good" sites may yield R2>0.5
even when total energy ZNE fails.

Uses FakeTorino + BackendEstimatorV2 with correct seed_simulator and precision.
"""

import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qiskit_ibm_runtime.fake_provider import FakeTorino

from qmbp_simulation import (
    ClassicalSolver,
    HamiltonianBuilder,
    HVACircuitBuilder,
    VQEOptimizer,
    make_lattice,
)
from qmbp_simulation.execution import (
    NoiselessBackend,
    NoisyEstimatorConfig,
    build_adjacency,
    find_layouts_bfs,
    run_zne_deployment,
    select_layouts_by_circuit_ces,
)
from qmbp_simulation.models import VQEConfig
from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "exp_noisy_variants"

# ── Configuration ──
N, p, J = 10, 2, 1.0
SEED = 43
SHOTS = 16384
N_LAYOUTS = 3
H_TEST_VALUES = [1.5, 2.0]

NOISY_CONFIG = NoisyEstimatorConfig(shots=SHOTS, seed_simulator=SEED)

print("=" * 70)
print("  VARIANT 3: Per-Observable ZNE at N=10 (p=2)")
print("  Hypothesis: Site-selective ZNE on good qubits recovers signal")
print("  Config: N=%d, p=%d, shots=%d, n_layouts=%d" % (N, p, SHOTS, N_LAYOUTS))
print("  Precision: %.6f" % NOISY_CONFIG.precision)
print("  H-test: %s" % H_TEST_VALUES)
print("=" * 70)

t0 = time.time()
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

builder = HamiltonianBuilder()
solver = ClassicalSolver()
hva = HVACircuitBuilder()
base_lattice = make_lattice("chain_1d", N, J=J, h=1.0)
qc, _ = hva.create(N, p, base_lattice)

# ── Phase 1-3: Standard N=10 p=2 pipeline ──
h_coarse = np.arange(0.0, 0.8, 0.1)
h_dense = np.arange(0.8, 1.45, 0.05)
h_coarse2 = np.arange(1.5, 2.05, 0.1)
h_values = np.unique(np.concatenate([h_coarse, h_dense, h_coarse2]))[::-1]

print("  Phase 1: Exact diag (%d h-points)..." % len(h_values))
t1 = time.time()
exact_data = []
for h in h_values:
    lat_h = make_lattice("chain_1d", N, J=J, h=h)
    H = builder.build(lat_h)
    exact_data.append(solver.solve(H, lat_h))
print("    Done in %.1fs" % (time.time() - t1))

print("  Phase 2: VQE descending sweep (5 restarts)...")
t2 = time.time()
vqe_config = VQEConfig(p_layers=p, n_restarts=5, maxiter=1000, enable_callbacks=False)
opt = VQEOptimizer(vqe_config)
vqe_results = opt.descending_sweep(h_values, qc, base_lattice, exact_data)
fids = np.array([r.fidelity for r in vqe_results])
print("    Done in %.1fs -- avg fid=%.1f%%" % (time.time() - t2, np.mean(fids) * 100))

print("  Phase 3: MPNN (h=128, L=3, 6000ep)...")
t3 = time.time()
dataset = build_graph_dataset(
    base_lattice,
    h_values,
    np.array([r.theta_opt for r in vqe_results]),
    np.array([d.ground_energy for d in exact_data]),
    fidelities=fids,
    fidelity_threshold=0.93,
)
model = MPNNPredictor(node_features=2, hidden_dim=128, n_layers=3, output_dim=2 * p)
train_result = train_mpnn(model, dataset, n_epochs=6000, lr=1e-3, patience=500)
print("    Done in %.1fs -- MSE=%.2e" % (time.time() - t3, train_result["final_mse"]))
model.eval()
edge_idx, coord = builder.build_graph_data(base_lattice)

# ── Phase 4: Per-site ZNE ──
print("")
print("-" * 70)
print("  Phase 4: Per-site ZNE (measure X_i per layout, fit per site)")
print("-" * 70)

backend = FakeTorino()
noiseless_backend = NoiselessBackend()

# Find candidate layouts using framework utilities
adj = build_adjacency(backend)
candidate_layouts = find_layouts_bfs(adj, n_qubits=N, n_candidates=20, seed=SEED)
print("  Found %d candidate layouts" % len(candidate_layouts))

results = []

for h_test in H_TEST_VALUES:
    print("")
    print("  h=%.2f:" % h_test)
    lat_test = make_lattice("chain_1d", N, J=J, h=h_test)
    H_test = builder.build(lat_test)
    exact_test = solver.solve(H_test, lat_test)

    # MPNN prediction
    x_test = torch.tensor(
        np.stack([np.full(N, h_test), coord.astype(float)], axis=1),
        dtype=torch.float32,
    )
    test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))
    with torch.no_grad():
        theta_pred = model(test_graph).numpy().flatten()

    bound_circuit = qc.assign_parameters(theta_pred)
    noiseless_per_site = exact_test.per_site_mag_x

    # Select layouts by circuit CES
    selection = select_layouts_by_circuit_ces(
        bound_circuit, backend, candidate_layouts, n_select=N_LAYOUTS
    )

    # Run full ZNE deployment with per-site measurements
    deploy_result = run_zne_deployment(
        bound_circuit=bound_circuit,
        hamiltonian=H_test,
        backend=backend,
        layout_selection=selection,
        config=NOISY_CONFIG,
        n_qubits=N,
        per_site=True,
    )

    # Extract results
    ces_arr = np.array(selection.ces_values)
    e_arr = np.array([d["energy"] for d in deploy_result.per_layout_data])
    r2_total = deploy_result.energy_zne.r_squared

    # Per-site ZNE analysis
    per_site_r2 = [r.r_squared for r in deploy_result.per_site_zne]
    per_site_extrap = [r.extrapolated_value for r in deploy_result.per_site_zne]

    # Identify good sites (R2 > 0.5)
    good_sites = [i for i, r2 in enumerate(per_site_r2) if r2 > 0.5]
    bad_sites = [i for i, r2 in enumerate(per_site_r2) if r2 <= 0.5]

    # Reconstruct magnetization from good sites only
    if good_sites:
        mag_x_good = np.mean([per_site_extrap[i] for i in good_sites])
    else:
        mag_x_good = np.mean(per_site_extrap)

    mag_x_noiseless = float(np.mean(noiseless_per_site))
    per_site_arr = np.array([d["per_site_x"] for d in deploy_result.per_layout_data])
    mag_x_raw = float(np.mean(per_site_arr[0]))
    mag_x_error_raw = abs(mag_x_raw - mag_x_noiseless)
    mag_x_error_good = abs(mag_x_good - mag_x_noiseless)

    improvement = 0.0
    if mag_x_error_raw > 0:
        improvement = (mag_x_error_raw - mag_x_error_good) / mag_x_error_raw

    print("    CES: %s" % [round(c, 4) for c in ces_arr.tolist()])
    print("    Total energy R2: %.4f" % r2_total)
    print("    Per-site R2: %s" % [round(r, 3) for r in per_site_r2])
    print("    Good sites (R2>0.5): %s (%d/%d)" % (good_sites, len(good_sites), N))
    print("    <X> noiseless: %.4f" % mag_x_noiseless)
    print("    <X> raw:       %.4f (error=%.4f)" % (mag_x_raw, mag_x_error_raw))
    print("    <X> good-site: %.4f (error=%.4f)" % (mag_x_good, mag_x_error_good))
    print("    Improvement: %.1f%%" % (improvement * 100))

    results.append(
        {
            "h_test": h_test,
            "ces_values": ces_arr.tolist(),
            "energy_per_layout": e_arr.tolist(),
            "r2_total_energy": r2_total,
            "per_site_r2": per_site_r2,
            "per_site_extrap": per_site_extrap,
            "good_sites": good_sites,
            "bad_sites": bad_sites,
            "mag_x_noiseless": mag_x_noiseless,
            "mag_x_raw": mag_x_raw,
            "mag_x_good_site_zne": mag_x_good,
            "mag_x_error_raw": mag_x_error_raw,
            "mag_x_error_good_site": mag_x_error_good,
            "improvement": improvement,
        }
    )

# ── Summary ──
print("")
print("=" * 70)
print("  SUMMARY: Per-Observable ZNE Results")
print("=" * 70)
for r in results:
    print(
        "  h=%.2f: %d/%d good sites, improvement=%.1f%%"
        % (r["h_test"], len(r["good_sites"]), N, r["improvement"] * 100)
    )
print("")
avg_good = np.mean([len(r["good_sites"]) for r in results])
avg_improvement = np.mean([r["improvement"] for r in results])
print("  Avg good sites: %.1f/%d" % (avg_good, N))
print("  Avg improvement: %.1f%%" % (avg_improvement * 100))

if avg_improvement > 0.1:
    print("  HYPOTHESIS CONFIRMED: Per-site ZNE improves over raw measurement")
else:
    print("  HYPOTHESIS REJECTED: Per-site ZNE does not help significantly")

elapsed = time.time() - t0
print("  Total time: %.0fs" % elapsed)

# ── Save ──
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = RESULTS_DIR / ("v3_per_obs_zne_%s.json" % ts)
output = {
    "variant": "V3_per_observable_zne",
    "timestamp": datetime.now().isoformat(),
    "config": {
        "N": N,
        "p": p,
        "shots": SHOTS,
        "n_layouts": N_LAYOUTS,
        "seed": SEED,
        "precision": NOISY_CONFIG.precision,
        "h_test": H_TEST_VALUES,
    },
    "mpnn_mse": train_result["final_mse"],
    "results": results,
    "summary": {
        "avg_good_sites": avg_good,
        "avg_improvement": avg_improvement,
        "elapsed_s": elapsed,
    },
}
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print("  Saved: %s" % out_path)
