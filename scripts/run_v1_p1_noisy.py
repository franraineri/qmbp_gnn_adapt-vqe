#!/usr/bin/env python3
"""Variant 1: p=1 Noisy Sweep at N=10 with inhomogeneous ZNE.

Hypothesis: p=1 N=10 has ~18 CX gates (similar to p=2 N=6 with ~20 CX).
Since p=2 N=6 achieves R²>0.99 with 3 layouts, p=1 N=10 should restore
ZNE linearity (R² > 0.8).

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
    compute_circuit_ces,
    find_layouts_bfs,
    linear_zne,
    noisy_estimate,
    select_layouts_by_circuit_ces,
)
from qmbp_simulation.models import VQEConfig
from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "exp_noisy_variants"

# ── Configuration ──
N, p, J = 10, 1, 1.0
SEED = 43
SHOTS = 16384
N_LAYOUTS = 3
H_TEST_VALUES = [2.0, 2.5, 3.0]

# Centralized noisy estimation config (enforces seed + precision)
NOISY_CONFIG = NoisyEstimatorConfig(shots=SHOTS, seed_simulator=SEED)

print("=" * 70)
print("  VARIANT 1: p=1 Noisy Sweep at N=10")
print("  Hypothesis: 50% fewer CX -> ZNE linearity restored")
print("  Config: N=%d, p=%d, shots=%d, n_layouts=%d" % (N, p, SHOTS, N_LAYOUTS))
print("  Precision: %.6f (= 1/sqrt(%d))" % (NOISY_CONFIG.precision, SHOTS))
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
print("  Circuit params: %d" % qc.num_parameters)

# ── Phase 1: Exact diag ──
h_values = np.arange(1.9, 4.05, 0.1)[::-1]  # Descending: h=4.0 -> 1.9
print("  Phase 1: Exact diag (%d h-points)..." % len(h_values))
t1 = time.time()
exact_data = []
for h in h_values:
    lat_h = make_lattice("chain_1d", N, J=J, h=h)
    H = builder.build(lat_h)
    exact_data.append(solver.solve(H, lat_h))
print("    Done in %.1fs" % (time.time() - t1))

# ── Phase 2: VQE descending sweep ──
print("  Phase 2: VQE descending sweep (5 restarts)...")
t2 = time.time()
vqe_config = VQEConfig(p_layers=p, n_restarts=5, maxiter=1000, enable_callbacks=False)
opt = VQEOptimizer(vqe_config)
vqe_results = opt.descending_sweep(h_values, qc, base_lattice, exact_data)
fids = np.array([r.fidelity for r in vqe_results])
print("    Done in %.1fs -- avg fid=%.1f%%" % (time.time() - t2, np.mean(fids) * 100))

# ── Phase 3: MPNN ──
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

# ── Phase 4: Noisy deployment ──
print("")
print("-" * 70)
print("  Phase 4: Noisy deployment (FakeTorino + BackendEstimatorV2)")
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

    # Noiseless energy
    e_noiseless = noiseless_backend.evaluate(qc, H_test, theta_pred)
    de_noiseless = abs(e_noiseless - exact_test.ground_energy) / exact_test.gap

    # Select layouts by CIRCUIT CES (framework utility)
    selection = select_layouts_by_circuit_ces(
        bound_circuit, backend, candidate_layouts, n_select=N_LAYOUTS
    )
    print(
        "    Selected %d layouts with circuit CES: %s"
        % (len(selection.layouts), [round(c, 4) for c in selection.ces_values])
    )

    # Execute on each layout with correct seed + precision
    layout_energies = []
    for li, transpiled in enumerate(selection.transpiled_circuits):
        h_mapped = H_test.apply_layout(transpiled.layout)
        energy = noisy_estimate(transpiled, h_mapped, backend, NOISY_CONFIG, seed_offset=li)
        layout_energies.append(energy)

    # ZNE: linear extrapolation E(CES) -> CES=0
    ces_arr = np.array(selection.ces_values)
    e_arr = np.array(layout_energies)
    zne_result = linear_zne(ces_arr, e_arr)

    de_zne = abs(zne_result.extrapolated_value - exact_test.ground_energy) / exact_test.gap
    de_raw = abs(e_arr[0] - exact_test.ground_energy) / exact_test.gap
    zne_gain = (de_raw - de_zne) / de_raw if de_raw > 0 else 0.0
    _, n_2q = compute_circuit_ces(selection.transpiled_circuits[0], backend)

    print("    2Q gates: %d" % n_2q)
    print("    CES: %s" % [round(c, 4) for c in ces_arr.tolist()])
    print("    Energies: %s" % [round(e, 4) for e in e_arr.tolist()])
    print("    Noiseless dE/gap: %.4f" % de_noiseless)
    print("    Noisy raw dE/gap: %.4f" % de_raw)
    print(
        "    ZNE dE/gap: %.4f, R2=%.4f, gain=%.1f%%"
        % (de_zne, zne_result.r_squared, zne_gain * 100)
    )
    if zne_result.r_squared > 0.8:
        print("    >> R2>0.8 -- LINEARITY RESTORED")
    else:
        print("    >> R2<0.8")

    results.append(
        {
            "h_test": h_test,
            "exact_energy": exact_test.ground_energy,
            "gap": exact_test.gap,
            "noiseless_de_gap": de_noiseless,
            "noisy_raw_de_gap": de_raw,
            "zne_de_gap": de_zne,
            "zne_r2": zne_result.r_squared,
            "zne_gain": zne_gain,
            "ces_values": ces_arr.tolist(),
            "energies": e_arr.tolist(),
            "n_2q_gates": n_2q,
        }
    )

# ── Summary ──
print("")
print("=" * 70)
print("  SUMMARY: p=1 N=10 ZNE Results")
print("=" * 70)
avg_r2 = np.mean([r["zne_r2"] for r in results])
good_r2 = sum(1 for r in results if r["zne_r2"] > 0.8)

for r in results:
    status = "PASS" if r["zne_r2"] > 0.8 else "FAIL"
    print(
        "  h=%.2f: R2=%.4f, dE/gap(raw)=%.4f, dE/gap(ZNE)=%.4f [%s]"
        % (r["h_test"], r["zne_r2"], r["noisy_raw_de_gap"], r["zne_de_gap"], status)
    )

print("")
print("  Average R2: %.4f" % avg_r2)
print("  Good R2 (>0.8): %d/%d" % (good_r2, len(results)))
print("  Comparison: p=2 N=10 had R2 < 0.05 (FAILS)")
print("  Comparison: p=2 N=6  had R2 > 0.99 (PASSES)")
print("")

if avg_r2 > 0.5:
    print("  HYPOTHESIS CONFIRMED: Reducing depth restores ZNE linearity!")
else:
    print("  HYPOTHESIS REJECTED: p=1 does not restore linearity.")

elapsed = time.time() - t0
print("  Total time: %.0fs" % elapsed)

# ── Save ──
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = RESULTS_DIR / ("v1_p1_noisy_%s.json" % ts)
output = {
    "variant": "V1_p1_noisy_sweep",
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
    "summary": {"avg_r2": avg_r2, "good_r2_count": good_r2, "elapsed_s": elapsed},
}
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print("  Saved: %s" % out_path)
