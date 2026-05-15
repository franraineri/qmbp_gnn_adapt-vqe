#!/usr/bin/env python
"""
Experiment A: Test ZNE with increased n_layouts at N=10.

Hypothesis: More layouts (7, 10) provide enough CES diversity for linear
extrapolation to work at N=10, where 3 layouts completely failed (R²<0.05).

Expected learning:
  - If R² improves to >0.8 → failure was statistical (too few points)
  - If R² stays <0.5 → failure is fundamental (no linear E(CES) at this depth)

Reference: Rabinovich et al. (2025, arXiv:2511.02901) — CLP-ZNE uses O(n) layouts.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
from torch_geometric.data import Data

from src.poc.v6.classical_solver import ClassicalSolver
from src.poc.v6.config import VQEConfig
from src.poc.v6.diagnostics import DiagnosticCollector
from src.poc.v6.hamiltonian_builder import HamiltonianBuilder, make_lattice
from src.poc.v6.hardware_deployer_v61 import HardwareDeployerV61
from src.poc.v6.hva_builder import HVACircuitBuilder
from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn
from src.poc.v6.vqe_optimizer import VQEOptimizer

RESULTS_DIR = _project_root / "scripts" / "notebook_results"

# Test configurations: 3 (baseline/control), 7, 10 layouts
LAYOUT_CONFIGS = [3, 7, 10]
H_TEST_VALUES = [1.5, 1.7, 2.0]  # Use h-values where noiseless passes easily
N = 10
SEED_MPNN = 43
SEED_LAYOUT = 42
SHOTS = 16384


def main() -> int:
    t0 = time.time()
    np.random.seed(SEED_MPNN)
    torch.manual_seed(SEED_MPNN)

    print("=" * 70)
    print("  Experiment A: ZNE Layout Scaling at N=10")
    print(f"  Layout configs: {LAYOUT_CONFIGS}")
    print(f"  H-test values: {H_TEST_VALUES}")
    print(f"  N={N}, shots={SHOTS}, seed_mpnn={SEED_MPNN}, seed_layout={SEED_LAYOUT}")
    print("=" * 70)

    # ── Diagnostics ──
    collector = DiagnosticCollector(verbose=False, save_dir=RESULTS_DIR)

    # ── Phase 1-3: Build MPNN (same as standard N=10 pipeline) ──
    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    J, p = 1.0, 2
    base_lattice = make_lattice("chain_1d", N, J=J, h=1.0)
    qc, _ = hva.create(N, p, base_lattice)

    h_coarse = np.arange(0.0, 0.8, 0.1)
    h_dense = np.arange(0.8, 1.45, 0.05)
    h_coarse2 = np.arange(1.5, 2.05, 0.1)
    h_values = np.unique(np.concatenate([h_coarse, h_dense, h_coarse2]))

    print(f"\n  Phase 1: Exact diag ({len(h_values)} h-points)...")
    t1 = time.time()
    exact_data = []
    for h in h_values:
        lat_h = make_lattice("chain_1d", N, J=J, h=h)
        H = builder.build(lat_h)
        exact_data.append(solver.solve(H, lat_h))
    print(f"    Done in {time.time() - t1:.1f}s")

    collector.record_phase1(
        n_points=len(exact_data),
        elapsed_s=time.time() - t1,
        gap_min=min(d.gap for d in exact_data),
    )

    print("  Phase 2: VQE descending sweep (5 restarts, 1000 iter)...")
    t2 = time.time()
    vqe_config = VQEConfig(p_layers=p, n_restarts=5, maxiter=1000, enable_callbacks=False)
    opt = VQEOptimizer(vqe_config)
    vqe_results = opt.descending_sweep(h_values, qc, base_lattice, exact_data)
    fids = np.array([r.fidelity for r in vqe_results])
    print(f"    Done in {time.time() - t2:.1f}s — avg fid={np.mean(fids) * 100:.1f}%")

    print("  Phase 3: MPNN (h=128, L=3, epochs=6000, patience=500)...")
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
    print(f"    Done in {time.time() - t3:.1f}s — MSE={train_result['final_mse']:.2e}")

    # ── Phase 4: Deploy with varying n_layouts ──
    print(f"\n{'═' * 70}")
    print("  DEPLOYMENT: Comparing n_layouts = 3, 7, 10")
    print(f"{'═' * 70}")

    model.eval()
    edge_idx, coord = builder.build_graph_data(base_lattice)

    results_by_layouts: dict[int, list[dict]] = {}

    for n_layouts in LAYOUT_CONFIGS:
        print(f"\n  ── n_layouts = {n_layouts} ──")
        deployer = HardwareDeployerV61(
            mode="noisy_simulation", n_layouts=n_layouts, seed=SEED_LAYOUT
        )
        results_by_layouts[n_layouts] = []

        for h_test in H_TEST_VALUES:
            lat_test = make_lattice("chain_1d", N, J=J, h=h_test)
            H_test = builder.build(lat_test)
            exact_test = solver.solve(H_test, lat_test)

            x_test = torch.tensor(
                np.stack([np.full(N, h_test), coord.astype(float)], axis=1),
                dtype=torch.float32,
            )
            test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))
            with torch.no_grad():
                theta_pred = model(test_graph).numpy().flatten()

            t_dep = time.time()
            res = deployer.deploy_adapt_vqe(qc, H_test, theta_pred, lat_test, exact_test)
            elapsed = time.time() - t_dep

            r2_str = f"{res.zne_r_squared:.4f}" if res.zne_r_squared is not None else "N/A"
            print(
                f"    h={h_test:.2f}: ΔE/gap={res.delta_e_over_gap:.4f}, "
                f"R²={r2_str}, CES={res.ces_values}, ({elapsed:.1f}s)"
            )

            results_by_layouts[n_layouts].append(
                {
                    "h_test": h_test,
                    "delta_e_over_gap": res.delta_e_over_gap,
                    "predicted_energy": res.predicted_energy,
                    "zne_r_squared": res.zne_r_squared,
                    "ces_values": res.ces_values,
                    "energies_per_layout": res.energies_per_layout,
                    "extrapolation_method": res.extrapolation_method,
                    "mag_x_error": res.mag_x_error,
                    "elapsed_s": elapsed,
                }
            )

    # ── Summary table ──
    print(f"\n\n{'═' * 70}")
    print("  RESULTS: R² by n_layouts × h_test")
    print(f"{'═' * 70}")
    print(f"  {'h_test':<8}", end="")
    for nl in LAYOUT_CONFIGS:
        print(f"  {'n=' + str(nl) + ' R²':<14} {'ΔE/gap':<10}", end="")
    print()
    print(f"  {'─' * 66}")

    for i, h_test in enumerate(H_TEST_VALUES):
        print(f"  {h_test:<8.2f}", end="")
        for nl in LAYOUT_CONFIGS:
            r = results_by_layouts[nl][i]
            r2 = r["zne_r_squared"]
            r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
            de = r["delta_e_over_gap"]
            print(f"  {r2_str:<14} {de:<10.4f}", end="")
        print()

    # ── Conclusion ──
    print("\n  Analysis:")
    for nl in LAYOUT_CONFIGS:
        r2_vals = [
            r["zne_r_squared"] for r in results_by_layouts[nl] if r["zne_r_squared"] is not None
        ]
        if r2_vals:
            avg_r2 = np.mean(r2_vals)
            good_r2 = sum(1 for r2 in r2_vals if r2 > 0.8)
            print(
                f"    n_layouts={nl}: avg R²={avg_r2:.4f}, good R² (>0.8): {good_r2}/{len(r2_vals)}"
            )

    # ── Save results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = hashlib.sha1(f"layout_scaling:{ts}".encode()).hexdigest()[:8]
    json_path = RESULTS_DIR / f"layout_scaling_{ts}_{run_id}.json"

    output = {
        "experiment": "A_layout_scaling",
        "hypothesis": "More layouts (7, 10) improve ZNE R² at N=10 where 3 layouts failed",
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "config": {
            "N": N,
            "p": p,
            "seed_mpnn": SEED_MPNN,
            "seed_layout": SEED_LAYOUT,
            "shots": SHOTS,
            "h_test_values": H_TEST_VALUES,
            "layout_configs": LAYOUT_CONFIGS,
        },
        "mpnn_mse": train_result["final_mse"],
        "results_by_layouts": {str(k): v for k, v in results_by_layouts.items()},
        "diagnostics": collector.to_dict(),
        "total_elapsed_s": round(time.time() - t0, 1),
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved: {json_path}")
    print(f"  Total time: {time.time() - t0:.0f}s")
    print(f"{'═' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
