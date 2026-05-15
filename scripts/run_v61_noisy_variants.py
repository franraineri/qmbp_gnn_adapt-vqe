#!/usr/bin/env python
"""
Experiments A' + Gate Folding: Two alternative ZNE strategies at N=10.

A' (Filtered layouts): Lower MAX_CES_RATIO to 3.0 to exclude the pathological
   CES=6.29 outlier layout. Use 5 layouts all in the perturbative regime [0.3, 1.5].
   Hypothesis: The failure was caused by one bad layout, not fundamental physics.

Gate Folding ZNE: Instead of varying layouts, amplify noise uniformly by inserting
   identity-equivalent gate pairs (RZZ(θ)·RZZ(-θ) = I). Noise factors 1×, 3×, 5×.
   Hypothesis: A controlled, monotonic noise axis enables extrapolation even at N=10.

References:
  - Uvarov et al. (2024): inhomogeneous ZNE (our current approach)
  - Mitiq / Li et al. (2017): gate folding ZNE (alternative approach)
  - Tsubouchi et al. (2023): exponential cost bound (motivates both experiments)
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
from qiskit.circuit import QuantumCircuit
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
N = 10
SEED_MPNN = 43
SEED_LAYOUT = 42
SHOTS = 16384
H_TEST_VALUES = [1.5, 2.0]  # Two h-values (quick mode — hypothesis testing)


def fold_circuit(circuit: QuantumCircuit, noise_factor: int) -> QuantumCircuit:
    """Apply global unitary folding to amplify noise by noise_factor.

    For noise_factor=3: circuit → circuit · circuit† · circuit
    For noise_factor=5: circuit → circuit · circuit† · circuit · circuit† · circuit

    The folded circuit implements the same unitary (U·U†·U = U) but has
    noise_factor× more gates, amplifying the noise proportionally.

    Parameters
    ----------
    circuit : QuantumCircuit
        The original (bound, non-parameterized) circuit.
    noise_factor : int
        Odd integer ≥ 1. Number of times the circuit unitary is effectively applied.

    Returns
    -------
    QuantumCircuit
        Folded circuit with noise_factor× more gates.
    """
    if noise_factor == 1:
        return circuit.copy()

    assert noise_factor % 2 == 1, "noise_factor must be odd (1, 3, 5, ...)"

    # Build folded circuit: U · (U† · U) repeated (noise_factor-1)/2 times
    n_folds = (noise_factor - 1) // 2
    folded = circuit.copy()
    for _ in range(n_folds):
        folded.compose(circuit.inverse(), inplace=True)
        folded.compose(circuit, inplace=True)

    return folded


def main() -> int:
    t0 = time.time()
    np.random.seed(SEED_MPNN)
    torch.manual_seed(SEED_MPNN)

    print("=" * 70)
    print("  Experiments A' + Gate Folding ZNE at N=10")
    print(f"  H-test values: {H_TEST_VALUES}")
    print(f"  N={N}, shots={SHOTS}")
    print("  A': 5 layouts, MAX_CES_RATIO=3.0 (filter outliers)")
    print("  Gate Folding: noise factors [1, 3, 5], single best layout")
    print("=" * 70)

    collector = DiagnosticCollector(verbose=False, save_dir=RESULTS_DIR)

    # ── Phase 1-3: Standard pipeline ──
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

    print("  Phase 2: VQE descending sweep...")
    t2 = time.time()
    vqe_config = VQEConfig(p_layers=p, n_restarts=5, maxiter=1000, enable_callbacks=False)
    opt = VQEOptimizer(vqe_config)
    vqe_results = opt.descending_sweep(h_values, qc, base_lattice, exact_data)
    fids = np.array([r.fidelity for r in vqe_results])
    print(f"    Done in {time.time() - t2:.1f}s — avg fid={np.mean(fids) * 100:.1f}%")

    print("  Phase 3: MPNN training...")
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

    model.eval()
    edge_idx, coord = builder.build_graph_data(base_lattice)

    # ══════════════════════════════════════════════════════════════════════
    # Experiment A': Filtered layouts (MAX_CES_RATIO=3.0, 5 layouts)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("  A': Inhomogeneous ZNE with filtered layouts (5 layouts, CES ratio ≤ 3)")
    print(f"{'═' * 70}")

    deployer_filtered = HardwareDeployerV61(mode="noisy_simulation", n_layouts=5, seed=SEED_LAYOUT)

    results_a_prime = []
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
        res = deployer_filtered.deploy_adapt_vqe(qc, H_test, theta_pred, lat_test, exact_test)
        elapsed = time.time() - t_dep

        r2_str = f"{res.zne_r_squared:.4f}" if res.zne_r_squared is not None else "N/A"
        print(
            f"    h={h_test:.2f}: ΔE/gap={res.delta_e_over_gap:.4f}, R²={r2_str}, "
            f"CES={[f'{c:.3f}' for c in res.ces_values]} ({elapsed:.1f}s)"
        )

        results_a_prime.append(
            {
                "h_test": h_test,
                "delta_e_over_gap": res.delta_e_over_gap,
                "zne_r_squared": res.zne_r_squared,
                "ces_values": res.ces_values,
                "energies_per_layout": res.energies_per_layout,
                "n_layouts_used": len(res.ces_values),
                "elapsed_s": elapsed,
            }
        )

    # ══════════════════════════════════════════════════════════════════════
    # Gate Folding ZNE: noise factors [1, 3, 5]
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("  Gate Folding ZNE: noise factors [1, 3, 5] on single best layout")
    print(f"{'═' * 70}")

    from qiskit.primitives import BackendEstimatorV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    backend = FakeTorino()
    precision = 1.0 / np.sqrt(SHOTS)

    # Use the best (lowest CES) layout from the filtered deployer
    deployer_single = HardwareDeployerV61(mode="noisy_simulation", n_layouts=1, seed=SEED_LAYOUT)

    noise_factors = [1, 3, 5]
    results_gate_fold = []

    for h_test in H_TEST_VALUES:
        print(f"\n    h={h_test:.2f}:")
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

        bound_circuit = qc.assign_parameters(theta_pred)

        # Get best layout
        layouts = deployer_single._layout_selector.select_layouts(N, 1)
        best_layout = layouts[0].initial_layout

        # Transpile the base circuit once
        pm = generate_preset_pass_manager(
            optimization_level=2, backend=backend, initial_layout=best_layout
        )
        transpiled_base = pm.run(bound_circuit)

        # Run at each noise factor
        energies_by_nf = []
        t_dep = time.time()

        for nf in noise_factors:
            # Fold the transpiled circuit
            folded = fold_circuit(transpiled_base, nf)

            # Map Hamiltonian to the transpiled layout
            h_mapped = H_test.apply_layout(transpiled_base.layout)

            # Execute
            estimator = BackendEstimatorV2(
                backend=backend,
                options={"default_precision": precision, "seed_simulator": SEED_LAYOUT + nf},
            )
            job = estimator.run([(folded, h_mapped)])
            result = job.result()
            energy = float(result[0].data.evs)
            energies_by_nf.append(energy)
            print(f"      nf={nf}: E={energy:.4f}")

        # Linear extrapolation to noise_factor=0
        nf_arr = np.array(noise_factors, dtype=float)
        e_arr = np.array(energies_by_nf)

        if len(noise_factors) >= 2:
            coeffs = np.polyfit(nf_arr, e_arr, 1)
            extrapolated_energy = float(np.polyval(coeffs, 0.0))
            y_pred = np.polyval(coeffs, nf_arr)
            ss_res = np.sum((e_arr - y_pred) ** 2)
            ss_tot = np.sum((e_arr - np.mean(e_arr)) ** 2)
            r2_gf = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            extrapolated_energy = energies_by_nf[0]
            r2_gf = 0.0

        delta_e_gf = abs(extrapolated_energy - exact_test.ground_energy)
        de_gap_gf = delta_e_gf / exact_test.gap if exact_test.gap > 0 else float("inf")
        elapsed = time.time() - t_dep

        print(
            f"      Extrapolated E={extrapolated_energy:.4f}, ΔE/gap={de_gap_gf:.4f}, R²={r2_gf:.4f} ({elapsed:.1f}s)"
        )

        results_gate_fold.append(
            {
                "h_test": h_test,
                "noise_factors": noise_factors,
                "energies": energies_by_nf,
                "extrapolated_energy": extrapolated_energy,
                "delta_e_over_gap": de_gap_gf,
                "r_squared": r2_gf,
                "elapsed_s": elapsed,
            }
        )

    # ── Summary ──
    print(f"\n\n{'═' * 70}")
    print("  SUMMARY")
    print(f"{'═' * 70}")
    print(f"  {'Method':<25} {'h_test':<8} {'R²':<10} {'ΔE/gap':<12} {'Layouts/NF'}")
    print(f"  {'─' * 65}")
    for r in results_a_prime:
        r2 = r["zne_r_squared"] if r["zne_r_squared"] is not None else 0
        print(
            f"  {'A (filtered layouts)':<25} {r['h_test']:<8.2f} {r2:<10.4f} {r['delta_e_over_gap']:<12.4f} {r['n_layouts_used']} layouts"
        )
    for r in results_gate_fold:
        print(
            f"  {'Gate Folding':<25} {r['h_test']:<8.2f} {r['r_squared']:<10.4f} {r['delta_e_over_gap']:<12.4f} nf={r['noise_factors']}"
        )

    # ── Save ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = hashlib.sha1(f"variants:{ts}".encode()).hexdigest()[:8]
    json_path = RESULTS_DIR / f"zne_variants_{ts}_{run_id}.json"

    output = {
        "experiment": "A_prime_and_gate_folding",
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "config": {
            "N": N,
            "p": p,
            "seed_mpnn": SEED_MPNN,
            "seed_layout": SEED_LAYOUT,
            "shots": SHOTS,
            "h_test_values": H_TEST_VALUES,
            "max_ces_ratio": 3.0,
        },
        "mpnn_mse": train_result["final_mse"],
        "results_a_prime": results_a_prime,
        "results_gate_folding": results_gate_fold,
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
