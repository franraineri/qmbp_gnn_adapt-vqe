#!/usr/bin/env python
"""
Experiment B: DD pre-mitigation + ZNE at N=10.

Hypothesis: Dynamical Decoupling reduces effective noise during idle periods,
bringing the circuit back into the perturbative regime where linear E(CES) holds.
If DD reduces the effective CES spread, R² should improve from ~0.03 to >0.5.

Approach: Apply PadDynamicalDecoupling (XY4 sequence) as a transpiler pass
AFTER standard transpilation but BEFORE submitting to BackendEstimatorV2.
Since BackendEstimatorV2 re-transpiles internally, we submit ISA circuits
(already transpiled + DD applied) and rely on the estimator's internal
transpilation being a no-op for already-ISA circuits.

Reference: Pokharel et al. (2025, arXiv:2403.02294) — learned DD on IBM.
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
from qiskit.circuit.library import XGate, YGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
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
H_TEST_VALUES = [1.5, 1.7, 2.0]


def main() -> int:
    t0 = time.time()
    np.random.seed(SEED_MPNN)
    torch.manual_seed(SEED_MPNN)

    print("=" * 70)
    print("  Experiment B: DD Pre-Mitigation + ZNE at N=10")
    print(f"  H-test values: {H_TEST_VALUES}")
    print(f"  N={N}, shots={SHOTS}, n_layouts=3")
    print("  DD sequence: XY4 (X-Y-X-Y)")
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

    # ── Phase 4: Compare no-DD vs DD ──
    print(f"\n{'═' * 70}")
    print("  DEPLOYMENT: Comparing no-DD (baseline) vs DD (XY4)")
    print(f"{'═' * 70}")

    model.eval()
    edge_idx, coord = builder.build_graph_data(base_lattice)

    # Standard deployer (no DD — baseline, same as previous experiments)
    deployer_no_dd = HardwareDeployerV61(mode="noisy_simulation", n_layouts=3, seed=SEED_LAYOUT)

    # For DD experiment: we'll use the same deployer but apply DD post-transpilation
    # by monkey-patching the transpilation step. However, since BackendEstimatorV2
    # re-transpiles, the cleanest test is to compare the raw energies and see if
    # the DD-applied circuit produces different noise characteristics.
    #
    # Alternative approach: Use Qiskit's AerSimulator directly with a noise model
    # and manually apply DD. But this changes the simulation backend.
    #
    # Simplest valid approach: Check if BackendEstimatorV2 respects pre-transpiled
    # ISA circuits (it should, since they're already in the backend's basis).

    from qiskit.primitives import BackendEstimatorV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    backend = FakeTorino()
    precision = 1.0 / np.sqrt(SHOTS)

    # Get DD pass components
    target = backend.target
    # XY4 sequence: X-Y-X-Y
    dd_sequence = [XGate(), YGate(), XGate(), YGate()]

    results_no_dd = []
    results_with_dd = []

    for h_test in H_TEST_VALUES:
        print(f"\n  h={h_test:.2f}:")
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

        # ── No-DD baseline ──
        t_dep = time.time()
        res_no_dd = deployer_no_dd.deploy_adapt_vqe(qc, H_test, theta_pred, lat_test, exact_test)
        t_no_dd = time.time() - t_dep
        r2_no_dd = res_no_dd.zne_r_squared
        print(
            f"    no-DD:   ΔE/gap={res_no_dd.delta_e_over_gap:.4f}, "
            f"R²={r2_no_dd:.4f} ({t_no_dd:.1f}s)"
        )
        results_no_dd.append(
            {
                "h_test": h_test,
                "delta_e_over_gap": res_no_dd.delta_e_over_gap,
                "zne_r_squared": r2_no_dd,
                "ces_values": res_no_dd.ces_values,
                "energies_per_layout": res_no_dd.energies_per_layout,
                "elapsed_s": t_no_dd,
            }
        )

        # ── With DD: manual transpilation + DD pass + direct estimator call ──
        t_dep = time.time()
        bound_circuit = qc.assign_parameters(theta_pred)

        # Get layouts from the deployer's selector
        layouts = deployer_no_dd._layout_selector.select_layouts(N, 3)

        dd_energies = []
        dd_ces = []
        for layout in layouts:
            # Transpile with optimization_level=2
            pm = generate_preset_pass_manager(
                optimization_level=2,
                backend=backend,
                initial_layout=layout.initial_layout,
            )
            transpiled = pm.run(bound_circuit)

            # Apply DD pass
            try:
                dd_pm = PassManager(
                    [
                        ALAPScheduleAnalysis(target=target),
                        PadDynamicalDecoupling(target=target, dd_sequence=dd_sequence),
                    ]
                )
                transpiled_dd = dd_pm.run(transpiled)
            except Exception as e:
                # If DD fails (e.g., no idle time), use original
                print(f"      DD failed for layout: {e}")
                transpiled_dd = transpiled

            # Compute CES on the DD circuit
            ces = deployer_no_dd._layout_selector.compute_ces(transpiled_dd)
            dd_ces.append(ces)

            # Execute with BackendEstimatorV2
            h_mapped = H_test.apply_layout(transpiled_dd.layout)
            estimator = BackendEstimatorV2(
                backend=backend,
                options={"default_precision": precision, "seed_simulator": SEED_LAYOUT},
            )
            job = estimator.run([(transpiled_dd, h_mapped)])
            result = job.result()
            energy = float(result[0].data.evs)
            dd_energies.append(energy)

        # Linear extrapolation
        ces_arr = np.array(dd_ces)
        energy_arr = np.array(dd_energies)
        if len(dd_ces) >= 2:
            coeffs = np.polyfit(ces_arr, energy_arr, 1)
            extrapolated_energy = float(np.polyval(coeffs, 0.0))
            y_pred = np.polyval(coeffs, ces_arr)
            ss_res = np.sum((energy_arr - y_pred) ** 2)
            ss_tot = np.sum((energy_arr - np.mean(energy_arr)) ** 2)
            r2_dd = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            r2_dd = 0.0
            extrapolated_energy = dd_energies[0] if dd_energies else 0.0

        delta_e_dd = abs(extrapolated_energy - exact_test.ground_energy)
        de_gap_dd = delta_e_dd / exact_test.gap if exact_test.gap > 0 else float("inf")
        t_dd = time.time() - t_dep

        print(
            f"    with-DD: ΔE/gap={de_gap_dd:.4f}, "
            f"R²={r2_dd:.4f}, CES={[f'{c:.3f}' for c in dd_ces]} ({t_dd:.1f}s)"
        )
        results_with_dd.append(
            {
                "h_test": h_test,
                "delta_e_over_gap": de_gap_dd,
                "zne_r_squared": r2_dd,
                "ces_values": dd_ces,
                "energies_per_layout": dd_energies,
                "extrapolated_energy": extrapolated_energy,
                "elapsed_s": t_dd,
            }
        )

    # ── Summary ──
    print(f"\n{'═' * 70}")
    print("  COMPARISON: no-DD vs with-DD")
    print(f"{'═' * 70}")
    print(f"  {'h_test':<8} {'no-DD R²':<12} {'DD R²':<12} {'no-DD ΔE/gap':<14} {'DD ΔE/gap':<14}")
    print(f"  {'─' * 58}")
    for i, h_test in enumerate(H_TEST_VALUES):
        r2_n = results_no_dd[i]["zne_r_squared"] or 0
        r2_d = results_with_dd[i]["zne_r_squared"] or 0
        de_n = results_no_dd[i]["delta_e_over_gap"]
        de_d = results_with_dd[i]["delta_e_over_gap"]
        print(f"  {h_test:<8.2f} {r2_n:<12.4f} {r2_d:<12.4f} {de_n:<14.4f} {de_d:<14.4f}")

    avg_r2_no_dd = np.mean([r["zne_r_squared"] or 0 for r in results_no_dd])
    avg_r2_dd = np.mean([r["zne_r_squared"] or 0 for r in results_with_dd])
    print(f"\n  Average R²: no-DD={avg_r2_no_dd:.4f}, with-DD={avg_r2_dd:.4f}")
    if avg_r2_dd > avg_r2_no_dd:
        print(f"  DD improvement: {(avg_r2_dd - avg_r2_no_dd) / max(avg_r2_no_dd, 1e-6):.1%}")
    else:
        print("  DD did NOT improve R²")

    # ── Save ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = hashlib.sha1(f"dd_experiment:{ts}".encode()).hexdigest()[:8]
    json_path = RESULTS_DIR / f"dd_experiment_{ts}_{run_id}.json"

    output = {
        "experiment": "B_dd_pre_mitigation",
        "hypothesis": "DD reduces effective noise, restoring linear E(CES) at N=10",
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "config": {
            "N": N,
            "p": p,
            "seed_mpnn": SEED_MPNN,
            "seed_layout": SEED_LAYOUT,
            "shots": SHOTS,
            "n_layouts": 3,
            "dd_sequence": "XY4",
            "h_test_values": H_TEST_VALUES,
        },
        "mpnn_mse": train_result["final_mse"],
        "results_no_dd": results_no_dd,
        "results_with_dd": results_with_dd,
        "avg_r2_no_dd": avg_r2_no_dd,
        "avg_r2_dd": avg_r2_dd,
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
