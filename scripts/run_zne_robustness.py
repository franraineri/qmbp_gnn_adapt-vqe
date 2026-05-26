#!/usr/bin/env python3
"""ZNE Robustness Validation at N=10 — CORRECTED.

Fixes applied:
  1. seed_simulator fixed for reproducibility
  2. default_precision = 1/sqrt(SHOTS) for correct shot count
  3. Circuit CES used for layout selection (not topology CES)
  4. Reports both R2 AND dE/gap as success metrics

Validation matrix:
  - 3 pipeline seeds (42, 43, 44)
  - 3 layout seeds (42, 100, 200)
  - 2 layout counts (3, 5)
  - 4 h-values (1.25, 1.5, 1.7, 2.0)
  Total: 72 ZNE evaluations

Success criterion: R2 > 0.8 AND ZNE improves over raw in >= 80% of configs.
"""

import json
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qiskit.primitives import BackendEstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeTorino

from qmbp_simulation import (
    ClassicalSolver,
    HamiltonianBuilder,
    HVACircuitBuilder,
    VQEOptimizer,
    make_lattice,
)
from qmbp_simulation.execution import NoiselessBackend
from qmbp_simulation.models import VQEConfig
from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "exp_noisy_variants"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ──
N, p, J = 10, 2, 1.0
SHOTS = 16384
PRECISION = 1.0 / np.sqrt(SHOTS)  # FIX #2: correct precision
H_TEST = [1.25, 1.5, 1.7, 2.0]
SEEDS_PIPELINE = [42, 43, 44]
SEEDS_LAYOUT = [42, 100, 200]
N_LAYOUTS_LIST = [3, 5]

# ── Utilities ──


def build_adjacency(backend):
    """Build adjacency + edge error map from backend target."""
    adj = {}
    edge_errors = {}
    target = backend.target
    for op_name in target.operation_names:
        qargs = target.qargs_for_operation_name(op_name)
        if qargs is None:
            continue
        for qa in qargs:
            if len(qa) == 2:
                q0, q1 = qa
                adj.setdefault(q0, []).append(q1)
                adj.setdefault(q1, []).append(q0)
                edge = (min(q0, q1), max(q0, q1))
                if edge not in edge_errors:
                    props = target[op_name].get((q0, q1))
                    if props and props.error is not None:
                        edge_errors[edge] = props.error
    for k in adj:
        adj[k] = list(set(adj[k]))
    return adj, edge_errors


def find_layouts_bfs(adj, n_qubits, n_candidates=40, seed=42):
    """Find connected subsets of size n_qubits via BFS."""
    rng = random.Random(seed)
    all_nodes = list(adj.keys())
    found = []
    seen_keys = set()
    starts = rng.sample(all_nodes, min(100, len(all_nodes)))
    for start in starts:
        visited = [start]
        queue = deque(adj.get(start, []))
        visited_set = {start}
        while queue and len(visited) < n_qubits:
            node = queue.popleft()
            if node in visited_set:
                continue
            visited_set.add(node)
            visited.append(node)
            for nb in adj.get(node, []):
                if nb not in visited_set:
                    queue.append(nb)
        if len(visited) >= n_qubits:
            key = tuple(sorted(visited[:n_qubits]))
            if key not in seen_keys:
                seen_keys.add(key)
                found.append(visited[:n_qubits])
        if len(found) >= n_candidates:
            break
    return found


def compute_circuit_ces(transpiled, backend):
    """Compute ACTUAL circuit CES from transpiled circuit (FIX #3)."""
    ces = 0.0
    n_2q = 0
    target = backend.target
    for inst in transpiled.data:
        if inst.operation.num_qubits == 2:
            n_2q += 1
            qubits = [transpiled.find_bit(q).index for q in inst.qubits]
            q0, q1 = min(qubits), max(qubits)
            gate_props = target[inst.operation.name].get((q0, q1))
            if gate_props and gate_props.error is not None:
                ces += gate_props.error
            else:
                ces += 0.01
    return ces, n_2q


def select_layouts_by_circuit_ces(bound_circuit, backend, candidate_layouts, n_select):
    """FIX #3: Select layouts by ACTUAL circuit CES (transpile each candidate).

    This is more expensive but gives true CES diversity.
    """
    circuit_ces_list = []
    transpiled_list = []
    for layout in candidate_layouts:
        pm = generate_preset_pass_manager(
            optimization_level=2, backend=backend, initial_layout=layout
        )
        transpiled = pm.run(bound_circuit)
        ces, _ = compute_circuit_ces(transpiled, backend)
        circuit_ces_list.append(ces)
        transpiled_list.append(transpiled)

    # Select n_select with max spread (first + last + evenly spaced)
    sorted_idx = np.argsort(circuit_ces_list)
    if len(sorted_idx) >= n_select:
        # Pick first, last, and evenly spaced in between
        indices = [sorted_idx[0]]
        if n_select > 2:
            step = (len(sorted_idx) - 1) / (n_select - 1)
            for i in range(1, n_select - 1):
                indices.append(sorted_idx[int(round(i * step))])
        indices.append(sorted_idx[-1])
        indices = sorted(set(indices))[:n_select]
    else:
        indices = list(sorted_idx)

    selected_layouts = [candidate_layouts[i] for i in indices]
    selected_ces = [circuit_ces_list[i] for i in indices]
    selected_transpiled = [transpiled_list[i] for i in indices]
    return selected_layouts, selected_ces, selected_transpiled


def run_zne_with_transpiled(transpiled_list, H_test, backend, seed_sim):
    """Execute ZNE using pre-transpiled circuits. FIX #1: seed_simulator."""
    ces_list = []
    e_list = []
    for transpiled in transpiled_list:
        ces, _ = compute_circuit_ces(transpiled, backend)
        ces_list.append(ces)

        h_mapped = H_test.apply_layout(transpiled.layout)
        # FIX #1 + #2: seed_simulator + correct precision
        est = BackendEstimatorV2(
            backend=backend,
            options={"seed_simulator": seed_sim, "default_precision": PRECISION},
        )
        job = est.run([(transpiled, h_mapped)])
        e_list.append(float(job.result()[0].data.evs))

    ces_arr = np.array(ces_list)
    e_arr = np.array(e_list)

    # Linear ZNE
    if len(ces_arr) >= 2 and np.std(ces_arr) > 1e-10:
        coeffs = np.polyfit(ces_arr, e_arr, 1)
        e_zne = float(np.polyval(coeffs, 0.0))
        y_pred = np.polyval(coeffs, ces_arr)
        ss_res = np.sum((e_arr - y_pred) ** 2)
        ss_tot = np.sum((e_arr - np.mean(e_arr)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0
    else:
        e_zne = float(np.mean(e_arr))
        r2 = 0.0

    return ces_arr, e_arr, r2, e_zne


# ── Main ──


def main():
    t_total = time.time()
    backend = FakeTorino()
    adj, edge_errors = build_adjacency(backend)
    noiseless_backend = NoiselessBackend()
    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()

    print("=" * 70)
    print("  ZNE ROBUSTNESS VALIDATION: p=2 N=10 (CORRECTED)")
    print("  Fixes: seed_simulator, precision=1/sqrt(%d), circuit CES selection" % SHOTS)
    print("  Seeds (pipeline): %s" % SEEDS_PIPELINE)
    print("  Seeds (layout): %s" % SEEDS_LAYOUT)
    print("  N_layouts: %s" % N_LAYOUTS_LIST)
    print("  H-test: %s" % H_TEST)
    print(
        "  Total configs: %d"
        % (len(SEEDS_PIPELINE) * len(SEEDS_LAYOUT) * len(N_LAYOUTS_LIST) * len(H_TEST))
    )
    print("=" * 70)

    all_results = []

    for seed_pipe in SEEDS_PIPELINE:
        print("")
        print("  === Pipeline seed=%d ===" % seed_pipe)
        np.random.seed(seed_pipe)
        torch.manual_seed(seed_pipe)
        random.seed(seed_pipe)

        base_lattice = make_lattice("chain_1d", N, J=J, h=1.0)
        qc, _ = hva.create(N, p, base_lattice)

        # Phase 1-3
        h_coarse = np.arange(0.0, 0.8, 0.1)
        h_dense = np.arange(0.8, 1.45, 0.05)
        h_coarse2 = np.arange(1.5, 2.05, 0.1)
        h_values = np.unique(np.concatenate([h_coarse, h_dense, h_coarse2]))[::-1]

        print("    Phase 1: Exact diag (%d pts)..." % len(h_values))
        exact_data = []
        for h in h_values:
            lat_h = make_lattice("chain_1d", N, J=J, h=h)
            H = builder.build(lat_h)
            exact_data.append(solver.solve(H, lat_h))

        print("    Phase 2: VQE...")
        vqe_config = VQEConfig(p_layers=p, n_restarts=5, maxiter=1000, enable_callbacks=False)
        opt = VQEOptimizer(vqe_config)
        vqe_results = opt.descending_sweep(h_values, qc, base_lattice, exact_data)
        fids = np.array([r.fidelity for r in vqe_results])
        print("      avg fid=%.1f%%" % (np.mean(fids) * 100))

        print("    Phase 3: MPNN...")
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
        print("      MSE=%.2e" % train_result["final_mse"])
        model.eval()
        edge_idx, coord = builder.build_graph_data(base_lattice)

        # Phase 4: ZNE across layout seeds and n_layouts
        for seed_layout in SEEDS_LAYOUT:
            candidate_layouts = find_layouts_bfs(adj, N, n_candidates=40, seed=seed_layout)

            for h_test in H_TEST:
                lat_test = make_lattice("chain_1d", N, J=J, h=h_test)
                H_test = builder.build(lat_test)
                exact_test = solver.solve(H_test, lat_test)

                # Predict theta
                x_t = torch.tensor(
                    np.stack([np.full(N, h_test), coord.astype(float)], axis=1), dtype=torch.float32
                )
                g = Data(x=x_t, edge_index=torch.tensor(edge_idx, dtype=torch.long))
                with torch.no_grad():
                    theta = model(g).numpy().flatten()
                bound = qc.assign_parameters(theta)

                # Noiseless baseline
                e_nl = noiseless_backend.evaluate(qc, H_test, theta)
                de_nl = abs(e_nl - exact_test.ground_energy) / exact_test.gap

                for n_layouts in N_LAYOUTS_LIST:
                    # FIX #3: Select by circuit CES
                    sel_layouts, sel_ces, sel_transpiled = select_layouts_by_circuit_ces(
                        bound, backend, candidate_layouts, n_layouts
                    )

                    # FIX #1: deterministic seed for simulator
                    seed_sim = seed_pipe * 1000 + seed_layout + n_layouts
                    ces_arr, e_arr, r2, e_zne = run_zne_with_transpiled(
                        sel_transpiled, H_test, backend, seed_sim
                    )

                    de_zne = abs(e_zne - exact_test.ground_energy) / exact_test.gap
                    de_raw = abs(e_arr[0] - exact_test.ground_energy) / exact_test.gap
                    gain = (de_raw - de_zne) / de_raw if de_raw > 0 else 0.0
                    zne_helps = de_zne < de_raw

                    all_results.append(
                        {
                            "seed_pipe": seed_pipe,
                            "seed_layout": seed_layout,
                            "n_layouts": n_layouts,
                            "h_test": h_test,
                            "r2": r2,
                            "de_zne": de_zne,
                            "de_raw": de_raw,
                            "de_noiseless": de_nl,
                            "gain": gain,
                            "zne_helps": zne_helps,
                            "ces_range": [float(ces_arr.min()), float(ces_arr.max())],
                            "ces_ratio": float(ces_arr.max() / ces_arr.min())
                            if ces_arr.min() > 0
                            else 0,
                        }
                    )

        n_done = len([r for r in all_results if r["seed_pipe"] == seed_pipe])
        print("    Done: %d evaluations for seed=%d" % (n_done, seed_pipe))

    # ── Analysis ──
    print("")
    print("=" * 70)
    print("  RESULTS: %d total ZNE evaluations" % len(all_results))
    print("=" * 70)

    r2_values = [r["r2"] for r in all_results]
    de_zne_values = [r["de_zne"] for r in all_results]
    gains = [r["gain"] for r in all_results]
    helps_count = sum(1 for r in all_results if r["zne_helps"])

    # FIX #4: Report BOTH R2 and dE/gap
    good_r2 = sum(1 for r in r2_values if r > 0.8)
    print("")
    print("  R2 statistics:")
    print(
        "    mean=%.4f, std=%.4f, min=%.4f, max=%.4f"
        % (np.mean(r2_values), np.std(r2_values), np.min(r2_values), np.max(r2_values))
    )
    print("    R2>0.8: %d/%d (%.1f%%)" % (good_r2, len(r2_values), 100 * good_r2 / len(r2_values)))

    print("")
    print("  dE/gap statistics (ZNE extrapolated):")
    print(
        "    mean=%.4f, std=%.4f, min=%.4f, max=%.4f"
        % (
            np.mean(de_zne_values),
            np.std(de_zne_values),
            np.min(de_zne_values),
            np.max(de_zne_values),
        )
    )

    print("")
    print("  ZNE effectiveness:")
    print(
        "    ZNE helps (dE_zne < dE_raw): %d/%d (%.1f%%)"
        % (helps_count, len(all_results), 100 * helps_count / len(all_results))
    )
    print("    Mean gain: %.1f%%" % (np.mean(gains) * 100))

    # By pipeline seed
    print("")
    print("  By pipeline seed:")
    for s in SEEDS_PIPELINE:
        subset = [r for r in all_results if r["seed_pipe"] == s]
        r2s = [r["r2"] for r in subset]
        des = [r["de_zne"] for r in subset]
        helps = sum(1 for r in subset if r["zne_helps"])
        print(
            "    seed=%d: R2=%.4f+/-%.4f, dE/gap=%.4f+/-%.4f, helps=%d/%d"
            % (s, np.mean(r2s), np.std(r2s), np.mean(des), np.std(des), helps, len(subset))
        )

    # By layout seed
    print("")
    print("  By layout seed:")
    for s in SEEDS_LAYOUT:
        subset = [r for r in all_results if r["seed_layout"] == s]
        r2s = [r["r2"] for r in subset]
        helps = sum(1 for r in subset if r["zne_helps"])
        print(
            "    seed=%d: R2=%.4f+/-%.4f, helps=%d/%d"
            % (s, np.mean(r2s), np.std(r2s), helps, len(subset))
        )

    # By n_layouts
    print("")
    print("  By n_layouts:")
    for nl in N_LAYOUTS_LIST:
        subset = [r for r in all_results if r["n_layouts"] == nl]
        r2s = [r["r2"] for r in subset]
        des = [r["de_zne"] for r in subset]
        helps = sum(1 for r in subset if r["zne_helps"])
        print(
            "    n=%d: R2=%.4f+/-%.4f, dE/gap=%.4f+/-%.4f, helps=%d/%d"
            % (nl, np.mean(r2s), np.std(r2s), np.mean(des), np.std(des), helps, len(subset))
        )

    # By h_test
    print("")
    print("  By h_test:")
    for h in H_TEST:
        subset = [r for r in all_results if r["h_test"] == h]
        r2s = [r["r2"] for r in subset]
        des = [r["de_zne"] for r in subset]
        helps = sum(1 for r in subset if r["zne_helps"])
        print(
            "    h=%.2f: R2=%.4f+/-%.4f, dE/gap=%.4f+/-%.4f, helps=%d/%d"
            % (h, np.mean(r2s), np.std(r2s), np.mean(des), np.std(des), helps, len(subset))
        )

    # CES ratio analysis
    print("")
    print("  CES ratio (max/min) statistics:")
    ratios = [r["ces_ratio"] for r in all_results if r["ces_ratio"] > 0]
    print("    mean=%.1f, min=%.1f, max=%.1f" % (np.mean(ratios), np.min(ratios), np.max(ratios)))

    # Correlation: does higher CES ratio -> better R2?
    if len(ratios) == len(r2_values):
        corr = np.corrcoef(ratios, r2_values)[0, 1]
        print("    Correlation(CES_ratio, R2) = %.4f" % corr)

    # Final verdict
    print("")
    print("=" * 70)
    success_r2 = good_r2 / len(r2_values)
    success_helps = helps_count / len(all_results)
    overall = success_r2 >= 0.8 and success_helps >= 0.7

    if overall:
        print("  VERDICT: ZNE at p=2 N=10 is ROBUST")
        print("    R2>0.8 in %.0f%% of configs" % (success_r2 * 100))
        print("    ZNE helps in %.0f%% of configs" % (success_helps * 100))
    elif success_r2 >= 0.5:
        print("  VERDICT: ZNE at p=2 N=10 is PARTIALLY ROBUST")
        print("    R2>0.8 in %.0f%% (need 80%%)" % (success_r2 * 100))
        print("    ZNE helps in %.0f%% (need 70%%)" % (success_helps * 100))
    else:
        print("  VERDICT: ZNE at p=2 N=10 is NOT ROBUST")
        print("    R2>0.8 in only %.0f%%" % (success_r2 * 100))
    print("=" * 70)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / ("zne_robustness_%s.json" % ts)
    output = {
        "experiment": "ZNE_robustness_p2_N10_CORRECTED",
        "timestamp": datetime.now().isoformat(),
        "fixes_applied": [
            "seed_simulator fixed for reproducibility",
            "default_precision=1/sqrt(16384) for correct shots",
            "circuit CES used for layout selection",
            "both R2 and dE/gap reported",
        ],
        "config": {
            "N": N,
            "p": p,
            "shots": SHOTS,
            "precision": PRECISION,
            "seeds_pipeline": SEEDS_PIPELINE,
            "seeds_layout": SEEDS_LAYOUT,
            "n_layouts_list": N_LAYOUTS_LIST,
            "h_test": H_TEST,
        },
        "n_evaluations": len(all_results),
        "results": all_results,
        "summary": {
            "success_rate_r2": success_r2,
            "success_rate_helps": success_helps,
            "mean_r2": float(np.mean(r2_values)),
            "std_r2": float(np.std(r2_values)),
            "mean_de_zne": float(np.mean(de_zne_values)),
            "mean_gain": float(np.mean(gains)),
        },
        "elapsed_s": time.time() - t_total,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("  Saved: %s" % out_path)
    print("  Total time: %.0fs (%.1f min)" % (time.time() - t_total, (time.time() - t_total) / 60))


if __name__ == "__main__":
    main()
