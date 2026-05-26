#!/usr/bin/env python3
"""Noisy Variants Batch 2: Experiments #1, #3, #4, #8.

#1: p=2 N=10 with correct layout selection (re-validate ZNE failure)
#3: CES outlier filtering (post-processing of #1 data)
#4: p=1 N=10 + per-site ZNE combined
#8: ZNE + SPSA refinement

All use FakeTorino + explicit BFS layout selection.
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
from qiskit.quantum_info import SparsePauliOp
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

SEED = 43
SHOTS = 16384
PRECISION = 1.0 / np.sqrt(SHOTS)  # Correct precision for BackendEstimatorV2

# ═══════════════════════════════════════════════════════════════════════
# SHARED: Layout selection on FakeTorino heavy-hex
# ═══════════════════════════════════════════════════════════════════════


def build_adjacency(backend):
    """Build adjacency graph from backend target."""
    adj = {}
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
    # Deduplicate
    for k in adj:
        adj[k] = list(set(adj[k]))
    return adj


def find_layouts_bfs(adj, n_qubits, n_candidates=30, seed=42):
    """Find connected subsets of size n_qubits via BFS."""
    rng = random.Random(seed)
    all_nodes = list(adj.keys())
    found = []
    starts = rng.sample(all_nodes, min(80, len(all_nodes)))
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
            if key not in {tuple(sorted(f)) for f in found}:
                found.append(visited[:n_qubits])
        if len(found) >= n_candidates:
            break
    return found


def compute_topology_ces(layout, adj, backend):
    """Compute topology CES for a layout (sum of internal edge errors)."""
    target = backend.target
    layout_set = set(layout)
    ces = 0.0
    for q in layout:
        for nb in adj.get(q, []):
            if nb in layout_set and nb > q:
                # Find 2Q gate error for this edge
                for op_name in target.operation_names:
                    props = target[op_name].get((q, nb))
                    if props and props.error is not None:
                        ces += props.error
                        break
                else:
                    ces += 0.01
    return ces


def select_diverse_layouts(layouts, adj, backend, n_select=3):
    """Select n_select layouts with maximum CES spread."""
    ces_list = [compute_topology_ces(l, adj, backend) for l in layouts]
    sorted_idx = np.argsort(ces_list)
    if len(sorted_idx) >= n_select:
        step = max(1, (len(sorted_idx) - 1) // (n_select - 1))
        selected = [sorted_idx[i * step] for i in range(n_select)]
        # Ensure we include first and last
        selected[0] = sorted_idx[0]
        selected[-1] = sorted_idx[-1]
        selected = sorted(set(selected))[:n_select]
    else:
        selected = list(sorted_idx)
    return [layouts[i] for i in selected], [ces_list[i] for i in selected]


def compute_circuit_ces(transpiled, backend):
    """Compute actual circuit CES from transpiled circuit."""
    ces = 0.0
    n_2q = 0
    for inst in transpiled.data:
        if inst.operation.num_qubits == 2:
            n_2q += 1
            qubits = [transpiled.find_bit(q).index for q in inst.qubits]
            q0, q1 = min(qubits), max(qubits)
            gate_props = backend.target[inst.operation.name].get((q0, q1))
            if gate_props and gate_props.error is not None:
                ces += gate_props.error
            else:
                ces += 0.01
    return ces, n_2q


def select_layouts_by_circuit_ces(bound_circuit, backend, candidate_layouts, n_select):
    """Select layouts by ACTUAL circuit CES (transpile each candidate).

    This is more expensive but gives true CES diversity instead of topology CES.
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


def run_zne_deployment(
    bound_circuit, H_test, exact_test, backend, layouts, n_qubits, seed_base=SEED
):
    """Run ZNE deployment with given layouts. Returns per-layout data.

    Uses seed_simulator and default_precision for reproducibility and correct shots.
    """
    layout_data = []
    for li, init_layout in enumerate(layouts):
        pm = generate_preset_pass_manager(
            optimization_level=2,
            backend=backend,
            initial_layout=init_layout,
        )
        transpiled = pm.run(bound_circuit)
        ces, n_2q = compute_circuit_ces(transpiled, backend)

        # Energy — with seed_simulator + correct precision
        h_mapped = H_test.apply_layout(transpiled.layout)
        estimator = BackendEstimatorV2(
            backend=backend,
            options={"seed_simulator": seed_base + li, "default_precision": PRECISION},
        )
        job = estimator.run([(transpiled, h_mapped)])
        energy = float(job.result()[0].data.evs)

        # Per-site X
        per_site_x = []
        for i in range(n_qubits):
            op = SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=n_qubits)
            x_mapped = op.apply_layout(transpiled.layout)
            job_x = estimator.run([(transpiled, x_mapped)])
            per_site_x.append(float(job_x.result()[0].data.evs))

        layout_data.append(
            {
                "ces": ces,
                "n_2q": n_2q,
                "energy": energy,
                "per_site_x": per_site_x,
            }
        )
    return layout_data


def linear_zne(ces_arr, values_arr):
    """Linear ZNE extrapolation to CES=0. Returns (extrap_value, r2)."""
    coeffs = np.polyfit(ces_arr, values_arr, 1)
    extrap = float(np.polyval(coeffs, 0.0))
    y_pred = np.polyval(coeffs, ces_arr)
    ss_res = np.sum((values_arr - y_pred) ** 2)
    ss_tot = np.sum((values_arr - np.mean(values_arr)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0
    return extrap, r2


# ═══════════════════════════════════════════════════════════════════════
# SHARED: Pipeline builder (Phase 1-3)
# ═══════════════════════════════════════════════════════════════════════


def build_pipeline(N, p, h_range, seed=SEED):
    """Run Phase 1-3 and return (model, qc, builder, solver, exact_cache)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    base_lattice = make_lattice("chain_1d", N, J=1.0, h=1.0)
    qc, _ = hva.create(N, p, base_lattice)

    h_values = np.arange(h_range[0], h_range[1] + 0.01, 0.1)[::-1]  # Descending
    print("    Phase 1: Exact diag (%d h-points)..." % len(h_values))
    t1 = time.time()
    exact_data = []
    for h in h_values:
        lat_h = make_lattice("chain_1d", N, J=1.0, h=h)
        H = builder.build(lat_h)
        exact_data.append(solver.solve(H, lat_h))
    print("      Done in %.1fs" % (time.time() - t1))

    print("    Phase 2: VQE descending sweep (5 restarts)...")
    t2 = time.time()
    vqe_config = VQEConfig(p_layers=p, n_restarts=5, maxiter=1000, enable_callbacks=False)
    opt = VQEOptimizer(vqe_config)
    vqe_results = opt.descending_sweep(h_values, qc, base_lattice, exact_data)
    fids = np.array([r.fidelity for r in vqe_results])
    print("      Done in %.1fs -- avg fid=%.1f%%" % (time.time() - t2, np.mean(fids) * 100))

    print("    Phase 3: MPNN (h=128, L=3, 6000ep)...")
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
    print("      Done in %.1fs -- MSE=%.2e" % (time.time() - t3, train_result["final_mse"]))
    model.eval()

    return model, qc, builder, solver, base_lattice, train_result["final_mse"]


def predict_theta(model, builder, base_lattice, N, h_test):
    """Get MPNN prediction for a given h_test."""
    edge_idx, coord = builder.build_graph_data(base_lattice)
    x_test = torch.tensor(
        np.stack([np.full(N, h_test), coord.astype(float)], axis=1),
        dtype=torch.float32,
    )
    test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))
    with torch.no_grad():
        return model(test_graph).numpy().flatten()


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT #1: p=2 N=10 with correct layout selection
# ═══════════════════════════════════════════════════════════════════════


def run_exp1_p2_correct_layouts():
    """Re-validate ZNE at p=2 N=10 with proper BFS layout selection."""
    print("")
    print("=" * 70)
    print("  EXP #1: p=2 N=10 with CORRECT layout selection")
    print("  Question: Was the original ZNE failure just bad layout selection?")
    print("=" * 70)

    N, p = 10, 2
    H_TEST = [1.0, 1.25, 1.5, 1.7, 2.0]
    N_LAYOUTS = 5  # Use 5 for better statistics

    print("  Building pipeline (N=%d, p=%d)..." % (N, p))
    model, qc, builder, solver, base_lattice, mse = build_pipeline(N, p, (0.0, 2.0))

    backend = FakeTorino()
    adj = build_adjacency(backend)
    all_layouts = find_layouts_bfs(adj, N, n_candidates=30, seed=SEED)

    noiseless_backend = NoiselessBackend()
    results = []

    for h_test in H_TEST:
        print("")
        print("  h=%.2f:" % h_test)
        lat_test = make_lattice("chain_1d", N, J=1.0, h=h_test)
        H_test = builder.build(lat_test)
        exact_test = solver.solve(H_test, lat_test)
        theta_pred = predict_theta(model, builder, base_lattice, N, h_test)
        bound = qc.assign_parameters(theta_pred)

        # Select layouts by CIRCUIT CES (not topology CES)
        selected, circuit_ces, _ = select_layouts_by_circuit_ces(
            bound, backend, all_layouts, N_LAYOUTS
        )
        print(
            "    Selected %d layouts, circuit CES: %s"
            % (len(selected), [round(c, 4) for c in circuit_ces])
        )

        # Noiseless
        e_noiseless = noiseless_backend.evaluate(qc, H_test, theta_pred)
        de_noiseless = abs(e_noiseless - exact_test.ground_energy) / exact_test.gap

        # Noisy with ZNE
        layout_data = run_zne_deployment(bound, H_test, exact_test, backend, selected, N)
        ces_arr = np.array([d["ces"] for d in layout_data])
        e_arr = np.array([d["energy"] for d in layout_data])

        # ZNE extrapolation
        e_zne, r2 = linear_zne(ces_arr, e_arr)
        de_zne = abs(e_zne - exact_test.ground_energy) / exact_test.gap
        de_raw = abs(e_arr[0] - exact_test.ground_energy) / exact_test.gap
        gain = (de_raw - de_zne) / de_raw if de_raw > 0 else 0.0

        print("    CES: %s" % [round(c, 3) for c in ces_arr.tolist()])
        print("    Noiseless dE/gap: %.4f" % de_noiseless)
        print("    Raw dE/gap: %.4f" % de_raw)
        print("    ZNE dE/gap: %.4f, R2=%.4f, gain=%.1f%%" % (de_zne, r2, gain * 100))
        status = "PASS" if r2 > 0.8 else "FAIL"
        print("    [%s]" % status)

        results.append(
            {
                "h_test": h_test,
                "noiseless_de_gap": de_noiseless,
                "raw_de_gap": de_raw,
                "zne_de_gap": de_zne,
                "r2": r2,
                "gain": gain,
                "ces_values": ces_arr.tolist(),
                "energies": e_arr.tolist(),
                "n_2q": layout_data[0]["n_2q"],
            }
        )

    # Summary
    avg_r2 = np.mean([r["r2"] for r in results])
    good = sum(1 for r in results if r["r2"] > 0.8)
    print("")
    print("  SUMMARY: avg R2=%.4f, good R2: %d/%d" % (avg_r2, good, len(results)))
    if avg_r2 > 0.8:
        print("  CONCLUSION: ZNE WORKS at p=2 N=10 with correct layout selection!")
        print("  The original failure was ENTIRELY due to bad layout selection.")
    else:
        print("  CONCLUSION: ZNE still fails even with correct layouts.")

    return {
        "exp": "1_p2_correct_layouts",
        "results": results,
        "summary": {"avg_r2": avg_r2, "good_count": good},
        "mpnn_mse": mse,
    }


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT #3: CES outlier filtering (post-processing of #1 data)
# ═══════════════════════════════════════════════════════════════════════


def run_exp3_ces_filtering(exp1_results):
    """Filter CES outliers and re-do ZNE with only perturbative layouts."""
    print("")
    print("=" * 70)
    print("  EXP #3: CES Outlier Filtering")
    print("  Question: Does removing high-CES layouts improve extrapolation?")
    print("=" * 70)

    results = []
    for r in exp1_results["results"]:
        h_test = r["h_test"]
        ces_arr = np.array(r["ces_values"])
        e_arr = np.array(r["energies"])

        # Full ZNE (all layouts)
        e_full, r2_full = linear_zne(ces_arr, e_arr)

        # Filtered: keep only CES < 2 * median
        median_ces = np.median(ces_arr)
        mask = ces_arr < 2.0 * median_ces
        ces_filt = ces_arr[mask]
        e_filt = e_arr[mask]

        if len(ces_filt) >= 2:
            e_filtered, r2_filtered = linear_zne(ces_filt, e_filt)
        else:
            e_filtered, r2_filtered = e_full, r2_full

        # Strict filter: CES < 1.0 only
        mask_strict = ces_arr < 1.0
        ces_strict = ces_arr[mask_strict]
        e_strict = e_arr[mask_strict]
        if len(ces_strict) >= 2:
            e_strict_zne, r2_strict = linear_zne(ces_strict, e_strict)
        else:
            e_strict_zne, r2_strict = e_full, r2_full

        print(
            "  h=%.2f: full R2=%.4f (%d pts), filtered R2=%.4f (%d pts), strict R2=%.4f (%d pts)"
            % (
                h_test,
                r2_full,
                len(ces_arr),
                r2_filtered,
                len(ces_filt),
                r2_strict,
                len(ces_strict),
            )
        )

        results.append(
            {
                "h_test": h_test,
                "full": {"r2": r2_full, "n_points": len(ces_arr)},
                "filtered_2x_median": {"r2": r2_filtered, "n_points": int(np.sum(mask))},
                "strict_lt_1": {"r2": r2_strict, "n_points": int(np.sum(mask_strict))},
            }
        )

    print("")
    print("  CONCLUSION:")
    avg_full = np.mean([r["full"]["r2"] for r in results])
    avg_filt = np.mean([r["filtered_2x_median"]["r2"] for r in results])
    avg_strict = np.mean([r["strict_lt_1"]["r2"] for r in results])
    print("    Full (all layouts): avg R2=%.4f" % avg_full)
    print("    Filtered (CES<2*median): avg R2=%.4f" % avg_filt)
    print("    Strict (CES<1.0): avg R2=%.4f" % avg_strict)
    if avg_filt > avg_full:
        print("    -> Filtering IMPROVES R2")
    else:
        print("    -> Filtering does NOT help (full set is already good)")

    return {"exp": "3_ces_filtering", "results": results}


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT #4: p=1 N=10 + per-site ZNE combined
# ═══════════════════════════════════════════════════════════════════════


def run_exp4_p1_persite():
    """Combine p=1 (shallow) with per-site ZNE for maximum improvement."""
    print("")
    print("=" * 70)
    print("  EXP #4: p=1 N=10 + Per-Site ZNE Combined")
    print("  Question: Can we approach 5% threshold with both techniques?")
    print("=" * 70)

    N, p = 10, 1
    H_TEST = [2.0, 2.5, 3.0]
    N_LAYOUTS = 5

    print("  Building pipeline (N=%d, p=%d)..." % (N, p))
    model, qc, builder, solver, base_lattice, mse = build_pipeline(N, p, (1.9, 4.0))

    backend = FakeTorino()
    adj = build_adjacency(backend)
    all_layouts = find_layouts_bfs(adj, N, n_candidates=30, seed=SEED)

    noiseless_backend = NoiselessBackend()
    results = []

    for h_test in H_TEST:
        print("")
        print("  h=%.2f:" % h_test)
        lat_test = make_lattice("chain_1d", N, J=1.0, h=h_test)
        H_test = builder.build(lat_test)
        exact_test = solver.solve(H_test, lat_test)
        theta_pred = predict_theta(model, builder, base_lattice, N, h_test)
        bound = qc.assign_parameters(theta_pred)

        # Select layouts by CIRCUIT CES (not topology CES)
        selected, circuit_ces, _ = select_layouts_by_circuit_ces(
            bound, backend, all_layouts, N_LAYOUTS
        )
        print(
            "    Selected %d layouts, circuit CES: %s"
            % (len(selected), [round(c, 4) for c in circuit_ces])
        )

        # Noiseless
        e_noiseless = noiseless_backend.evaluate(qc, H_test, theta_pred)
        de_noiseless = abs(e_noiseless - exact_test.ground_energy) / exact_test.gap

        # Noisy with per-site data
        layout_data = run_zne_deployment(bound, H_test, exact_test, backend, selected, N)
        ces_arr = np.array([d["ces"] for d in layout_data])
        e_arr = np.array([d["energy"] for d in layout_data])
        per_site_arr = np.array([d["per_site_x"] for d in layout_data])

        # Total energy ZNE
        e_zne, r2_total = linear_zne(ces_arr, e_arr)
        de_zne = abs(e_zne - exact_test.ground_energy) / exact_test.gap
        de_raw = abs(e_arr[0] - exact_test.ground_energy) / exact_test.gap

        # Per-site ZNE
        noiseless_per_site = exact_test.per_site_mag_x
        mag_x_noiseless = float(np.mean(noiseless_per_site))

        per_site_r2 = []
        per_site_extrap = []
        for site_i in range(N):
            site_vals = per_site_arr[:, site_i]
            extrap, r2_s = linear_zne(ces_arr, site_vals)
            per_site_r2.append(r2_s)
            per_site_extrap.append(extrap)

        # Good sites
        good_sites = [i for i, r2 in enumerate(per_site_r2) if r2 > 0.5]
        if good_sites:
            mag_x_good = np.mean([per_site_extrap[i] for i in good_sites])
        else:
            mag_x_good = np.mean(per_site_extrap)

        mag_x_raw = float(np.mean(per_site_arr[0]))
        mag_x_error_raw = abs(mag_x_raw - mag_x_noiseless)
        mag_x_error_good = abs(mag_x_good - mag_x_noiseless)
        improvement = (
            (mag_x_error_raw - mag_x_error_good) / mag_x_error_raw if mag_x_error_raw > 0 else 0
        )

        print("    CES: %s" % [round(c, 3) for c in ces_arr.tolist()])
        print("    Energy R2: %.4f" % r2_total)
        print("    Noiseless dE/gap: %.4f" % de_noiseless)
        print("    Raw dE/gap: %.4f" % de_raw)
        print("    ZNE dE/gap: %.4f" % de_zne)
        print(
            "    Good sites: %d/%d, <X> improvement: %.1f%%"
            % (len(good_sites), N, improvement * 100)
        )
        passes = de_zne < 0.05
        print("    Passes 5%%: %s" % ("YES!" if passes else "no (%.1f%%)" % (de_zne * 100)))

        results.append(
            {
                "h_test": h_test,
                "de_noiseless": de_noiseless,
                "de_raw": de_raw,
                "de_zne": de_zne,
                "r2_total": r2_total,
                "good_sites": good_sites,
                "per_site_r2": per_site_r2,
                "mag_x_improvement": improvement,
                "passes_5pct": passes,
                "ces_values": ces_arr.tolist(),
            }
        )

    avg_r2 = np.mean([r["r2_total"] for r in results])
    avg_de_zne = np.mean([r["de_zne"] for r in results])
    print("")
    print("  SUMMARY: avg R2=%.4f, avg ZNE dE/gap=%.4f" % (avg_r2, avg_de_zne))
    return {
        "exp": "4_p1_persite",
        "results": results,
        "mpnn_mse": mse,
        "summary": {"avg_r2": avg_r2, "avg_de_zne": avg_de_zne},
    }


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT #8: ZNE + SPSA refinement
# ═══════════════════════════════════════════════════════════════════════


def run_exp8_zne_plus_spsa():
    """After ZNE gives a mitigated energy estimate, refine theta with SPSA."""
    print("")
    print("=" * 70)
    print("  EXP #8: ZNE + SPSA Refinement")
    print("  Question: Can SPSA refine theta after ZNE gives a good starting point?")
    print("  Note: V7 4B showed SPSA hurts warm-start in noiseless. Under noise")
    print("  with ZNE-corrected cost, it might help.")
    print("=" * 70)

    N, p = 10, 2
    H_TEST = [1.5, 2.0]
    N_LAYOUTS = 3
    SPSA_ITERS = 50
    SPSA_A = 0.1
    SPSA_C = 0.05
    SPSA_STAB = 10

    print("  Building pipeline (N=%d, p=%d)..." % (N, p))
    model, qc, builder, solver, base_lattice, mse = build_pipeline(N, p, (0.0, 2.0))

    backend = FakeTorino()
    adj = build_adjacency(backend)
    all_layouts = find_layouts_bfs(adj, N, n_candidates=30, seed=SEED)

    noiseless_backend = NoiselessBackend()
    results = []

    for h_test in H_TEST:
        print("")
        print("  h=%.2f:" % h_test)
        lat_test = make_lattice("chain_1d", N, J=1.0, h=h_test)
        H_test = builder.build(lat_test)
        exact_test = solver.solve(H_test, lat_test)
        theta_pred = predict_theta(model, builder, base_lattice, N, h_test)

        # Noiseless baseline
        e_noiseless = noiseless_backend.evaluate(qc, H_test, theta_pred)
        de_noiseless = abs(e_noiseless - exact_test.ground_energy) / exact_test.gap

        # Select layouts by CIRCUIT CES
        bound_before = qc.assign_parameters(theta_pred)
        selected, circuit_ces, _ = select_layouts_by_circuit_ces(
            bound_before, backend, all_layouts, N_LAYOUTS
        )

        # ZNE with MPNN prediction (before SPSA)
        layout_data_before = run_zne_deployment(
            bound_before, H_test, exact_test, backend, selected, N
        )
        ces_arr = np.array([d["ces"] for d in layout_data_before])
        e_before = np.array([d["energy"] for d in layout_data_before])
        e_zne_before, r2_before = linear_zne(ces_arr, e_before)
        de_zne_before = abs(e_zne_before - exact_test.ground_energy) / exact_test.gap

        # SPSA refinement: optimize theta using ZNE-mitigated cost
        # Cost function: for given theta, transpile to all layouts, measure, extrapolate
        print("    Running SPSA (%d iterations)..." % SPSA_ITERS)
        t_spsa = time.time()
        theta_current = theta_pred.copy()
        best_theta = theta_current.copy()
        best_energy = e_zne_before

        def zne_cost(theta):
            """ZNE-mitigated energy for given theta."""
            bound = qc.assign_parameters(theta)
            energies = []
            for li, init_layout in enumerate(selected):
                pm = generate_preset_pass_manager(
                    optimization_level=2, backend=backend, initial_layout=init_layout
                )
                transpiled = pm.run(bound)
                h_mapped = H_test.apply_layout(transpiled.layout)
                est = BackendEstimatorV2(
                    backend=backend,
                    options={"seed_simulator": SEED + li + 500, "default_precision": PRECISION},
                )
                job = est.run([(transpiled, h_mapped)])
                energies.append(float(job.result()[0].data.evs))
            ces_local = ces_arr  # Reuse CES from initial run (same layouts)
            extrap, _ = linear_zne(ces_local, np.array(energies))
            return extrap

        for k in range(SPSA_ITERS):
            ak = SPSA_A / (k + 1 + SPSA_STAB) ** 0.602
            ck = SPSA_C / (k + 1) ** 0.101

            # Random perturbation direction
            delta = np.random.choice([-1, 1], size=len(theta_current))

            # Evaluate at theta +/- ck*delta
            theta_plus = theta_current + ck * delta
            theta_minus = theta_current - ck * delta

            # Use single-layout noisy eval for SPSA (ZNE too expensive per iteration)
            # Use the best layout (lowest CES)
            best_layout = selected[0]
            pm = generate_preset_pass_manager(
                optimization_level=2, backend=backend, initial_layout=best_layout
            )

            bound_plus = qc.assign_parameters(theta_plus)
            bound_minus = qc.assign_parameters(theta_minus)
            t_plus = pm.run(bound_plus)
            t_minus = pm.run(bound_minus)
            h_mapped = H_test.apply_layout(t_plus.layout)

            est = BackendEstimatorV2(
                backend=backend,
                options={"seed_simulator": SEED + k * 2, "default_precision": PRECISION},
            )
            job_p = est.run([(t_plus, h_mapped)])
            job_m = est.run([(t_minus, H_test.apply_layout(t_minus.layout))])
            e_plus = float(job_p.result()[0].data.evs)
            e_minus = float(job_m.result()[0].data.evs)

            # SPSA gradient estimate
            grad = (e_plus - e_minus) / (2 * ck * delta)

            # Update
            theta_current = theta_current - ak * grad

            # Track best (every 10 iterations, do full ZNE eval)
            if (k + 1) % 25 == 0:
                bound_check = qc.assign_parameters(theta_current)
                check_data = []
                for li, init_layout in enumerate(selected):
                    pm_c = generate_preset_pass_manager(
                        optimization_level=2, backend=backend, initial_layout=init_layout
                    )
                    t_c = pm_c.run(bound_check)
                    h_m = H_test.apply_layout(t_c.layout)
                    est_c = BackendEstimatorV2(
                        backend=backend,
                        options={"seed_simulator": SEED + k + li, "default_precision": PRECISION},
                    )
                    job_c = est_c.run([(t_c, h_m)])
                    check_data.append(float(job_c.result()[0].data.evs))
                e_check, _ = linear_zne(ces_arr, np.array(check_data))
                if e_check < best_energy:
                    best_energy = e_check
                    best_theta = theta_current.copy()

        spsa_time = time.time() - t_spsa

        # Final ZNE evaluation with best theta
        bound_after = qc.assign_parameters(best_theta)
        layout_data_after = run_zne_deployment(
            bound_after, H_test, exact_test, backend, selected, N
        )
        e_after = np.array([d["energy"] for d in layout_data_after])
        e_zne_after, r2_after = linear_zne(ces_arr, e_after)
        de_zne_after = abs(e_zne_after - exact_test.ground_energy) / exact_test.gap

        spsa_gain = (de_zne_before - de_zne_after) / de_zne_before if de_zne_before > 0 else 0

        print("    SPSA time: %.1fs" % spsa_time)
        print("    Before SPSA: ZNE dE/gap=%.4f (R2=%.4f)" % (de_zne_before, r2_before))
        print("    After SPSA:  ZNE dE/gap=%.4f (R2=%.4f)" % (de_zne_after, r2_after))
        print("    SPSA gain: %.1f%%" % (spsa_gain * 100))
        if de_zne_after < de_zne_before:
            print("    -> SPSA HELPS")
        else:
            print("    -> SPSA HURTS (confirms V7 4B)")

        results.append(
            {
                "h_test": h_test,
                "de_noiseless": de_noiseless,
                "de_zne_before": de_zne_before,
                "r2_before": r2_before,
                "de_zne_after": de_zne_after,
                "r2_after": r2_after,
                "spsa_gain": spsa_gain,
                "spsa_time": spsa_time,
                "spsa_iters": SPSA_ITERS,
            }
        )

    print("")
    avg_gain = np.mean([r["spsa_gain"] for r in results])
    print("  SUMMARY: avg SPSA gain=%.1f%%" % (avg_gain * 100))
    if avg_gain > 0:
        print("  CONCLUSION: SPSA refinement HELPS after ZNE (unlike noiseless V7 4B)")
    else:
        print("  CONCLUSION: SPSA still HURTS even with ZNE cost (confirms V7 4B)")

    return {
        "exp": "8_zne_spsa",
        "results": results,
        "mpnn_mse": mse,
        "summary": {"avg_spsa_gain": avg_gain},
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_total = time.time()
    all_results = {}

    # Exp #1: p=2 N=10 correct layouts
    exp1 = run_exp1_p2_correct_layouts()
    all_results["exp1"] = exp1

    # Exp #3: CES filtering (uses exp1 data)
    exp3 = run_exp3_ces_filtering(exp1)
    all_results["exp3"] = exp3

    # Exp #4: p=1 + per-site
    exp4 = run_exp4_p1_persite()
    all_results["exp4"] = exp4

    # Exp #8: ZNE + SPSA
    exp8 = run_exp8_zne_plus_spsa()
    all_results["exp8"] = exp8

    # Save all
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / ("batch2_%s.json" % ts)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t_total
    print("")
    print("=" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print("  Total time: %.0fs (%.1f min)" % (elapsed, elapsed / 60))
    print("  Saved: %s" % out_path)
    print("=" * 70)
