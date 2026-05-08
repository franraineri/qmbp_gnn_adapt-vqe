#!/usr/bin/env python
"""
GNN-HVA v6.1 — Thesis Results Consolidation

Runs the definitive experiments for Chapter 4 (Results):
  - Table 4.2: N=6, 3 seeds × 3 h_test values = 9 runs
  - Table 4.3: N=10, 3 seeds × 2 h_test values = 6 runs

All with optimal config per system size. Reports mean ± std for each metric.

Usage:
    python scripts/run_thesis_results.py              # all 15 runs
    python scripts/run_thesis_results.py --table 4.2  # N=6 only (9 runs)
    python scripts/run_thesis_results.py --table 4.3  # N=10 only (6 runs)
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

RESULTS_DIR = _project_root / "scripts" / "notebook_results"
BINNACLE_DIR = _project_root / "documentation" / "binnacles"


def run_single(N: int, h_test: float, seed: int, mpnn_hidden: int, patience: int) -> dict:
    """Run a single pipeline execution and return metrics."""
    import torch
    from torch_geometric.data import Data

    from src.poc.v6.analysis_utils import WeightGradientAnalyzer
    from src.poc.v6.classical_solver import ClassicalSolver
    from src.poc.v6.config import VQEConfig
    from src.poc.v6.hamiltonian_builder import HamiltonianBuilder, make_lattice
    from src.poc.v6.hardware_deployer_v61 import HardwareDeployerV61
    from src.poc.v6.hva_builder import HVACircuitBuilder
    from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn
    from src.poc.v6.vqe_optimizer import VQEOptimizer

    np.random.seed(seed)
    torch.manual_seed(seed)

    J, p = 1.0, 2

    # Standard h-grid
    h_coarse = np.arange(0.0, 0.8, 0.1)
    h_dense = np.arange(0.8, 1.45, 0.05)
    h_coarse2 = np.arange(1.5, 2.05, 0.1)
    h_values = np.unique(np.concatenate([h_coarse, h_dense, h_coarse2]))

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    base_lattice = make_lattice("chain_1d", N, J=J, h=1.0)
    qc, _ = hva.create(N, p, base_lattice)

    # Phase 1
    exact_data = []
    for h in h_values:
        lat_h = make_lattice("chain_1d", N, J=J, h=h)
        H = builder.build(lat_h)
        exact_data.append(solver.solve(H, lat_h))

    # Phase 2
    vqe_config = VQEConfig(p_layers=p, n_restarts=5, maxiter=1000, enable_callbacks=False)
    opt = VQEOptimizer(vqe_config)
    vqe_results = opt.descending_sweep(h_values, qc, base_lattice, exact_data)
    fids = np.array([r.fidelity for r in vqe_results])

    # Phase 3
    dataset = build_graph_dataset(
        base_lattice,
        h_values,
        np.array([r.theta_opt for r in vqe_results]),
        np.array([d.ground_energy for d in exact_data]),
        fidelities=fids,
        fidelity_threshold=0.93,
    )
    model = MPNNPredictor(node_features=2, hidden_dim=mpnn_hidden, n_layers=3, output_dim=2 * p)
    train_result = train_mpnn(model, dataset, n_epochs=6000, lr=1e-3, patience=patience)

    # Phase 4
    lat_test = make_lattice("chain_1d", N, J=J, h=h_test)
    H_test = builder.build(lat_test)
    exact_test = solver.solve(H_test, lat_test)

    model.eval()
    edge_idx, coord = builder.build_graph_data(base_lattice)
    x_test = torch.tensor(
        np.stack([np.full(N, h_test), coord.astype(float)], axis=1),
        dtype=torch.float32,
    )
    test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))
    with torch.no_grad():
        theta_pred = model(test_graph).numpy().flatten()

    deployer = HardwareDeployerV61(mode="simulation")
    deploy_result = deployer.deploy_adapt_vqe(qc, H_test, theta_pred, lat_test, exact_test)

    # Gradient analysis
    analyzer = WeightGradientAnalyzer(model)
    grad_result = analyzer.analyze(dataset)

    checklist = deploy_result.metrics_checklist
    return {
        "N": N,
        "h_test": h_test,
        "seed": seed,
        "delta_e_over_gap": deploy_result.delta_e_over_gap,
        "delta_e": deploy_result.delta_e,
        "mag_x_error": deploy_result.mag_x_error,
        "corr_zz_error": deploy_result.corr_zz_error,
        "fidelity": deploy_result.fidelity,
        "adapt_iterations": deploy_result.adapt_iterations,
        "phase_label": deploy_result.phase_label,
        "checklist_pass": sum(checklist.values()),
        "checklist_total": len(checklist),
        "mpnn_mse": train_result["final_mse"],
        "training_points": len(dataset),
        "avg_vqe_fidelity": float(np.mean(fids)),
        "gradient_peaks": len(grad_result.peak_h_values),
        "critical_region": grad_result.critical_region_detected,
    }


def compute_stats(results: list[dict], key: str) -> tuple[float, float]:
    """Compute mean and std for a metric across runs."""
    values = [r[key] for r in results if r.get(key) is not None]
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Thesis Results Consolidation")
    parser.add_argument("--table", choices=["4.2", "4.3", "all"], default="all")
    parser.add_argument("--binnacle", action="store_true", help="Append to binnacle")
    args = parser.parse_args()

    seeds = [42, 43, 44]

    # Define experiment matrix
    experiments = []
    if args.table in ("4.2", "all"):
        for h_test in [1.25, 1.4, 1.5]:
            for seed in seeds:
                experiments.append(
                    {"N": 6, "h_test": h_test, "seed": seed, "mpnn_hidden": 64, "patience": 300}
                )
    if args.table in ("4.3", "all"):
        for h_test in [1.4, 1.5]:
            for seed in seeds:
                experiments.append(
                    {"N": 10, "h_test": h_test, "seed": seed, "mpnn_hidden": 128, "patience": 500}
                )

    print("=" * 70)
    print("  GNN-HVA v6.1 — Thesis Results Consolidation")
    print(
        f"  {len(experiments)} experiments ({len([e for e in experiments if e['N'] == 6])} N=6, "
        f"{len([e for e in experiments if e['N'] == 10])} N=10)"
    )
    print("=" * 70)

    all_results = []
    t_total = time.time()

    for i, exp in enumerate(experiments, 1):
        print(
            f"\n  [{i}/{len(experiments)}] N={exp['N']}, h_test={exp['h_test']}, seed={exp['seed']}...",
            end=" ",
            flush=True,
        )
        t0 = time.time()
        try:
            result = run_single(**exp)
            all_results.append(result)
            de_tag = (
                "✅"
                if result["delta_e_over_gap"] < 0.05
                else "⚠️"
                if result["delta_e_over_gap"] < 0.10
                else "❌"
            )
            print(
                f"ΔE/gap={result['delta_e_over_gap']:.4f} {de_tag} "
                f"CL={result['checklist_pass']}/{result['checklist_total']} "
                f"({time.time() - t0:.0f}s)"
            )
        except Exception as e:
            print(f"FAILED: {e}")
            all_results.append(
                {"N": exp["N"], "h_test": exp["h_test"], "seed": exp["seed"], "error": str(e)}
            )

    total_time = time.time() - t_total

    # ── Generate Tables ──
    print(f"\n\n{'=' * 70}")
    print("  THESIS RESULTS — TABLE 4.2 (N=6)")
    print(f"{'=' * 70}")

    n6_results = [r for r in all_results if r.get("N") == 6 and "error" not in r]
    if n6_results:
        # Group by h_test
        by_h = defaultdict(list)
        for r in n6_results:
            by_h[r["h_test"]].append(r)

        print(
            f"\n  {'h_test':<8} {'ΔE/gap':<16} {'⟨X⟩ err':<16} {'⟨ZZ⟩ err':<16} "
            f"{'Fidelity':<16} {'Checklist':<12} {'MSE':<14}"
        )
        print(f"  {'─' * 98}")
        for h_test in sorted(by_h.keys()):
            runs = by_h[h_test]
            de_m, de_s = compute_stats(runs, "delta_e_over_gap")
            mx_m, mx_s = compute_stats(runs, "mag_x_error")
            zz_m, zz_s = compute_stats(runs, "corr_zz_error")
            fi_m, fi_s = compute_stats(runs, "fidelity")
            cl_m, cl_s = compute_stats(runs, "checklist_pass")
            ms_m, ms_s = compute_stats(runs, "mpnn_mse")
            de_tag = "✅" if de_m < 0.05 else "⚠️" if de_m < 0.10 else "❌"
            print(
                f"  {h_test:<8} {de_m:.4f}±{de_s:.4f} {de_tag} "
                f"{mx_m:.2e}±{mx_s:.2e}  {zz_m:.2e}±{zz_s:.2e}  "
                f"{fi_m:.4f}±{fi_s:.4f}  {cl_m:.1f}±{cl_s:.1f}/{runs[0]['checklist_total']}  "
                f"{ms_m:.2e}±{ms_s:.2e}"
            )

    print(f"\n\n{'=' * 70}")
    print("  THESIS RESULTS — TABLE 4.3 (N=10)")
    print(f"{'=' * 70}")

    n10_results = [r for r in all_results if r.get("N") == 10 and "error" not in r]
    if n10_results:
        by_h = defaultdict(list)
        for r in n10_results:
            by_h[r["h_test"]].append(r)

        print(
            f"\n  {'h_test':<8} {'ΔE/gap':<16} {'⟨X⟩ err':<16} {'⟨ZZ⟩ err':<16} "
            f"{'Fidelity':<16} {'Checklist':<12} {'MSE':<14}"
        )
        print(f"  {'─' * 98}")
        for h_test in sorted(by_h.keys()):
            runs = by_h[h_test]
            de_m, de_s = compute_stats(runs, "delta_e_over_gap")
            mx_m, mx_s = compute_stats(runs, "mag_x_error")
            zz_m, zz_s = compute_stats(runs, "corr_zz_error")
            fi_m, fi_s = compute_stats(runs, "fidelity")
            cl_m, cl_s = compute_stats(runs, "checklist_pass")
            ms_m, ms_s = compute_stats(runs, "mpnn_mse")
            de_tag = "✅" if de_m < 0.05 else "⚠️" if de_m < 0.10 else "❌"
            print(
                f"  {h_test:<8} {de_m:.4f}±{de_s:.4f} {de_tag} "
                f"{mx_m:.2e}±{mx_s:.2e}  {zz_m:.2e}±{zz_s:.2e}  "
                f"{fi_m:.4f}±{fi_s:.4f}  {cl_m:.1f}±{cl_s:.1f}/{runs[0]['checklist_total']}  "
                f"{ms_m:.2e}±{ms_s:.2e}"
            )

    # ── Per-run detail ──
    print(f"\n\n{'=' * 70}")
    print("  PER-RUN DETAIL")
    print(f"{'=' * 70}")
    print(
        f"  {'N':<4} {'h_test':<8} {'seed':<6} {'ΔE/gap':<10} {'⟨X⟩ err':<10} "
        f"{'⟨ZZ⟩ err':<10} {'Fid':<8} {'CL':<6} {'MSE':<10} {'Peaks'}"
    )
    print(f"  {'─' * 90}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['N']:<4} {r['h_test']:<8} {r['seed']:<6} FAILED: {r['error'][:40]}")
            continue
        de_tag = (
            "✅" if r["delta_e_over_gap"] < 0.05 else "⚠️" if r["delta_e_over_gap"] < 0.10 else "❌"
        )
        fid_str = f"{r['fidelity']:.4f}" if r.get("fidelity") is not None else "N/A   "
        print(
            f"  {r['N']:<4} {r['h_test']:<8} {r['seed']:<6} "
            f"{r['delta_e_over_gap']:.4f}{de_tag}  "
            f"{r['mag_x_error']:.2e}  {r['corr_zz_error']:.2e}  "
            f"{fid_str}  {r['checklist_pass']}/{r['checklist_total']}  "
            f"{r['mpnn_mse']:.2e}  {r['gradient_peaks']}"
        )

    print(f"\n  Total time: {total_time:.0f}s")

    # ── Save JSON ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = hashlib.sha1(f"thesis:{ts}".encode()).hexdigest()[:8]
    summary_path = RESULTS_DIR / f"thesis_results_{ts}_{run_id}.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "runner": "run_thesis_results.py",
        "total_experiments": len(experiments),
        "total_elapsed_s": round(total_time, 1),
        "results": all_results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Results saved: {summary_path}")

    # ── Binnacle ──
    if args.binnacle:
        BINNACLE_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # N=6 binnacle
        if n6_results:
            binnacle_path = BINNACLE_DIR / "binnacle-N6.md"
            lines = [f"\n---\n\n## {now} — Thesis Table 4.2 (N=6, 3 seeds × 3 h_test)\n\n"]
            lines.append(
                "| h_test | Seed | ΔE/gap | ⟨X⟩ err | ⟨ZZ⟩ err | Fidelity | Checklist | MSE |\n"
            )
            lines.append(
                "|--------|------|--------|---------|----------|----------|-----------|-----|\n"
            )
            for r in sorted(n6_results, key=lambda x: (x["h_test"], x["seed"])):
                fid_str = f"{r['fidelity']:.4f}" if r.get("fidelity") is not None else "N/A"
                lines.append(
                    f"| {r['h_test']} | {r['seed']} | {r['delta_e_over_gap']:.4f} | "
                    f"{r['mag_x_error']:.2e} | {r['corr_zz_error']:.2e} | "
                    f"{fid_str} | {r['checklist_pass']}/{r['checklist_total']} | "
                    f"{r['mpnn_mse']:.2e} |\n"
                )
            lines.append("\n**Aggregated (mean ± std):**\n\n")
            lines.append("| h_test | ΔE/gap | ⟨X⟩ err | ⟨ZZ⟩ err | Fidelity | Checklist |\n")
            lines.append("|--------|--------|---------|----------|----------|----------|\n")
            by_h = defaultdict(list)
            for r in n6_results:
                by_h[r["h_test"]].append(r)
            for h_test in sorted(by_h.keys()):
                runs = by_h[h_test]
                de_m, de_s = compute_stats(runs, "delta_e_over_gap")
                mx_m, mx_s = compute_stats(runs, "mag_x_error")
                zz_m, zz_s = compute_stats(runs, "corr_zz_error")
                fi_m, fi_s = compute_stats(runs, "fidelity")
                cl_m, cl_s = compute_stats(runs, "checklist_pass")
                lines.append(
                    f"| {h_test} | {de_m:.4f}±{de_s:.4f} | {mx_m:.2e}±{mx_s:.2e} | "
                    f"{zz_m:.2e}±{zz_s:.2e} | {fi_m:.4f}±{fi_s:.4f} | "
                    f"{cl_m:.1f}±{cl_s:.1f} |\n"
                )
            lines.append("\n")
            with open(binnacle_path, "a") as f:
                f.writelines(lines)
            print(f"  Binnacle (N=6): {binnacle_path}")

        # N=10 binnacle
        if n10_results:
            binnacle_path = BINNACLE_DIR / "binnacle-N10.md"
            lines = [f"\n---\n\n## {now} — Thesis Table 4.3 (N=10, 3 seeds × 2 h_test)\n\n"]
            lines.append(
                "| h_test | Seed | ΔE/gap | ⟨X⟩ err | ⟨ZZ⟩ err | Fidelity | Checklist | MSE |\n"
            )
            lines.append(
                "|--------|------|--------|---------|----------|----------|-----------|-----|\n"
            )
            for r in sorted(n10_results, key=lambda x: (x["h_test"], x["seed"])):
                fid_str = f"{r['fidelity']:.4f}" if r.get("fidelity") is not None else "N/A"
                lines.append(
                    f"| {r['h_test']} | {r['seed']} | {r['delta_e_over_gap']:.4f} | "
                    f"{r['mag_x_error']:.2e} | {r['corr_zz_error']:.2e} | "
                    f"{fid_str} | {r['checklist_pass']}/{r['checklist_total']} | "
                    f"{r['mpnn_mse']:.2e} |\n"
                )
            lines.append("\n**Aggregated (mean ± std):**\n\n")
            lines.append("| h_test | ΔE/gap | ⟨X⟩ err | ⟨ZZ⟩ err | Fidelity | Checklist |\n")
            lines.append("|--------|--------|---------|----------|----------|----------|\n")
            by_h = defaultdict(list)
            for r in n10_results:
                by_h[r["h_test"]].append(r)
            for h_test in sorted(by_h.keys()):
                runs = by_h[h_test]
                de_m, de_s = compute_stats(runs, "delta_e_over_gap")
                mx_m, mx_s = compute_stats(runs, "mag_x_error")
                zz_m, zz_s = compute_stats(runs, "corr_zz_error")
                fi_m, fi_s = compute_stats(runs, "fidelity")
                cl_m, cl_s = compute_stats(runs, "checklist_pass")
                lines.append(
                    f"| {h_test} | {de_m:.4f}±{de_s:.4f} | {mx_m:.2e}±{mx_s:.2e} | "
                    f"{zz_m:.2e}±{zz_s:.2e} | {fi_m:.4f}±{fi_s:.4f} | "
                    f"{cl_m:.1f}±{cl_s:.1f} |\n"
                )
            lines.append("\n")
            with open(binnacle_path, "a") as f:
                f.writelines(lines)
            print(f"  Binnacle (N=10): {binnacle_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
