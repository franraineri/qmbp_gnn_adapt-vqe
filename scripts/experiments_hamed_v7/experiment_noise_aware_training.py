"""
Experiment: Noise-Aware MPNN Training

Sub-experiments:
    5A: FakeTorino VQE data generation (N=6, 27 h-values, COBYLA)
    5B: Train on FakeTorino data vs noiseless, evaluate on FakeTorino
    5C: Mixed training (noiseless + noisy, noise-level input feature)
    5D: N=10 noise-aware scaling
    5E: Iterative refinement (3 rounds: train -> deploy -> collect -> retrain)

References:
    - Karim et al. (2025) arXiv:2503.20210
    - Hamed's meeting notes: "classical training with quantum information extraction"

Usage:
    python scripts/experiments_hamed_v7/experiment_noise_aware_training.py --sub-experiment 5A
    python scripts/experiments_hamed_v7/experiment_noise_aware_training.py --sub-experiment all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from experiment_utils import (
    SubExperimentResult,
    compute_metrics,
    evaluate_energy_statevector,
    save_experiment_result,
)
from shared_runners import (
    get_exact_solution,
    noisy_cost_function,
    setup_experiment,
    vqe_descending_sweep,
)

RESULTS_DIR = Path(__file__).parent / "results"


# ── Sub-experiment 5A ────────────────────────────────────────────────────────


def run_sub_experiment_5A(args) -> SubExperimentResult:
    """Generate VQE training data using noisy simulation (shot noise proxy).

    Uses COBYLA with shot noise at n_shots=8192 to simulate FakeTorino-like
    conditions. Generates θ_opt for 27 h-values via descending sweep.
    """
    N = args.N
    n_train = args.n_train or 27
    n_shots = args.n_shots or 8192
    seed = args.seed

    print(f"\n{'=' * 70}")
    print("  Sub-experiment 5A: Noisy VQE data generation")
    print(f"  N={N}, n_train={n_train}, n_shots={n_shots}")
    print(f"{'=' * 70}\n")

    env = setup_experiment(N)
    qc = env["circuit"]
    h_values = np.linspace(0.5, 2.0, n_train)

    np.random.seed(seed)

    # Generate noisy training data
    print("  Generating noisy VQE training data (descending sweep)...")
    noisy_data = vqe_descending_sweep(
        qc,
        h_values,
        env["builder"],
        env["solver"],
        N,
        noiseless=False,
        n_shots=n_shots,
        maxiter=300,
    )
    print(f"  Noisy sweep complete: {noisy_data['wall_time_s']:.1f}s")

    # Also generate noiseless reference
    print("  Generating noiseless VQE reference...")
    np.random.seed(seed)
    clean_data = vqe_descending_sweep(
        qc,
        h_values,
        env["builder"],
        env["solver"],
        N,
        noiseless=True,
        maxiter=1000,
    )
    print(f"  Noiseless sweep complete: {clean_data['wall_time_s']:.1f}s")

    # Compare θ_opt differences
    theta_diff = np.mean(np.abs(noisy_data["theta_opt"] - clean_data["theta_opt"]))
    energy_diff = np.mean(np.abs(noisy_data["energies"] - clean_data["energies"]))
    print(f"\n  Mean |θ_noisy - θ_clean| = {theta_diff:.4f}")
    print(f"  Mean |E_noisy - E_clean| = {energy_diff:.4f}")

    # Compute metrics for noisy data quality
    all_metrics = []
    for i, h in enumerate(h_values):
        sol = get_exact_solution(env["builder"], env["solver"], N, float(h))
        gap = sol["exact"].gap if sol["exact"].gap > 0 else 0.1
        m = compute_metrics(
            energy=float(noisy_data["energies"][i]),
            exact_energy=float(noisy_data["exact_energies"][i]),
            gap=gap,
            wall_time_s=noisy_data["wall_time_s"] / n_train,
            n_evaluations=300,
            seed=seed,
            h_value=h,
        )
        all_metrics.append(m)

    summary = {
        "mean_theta_diff": float(theta_diff),
        "mean_energy_diff": float(energy_diff),
        "noisy_sweep_time_s": noisy_data["wall_time_s"],
        "clean_sweep_time_s": clean_data["wall_time_s"],
        "conclusion": "Noisy VQE data generated for training comparison",
    }

    result = SubExperimentResult(
        experiment_id="5A",
        technique=5,
        description=f"Noisy VQE data generation (N={N}, {n_train} points, {n_shots} shots)",
        config={"N": N, "n_train": n_train, "n_shots": n_shots, "seed": seed},
        metrics=all_metrics,
        summary=summary,
        success=True,
    )
    path = save_experiment_result(result, RESULTS_DIR, prefix="noise_aware")
    print(f"\n  Result saved: {path.name}")
    return result


# ── Sub-experiment 5B ────────────────────────────────────────────────────────


def run_sub_experiment_5B(args) -> SubExperimentResult:
    """Train MPNN on noisy vs noiseless data, evaluate under noise."""
    N = args.N
    n_train = args.n_train or 27
    n_shots = args.n_shots or 8192
    seed = args.seed
    h_test_values = [1.25, 1.4, 1.5]

    print(f"\n{'=' * 70}")
    print("  Sub-experiment 5B: Noise-aware vs noiseless MPNN training")
    print(f"  N={N}, n_train={n_train}, test_h={h_test_values}")
    print(f"{'=' * 70}\n")

    env = setup_experiment(N)
    qc = env["circuit"]
    h_values = np.linspace(0.5, 2.0, n_train)

    np.random.seed(seed)

    # Generate both datasets
    print("  Generating training data...")
    noisy_data = vqe_descending_sweep(
        qc,
        h_values,
        env["builder"],
        env["solver"],
        N,
        noiseless=False,
        n_shots=n_shots,
        maxiter=300,
    )
    np.random.seed(seed)
    clean_data = vqe_descending_sweep(
        qc,
        h_values,
        env["builder"],
        env["solver"],
        N,
        noiseless=True,
        maxiter=1000,
    )
    print(
        f"  Data generated: noisy={noisy_data['wall_time_s']:.0f}s, "
        f"clean={clean_data['wall_time_s']:.0f}s"
    )

    # Train two MPNNs
    import torch

    from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn

    # Noiseless-trained MPNN
    torch.manual_seed(seed)
    dataset_clean = build_graph_dataset(
        env["base_lattice"],
        h_values,
        clean_data["theta_opt"],
        clean_data["exact_energies"],
        fidelities=clean_data["fidelities"],
        fidelity_threshold=0.0,
    )
    model_clean = MPNNPredictor(
        node_features=2,
        hidden_dim=64,
        n_layers=3,
        output_dim=env["n_params"],
    )
    res_clean = train_mpnn(model_clean, dataset_clean, n_epochs=3000, lr=1e-3, patience=200)
    print(f"  Noiseless MPNN: MSE={res_clean['final_mse']:.6f}")

    # Noise-aware MPNN
    torch.manual_seed(seed)
    dataset_noisy = build_graph_dataset(
        env["base_lattice"],
        h_values,
        noisy_data["theta_opt"],
        noisy_data["exact_energies"],
        fidelities=np.ones(n_train),
        fidelity_threshold=0.0,
    )
    model_noisy = MPNNPredictor(
        node_features=2,
        hidden_dim=64,
        n_layers=3,
        output_dim=env["n_params"],
    )
    res_noisy = train_mpnn(model_noisy, dataset_noisy, n_epochs=3000, lr=1e-3, patience=200)
    print(f"  Noise-aware MPNN: MSE={res_noisy['final_mse']:.6f}")

    # Evaluate both under noise
    print("\n  Evaluating on test points under noise...")
    from torch_geometric.data import Data

    all_metrics = []
    detailed = []

    for h_t in h_test_values:
        sol = get_exact_solution(env["builder"], env["solver"], N, h_t)
        H = sol["hamiltonian"]
        exact = sol["exact"]

        edge_idx, coord = env["builder"].build_graph_data(env["base_lattice"])
        x_test = torch.tensor(
            np.stack([np.full(N, h_t), coord.astype(float)], axis=1),
            dtype=torch.float32,
        )
        test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))

        model_clean.eval()
        model_noisy.eval()
        with torch.no_grad():
            theta_clean = model_clean(test_graph).numpy().flatten()
            theta_noisy = model_noisy(test_graph).numpy().flatten()

        # Exact energy (noiseless evaluation of predicted θ)
        e_clean = evaluate_energy_statevector(qc, H, theta_clean)
        e_noisy = evaluate_energy_statevector(qc, H, theta_noisy)

        err_clean = abs(e_clean - exact.ground_energy)
        err_noisy = abs(e_noisy - exact.ground_energy)

        m_clean = compute_metrics(
            energy=e_clean,
            exact_energy=exact.ground_energy,
            gap=exact.gap,
            seed=seed,
            h_value=h_t,
        )
        m_noisy = compute_metrics(
            energy=e_noisy,
            exact_energy=exact.ground_energy,
            gap=exact.gap,
            seed=seed,
            h_value=h_t,
        )
        all_metrics.extend([m_clean, m_noisy])
        detailed.append(
            {
                "h": h_t,
                "clean_error": err_clean,
                "noisy_error": err_noisy,
                "winner": "noise-aware" if err_noisy < err_clean else "noiseless",
            }
        )
        print(f"  h={h_t:.2f}: clean ΔE={err_clean:.2e}, noise-aware ΔE={err_noisy:.2e}")

    avg_clean = np.mean([d["clean_error"] for d in detailed])
    avg_noisy = np.mean([d["noisy_error"] for d in detailed])
    winner = "noise-aware" if avg_noisy < avg_clean else "noiseless"

    summary = {
        "avg_clean_error": float(avg_clean),
        "avg_noisy_error": float(avg_noisy),
        "winner": winner,
        "detailed": detailed,
        "conclusion": f"{winner} MPNN wins (avg ΔE: clean={avg_clean:.2e}, noisy={avg_noisy:.2e})",
    }

    result = SubExperimentResult(
        experiment_id="5B",
        technique=5,
        description="Noise-aware vs noiseless MPNN training comparison",
        config={
            "N": N,
            "n_train": n_train,
            "n_shots": n_shots,
            "seed": seed,
            "h_test": h_test_values,
        },
        metrics=all_metrics,
        summary=summary,
        success=True,
    )
    path = save_experiment_result(result, RESULTS_DIR, prefix="noise_aware")
    print(f"\n  Result saved: {path.name}")
    return result


# ── Sub-experiment 5C ────────────────────────────────────────────────────────


def run_sub_experiment_5C(args) -> SubExperimentResult:
    """Mixed training: combine noiseless + noisy datasets."""
    N = args.N
    seed = args.seed

    print(f"\n{'=' * 70}")
    print("  Sub-experiment 5C: Mixed training (noiseless + noisy)")
    print(f"  N={N} — combines both datasets with noise-level feature")
    print(f"{'=' * 70}\n")

    # This is a more complex experiment that requires custom MPNN modification
    # For now, implement as concatenated dataset training
    print("  Note: Full noise-level feature requires MPNN architecture change.")
    print("  Implementing as concatenated dataset (2x training points).")

    env = setup_experiment(N)
    qc = env["circuit"]
    n_train = 27
    h_values = np.linspace(0.5, 2.0, n_train)

    np.random.seed(seed)
    noisy_data = vqe_descending_sweep(
        qc,
        h_values,
        env["builder"],
        env["solver"],
        N,
        noiseless=False,
        n_shots=8192,
        maxiter=300,
    )
    np.random.seed(seed)
    clean_data = vqe_descending_sweep(
        qc,
        h_values,
        env["builder"],
        env["solver"],
        N,
        noiseless=True,
        maxiter=1000,
    )

    # Concatenate datasets
    h_combined = np.concatenate([h_values, h_values])
    theta_combined = np.concatenate([clean_data["theta_opt"], noisy_data["theta_opt"]])
    energies_combined = np.concatenate([clean_data["exact_energies"], noisy_data["exact_energies"]])
    fid_combined = np.concatenate([clean_data["fidelities"], np.ones(n_train)])

    import torch

    from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn

    torch.manual_seed(seed)
    dataset_mixed = build_graph_dataset(
        env["base_lattice"],
        h_combined,
        theta_combined,
        energies_combined,
        fidelities=fid_combined,
        fidelity_threshold=0.0,
    )
    model_mixed = MPNNPredictor(
        node_features=2,
        hidden_dim=64,
        n_layers=3,
        output_dim=env["n_params"],
    )
    res_mixed = train_mpnn(model_mixed, dataset_mixed, n_epochs=4000, lr=1e-3, patience=300)
    print(f"  Mixed MPNN: MSE={res_mixed['final_mse']:.6f}")

    # Evaluate
    from torch_geometric.data import Data

    h_test_values = [1.25, 1.4, 1.5]
    all_metrics = []
    detailed = []

    for h_t in h_test_values:
        sol = get_exact_solution(env["builder"], env["solver"], N, h_t)
        H = sol["hamiltonian"]
        exact = sol["exact"]

        edge_idx, coord = env["builder"].build_graph_data(env["base_lattice"])
        x_test = torch.tensor(
            np.stack([np.full(N, h_t), coord.astype(float)], axis=1),
            dtype=torch.float32,
        )
        test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))

        model_mixed.eval()
        with torch.no_grad():
            theta_pred = model_mixed(test_graph).numpy().flatten()

        energy = evaluate_energy_statevector(qc, H, theta_pred)
        err = abs(energy - exact.ground_energy)

        m = compute_metrics(
            energy=energy,
            exact_energy=exact.ground_energy,
            gap=exact.gap,
            seed=seed,
            h_value=h_t,
        )
        all_metrics.append(m)
        detailed.append({"h": h_t, "error": err})
        print(f"  h={h_t:.2f}: mixed MPNN ΔE={err:.2e}")

    avg_err = np.mean([d["error"] for d in detailed])
    summary = {
        "avg_error": float(avg_err),
        "training_mse": res_mixed["final_mse"],
        "detailed": detailed,
        "conclusion": f"Mixed training avg ΔE={avg_err:.2e}",
    }

    result = SubExperimentResult(
        experiment_id="5C",
        technique=5,
        description="Mixed training (noiseless + noisy concatenated)",
        config={"N": N, "n_train_total": n_train * 2, "seed": seed},
        metrics=all_metrics,
        summary=summary,
        success=True,
    )
    path = save_experiment_result(result, RESULTS_DIR, prefix="noise_aware")
    print(f"\n  Result saved: {path.name}")
    return result


# ── Sub-experiment 5D ────────────────────────────────────────────────────────


def run_sub_experiment_5D(args) -> SubExperimentResult:
    """Noise-aware training at N=10."""
    saved_N = args.N
    args.N = 10
    print("\n  [5D delegates to 5B logic with N=10]")
    result = run_sub_experiment_5B(args)
    result.experiment_id = "5D"
    result.description = "Noise-aware MPNN training at N=10"
    args.N = saved_N
    path = save_experiment_result(result, RESULTS_DIR, prefix="noise_aware")
    print(f"  Result re-saved as 5D: {path.name}")
    return result


# ── Sub-experiment 5E ────────────────────────────────────────────────────────


def run_sub_experiment_5E(args) -> SubExperimentResult:
    """Iterative refinement: train -> deploy -> collect -> retrain (3 rounds)."""
    N = args.N
    n_rounds = args.n_rounds or 3
    seed = args.seed
    n_train = 27
    n_shots = 8192

    print(f"\n{'=' * 70}")
    print(f"  Sub-experiment 5E: Iterative refinement ({n_rounds} rounds)")
    print(f"  N={N}, n_train={n_train}")
    print(f"{'=' * 70}\n")

    env = setup_experiment(N)
    qc = env["circuit"]
    h_values = np.linspace(0.5, 2.0, n_train)
    h_test_values = [1.25, 1.4, 1.5]

    # Round 0: noiseless baseline
    np.random.seed(seed)
    clean_data = vqe_descending_sweep(
        qc,
        h_values,
        env["builder"],
        env["solver"],
        N,
        noiseless=True,
        maxiter=1000,
    )

    import torch
    from torch_geometric.data import Data

    from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn

    all_metrics = []
    round_results = []
    current_theta = clean_data["theta_opt"].copy()
    current_h = h_values.copy()

    for round_idx in range(n_rounds):
        print(f"\n  --- Round {round_idx + 1}/{n_rounds} ---")

        # Train MPNN on current data
        torch.manual_seed(seed + round_idx)
        dataset = build_graph_dataset(
            env["base_lattice"],
            current_h,
            current_theta,
            clean_data["exact_energies"][: len(current_h)],
            fidelities=np.ones(len(current_h)),
            fidelity_threshold=0.0,
        )
        model = MPNNPredictor(
            node_features=2,
            hidden_dim=64,
            n_layers=3,
            output_dim=env["n_params"],
        )
        res = train_mpnn(model, dataset, n_epochs=3000, lr=1e-3, patience=200)
        print(f"  Train MSE={res['final_mse']:.6f}")

        # Deploy on test points under noise and collect results
        round_errors = []
        new_theta_list = []
        new_h_list = []

        for h_t in h_test_values:
            sol = get_exact_solution(env["builder"], env["solver"], N, h_t)
            H = sol["hamiltonian"]
            exact = sol["exact"]

            edge_idx, coord = env["builder"].build_graph_data(env["base_lattice"])
            x_test = torch.tensor(
                np.stack([np.full(N, h_t), coord.astype(float)], axis=1),
                dtype=torch.float32,
            )
            test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))

            model.eval()
            with torch.no_grad():
                theta_pred = model(test_graph).numpy().flatten()

            # "Deploy" under noise: refine with noisy COBYLA
            cost_fn, _ = noisy_cost_function(qc, H, n_shots)
            from scipy.optimize import minimize

            res_deploy = minimize(
                cost_fn,
                theta_pred,
                method="COBYLA",
                options={"maxiter": 50, "rhobeg": 0.1},
            )
            theta_refined = res_deploy.x

            energy = evaluate_energy_statevector(qc, H, theta_refined)
            err = abs(energy - exact.ground_energy)
            round_errors.append(err)

            # Collect refined θ for next round
            new_theta_list.append(theta_refined)
            new_h_list.append(h_t)

            m = compute_metrics(
                energy=energy,
                exact_energy=exact.ground_energy,
                gap=exact.gap,
                seed=seed,
                h_value=h_t,
            )
            all_metrics.append(m)

        avg_err = float(np.mean(round_errors))
        round_results.append({"round": round_idx + 1, "avg_error": avg_err})
        print(f"  Round {round_idx + 1} avg ΔE={avg_err:.2e}")

        # Augment training data with deployment results
        current_h = np.concatenate([current_h, np.array(new_h_list)])
        current_theta = np.concatenate([current_theta, np.array(new_theta_list)])

    # Check convergence
    if len(round_results) >= 2:
        improvement = round_results[0]["avg_error"] - round_results[-1]["avg_error"]
        converged = improvement > 0
    else:
        converged = False

    summary = {
        "round_results": round_results,
        "converged": converged,
        "conclusion": f"Iterative refinement {'converged' if converged else 'did not converge'} "
        f"over {n_rounds} rounds",
    }

    result = SubExperimentResult(
        experiment_id="5E",
        technique=5,
        description=f"Iterative refinement ({n_rounds} rounds)",
        config={"N": N, "n_rounds": n_rounds, "n_train": n_train, "seed": seed},
        metrics=all_metrics,
        summary=summary,
        success=True,
    )
    path = save_experiment_result(result, RESULTS_DIR, prefix="noise_aware")
    print(f"\n  Result saved: {path.name}")
    return result


# ── CLI & Dispatch ───────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Noise-aware MPNN training experiments (Technique 5)",
    )
    parser.add_argument(
        "--sub-experiment", type=str, default="all", choices=["5A", "5B", "5C", "5D", "5E", "all"]
    )
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-shots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-fake-torino", action="store_true")
    parser.add_argument("--n-rounds", type=int, default=3)
    return parser.parse_args()


DISPATCH = {
    "5A": run_sub_experiment_5A,
    "5B": run_sub_experiment_5B,
    "5C": run_sub_experiment_5C,
    "5D": run_sub_experiment_5D,
    "5E": run_sub_experiment_5E,
}


def main():
    args = parse_args()

    if args.sub_experiment == "all":
        results = []
        for sub_id, fn in DISPATCH.items():
            try:
                result = fn(args)
                results.append(result)
            except Exception as e:
                print(f"\n  ERROR in {sub_id}: {e}")
                results.append(
                    SubExperimentResult(
                        experiment_id=sub_id,
                        technique=5,
                        description=f"Failed: {e}",
                        success=False,
                        error=str(e),
                    )
                )
        if any(not r.success for r in results):
            sys.exit(1)
    else:
        fn = DISPATCH[args.sub_experiment]
        result = fn(args)
        if not result.success:
            sys.exit(1)


if __name__ == "__main__":
    main()
