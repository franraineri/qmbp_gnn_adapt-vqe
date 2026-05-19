"""
Experiment: QRC Features → NN Warm-Start (Hamed's Suggestion)

Hypothesis:
    Using a quantum reservoir to extract features, then feeding those features
    into a classical neural network to predict VQE parameters, could provide
    an alternative warm-start strategy.

Sub-experiments:
    2A: 4 reservoir designs comparison (random, near-identity, entangling, optimized)
    2B: N=10 QRC vs MPNN comparison (27 training points, production MPNN config)
    2C: Data efficiency (vary training set size [8, 12, 16, 20, 27])
    2D: Hybrid QRC+MPNN (concatenated features → shared MLP)

References:
    - Kutvonen et al. (2020) Nature Sci Rep
    - arXiv:2510.00171 — QRC with Jaynes-Cummings model

Usage:
    python scripts/experiments_hamed_v7/experiment_qrc_warmstart.py --sub-experiment 2A
    python scripts/experiments_hamed_v7/experiment_qrc_warmstart.py --sub-experiment all
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

from src.poc.v6 import (
    HVACircuitBuilder,
    VQEOptimizer,
    make_lattice,
)
from src.poc.v6.config import VQEConfig

RESULTS_DIR = Path(__file__).parent / "results"

from shared_runners import (
    clear_exact_cache,
    get_exact_solution,
    setup_experiment,
    train_mpnn_with_best_state,
)

# ── QRC Predictor ────────────────────────────────────────────────────────


class QRCPredictor:
    """QRC-based warm-start predictor: reservoir features → MLP → θ_pred.

    Supports multiple reservoir designs via `reservoir_type` parameter.
    """

    def __init__(
        self, n_qubits: int, p_layers: int = 2, seed: int = 42, reservoir_type: str = "random"
    ):
        self.n_qubits = n_qubits
        self.p_layers = p_layers
        self.seed = seed
        self.reservoir_type = reservoir_type
        self._reservoir_circuit = None
        self._mlp = None
        self._feature_dim = n_qubits + (n_qubits - 1)  # ⟨Xi⟩ + ⟨ZiZj⟩

    def build_reservoir(self, optimized_params=None):
        """Create a fixed HVA reservoir with parameters based on reservoir_type."""

        hva = HVACircuitBuilder()
        base_lattice = make_lattice("chain_1d", self.n_qubits, J=1.0, h=1.0)
        qc, theta = hva.create(self.n_qubits, self.p_layers, base_lattice)

        rng = np.random.RandomState(self.seed)
        n_params = len(theta)

        if self.reservoir_type == "random":
            params = rng.uniform(-np.pi, np.pi, n_params)
        elif self.reservoir_type == "near_identity":
            params = rng.uniform(-0.1, 0.1, n_params)
        elif self.reservoir_type == "entangling":
            # Structured: θ_ZZ = π/4, θ_X = π/4
            params = np.full(n_params, np.pi / 4)
        elif self.reservoir_type == "optimized":
            if optimized_params is not None:
                params = optimized_params
            else:
                params = rng.uniform(-np.pi, np.pi, n_params)
        else:
            raise ValueError(f"Unknown reservoir_type: {self.reservoir_type}")

        self._reservoir_circuit = qc.assign_parameters(params)

    def extract_features(self, h_value: float) -> np.ndarray:
        """Encode h via Rx gates and extract observable features."""
        from qiskit.quantum_info import SparsePauliOp, Statevector

        encoded = self._reservoir_circuit.copy()
        for i in range(self.n_qubits):
            encoded.rx(h_value, i)

        sv = Statevector(encoded)
        features = []

        # Per-site ⟨Xi⟩
        for i in range(self.n_qubits):
            label = "I" * (self.n_qubits - 1 - i) + "X" + "I" * i
            op = SparsePauliOp.from_list([(label, 1.0)])
            features.append(float(sv.expectation_value(op).real))

        # Per-bond ⟨ZiZj⟩
        for i in range(self.n_qubits - 1):
            j = i + 1
            label = ["I"] * self.n_qubits
            label[self.n_qubits - 1 - i] = "Z"
            label[self.n_qubits - 1 - j] = "Z"
            op = SparsePauliOp.from_list([("".join(label), 1.0)])
            features.append(float(sv.expectation_value(op).real))

        return np.array(features)

    def train(self, h_values: np.ndarray, theta_targets: np.ndarray) -> float:
        """Train MLP: reservoir features → θ_pred. Returns training MSE."""
        from sklearn.neural_network import MLPRegressor

        X_train = np.array([self.extract_features(h) for h in h_values])

        self._mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=2000,
            random_state=self.seed,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
            learning_rate_init=1e-3,
        )
        self._mlp.fit(X_train, theta_targets)
        train_pred = self._mlp.predict(X_train)
        return float(np.mean((train_pred - theta_targets) ** 2))

    def predict(self, h_value: float) -> np.ndarray:
        """Predict θ for a given h."""
        features = self.extract_features(h_value).reshape(1, -1)
        return self._mlp.predict(features).flatten()


# ── Training Data Generation ─────────────────────────────────────────────


def generate_training_data(N: int, n_train: int, seed: int = 42):
    """Generate VQE training data via descending sweep.

    Returns dict with h_values, theta_opt, fidelities, exact_data.
    """
    env = setup_experiment(N)
    qc = env["circuit"]
    h_values = np.linspace(0.2, 2.0, n_train)

    vqe_config = VQEConfig(p_layers=2, n_restarts=3, maxiter=500, enable_callbacks=False)
    opt = VQEOptimizer(vqe_config)

    exact_data = []
    for h in h_values:
        sol = get_exact_solution(env["builder"], env["solver"], N, h)
        exact_data.append(sol["exact"])

    np.random.seed(seed)
    vqe_results = opt.descending_sweep(h_values, qc, env["base_lattice"], exact_data)
    theta_opt = np.array([r.theta_opt for r in vqe_results])
    fidelities = np.array([r.fidelity for r in vqe_results])

    return {
        "env": env,
        "qc": qc,
        "h_values": h_values,
        "theta_opt": theta_opt,
        "fidelities": fidelities,
        "exact_data": exact_data,
    }


# ── Sub-experiment 2A ────────────────────────────────────────────────────


def run_sub_experiment_2A(args) -> SubExperimentResult:
    """4 reservoir designs comparison at N=6."""
    N = args.N
    n_train = args.n_train or 20
    seed = args.seed
    h_test_values = [0.5, 1.0, 1.25, 1.5, 1.8]

    print(f"\n{'=' * 70}")
    print(f"  Sub-experiment 2A: 4 reservoir designs (N={N})")
    print(f"  n_train={n_train}, test_h={h_test_values}")
    print(f"{'=' * 70}\n")

    # Generate training data
    data = generate_training_data(N, n_train, seed)
    qc = data["qc"]
    mask = data["fidelities"] >= 0.93
    h_filtered = data["h_values"][mask]
    theta_filtered = data["theta_opt"][mask]
    print(f"  Training: {mask.sum()}/{len(mask)} points above fidelity threshold")

    # Get optimized params for "optimized" reservoir (VQE at h=1.0)
    idx_h1 = np.argmin(np.abs(data["h_values"] - 1.0))
    optimized_params = data["theta_opt"][idx_h1]

    reservoir_types = ["random", "near_identity", "entangling", "optimized"]
    all_metrics = []
    detailed = []

    for rtype in reservoir_types:
        print(f"\n  Reservoir: {rtype}")
        qrc = QRCPredictor(N, p_layers=2, seed=seed, reservoir_type=rtype)
        qrc.build_reservoir(optimized_params=optimized_params if rtype == "optimized" else None)
        train_mse = qrc.train(h_filtered, theta_filtered)
        print(f"    Train MSE={train_mse:.6f}")

        for h_t in h_test_values:
            sol = get_exact_solution(data["env"]["builder"], data["env"]["solver"], N, h_t)
            exact = sol["exact"]
            H = sol["hamiltonian"]

            theta_pred = qrc.predict(h_t)
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
            detailed.append({"reservoir": rtype, "h": h_t, "error": err})

        avg_err = np.mean([d["error"] for d in detailed if d["reservoir"] == rtype])
        print(f"    Avg test ΔE={avg_err:.2e}")

    # Summary
    summary_by_reservoir = {}
    for rtype in reservoir_types:
        errs = [d["error"] for d in detailed if d["reservoir"] == rtype]
        summary_by_reservoir[rtype] = float(np.mean(errs))

    best_reservoir = min(summary_by_reservoir, key=summary_by_reservoir.get)
    summary = {
        "by_reservoir": summary_by_reservoir,
        "best_reservoir": best_reservoir,
        "detailed": detailed,
        "conclusion": f"Best reservoir: {best_reservoir} (avg ΔE={summary_by_reservoir[best_reservoir]:.2e})",
    }

    result = SubExperimentResult(
        experiment_id="2A",
        technique=2,
        description="4 reservoir designs comparison (N=6)",
        config={
            "N": N,
            "n_train": n_train,
            "seed": seed,
            "h_test": h_test_values,
            "reservoir_types": reservoir_types,
        },
        metrics=all_metrics,
        summary=summary,
        success=True,
    )
    path = save_experiment_result(result, RESULTS_DIR, prefix="qrc")
    print(f"\n  Result saved: {path.name}")
    return result


# ── Sub-experiment 2B ────────────────────────────────────────────────────


def run_sub_experiment_2B(args) -> SubExperimentResult:
    """N=10 QRC vs MPNN comparison (27 training points, production config)."""
    N = 10
    n_train = 27
    seed = args.seed
    h_test_values = [1.25, 1.4, 1.5]

    print(f"\n{'=' * 70}")
    print(f"  Sub-experiment 2B: QRC vs MPNN at N={N}")
    print(f"  n_train={n_train}, test_h={h_test_values}")
    print(f"{'=' * 70}\n")

    clear_exact_cache()
    data = generate_training_data(N, n_train, seed)
    qc = data["qc"]
    env = data["env"]
    mask = data["fidelities"] >= 0.93
    h_filtered = data["h_values"][mask]
    theta_filtered = data["theta_opt"][mask]
    print(f"  Training: {mask.sum()}/{len(mask)} points above fidelity threshold")

    # QRC predictor (best reservoir from 2A, default to random)
    qrc = QRCPredictor(N, p_layers=2, seed=seed, reservoir_type="random")
    qrc.build_reservoir()
    qrc_mse = qrc.train(h_filtered, theta_filtered)
    print(f"  QRC train MSE={qrc_mse:.6f}")

    # MPNN predictor (production config: h=128, L=3, 6000 epochs)
    import torch

    from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset

    torch.manual_seed(seed)
    exact_energies = np.array(
        [data["exact_data"][i].ground_energy for i in range(len(data["h_values"])) if mask[i]]
    )
    dataset = build_graph_dataset(
        env["base_lattice"],
        h_filtered,
        theta_filtered,
        exact_energies,
        fidelities=data["fidelities"][mask],
        fidelity_threshold=0.0,  # noqa — intentional: QRC uses pre-filtered mask, no double filtering
    )
    model = MPNNPredictor(
        node_features=2,
        hidden_dim=128,
        n_layers=3,
        output_dim=env["n_params"],
    )
    mpnn_result = train_mpnn_with_best_state(
        model,
        dataset,
        n_epochs=6000,
        lr=1e-3,
        patience=500,
    )
    print(f"  MPNN train MSE={mpnn_result['final_mse']:.6f}")

    # Evaluate both
    from torch_geometric.data import Data

    all_metrics = []
    detailed = []

    for h_t in h_test_values:
        sol = get_exact_solution(env["builder"], env["solver"], N, h_t)
        H = sol["hamiltonian"]
        exact = sol["exact"]

        # QRC prediction
        theta_qrc = qrc.predict(h_t)
        e_qrc = evaluate_energy_statevector(qc, H, theta_qrc)
        err_qrc = abs(e_qrc - exact.ground_energy)

        # MPNN prediction
        edge_idx, coord = env["builder"].build_graph_data(env["base_lattice"])
        x_test = torch.tensor(
            np.stack([np.full(N, h_t), coord.astype(float)], axis=1),
            dtype=torch.float32,
        )
        test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))
        model.eval()
        with torch.no_grad():
            theta_mpnn = model(test_graph).numpy().flatten()
        e_mpnn = evaluate_energy_statevector(qc, H, theta_mpnn)
        err_mpnn = abs(e_mpnn - exact.ground_energy)

        m_qrc = compute_metrics(
            energy=e_qrc, exact_energy=exact.ground_energy, gap=exact.gap, seed=seed, h_value=h_t
        )
        m_mpnn = compute_metrics(
            energy=e_mpnn, exact_energy=exact.ground_energy, gap=exact.gap, seed=seed, h_value=h_t
        )
        all_metrics.extend([m_qrc, m_mpnn])
        detailed.append({"h": h_t, "qrc_error": err_qrc, "mpnn_error": err_mpnn})
        print(f"  h={h_t:.2f}: QRC ΔE={err_qrc:.2e}, MPNN ΔE={err_mpnn:.2e}")

    avg_qrc = np.mean([d["qrc_error"] for d in detailed])
    avg_mpnn = np.mean([d["mpnn_error"] for d in detailed])
    winner = "MPNN" if avg_mpnn < avg_qrc else "QRC"

    summary = {
        "avg_qrc_error": float(avg_qrc),
        "avg_mpnn_error": float(avg_mpnn),
        "winner": winner,
        "detailed": detailed,
        "conclusion": f"{winner} wins at N=10 (QRC={avg_qrc:.2e}, MPNN={avg_mpnn:.2e})",
    }

    result = SubExperimentResult(
        experiment_id="2B",
        technique=2,
        description=f"QRC vs MPNN comparison (N={N}, 27 training points)",
        config={
            "N": N,
            "n_train": n_train,
            "seed": seed,
            "h_test": h_test_values,
            "mpnn_config": {"hidden": 128, "layers": 3, "epochs": 6000},
        },
        metrics=all_metrics,
        summary=summary,
        success=True,
    )
    path = save_experiment_result(result, RESULTS_DIR, prefix="qrc")
    print(f"\n  Result saved: {path.name}")
    return result


# ── Sub-experiment 2C ────────────────────────────────────────────────────


def run_sub_experiment_2C(args) -> SubExperimentResult:
    """Data efficiency: vary training set size [8, 12, 16, 20, 27]."""
    N = args.N
    seed = args.seed
    train_sizes = [8, 12, 16, 20, 27]
    h_test_values = [1.25, 1.4, 1.5]

    print(f"\n{'=' * 70}")
    print(f"  Sub-experiment 2C: Data efficiency (N={N})")
    print(f"  Training sizes={train_sizes}")
    print(f"{'=' * 70}\n")

    all_metrics = []
    detailed = []

    for n_train in train_sizes:
        clear_exact_cache()
        data = generate_training_data(N, n_train, seed)
        qc = data["qc"]
        env = data["env"]
        mask = data["fidelities"] >= 0.93
        h_filtered = data["h_values"][mask]
        theta_filtered = data["theta_opt"][mask]

        # QRC
        qrc = QRCPredictor(N, p_layers=2, seed=seed, reservoir_type="random")
        qrc.build_reservoir()
        qrc.train(h_filtered, theta_filtered)

        # MPNN
        import torch

        from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset

        torch.manual_seed(seed)
        exact_energies = np.array(
            [data["exact_data"][i].ground_energy for i in range(len(data["h_values"])) if mask[i]]
        )
        dataset = build_graph_dataset(
            env["base_lattice"],
            h_filtered,
            theta_filtered,
            exact_energies,
            fidelities=data["fidelities"][mask],
            fidelity_threshold=0.0,  # noqa — intentional: QRC uses pre-filtered mask
        )
        model = MPNNPredictor(
            node_features=2,
            hidden_dim=64,
            n_layers=3,
            output_dim=env["n_params"],
        )
        train_mpnn_with_best_state(model, dataset, n_epochs=3000, lr=1e-3, patience=200)

        # Evaluate
        from torch_geometric.data import Data

        qrc_errors = []
        mpnn_errors = []

        for h_t in h_test_values:
            sol = get_exact_solution(env["builder"], env["solver"], N, h_t)
            H = sol["hamiltonian"]
            exact = sol["exact"]

            theta_qrc = qrc.predict(h_t)
            e_qrc = evaluate_energy_statevector(qc, H, theta_qrc)
            qrc_errors.append(abs(e_qrc - exact.ground_energy))

            edge_idx, coord = env["builder"].build_graph_data(env["base_lattice"])
            x_test = torch.tensor(
                np.stack([np.full(N, h_t), coord.astype(float)], axis=1),
                dtype=torch.float32,
            )
            test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))
            model.eval()
            with torch.no_grad():
                theta_mpnn = model(test_graph).numpy().flatten()
            e_mpnn = evaluate_energy_statevector(qc, H, theta_mpnn)
            mpnn_errors.append(abs(e_mpnn - exact.ground_energy))

        avg_qrc = float(np.mean(qrc_errors))
        avg_mpnn = float(np.mean(mpnn_errors))
        detailed.append(
            {
                "n_train": n_train,
                "n_filtered": int(mask.sum()),
                "qrc_avg_error": avg_qrc,
                "mpnn_avg_error": avg_mpnn,
            }
        )
        print(
            f"  n_train={n_train:2d} (filtered={mask.sum():2d}): "
            f"QRC={avg_qrc:.2e}, MPNN={avg_mpnn:.2e}"
        )

    summary = {
        "detailed": detailed,
        "conclusion": "Data efficiency comparison across training sizes",
    }

    result = SubExperimentResult(
        experiment_id="2C",
        technique=2,
        description="Data efficiency (vary training set size)",
        config={"N": N, "seed": seed, "train_sizes": train_sizes, "h_test": h_test_values},
        metrics=all_metrics,
        summary=summary,
        success=True,
    )
    path = save_experiment_result(result, RESULTS_DIR, prefix="qrc")
    print(f"\n  Result saved: {path.name}")
    return result


# ── Sub-experiment 2D ────────────────────────────────────────────────────


def run_sub_experiment_2D(args) -> SubExperimentResult:
    """Hybrid QRC+MPNN: concatenate QRC features with MPNN embeddings."""
    N = args.N
    n_train = args.n_train or 20
    seed = args.seed
    h_test_values = [1.25, 1.4, 1.5]

    print(f"\n{'=' * 70}")
    print(f"  Sub-experiment 2D: Hybrid QRC+MPNN (N={N})")
    print("  Concatenate QRC features + MPNN graph embedding → MLP → θ")
    print(f"{'=' * 70}\n")

    data = generate_training_data(N, n_train, seed)
    qc = data["qc"]
    env = data["env"]
    mask = data["fidelities"] >= 0.93
    h_filtered = data["h_values"][mask]
    theta_filtered = data["theta_opt"][mask]

    # Extract QRC features for all training points
    qrc = QRCPredictor(N, p_layers=2, seed=seed, reservoir_type="random")
    qrc.build_reservoir()
    qrc_features = np.array([qrc.extract_features(h) for h in h_filtered])
    print(f"  QRC features: shape={qrc_features.shape}")

    # Combine QRC features + h-value as input to MLP
    # (Simpler hybrid: QRC features + h → MLP → θ, since full MPNN embedding
    #  would require architectural changes to MPNNPredictor)
    from sklearn.neural_network import MLPRegressor

    # Hybrid input: [QRC_features, h_value]
    X_hybrid = np.column_stack([qrc_features, h_filtered.reshape(-1, 1)])
    print(f"  Hybrid input: shape={X_hybrid.shape}")

    mlp_hybrid = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=3000,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=100,
        learning_rate_init=1e-3,
    )
    mlp_hybrid.fit(X_hybrid, theta_filtered)
    train_pred = mlp_hybrid.predict(X_hybrid)
    hybrid_mse = float(np.mean((train_pred - theta_filtered) ** 2))
    print(f"  Hybrid MLP train MSE={hybrid_mse:.6f}")

    # Also train standalone QRC for comparison
    qrc_mse = qrc.train(h_filtered, theta_filtered)
    print(f"  QRC-only train MSE={qrc_mse:.6f}")

    # Evaluate
    all_metrics = []
    detailed = []

    for h_t in h_test_values:
        sol = get_exact_solution(env["builder"], env["solver"], N, h_t)
        H = sol["hamiltonian"]
        exact = sol["exact"]

        # Hybrid prediction
        feat_test = qrc.extract_features(h_t)
        x_hybrid_test = np.concatenate([feat_test, [h_t]]).reshape(1, -1)
        theta_hybrid = mlp_hybrid.predict(x_hybrid_test).flatten()
        e_hybrid = evaluate_energy_statevector(qc, H, theta_hybrid)
        err_hybrid = abs(e_hybrid - exact.ground_energy)

        # QRC-only prediction
        theta_qrc = qrc.predict(h_t)
        e_qrc = evaluate_energy_statevector(qc, H, theta_qrc)
        err_qrc = abs(e_qrc - exact.ground_energy)

        m = compute_metrics(
            energy=e_hybrid, exact_energy=exact.ground_energy, gap=exact.gap, seed=seed, h_value=h_t
        )
        all_metrics.append(m)
        detailed.append({"h": h_t, "hybrid_error": err_hybrid, "qrc_error": err_qrc})
        print(f"  h={h_t:.2f}: Hybrid ΔE={err_hybrid:.2e}, QRC-only ΔE={err_qrc:.2e}")

    avg_hybrid = np.mean([d["hybrid_error"] for d in detailed])
    avg_qrc = np.mean([d["qrc_error"] for d in detailed])

    summary = {
        "avg_hybrid_error": float(avg_hybrid),
        "avg_qrc_error": float(avg_qrc),
        "hybrid_wins": avg_hybrid < avg_qrc,
        "detailed": detailed,
        "conclusion": f"Hybrid {'wins' if avg_hybrid < avg_qrc else 'loses'} vs QRC-only",
    }

    result = SubExperimentResult(
        experiment_id="2D",
        technique=2,
        description="Hybrid QRC+MPNN (concatenated features → MLP)",
        config={"N": N, "n_train": n_train, "seed": seed, "h_test": h_test_values},
        metrics=all_metrics,
        summary=summary,
        success=True,
    )
    path = save_experiment_result(result, RESULTS_DIR, prefix="qrc")
    print(f"\n  Result saved: {path.name}")
    return result


# ── CLI & Dispatch ───────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QRC warm-start experiments (Technique 2)",
    )
    parser.add_argument(
        "--sub-experiment", type=str, default="all", choices=["2A", "2B", "2C", "2D", "all"]
    )
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reservoir-type",
        type=str,
        default="random",
        choices=["random", "near_identity", "entangling", "optimized"],
    )
    parser.add_argument("--mpnn-epochs", type=int, default=6000)
    return parser.parse_args()


DISPATCH = {
    "2A": run_sub_experiment_2A,
    "2B": run_sub_experiment_2B,
    "2C": run_sub_experiment_2C,
    "2D": run_sub_experiment_2D,
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
                        technique=2,
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
