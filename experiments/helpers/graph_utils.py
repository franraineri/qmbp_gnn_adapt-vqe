"""Shared graph construction utilities for experiments.

Provides topology-aware graph building for MPNN dataset construction
and prediction. Replaces hardcoded chain_1d edge patterns that were
duplicated across G1, G2, G3, G5, and other predictor experiments.

Usage:
    from experiments.helpers.graph_utils import build_experiment_dataset, predict_theta

    dataset = build_experiment_dataset(experiment, h_values, theta_array)
    theta_pred = predict_theta(experiment, model, h_test)

    # Batch prediction at multiple h-values:
    predictions = predict_theta_batch(experiment, model, [1.5, 1.75, 2.0])
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch, Data


def _get_graph_structure(experiment) -> tuple[torch.Tensor, np.ndarray, int]:
    """Extract graph structure from experiment's lattice configuration.

    Returns the edge_index tensor, coordination numbers, and N.
    Validates that the experiment has been properly set up.

    Returns
    -------
    tuple[torch.Tensor, np.ndarray, int]
        (edge_index, coordination_numbers, n_qubits)
    """
    from qmbp_simulation import make_lattice

    if experiment.builder is None:
        raise RuntimeError(
            "Experiment.builder is None — call experiment.setup() before "
            "using graph_utils. The builder is initialized during setup()."
        )

    N = experiment.config.system.n_qubits
    topology = experiment.config.system.topology
    J = experiment.config.system.J

    lattice = make_lattice(topology, N, J=J, h=1.0)
    edge_index_np, coord = experiment.builder.build_graph_data(lattice)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    return edge_index, coord, N


def build_experiment_dataset(
    experiment,
    h_values: np.ndarray,
    theta_array: np.ndarray,
) -> list[Data]:
    """Build a PyG dataset using the experiment's lattice topology.

    Uses HamiltonianBuilder.build_graph_data() to get correct edges
    and coordination numbers for any supported topology (chain_1d,
    ladder, triangular, kagome).

    Parameters
    ----------
    experiment : BaseExperiment
        The experiment instance (must have setup() called).
    h_values : np.ndarray
        Array of h-values, shape [n_points].
    theta_array : np.ndarray
        Array of optimized parameters, shape [n_points, n_params].

    Returns
    -------
    list[Data]
        PyG dataset ready for MPNN training.

    Raises
    ------
    ValueError
        If h_values and theta_array have mismatched lengths.
    RuntimeError
        If experiment.builder is None (setup not called).
    """
    h_values = np.asarray(h_values)
    theta_array = np.asarray(theta_array)

    if len(h_values) != len(theta_array):
        raise ValueError(
            f"h_values length ({len(h_values)}) != theta_array length ({len(theta_array)})"
        )
    if len(h_values) == 0:
        raise ValueError("h_values is empty — cannot build dataset.")

    edge_index, coord, N = _get_graph_structure(experiment)

    if theta_array.ndim == 1:
        # Single point — reshape to [1, n_params]
        theta_array = theta_array.reshape(1, -1)

    dataset = []
    for i, h in enumerate(h_values):
        h_feat = np.full(N, float(h))
        x = torch.tensor(
            np.stack([h_feat, coord.astype(float)], axis=1),
            dtype=torch.float32,
        )
        y = torch.tensor(theta_array[i], dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index, y=y)
        data.h_value = float(h)
        dataset.append(data)

    return dataset


def predict_theta(
    experiment,
    model,
    h: float,
) -> np.ndarray:
    """Predict θ at a single h-value using the experiment's lattice topology.

    Parameters
    ----------
    experiment : BaseExperiment
        The experiment instance (must have setup() called).
    model : MPNNPredictor
        Trained MPNN model.
    h : float
        Transverse field value for prediction.

    Returns
    -------
    np.ndarray
        Predicted parameter vector, shape [n_params].

    Raises
    ------
    RuntimeError
        If experiment.builder is None (setup not called).
    """
    edge_index, coord, N = _get_graph_structure(experiment)

    h_feat = np.full(N, float(h))
    x = torch.tensor(
        np.stack([h_feat, coord.astype(float)], axis=1),
        dtype=torch.float32,
    )
    graph = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([graph])

    model.eval()
    with torch.no_grad():
        return model(batch).numpy().flatten()


def predict_theta_batch(
    experiment,
    model,
    h_values: list[float] | np.ndarray,
) -> np.ndarray:
    """Predict θ at multiple h-values in a single batched forward pass.

    More efficient than calling predict_theta() in a loop because
    it batches all graphs into a single forward pass.

    Parameters
    ----------
    experiment : BaseExperiment
        The experiment instance (must have setup() called).
    model : MPNNPredictor
        Trained MPNN model.
    h_values : list[float] | np.ndarray
        Transverse field values for prediction.

    Returns
    -------
    np.ndarray
        Predicted parameters, shape [n_points, n_params].
    """
    h_values = np.asarray(h_values, dtype=float)
    edge_index, coord, N = _get_graph_structure(experiment)

    graphs = []
    for h in h_values:
        h_feat = np.full(N, float(h))
        x = torch.tensor(
            np.stack([h_feat, coord.astype(float)], axis=1),
            dtype=torch.float32,
        )
        graphs.append(Data(x=x, edge_index=edge_index))

    batch = Batch.from_data_list(graphs)

    model.eval()
    with torch.no_grad():
        predictions = model(batch).numpy()

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Bond-Resolved MPNN Training & Evaluation Helpers
# ─────────────────────────────────────────────────────────────────────────────


def train_bond_resolved_variant(
    lattice,
    h_values: np.ndarray,
    theta_opts: np.ndarray,
    *,
    include_circuit_nodes: bool = False,
    p_layers: int = 1,
    hidden_dim: int = 256,
    n_layers: int = 3,
    n_epochs: int = 6000,
    lr: float = 1e-3,
    patience: int = 300,
    seed: int = 42,
    dropout: float = 0.1,
    weight_decay: float = 0.0,
    val_fraction: float = 0.2,
    norm_type: str = "none",
) -> tuple:
    """Train a BondResolvedMPNN on given data with configurable graph type.

    This is the reusable training pattern for the 2×2 variant matrix:
    - include_circuit_nodes=False: Hamiltonian-only graph (baseline, 3 features)
    - include_circuit_nodes=True: Unified Hamiltonian+Circuit graph (4 features)

    The model architecture adapts automatically based on the graph type
    (3 vs 4 node features).

    Parameters
    ----------
    lattice : LatticeConfig
        Lattice specification (topology, edges, N).
    h_values : np.ndarray of shape [n_points]
        Training h-values.
    theta_opts : np.ndarray of shape [n_points, n_params]
        Training targets (noiseless or noisy θ_opt).
    include_circuit_nodes : bool
        If True, uses unified Hamiltonian+Circuit graph.
    p_layers : int
        Number of HVA layers (for unified graph gate node count).
    hidden_dim : int
        GNN hidden dimension.
    n_layers : int
        Number of GINConv layers.
    n_epochs : int
        Training epochs.
    lr : float
        Learning rate.
    patience : int
        ReduceLROnPlateau patience.
    seed : int
        Random seed.
    dropout : float
        Dropout rate in prediction heads.
    weight_decay : float
        L2 regularization weight for Adam optimizer. Use 1e-4 to 1e-3
        for unified graphs (larger models, overfitting risk).
    val_fraction : float
        Fraction of data held out for validation (0.0 disables).
        Validation MSE is reported separately to detect overfitting.
    norm_type : str
        Normalization type: "none" (default, best for cross-N),
        "batch", or "layer".

    Returns
    -------
    tuple[BondResolvedMPNN, dict]
        (trained_model, training_metrics)
        training_metrics contains:
            - final_mse: training MSE at last epoch
            - val_mse: validation MSE (None if val_fraction=0)
            - generalization_gap: val_mse - final_mse (overfitting indicator)
            - mse_history: list of per-epoch training MSE
            - val_mse_history: list of per-epoch val MSE (sampled every 50 epochs)
            - n_epochs_run: actual epochs executed
    """
    import torch
    import torch.nn.functional as F

    from qmbp_simulation.predictors import (
        BondResolvedMPNN,
        build_bond_resolved_graph,
    )
    from qmbp_simulation.predictors.unified_graph import (
        build_unified_bond_resolved_graph,
    )
    from qmbp_simulation.utils.helpers import canonicalize_theta

    # Build dataset
    dataset = []
    for i, h in enumerate(h_values):
        theta = canonicalize_theta(theta_opts[i])
        if include_circuit_nodes:
            graph = build_unified_bond_resolved_graph(
                lattice, h_value=float(h), p_layers=p_layers,
                theta_opt=theta, include_circuit_nodes=True,
            )
        else:
            graph = build_bond_resolved_graph(
                lattice, h_value=float(h), theta_opt=theta,
            )
        dataset.append(graph)

    # Train/val split (deterministic based on seed)
    rng = np.random.default_rng(seed)
    n_total = len(dataset)
    n_val = int(n_total * val_fraction) if val_fraction > 0 else 0
    if n_val > 0 and n_total - n_val < 3:
        # Not enough training data — disable validation
        n_val = 0

    if n_val > 0:
        indices = rng.permutation(n_total)
        val_indices = set(indices[:n_val].tolist())
        train_dataset = [dataset[i] for i in range(n_total) if i not in val_indices]
        val_dataset = [dataset[i] for i in val_indices]
    else:
        train_dataset = dataset
        val_dataset = []

    # Determine node_features from first graph
    node_features = dataset[0].x.shape[1]

    # Create model
    model = BondResolvedMPNN(
        node_features=node_features,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        norm_type=norm_type,
        dropout=dropout,
    )

    # Training loop (custom to support weight_decay + val tracking)
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience, factor=0.5, min_lr=1e-6
    )

    mse_history = []
    val_mse_history = []

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for data in train_dataset:
            optimizer.zero_grad()
            pred = model(data).squeeze(0)
            target = data.y
            n_e = data.n_edges_unique
            loss = F.mse_loss(pred[:n_e], target[:n_e]) + F.mse_loss(pred[n_e:], target[n_e:])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataset)
        mse_history.append(avg_loss)
        scheduler.step(avg_loss)

        # Validation MSE (every 50 epochs to avoid overhead)
        if val_dataset and epoch % 50 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in val_dataset:
                    pred = model(data).squeeze(0)
                    target = data.y
                    n_e = data.n_edges_unique
                    loss_v = F.mse_loss(pred[:n_e], target[:n_e]) + F.mse_loss(pred[n_e:], target[n_e:])
                    val_loss += loss_v.item()
            val_mse_history.append(val_loss / len(val_dataset))
            model.train()

        # Early stopping on LR exhaustion
        if optimizer.param_groups[0]["lr"] <= 1e-6 and epoch > 500:
            break

    # Final val MSE
    final_val_mse = None
    if val_dataset:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_dataset:
                pred = model(data).squeeze(0)
                target = data.y
                n_e = data.n_edges_unique
                loss_v = F.mse_loss(pred[:n_e], target[:n_e]) + F.mse_loss(pred[n_e:], target[n_e:])
                val_loss += loss_v.item()
        final_val_mse = val_loss / len(val_dataset)

    final_mse = mse_history[-1] if mse_history else float("inf")
    gen_gap = (final_val_mse - final_mse) if final_val_mse is not None else None

    return model, {
        "final_mse": float(final_mse),
        "val_mse": float(final_val_mse) if final_val_mse is not None else None,
        "generalization_gap": float(gen_gap) if gen_gap is not None else None,
        "mse_history": mse_history,
        "val_mse_history": val_mse_history,
        "n_epochs_run": len(mse_history),
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "weight_decay": weight_decay,
        "node_features": node_features,
    }


def evaluate_bond_resolved_variant(
    model,
    lattice,
    h_test_values: np.ndarray,
    circuit,
    spec,
    backend,
    *,
    include_circuit_nodes: bool = False,
    p_layers: int = 1,
    e_exact: np.ndarray | None = None,
    gaps: np.ndarray | None = None,
    n_eval_seeds: int = 1,
    base_seed: int = 200,
) -> dict:
    """Evaluate a trained BondResolvedMPNN on test h-points.

    Reusable evaluation pattern: predicts θ at each test point, evaluates
    energy on the given backend, and computes ΔE/gap metrics.

    When n_eval_seeds > 1 and backend is NoisyBackend, evaluates each
    prediction multiple times with different noise seeds to obtain
    mean ± std of ΔE/gap (accounts for shot noise stochasticity).

    Parameters
    ----------
    model : BondResolvedMPNN
        Trained model.
    lattice : LatticeConfig
        Lattice specification (must match training topology).
    h_test_values : np.ndarray
        Test h-values.
    circuit : QuantumCircuit
        Parameterized HVA circuit.
    spec : ModelSpec
        Model spec for Hamiltonian construction.
    backend : ExecutionBackend
        Backend for energy evaluation (NoiselessBackend or NoisyBackend).
    include_circuit_nodes : bool
        Must match training graph type.
    p_layers : int
        Must match training p_layers.
    e_exact : np.ndarray | None
        Exact energies at test points. If None, computed via ClassicalSolver.
    gaps : np.ndarray | None
        Spectral gaps at test points. If None, computed via ClassicalSolver.
    n_eval_seeds : int
        Number of evaluation repetitions for noisy backends. Default 1
        (deterministic). Use 5-10 for reliable statistics on NoisyBackend.
    base_seed : int
        Starting seed for multi-eval (seeds: base_seed, base_seed+1, ...).

    Returns
    -------
    dict with keys:
        - per_point: list of {h, e_pred, e_exact, gap, de_gap, de_abs, theta_pred}
        - mean_de_gap: float
        - median_de_gap: float
        - max_de_gap: float
        - std_de_gap: float (0 if n_eval_seeds=1)
        - mean_de_abs: float (|ΔE| absolute, not normalized by gap)
        - pass_rate: fraction with ΔE/gap < 5%
        - n_pass: count passing
        - n_total: total points
        - n_eval_seeds: seeds used
    """
    import torch

    from qmbp_simulation import ClassicalSolver, make_lattice
    from qmbp_simulation.models import HamiltonianBuilder
    from qmbp_simulation.models.constants import DE_GAP_THRESHOLD
    from qmbp_simulation.predictors import build_bond_resolved_graph
    from qmbp_simulation.predictors.unified_graph import (
        build_unified_bond_resolved_graph,
    )

    solver = ClassicalSolver()
    model.eval()

    per_point = []
    for i, h in enumerate(h_test_values):
        # Build prediction graph (no target)
        if include_circuit_nodes:
            graph = build_unified_bond_resolved_graph(
                lattice, h_value=float(h), p_layers=p_layers,
                include_circuit_nodes=True,
            )
        else:
            graph = build_bond_resolved_graph(lattice, h_value=float(h))

        # Predict θ (deterministic — same for all eval seeds)
        with torch.no_grad():
            theta_pred = model(graph).numpy().flatten()

        # Get exact reference
        if e_exact is not None and gaps is not None:
            e_ex = float(e_exact[i])
            gap = float(gaps[i])
        else:
            lat_h = make_lattice(lattice.topology, lattice.n_qubits, J=1.0, h=float(h))
            H_exact = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)
            gt = solver.solve(H_exact, lat_h)
            e_ex = gt.ground_energy
            gap = gt.gap

        # Evaluate energy (possibly multiple times for noisy stats)
        lat_eval = make_lattice(lattice.topology, lattice.n_qubits, J=1.0, h=float(h))
        H_eval = spec.build_hamiltonian(lat_eval, **spec.hamiltonian_kwargs)

        if n_eval_seeds > 1:
            e_preds = []
            for s in range(n_eval_seeds):
                # Re-seed noisy backend if possible
                if hasattr(backend, "_rng"):
                    backend._rng = np.random.default_rng(base_seed + s)
                e_preds.append(backend.evaluate(circuit, H_eval, theta_pred))
            e_pred = float(np.mean(e_preds))
            e_pred_std = float(np.std(e_preds))
        else:
            e_pred = backend.evaluate(circuit, H_eval, theta_pred)
            e_pred_std = 0.0

        de_gap = abs(e_pred - e_ex) / max(gap, 1e-10)
        de_abs = abs(e_pred - e_ex)

        per_point.append({
            "h": float(h),
            "e_pred": float(e_pred),
            "e_pred_std": e_pred_std,
            "e_exact": e_ex,
            "gap": gap,
            "de_gap": float(de_gap),
            "de_abs": float(de_abs),
            "theta_pred": theta_pred.tolist(),
        })

    de_gaps = np.array([p["de_gap"] for p in per_point])
    de_abs_arr = np.array([p["de_abs"] for p in per_point])
    n_pass = int((de_gaps < DE_GAP_THRESHOLD).sum())

    return {
        "per_point": per_point,
        "mean_de_gap": float(de_gaps.mean()),
        "median_de_gap": float(np.median(de_gaps)),
        "max_de_gap": float(de_gaps.max()),
        "std_de_gap": float(de_gaps.std()),
        "mean_de_abs": float(de_abs_arr.mean()),
        "pass_rate": n_pass / len(de_gaps) if len(de_gaps) > 0 else 0.0,
        "n_pass": n_pass,
        "n_total": len(de_gaps),
        "n_eval_seeds": n_eval_seeds,
    }


def compare_theta_arrays(
    theta_a: np.ndarray,
    theta_b: np.ndarray,
    h_values: np.ndarray | None = None,
) -> dict:
    """Compare two θ_opt arrays element-wise (e.g., noiseless vs noisy).

    Reusable for any pairwise θ comparison: noise vs noiseless, seed A vs B,
    topology A vs B, etc. All metrics computed here can also be derived
    post-hoc from saved arrays, but this function standardizes the format.

    Parameters
    ----------
    theta_a : np.ndarray of shape [n_points, n_params]
        First θ array (e.g., noiseless).
    theta_b : np.ndarray of shape [n_points, n_params]
        Second θ array (e.g., noisy).
    h_values : np.ndarray | None
        If provided, returned in per-point data for correlation analysis.

    Returns
    -------
    dict with keys:
        - mean_l2_displacement: mean ‖θ_a - θ_b‖₂ across points
        - std_l2_displacement: std of the same
        - max_l2_displacement: maximum displacement (worst point)
        - per_param_correlation: Pearson r per parameter column
        - mean_correlation: average Pearson r across parameters
        - per_point_l2: list of ‖θ_a_i - θ_b_i‖₂ per h-point
    """
    theta_a = np.asarray(theta_a)
    theta_b = np.asarray(theta_b)

    if theta_a.shape != theta_b.shape:
        raise ValueError(
            f"Shape mismatch: theta_a={theta_a.shape}, theta_b={theta_b.shape}"
        )

    # Per-point L2 displacement
    diffs = theta_a - theta_b
    per_point_l2 = np.linalg.norm(diffs, axis=1)

    # Per-parameter Pearson correlation
    n_params = theta_a.shape[1]
    per_param_corr = []
    for j in range(n_params):
        col_a = theta_a[:, j]
        col_b = theta_b[:, j]
        if np.std(col_a) < 1e-10 or np.std(col_b) < 1e-10:
            per_param_corr.append(0.0)
        else:
            r = np.corrcoef(col_a, col_b)[0, 1]
            per_param_corr.append(float(r) if np.isfinite(r) else 0.0)

    result = {
        "mean_l2_displacement": float(per_point_l2.mean()),
        "std_l2_displacement": float(per_point_l2.std()),
        "max_l2_displacement": float(per_point_l2.max()),
        "per_param_correlation": per_param_corr,
        "mean_correlation": float(np.mean(per_param_corr)),
        "per_point_l2": per_point_l2.tolist(),
    }

    if h_values is not None:
        result["h_values"] = list(h_values)

    return result


def train_unified_mpnn_variant(
    lattice,
    h_values: np.ndarray,
    theta_opts: np.ndarray,
    *,
    p_layers: int = 1,
    hidden_dim: int = 256,
    n_layers: int = 3,
    n_epochs: int = 6000,
    lr: float = 1e-3,
    patience: int = 300,
    seed: int = 42,
    dropout: float = 0.1,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.2,
    norm_type: str = "none",
    type_embedding_dim: int = 16,
    gate_readout: bool = True,
) -> tuple:
    """Train a UnifiedMPNN (type-aware architecture) on unified graphs.

    This is the Qracle-style variant for the comparison matrix. Unlike
    train_bond_resolved_variant which uses BondResolvedMPNN with uniform
    message passing, this uses UnifiedMPNN with learned type embeddings
    and gate-node readout for θ_zz predictions.

    Parameters
    ----------
    lattice : LatticeConfig
        Lattice specification.
    h_values : np.ndarray of shape [n_points]
        Training h-values.
    theta_opts : np.ndarray of shape [n_points, n_params]
        Training targets.
    p_layers : int
        Number of HVA layers.
    hidden_dim : int
        GNN hidden dimension.
    n_layers : int
        Number of GINConv layers.
    n_epochs : int
        Training epochs.
    lr : float
        Learning rate.
    patience : int
        ReduceLROnPlateau patience.
    seed : int
        Random seed.
    dropout : float
        Dropout rate.
    weight_decay : float
        L2 regularization (default 1e-4).
    val_fraction : float
        Fraction held out for validation.
    norm_type : str
        Normalization type.
    type_embedding_dim : int
        Learned type embedding dimension (0 to disable).
    gate_readout : bool
        If True, predict θ_zz from ZZ gate node embeddings directly.

    Returns
    -------
    tuple[UnifiedMPNN, dict]
        (trained_model, training_metrics)
    """
    from qmbp_simulation.predictors.unified_graph import (
        build_unified_bond_resolved_graph,
        UNIFIED_NODE_FEATURES,
    )
    from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn
    from qmbp_simulation.utils.helpers import canonicalize_theta

    # Build unified dataset
    dataset = []
    for i, h in enumerate(h_values):
        theta = canonicalize_theta(theta_opts[i])
        graph = build_unified_bond_resolved_graph(
            lattice, h_value=float(h), p_layers=p_layers,
            theta_opt=theta, include_circuit_nodes=True,
        )
        dataset.append(graph)

    # Create model
    model = UnifiedMPNN(
        node_features=UNIFIED_NODE_FEATURES,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        norm_type=norm_type,
        dropout=dropout,
        type_embedding_dim=type_embedding_dim,
        gate_readout=gate_readout,
    )

    # Train
    metrics = train_unified_mpnn(
        model, dataset,
        n_epochs=n_epochs,
        lr=lr,
        patience=patience,
        seed=seed,
        weight_decay=weight_decay,
        val_fraction=val_fraction,
    )

    # Add architecture metadata
    metrics["architecture"] = "UnifiedMPNN"
    metrics["type_embedding_dim"] = type_embedding_dim
    metrics["gate_readout"] = gate_readout
    metrics["node_features"] = UNIFIED_NODE_FEATURES

    return model, metrics
