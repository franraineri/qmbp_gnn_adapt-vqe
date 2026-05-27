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
