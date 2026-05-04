"""
MPNN Predictor — Lattice-agnostic parameter predictor via PyTorch Geometric
message passing.

Replaces the V4 MLP with a Message Passing Neural Network that generalizes
across different lattice topologies and system sizes via global pooling.

Architecture:
  Input: Data(x=[n_qubits, 2], edge_index=[2, n_edges])
    - Node features: (h_i, coordination_number_i)
  Message Passing: k GINConv layers with ReLU + BatchNorm
  Readout: global_mean_pool → MLP head → θ_pred ∈ ℝ^(2p)

References
----------
- Qracle (Zhang et al., 2025): GNN-based VQE parameter initializer.
- NN-VQE (Miao et al., PRApplied 2024): MLP h→θ for spin Hamiltonians.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool

from .config import LatticeConfig

logger = logging.getLogger(__name__)


# ── Task 6.1: MPNNPredictor ──────────────────────────────────────────────


class MPNNPredictor(nn.Module):
    """Lattice-agnostic MPNN that maps graph-structured Hamiltonian data
    to optimal HVA parameters.

    Parameters
    ----------
    node_features : int
        Number of input node features (default 2: h_i, coord_i).
    hidden_dim : int
        Hidden dimension for message passing layers.
    n_layers : int
        Number of GINConv message passing layers.
    output_dim : int
        Output dimension (2 * p_layers).
    """

    def __init__(
        self,
        node_features: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 3,
        output_dim: int = 4,
    ) -> None:
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer: node_features → hidden_dim
        mlp0 = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINConv(mlp0))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Subsequent layers: hidden_dim → hidden_dim
        for _ in range(n_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Readout MLP: global pooled vector → θ_pred
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Predict θ_pred from graph-structured Hamiltonian data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Must have ``x``, ``edge_index``, and ``batch`` attributes.

        Returns
        -------
        torch.Tensor of shape [batch_size, output_dim]
        """
        x, edge_index = data.x, data.edge_index
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for conv, bn in zip(self.convs, self.bns, strict=False):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)

        # Global mean pooling: lattice-agnostic fixed-size output
        x = global_mean_pool(x, batch)

        return self.head(x)


# ── Task 6.2: Graph data construction utility ────────────────────────────


def build_graph_dataset(
    lattice: LatticeConfig,
    h_values: np.ndarray,
    theta_opt: np.ndarray,
    e_exact: np.ndarray,
    fidelities: np.ndarray | None = None,
    fidelity_threshold: float = 0.93,
) -> list[Data]:
    """Convert LatticeConfig + θ_opt arrays into torch_geometric Data objects.

    Parameters
    ----------
    lattice : LatticeConfig
        Lattice specification (edges, coordination numbers).
    h_values : np.ndarray [n_points]
        Transverse field values.
    theta_opt : np.ndarray [n_points, 2p]
        Optimized parameters per h-point.
    e_exact : np.ndarray [n_points]
        Exact ground state energies.
    fidelities : np.ndarray | None [n_points]
        VQE fidelities for filtering.
    fidelity_threshold : float
        Minimum fidelity to include in dataset.

    Returns
    -------
    list[Data] — filtered graph dataset for MPNN training.
    """
    from .hamiltonian_builder import HamiltonianBuilder

    builder = HamiltonianBuilder()
    edge_index_np, coord = builder.build_graph_data(lattice)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    dataset: list[Data] = []
    for i, h in enumerate(h_values):
        # Task 6.3 (fidelity filter): skip low-fidelity samples
        if fidelities is not None and fidelities[i] < fidelity_threshold:
            continue

        # Node features: [h_i, coordination_number_i] per site
        h_feat = np.full(lattice.n_qubits, float(h))
        x = torch.tensor(
            np.stack([h_feat, coord.astype(float)], axis=1),
            dtype=torch.float32,
        )
        y = torch.tensor(theta_opt[i], dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.e_exact = float(e_exact[i])
        data.h_value = float(h)
        dataset.append(data)

    logger.info(
        f"Built graph dataset: {len(dataset)}/{len(h_values)} points "
        f"(fidelity threshold={fidelity_threshold})"
    )
    return dataset


# ── Task 6.3: Training loop ─────────────────────────────────────────────


def train_mpnn(
    model: MPNNPredictor,
    dataset: list[Data],
    n_epochs: int = 4000,
    lr: float = 1e-3,
    patience: int = 150,
    energy_val_interval: int = 50,
    energy_val_fn: callable | None = None,
    divergence_window: int = 5,
) -> dict:
    """Train the MPNN with MSE loss, ReduceLROnPlateau, and optional
    energy-driven validation.

    Parameters
    ----------
    model : MPNNPredictor
    dataset : list[Data] — training data (already fidelity-filtered)
    n_epochs : int
    lr : float — initial learning rate
    patience : int — scheduler patience
    energy_val_interval : int — epochs between energy validation callbacks
    energy_val_fn : callable | None
        Function(theta_pred_batch, data_batch) -> list[float] of energy errors.
        If provided, enables energy-driven validation (Task 6.4).
    divergence_window : int — window for divergence detection (Task 6.5)

    Returns
    -------
    dict with keys: 'mse_history', 'energy_val_history', 'final_mse',
                    'stopped_early', 'stop_reason'
    """
    from torch_geometric.loader import DataLoader

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience, factor=0.5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    mse_history: list[float] = []
    energy_val_history: list[float] = []
    stopped_early = False
    stop_reason = "completed"

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            pred = model(batch)
            target = batch.y.view(pred.shape)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()

        mse_history.append(epoch_loss)
        scheduler.step(epoch_loss)

        # ── Task 6.4: energy-driven validation callback ──────────────
        if energy_val_fn is not None and (epoch + 1) % energy_val_interval == 0:
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    pred = model(batch)
                    energy_errors = energy_val_fn(pred, batch)
                    mean_de = float(np.mean(energy_errors))
                    energy_val_history.append(mean_de)
                    if (epoch + 1) % (energy_val_interval * 4) == 0:
                        logger.info(
                            f"  Epoch {epoch + 1}: MSE={epoch_loss:.2e}, ΔE_mean={mean_de:.2e}"
                        )
                    break  # single batch
            model.train()

            # ── Task 6.5: divergence detection ───────────────────────
            if len(energy_val_history) >= divergence_window:
                recent_de = energy_val_history[-divergence_window:]
                recent_mse = mse_history[-divergence_window * energy_val_interval :]
                mse_improving = recent_mse[-1] < recent_mse[0] * 0.99
                de_stagnant = all(
                    abs(recent_de[k] - recent_de[k - 1]) < 0.01 * abs(recent_de[0])
                    for k in range(1, len(recent_de))
                )
                if mse_improving and de_stagnant and mean_de > 0.01:
                    logger.warning(
                        f"Training divergence detected at epoch {epoch + 1}: "
                        f"MSE converging but ΔE stagnant ({mean_de:.2e}). "
                        f"Check Phase 2 data quality."
                    )
                    stopped_early = True
                    stop_reason = "divergence_detected"
                    break

    return {
        "mse_history": mse_history,
        "energy_val_history": energy_val_history,
        "final_mse": mse_history[-1] if mse_history else float("inf"),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
    }
