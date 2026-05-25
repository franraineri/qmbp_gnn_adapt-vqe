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
from torch_geometric.nn import GINConv, NNConv, global_mean_pool

from qmbp_simulation.models import HamiltonianBuilder, LatticeConfig

logger = logging.getLogger(__name__)

# NNConv edge MLP hidden dimension (from V6.1 config)
NNCONV_EDGE_MLP_HIDDEN = 32


# ── MPNNPredictor ────────────────────────────────────────────────────────


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
    per_parameter_heads : bool
        When True, use separate MLP heads for θ_zz and θ_x predictions
        instead of a single head. Default False preserves V6.0 behavior.
    use_edge_features : bool
        When True, use NNConv layers (processes edge attributes) instead
        of GINConv. Requires Data objects with ``edge_attr`` tensors.
        Default False preserves V6.0 GINConv behavior.
    edge_feature_dim : int
        Dimension of edge features (default 1 for scalar J_ij).
    """

    def __init__(
        self,
        node_features: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 3,
        output_dim: int = 4,
        per_parameter_heads: bool = False,
        use_edge_features: bool = False,
        edge_feature_dim: int = 1,
    ) -> None:
        super().__init__()

        # Store architecture attributes for checkpoint metadata
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.per_parameter_heads = per_parameter_heads
        self.use_edge_features = use_edge_features
        self.edge_feature_dim = edge_feature_dim

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if use_edge_features:
            # NNConv architecture: edge features processed through learned MLP
            # First layer: node_features → hidden_dim
            edge_nn_0 = nn.Sequential(
                nn.Linear(edge_feature_dim, NNCONV_EDGE_MLP_HIDDEN),
                nn.ReLU(),
                nn.Linear(NNCONV_EDGE_MLP_HIDDEN, node_features * hidden_dim),
            )
            self.convs.append(NNConv(node_features, hidden_dim, edge_nn_0, aggr="add"))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

            # Subsequent layers: hidden_dim → hidden_dim
            for _ in range(n_layers - 1):
                edge_nn = nn.Sequential(
                    nn.Linear(edge_feature_dim, NNCONV_EDGE_MLP_HIDDEN),
                    nn.ReLU(),
                    nn.Linear(NNCONV_EDGE_MLP_HIDDEN, hidden_dim * hidden_dim),
                )
                self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr="add"))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
        else:
            # GINConv architecture (V6.0 default)
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

        # Readout head(s): global pooled vector → θ_pred
        if per_parameter_heads:
            # Separate heads for θ_zz and θ_x (physics-informed specialization)
            self.head_zz = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim // 2),
            )
            self.head_x = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim // 2),
            )
            self.head = None  # Not used in per-parameter mode
        else:
            # Single head (V6.0 default, backward compatible)
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
            When ``use_edge_features=True``, must also have ``edge_attr``.

        Returns
        -------
        torch.Tensor of shape [batch_size, output_dim]

        Raises
        ------
        RuntimeError
            If NNConv mode is active but ``data.edge_attr`` is missing.
        """
        x, edge_index = data.x, data.edge_index
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self.use_edge_features:
            # NNConv requires edge_attr
            if not hasattr(data, "edge_attr") or data.edge_attr is None:
                raise RuntimeError(
                    "MPNNPredictor with use_edge_features=True requires "
                    "Data objects with edge_attr tensors. Got Data without "
                    "edge_attr. Ensure build_graph_dataset() was called with "
                    "include_edge_features=True."
                )
            edge_attr = data.edge_attr
            for conv, bn in zip(self.convs, self.bns, strict=False):
                x = conv(x, edge_index, edge_attr)
                x = bn(x)
                x = torch.relu(x)
        else:
            for conv, bn in zip(self.convs, self.bns, strict=False):
                x = conv(x, edge_index)
                x = bn(x)
                x = torch.relu(x)

        # Global mean pooling: lattice-agnostic fixed-size output
        x = global_mean_pool(x, batch)

        if self.per_parameter_heads:
            zz_out = self.head_zz(x)
            x_out = self.head_x(x)
            return torch.cat([zz_out, x_out], dim=-1)
        return self.head(x)


# ── Graph data construction utility ──────────────────────────────────────


def build_graph_dataset(
    lattice: LatticeConfig,
    h_values: np.ndarray,
    theta_opt: np.ndarray,
    e_exact: np.ndarray,
    fidelities: np.ndarray | None = None,
    fidelity_threshold: float = 0.93,
    include_edge_features: bool = False,
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
        Minimum fidelity to include in dataset (default 0.93).
    include_edge_features : bool
        When True, add ``edge_attr`` tensors containing coupling J_ij.

    Returns
    -------
    list[Data] — filtered graph dataset for MPNN training.

    Raises
    ------
    ValueError
        If fewer than 3 data points pass the fidelity filter.
    """
    builder = HamiltonianBuilder()
    edge_index_np, coord = builder.build_graph_data(lattice)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    # ── Edge feature construction ────────────────────────────────────
    edge_attr: torch.Tensor | None = None
    if include_edge_features:
        if isinstance(lattice.J, np.ndarray):
            # Per-bond J array: assign J_ij to each directed edge
            # edge_index has shape [2, 2*n_bonds] (both directions)
            j_values = np.concatenate([lattice.J, lattice.J])
            edge_attr = torch.tensor(j_values.reshape(-1, 1), dtype=torch.float32)
        else:
            # Scalar J: uniform coupling — no information gain
            logger.warning(
                "include_edge_features=True but LatticeConfig has scalar J "
                f"(J={lattice.J}). Skipping edge features — uniform coupling "
                "provides no information gain for the MPNN."
            )
            include_edge_features = False  # Disable for this dataset

    dataset: list[Data] = []
    for i, h in enumerate(h_values):
        # Fidelity filter: skip low-fidelity samples
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

        # Attach edge features if available
        if include_edge_features and edge_attr is not None:
            data.edge_attr = edge_attr

        dataset.append(data)

    # Enforce fidelity filter constraint: need at least 3 points
    if len(dataset) < 3:
        raise ValueError(
            f"Fewer than 3 data points ({len(dataset)}) passed the fidelity "
            f"filter (threshold={fidelity_threshold}). Cannot build a reliable "
            f"training dataset. Consider widening the h-grid or lowering the "
            f"fidelity threshold."
        )

    logger.info(
        f"Built graph dataset: {len(dataset)}/{len(h_values)} points "
        f"(fidelity threshold={fidelity_threshold}, "
        f"edge_features={include_edge_features})"
    )
    return dataset


# ── Training loop ────────────────────────────────────────────────────────


def train_mpnn(
    model: MPNNPredictor,
    dataset: list[Data],
    n_epochs: int = 4000,
    lr: float = 1e-3,
    patience: int = 150,
    energy_val_interval: int = 50,
    energy_val_fn: callable | None = None,
    divergence_window: int = 5,
    divergence_threshold: float = 0.01,
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
    divergence_window : int — window for divergence detection
    divergence_threshold : float — ΔE threshold for divergence detection.

    Returns
    -------
    dict with keys: 'mse_history', 'energy_val_history', 'final_mse',
                    'stopped_early', 'stop_reason',
                    and optionally 'zz_head_loss_history', 'x_head_loss_history'
                    when per_parameter_heads is enabled.

    Raises
    ------
    ValueError
        If dataset has fewer than 3 points (insufficient for training).
    """
    # ── Dataset validation ────────────────────────────────────────────
    if len(dataset) == 0:
        raise ValueError(
            "Empty dataset passed to train_mpnn(). "
            "Check fidelity filter threshold or h-grid range — "
            "all points may have been filtered out."
        )
    if len(dataset) < 3:
        raise ValueError(
            f"Dataset too small ({len(dataset)} points) for reliable MPNN training. "
            f"Need at least 3 points. Consider widening the h-grid or "
            f"lowering the fidelity threshold."
        )
    if len(dataset) < 10:
        logger.warning(
            f"Small dataset ({len(dataset)} points). MPNN predictions may be unreliable. "
            f"Consider adding more training points in the valid regime."
        )

    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience, factor=0.5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    # Detect per-parameter heads mode
    has_per_param_heads = hasattr(model, "per_parameter_heads") and model.per_parameter_heads

    mse_history: list[float] = []
    energy_val_history: list[float] = []
    zz_head_loss_history: list[float] = []
    x_head_loss_history: list[float] = []
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

        # ── Per-head loss reporting ──────────────────────────────────
        if has_per_param_heads and (epoch + 1) % energy_val_interval == 0:
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    pred = model(batch)
                    target = batch.y.view(pred.shape)
                    p = model.output_dim // 2
                    loss_zz = F.mse_loss(pred[:, :p], target[:, :p]).item()
                    loss_x = F.mse_loss(pred[:, p:], target[:, p:]).item()
                    zz_head_loss_history.append(loss_zz)
                    x_head_loss_history.append(loss_x)
                    logger.info(
                        f"  Epoch {epoch + 1}: MSE={epoch_loss:.2e}, "
                        f"ZZ-head={loss_zz:.2e}, X-head={loss_x:.2e}"
                    )
                    break  # single batch
            model.train()

        # ── Energy-driven validation callback ────────────────────────
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

            # ── Divergence detection ─────────────────────────────────
            if len(energy_val_history) >= divergence_window:
                recent_de = energy_val_history[-divergence_window:]
                recent_mse = mse_history[-divergence_window * energy_val_interval :]
                mse_improving = recent_mse[-1] < recent_mse[0] * 0.99
                de_stagnant = all(
                    abs(recent_de[k] - recent_de[k - 1]) < 0.01 * abs(recent_de[0])
                    for k in range(1, len(recent_de))
                )
                if mse_improving and de_stagnant and mean_de > divergence_threshold:
                    logger.warning(
                        f"Training divergence detected at epoch {epoch + 1}: "
                        f"MSE converging but ΔE stagnant ({mean_de:.2e}). "
                        f"Check Phase 2 data quality."
                    )
                    stopped_early = True
                    stop_reason = "divergence_detected"
                    break

    result = {
        "mse_history": mse_history,
        "energy_val_history": energy_val_history,
        "final_mse": mse_history[-1] if mse_history else float("inf"),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
    }

    if has_per_param_heads:
        result["zz_head_loss_history"] = zz_head_loss_history
        result["x_head_loss_history"] = x_head_loss_history

    return result


# ── Model save/load with architecture metadata ──────────────────────────


def save_mpnn_checkpoint(
    model: MPNNPredictor,
    path: str,
    training_metadata: dict | None = None,
) -> None:
    """Save model with architecture metadata for correct reconstruction.

    Parameters
    ----------
    model : MPNNPredictor
        Trained model to save.
    path : str
        File path for the checkpoint (.pt file).
    training_metadata : dict | None
        Optional training info (epoch, loss, dataset details).
    """
    architecture = "nnconv" if getattr(model, "use_edge_features", False) else "ginconv"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": architecture,
            "per_parameter_heads": getattr(model, "per_parameter_heads", False),
            "node_features": getattr(model, "node_features", 2),
            "hidden_dim": getattr(model, "hidden_dim", 64),
            "n_layers": getattr(model, "n_layers", 3),
            "output_dim": getattr(model, "output_dim", 4),
            "use_edge_features": getattr(model, "use_edge_features", False),
            "edge_feature_dim": getattr(model, "edge_feature_dim", 1),
            "training_metadata": training_metadata or {},
        },
        path,
    )

    logger.info(f"Saved MPNNCheckpoint to {path}")


def load_mpnn_checkpoint(path: str) -> MPNNPredictor:
    """Load model from checkpoint, reconstructing correct architecture.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.

    Returns
    -------
    MPNNPredictor
        Reconstructed model with loaded weights.
    """
    data = torch.load(path, map_location="cpu", weights_only=False)

    # Handle legacy checkpoints without metadata
    if "state_dict" not in data:
        # Assume it's a raw state_dict (V6.0 format)
        logger.warning(
            "Loading legacy checkpoint without architecture metadata. "
            "Assuming single-head GINConv (V6.0 defaults)."
        )
        model = MPNNPredictor()
        model.load_state_dict(data)
        return model

    # Reconstruct architecture from metadata
    model = MPNNPredictor(
        node_features=data.get("node_features", 2),
        hidden_dim=data.get("hidden_dim", 64),
        n_layers=data.get("n_layers", 3),
        output_dim=data.get("output_dim", 4),
        per_parameter_heads=data.get("per_parameter_heads", False),
        use_edge_features=data.get("use_edge_features", False),
        edge_feature_dim=data.get("edge_feature_dim", 1),
    )
    model.load_state_dict(data["state_dict"])
    logger.info(
        f"Loaded MPNNCheckpoint: arch={data['architecture']}, "
        f"per_param_heads={data.get('per_parameter_heads', False)}, "
        f"edge_features={data.get('use_edge_features', False)}"
    )
    return model
