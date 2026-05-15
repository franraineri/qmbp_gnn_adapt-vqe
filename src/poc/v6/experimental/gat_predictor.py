"""
GATPredictor — Attention-based GNN predictor (DEPRECATED).

Status: Tested and REJECTED during V6.0 development.
Reason: GATConv adds instability for 1D chains without improving accuracy.
        GINConv (MPNNPredictor) is strictly superior for TFIM chain topology.

Kept for reproducibility of V6.0 benchmark results.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool


class GATPredictor(nn.Module):
    """Attention-based GNN predictor using GATConv.

    DEPRECATED: Use MPNNPredictor (GINConv) instead.

    Uses multi-head attention to learn which neighbors are most informative
    for parameter prediction. May perform better near phase transitions where
    correlations are long-range.

    Parameters
    ----------
    node_features : int
        Number of input node features (default 2: h_i, coord_i).
    hidden_dim : int
        Hidden dimension per attention head.
    n_layers : int
        Number of GATConv layers.
    n_heads : int
        Number of attention heads per layer.
    output_dim : int
        Output dimension (2 * p_layers).
    """

    def __init__(
        self,
        node_features: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        output_dim: int = 4,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GATConv

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer: node_features → hidden_dim * n_heads
        self.convs.append(GATConv(node_features, hidden_dim, heads=n_heads, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_dim * n_heads))

        # Middle layers: hidden_dim * n_heads → hidden_dim * n_heads
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(hidden_dim * n_heads, hidden_dim, heads=n_heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim * n_heads))

        # Last conv: hidden_dim * n_heads → hidden_dim (concat=False, average heads)
        self.convs.append(GATConv(hidden_dim * n_heads, hidden_dim, heads=1, concat=False))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Readout MLP
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Predict θ_pred from graph data using attention."""
        x, edge_index = data.x, data.edge_index
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for conv, bn in zip(self.convs, self.bns, strict=False):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)

        x = global_mean_pool(x, batch)
        return self.head(x)
