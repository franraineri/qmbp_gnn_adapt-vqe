"""Unified MPNN — Type-aware GNN for Qracle-style Hamiltonian+Circuit graphs.

Extends BondResolvedMPNN with type-aware message passing: different node types
(qubit, ZZ gate, RX gate) use distinct message functions, enabling the GNN to
exploit the heterogeneous structure of the unified graph.

Key differences from BondResolvedMPNN:
  1. Type-conditioned message passing via per-type GINConv layers
  2. Gate-node readout: θ_zz predicted directly from ZZ gate embeddings
  3. Qubit-node readout: θ_x predicted directly from qubit node embeddings
  4. No concatenation trick for edge prediction — gate nodes carry the info

Architecture (Qracle-inspired, adapted for bond-resolved HVA):
  Input: Unified graph from build_unified_bond_resolved_graph()
  Type-Conditioned MP: k layers of {GINConv_qubit, GINConv_gate} with shared state
  Readout:
    - θ_x: qubit_embedding → node_head → [N values]
    - θ_zz: zz_gate_embedding → gate_head → [n_edges values]

References
----------
- Zhang et al. (2025) "Qracle" arXiv:2505.01236
- Integration plan: internal/documentation/next-steps/04_qracle_unified_graph.md
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GINConv

from qmbp_simulation.predictors.unified_graph import (
    NODE_TYPE_QUBIT,
    NODE_TYPE_RX_GATE,
    NODE_TYPE_ZZ_GATE,
    UNIFIED_NODE_FEATURES,
)

logger = logging.getLogger(__name__)


class UnifiedMPNN(nn.Module):
    """Type-aware MPNN for Qracle-style unified Hamiltonian+Circuit graphs.

    This model processes the heterogeneous unified graph with type-conditioned
    message passing. Unlike BondResolvedMPNN which applies uniform GINConv
    to all nodes, UnifiedMPNN routes messages through type-specific pathways:

    - **Shared backbone**: All nodes participate in the same GINConv layers
      (messages flow freely between qubit, ZZ gate, and RX gate nodes)
    - **Type embedding**: A learned type embedding is added to node features
      at input, giving the GNN explicit node-type awareness
    - **Type-aware readout**: Predictions extracted from specific node types:
      - θ_zz[i]: from ZZ gate node i (one per lattice edge per layer)
      - θ_x[j]: from qubit node j (one per qubit per layer)

    This is more principled than BondResolvedMPNN's approach of masking at
    readout, because here the ZZ gate nodes LEARN to represent θ_zz during
    message passing (they aggregate information from their connected qubits).

    Parameters
    ----------
    node_features : int
        Input node features from unified graph (default 4: UNIFIED_NODE_FEATURES).
    hidden_dim : int
        Hidden dimension for GNN layers.
    n_layers : int
        Number of GINConv message passing layers.
    norm_type : str
        Normalization: 'none' (default, best cross-N), 'batch', or 'layer'.
    dropout : float
        Dropout rate in prediction heads.
    type_embedding_dim : int
        Learned type embedding dimension (added to hidden representation).
        Set to 0 to disable type embeddings (pure positional encoding via
        the node_type feature in x[:, 3]).
    gate_readout : bool
        If True (default), predict θ_zz from ZZ gate node embeddings.
        If False, use edge concatenation (BondResolvedMPNN-style fallback).
    """

    def __init__(
        self,
        node_features: int = UNIFIED_NODE_FEATURES,
        hidden_dim: int = 256,
        n_layers: int = 3,
        norm_type: str = "none",
        dropout: float = 0.1,
        type_embedding_dim: int = 16,
        gate_readout: bool = True,
    ) -> None:
        super().__init__()

        if norm_type not in ("batch", "layer", "none"):
            raise ValueError(f"norm_type must be 'batch', 'layer', or 'none'. Got: {norm_type!r}")

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.norm_type = norm_type
        self.dropout_rate = dropout
        self.type_embedding_dim = type_embedding_dim
        self.gate_readout = gate_readout

        # ── Type embedding (learned) ─────────────────────────────────
        n_types = 3  # qubit=0, zz_gate=1, rx_gate=2
        if type_embedding_dim > 0:
            self.type_emb = nn.Embedding(n_types, type_embedding_dim)
            effective_input_dim = node_features + type_embedding_dim
        else:
            self.type_emb = None
            effective_input_dim = node_features

        # ── Message passing backbone ─────────────────────────────────
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        def _make_norm(dim: int) -> nn.Module:
            if norm_type == "batch":
                return nn.BatchNorm1d(dim)
            elif norm_type == "layer":
                return nn.LayerNorm(dim)
            return nn.Identity()

        # First layer: effective_input_dim → hidden_dim
        mlp0 = nn.Sequential(
            nn.Linear(effective_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINConv(mlp0))
        self.norms.append(_make_norm(hidden_dim))

        # Subsequent layers: hidden_dim → hidden_dim
        for _ in range(n_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(_make_norm(hidden_dim))

        # ── Readout heads ────────────────────────────────────────────
        # θ_x: per-qubit prediction from qubit node embeddings
        self.qubit_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        if gate_readout:
            # θ_zz: per-gate prediction from ZZ gate node embeddings
            # Each ZZ gate node represents one bond — predict its θ_zz directly
            self.gate_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.edge_head = None
        else:
            # Fallback: edge concatenation (same as BondResolvedMPNN)
            self.gate_head = None
            self.edge_head = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, data: Data) -> torch.Tensor:
        """Predict bond-resolved θ = [θ_zz_edges, θ_x_nodes] × p_layers.

        Parameters
        ----------
        data : Data
            Unified graph from build_unified_bond_resolved_graph() with:
            - x: [total_nodes, UNIFIED_NODE_FEATURES]
            - edge_index: [2, total_edges]
            - node_type: [total_nodes] with values 0/1/2
            - edge_list: [n_edges, 2] unique lattice edges (qubit indices)
            - n_qubit_nodes: int
            - n_edges_unique: int

        Returns
        -------
        torch.Tensor of shape [1, (n_edges + n_qubit_nodes) * p_layers]
            For p=1: [1, n_edges + N]
            For p>1: [1, (n_edges + N) * p] with layout matching theta_opt

        Raises
        ------
        RuntimeError
            If data is missing required attributes (node_type, edge_list, etc.)
        """
        x = data.x
        edge_index = data.edge_index

        # Input validation
        if not hasattr(data, "node_type") or data.node_type is None:
            raise RuntimeError(
                "UnifiedMPNN requires unified graphs with node_type attribute. "
                "Use build_unified_bond_resolved_graph(include_circuit_nodes=True)."
            )
        node_type = data.node_type

        # ── Prepend type embedding if enabled ────────────────────────
        if self.type_emb is not None:
            type_emb = self.type_emb(node_type)  # [total_nodes, type_emb_dim]
            x = torch.cat([x, type_emb], dim=-1)  # [total_nodes, feat + emb]

        # ── Message passing (all nodes participate) ──────────────────
        for conv, norm in zip(self.convs, self.norms, strict=False):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)

        # ── Extract embeddings by node type ──────────────────────────
        qubit_mask = node_type == NODE_TYPE_QUBIT
        zz_gate_mask = node_type == NODE_TYPE_ZZ_GATE
        rx_gate_mask = node_type == NODE_TYPE_RX_GATE

        x_qubit = x[qubit_mask]  # [N, hidden_dim]
        x_zz_gates = x[zz_gate_mask]  # [n_edges * p_layers, hidden_dim]
        x_rx_gates = x[rx_gate_mask]  # [N * p_layers, hidden_dim]

        n_edges = data.n_edges_unique
        N = data.n_qubit_nodes

        # Infer p_layers from gate node counts
        n_zz_total = x_zz_gates.shape[0]
        p_layers = n_zz_total // n_edges if n_edges > 0 else 1

        # ── θ_zz: predict from ZZ gate embeddings ────────────────────
        if self.gate_readout:
            # Direct gate readout: each ZZ gate node → one θ_zz value.
            theta_zz = self.gate_head(x_zz_gates).squeeze(-1)  # [n_edges * p]
        else:
            # Fallback: concatenate qubit embeddings at edge endpoints
            # For p>1, repeat edge_list predictions per layer
            edge_list = data.edge_list  # [n_edges, 2]
            src_emb = x_qubit[edge_list[:, 0]]
            tgt_emb = x_qubit[edge_list[:, 1]]
            edge_emb = torch.cat([src_emb, tgt_emb], dim=-1)
            theta_zz_layer = self.edge_head(edge_emb).squeeze(-1)  # [n_edges]
            # For p>1 with edge fallback, repeat (qubit embeddings don't change per layer)
            theta_zz = theta_zz_layer.repeat(p_layers)

        # ── θ_x: predict from RX gate embeddings (p>1) or qubits (p=1) ─
        if p_layers > 1 and x_rx_gates.shape[0] > 0:
            # For p>1: use RX gate node embeddings (one per qubit per layer)
            theta_x = self.qubit_head(x_rx_gates).squeeze(-1)  # [N * p]
        else:
            # For p=1: use qubit node embeddings directly
            theta_x = self.qubit_head(x_qubit).squeeze(-1)  # [N]

        # ── Concatenate: [θ_zz_layer1, θ_x_layer1, θ_zz_layer2, ...] ─
        # Target layout from build_unified_bond_resolved_graph:
        #   [θ_zz_all_layers, θ_x_all_layers] (not interleaved)
        # = [zz_l1(n_e), zz_l2(n_e), ..., x_l1(N), x_l2(N), ...]
        # Actually the layout is: (n_edges + N) * p_layers as flat vector
        # with first n_edges*p = ZZ params, then N*p = X params
        return torch.cat([theta_zz, theta_x], dim=-1).unsqueeze(0)


def train_unified_mpnn(
    model: UnifiedMPNN,
    dataset: list[Data],
    n_epochs: int = 6000,
    lr: float = 1e-3,
    patience: int = 300,
    seed: int = 42,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.2,
    mse_floor: float = 0.0,
    _layerwise_lr: dict | None = None,
) -> dict:
    """Train the UnifiedMPNN with per-edge/per-node MSE loss.

    Uses the same training pattern as train_bond_resolved_mpnn but with
    weight decay by default (unified graphs are larger → higher overfitting
    risk) and train/val split for generalization monitoring.

    Parameters
    ----------
    model : UnifiedMPNN
        Model to train.  Must NOT be in eval mode — this function calls
        ``model.train()`` at the start, but callers should pass a model that
        is ready for gradient updates.
    dataset : list[Data]
        Training graphs from build_unified_bond_resolved_graph() with y targets.
        Every graph must have attributes: x, edge_index, node_type, n_edges_unique,
        n_qubit_nodes, y.  Graphs with mismatched y length are skipped with a warning.
    n_epochs : int
        Maximum training epochs.
    lr : float
        Initial learning rate.
    patience : int
        ReduceLROnPlateau patience.
    seed : int
        Random seed.
    weight_decay : float
        L2 regularization (default 1e-4, higher than BondResolvedMPNN).
    val_fraction : float
        Fraction held out for validation (0.0 disables).
    mse_floor : float
        If > 0, stop early when training MSE drops below this threshold and
        at least 50 epochs have elapsed.  Set to 0.0 (default) to disable.
        Useful for fine-tuning when the model is already near-optimal.
    _layerwise_lr : dict | None
        Internal parameter injected by fine_tune_unified_mpnn to override
        per-parameter-group LRs for layer-wise decay.  Format::

            {
                "early_conv": 0.1,   # multiplier on `lr` for early layers
                "last_conv":  0.5,   # multiplier for last conv layer
                "heads":      1.0,   # multiplier for readout heads
                "type_emb":   0.3,   # multiplier for type embedding
            }

        If None, a single AdamW optimizer is used with uniform ``lr``.

    Returns
    -------
    dict with training metrics matching train_bond_resolved_variant output format.
    """
    import torch.nn.functional as F

    # ── Input validation ────────────────────────────────────────────────────
    if not isinstance(model, UnifiedMPNN):
        raise TypeError(
            f"Expected UnifiedMPNN, got {type(model).__name__}. "
            "Use train_bond_resolved_mpnn for BondResolvedMPNN."
        )
    if len(dataset) < 3:
        raise ValueError(
            f"Need ≥3 training points, got {len(dataset)}. "
            "Collect more VQE data before training."
        )
    # Validate graph attributes on first sample (fast check, fail early)
    _required_attrs = ("x", "edge_index", "node_type", "n_edges_unique", "n_qubit_nodes", "y")
    _sample = dataset[0]
    _missing = [a for a in _required_attrs if not hasattr(_sample, a)]
    if _missing:
        raise ValueError(
            f"Dataset graphs missing required attributes: {_missing}. "
            "Use build_unified_bond_resolved_graph(include_circuit_nodes=True)."
        )
    if mse_floor < 0:
        raise ValueError(f"mse_floor must be ≥ 0, got {mse_floor}.")

    torch.manual_seed(seed)

    # Train/val split
    rng = np.random.default_rng(seed)
    n_total = len(dataset)
    n_val = int(n_total * val_fraction) if val_fraction > 0 else 0
    if n_val > 0 and n_total - n_val < 3:
        n_val = 0

    if n_val > 0:
        indices = rng.permutation(n_total)
        val_indices = set(indices[:n_val].tolist())
        train_dataset = [dataset[i] for i in range(n_total) if i not in val_indices]
        val_dataset = [dataset[i] for i in val_indices]
    else:
        train_dataset = dataset
        val_dataset = []

    # ── Optimizer: layer-wise LR or uniform ────────────────────────────────
    if _layerwise_lr is not None:
        early_lr = lr * _layerwise_lr.get("early_conv", 1.0)
        last_lr = lr * _layerwise_lr.get("last_conv", 1.0)
        head_lr = lr * _layerwise_lr.get("heads", 1.0)
        emb_lr = lr * _layerwise_lr.get("type_emb", 1.0)

        param_groups: list[dict] = []
        # Early conv layers (all but last)
        if len(model.convs) > 1:
            param_groups.append({
                "params": list(model.convs[:-1].parameters()),
                "lr": early_lr,
                "name": "early_convs",
            })
        # Last conv layer
        param_groups.append({
            "params": list(model.convs[-1].parameters()),
            "lr": last_lr,
            "name": "last_conv",
        })
        # Readout heads
        head_params = list(model.qubit_head.parameters())
        if model.gate_head is not None:
            head_params += list(model.gate_head.parameters())
        if model.edge_head is not None:
            head_params += list(model.edge_head.parameters())
        param_groups.append({"params": head_params, "lr": head_lr, "name": "heads"})
        # Type embedding
        if model.type_emb is not None:
            param_groups.append({
                "params": list(model.type_emb.parameters()),
                "lr": emb_lr,
                "name": "type_emb",
            })
        # Norms (if any)
        norm_params = [p for n in model.norms for p in n.parameters()]
        if norm_params:
            param_groups.append({"params": norm_params, "lr": last_lr, "name": "norms"})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        logger.debug(
            "  Layer-wise optimizer: early_conv lr=%.2e, last_conv lr=%.2e, "
            "heads lr=%.2e, type_emb lr=%.2e",
            early_lr, last_lr, head_lr, emb_lr,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience, factor=0.5, min_lr=1e-6
    )

    mse_history: list[float] = []
    val_mse_history: list[float] = []
    zz_loss_history: list[float] = []
    x_loss_history: list[float] = []
    stop_reason = "completed"

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        total_zz = 0.0
        total_x = 0.0
        n_skipped = 0

        for data in train_dataset:
            optimizer.zero_grad()

            # Guard: skip graphs with missing or malformed y tensor
            if not hasattr(data, "y") or data.y is None or len(data.y) == 0:
                n_skipped += 1
                continue

            pred = model(data).squeeze(0)
            target = data.y
            n_e = data.n_edges_unique

            # Infer p_layers from target size vs n_edges + N
            N = data.n_qubit_nodes
            p_inferred = len(target) // (n_e + N) if (n_e + N) > 0 else 1

            # Guard: skip if predicted output length doesn't match target
            expected_len = n_e * p_inferred + N * p_inferred
            if len(pred) != expected_len:
                logger.warning(
                    "  Skipping graph: pred len=%d ≠ expected %d "
                    "(n_e=%d, N=%d, p=%d). Check build_unified_bond_resolved_graph.",
                    len(pred), expected_len, n_e, N, p_inferred,
                )
                n_skipped += 1
                continue

            # Prediction layout: [θ_zz_all_layers, θ_x_all_layers]
            # Target layout: [θ_zz_l1, θ_x_l1, θ_zz_l2, θ_x_l2, ...]
            # Need to rearrange target to match prediction layout
            if p_inferred > 1:
                # Reshape target from interleaved to grouped
                target_layers = target.reshape(p_inferred, n_e + N)
                target_zz = target_layers[:, :n_e].reshape(-1)  # all ZZ
                target_x = target_layers[:, n_e:].reshape(-1)   # all X
                target_grouped = torch.cat([target_zz, target_x])
            else:
                target_grouped = target

            n_zz_total = n_e * p_inferred
            loss_zz = F.mse_loss(pred[:n_zz_total], target_grouped[:n_zz_total])
            loss_x = F.mse_loss(pred[n_zz_total:], target_grouped[n_zz_total:])
            loss = loss_zz + loss_x

            # ── Weighted loss: scale by sample_weight (quality tier) ──────
            # verified=1.0, approximate=0.7, unverified=0.5
            # This makes the model learn more from high-quality VQE data.
            if hasattr(data, "sample_weight") and data.sample_weight is not None:
                from qmbp_simulation.analysis.metrics import SAMPLE_WEIGHT_MIN, SAMPLE_WEIGHT_MAX
                w = data.sample_weight.item()
                # Guard: clamp weight to valid range
                if w < SAMPLE_WEIGHT_MIN or w > SAMPLE_WEIGHT_MAX:
                    logger.warning(
                        "  sample_weight=%.2f out of range [%.1f, %.1f] — clamping",
                        w, SAMPLE_WEIGHT_MIN, SAMPLE_WEIGHT_MAX,
                    )
                    w = max(SAMPLE_WEIGHT_MIN, min(SAMPLE_WEIGHT_MAX, w))
                loss = loss * w

            # Guard: skip NaN/Inf losses (can occur with bad θ_opt data)
            if not torch.isfinite(loss):
                logger.warning("  Non-finite loss at epoch %d — skipping batch.", epoch)
                n_skipped += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_zz += loss_zz.item()
            total_x += loss_x.item()

        n_effective = len(train_dataset) - n_skipped
        if n_effective == 0:
            logger.error(
                "All %d training graphs were skipped at epoch %d. "
                "Check dataset integrity.",
                len(train_dataset), epoch,
            )
            stop_reason = "all_graphs_skipped"
            break

        avg_loss = total_loss / n_effective
        mse_history.append(avg_loss)
        zz_loss_history.append(total_zz / n_effective)
        x_loss_history.append(total_x / n_effective)
        scheduler.step(avg_loss)

        # Validation every 50 epochs
        if val_dataset and epoch % 50 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in val_dataset:
                    pred = model(data).squeeze(0)
                    target = data.y
                    n_e = data.n_edges_unique
                    N_v = data.n_qubit_nodes
                    p_v = len(target) // (n_e + N_v) if (n_e + N_v) > 0 else 1
                    if p_v > 1:
                        tl = target.reshape(p_v, n_e + N_v)
                        tg = torch.cat([tl[:, :n_e].reshape(-1), tl[:, n_e:].reshape(-1)])
                    else:
                        tg = target
                    n_zz_v = n_e * p_v
                    loss_v = F.mse_loss(pred[:n_zz_v], tg[:n_zz_v]) + \
                             F.mse_loss(pred[n_zz_v:], tg[n_zz_v:])
                    val_loss += loss_v.item()
            val_mse_history.append(val_loss / len(val_dataset))
            model.train()

            # ── Overfitting detection: val_mse rising while train_mse falling ──
            # If val_mse increased for 3 consecutive checks while train_mse
            # decreased, the model is overfitting → early stop to preserve
            # generalization. This is more granular than LR exhaustion.
            if len(val_mse_history) >= 4 and epoch > 200:
                val_recent = val_mse_history[-3:]
                val_rising = all(val_recent[i] > val_recent[i-1] for i in range(1, 3))
                train_falling = avg_loss < mse_history[-50] if len(mse_history) > 50 else False
                if val_rising and train_falling:
                    logger.info(
                        "  Early stop at epoch %d: overfitting detected "
                        "(val_mse rising for 3 checks: %.2e → %.2e → %.2e, "
                        "train_mse=%.2e still decreasing)",
                        epoch + 1,
                        val_mse_history[-3], val_mse_history[-2], val_mse_history[-1],
                        avg_loss,
                    )
                    stop_reason = "overfitting_detected"
                    break

        if (epoch + 1) % 1000 == 0:
            logger.info(
                "  Epoch %d: MSE=%.2e (ZZ=%.2e, X=%.2e)",
                epoch + 1, avg_loss,
                total_zz / n_effective,
                total_x / n_effective,
            )

        # Early stopping: mse_floor reached (model already excellent)
        if mse_floor > 0 and avg_loss < mse_floor and epoch >= 50:
            logger.info(
                "  Early stop at epoch %d: MSE=%.2e < floor=%.2e",
                epoch + 1, avg_loss, mse_floor,
            )
            stop_reason = "mse_floor_reached"
            break

        # Early stopping on LR exhaustion
        if optimizer.param_groups[0]["lr"] <= 1e-6 and epoch > 500:
            stop_reason = "lr_exhausted"
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
                N_v = data.n_qubit_nodes
                p_v = len(target) // (n_e + N_v) if (n_e + N_v) > 0 else 1
                if p_v > 1:
                    tl = target.reshape(p_v, n_e + N_v)
                    tg = torch.cat([tl[:, :n_e].reshape(-1), tl[:, n_e:].reshape(-1)])
                else:
                    tg = target
                n_zz_v = n_e * p_v
                loss_v = F.mse_loss(pred[:n_zz_v], tg[:n_zz_v]) + \
                         F.mse_loss(pred[n_zz_v:], tg[n_zz_v:])
                val_loss += loss_v.item()
        final_val_mse = val_loss / len(val_dataset)

    final_mse = mse_history[-1] if mse_history else float("inf")
    gen_gap = (final_val_mse - final_mse) if final_val_mse is not None else None

    # ── Log weight distribution (quality tier visibility) ─────────────────
    weight_counts = {"verified (1.0)": 0, "augmented (0.8)": 0, "approximate (0.7)": 0, "unverified (0.5)": 0, "other": 0}
    for d in train_dataset:
        if hasattr(d, "sample_weight") and d.sample_weight is not None:
            w = round(float(d.sample_weight.item()), 1)
            if w == 1.0:
                weight_counts["verified (1.0)"] += 1
            elif w == 0.8:
                weight_counts["augmented (0.8)"] += 1
            elif w == 0.7:
                weight_counts["approximate (0.7)"] += 1
            elif w == 0.5:
                weight_counts["unverified (0.5)"] += 1
            else:
                weight_counts["other"] += 1
        else:
            weight_counts["unverified (0.5)"] += 1
    non_zero = {k: v for k, v in weight_counts.items() if v > 0}
    if non_zero:
        logger.info("  Weight distribution: %s", non_zero)

    return {
        "final_mse": float(final_mse),
        "final_zz_mse": float(zz_loss_history[-1]) if zz_loss_history else float("inf"),
        "final_x_mse": float(x_loss_history[-1]) if x_loss_history else float("inf"),
        "val_mse": float(final_val_mse) if final_val_mse is not None else None,
        "generalization_gap": float(gen_gap) if gen_gap is not None else None,
        "mse_history": mse_history,
        "zz_loss_history": zz_loss_history,
        "x_loss_history": x_loss_history,
        "val_mse_history": val_mse_history,
        "n_epochs_run": len(mse_history),
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "stopped_early": stop_reason != "completed",
        "stop_reason": stop_reason,
        "weight_distribution": non_zero,
    }


# ── Fine-tuning (reuse existing model weights) ──────────────────────────


def fine_tune_unified_mpnn(
    model: UnifiedMPNN,
    dataset: list[Data],
    n_epochs: int = 1000,
    lr: float = 3e-4,
    patience: int = 150,
    seed: int = 42,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.2,
    mse_floor: float = 1e-5,
    freeze_early_layers: bool = True,
    layerwise_decay: float = 0.1,
) -> dict:
    """Fine-tune an existing UnifiedMPNN on an updated dataset.

    Unlike ``train_unified_mpnn`` (trains from scratch with 5000+ epochs),
    this function implements *genuine* fine-tuning:

    - **Starts from existing learned weights** (warm-start from pre-trained state)
    - **Layer-wise LR decay** (``freeze_early_layers=True``): early backbone
      layers receive ``lr * layerwise_decay`` to prevent catastrophic forgetting,
      while readout heads receive the full ``lr`` for task adaptation.
    - **Lower global LR** (3e-4 vs 1e-3) to make small targeted corrections.
    - **Real mse_floor early-stop**: training actually terminates when MSE
      drops below ``mse_floor`` (previously this was only a post-hoc flag).
    - **Fewer epochs** (1000 vs 6000) — converges faster from a good init.

    Use this when the training dataset has been incrementally updated
    (e.g. a few new VQE-refined points added) and a full retrain from
    scratch would be wasteful.

    Parameters
    ----------
    model : UnifiedMPNN
        Pre-trained model to fine-tune IN PLACE.  Will be set to train mode
        at entry; caller should call ``model.eval()`` after if needed.
    dataset : list[Data]
        Updated training dataset (should be a superset of original training data).
    n_epochs : int
        Maximum fine-tuning epochs (default 1000).
    lr : float
        Base learning rate for heads and last backbone layer (default 3e-4).
    patience : int
        ReduceLROnPlateau patience (default 150).
    seed : int
        Random seed for reproducibility.
    weight_decay : float
        L2 regularization strength.
    val_fraction : float
        Fraction held out for generalization monitoring.
    mse_floor : float
        Stop training when MSE drops below this value (real early-stop, not
        just a diagnostic flag).  Set to 0.0 to disable.
    freeze_early_layers : bool
        If True (default), apply layer-wise LR decay: early backbone layers
        receive ``lr * layerwise_decay``, last backbone layer receives
        ``lr * 0.5``, and readout heads receive the full ``lr``.
        This is the principal protection against catastrophic forgetting.
    layerwise_decay : float
        Multiplier applied to ``lr`` for early backbone layers when
        ``freeze_early_layers=True`` (default 0.1 = 10× slower than heads).

    Returns
    -------
    dict
        Training metrics including:

        - ``final_mse``, ``val_mse``, ``generalization_gap``
        - ``n_epochs_run``, ``stopped_early``, ``stop_reason``
        - ``initial_mse``: MSE at epoch 0 (before any weight updates)
        - ``improvement_ratio``: ``final_mse / initial_mse`` (< 1 = improved)
        - ``mode``: always ``"fine_tune"``
        - ``notes``: ``"minimal_improvement"`` / ``"below_mse_floor"`` / ``"improved"``
        - ``layerwise_lr_used``: bool indicating whether layer-wise LR was applied
    """
    if not isinstance(model, UnifiedMPNN):
        raise TypeError(
            f"Expected UnifiedMPNN, got {type(model).__name__}."
        )
    if len(dataset) < 3:
        raise ValueError(
            f"fine_tune_unified_mpnn needs ≥3 points, got {len(dataset)}."
        )
    if layerwise_decay <= 0 or layerwise_decay > 1:
        raise ValueError(
            f"layerwise_decay must be in (0, 1], got {layerwise_decay}."
        )

    # Build layer-wise LR dict for train_unified_mpnn
    _lw_lr: dict | None = None
    if freeze_early_layers and len(model.convs) >= 1:
        _lw_lr = {
            "early_conv": layerwise_decay,   # early backbone layers: lr * decay
            "last_conv": 0.5,                # last backbone layer: lr * 0.5
            "heads": 1.0,                    # readout heads: full lr
            "type_emb": layerwise_decay * 2, # type embedding: slightly faster than early
        }
        logger.info(
            "  Fine-tune: layer-wise LR — early=%.2e, last=%.2e, heads=%.2e",
            lr * layerwise_decay, lr * 0.5, lr,
        )
    else:
        logger.info("  Fine-tune: uniform LR=%.2e (freeze_early_layers=False)", lr)

    # Run training with fine-tuning hyperparameters and real mse_floor
    result = train_unified_mpnn(
        model=model,
        dataset=dataset,
        n_epochs=n_epochs,
        lr=lr,
        patience=patience,
        seed=seed,
        weight_decay=weight_decay,
        val_fraction=val_fraction,
        mse_floor=mse_floor,
        _layerwise_lr=_lw_lr,
    )

    # Enrich result with fine-tuning diagnostics
    mse_history = result.get("mse_history", [])
    initial_mse = mse_history[0] if mse_history else float("inf")
    final_mse = result.get("final_mse", float("inf"))

    result["initial_mse"] = float(initial_mse)
    result["improvement_ratio"] = (
        final_mse / initial_mse if initial_mse > 0 else 1.0
    )
    result["mode"] = "fine_tune"
    result["layerwise_lr_used"] = _lw_lr is not None

    # Categorize outcome
    if result.get("stop_reason") == "overfitting_detected":
        result["notes"] = "overfitting_stopped"
        logger.info(
            "  Fine-tune: stopped early due to overfitting at epoch %d. "
            "Val MSE was rising while train MSE decreased.",
            result.get("n_epochs_run", 0),
        )
    elif result["improvement_ratio"] > 0.95:
        result["notes"] = "minimal_improvement"
        logger.info(
            "  Fine-tune: minimal improvement (ratio=%.3f). "
            "Model may already be near-optimal for this dataset.",
            result["improvement_ratio"],
        )
    elif result.get("stop_reason") == "mse_floor_reached":
        result["notes"] = "below_mse_floor"
        logger.info(
            "  Fine-tune: MSE below floor (%.2e). Stopped early at epoch %d.",
            mse_floor, result.get("n_epochs_run", 0),
        )
    else:
        result["notes"] = "improved"
        logger.info(
            "  Fine-tune: MSE %.2e → %.2e (ratio=%.3f, %d epochs).",
            initial_mse, final_mse,
            result["improvement_ratio"],
            result.get("n_epochs_run", 0),
        )

    return result


def should_retrain(
    n_new_points: int,
    current_pass_rate: float,
    prev_pass_rate: float,
    dataset_size: int,
    *,
    min_new_fraction: float = 0.05,
    min_new_points: int = 1,
) -> tuple[bool, str]:
    """Decide whether retraining the MPNN is worthwhile.

    Implements the "skip retrain if no improvement" heuristic (Fix A)
    with safety checks to avoid skipping when it matters.

    Parameters
    ----------
    n_new_points : int
        Number of newly refined VQE points added to the dataset.
    current_pass_rate : float
        Current iteration pass rate (after evaluation).
    prev_pass_rate : float
        Previous iteration pass rate.
    dataset_size : int
        Total number of points in the training dataset.
    min_new_fraction : float
        Minimum fraction of new data to justify retrain (default 5%).
    min_new_points : int
        Minimum absolute new points to justify retrain (default 1).

    Returns
    -------
    tuple[bool, str]
        (should_retrain, reason) where reason explains the decision.

    Examples
    --------
    >>> should_retrain(0, 0.69, 0.69, 45)
    (False, "no_new_data")
    >>> should_retrain(3, 0.75, 0.69, 45)
    (True, "pass_rate_improved")
    >>> should_retrain(1, 0.69, 0.69, 200)
    (False, "below_min_fraction")
    """
    if n_new_points == 0:
        return False, "no_new_data"

    if n_new_points < min_new_points:
        return False, "below_min_points"

    # Clamp pass rates to valid range (prevents garbage-in from averaging bugs)
    current_pass_rate = max(0.0, min(1.0, current_pass_rate))
    prev_pass_rate = max(0.0, min(1.0, prev_pass_rate))

    # Always retrain if pass_rate meaningfully improved (>= +3pp).
    # This check MUST come BEFORE the fraction threshold so that real
    # improvements on large datasets are not silently skipped.
    if current_pass_rate > prev_pass_rate + 0.03:
        return True, "pass_rate_improved"

    new_fraction = n_new_points / max(dataset_size, 1)
    if new_fraction < min_new_fraction: #and dataset_size > 20:
        # New data is a negligible fraction of an already-large dataset —
        # the model has seen much more data than what was added, so retraining
        # is unlikely to change predictions meaningfully.
        return False, "below_min_fraction"

    # Retrain if we have meaningful new data
    return True, "new_data_available"


# ── Checkpoint save/load (same pattern as mpnn.py) ───────────────────────


def save_unified_checkpoint(
    model: UnifiedMPNN,
    path: str,
    training_metadata: dict | None = None,
) -> None:
    """Save UnifiedMPNN with architecture metadata for reconstruction.

    Uses the same envelope format as save_mpnn_checkpoint for consistency.
    Checkpoints are interoperable with the artifact_store system.

    Parameters
    ----------
    model : UnifiedMPNN
        Trained model to save.
    path : str
        File path for the checkpoint (.pt file).
    training_metadata : dict | None
        Optional training info (epochs, loss, dataset details).

    Raises
    ------
    ValueError
        If model weights contain NaN/Inf (corrupted model, should not be persisted).
    """
    # ── Validate model integrity before persisting ────────────────────────
    # A model with NaN weights is useless and should never be saved to disk
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        if not torch.all(torch.isfinite(param)):
            n_bad = int((~torch.isfinite(param)).sum())
            raise ValueError(
                f"Cannot save checkpoint: parameter '{name}' contains "
                f"{n_bad} non-finite values. The model is corrupted — "
                f"check training for NaN loss or gradient explosions."
            )

    from pathlib import Path as _Path
    _Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": state_dict,
            "architecture": "unified_mpnn",
            "node_features": model.node_features,
            "hidden_dim": model.hidden_dim,
            "n_layers": model.n_layers,
            "norm_type": model.norm_type,
            "dropout": model.dropout_rate,
            "type_embedding_dim": model.type_embedding_dim,
            "gate_readout": model.gate_readout,
            "training_metadata": training_metadata or {},
        },
        path,
    )
    logger.info("Saved UnifiedMPNN checkpoint to %s", path)


def load_unified_checkpoint(path: str, eval_mode: bool = True) -> UnifiedMPNN:
    """Load UnifiedMPNN from checkpoint, reconstructing architecture.

    Handles two checkpoint formats:
    - **New format** (saved by ``save_unified_checkpoint``): has ``architecture``,
      ``node_features``, ``hidden_dim``, etc. at top level.
    - **Legacy format** (saved by ``save_mpnn_checkpoint`` before the zoo fix):
      only has ``state_dict``.  Architecture is inferred from weight tensor shapes.

    Parameters
    ----------
    path : str
        Path to the checkpoint file (.pt).
    eval_mode : bool
        If True (default), set model to eval mode after loading.
        Set to False for fine-tuning (caller will call model.train()).

    Returns
    -------
    UnifiedMPNN
        Reconstructed model with loaded weights.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    RuntimeError
        If state_dict loading fails (likely architecture mismatch with legacy file).
    """
    from pathlib import Path as _Path

    if not _Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)

    # ── Infer architecture from checkpoint ───────────────────────────────
    # New format (post-fix): architecture='unified_mpnn' + full metadata
    if "architecture" in data and data["architecture"] == "unified_mpnn":
        node_features = data.get("node_features", UNIFIED_NODE_FEATURES)
        hidden_dim = data.get("hidden_dim", 256)
        n_layers = data.get("n_layers", 3)
        norm_type = data.get("norm_type", "none")
        dropout = data.get("dropout", 0.1)
        type_embedding_dim = data.get("type_embedding_dim", 16)
        gate_readout = data.get("gate_readout", True)
    elif (
        "architecture" in data
        and data["architecture"] == "ginconv"
        and "hidden_dim" in data
        and any("qubit_head" in k for k in data.get("state_dict", {}).keys())
    ):
        # Intermediate format: saved by save_mpnn_checkpoint with top-level
        # metadata keys (hidden_dim, node_features, etc.) but architecture marked as 'ginconv' instead of 'unified_mpnn'. These keys ARE
        # correct — trust them directly rather than inferring from weights.
        hidden_dim = data.get("hidden_dim", 256)
        node_features = data.get("node_features", UNIFIED_NODE_FEATURES)
        n_layers = data.get("n_layers", 3)
        norm_type = data.get("norm_type", "none")
        dropout = data.get("dropout", 0.1)
        # save_mpnn_checkpoint doesn't save type_embedding_dim — infer it
        state_dict = data.get("state_dict", {})
        if "type_emb.weight" in state_dict:
            type_embedding_dim = state_dict["type_emb.weight"].shape[1]
        else:
            type_embedding_dim = 0
        gate_readout = any("gate_head" in k for k in state_dict)
        logger.info(
            "  Loading intermediate-format UnifiedMPNN (arch='ginconv' + qubit_head). "
            "Inferred: hidden=%d, layers=%d, type_emb=%d, gate_readout=%s",
            hidden_dim, n_layers, type_embedding_dim, gate_readout,
        )
    else:
        # Legacy format: infer hidden_dim from weight shapes in state_dict
        state_dict = data.get("state_dict", data)
        # Infer hidden_dim from qubit_head first Linear layer
        if "qubit_head.0.weight" in state_dict:
            hidden_dim = state_dict["qubit_head.0.weight"].shape[1]
        elif "convs.0.nn.0.weight" in state_dict:
            hidden_dim = state_dict["convs.0.nn.0.weight"].shape[0]
        else:
            hidden_dim = 256
        # Infer n_layers from number of conv keys
        n_layers = sum(1 for k in state_dict if k.startswith("convs.") and ".nn.0.weight" in k)
        n_layers = max(n_layers, 1)
        # Infer type_embedding_dim from type_emb.weight shape (0 if absent)
        if "type_emb.weight" in state_dict:
            type_embedding_dim = state_dict["type_emb.weight"].shape[1]
        else:
            type_embedding_dim = 0
        # Infer gate_readout from presence of gate_head vs edge_head
        gate_readout = any("gate_head" in k for k in state_dict)
        # node_features from convs.0.nn.0.weight input dim
        if "convs.0.nn.0.weight" in state_dict:
            effective_input = state_dict["convs.0.nn.0.weight"].shape[1]
            node_features = effective_input - type_embedding_dim
        else:
            node_features = UNIFIED_NODE_FEATURES
        norm_type = "none"
        dropout = 0.1
        logger.info(
            "  Loading legacy UnifiedMPNN checkpoint (no architecture metadata). "
            "Inferred: hidden=%d, layers=%d, type_emb=%d, gate_readout=%s",
            hidden_dim, n_layers, type_embedding_dim, gate_readout,
        )

    model = UnifiedMPNN(
        node_features=node_features,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        norm_type=norm_type,
        dropout=dropout,
        type_embedding_dim=type_embedding_dim,
        gate_readout=gate_readout,
    )

    state_dict = data.get("state_dict", data)
    model.load_state_dict(state_dict)

    if eval_mode:
        model.eval()

    logger.info(
        "Loaded UnifiedMPNN: hidden=%d, layers=%d, type_emb=%d, gate_readout=%s",
        model.hidden_dim, model.n_layers,
        model.type_embedding_dim, model.gate_readout,
    )
    return model
