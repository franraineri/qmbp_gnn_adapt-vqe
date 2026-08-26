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
from collections.abc import Callable

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
    norm_type : str
        Normalization layer type between GNN convolutions. One of:
        - ``"batch"`` (default): BatchNorm1d — best for fixed-N training
          where all graphs have the same size. Standard GIN behavior.
        - ``"layer"``: LayerNorm — normalizes per-node across features.
          Size-invariant; recommended for cross-N generalization.
        - ``"none"``: No normalization — recommended for chain_1d cross-N
          where all nodes have identical features (BN variance=0 causes
          distortion). Validated: 0.13% ΔE/gap vs 18.5% with BN.

        .. note::
           For topologies with nodal symmetry (chain_1d), all nodes in a
           graph have identical features after message passing. BatchNorm
           accumulates running_stats that reflect graph-size differences
           rather than meaningful feature variation, causing 25-40%
           underprediction of θ_x during cross-N deployment. Use
           ``norm_type="none"`` or ``norm_type="layer"`` for cross-N.
    n_edges : int | None
        Number of lattice edges for asymmetric head split. When
        ``per_parameter_heads=True`` and ``n_edges`` is provided,
        head_zz outputs ``n_edges`` values and head_x outputs
        ``output_dim - n_edges`` values. When None, falls back to
        symmetric ``output_dim // 2`` split (backward compatible).
        Required for bond-resolved HVA where n_edges ≠ n_qubits.
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
        norm_type: str = "batch",
        n_edges: int | None = None,
        dropout_rate: float = 0.1,
        dropout_between_layers: float = 0.0,
    ) -> None:
        super().__init__()

        if norm_type not in ("batch", "layer", "none"):
            raise ValueError(f"norm_type must be 'batch', 'layer', or 'none'. Got: {norm_type!r}")

        # Store architecture attributes for checkpoint metadata
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.per_parameter_heads = per_parameter_heads
        self.use_edge_features = use_edge_features
        self.edge_feature_dim = edge_feature_dim
        self.norm_type = norm_type
        self.n_edges = n_edges
        self.dropout_rate = dropout_rate
        self.dropout_between_layers = dropout_between_layers

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Dropout between GINConv/NNConv layers (for MC-Dropout UQ)
        # When dropout_between_layers > 0, enables meaningful variance
        # estimation across the entire representation, not just the head.
        self._inter_layer_dropout = (
            nn.Dropout(dropout_between_layers) if dropout_between_layers > 0 else None
        )

        def _make_norm(dim: int) -> nn.Module:
            """Create normalization layer based on norm_type."""
            if norm_type == "batch":
                return nn.BatchNorm1d(dim)
            elif norm_type == "layer":
                return nn.LayerNorm(dim)
            else:  # "none"
                return nn.Identity()

        if use_edge_features:
            # NNConv architecture: edge features processed through learned MLP
            # First layer: node_features → hidden_dim
            edge_nn_0 = nn.Sequential(
                nn.Linear(edge_feature_dim, NNCONV_EDGE_MLP_HIDDEN),
                nn.ReLU(),
                nn.Linear(NNCONV_EDGE_MLP_HIDDEN, node_features * hidden_dim),
            )
            self.convs.append(NNConv(node_features, hidden_dim, edge_nn_0, aggr="add"))
            self.norms.append(_make_norm(hidden_dim))

            # Subsequent layers: hidden_dim → hidden_dim
            for _ in range(n_layers - 1):
                edge_nn = nn.Sequential(
                    nn.Linear(edge_feature_dim, NNCONV_EDGE_MLP_HIDDEN),
                    nn.ReLU(),
                    nn.Linear(NNCONV_EDGE_MLP_HIDDEN, hidden_dim * hidden_dim),
                )
                self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr="add"))
                self.norms.append(_make_norm(hidden_dim))
        else:
            # GINConv architecture (V6.0 default)
            # First layer: node_features → hidden_dim
            mlp0 = nn.Sequential(
                nn.Linear(node_features, hidden_dim),
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

        # Backward compatibility: expose self.bns as alias for self.norms
        # (existing code may reference model.bns for checkpoint loading)
        # Note: We don't set self.bns = self.norms because nn.Module.__setattr__
        # would register it as a duplicate ModuleList. Instead, use a property.
        # Legacy checkpoints with 'bns' keys are handled via state_dict remapping.

        # Readout head(s): global pooled vector → θ_pred
        if per_parameter_heads:
            # Separate heads for θ_zz and θ_x (physics-informed specialization).
            # For bond-resolved HVA: n_edges ZZ params + (output_dim - n_edges) X params.
            # When n_edges is None, fall back to symmetric split (backward compat).
            dim_zz = n_edges if n_edges is not None else output_dim // 2
            dim_x = output_dim - dim_zz
            self._dim_zz = dim_zz
            self._dim_x = dim_x
            self.head_zz = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, dim_zz),
            )
            self.head_x = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, dim_x),
            )
            self.head = None  # Not used in per-parameter mode
        else:
            # Single head (V6.0 default, backward compatible)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, output_dim),
            )

    @property
    def bns(self) -> nn.ModuleList:
        """Backward-compatible alias for self.norms (legacy code uses model.bns)."""
        return self.norms

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
            for conv, norm in zip(self.convs, self.norms, strict=False):
                x = conv(x, edge_index, edge_attr)
                x = norm(x)
                x = torch.relu(x)
                if self._inter_layer_dropout is not None:
                    x = self._inter_layer_dropout(x)
        else:
            for conv, norm in zip(self.convs, self.norms, strict=False):
                x = conv(x, edge_index)
                x = norm(x)
                x = torch.relu(x)
                if self._inter_layer_dropout is not None:
                    x = self._inter_layer_dropout(x)

        # Global mean pooling: lattice-agnostic fixed-size output
        x = global_mean_pool(x, batch)

        if self.per_parameter_heads:
            zz_out = self.head_zz(x)
            x_out = self.head_x(x)
            return torch.cat([zz_out, x_out], dim=-1)
        return self.head(x)  # type: ignore[misc, no-any-return]


# ── Reusable prediction utility ──────────────────────────────────────────


def predict_theta(
    model: MPNNPredictor,
    lattice: LatticeConfig,
    h_values: list[float] | np.ndarray,
    *,
    rescale_h_by_j: bool = False,
    clip_bounds: tuple[float, float] = (-np.pi, np.pi),
    extra_node_features: np.ndarray | None = None,
) -> dict[float, np.ndarray]:
    """Generate MPNN predictions for given h-values with full safety guards.

    This is the canonical prediction function — all callers (pipeline runners,
    experiments, notebooks) should use this instead of manually constructing
    graphs and calling model(). It applies:

    1. h/J rescaling (when ``rescale_h_by_j=True``)
    2. NaN/Inf guard (replaces corrupt predictions with zeros + warning)
    3. Bounds clipping (ensures θ ∈ [-π, π] for valid HVA circuits)

    Parameters
    ----------
    model : MPNNPredictor
        Trained MPNN model (will be set to eval mode).
    lattice : LatticeConfig
        Lattice specification (topology, n_qubits, J, edges).
    h_values : list[float] | np.ndarray
        Transverse field values to predict θ for.
    rescale_h_by_j : bool
        If True, use h/J as the node feature instead of h.
        Requires ``lattice.J`` to be a positive scalar.
    clip_bounds : tuple[float, float]
        Parameter bounds for clipping (default [-π, π]).
    extra_node_features : np.ndarray | None
        Additional per-point features broadcast to all nodes, shape
        [len(h_values), n_extra]. Must match what was used in
        ``build_graph_dataset(extra_node_features=...)`` during training.
        For models with uniform extra features (e.g., single J₂ value),
        pass a constant array like ``np.full((len(h_values), 1), j2_val)``.

    Returns
    -------
    dict[float, np.ndarray]
        Mapping from h-value to predicted θ array.

    Raises
    ------
    ValueError
        If rescale_h_by_j=True but J is non-scalar or non-positive.
    """
    import torch
    from torch_geometric.data import Data

    from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

    # Validate rescaling
    j_scalar = 1.0
    if rescale_h_by_j:
        if isinstance(lattice.J, np.ndarray):
            raise ValueError(
                "predict_theta: rescale_h_by_j=True requires scalar J, "
                f"but lattice.J is per-bond array."
            )
        j_scalar = float(lattice.J)
        if j_scalar <= 0:
            raise ValueError(
                f"predict_theta: rescale_h_by_j=True requires J>0, got J={j_scalar}."
            )

    # Validate extra_node_features shape
    if extra_node_features is not None:
        extra_node_features = np.asarray(extra_node_features)
        if extra_node_features.ndim == 1:
            extra_node_features = extra_node_features.reshape(-1, 1)
        if len(extra_node_features) != len(h_values):
            raise ValueError(
                f"predict_theta: extra_node_features has {len(extra_node_features)} rows "
                f"but h_values has {len(h_values)} entries. Must match."
            )

    builder = HamiltonianBuilder()
    edge_index_np, coord = builder.build_graph_data(lattice)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    model.eval()
    predictions: dict[float, np.ndarray] = {}

    with torch.no_grad():
        for idx, h in enumerate(h_values):
            h_feat_val = float(h) / j_scalar if rescale_h_by_j else float(h)
            h_feat = np.full(lattice.n_qubits, h_feat_val)
            base_features = [h_feat, coord.astype(float)]

            # Add extra features (broadcast scalar per-point values to all nodes)
            if extra_node_features is not None:
                for col in range(extra_node_features.shape[1]):
                    val = extra_node_features[idx, col]
                    base_features.append(np.full(lattice.n_qubits, float(val)))

            x = torch.tensor(
                np.stack(base_features, axis=1),
                dtype=torch.float32,
            )
            graph = Data(x=x, edge_index=edge_index)
            theta_pred = model(graph).numpy().flatten()

            # Guard: NaN/Inf → zeros
            if not np.all(np.isfinite(theta_pred)):
                n_bad = int(np.sum(~np.isfinite(theta_pred)))
                logger.warning(
                    "predict_theta: %d/%d NaN/Inf values at h=%.4f. "
                    "Replacing with zeros.",
                    n_bad, len(theta_pred), h,
                )
                theta_pred = np.where(np.isfinite(theta_pred), theta_pred, 0.0)

            # Guard: clip to valid HVA range
            if clip_bounds is not None:
                theta_pred = np.clip(theta_pred, clip_bounds[0], clip_bounds[1])

            predictions[float(h)] = theta_pred

    return predictions


# ── Graph data construction utility ──────────────────────────────────────


def build_graph_dataset(
    lattice: LatticeConfig,
    h_values: np.ndarray,
    theta_opt: np.ndarray,
    e_exact: np.ndarray,
    fidelities: np.ndarray | None = None,
    fidelity_threshold: float = 0.93,
    de_gaps: np.ndarray | None = None,
    de_gap_threshold: float = 0.20,
    include_edge_features: bool = False,
    extra_node_features: np.ndarray | None = None,
    rescale_h_by_j: bool = False,
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
        VQE fidelities for filtering (used when available, typically N≤22).
    fidelity_threshold : float
        Minimum fidelity to include in dataset (default 0.93).
    de_gaps : np.ndarray | None [n_points]
        ΔE/gap values per point. Used as quality gate when fidelities are
        unavailable (N>22, MPS backend). If both fidelities and de_gaps are
        provided, a point must pass BOTH filters.
    de_gap_threshold : float
        Maximum ΔE/gap to include in dataset (default 0.20 = 20%).
        Points with ΔE/gap > threshold are considered unconverged VQE
        and excluded from training to prevent garbage-in/garbage-out.
    include_edge_features : bool
        When True, add ``edge_attr`` tensors containing coupling J_ij.
    extra_node_features : np.ndarray | None [n_points, n_extra]
        Additional per-point features broadcast to all nodes. Each row is
        replicated across all N sites as extra columns in the node feature
        matrix. Use for model parameters like J₂, g, delta that vary across
        the dataset but are uniform within a single graph.

        Example: for frustrated TFIM with J₂ varying per point,
        pass extra_node_features=J2_values.reshape(-1, 1) to get
        node features [h_i, coord_i, J₂] (3 features per node).
    rescale_h_by_j : bool
        When True, the node h-feature is rescaled as h/J (control parameter
        divided by coupling). This makes the MPNN learn θ(h/J) instead of
        θ(h), enabling zero-shot generalization across different J couplings.
        Requires ``lattice.J`` to be a positive scalar. Default False.

    Returns
    -------
    list[Data] — filtered graph dataset for MPNN training.

    Raises
    ------
    ValueError
        If fewer than 3 data points pass the quality filter.
    """
    # ── Input shape validation ───────────────────────────────────────
    # Ensure float64 (handles legacy dtype=object arrays from NPZ)
    theta_opt = np.asarray(theta_opt, dtype=np.float64)
    if theta_opt.ndim != 2:
        raise ValueError(f"theta_opt must be 2D (n_points, n_params), got shape {theta_opt.shape}.")
    if len(h_values) != theta_opt.shape[0]:
        raise ValueError(
            f"h_values length ({len(h_values)}) must match theta_opt rows ({theta_opt.shape[0]})."
        )
    if len(e_exact) != len(h_values):
        raise ValueError(f"e_exact length ({len(e_exact)}) must match h_values ({len(h_values)}).")
    if np.any(~np.isfinite(theta_opt)):
        raise ValueError(
            "theta_opt contains NaN or Inf values. Check VQE convergence "
            "before building training dataset."
        )
    # ── h/J rescaling validation ─────────────────────────────────────
    _j_scalar: float = 1.0
    if rescale_h_by_j:
        if isinstance(lattice.J, np.ndarray):
            raise ValueError(
                "rescale_h_by_j=True requires scalar J coupling, "
                f"but lattice.J is an array (per-bond). Use extra_node_features "
                f"with h/J ratios for non-uniform couplings."
            )
        _j_scalar = float(lattice.J)
        if _j_scalar <= 0:
            raise ValueError(
                f"rescale_h_by_j=True requires positive J, got J={_j_scalar}."
            )
        logger.info(
            "  h/J rescaling enabled: h_feature = h / J (J=%.4f)", _j_scalar
        )

    logger.debug(
        "build_graph_dataset: n_qubits=%d, n_h_points=%d, theta_shape=%s, "
        "fidelity_threshold=%.2f, de_gap_threshold=%.2f, rescale_h_by_j=%s",
        lattice.n_qubits,
        len(h_values),
        theta_opt.shape,
        fidelity_threshold,
        de_gap_threshold,
        rescale_h_by_j,
    )
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
    n_fidelity_filtered = 0
    n_de_gap_filtered = 0
    n_basin_filtered = 0

    # ── Mandatory canonicalization + basin filtering ──────────────────
    # HVA circuits have gauge symmetries (period π + Z₂) that produce
    # multiple equivalent θ for the same state. Canonicalize to the
    # fundamental domain BEFORE building the dataset.
    from qmbp_simulation.utils import canonicalize_theta, filter_consistent_theta

    theta_canonical = np.array([canonicalize_theta(t) for t in theta_opt])

    # Filter periodic basin outliers (local minima with different θ)
    _, basin_mask = filter_consistent_theta(theta_canonical)
    n_basin_filtered = int((~basin_mask).sum())
    if n_basin_filtered > 0:
        logger.info(
            f"  ⚠️  Theta basin filter: removed {n_basin_filtered}/{len(theta_opt)} "
            f"outlier points (different periodic basins)."
        )

    for i, h in enumerate(h_values):
        # Basin filter: skip points in different local minima
        if not basin_mask[i]:
            continue

        # Quality gate: fidelity filter (for N≤22 with statevector)
        if fidelities is not None and fidelities[i] < fidelity_threshold:
            n_fidelity_filtered += 1
            continue

        # Quality gate: ΔE/gap filter (for all N, especially N>22 with MPS)
        if de_gaps is not None and de_gaps[i] > de_gap_threshold:
            n_de_gap_filtered += 1
            continue

        # Node features: [h_i (or h/J), coordination_number_i, ...extra...] per site
        h_value_feat = float(h) / _j_scalar if rescale_h_by_j else float(h)
        h_feat = np.full(lattice.n_qubits, h_value_feat)
        base_features = [h_feat, coord.astype(float)]

        # Add extra features (broadcast scalar per-point values to all nodes)
        if extra_node_features is not None:
            for col in range(extra_node_features.shape[1]):
                val = extra_node_features[i, col]
                base_features.append(np.full(lattice.n_qubits, float(val)))

        x = torch.tensor(
            np.stack(base_features, axis=1),
            dtype=torch.float32,
        )
        y = torch.tensor(theta_canonical[i], dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.e_exact = float(e_exact[i])
        data.h_value = float(h)
        data.h_over_j = h_value_feat  # Store rescaled feature for deployment

        # Attach edge features if available
        if include_edge_features and edge_attr is not None:
            data.edge_attr = edge_attr

        dataset.append(data)

    # Enforce quality filter constraint: need at least 3 points
    if len(dataset) < 3:
        filter_details = []
        if n_fidelity_filtered > 0:
            filter_details.append(f"fidelity<{fidelity_threshold}: {n_fidelity_filtered} removed")
        if n_de_gap_filtered > 0:
            filter_details.append(
                f"ΔE/gap>{de_gap_threshold * 100:.0f}%: {n_de_gap_filtered} removed"
            )
        raise ValueError(
            f"Fewer than 3 data points ({len(dataset)}) passed quality filters "
            f"({', '.join(filter_details) if filter_details else 'no filter active'}). "
            f"Cannot build a reliable training dataset. Consider widening the "
            f"h-grid, increasing VQE maxiter/restarts, or relaxing thresholds."
        )

    n_features = 2 + (extra_node_features.shape[1] if extra_node_features is not None else 0)
    logger.info(
        f"Built graph dataset: {len(dataset)}/{len(h_values)} points "
        f"(fidelity_filter={n_fidelity_filtered}, de_gap_filter={n_de_gap_filtered}, "
        f"basin_filter={n_basin_filtered}, "
        f"node_features={n_features}, edge_features={include_edge_features})"
    )

    # ── Data quality guards ──────────────────────────────────────────
    # Warn if h-values have no variance (MPNN can't learn h→θ mapping)
    h_vals_in_dataset = [float(d.h_value) for d in dataset]
    if len(h_vals_in_dataset) >= 3:
        h_range = max(h_vals_in_dataset) - min(h_vals_in_dataset)
        if h_range < 0.01:
            logger.warning(
                "⚠️ h-value range in dataset is tiny (%.4f). "
                "The MPNN cannot learn a meaningful h→θ mapping from "
                "nearly identical inputs. Check h_values array.",
                h_range,
            )
    # Warn if theta targets have no variance (constant θ across all h)
    theta_stack = np.array([d.y.numpy() for d in dataset])
    theta_std = float(np.std(theta_stack))
    if theta_std < 1e-4:
        logger.warning(
            "⚠️ θ targets have near-zero variance (std=%.2e). "
            "The MPNN will learn a constant function. Check VQE convergence "
            "or theta canonicalization.",
            theta_std,
        )

    return dataset


# ── Training loop ────────────────────────────────────────────────────────


def train_mpnn(
    model: MPNNPredictor | None,
    dataset: list[Data],
    n_epochs: int = 4000,
    lr: float = 1e-3,
    patience: int = 150,
    energy_val_interval: int = 50,
    energy_val_fn: Callable[..., float] | None = None,
    divergence_window: int = 5,
    divergence_threshold: float = 0.01,
    seed: int = 42,
    physics_loss_fn: Callable[..., float] | None = None,
    physics_loss_weight: float = 0.1,
    physics_loss_start_epoch: int = 500,
    weight_decay: float = 1e-4,
    grad_clip_norm: float | None = 1.0,
    # ── Model construction kwargs (used when model=None) ──
    hidden_dim: int = 64,
    n_layers: int = 3,
    output_dim: int | None = None,
    node_features: int = 2,
    dropout_rate: float = 0.1,
    norm_type: str = "batch",
    per_parameter_heads: bool = False,
    use_edge_features: bool = False,
    edge_feature_dim: int = 1,
    n_edges: int | None = None,
) -> tuple[MPNNPredictor, dict] | dict:
    """Train the MPNN with MSE loss, ReduceLROnPlateau, and optional
    energy-driven validation.

    Parameters
    ----------
    model : MPNNPredictor | None
        Pre-instantiated model to train. If None, a new MPNNPredictor is
        created using the model construction kwargs (hidden_dim, n_layers,
        etc.). This allows two calling patterns:

        1. Explicit model (backward compatible, returns dict only):
           ``result = train_mpnn(model, dataset)``

        2. Auto-create (returns tuple (model, dict)):
           ``model, result = train_mpnn(None, dataset, hidden_dim=128)``

    dataset : list[Data] — training data (already fidelity-filtered)
    n_epochs : int
    lr : float — initial learning rate
    patience : int — scheduler patience
    energy_val_interval : int — epochs between energy validation callbacks
    energy_val_fn : callable | None
        Function(theta_pred_batch, data_batch) -> list[float] of energy errors.
    divergence_window : int — window for divergence detection
    divergence_threshold : float — ΔE threshold for divergence detection.
    seed : int — random seed for reproducibility (torch + DataLoader).
    physics_loss_fn : callable | None
        Function(theta_pred_batch, data_batch) -> Tensor of physics penalty.
        Adds a regularizer that penalizes predictions violating physical
        constraints (e.g., energy > E_exact, wrong observable signs).
        Activated after `physics_loss_start_epoch` epochs.
    physics_loss_weight : float
        Weight of the physics loss relative to MSE (default 0.1).
    physics_loss_start_epoch : int
        Epoch at which to start applying physics loss (default 500).
        Allows MSE to converge first, then physics regularizes.
    grad_clip_norm : float | None
        Maximum L2 norm for gradient clipping. Prevents training divergence
        from gradient spikes (common with discontinuous θ(h) data).
        Default 1.0. Set to None to disable clipping.
    hidden_dim : int
        Hidden dimension for MPNNPredictor (used when model=None). Default 64.
    n_layers : int
        Number of GINConv layers (used when model=None). Default 3.
    output_dim : int | None
        Output dimension. If None and model=None, inferred from dataset[0].y.
    node_features : int
        Number of input node features (used when model=None). Default 2.
    dropout_rate : float
        Dropout rate for readout head (used when model=None). Default 0.1.
    norm_type : str
        Normalization type: "batch", "layer", or "none" (used when model=None).
    per_parameter_heads : bool
        Use separate ZZ/X heads (used when model=None). Default False.
    use_edge_features : bool
        Use NNConv with edge features (used when model=None). Default False.
    edge_feature_dim : int
        Edge feature dimension (used when model=None). Default 1.
    n_edges : int | None
        Number of edges for asymmetric head split (used when model=None).

    Returns
    -------
    dict (when model is provided) or tuple[MPNNPredictor, dict] (when model=None)
        When model=None: returns (model, history_dict).
        When model is provided: returns history_dict only (backward compatible).

        history_dict has keys: 'mse_history', 'energy_val_history', 'final_mse',
        'stopped_early', 'stop_reason',
        and optionally 'zz_head_loss_history', 'x_head_loss_history'
        when per_parameter_heads is enabled.

    Raises
    ------
    ValueError
        If dataset has fewer than 3 points (insufficient for training).
    """
    # ── Auto-construct model if None ──────────────────────────────────
    _model_was_none = model is None
    if model is None:
        # Infer output_dim from dataset if not provided
        if output_dim is None:
            if len(dataset) > 0 and hasattr(dataset[0], "y"):
                output_dim = dataset[0].y.shape[-1] if dataset[0].y.dim() > 0 else 1
            else:
                raise ValueError(
                    "output_dim must be specified when model=None and dataset "
                    "has no .y attribute to infer from."
                )
        # Infer node_features from dataset if possible
        if len(dataset) > 0 and hasattr(dataset[0], "x"):
            node_features = dataset[0].x.shape[-1]

        model = MPNNPredictor(
            node_features=node_features,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            output_dim=output_dim,
            per_parameter_heads=per_parameter_heads,
            use_edge_features=use_edge_features,
            edge_feature_dim=edge_feature_dim,
            norm_type=norm_type,
            n_edges=n_edges,
            dropout_rate=dropout_rate,
        )
        logger.info(
            "  Auto-created MPNNPredictor: hidden=%d, layers=%d, output=%d, "
            "norm=%s, params=%d",
            hidden_dim, n_layers, output_dim, norm_type,
            sum(p.numel() for p in model.parameters()),
        )

    # ── Dataset validation ────────────────────────────────────────────
    logger.info(
        "  🧠 train_mpnn: dataset=%d pts, epochs=%d, lr=%.1e, patience=%d, "
        "physics_loss=%s (λ=%.2f, start=%d), weight_decay=%.1e, grad_clip=%.1f",
        len(dataset),
        n_epochs,
        lr,
        patience,
        "ON" if physics_loss_fn else "OFF",
        physics_loss_weight,
        physics_loss_start_epoch,
        weight_decay,
        grad_clip_norm if grad_clip_norm is not None else 0.0,
    )
    logger.debug(
        "train_mpnn: dataset_size=%d, n_epochs=%d, lr=%.1e, patience=%d, seed=%d",
        len(dataset),
        n_epochs,
        lr,
        patience,
        seed,
    )
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

    # Seed PyTorch for reproducible weight initialization and DataLoader shuffling
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _loader_generator = torch.Generator().manual_seed(seed)

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True, generator=_loader_generator)

    # ── Compute device (GPU if available, else CPU) — portable local ↔ server ──
    from qmbp_simulation.utils.helpers import describe_device, resolve_device

    device = resolve_device()
    model = model.to(device)
    logger.info("  train_mpnn device: %s", describe_device(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            target = batch.y.view(pred.shape)
            loss = criterion(pred, target)

            # Physics-informed regularization (after warm-up period)
            if physics_loss_fn is not None and epoch >= physics_loss_start_epoch:
                phys_loss = physics_loss_fn(pred, batch)
                loss = loss + physics_loss_weight * phys_loss

            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            epoch_loss = loss.item()

        # ── NaN detection: abort early on degenerate training ────────
        if not np.isfinite(epoch_loss):
            logger.error(
                f"NaN/Inf loss detected at epoch {epoch + 1}. "
                f"Aborting training. Check input data for NaN in theta_opt "
                f"or degenerate graph features."
            )
            stopped_early = True
            stop_reason = "nan_loss"
            break

        # ── Early garbage detection: high loss after warm-up ─────────
        # If MSE is still very high after 500 epochs, the data is likely
        # corrupted (wrong feature dimensions, topology mismatch, etc.)
        if epoch == 499 and epoch_loss > 0.5:
            logger.warning(
                f"⚠️ MSE still very high ({epoch_loss:.4f}) after 500 epochs. "
                f"Possible issues: wrong node features, topology mismatch, "
                f"or corrupted theta_opt data. Training will continue but "
                f"predictions may be unreliable."
            )

        mse_history.append(epoch_loss)
        scheduler.step(epoch_loss)

        # ── Per-head loss reporting ──────────────────────────────────
        if has_per_param_heads and (epoch + 1) % energy_val_interval == 0:
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    target = batch.y.view(pred.shape)
                    p = model._dim_zz if hasattr(model, "_dim_zz") else model.output_dim // 2
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
                    batch = batch.to(device)
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
                # Guard against division-by-zero when recent_de[0] == 0
                # (perfect energy → stagnancy check is meaningless)
                ref_de = abs(recent_de[0]) if abs(recent_de[0]) > 1e-12 else 1e-12
                de_stagnant = all(
                    abs(recent_de[k] - recent_de[k - 1]) < 0.01 * ref_de
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

    # Return model to CPU so downstream inference (CPU graphs) and checkpoint
    # saving stay device-agnostic; free GPU memory for the next job.
    model = model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

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

    # Return (model, result) when model was auto-created (model=None),
    # otherwise return just result for backward compatibility with all
    # existing callers that pass a pre-built model.
    if _model_was_none:
        return model, result
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
            "norm_type": getattr(model, "norm_type", "batch"),
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
    data = torch.load(path, map_location="cpu", weights_only=False)  # nosec: trusted checkpoint

    # Handle legacy checkpoints without metadata
    if "state_dict" not in data:
        # Assume it's a raw state_dict (V6.0 format)
        logger.warning(
            "Loading legacy checkpoint without architecture metadata. "
            "Assuming single-head GINConv (V6.0 defaults)."
        )
        model = MPNNPredictor()
        # Remap legacy keys if needed
        remapped = {}
        for key, value in data.items():
            new_key = key.replace("bns.", "norms.", 1) if key.startswith("bns.") else key
            remapped[new_key] = value
        model.load_state_dict(remapped)
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
        norm_type=data.get("norm_type", "batch"),
    )

    # Remap legacy state_dict keys: "bns.*" → "norms.*"
    state_dict = data["state_dict"]
    remapped = {}
    for key, value in state_dict.items():
        new_key = key.replace("bns.", "norms.", 1) if key.startswith("bns.") else key
        remapped[new_key] = value

    model.load_state_dict(remapped)
    logger.info(
        f"Loaded MPNNCheckpoint: arch={data['architecture']}, "
        f"per_param_heads={data.get('per_parameter_heads', False)}, "
        f"edge_features={data.get('use_edge_features', False)}, "
        f"norm_type={data.get('norm_type', 'batch')}"
    )
    return model


# ── BondResolvedMPNN: Per-node/per-edge prediction for cross-N transfer ──────


class BondResolvedMPNN(nn.Module):
    """Size-agnostic MPNN for bond-resolved HVA parameter prediction.

    Unlike ``MPNNPredictor`` which uses global pooling → fixed output_dim,
    this model predicts **per-node** (θ_x) and **per-edge** (θ_zz) values,
    allowing it to generalize across system sizes without retraining.

    Architecture:
        Input: Data(x=[N, node_features], edge_index=[2, 2*n_edges])
        Message Passing: k GINConv layers (norm_type='none' for cross-N)
        Per-node head: node_embedding → θ_x_i  (N outputs)
        Per-edge head: edge_embedding → θ_zz_ij  (n_edges outputs)

    The key insight: for bond-resolved HVA on chain_1d,
    θ = [θ_zz_1, ..., θ_zz_{N-1}, θ_x_1, ..., θ_x_N] has dimension 2N-1.
    By predicting per-node and per-edge, the model naturally adapts to any N.

    Parameters
    ----------
    node_features : int
        Input node features (default 3: h_i, coord_i, N/100).
    hidden_dim : int
        Hidden dimension for GNN layers.
    n_layers : int
        Number of GINConv message passing layers.
    norm_type : str
        Normalization type ('none' recommended for cross-N on chain_1d).
    dropout : float
        Dropout rate in output heads.

    References
    ----------
    - Cross-N zero-shot (binnacle-cross-n-zero-shot.md): norm_type='none' is essential.
    - Bond-resolved HVA: N-1 ZZ bonds + N X fields = 2N-1 params.
    """

    def __init__(
        self,
        node_features: int = 3,
        hidden_dim: int = 256,
        n_layers: int = 3,
        norm_type: str = "none",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if norm_type not in ("batch", "layer", "none"):
            raise ValueError(f"norm_type must be 'batch', 'layer', or 'none'. Got: {norm_type!r}")

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.norm_type = norm_type
        self.dropout_rate = dropout

        # ── Message passing backbone ─────────────────────────────────
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        def _make_norm(dim: int) -> nn.Module:
            if norm_type == "batch":
                return nn.BatchNorm1d(dim)
            elif norm_type == "layer":
                return nn.LayerNorm(dim)
            return nn.Identity()

        # First layer: node_features → hidden_dim
        mlp0 = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
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

        # ── Per-node head: predicts θ_x per site ────────────────────
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ── Per-edge head: predicts θ_zz per bond ───────────────────
        # Uses concatenation of source + target node embeddings
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Predict bond-resolved θ = [θ_zz_edges, θ_x_nodes].

        Parameters
        ----------
        data : Data
            Must have x, edge_index, and edge_list (undirected edge pairs).
            - ``edge_list``: tensor of shape [n_edges, 2] with unique undirected
              edges (i < j). Used for θ_zz predictions.
            - ``batch``: batch indices (for batched graphs).
            - ``node_type`` (optional): tensor [total_nodes] with 0=qubit, 1=ZZ gate,
              2=RX gate. When present, predictions are masked to qubit nodes only
              (unified Hamiltonian+Circuit graph mode, Qracle-style).
            - ``n_qubit_nodes`` (optional): int, number of qubit nodes per graph.

        Returns
        -------
        torch.Tensor of shape [batch_size, n_edges + n_qubit_nodes]
            Concatenated [θ_zz_per_edge, θ_x_per_qubit] for each graph.
            When batch_size=1, shape is [1, 2N-1] for chain_1d.
        """
        x, edge_index = data.x, data.edge_index

        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # ── Message passing (all nodes participate) ──────────────────
        for conv, norm in zip(self.convs, self.norms, strict=False):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)

        # ── Determine qubit mask ─────────────────────────────────────
        # When node_type is available (unified graph), only predict on qubits.
        # Otherwise (Hamiltonian-only graph), all nodes are qubits.
        has_node_type = hasattr(data, "node_type") and data.node_type is not None
        if has_node_type:
            qubit_mask = data.node_type == 0  # [total_nodes]
            x_qubit = x[qubit_mask]  # [n_qubit_nodes_total, hidden]
        else:
            qubit_mask = None
            x_qubit = x  # all nodes are qubits

        # ── Per-node θ_x predictions (qubit nodes only) ──────────────
        theta_x = self.node_head(x_qubit).squeeze(-1)  # [n_qubit_nodes_total]

        # ── Per-edge θ_zz predictions ────────────────────────────────
        # edge_list references qubit indices (always 0..N-1), so we index
        # into x_qubit directly (qubit nodes are always the first N nodes).
        edge_list = data.edge_list  # [n_edges_total, 2] undirected

        # Defensive: verify edge_list indices are within x_qubit range
        max_edge_idx = edge_list.max().item() if edge_list.numel() > 0 else 0
        if max_edge_idx >= x_qubit.size(0):
            raise RuntimeError(
                f"edge_list contains index {max_edge_idx} but x_qubit has "
                f"only {x_qubit.size(0)} nodes. This indicates a graph "
                f"construction error (qubit nodes must be first N nodes)."
            )

        src_emb = x_qubit[edge_list[:, 0]]  # [n_edges_total, hidden]
        tgt_emb = x_qubit[edge_list[:, 1]]  # [n_edges_total, hidden]
        edge_emb = torch.cat([src_emb, tgt_emb], dim=-1)
        theta_zz = self.edge_head(edge_emb).squeeze(-1)  # [n_edges_total]

        # ── Reassemble per-graph: [θ_zz, θ_x] ───────────────────────
        # For single graph (no batching), just concatenate
        if batch.max() == 0:
            return torch.cat([theta_zz, theta_x], dim=-1).unsqueeze(0)

        # Batched: assemble per graph using qubit-level batching
        if has_node_type:
            qubit_batch = batch[qubit_mask]
        else:
            qubit_batch = batch

        outputs = []
        for g in range(batch.max().item() + 1):
            node_mask_g = qubit_batch == g
            edge_mask = data.edge_batch == g if hasattr(data, "edge_batch") else None

            if edge_mask is None:
                # Infer edge_batch from qubit batch
                edge_mask = qubit_batch[edge_list[:, 0]] == g

            g_theta_zz = theta_zz[edge_mask]
            g_theta_x = theta_x[node_mask_g]
            outputs.append(torch.cat([g_theta_zz, g_theta_x], dim=-1))

        # Pad to max length for batching (needed for loss computation)
        max_len = max(o.size(0) for o in outputs)
        padded = torch.zeros(len(outputs), max_len, device=x.device)
        for i, o in enumerate(outputs):
            padded[i, : o.size(0)] = o

        return padded


def build_bond_resolved_graph(
    lattice: LatticeConfig,
    h_value: float,
    theta_opt: np.ndarray | None = None,
    n_feature: bool = True,
) -> Data:
    """Build a single graph for bond-resolved MPNN prediction.

    Parameters
    ----------
    lattice : LatticeConfig
        Lattice with edges defined.
    h_value : float
        Transverse field value.
    theta_opt : np.ndarray | None
        Target θ_opt vector [θ_zz_edges, θ_x_nodes]. None for inference.
    n_feature : bool
        Include N/100 as third node feature.

    Returns
    -------
    Data
        Graph with x, edge_index, edge_list, and optionally y.
    """
    builder = HamiltonianBuilder()
    edge_index_np, coord = builder.build_graph_data(lattice)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    N = lattice.n_qubits
    h_feat = np.full(N, float(h_value))
    cols = [h_feat, coord.astype(float)]
    if n_feature:
        cols.append(np.full(N, N / 100.0))

    x = torch.tensor(np.stack(cols, axis=1), dtype=torch.float32)

    # Undirected edge list (i < j) for θ_zz prediction
    edges_unique = np.array(sorted(lattice.edges))
    edge_list = torch.tensor(edges_unique, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_list=edge_list)
    data.n_nodes = N
    data.n_edges_unique = len(edges_unique)

    if theta_opt is not None:
        # Ensure float64 before torch conversion (handles legacy dtype=object)
        theta_opt = np.asarray(theta_opt, dtype=np.float64)
        data.y = torch.tensor(theta_opt, dtype=torch.float32)

    return data


def train_bond_resolved_mpnn(
    model: BondResolvedMPNN,
    dataset: list[Data],
    n_epochs: int = 8000,
    lr: float = 1e-3,
    patience: int = 300,
    seed: int = 42,
    weight_decay: float = 1e-5,
    val_fraction: float = 0.2,
) -> dict:
    """Train the BondResolvedMPNN with per-node/per-edge MSE loss.

    Parameters
    ----------
    model : BondResolvedMPNN
    dataset : list[Data]
        Training graphs with `y` targets [θ_zz, θ_x].
    n_epochs : int
    lr : float
    patience : int
        Scheduler patience for ReduceLROnPlateau.
    seed : int
    weight_decay : float
        L2 regularization for Adam optimizer. Default 1e-5 (minimal
        regularization always active to prevent weight explosion).
    val_fraction : float
        Fraction of dataset held out for validation. Default 0.2.
        Reports val_mse and generalization_gap to detect overfitting.
        Set to 0.0 to disable (uses all data for training).

    Returns
    -------
    dict with training metrics:
        - final_mse, final_zz_mse, final_x_mse
        - val_mse (None if val_fraction=0)
        - generalization_gap (val_mse - final_mse, None if no val)
        - mse_history, zz_loss_history, x_loss_history
        - stopped_early, stop_reason
    """
    if len(dataset) < 3:
        raise ValueError(f"Need ≥3 training points, got {len(dataset)}.")

    import torch.nn.functional as F

    torch.manual_seed(seed)

    # Train/val split
    n_total = len(dataset)
    n_val = int(n_total * val_fraction) if val_fraction > 0 else 0
    if n_val > 0 and n_total - n_val < 3:
        n_val = 0  # Not enough for meaningful split

    if n_val > 0:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_total)
        val_idx = set(indices[:n_val].tolist())
        train_data = [dataset[i] for i in range(n_total) if i not in val_idx]
        val_data = [dataset[i] for i in val_idx]
    else:
        train_data = dataset
        val_data = []

    # ── Compute device (GPU if available, else CPU) — portable local ↔ server ──
    from qmbp_simulation.utils.helpers import describe_device, resolve_device

    device = resolve_device()
    model = model.to(device)
    logger.info("  train_bond_resolved_mpnn device: %s", describe_device(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience, factor=0.5, min_lr=1e-6
    )

    mse_history: list[float] = []
    zz_loss_history: list[float] = []
    x_loss_history: list[float] = []
    val_mse_history: list[float] = []

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        total_zz_loss = 0.0
        total_x_loss = 0.0

        for data in train_data:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data).squeeze(0)  # [n_edges + n_nodes]
            target = data.y

            n_e = data.n_edges_unique
            loss_zz = F.mse_loss(pred[:n_e], target[:n_e])
            loss_x = F.mse_loss(pred[n_e:], target[n_e:])
            loss = loss_zz + loss_x

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_zz_loss += loss_zz.item()
            total_x_loss += loss_x.item()

        avg_loss = total_loss / len(train_data)
        mse_history.append(avg_loss)
        zz_loss_history.append(total_zz_loss / len(train_data))
        x_loss_history.append(total_x_loss / len(train_data))
        scheduler.step(avg_loss)

        # Validation (every 50 epochs to limit overhead)
        if val_data and epoch % 50 == 0:
            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for data in val_data:
                    data = data.to(device)
                    pred = model(data).squeeze(0)
                    target = data.y
                    n_e = data.n_edges_unique
                    v_loss += (F.mse_loss(pred[:n_e], target[:n_e]) +
                               F.mse_loss(pred[n_e:], target[n_e:])).item()
            val_mse_history.append(v_loss / len(val_data))
            model.train()

        if (epoch + 1) % 250 == 0:
            val_str = f", val={val_mse_history[-1]:.2e}" if val_mse_history else ""
            logger.info(
                f"  Epoch {epoch + 1}: MSE={avg_loss:.2e} "
                f"(ZZ={total_zz_loss / len(train_data):.2e}, "
                f"X={total_x_loss / len(train_data):.2e}{val_str})"
            )

    # Final validation MSE
    final_val_mse = None
    if val_data:
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for data in val_data:
                data = data.to(device)
                pred = model(data).squeeze(0)
                target = data.y
                n_e = data.n_edges_unique
                v_loss += (F.mse_loss(pred[:n_e], target[:n_e]) +
                           F.mse_loss(pred[n_e:], target[n_e:])).item()
        final_val_mse = v_loss / len(val_data)

    # Return model to CPU (device-agnostic checkpoints + CPU inference).
    model = model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    final_mse = mse_history[-1] if mse_history else float("inf")
    gen_gap = (final_val_mse - final_mse) if final_val_mse is not None else None

    return {
        "mse_history": mse_history,
        "zz_loss_history": zz_loss_history,
        "x_loss_history": x_loss_history,
        "val_mse_history": val_mse_history,
        "final_mse": final_mse,
        "final_zz_mse": zz_loss_history[-1] if zz_loss_history else float("inf"),
        "final_x_mse": x_loss_history[-1] if x_loss_history else float("inf"),
        "val_mse": final_val_mse,
        "generalization_gap": gen_gap,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "stopped_early": False,
        "stop_reason": "completed",
    }
