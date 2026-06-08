"""
GNN-QEM — Graph Neural Network for Quantum Error Mitigation.

Uses the hardware coupling topology as a graph to learn noise propagation
patterns and predict corrected (ideal) energies from noisy measurements.

Architecture:
  Input: Data(x=[n_qubits, node_feat_dim], edge_index=[2, n_edges],
              y=[1] (noisy_energy scalar), context=[context_dim])
    - Node features: (T1_i, T2_i, readout_error_i, gate_error_mean_i)
    - Edge features: 2Q gate error for each coupling
    - Context: (h_value, n_2q_gates, CES, noisy_energy)
  Message Passing: k GINConv layers (same architecture as Phase 3 MPNN)
  Readout: global_mean_pool → concat(pooled, context) → MLP → ΔE_correction
  Output: E_corrected = E_noisy + ΔE_correction

References
----------
- Wang et al. (2026). "Scalable Quantum Error Mitigation with Physically
  Informed Graph Neural Networks." arXiv:2604.16815.
- Czarnik et al. (2024). "Machine Learning for Practical Quantum Error
  Mitigation." arXiv:2309.17368.
- Xu, Huang et al. (2025). "Physics-inspired Machine Learning for Quantum
  Error Mitigation." arXiv:2501.04558.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class GNNQEMConfig:
    """Configuration for the GNN-QEM model.

    Parameters
    ----------
    node_feature_dim : int
        Number of node features (T1, T2, readout_err, mean_gate_err).
    context_dim : int
        Number of scalar context features (h, n_2q, CES, E_noisy).
    hidden_dim : int
        Hidden dimension for GINConv layers.
    n_layers : int
        Number of message-passing layers.
    dropout : float
        Dropout rate in MLP head.
    lr : float
        Learning rate for training.
    epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience.
    """

    node_feature_dim: int = 4
    context_dim: int = 4
    hidden_dim: int = 64
    n_layers: int = 3
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 2000
    patience: int = 200


# ── Model ────────────────────────────────────────────────────────────────


class GNNQEMCorrector(nn.Module):
    """GNN-based energy correction model for quantum error mitigation.

    Learns to predict ΔE = E_ideal - E_noisy from the hardware topology
    graph with calibration data as node/edge features. The correction is
    additive: E_corrected = E_noisy + ΔE_predicted.

    Architecture follows Wang et al. (arXiv:2604.16815) GEM framework:
    - Graph encodes hardware topology (nodes=qubits, edges=couplings)
    - Node features capture local noise (T1, T2, readout error)
    - Edge features capture coupling noise (2Q gate errors)
    - GINConv propagates error correlations along coupling structure
    - Context vector provides circuit-level information (h, CES, E_noisy)
    - Dual output: ΔE correction + confidence estimate

    Parameters
    ----------
    config : GNNQEMConfig
        Model configuration.
    """

    def __init__(self, config: GNNQEMConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = GNNQEMConfig()
        self.config = config

        # GINConv message passing (same architecture as Phase 3 MPNN)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer: node_feature_dim → hidden_dim
        mlp0 = nn.Sequential(
            nn.Linear(config.node_feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.convs.append(GINConv(mlp0))
        self.bns.append(nn.BatchNorm1d(config.hidden_dim))

        # Subsequent layers: hidden_dim → hidden_dim
        for _ in range(config.n_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(config.hidden_dim))

        # Correction head: pooled graph features + context → ΔE
        head_input_dim = config.hidden_dim + config.context_dim
        self.correction_head = nn.Sequential(
            nn.Linear(head_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),  # ΔE scalar
        )

        # Confidence head: estimates uncertainty of correction
        self.confidence_head = nn.Sequential(
            nn.Linear(head_input_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict energy correction and confidence from hardware graph.

        Parameters
        ----------
        data : Data
            Must have: x, edge_index, batch, context.
            - x: [n_nodes, node_feature_dim] — qubit calibration features
            - edge_index: [2, n_edges] — hardware coupling map
            - batch: [n_nodes] — batch assignment
            - context: [batch_size, context_dim] — circuit-level features

        Returns
        -------
        tuple[Tensor, Tensor]
            (delta_e, confidence) — correction and confidence per sample.
            delta_e: [batch_size, 1] — additive correction
            confidence: [batch_size, 1] — prediction confidence ∈ [0, 1]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Message passing
        for conv, bn in zip(self.convs, self.bns, strict=False):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)

        # Global pooling → graph-level representation
        graph_repr = global_mean_pool(x, batch)

        # Concatenate with context
        context = (
            data.context
            if hasattr(data, "context")
            else torch.zeros(graph_repr.size(0), self.config.context_dim, device=graph_repr.device)
        )
        combined = torch.cat([graph_repr, context], dim=-1)

        # Predict correction and confidence
        delta_e = self.correction_head(combined)
        confidence = self.confidence_head(combined)

        return delta_e, confidence


# ── Data construction ────────────────────────────────────────────────────


@dataclass
class QEMSample:
    """A single training/inference sample for GNN-QEM.

    Attributes
    ----------
    noisy_energy : float
        Measured energy from noisy execution.
    exact_energy : float
        Ground truth energy from exact diag.
    h_value : float
        Transverse field value.
    n_2q_gates : int
        Number of 2-qubit gates in transpiled circuit.
    ces : float
        Circuit Error Score.
    topology : str
        Lattice topology name.
    n_qubits : int
        System size.
    qubit_t1 : list[float]
        T1 (µs) per qubit in layout.
    qubit_t2 : list[float]
        T2 (µs) per qubit in layout.
    readout_errors : list[float]
        Readout error per qubit.
    gate_errors_2q : list[float]
        2Q gate errors for edges in layout.
    edge_index : np.ndarray
        [2, n_edges] coupling map of the layout.
    """

    noisy_energy: float
    exact_energy: float
    h_value: float
    n_2q_gates: int
    ces: float
    topology: str
    n_qubits: int
    qubit_t1: list[float] = field(default_factory=list)
    qubit_t2: list[float] = field(default_factory=list)
    readout_errors: list[float] = field(default_factory=list)
    gate_errors_2q: list[float] = field(default_factory=list)
    edge_index: np.ndarray = field(default_factory=lambda: np.zeros((2, 0), dtype=int))


def build_qem_graph(sample: QEMSample) -> Data:
    """Convert a QEMSample into a torch_geometric Data object.

    Node features (per qubit): [T1_normalized, T2_normalized, readout_err, mean_gate_err]
    Edge index: hardware coupling map of the layout
    Context: [h_value, n_2q_gates_normalized, CES, noisy_energy_normalized]
    Target: delta_e = exact_energy - noisy_energy (what the model learns)

    Normalization strategy:
      - T1, T2: divide by 100 µs (typical scale for IBM processors)
      - n_2q_gates: divide by 50 (max practical for ZNE regime)
      - Energy: divide by n_qubits (energy is extensive)

    Parameters
    ----------
    sample : QEMSample
        Input sample with calibration and measurement data.

    Returns
    -------
    Data
        PyTorch Geometric graph ready for GNNQEMCorrector.
    """
    n = sample.n_qubits

    # Node features: [T1/100, T2/100, readout_err, mean_gate_err]
    # Pad with defaults if calibration data is shorter than n_qubits
    def _pad_to_n(arr: list[float], default: float) -> np.ndarray:
        """Pad list to exactly n entries with default value."""
        padded = np.full(n, default, dtype=np.float32)
        available = min(len(arr), n)
        padded[:available] = arr[:available]
        return padded

    t1_norm = (
        _pad_to_n(sample.qubit_t1, 100.0) / 100.0
        if sample.qubit_t1
        else np.ones(n, dtype=np.float32)
    )
    t2_norm = (
        _pad_to_n(sample.qubit_t2, 80.0) / 100.0
        if sample.qubit_t2
        else np.full(n, 0.8, dtype=np.float32)
    )
    readout = (
        _pad_to_n(sample.readout_errors, 0.01)
        if sample.readout_errors
        else np.full(n, 0.01, dtype=np.float32)
    )

    # Per-qubit gate error: distribute 2Q errors to their participating qubits
    if sample.gate_errors_2q and sample.edge_index.size > 0:
        gate_feat = np.full(n, 0.01, dtype=np.float32)
        edge_idx = sample.edge_index
        n_edges_per_direction = edge_idx.shape[1] // 2 if edge_idx.shape[1] > 0 else 0
        for i, err in enumerate(sample.gate_errors_2q[:n_edges_per_direction]):
            if i < edge_idx.shape[1]:
                q0 = int(edge_idx[0, i])
                q1 = int(edge_idx[1, i])
                if q0 < n:
                    gate_feat[q0] = max(gate_feat[q0], err)
                if q1 < n:
                    gate_feat[q1] = max(gate_feat[q1], err)
    elif sample.gate_errors_2q:
        mean_gate = float(np.mean(sample.gate_errors_2q))
        gate_feat = np.full(n, mean_gate, dtype=np.float32)
    else:
        gate_feat = np.full(n, 0.01, dtype=np.float32)

    x = torch.tensor(
        np.stack([t1_norm, t2_norm, readout, gate_feat], axis=1),
        dtype=torch.float32,
    )

    # Edge index
    if sample.edge_index.size > 0:
        edge_index = torch.tensor(sample.edge_index, dtype=torch.long)
    else:
        # Fallback: linear chain
        sources = list(range(n - 1))
        targets = list(range(1, n))
        edge_index = torch.tensor([sources + targets, targets + sources], dtype=torch.long)

    # Context features: [h, n_2q/50, CES, E_noisy/N]
    context = torch.tensor(
        [
            [
                sample.h_value,
                sample.n_2q_gates / 50.0,
                sample.ces,
                sample.noisy_energy / max(n, 1),
            ]
        ],
        dtype=torch.float32,
    )

    # Target: correction the model should learn
    delta_e = sample.exact_energy - sample.noisy_energy

    return Data(
        x=x,
        edge_index=edge_index,
        context=context,
        y=torch.tensor([[delta_e]], dtype=torch.float32),
        noisy_energy=torch.tensor([[sample.noisy_energy]], dtype=torch.float32),
    )


def build_qem_dataset(samples: list[QEMSample]) -> list[Data]:
    """Convert a list of QEMSamples into a graph dataset.

    Parameters
    ----------
    samples : list[QEMSample]
        Training/validation samples.

    Returns
    -------
    list[Data]
        Graph dataset for GNN-QEM training.
    """
    dataset = []
    for s in samples:
        try:
            data = build_qem_graph(s)
            dataset.append(data)
        except Exception as e:
            logger.warning(f"[gnn_qem] Skipping sample h={s.h_value}: {e}")
    logger.info(f"[gnn_qem] Built dataset: {len(dataset)} graphs from {len(samples)} samples")
    return dataset


# ── Training ─────────────────────────────────────────────────────────────


@dataclass
class QEMTrainResult:
    """Result of GNN-QEM training.

    Attributes
    ----------
    best_epoch : int
        Epoch with lowest validation loss.
    train_loss_final : float
        Final training MSE loss.
    val_loss_final : float
        Final validation MSE loss.
    val_mae : float
        Mean absolute error on validation set.
    val_improvement_pct : float
        Mean improvement in |ΔE| vs uncorrected (%), on validation set.
    n_train : int
        Number of training samples.
    n_val : int
        Number of validation samples.
    """

    best_epoch: int
    train_loss_final: float
    val_loss_final: float
    val_mae: float
    val_improvement_pct: float
    n_train: int
    n_val: int


def train_gnn_qem(
    model: GNNQEMCorrector,
    train_data: list[Data],
    val_data: list[Data],
    config: GNNQEMConfig | None = None,
) -> QEMTrainResult:
    """Train the GNN-QEM model on (noisy, exact) energy pairs.

    Uses MSE loss on ΔE predictions with early stopping. The confidence
    head is trained with an auxiliary loss: it learns to predict whether
    its ΔE correction will reduce error (1 = correction helps, 0 = hurts).

    Parameters
    ----------
    model : GNNQEMCorrector
        Model to train (modified in-place).
    train_data : list[Data]
        Training graph dataset (minimum 5 samples).
    val_data : list[Data]
        Validation graph dataset (minimum 2 samples).
    config : GNNQEMConfig | None
        Training config (uses model.config if None).

    Returns
    -------
    QEMTrainResult
        Training summary with metrics.

    Raises
    ------
    ValueError
        If insufficient training or validation data.
    """
    from torch_geometric.loader import DataLoader

    if config is None:
        config = model.config

    if len(train_data) < 5:
        raise ValueError(
            f"Need at least 5 training samples, got {len(train_data)}. "
            f"Generate more data with generate_qem_training_data()."
        )
    if len(val_data) < 2:
        raise ValueError(f"Need at least 2 validation samples, got {len(val_data)}.")

    # Clamp batch size to avoid BatchNorm issues with batch_size=1
    batch_size = min(32, len(train_data))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_batch_size = min(len(val_data), 64)
    val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=config.patience // 2, factor=0.5
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0
    train_loss = float("inf")

    model.train()
    for epoch in range(config.epochs):
        # Training
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            delta_e_pred, confidence = model(batch)

            # Primary loss: MSE on energy correction
            correction_loss = nn.functional.mse_loss(delta_e_pred, batch.y)

            # Auxiliary confidence loss: predict whether correction helps
            # Target: 1 if |E_corrected - E_exact| < |E_noisy - E_exact|, else 0
            with torch.no_grad():
                e_noisy = batch.noisy_energy
                e_exact = e_noisy + batch.y
                error_before = torch.abs(e_noisy - e_exact)
                error_after = torch.abs(e_noisy + delta_e_pred.detach() - e_exact)
                conf_target = (error_after < error_before).float()

            confidence_loss = nn.functional.binary_cross_entropy(confidence, conf_target)

            loss = correction_loss + 0.1 * confidence_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += correction_loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            n_val_batches = 0
            for batch in val_loader:
                delta_e_pred, _ = model(batch)
                val_loss += nn.functional.mse_loss(delta_e_pred, batch.y).item()
                n_val_batches += 1
            val_loss = val_loss / max(n_val_batches, 1)
        model.train()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            logger.info(f"[gnn_qem] Early stopping at epoch {epoch} (patience={config.patience})")
            break

        if epoch % 200 == 0:
            logger.info(
                f"[gnn_qem] Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Compute validation metrics
    model.eval()
    val_mae = 0.0
    improvements = []
    n_val_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            delta_e_pred, _ = model(batch)
            val_mae += torch.mean(torch.abs(delta_e_pred - batch.y)).item()
            n_val_batches += 1
            # Improvement: compare |E_noisy - E_exact| vs |E_corrected - E_exact|
            e_noisy = batch.noisy_energy
            e_exact = e_noisy + batch.y
            e_corrected = e_noisy + delta_e_pred
            error_before = torch.abs(e_noisy - e_exact)
            error_after = torch.abs(e_corrected - e_exact)
            # Avoid division by zero for samples where E_noisy == E_exact
            valid_mask = error_before > 1e-10
            if valid_mask.any():
                improvement = (
                    (error_before[valid_mask] - error_after[valid_mask])
                    / error_before[valid_mask]
                    * 100
                )
                improvements.append(improvement.mean().item())

    val_mae = val_mae / max(n_val_batches, 1)
    mean_improvement = float(np.mean(improvements)) if improvements else 0.0

    logger.info(
        f"[gnn_qem] Training complete: best_epoch={best_epoch}, "
        f"val_MAE={val_mae:.6f}, improvement={mean_improvement:.1f}%"
    )

    return QEMTrainResult(
        best_epoch=best_epoch,
        train_loss_final=train_loss,
        val_loss_final=best_val_loss,
        val_mae=val_mae,
        val_improvement_pct=mean_improvement,
        n_train=len(train_data),
        n_val=len(val_data),
    )


# ── Inference ────────────────────────────────────────────────────────────


@dataclass
class QEMCorrectionResult:
    """Result of applying GNN-QEM correction to a noisy energy.

    Attributes
    ----------
    corrected_energy : float
        E_noisy + ΔE_predicted.
    delta_e_predicted : float
        Predicted correction (additive).
    confidence : float
        Model's confidence in the correction [0, 1].
    correction_applied : bool
        False if confidence < threshold (returns uncorrected energy).
    original_energy : float
        Input noisy energy (unchanged).
    """

    corrected_energy: float
    delta_e_predicted: float
    confidence: float
    correction_applied: bool
    original_energy: float


def correct_energy(
    model: GNNQEMCorrector,
    sample: QEMSample,
    confidence_threshold: float = 0.5,
) -> QEMCorrectionResult:
    """Apply trained GNN-QEM model to correct a single noisy energy.

    If the model's confidence is below `confidence_threshold`, the
    correction is NOT applied (returns uncorrected energy). This prevents
    degrading results when the model is uncertain.

    Parameters
    ----------
    model : GNNQEMCorrector
        Trained model.
    sample : QEMSample
        Noisy measurement with hardware calibration features.
    confidence_threshold : float
        Minimum confidence to apply correction (default: 0.5).

    Returns
    -------
    QEMCorrectionResult
        Corrected energy with metadata.
    """
    model.eval()
    data = build_qem_graph(sample)
    # Add batch dimension
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

    with torch.no_grad():
        delta_e_pred, confidence = model(data)

    delta_e = float(delta_e_pred[0, 0])
    conf = float(confidence[0, 0])

    if conf >= confidence_threshold:
        corrected = sample.noisy_energy + delta_e
        applied = True
    else:
        corrected = sample.noisy_energy
        applied = False
        logger.debug(
            f"[gnn_qem] Confidence {conf:.3f} < {confidence_threshold}, "
            f"skipping correction (ΔE would be {delta_e:.6f})"
        )

    return QEMCorrectionResult(
        corrected_energy=corrected,
        delta_e_predicted=delta_e,
        confidence=conf,
        correction_applied=applied,
        original_energy=sample.noisy_energy,
    )


# ── Checkpoint management ────────────────────────────────────────────────


def save_qem_checkpoint(
    model: GNNQEMCorrector,
    path: Path | str,
    train_result: QEMTrainResult | None = None,
    metadata: dict | None = None,
) -> None:
    """Save GNN-QEM model checkpoint.

    Parameters
    ----------
    model : GNNQEMCorrector
        Trained model to save.
    path : Path | str
        Output file path (.pt).
    train_result : QEMTrainResult | None
        Training metrics to store alongside weights.
    metadata : dict | None
        Additional metadata (topology, dataset info, etc.)
    """
    from dataclasses import asdict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(model.config),
        "train_result": asdict(train_result) if train_result else None,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)
    logger.info(f"[gnn_qem] Checkpoint saved to {path}")


def load_qem_checkpoint(
    path: Path | str,
    device: str = "cpu",
) -> tuple[GNNQEMCorrector, QEMTrainResult | None, dict]:
    """Load GNN-QEM model from checkpoint.

    Parameters
    ----------
    path : Path | str
        Checkpoint file (.pt).
    device : str
        Device to load model onto.

    Returns
    -------
    tuple[GNNQEMCorrector, QEMTrainResult | None, dict]
        (model, train_result, metadata)
    """
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)

    config = GNNQEMConfig(**checkpoint["config"])
    model = GNNQEMCorrector(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_result = None
    if checkpoint.get("train_result"):
        train_result = QEMTrainResult(**checkpoint["train_result"])

    logger.info(f"[gnn_qem] Loaded checkpoint from {path}")
    return model, train_result, checkpoint.get("metadata", {})


# ── Training Data Generation ─────────────────────────────────────────────


def generate_qem_training_data(
    topologies: list[str] | None = None,
    n_qubits_list: list[int] | None = None,
    h_values: list[float] | None = None,
    p_layers: int = 1,
    seeds: list[int] | None = None,
    shots: int = 4096,
    model_name: str = "tfim",
) -> list[QEMSample]:
    """Generate (noisy, exact) energy pairs using FakeTorino for GNN-QEM training.

    Runs the pipeline: exact_diag → VQE → transpile → noisy_estimate for
    each (topology, N, h, seed) combination. Collects calibration data
    from the backend alongside energy measurements.

    This function is computationally expensive (~2-5s per sample on FakeTorino).
    Recommended: generate once, save to disk, load for repeated training.

    Parameters
    ----------
    topologies : list[str] | None
        Lattice topologies (default: ["chain_1d"]).
    n_qubits_list : list[int] | None
        System sizes (default: [6]).
    h_values : list[float] | None
        Transverse field values (default: [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]).
    p_layers : int
        HVA layers (default: 1).
    seeds : list[int] | None
        Random seeds for layout selection (default: [42, 43, 44]).
    shots : int
        Shots per noisy estimation (default: 4096).
    model_name : str
        Model from registry (default: "tfim").

    Returns
    -------
    list[QEMSample]
        Generated samples ready for build_qem_dataset().
    """
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    from qmbp_simulation.execution.noisy_utils import (
        NoisyEstimatorConfig,
        build_adjacency,
        compute_circuit_ces,
        find_layouts_bfs,
        noisy_estimate,
        select_layouts_low_ces,
        take_calibration_snapshot,
    )
    from qmbp_simulation.models import make_lattice
    from qmbp_simulation.models.model_registry import get_model_spec
    from qmbp_simulation.solvers.classical import ClassicalSolver

    if topologies is None:
        topologies = ["chain_1d"]
    if n_qubits_list is None:
        n_qubits_list = [6]
    if h_values is None:
        h_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    if seeds is None:
        seeds = [42, 43, 44]

    spec = get_model_spec(model_name)
    solver = ClassicalSolver()
    fake_backend = FakeTorino()
    adj = build_adjacency(fake_backend)

    # Take one calibration snapshot (static for FakeTorino)
    cal_snap = take_calibration_snapshot(fake_backend)

    samples: list[QEMSample] = []
    total = len(topologies) * len(n_qubits_list) * len(h_values) * len(seeds)
    count = 0

    for topo in topologies:
        for n_qubits in n_qubits_list:
            # Find candidate layouts for this size
            candidates = find_layouts_bfs(adj, n_qubits, n_candidates=20)
            if not candidates:
                logger.warning(f"[gen_data] No layouts for {topo} N={n_qubits}, skipping")
                continue

            for h in h_values:
                # Phase 1: Exact ground state
                lattice = make_lattice(topo, n_qubits, J=1.0, h=h)
                H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
                gt = solver.solve(H, lattice)
                e_exact = gt.ground_energy

                # Build circuit with random theta (simulates imperfect VQE)
                qc, params = spec.create_circuit(n_qubits, p_layers, lattice, **spec.circuit_kwargs)

                for seed in seeds:
                    count += 1
                    rng = np.random.default_rng(seed)
                    theta = rng.uniform(-1.0, 1.0, len(params))
                    bound = qc.assign_parameters(theta)

                    # Select one low-CES layout
                    layout_sel = select_layouts_low_ces(
                        bound,
                        fake_backend,
                        candidates,
                        n_select=1,
                        optimization_level=2,
                    )
                    if not layout_sel.transpiled_circuits:
                        continue

                    transpiled = layout_sel.transpiled_circuits[0]
                    ces = layout_sel.ces_values[0]
                    _, n_2q = compute_circuit_ces(transpiled, fake_backend)

                    # Noisy estimation
                    H_mapped = H.apply_layout(transpiled.layout)
                    config = NoisyEstimatorConfig(shots=shots, seed_simulator=seed)
                    e_noisy = noisy_estimate(transpiled, H_mapped, fake_backend, config)

                    # Extract calibration for qubits in this layout
                    layout_qubits = layout_sel.layouts[0]
                    t1_list = [cal_snap.qubit_t1.get(q, 100.0) for q in layout_qubits]
                    t2_list = [cal_snap.qubit_t2.get(q, 80.0) for q in layout_qubits]
                    ro_list = [cal_snap.readout_errors.get(q, 0.01) for q in layout_qubits]

                    # Gate errors for edges in layout
                    gate_errs = []
                    for i in range(len(layout_qubits)):
                        for j in range(i + 1, len(layout_qubits)):
                            qi, qj = layout_qubits[i], layout_qubits[j]
                            key = f"{min(qi, qj)}-{max(qi, qj)}"
                            if key in cal_snap.gate_errors_2q:
                                gate_errs.append(cal_snap.gate_errors_2q[key])

                    # Build edge_index for the layout subgraph
                    local_edges_src = []
                    local_edges_dst = []
                    qubit_to_local = {q: i for i, q in enumerate(layout_qubits)}
                    for i, qi in enumerate(layout_qubits):
                        for qj in adj.get(qi, []):
                            if qj in qubit_to_local:
                                local_edges_src.append(i)
                                local_edges_dst.append(qubit_to_local[qj])

                    edge_idx = (
                        np.array([local_edges_src, local_edges_dst], dtype=int)
                        if local_edges_src
                        else np.zeros((2, 0), dtype=int)
                    )

                    samples.append(
                        QEMSample(
                            noisy_energy=e_noisy,
                            exact_energy=e_exact,
                            h_value=h,
                            n_2q_gates=n_2q,
                            ces=ces,
                            topology=topo,
                            n_qubits=n_qubits,
                            qubit_t1=t1_list,
                            qubit_t2=t2_list,
                            readout_errors=ro_list,
                            gate_errors_2q=gate_errs,
                            edge_index=edge_idx,
                        )
                    )

                    if count % 10 == 0:
                        logger.info(
                            f"[gen_data] {count}/{total}: {topo} N={n_qubits} "
                            f"h={h:.2f} seed={seed} → E_noisy={e_noisy:.4f} "
                            f"(exact={e_exact:.4f}, Δ={abs(e_noisy - e_exact):.4f})"
                        )

    logger.info(f"[gen_data] Generated {len(samples)} samples from {count} attempts")
    return samples


def save_qem_samples(samples: list[QEMSample], path: Path | str) -> None:
    """Save QEM samples to JSON for reuse without regeneration.

    Parameters
    ----------
    samples : list[QEMSample]
        Samples to save.
    path : Path | str
        Output JSON path.
    """
    import json
    from dataclasses import asdict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = []
    for s in samples:
        d = asdict(s)
        d["edge_index"] = s.edge_index.tolist()
        serializable.append(d)

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"[gnn_qem] Saved {len(samples)} samples to {path}")


def load_qem_samples(path: Path | str) -> list[QEMSample]:
    """Load QEM samples from JSON.

    Parameters
    ----------
    path : Path | str
        Input JSON path (from save_qem_samples).

    Returns
    -------
    list[QEMSample]
        Loaded samples.
    """
    import json

    with open(Path(path)) as f:
        data = json.load(f)

    samples = []
    for d in data:
        d["edge_index"] = np.array(d["edge_index"], dtype=int)
        samples.append(QEMSample(**d))

    logger.info(f"[gnn_qem] Loaded {len(samples)} samples from {path}")
    return samples
