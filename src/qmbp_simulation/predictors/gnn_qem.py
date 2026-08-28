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

from qmbp_simulation.models.constants import DEFAULT_SEEDS

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

    # ── Compute device (GPU if available, else CPU) — portable local ↔ server ──
    from qmbp_simulation.utils.helpers import finalize_model_device, prepare_model_device

    model, device = prepare_model_device(model, log_prefix="gnn_qem")

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
            batch = batch.to(device)
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
                batch = batch.to(device)
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
            batch = batch.to(device)
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

    # Device-agnostic checkpoints + CPU inference; frees GPU for the next job.
    model = finalize_model_device(model, device)

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
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)  # nosec: trusted checkpoint

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
        Random seeds for layout selection (default: DEFAULT_SEEDS).
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
        seeds = list(DEFAULT_SEEDS)

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
        json.dump(serializable, f, indent=2, default=str)
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


# ═══════════════════════════════════════════════════════════════════════════
# GNN-QEM V2 — Scalable Architecture (Plan 08)
# ═══════════════════════════════════════════════════════════════════════════
#
# Changes from V1:
#   - GATv2Conv with per-edge 2Q gate errors (instead of GINConv)
#   - Expanded node features (6 dims): T1, T2, readout, gate_err, n_cx_local, degree
#   - Virtual global node with expanded context (7 dims)
#   - Residual connections, hidden_dim=128, heads=4, layers=4
#   - global_add_pool (preserves magnitude scaling with N)
#   - Calibration augmentation during training (±20%)
#   - QEMSampleV2 extends QEMSample with gap, per-qubit gate counts, degree
#
# Backward compatibility:
#   - V1 classes (GNNQEMConfig, GNNQEMCorrector, QEMSample) are UNTOUCHED
#   - V2 coexists as separate classes in the same module
#   - Checkpoints are versioned ("version": "2.0")
#   - HardwareBackend.load_gnn_qem() auto-detects V1 vs V2


@dataclass
class GNNQEMConfigV2:
    """Configuration for GNN-QEM V2 (scalable to N=127+).

    Differences from V1:
    - Per-edge features (2Q gate error as edge_attr)
    - GATv2Conv with multi-head attention
    - Virtual global node absorbs context
    - Larger capacity (hidden=128, 4 layers)
    - Calibration augmentation for drift robustness
    """

    node_feature_dim: int = 6  # T1, T2, readout, gate_err_max, n_cx_local, degree
    edge_feature_dim: int = 1  # per-edge 2Q gate error
    context_dim: int = 7  # h, n_2q, CES_2q, CES_readout, E_noisy/N, gap, sign_bias
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.15
    use_virtual_node: bool = True
    augment_calibration: bool = True
    augment_scale: float = 0.2
    lr: float = 5e-4
    epochs: int = 3000
    patience: int = 300


@dataclass
class QEMSampleV2(QEMSample):
    """Extended sample with V2 features. Inherits all V1 fields.

    New fields enable richer graph construction:
    - gap: spectral gap (informs noise sensitivity)
    - n_cx_per_qubit: 2Q gate count per qubit (local error exposure)
    - qubit_degree: coupling map degree per qubit
    - ces_2q: CES from 2Q gates only
    - ces_readout: CES from readout only
    """

    gap: float = 0.0
    n_cx_per_qubit: list[float] = field(default_factory=list)
    qubit_degree: list[int] = field(default_factory=list)
    ces_2q: float = 0.0
    ces_readout: float = 0.0


class GNNQEMCorrectorV2(nn.Module):
    """V2 GNN-based energy correction model — scalable to N=127+.

    Architecture improvements over V1:
    - GATv2Conv: attention-based message passing with per-edge features
    - Per-edge 2Q gate errors as edge_attr (not averaged to nodes)
    - Virtual global node: carries circuit-level context, provides
      infinite receptive field in 1 extra hop
    - Residual connections between layers
    - global_add_pool (preserves magnitude, better for variable N)
    - Dual output: ΔE correction + confidence (same as V1 interface)

    References
    ----------
    - arXiv:2604.16815 (2026): Physically-informed GNN for QEM
    - arXiv:2504.00464 (2025): GNN circuit output prediction with noise features
    - Brody et al. (2022): "How Attentive are GATs?" (GATv2)
    """

    def __init__(self, config: GNNQEMConfigV2 | None = None) -> None:
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        from torch_geometric.nn import global_add_pool as _gap

        if config is None:
            config = GNNQEMConfigV2()
        self.config = config
        self._global_add_pool = _gap

        # ── Virtual node projection: context_dim → hidden_dim ────────
        if config.use_virtual_node:
            self.virtual_node_proj = nn.Sequential(
                nn.Linear(config.context_dim, config.hidden_dim),
                nn.ReLU(),
            )

        # ── Input projection: node_feature_dim → hidden_dim ──────────
        self.input_proj = nn.Linear(config.node_feature_dim, config.hidden_dim)

        # ── GATv2Conv layers with edge features ──────────────────────
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(config.n_layers):
            conv = GATv2Conv(
                in_channels=config.hidden_dim,
                out_channels=config.hidden_dim // config.n_heads,
                heads=config.n_heads,
                edge_dim=config.edge_feature_dim,
                concat=True,  # output = n_heads * (hidden/n_heads) = hidden
                dropout=config.dropout,
                add_self_loops=False,  # virtual node handles global info
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(config.hidden_dim))

        # ── Correction head: pooled features → ΔE ────────────────────
        head_input_dim = config.hidden_dim * 2 if config.use_virtual_node else config.hidden_dim
        self.correction_head = nn.Sequential(
            nn.Linear(head_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

        # ── Confidence head ──────────────────────────────────────────
        self.confidence_head = nn.Sequential(
            nn.Linear(head_input_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict energy correction and confidence.

        Parameters
        ----------
        data : Data
            Must have: x [n_nodes, node_feat], edge_index [2, n_edges],
            edge_attr [n_edges, edge_feat], batch [n_nodes].
            If use_virtual_node: also needs context [batch_size, context_dim].

        Returns
        -------
        tuple[Tensor, Tensor]
            (delta_e [batch_size, 1], confidence [batch_size, 1])
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)

        # Project node features to hidden_dim
        x = self.input_proj(x)

        # Add virtual node if configured
        virtual_embs = None
        if self.config.use_virtual_node and hasattr(data, "context"):
            virtual_embs = self._add_virtual_node(x, edge_index, edge_attr, batch, data.context)
            x, edge_index, edge_attr, batch = virtual_embs

        # Message passing with residual connections
        for conv, norm in zip(self.convs, self.norms, strict=False):
            x_prev = x
            if edge_attr is not None:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = x + x_prev  # Residual connection

        # Readout
        if self.config.use_virtual_node:
            # Extract virtual node embeddings (last node per graph)
            pooled = self._global_add_pool(x, batch)
            # Virtual nodes are at the end — get their embeddings
            batch_size = int(batch.max().item()) + 1
            virtual_indices = self._get_virtual_node_indices(batch, batch_size)
            virtual_out = x[virtual_indices]
            combined = torch.cat([pooled, virtual_out], dim=-1)
        else:
            combined = self._global_add_pool(x, batch)

        delta_e = self.correction_head(combined)
        confidence = self.confidence_head(combined)

        return delta_e, confidence

    def _add_virtual_node(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        batch: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Add one virtual node per graph, connected to all real nodes."""
        batch_size = int(batch.max().item()) + 1
        device = x.device

        # Project context to hidden_dim for each graph
        virtual_feats = self.virtual_node_proj(context)  # [batch_size, hidden_dim]

        # Append virtual nodes to x
        x = torch.cat([x, virtual_feats], dim=0)  # [n_nodes + batch_size, hidden]

        # Update batch for virtual nodes
        virtual_batch = torch.arange(batch_size, device=device, dtype=torch.long)
        batch = torch.cat([batch, virtual_batch], dim=0)

        # Build edges: virtual_node ↔ all nodes in its graph
        n_real_nodes = x.size(0) - batch_size
        new_src, new_dst = [], []
        for g in range(batch_size):
            virtual_idx = n_real_nodes + g
            graph_nodes = (batch[:n_real_nodes] == g).nonzero(as_tuple=True)[0]
            for node_idx in graph_nodes:
                new_src.append(virtual_idx)
                new_dst.append(int(node_idx))
                new_src.append(int(node_idx))
                new_dst.append(virtual_idx)

        if new_src:
            virtual_edges = torch.tensor([new_src, new_dst], device=device, dtype=torch.long)
            edge_index = torch.cat([edge_index, virtual_edges], dim=1)

            # Edge attr for virtual edges: use 0 (no gate error on virtual connections)
            if edge_attr is not None:
                n_virtual_edges = len(new_src)
                virtual_edge_attr = torch.zeros(n_virtual_edges, edge_attr.size(1), device=device)
                edge_attr = torch.cat([edge_attr, virtual_edge_attr], dim=0)

        return x, edge_index, edge_attr, batch

    def _get_virtual_node_indices(self, batch: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Get indices of virtual nodes (last batch_size nodes)."""
        total_nodes = batch.size(0)
        return torch.arange(total_nodes - batch_size, total_nodes, device=batch.device)


def build_qem_graph_v2(
    sample: QEMSampleV2,
    augment: bool = False,
    augment_scale: float = 0.2,
) -> Data:
    """Convert a QEMSampleV2 into a torch_geometric Data object (V2 format).

    V2 differences from V1:
    - 6 node features (added n_cx_local, degree)
    - Per-edge 2Q gate error as edge_attr
    - Context tensor for virtual node (7 dims with gap and sign_bias)
    - Optional calibration augmentation (±scale noise for drift robustness)

    Parameters
    ----------
    sample : QEMSampleV2
        Input sample with calibration and expanded features.
    augment : bool
        If True, perturb calibration features ±augment_scale (for training).
    augment_scale : float
        Relative perturbation magnitude (default 0.2 = ±20%).

    Returns
    -------
    Data
        PyTorch Geometric graph ready for GNNQEMCorrectorV2.
    """
    n = sample.n_qubits

    # ── Helper: pad array to n entries ───────────────────────────────
    def _pad(arr: list[float], default: float) -> np.ndarray:
        padded = np.full(n, default, dtype=np.float32)
        available = min(len(arr), n)
        if available > 0:
            padded[:available] = np.array(arr[:available], dtype=np.float32)
        return padded

    # ── Node features [6 dims] ───────────────────────────────────────
    t1_norm = (
        _pad(sample.qubit_t1, 100.0) / 100.0 if sample.qubit_t1 else np.ones(n, dtype=np.float32)
    )
    t2_norm = (
        _pad(sample.qubit_t2, 80.0) / 100.0
        if sample.qubit_t2
        else np.full(n, 0.8, dtype=np.float32)
    )
    readout = (
        _pad(sample.readout_errors, 0.01)
        if sample.readout_errors
        else np.full(n, 0.01, dtype=np.float32)
    )

    # Gate error max per qubit (from 2Q edges — same as V1)
    if sample.gate_errors_2q and sample.edge_index.size > 0:
        gate_feat = np.full(n, 0.01, dtype=np.float32)
        edge_idx = sample.edge_index
        for i in range(min(len(sample.gate_errors_2q), edge_idx.shape[1] // 2)):
            if i < edge_idx.shape[1]:
                q0 = int(edge_idx[0, i])
                q1 = int(edge_idx[1, i])
                if q0 < n:
                    gate_feat[q0] = max(gate_feat[q0], sample.gate_errors_2q[i])
                if q1 < n:
                    gate_feat[q1] = max(gate_feat[q1], sample.gate_errors_2q[i])
    else:
        gate_feat = np.full(n, 0.01, dtype=np.float32)

    # n_cx_local: normalized 2Q gate count per qubit
    if sample.n_cx_per_qubit:
        cx_arr = _pad(sample.n_cx_per_qubit, 0.0)
        max_cx = max(cx_arr.max(), 1.0)
        n_cx_norm = cx_arr / max_cx
    else:
        n_cx_norm = np.full(n, 0.5, dtype=np.float32)  # default: uniform assumption

    # degree: coupling map connectivity per qubit (normalized)
    if sample.qubit_degree:
        degree_arr = np.array(sample.qubit_degree[:n], dtype=np.float32)
        if len(degree_arr) < n:
            degree_arr = np.pad(degree_arr, (0, n - len(degree_arr)), constant_values=2)
        max_deg = max(degree_arr.max(), 1.0)
        degree_norm = degree_arr / max_deg
    else:
        degree_norm = np.full(n, 0.5, dtype=np.float32)

    # Stack into [n_qubits, 6]
    node_features = np.stack([t1_norm, t2_norm, readout, gate_feat, n_cx_norm, degree_norm], axis=1)
    x = torch.tensor(node_features, dtype=torch.float32)

    # ── Edge index + edge_attr [per-edge 2Q gate error] ──────────────
    if sample.edge_index.size > 0:
        edge_index = torch.tensor(sample.edge_index, dtype=torch.long)
    else:
        # Fallback: linear chain
        sources = list(range(n - 1))
        targets = list(range(1, n))
        edge_index = torch.tensor([sources + targets, targets + sources], dtype=torch.long)

    # Per-edge 2Q gate error
    n_edges = edge_index.size(1)
    edge_attr = torch.full((n_edges, 1), 0.005, dtype=torch.float32)
    if sample.gate_errors_2q:
        # Map gate errors to edges (undirected: each error appears twice)
        n_undirected = len(sample.gate_errors_2q)
        for i in range(min(n_undirected, n_edges)):
            edge_attr[i, 0] = sample.gate_errors_2q[i % n_undirected]
        # Mirror for reverse direction
        if n_edges > n_undirected:
            for i in range(n_undirected, min(2 * n_undirected, n_edges)):
                edge_attr[i, 0] = sample.gate_errors_2q[i - n_undirected]

    # ── Calibration augmentation ─────────────────────────────────────
    if augment:
        noise = 1.0 + augment_scale * torch.randn_like(x)
        noise = noise.clamp(min=0.3, max=2.0)  # prevent negative/extreme values
        x = x * noise
        edge_noise = 1.0 + augment_scale * torch.randn_like(edge_attr)
        edge_noise = edge_noise.clamp(min=0.3, max=3.0)
        edge_attr = edge_attr * edge_noise

    # ── Context vector [7 dims] for virtual node ─────────────────────
    ces_2q = sample.ces_2q if sample.ces_2q > 0 else sample.ces
    ces_readout = sample.ces_readout
    gap_norm = sample.gap / 10.0 if sample.gap > 0 else 0.0
    sign_bias = -1.0  # TFIM: noise always raises energy

    context = torch.tensor(
        [
            [
                sample.h_value,
                sample.n_2q_gates / 50.0,
                ces_2q,
                ces_readout,
                sample.noisy_energy / max(n, 1),
                gap_norm,
                sign_bias,
            ]
        ],
        dtype=torch.float32,
    )

    # ── Target ───────────────────────────────────────────────────────
    delta_e = sample.exact_energy - sample.noisy_energy

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        context=context,
        y=torch.tensor([[delta_e]], dtype=torch.float32),
        noisy_energy=torch.tensor([[sample.noisy_energy]], dtype=torch.float32),
    )


def train_gnn_qem_v2(
    model: GNNQEMCorrectorV2,
    train_data: list[Data],
    val_data: list[Data],
    config: GNNQEMConfigV2 | None = None,
) -> QEMTrainResult:
    """Train GNN-QEM V2 model. Same interface as train_gnn_qem but for V2.

    Differences from V1 training:
    - Uses augmented graphs (calibration perturbation) during training
    - AdamW optimizer with cosine annealing
    - Gradient clipping at 1.0

    Parameters
    ----------
    model : GNNQEMCorrectorV2
        Model to train (modified in-place).
    train_data : list[Data]
        Training graphs (from build_qem_graph_v2).
    val_data : list[Data]
        Validation graphs.
    config : GNNQEMConfigV2 | None
        Training config (uses model.config if None).

    Returns
    -------
    QEMTrainResult
        Training summary (same type as V1 for compatibility).
    """
    from torch_geometric.loader import DataLoader

    if config is None:
        config = model.config

    if len(train_data) < 5:
        raise ValueError(f"Need ≥5 training samples, got {len(train_data)}.")
    if len(val_data) < 2:
        raise ValueError(f"Need ≥2 validation samples, got {len(val_data)}.")

    batch_size = min(32, len(train_data))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=min(len(val_data), 64), shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
    )

    # ── Compute device (GPU if available, else CPU) — portable local ↔ server ──
    from qmbp_simulation.utils.helpers import finalize_model_device, prepare_model_device

    model, device = prepare_model_device(model, log_prefix="gnn_qem_v2")

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0
    train_loss = float("inf")

    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            delta_e_pred, confidence = model(batch)

            # Primary loss: MSE on energy correction
            correction_loss = nn.functional.mse_loss(delta_e_pred, batch.y)

            # Auxiliary confidence loss (same as V1)
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

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            n_val = 0
            for batch in val_loader:
                batch = batch.to(device)
                delta_e_pred, _ = model(batch)
                val_loss += nn.functional.mse_loss(delta_e_pred, batch.y).item()
                n_val += 1
            val_loss /= max(n_val, 1)
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            logger.info(f"[gnn_qem_v2] Early stop at epoch {epoch} (patience={config.patience})")
            break

        if epoch % 200 == 0:
            logger.info(f"[gnn_qem_v2] Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Compute validation metrics (same logic as V1)
    model.eval()
    val_mae = 0.0
    improvements = []
    n_val = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            delta_e_pred, _ = model(batch)
            val_mae += torch.mean(torch.abs(delta_e_pred - batch.y)).item()
            n_val += 1
            e_noisy = batch.noisy_energy
            e_exact = e_noisy + batch.y
            e_corrected = e_noisy + delta_e_pred
            error_before = torch.abs(e_noisy - e_exact)
            error_after = torch.abs(e_corrected - e_exact)
            valid_mask = error_before > 1e-10
            if valid_mask.any():
                improvement = (
                    (error_before[valid_mask] - error_after[valid_mask])
                    / error_before[valid_mask]
                    * 100
                )
                improvements.append(improvement.mean().item())

    val_mae /= max(n_val, 1)
    mean_improvement = float(np.mean(improvements)) if improvements else 0.0

    logger.info(
        f"[gnn_qem_v2] Done: best_epoch={best_epoch}, "
        f"val_MAE={val_mae:.6f}, improvement={mean_improvement:.1f}%"
    )

    # Device-agnostic checkpoints + CPU inference; frees GPU for the next job.
    model = finalize_model_device(model, device)

    return QEMTrainResult(
        best_epoch=best_epoch,
        train_loss_final=train_loss,
        val_loss_final=best_val_loss,
        val_mae=val_mae,
        val_improvement_pct=mean_improvement,
        n_train=len(train_data),
        n_val=len(val_data),
    )


def correct_energy_v2(
    model: GNNQEMCorrectorV2,
    sample: QEMSampleV2,
    confidence_threshold: float = 0.5,
) -> QEMCorrectionResult:
    """Apply trained GNN-QEM V2 model to correct a single noisy energy.

    Same interface and return type as V1 correct_energy() for compatibility.
    """
    model.eval()
    data = build_qem_graph_v2(sample, augment=False)
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
        logger.debug(f"[gnn_qem_v2] Confidence {conf:.3f} < {confidence_threshold}, skipping")

    return QEMCorrectionResult(
        corrected_energy=corrected,
        delta_e_predicted=delta_e,
        confidence=conf,
        correction_applied=applied,
        original_energy=sample.noisy_energy,
    )


def save_qem_v2_checkpoint(
    model: GNNQEMCorrectorV2,
    path: Path | str,
    train_result: QEMTrainResult | None = None,
    metadata: dict | None = None,
) -> None:
    """Save V2 checkpoint with version header for auto-detection."""
    from dataclasses import asdict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "2.0",
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "train_result": asdict(train_result) if train_result else None,
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    logger.info(f"[gnn_qem_v2] Saved V2 checkpoint to {path}")


def load_qem_v2_checkpoint(
    path: Path | str,
) -> tuple[GNNQEMCorrectorV2, QEMTrainResult | None, dict]:
    """Load V2 checkpoint. Raises ValueError if not a V2 checkpoint."""
    raw = torch.load(Path(path), map_location="cpu", weights_only=False)  # nosec: trusted checkpoint

    if not isinstance(raw, dict) or raw.get("version") != "2.0":
        raise ValueError(
            f"Not a V2 checkpoint (version={raw.get('version', 'unknown')}). "
            f"Use load_qem_checkpoint() for V1."
        )

    config = GNNQEMConfigV2(**raw["config"])
    model = GNNQEMCorrectorV2(config)
    model.load_state_dict(raw["state_dict"])
    model.eval()

    train_result = None
    if raw.get("train_result"):
        train_result = QEMTrainResult(**raw["train_result"])

    metadata = raw.get("metadata", {})
    logger.info(f"[gnn_qem_v2] Loaded V2 checkpoint from {path}")
    return model, train_result, metadata


def generate_qem_training_data_v2(
    topologies: list[str] | None = None,
    n_qubits_list: list[int] | None = None,
    h_values: list[float] | None = None,
    p_layers: int = 1,
    shots: int = 8192,
    model_name: str = "tfim",
    theta_source: str = "zoo",
    npz_paths: list[Path] | None = None,
) -> list[QEMSampleV2]:
    """Generate V2 training data with realistic θ_opt and expanded features.

    Key difference from V1: uses θ_opt from the model Zoo or NPZ datasets
    instead of random θ, producing training data representative of actual
    deployment (MPNN warm-started circuits).

    Parameters
    ----------
    topologies : list[str] | None
        Lattice topologies (default: ["chain_1d"]).
    n_qubits_list : list[int] | None
        System sizes (default: [6, 10]).
    h_values : list[float] | None
        Transverse field values (default: paramagnetic regime).
    p_layers : int
        HVA layers (default: 1).
    shots : int
        Shots per noisy estimation (default: 8192).
    model_name : str
        Model from registry (default: "tfim").
    theta_source : {"zoo", "npz", "random"}
        Source of θ_opt for circuit binding.
    npz_paths : list[Path] | None
        Paths to .npz datasets (for theta_source="npz").

    Returns
    -------
    list[QEMSampleV2]
        Generated samples ready for build_qem_graph_v2().
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
    from qmbp_simulation.pipeline.dataset_io import load_phase12_dataset
    from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

    if topologies is None:
        topologies = ["chain_1d"]
    if n_qubits_list is None:
        n_qubits_list = [6, 10]
    if h_values is None:
        h_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    spec = get_model_spec(model_name)
    fake_backend = FakeTorino()
    adj = build_adjacency(fake_backend)
    cal_snap = take_calibration_snapshot(fake_backend)
    gt_cache = GroundTruthCache()

    # Pre-load NPZ theta data if requested
    npz_theta_map: dict[tuple[str, int, float], np.ndarray] = {}
    if theta_source == "npz" and npz_paths:
        for path in npz_paths:
            try:
                data = load_phase12_dataset(path)
                n_q = int(data["n_qubits"])
                for i, h in enumerate(data["h_values"]):
                    npz_theta_map[(str(path), n_q, float(h))] = data["theta_opt"][i]
            except Exception as e:
                logger.warning(f"[gen_data_v2] Failed to load {path}: {e}")

    # Pre-load MPNN from zoo if requested
    mpnn_cache: dict[tuple[str, int], object] = {}
    if theta_source == "zoo":
        from qmbp_simulation.predictors.model_zoo import load_pretrained

        for topo in topologies:
            for n_qubits in n_qubits_list:
                try:
                    mpnn = load_pretrained(
                        model=model_name,
                        topology=topo,
                        n_qubits=n_qubits,
                        p_layers=p_layers,
                    )
                    mpnn_cache[(topo, n_qubits)] = mpnn
                except Exception:
                    logger.debug(f"[gen_data_v2] No zoo model for {topo} N={n_qubits}")

    samples: list[QEMSampleV2] = []
    rng = np.random.default_rng(42)

    for topo in topologies:
        for n_qubits in n_qubits_list:
            candidates = find_layouts_bfs(adj, n_qubits, n_candidates=20)
            if not candidates:
                continue

            for h in h_values:
                # ── Get θ_opt from cascade ────────────────────────────
                lattice = make_lattice(topo, n_qubits, J=1.0, h=h)
                qc, params = spec.create_circuit(n_qubits, p_layers, lattice, **spec.circuit_kwargs)
                n_params = len(params)

                theta = None
                if theta_source == "npz":
                    for key, t in npz_theta_map.items():
                        if key[1] == n_qubits and abs(key[2] - h) < 0.01:
                            theta = t[:n_params] if len(t) >= n_params else None
                            break

                if theta is None and theta_source in ("zoo", "npz"):
                    mpnn = mpnn_cache.get((topo, n_qubits))
                    if mpnn is not None:
                        try:
                            from qmbp_simulation.predictors.mpnn import build_graph_dataset

                            graphs = build_graph_dataset(
                                lattice,
                                np.array([h]),
                                theta_opt=np.zeros((1, n_params)),
                                e_exact=np.array([0.0]),
                                fidelity_threshold=0.0,
                            )
                            if graphs:
                                import torch as _torch

                                with _torch.no_grad():
                                    theta = mpnn(graphs[0]).numpy().flatten()[:n_params]
                        except Exception:
                            pass

                if theta is None:
                    theta = rng.uniform(-0.5, 0.5, n_params)

                bound = qc.assign_parameters(theta)

                # ── Layout selection + transpilation ──────────────────
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

                # ── Noisy estimation ─────────────────────────────────
                H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
                H_mapped = H.apply_layout(transpiled.layout)
                config_noisy = NoisyEstimatorConfig(shots=shots, seed_simulator=42)
                e_noisy = noisy_estimate(transpiled, H_mapped, fake_backend, config_noisy)

                # ── Ground truth from cache or solver ─────────────────
                cached = gt_cache.get(topo, n_qubits, model_name, h)
                if cached:
                    e_exact = cached["energy"]
                    gap = cached.get("gap", 0.0)
                else:
                    from qmbp_simulation.solvers.classical import ClassicalSolver

                    solver = ClassicalSolver()
                    gt = solver.solve(H, lattice)
                    e_exact = gt.ground_energy
                    gap = gt.gap
                    gt_cache.put(
                        topo, n_qubits, model_name, h, energy=e_exact, gap=gap, method="exact_diag"
                    )

                # ── Extract per-qubit features ────────────────────────
                layout_qubits = layout_sel.layouts[0]
                t1_list = [cal_snap.qubit_t1.get(q, 100.0) for q in layout_qubits]
                t2_list = [cal_snap.qubit_t2.get(q, 80.0) for q in layout_qubits]
                ro_list = [cal_snap.readout_errors.get(q, 0.01) for q in layout_qubits]

                # Gate count per qubit from transpiled circuit
                cx_per_qubit = [0.0] * n_qubits
                for inst in transpiled.data:
                    if inst.operation.num_qubits == 2 and inst.operation.name not in (
                        "barrier",
                        "delay",
                    ):
                        for q in inst.qubits:
                            idx = transpiled.find_bit(q).index
                            if idx < n_qubits:
                                cx_per_qubit[idx] += 1.0

                # Degree in coupling map subgraph
                qubit_to_local = {q: i for i, q in enumerate(layout_qubits)}
                degrees = [0] * n_qubits
                for qi in layout_qubits:
                    local_i = qubit_to_local[qi]
                    for qj in adj.get(qi, []):
                        if qj in qubit_to_local:
                            degrees[local_i] += 1

                # Edge info
                gate_errs = []
                local_edges_src, local_edges_dst = [], []
                for qi in layout_qubits:
                    for qj in adj.get(qi, []):
                        if qj in qubit_to_local:
                            local_edges_src.append(qubit_to_local[qi])
                            local_edges_dst.append(qubit_to_local[qj])
                            key = f"{min(qi, qj)}-{max(qi, qj)}"
                            if key in cal_snap.gate_errors_2q:
                                gate_errs.append(cal_snap.gate_errors_2q[key])

                edge_idx = (
                    np.array([local_edges_src, local_edges_dst], dtype=int)
                    if local_edges_src
                    else np.zeros((2, 0), dtype=int)
                )

                # CES decomposition
                ces_2q = ces  # CES is dominated by 2Q gates
                ces_readout = 1.0 - np.prod([1.0 - r for r in ro_list])

                samples.append(
                    QEMSampleV2(
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
                        # V2 fields:
                        gap=gap,
                        n_cx_per_qubit=cx_per_qubit,
                        qubit_degree=degrees,
                        ces_2q=ces_2q,
                        ces_readout=ces_readout,
                    )
                )

                logger.debug(
                    f"[gen_v2] {topo} N={n_qubits} h={h:.2f} → "
                    f"E_noisy={e_noisy:.4f} (exact={e_exact:.4f}, Δ={abs(e_noisy - e_exact):.4f})"
                )

    logger.info(f"[gen_data_v2] Generated {len(samples)} V2 samples")
    return samples


def save_qem_samples_v2(samples: list[QEMSampleV2], path: Path | str) -> None:
    """Save V2 QEM samples to JSON. Backward compatible with V1 loader for V1 fields.

    Parameters
    ----------
    samples : list[QEMSampleV2]
        V2 samples to persist.
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
        d["_version"] = "2.0"  # Marker for loader
        serializable.append(d)

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"[gnn_qem_v2] Saved {len(samples)} V2 samples to {path}")


def load_qem_samples_v2(path: Path | str) -> list[QEMSampleV2]:
    """Load V2 QEM samples from JSON.

    Handles both V2 format (with gap, n_cx_per_qubit, etc.) and V1 format
    (fills V2 fields with defaults). This allows loading V1 training data
    as V2 samples for backward compatibility.

    Parameters
    ----------
    path : Path | str
        Input JSON path.

    Returns
    -------
    list[QEMSampleV2]
        Loaded samples (V2 format, with defaults for missing fields).
    """
    import json

    with open(Path(path)) as f:
        data = json.load(f)

    samples = []
    for d in data:
        d.pop("_version", None)  # Remove version marker if present
        d["edge_index"] = np.array(d.get("edge_index", [[], []]), dtype=int)

        # Ensure V2 fields have defaults if loading V1 data
        d.setdefault("gap", 0.0)
        d.setdefault("n_cx_per_qubit", [])
        d.setdefault("qubit_degree", [])
        d.setdefault("ces_2q", d.get("ces", 0.0))
        d.setdefault("ces_readout", 0.0)

        # Filter only known fields for QEMSampleV2
        import dataclasses

        v2_fields = {f.name for f in dataclasses.fields(QEMSampleV2)}
        filtered = {k: v for k, v in d.items() if k in v2_fields}
        samples.append(QEMSampleV2(**filtered))

    logger.info(f"[gnn_qem_v2] Loaded {len(samples)} V2 samples from {path}")
    return samples
