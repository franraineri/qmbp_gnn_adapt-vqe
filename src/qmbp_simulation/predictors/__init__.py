"""Predictors submodule — MPNN parameter prediction + GNN-QEM error correction."""

from qmbp_simulation.predictors.gnn_qem import (
    GNNQEMConfig,
    GNNQEMCorrector,
    QEMCorrectionResult,
    QEMSample,
    QEMTrainResult,
    build_qem_dataset,
    build_qem_graph,
    correct_energy,
    generate_qem_training_data,
    load_qem_checkpoint,
    load_qem_samples,
    save_qem_checkpoint,
    save_qem_samples,
    train_gnn_qem,
)
from qmbp_simulation.predictors.mpnn import (
    MPNNPredictor,
    build_graph_dataset,
    load_mpnn_checkpoint,
    save_mpnn_checkpoint,
    train_mpnn,
)

__all__ = [
    # Phase 3: MPNN parameter predictor
    "MPNNPredictor",
    "build_graph_dataset",
    "load_mpnn_checkpoint",
    "save_mpnn_checkpoint",
    "train_mpnn",
    # Phase 4+: GNN-QEM error correction
    "GNNQEMCorrector",
    "GNNQEMConfig",
    "QEMSample",
    "QEMTrainResult",
    "QEMCorrectionResult",
    "build_qem_graph",
    "build_qem_dataset",
    "train_gnn_qem",
    "correct_energy",
    "generate_qem_training_data",
    "save_qem_checkpoint",
    "load_qem_checkpoint",
    "save_qem_samples",
    "load_qem_samples",
]
