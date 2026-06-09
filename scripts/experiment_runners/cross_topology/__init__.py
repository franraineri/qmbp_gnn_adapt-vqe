"""Cross-topology transfer experiment runners.

This package implements bidirectional cross-topology transfer experiments
validating GNN (MPNNPredictor) generalization across heterogeneous lattice
topologies (triangular ↔ heavy_hex).

Modules:
    helpers: Shared utilities (data adapter, graph construction, evaluation)
    run_vqe_data_gen: VQE data generation for missing topologies
    run_cross_n_validation: Within-topology cross-N validation
    run_cross_topology: Cross-topology transfer (tri→hex, hex→tri)
    run_ablation: GNN vs MLP vs Scipy + BatchNorm ablation
    run_orchestrator: Full experiment orchestration
"""
