"""
Pipeline Core — Shared 4-phase pipeline execution logic.

Extracts the common Phase 1–4 execution pattern used by:
  - scripts/run_v61_parametric.py
  - scripts/run_thesis_results.py
  - scripts/smoke_test_v61.py

This avoids duplicating the same pipeline logic in 3+ scripts.
Any change to the pipeline pattern only needs to happen here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class PipelineCoreConfig:
    """Minimal configuration for a single pipeline execution.

    All scripts can extend this or map their own config to these fields.
    """

    N: int = 6
    J: float | np.ndarray = 1.0
    p_layers: int = 2
    topology: str = "chain_1d"
    # VQE
    n_restarts: int = 5
    maxiter: int = 1000
    # MPNN
    mpnn_hidden: int = 64
    mpnn_layers: int = 3
    mpnn_epochs: int = 6000
    mpnn_lr: float = 1e-3
    mpnn_patience: int = 300
    per_parameter_heads: bool = False
    use_edge_features: bool = False
    # Deployment
    h_test: float = 1.25
    fidelity_threshold: float = 0.93
    # Reproducibility
    seed: int = 42
    # H-grid
    h_grid: str = "standard"  # "standard" (27pts) or "dense" (45pts)
    # Callbacks
    enable_callbacks: bool = False


@dataclass
class PipelineResult:
    """Structured output from a pipeline execution."""

    # Phase 1
    h_values: np.ndarray = field(default_factory=lambda: np.array([]))
    exact_data: list = field(default_factory=list)
    phase1_time: float = 0.0

    # Phase 2
    vqe_results: list = field(default_factory=list)
    fidelities: np.ndarray = field(default_factory=lambda: np.array([]))
    phase2_time: float = 0.0

    # Phase 3
    dataset: list = field(default_factory=list)
    model: object = None
    train_result: dict = field(default_factory=dict)
    phase3_time: float = 0.0

    # Phase 4
    deploy_result: object = None
    theta_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    phase4_time: float = 0.0

    # Overall
    total_time: float = 0.0
    success: bool = True
    error: str | None = None


# ── H-grid generation ────────────────────────────────────────────────────


def generate_h_grid(grid_type: str = "standard") -> np.ndarray:
    """Generate the h-value grid for the VQE sweep.

    Parameters
    ----------
    grid_type : str
        "standard" — 27 points with Δh=0.05 near critical region.
        "dense" — 45 points with Δh=0.025 near critical region.

    Returns
    -------
    np.ndarray — sorted unique h-values.
    """
    h_coarse = np.arange(0.0, 0.8, 0.1)
    h_coarse2 = np.arange(1.5, 2.05, 0.1)
    h_dense = np.arange(0.8, 1.45, 0.025) if grid_type == "dense" else np.arange(0.8, 1.45, 0.05)

    return np.unique(np.concatenate([h_coarse, h_dense, h_coarse2]))


# ── Phase execution functions ────────────────────────────────────────────


def run_phase1(
    cfg: PipelineCoreConfig,
    h_values: np.ndarray,
) -> tuple[list, float]:
    """Phase 1: Exact diagonalization across h-grid.

    Returns
    -------
    (exact_data, elapsed_s)
    """
    from .classical_solver import ClassicalSolver
    from .hamiltonian_builder import HamiltonianBuilder, make_lattice

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()

    t0 = time.time()
    exact_data = []
    for h in h_values:
        lat_h = make_lattice(cfg.topology, cfg.N, J=cfg.J, h=h)
        H = builder.build(lat_h)
        exact_data.append(solver.solve(H, lat_h))
    elapsed = time.time() - t0

    logger.info("Phase 1: %d points in %.1fs", len(exact_data), elapsed)
    return exact_data, elapsed


def run_phase2(
    cfg: PipelineCoreConfig,
    h_values: np.ndarray,
    exact_data: list,
) -> tuple[list, np.ndarray, float]:
    """Phase 2: VQE descending sweep.

    Returns
    -------
    (vqe_results, fidelities, elapsed_s)
    """
    from .config import VQEConfig
    from .hamiltonian_builder import make_lattice
    from .hva_builder import HVACircuitBuilder
    from .vqe_optimizer import VQEOptimizer

    hva = HVACircuitBuilder()
    base_lattice = make_lattice(cfg.topology, cfg.N, J=cfg.J, h=1.0)
    qc, _ = hva.create(cfg.N, cfg.p_layers, base_lattice)

    vqe_config = VQEConfig(
        p_layers=cfg.p_layers,
        n_restarts=cfg.n_restarts,
        maxiter=cfg.maxiter,
        enable_callbacks=cfg.enable_callbacks,
    )
    opt = VQEOptimizer(vqe_config)

    t0 = time.time()
    vqe_results = opt.descending_sweep(h_values, qc, base_lattice, exact_data)
    elapsed = time.time() - t0

    fidelities = np.array([r.fidelity for r in vqe_results])
    logger.info(
        "Phase 2: avg fid=%.1f%%, %d/%d above threshold, %.1fs",
        np.mean(fidelities) * 100,
        np.sum(fidelities >= cfg.fidelity_threshold),
        len(fidelities),
        elapsed,
    )
    return vqe_results, fidelities, elapsed


def run_phase3(
    cfg: PipelineCoreConfig,
    h_values: np.ndarray,
    vqe_results: list,
    exact_data: list,
    fidelities: np.ndarray,
) -> tuple[list, object, dict, float]:
    """Phase 3: MPNN training.

    Returns
    -------
    (dataset, model, train_result, elapsed_s)
    """
    from .hamiltonian_builder import make_lattice
    from .mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn

    base_lattice = make_lattice(cfg.topology, cfg.N, J=cfg.J, h=1.0)

    dataset = build_graph_dataset(
        base_lattice,
        h_values,
        np.array([r.theta_opt for r in vqe_results]),
        np.array([d.ground_energy for d in exact_data]),
        fidelities=fidelities,
        fidelity_threshold=cfg.fidelity_threshold,
        include_edge_features=cfg.use_edge_features,
    )

    model = MPNNPredictor(
        node_features=2,
        hidden_dim=cfg.mpnn_hidden,
        n_layers=cfg.mpnn_layers,
        output_dim=2 * cfg.p_layers,
        per_parameter_heads=cfg.per_parameter_heads,
        use_edge_features=cfg.use_edge_features,
    )

    t0 = time.time()
    train_result = train_mpnn(
        model, dataset, n_epochs=cfg.mpnn_epochs, lr=cfg.mpnn_lr, patience=cfg.mpnn_patience
    )
    elapsed = time.time() - t0

    logger.info(
        "Phase 3: MSE=%.6f, %d training points, %.1fs",
        train_result["final_mse"],
        len(dataset),
        elapsed,
    )
    return dataset, model, train_result, elapsed


def run_phase4(
    cfg: PipelineCoreConfig,
    model: object,
    *,
    mode: str = "simulation",
    include_baseline: bool = True,
    n_baseline_seeds: int = 5,
) -> tuple[object, np.ndarray, float]:
    """Phase 4: MPNN inference + deployment.

    Returns
    -------
    (deploy_result, theta_pred, elapsed_s)
    """
    import torch
    from torch_geometric.data import Data

    from .classical_solver import ClassicalSolver
    from .hamiltonian_builder import HamiltonianBuilder, make_lattice
    from .hardware_deployer_v61 import HardwareDeployerV61
    from .hva_builder import HVACircuitBuilder

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    base_lattice = make_lattice(cfg.topology, cfg.N, J=cfg.J, h=1.0)
    qc, _ = hva.create(cfg.N, cfg.p_layers, base_lattice)

    lat_test = make_lattice(cfg.topology, cfg.N, J=cfg.J, h=cfg.h_test)
    H_test = builder.build(lat_test)
    exact_test = solver.solve(H_test, lat_test)

    model.eval()
    edge_idx, coord = builder.build_graph_data(base_lattice)
    x_test = torch.tensor(
        np.stack([np.full(cfg.N, cfg.h_test), coord.astype(float)], axis=1),
        dtype=torch.float32,
    )
    test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))

    t0 = time.time()
    with torch.no_grad():
        theta_pred = model(test_graph).numpy().flatten()

    deployer = HardwareDeployerV61(mode=mode)

    if include_baseline:
        deploy_result, baseline_comparison = deployer.deploy_with_baseline(
            qc,
            H_test,
            theta_pred,
            lat_test,
            exact_test,
            n_random_seeds=n_baseline_seeds,
        )
    else:
        deploy_result = deployer.deploy_adapt_vqe(qc, H_test, theta_pred, lat_test, exact_test)
        baseline_comparison = None

    elapsed = time.time() - t0

    logger.info(
        "Phase 4: ΔE/gap=%.4f, phase=%s, %.1fs",
        deploy_result.delta_e_over_gap,
        deploy_result.phase_label,
        elapsed,
    )
    if baseline_comparison is not None:
        logger.info(
            "Phase 4 baseline: gain=%.1f%%, cold mean ΔE/gap=%.4f",
            baseline_comparison.gain_energy_pct,
            baseline_comparison.cold_start_mean["delta_e_over_gap"],
        )

    return deploy_result, theta_pred, elapsed


# ── Full pipeline execution ──────────────────────────────────────────────


def run_full_pipeline(
    cfg: PipelineCoreConfig,
    *,
    deploy_mode: str = "simulation",
) -> PipelineResult:
    """Execute the full 4-phase pipeline with the given configuration.

    This is the single-source-of-truth implementation that all scripts
    should delegate to for the core pipeline logic.

    Parameters
    ----------
    cfg : PipelineCoreConfig
        Full pipeline configuration.
    deploy_mode : str
        Deployment mode for Phase 4 ("simulation", "noisy_simulation", "hardware").

    Returns
    -------
    PipelineResult with all phase outputs.
    """
    import random

    import torch

    # Set seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    result = PipelineResult()
    t_total = time.time()

    try:
        # H-grid
        result.h_values = generate_h_grid(cfg.h_grid)

        # Phase 1
        result.exact_data, result.phase1_time = run_phase1(cfg, result.h_values)

        # Phase 2
        result.vqe_results, result.fidelities, result.phase2_time = run_phase2(
            cfg, result.h_values, result.exact_data
        )

        # Phase 3
        result.dataset, result.model, result.train_result, result.phase3_time = run_phase3(
            cfg, result.h_values, result.vqe_results, result.exact_data, result.fidelities
        )

        # Phase 4
        result.deploy_result, result.theta_pred, result.phase4_time = run_phase4(
            cfg, result.model, mode=deploy_mode
        )

    except Exception as e:
        result.success = False
        result.error = str(e)
        logger.error("Pipeline failed: %s", e)

    result.total_time = time.time() - t_total
    return result
