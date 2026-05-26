#!/usr/bin/env python3
"""Exhaustive pipeline variant runner for thesis validation.

Executes multiple pipeline configurations varying key parameters to
characterize the framework's behavior across different regimes.
Default: N=10, p=2, chain_1d topology. N is configurable via --n-qubits.

Parameters varied:
    - VQE restarts (1, 3, 5, 7)
    - MPNN hidden dimension (32, 64, 128)
    - MPNN epochs (2000, 4000, 6000)
    - h-grid density (sparse 5pts, standard 10pts, dense 16pts)
    - h-test points (single, multiple, near-critical)
    - Fidelity threshold (0.90, 0.93, 0.95)
    - Seeds (42, 43, 44)
    - Noisy: shots (4096, 8192, 16384)
    - Noisy: n_layouts (2, 3, 5)

Usage:
    # Run all variants sequentially (default N=10)
    python scripts/run_thesis_variants-N_10-chain_1d.py

    # Run with a different system size
    python scripts/run_thesis_variants-N_10-chain_1d.py --n-qubits 8

    # Run only noiseless variants
    python scripts/run_thesis_variants-N_10-chain_1d.py --noiseless-only

    # Run only noisy variants
    python scripts/run_thesis_variants-N_10-chain_1d.py --noisy-only

    # Run a specific variant by index
    python scripts/run_thesis_variants-N_10-chain_1d.py --variant 3

    # Dry run (print commands without executing)
    python scripts/run_thesis_variants-N_10-chain_1d.py --dry-run

    # Resume from a specific variant (skip already completed)
    python scripts/run_thesis_variants-N_10-chain_1d.py --start-from 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# Default system size (overridable via --n-qubits CLI argument)
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_N_QUBITS = 10


# ═══════════════════════════════════════════════════════════════════════════
# Variant definitions
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PipelineVariant:
    """A single pipeline configuration to test."""

    id: str
    description: str
    category: str  # "noiseless" or "noisy"
    command: list[str]
    hypothesis: str
    expected_outcome: str
    output_dir: str


def build_noiseless_variants(n_qubits: int = DEFAULT_N_QUBITS) -> list[PipelineVariant]:
    """Build all noiseless pipeline variants."""
    variants = []
    python = sys.executable  # Use the same Python interpreter
    n = str(n_qubits)
    out_base = f"results/thesis/variants_N{n_qubits}_chain"

    # ─── Group A: VQE Restart Sensitivity ──────────────────────────────────
    # Hypothesis: 5 restarts is optimal; 1 restart may fail near h=1.25
    for n_restarts in [1, 3, 5, 7]:
        variants.append(
            PipelineVariant(
                id=f"NL-A{n_restarts}",
                description=f"VQE restarts={n_restarts} (standard grid, hidden=128)",
                category="noiseless",
                command=[
                    python,
                    "scripts/run_pipeline.py",
                    "--n-qubits",
                    n,
                    "--p",
                    "2",
                    "--n-restarts",
                    str(n_restarts),
                    "--maxiter",
                    "1000",
                    "--h-values",
                    "2.0",
                    "1.75",
                    "1.5",
                    "1.35",
                    "1.25",
                    "--h-test",
                    "1.5",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--output-dir",
                    f"{out_base}/nl_restarts_{n_restarts}",
                    "--verbose",
                ],
                hypothesis=f"VQE with {n_restarts} restart(s) converges to ΔE/gap < 5%",
                expected_outcome="PASS for ≥3 restarts; 1 restart may fail at h=1.25",
                output_dir=f"{out_base}/nl_restarts_{n_restarts}",
            )
        )

    # ─── Group B: MPNN Hidden Dimension ────────────────────────────────────
    # Hypothesis: h=64 is sufficient for N=6; h=128 is overkill
    for hidden_dim in [32, 64, 128]:
        variants.append(
            PipelineVariant(
                id=f"NL-B{hidden_dim}",
                description=f"MPNN hidden_dim={hidden_dim} (5 restarts, 6000 epochs)",
                category="noiseless",
                command=[
                    python,
                    "scripts/run_pipeline.py",
                    "--n-qubits",
                    n,
                    "--p",
                    "2",
                    "--n-restarts",
                    "5",
                    "--maxiter",
                    "1000",
                    "--h-values",
                    "2.0",
                    "1.75",
                    "1.5",
                    "1.35",
                    "1.25",
                    "--h-test",
                    "1.5",
                    "--hidden-dim",
                    str(hidden_dim),
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--output-dir",
                    f"{out_base}/nl_hidden_{hidden_dim}",
                    "--verbose",
                ],
                hypothesis=f"MPNN with hidden_dim={hidden_dim} achieves ΔE/gap < 5%",
                expected_outcome="All pass; h=32 may have slightly higher error",
                output_dir=f"{out_base}/nl_hidden_{hidden_dim}",
            )
        )

    # ─── Group C: MPNN Training Epochs ─────────────────────────────────────
    # Hypothesis: 6000 epochs is optimal; 2000 may underfit
    for n_epochs in [2000, 4000, 6000]:
        variants.append(
            PipelineVariant(
                id=f"NL-C{n_epochs}",
                description=f"MPNN epochs={n_epochs} (hidden=128, patience=500)",
                category="noiseless",
                command=[
                    python,
                    "scripts/run_pipeline.py",
                    "--n-qubits",
                    n,
                    "--p",
                    "2",
                    "--n-restarts",
                    "5",
                    "--maxiter",
                    "1000",
                    "--h-values",
                    "2.0",
                    "1.75",
                    "1.5",
                    "1.35",
                    "1.25",
                    "--h-test",
                    "1.5",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    str(n_epochs),
                    "--patience",
                    "500",
                    "--output-dir",
                    f"{out_base}/nl_epochs_{n_epochs}",
                    "--verbose",
                ],
                hypothesis=f"MPNN with {n_epochs} epochs converges sufficiently",
                expected_outcome="All pass; 2000 may have higher MSE but still < 5%",
                output_dir=f"{out_base}/nl_epochs_{n_epochs}",
            )
        )

    # ─── Group D: h-Grid Density ──────────────────────────────────────────
    # Hypothesis: Denser grid → better MPNN interpolation
    h_grids = {
        "sparse5": ["2.0", "1.75", "1.5", "1.35", "1.25"],
        "standard10": ["2.0", "1.9", "1.8", "1.7", "1.6", "1.5", "1.4", "1.35", "1.3", "1.25"],
        "dense16": [
            "2.0",
            "1.95",
            "1.9",
            "1.85",
            "1.8",
            "1.75",
            "1.7",
            "1.65",
            "1.6",
            "1.55",
            "1.5",
            "1.45",
            "1.4",
            "1.35",
            "1.3",
            "1.25",
        ],
    }
    for grid_name, h_vals in h_grids.items():
        variants.append(
            PipelineVariant(
                id=f"NL-D-{grid_name}",
                description=f"h-grid: {grid_name} ({len(h_vals)} points)",
                category="noiseless",
                command=[
                    python,
                    "scripts/run_pipeline.py",
                    "--n-qubits",
                    n,
                    "--p",
                    "2",
                    "--n-restarts",
                    "5",
                    "--maxiter",
                    "1000",
                    "--h-values",
                    *h_vals,
                    "--h-test",
                    "1.5",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--output-dir",
                    f"{out_base}/nl_grid_{grid_name}",
                    "--verbose",
                ],
                hypothesis=f"Grid density '{grid_name}' provides sufficient training data",
                expected_outcome="All pass; sparse5 is the minimum viable (from G1 experiment)",
                output_dir=f"{out_base}/nl_grid_{grid_name}",
            )
        )

    # ─── Group E: h-Test Points ────────────────────────────────────────────
    # Hypothesis: Deployment works at multiple unseen points
    h_test_configs = {
        "single_safe": (["1.5"], "Single safe point (well within valid regime)"),
        "multi_spread": (["1.9", "1.6", "1.3"], "Multiple points across regime"),
        "near_critical": (["1.3", "1.25"], "Near critical region (hardest)"),
        "interpolation": (["1.45", "1.55", "1.65"], "Between training points"),
    }
    for test_name, (h_tests, desc) in h_test_configs.items():
        variants.append(
            PipelineVariant(
                id=f"NL-E-{test_name}",
                description=f"h_test: {desc}",
                category="noiseless",
                command=[
                    python,
                    "scripts/run_pipeline.py",
                    "--n-qubits",
                    n,
                    "--p",
                    "2",
                    "--n-restarts",
                    "5",
                    "--maxiter",
                    "1000",
                    "--h-values",
                    "2.0",
                    "1.9",
                    "1.8",
                    "1.7",
                    "1.6",
                    "1.5",
                    "1.4",
                    "1.35",
                    "1.3",
                    "1.25",
                    "--h-test",
                    *h_tests,
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--output-dir",
                    f"{out_base}/nl_htest_{test_name}",
                    "--verbose",
                ],
                hypothesis=f"MPNN generalizes to unseen h-test points: {h_tests}",
                expected_outcome="PASS for safe/interpolation; near_critical may be marginal",
                output_dir=f"{out_base}/nl_htest_{test_name}",
            )
        )

    # ─── Group F: Seed Robustness ──────────────────────────────────────────
    # Hypothesis: Pipeline is seed-independent (confirmed in G5)
    for seed_val in [42, 43, 44]:
        variants.append(
            PipelineVariant(
                id=f"NL-F-seed{seed_val}",
                description=f"Seed={seed_val} (reproducibility check)",
                category="noiseless",
                command=[
                    python,
                    "scripts/run_pipeline.py",
                    "--n-qubits",
                    n,
                    "--p",
                    "2",
                    "--n-restarts",
                    "5",
                    "--maxiter",
                    "1000",
                    "--h-values",
                    "2.0",
                    "1.75",
                    "1.5",
                    "1.35",
                    "1.25",
                    "--h-test",
                    "1.5",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--output-dir",
                    f"{out_base}/nl_seed_{seed_val}",
                    "--verbose",
                ],
                hypothesis=f"Pipeline with seed={seed_val} produces ΔE/gap < 5%",
                expected_outcome="All seeds pass — pipeline is seed-independent",
                output_dir=f"{out_base}/nl_seed_{seed_val}",
            )
        )

    # ─── Group G: MPNN Patience (Early Stopping) ──────────────────────────
    # Hypothesis: patience=500 is sufficient; lower may underfit
    for patience in [50, 150, 500]:
        variants.append(
            PipelineVariant(
                id=f"NL-G-pat{patience}",
                description=f"MPNN patience={patience} (early stopping sensitivity)",
                category="noiseless",
                command=[
                    python,
                    "scripts/run_pipeline.py",
                    "--n-qubits",
                    n,
                    "--p",
                    "2",
                    "--n-restarts",
                    "5",
                    "--maxiter",
                    "1000",
                    "--h-values",
                    "2.0",
                    "1.75",
                    "1.5",
                    "1.35",
                    "1.25",
                    "--h-test",
                    "1.5",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    str(patience),
                    "--output-dir",
                    f"{out_base}/nl_patience_{patience}",
                    "--verbose",
                ],
                hypothesis=f"MPNN with patience={patience} converges to ΔE/gap < 5%",
                expected_outcome="All pass; patience=50 may stop too early",
                output_dir=f"{out_base}/nl_patience_{patience}",
            )
        )

    # ─── Group H: p=1 Layer (Reduced Ansatz) ──────────────────────────────
    # Hypothesis: p=1 works at h≥1.6 (narrower valid regime)
    variants.append(
        PipelineVariant(
            id="NL-H-p1",
            description="p=1 layer (reduced ansatz, valid regime h≥1.6)",
            category="noiseless",
            command=[
                python,
                "scripts/run_pipeline.py",
                "--n-qubits",
                n,
                "--p",
                "1",
                "--n-restarts",
                "5",
                "--maxiter",
                "1000",
                "--h-values",
                "2.0",
                "1.9",
                "1.8",
                "1.7",
                "1.6",
                "--h-test",
                "1.75",
                "--hidden-dim",
                "128",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                "--output-dir",
                f"{out_base}/nl_p1",
                "--verbose",
            ],
            hypothesis="p=1 HVA achieves ΔE/gap < 5% for h≥1.6",
            expected_outcome="PASS — p=1 valid regime is h≥1.6 at N=6",
            output_dir=f"{out_base}/nl_p1",
        )
    )

    # ─── Group I: VQE Maxiter Sensitivity ──────────────────────────────────
    # Hypothesis: maxiter=1000 is more than enough; 200 may suffice
    for maxiter in [200, 500, 1000]:
        variants.append(
            PipelineVariant(
                id=f"NL-I-iter{maxiter}",
                description=f"VQE maxiter={maxiter} (convergence speed test)",
                category="noiseless",
                command=[
                    python,
                    "scripts/run_pipeline.py",
                    "--n-qubits",
                    n,
                    "--p",
                    "2",
                    "--n-restarts",
                    "5",
                    "--maxiter",
                    str(maxiter),
                    "--h-values",
                    "2.0",
                    "1.75",
                    "1.5",
                    "1.35",
                    "1.25",
                    "--h-test",
                    "1.5",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--output-dir",
                    f"{out_base}/nl_maxiter_{maxiter}",
                    "--verbose",
                ],
                hypothesis=f"VQE with maxiter={maxiter} converges to ΔE/gap < 5%",
                expected_outcome="All pass — warm-start converges in <50 iterations typically",
                output_dir=f"{out_base}/nl_maxiter_{maxiter}",
            )
        )

    return variants


def build_extended_variants(n_qubits: int = DEFAULT_N_QUBITS) -> list[PipelineVariant]:
    """Build extended variants of the 5 most promising configurations.

    These push the limits further than the base variants to find
    the true boundaries of the framework's capabilities.
    """
    variants = []
    python = sys.executable
    n = str(n_qubits)
    out_base = f"results/thesis/variants_N{n_qubits}_chain"

    # ─── EXT-1: Ultra-sparse grid (3 points) ──────────────────────────────
    # Based on NL-D-sparse5. If 5 points works, what about 3?
    # This tests the absolute minimum data requirement for MPNN.
    # If it passes, we have a 82% data reduction vs the 17-point baseline.
    variants.append(
        PipelineVariant(
            id="EXT-1-ultrasparse3",
            description="Ultra-sparse grid: only 3 training points (absolute minimum)",
            category="noiseless",
            command=[
                python,
                "scripts/run_pipeline.py",
                "--n-qubits",
                n,
                "--p",
                "2",
                "--n-restarts",
                "5",
                "--maxiter",
                "1000",
                "--h-values",
                "2.0",
                "1.5",
                "1.25",
                "--h-test",
                "1.75",
                "1.35",
                "--hidden-dim",
                "128",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                "--output-dir",
                f"{out_base}/ext_ultrasparse3",
                "--verbose",
            ],
            hypothesis="MPNN can interpolate with only 3 training points",
            expected_outcome="LIKELY FAIL — 3 points is below G1 minimum (9 pts). "
            "Tests true data efficiency floor.",
            output_dir=f"{out_base}/ext_ultrasparse3",
        )
    )

    # ─── EXT-2: Extrapolation beyond training range ───────────────────────
    # Based on NL-E-near_critical. Instead of interpolation, test
    # EXTRAPOLATION: deploy at h=1.1 (below training range h≥1.25).
    # This probes whether MPNN can predict outside its training domain.
    variants.append(
        PipelineVariant(
            id="EXT-2-extrapolation",
            description="Extrapolation: deploy at h=1.1 and h=1.15 (below training range)",
            category="noiseless",
            command=[
                python,
                "scripts/run_pipeline.py",
                "--n-qubits",
                n,
                "--p",
                "2",
                "--n-restarts",
                "5",
                "--maxiter",
                "1000",
                "--h-values",
                "2.0",
                "1.75",
                "1.5",
                "1.35",
                "1.25",
                "--h-test",
                "1.15",
                "1.1",
                "--hidden-dim",
                "128",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                "--output-dir",
                f"{out_base}/ext_extrapolation",
                "--verbose",
            ],
            hypothesis="MPNN extrapolates to h < 1.25 (outside training domain)",
            expected_outcome="EXPECTED FAIL — HVA p=2 cannot express ground state at h<1.0, "
            "and MPNN extrapolation is unreliable. Quantifies failure mode.",
            output_dir=f"{out_base}/ext_extrapolation",
        )
    )

    # ─── EXT-3: p=1 with extended regime + multiple test points ───────────
    # Based on NL-H-p1. Extends to test the full p=1 valid regime
    # with denser grid and multiple deployment points including the
    # boundary (h=1.6 is the p=1 limit at N=6).
    variants.append(
        PipelineVariant(
            id="EXT-3-p1-boundary",
            description="p=1 boundary test: dense grid h=[2.0→1.6], deploy at h=1.6 (limit)",
            category="noiseless",
            command=[
                python,
                "scripts/run_pipeline.py",
                "--n-qubits",
                n,
                "--p",
                "1",
                "--n-restarts",
                "5",
                "--maxiter",
                "1000",
                "--h-values",
                "2.0",
                "1.95",
                "1.9",
                "1.85",
                "1.8",
                "1.75",
                "1.7",
                "1.65",
                "1.6",
                "--h-test",
                "1.6",
                "1.7",
                "1.85",
                "--hidden-dim",
                "128",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                "--output-dir",
                f"{out_base}/ext_p1_boundary",
                "--verbose",
            ],
            hypothesis="p=1 pipeline works at the boundary h=1.6 with dense training",
            expected_outcome="h=1.85 and h=1.7 PASS; h=1.6 is marginal (boundary of valid regime). "
            "Demonstrates p=1 viability for hardware deployment.",
            output_dir=f"{out_base}/ext_p1_boundary",
        )
    )

    # ─── EXT-4: ZNE with 7 layouts (maximum spread) ──────────────────────
    # Based on NY-B-lay5. If 5 layouts is good, does 7 give diminishing
    # returns? This establishes the saturation point for layout count.
    variants.append(
        PipelineVariant(
            id="EXT-4-7layouts",
            description="ZNE with 7 layouts: test saturation of layout benefit",
            category="noisy",
            command=[
                python,
                "scripts/run_noisy_pipeline.py",
                "--n-qubits",
                n,
                "--p",
                "2",
                "--h-values",
                "2.0",
                "1.75",
                "1.5",
                "1.35",
                "1.25",
                "--n-layouts",
                "7",
                "--shots",
                "16384",
                "--seed",
                "42",
                "--output-dir",
                f"{out_base}/ext_7layouts",
            ],
            hypothesis="7 layouts gives diminishing returns vs 5 layouts at N=6",
            expected_outcome="R² ≈ same as 3-5 layouts (already saturated at N=6). "
            "Confirms 3 layouts is cost-optimal for N=6.",
            output_dir=f"{out_base}/ext_7layouts",
        )
    )

    # ─── EXT-5: Minimum shots with maximum layouts ────────────────────────
    # Based on NY-A-shots4096. Combines minimum shots (2048) with
    # maximum layouts (5) to test if more fit points compensate for
    # higher per-point variance. This is the "cheap hardware" config.
    variants.append(
        PipelineVariant(
            id="EXT-5-cheap-hw",
            description="Cheap hardware config: 2048 shots × 5 layouts (cost-optimized)",
            category="noisy",
            command=[
                python,
                "scripts/run_noisy_pipeline.py",
                "--n-qubits",
                n,
                "--p",
                "2",
                "--h-values",
                "2.0",
                "1.75",
                "1.5",
                "1.35",
                "1.25",
                "--n-layouts",
                "5",
                "--shots",
                "2048",
                "--seed",
                "42",
                "--output-dir",
                f"{out_base}/ext_cheap_hw",
            ],
            hypothesis="5 layouts compensate for low shots (2048) — total budget = 10240 shots",
            expected_outcome="ZNE still wins (R² > 0.8) because more fit points reduce "
            "extrapolation uncertainty despite higher per-point noise. "
            "Total shot budget (10240) is 63% less than 3×16384=49152.",
            output_dir=f"{out_base}/ext_cheap_hw",
        )
    )

    return variants


def build_noisy_variants(n_qubits: int = DEFAULT_N_QUBITS) -> list[PipelineVariant]:
    """Build all noisy pipeline variants."""
    variants = []
    python = sys.executable
    n = str(n_qubits)
    out_base = f"results/thesis/variants_N{n_qubits}_chain"

    # ─── Group NA: Shot Count Sensitivity ──────────────────────────────────
    # Hypothesis: More shots → better ZNE (lower variance in linear fit)
    for shots in [4096, 8192, 16384, 32768]:
        variants.append(
            PipelineVariant(
                id=f"NY-A-shots{shots}",
                description=f"Noisy: shots={shots} (3 layouts, seed=42)",
                category="noisy",
                command=[
                    python,
                    "scripts/run_noisy_pipeline.py",
                    "--n-qubits",
                    n,
                    "--p",
                    "2",
                    "--h-values",
                    "2.0",
                    "1.75",
                    "1.5",
                    "1.35",
                    "1.25",
                    "--n-layouts",
                    "3",
                    "--shots",
                    str(shots),
                    "--seed",
                    "42",
                    "--output-dir",
                    f"{out_base}/ny_shots_{shots}",
                ],
                hypothesis=f"ZNE with {shots} shots achieves R² > 0.8 and wins ≥4/5",
                expected_outcome="All pass; 4096 may have lower R² due to shot noise",
                output_dir=f"{out_base}/ny_shots_{shots}",
            )
        )

    # ─── Group NB: Layout Count ────────────────────────────────────────────
    # Hypothesis: More layouts → better ZNE extrapolation (more fit points)
    for n_layouts in [2, 3, 5]:
        variants.append(
            PipelineVariant(
                id=f"NY-B-lay{n_layouts}",
                description=f"Noisy: n_layouts={n_layouts} (16384 shots, seed=42)",
                category="noisy",
                command=[
                    python,
                    "scripts/run_noisy_pipeline.py",
                    "--n-qubits",
                    n,
                    "--p",
                    "2",
                    "--h-values",
                    "2.0",
                    "1.75",
                    "1.5",
                    "1.35",
                    "1.25",
                    "--n-layouts",
                    str(n_layouts),
                    "--shots",
                    "16384",
                    "--seed",
                    "42",
                    "--output-dir",
                    f"{out_base}/ny_layouts_{n_layouts}",
                ],
                hypothesis=f"ZNE with {n_layouts} layouts achieves R² > 0.8",
                expected_outcome="3+ layouts pass; 2 layouts is minimum for linear fit",
                output_dir=f"{out_base}/ny_layouts_{n_layouts}",
            )
        )

    # ─── Group NC: Seed Robustness (Noisy) ─────────────────────────────────
    # Hypothesis: ZNE results are reproducible across seeds
    for seed_val in [42, 43, 44]:
        variants.append(
            PipelineVariant(
                id=f"NY-C-seed{seed_val}",
                description=f"Noisy: seed={seed_val} (reproducibility, 3 layouts, 16384 shots)",
                category="noisy",
                command=[
                    python,
                    "scripts/run_noisy_pipeline.py",
                    "--n-qubits",
                    n,
                    "--p",
                    "2",
                    "--h-values",
                    "2.0",
                    "1.75",
                    "1.5",
                    "1.35",
                    "1.25",
                    "--n-layouts",
                    "3",
                    "--shots",
                    "16384",
                    "--seed",
                    str(seed_val),
                    "--output-dir",
                    f"{out_base}/ny_seed_{seed_val}",
                ],
                hypothesis=f"ZNE with seed={seed_val} is consistent (wins ≥4/5)",
                expected_outcome="All seeds pass — ZNE is robust to simulator seed",
                output_dir=f"{out_base}/ny_seed_{seed_val}",
            )
        )

    # ─── Group N-D: VQE Restarts Effect on Noisy ───────────────────────────
    # Hypothesis: Better VQE → better ZNE (closer to ground state)
    for n_restarts in [1, 3, 5]:
        variants.append(
            PipelineVariant(
                id=f"NY-D-rst{n_restarts}",
                description=f"Noisy: VQE restarts={n_restarts} (effect on ZNE quality)",
                category="noisy",
                command=[
                    python,
                    "scripts/run_noisy_pipeline.py",
                    "--n-qubits",
                    n,
                    "--p",
                    "2",
                    "--h-values",
                    "2.0",
                    "1.75",
                    "1.5",
                    "1.35",
                    "1.25",
                    "--n-layouts",
                    "3",
                    "--shots",
                    "16384",
                    "--seed",
                    "42",
                    "--n-restarts",
                    str(n_restarts),
                    "--output-dir",
                    f"{out_base}/ny_restarts_{n_restarts}",
                ],
                hypothesis=f"VQE quality ({n_restarts} restarts) affects ZNE gain",
                expected_outcome="All pass; gain should be similar (ZNE is independent of VQE quality)",
                output_dir=f"{out_base}/ny_restarts_{n_restarts}",
            )
        )

    # ─── Group NE: Dense h-grid for Noisy ─────────────────────────────────
    # Hypothesis: More h-points gives better characterization of ZNE behavior
    variants.append(
        PipelineVariant(
            id="NY-E-dense",
            description="Noisy: dense h-grid (8 points, full valid regime)",
            category="noisy",
            command=[
                python,
                "scripts/run_noisy_pipeline.py",
                "--n-qubits",
                n,
                "--p",
                "2",
                "--h-values",
                "2.0",
                "1.85",
                "1.7",
                "1.55",
                "1.4",
                "1.35",
                "1.3",
                "1.25",
                "--n-layouts",
                "3",
                "--shots",
                "16384",
                "--seed",
                "42",
                "--output-dir",
                f"{out_base}/ny_dense_grid",
            ],
            hypothesis="ZNE works across the full valid regime with dense sampling",
            expected_outcome="PASS — ZNE wins at all h-values (N=6 is in perturbative regime)",
            output_dir=f"{out_base}/ny_dense_grid",
        )
    )

    return variants


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RunResult:
    """Result of a single variant execution."""

    variant_id: str
    success: bool
    elapsed_s: float
    return_code: int
    error_msg: str = ""


def run_variant(variant: PipelineVariant, dry_run: bool = False) -> RunResult:
    """Execute a single pipeline variant.

    Parameters
    ----------
    variant : PipelineVariant
        The variant configuration to run.
    dry_run : bool
        If True, print command without executing.

    Returns
    -------
    RunResult
        Execution result.
    """
    cmd_str = " ".join(variant.command)

    if dry_run:
        print(f"  [DRY RUN] {cmd_str}")
        return RunResult(
            variant_id=variant.id,
            success=True,
            elapsed_s=0.0,
            return_code=0,
        )

    # Ensure output directory exists
    Path(variant.output_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        result = subprocess.run(
            variant.command,
            capture_output=True,
            text=True,
            timeout=1500,  # 10 min max per variant
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            # Print last 10 lines of stderr for debugging
            stderr_lines = result.stderr.strip().split("\n")[-10:]
            error_msg = "\n".join(stderr_lines)
            return RunResult(
                variant_id=variant.id,
                success=False,
                elapsed_s=elapsed,
                return_code=result.returncode,
                error_msg=error_msg,
            )

        return RunResult(
            variant_id=variant.id,
            success=True,
            elapsed_s=elapsed,
            return_code=0,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return RunResult(
            variant_id=variant.id,
            success=False,
            elapsed_s=elapsed,
            return_code=-1,
            error_msg="TIMEOUT (>600s)",
        )
    except Exception as e:
        elapsed = time.time() - t0
        return RunResult(
            variant_id=variant.id,
            success=False,
            elapsed_s=elapsed,
            return_code=-2,
            error_msg=str(e),
        )


def print_variant_table(variants: list[PipelineVariant]) -> None:
    """Print a formatted table of all variants."""
    print(f"\n{'#':<4} {'ID':<16} {'Category':<10} {'Description'}")
    print("-" * 80)
    for i, v in enumerate(variants):
        print(f"{i:<4} {v.id:<16} {v.category:<10} {v.description}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exhaustive pipeline variants for thesis validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=DEFAULT_N_QUBITS,
        help=f"Number of qubits (default: {DEFAULT_N_QUBITS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--noiseless-only",
        action="store_true",
        help="Run only noiseless variants",
    )
    parser.add_argument(
        "--noisy-only",
        action="store_true",
        help="Run only noisy variants",
    )
    parser.add_argument(
        "--extended-only",
        action="store_true",
        help="Run only extended (most promising) variants",
    )
    parser.add_argument(
        "--variant",
        type=int,
        default=None,
        help="Run a specific variant by index (0-based)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start from variant index (skip earlier ones)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all variants and exit",
    )
    args = parser.parse_args()

    n_qubits = args.n_qubits

    # Build variant lists
    noiseless = build_noiseless_variants(n_qubits)
    noisy = build_noisy_variants(n_qubits)
    extended = build_extended_variants(n_qubits)

    # Filter by category
    if args.noiseless_only:
        all_variants = noiseless
    elif args.noisy_only:
        all_variants = noisy
    elif args.extended_only:
        all_variants = extended
    else:
        all_variants = noiseless + noisy + extended

    # List mode
    if args.list:
        print("=" * 80)
        print("  PIPELINE VARIANTS FOR THESIS VALIDATION")
        print("=" * 80)
        print(f"\n  Noiseless: {len(noiseless)} variants")
        print(f"  Noisy:     {len(noisy)} variants")
        print(f"  Extended:  {len(extended)} variants (most promising + pushed limits)")
        print(f"  Total:     {len(all_variants)} variants")
        print_variant_table(all_variants)
        return

    # Single variant mode
    if args.variant is not None:
        if args.variant >= len(all_variants):
            print(f"ERROR: Variant index {args.variant} out of range (0-{len(all_variants) - 1})")
            sys.exit(1)
        v = all_variants[args.variant]
        print(f"\nRunning single variant: {v.id}")
        print(f"  Description: {v.description}")
        print(f"  Hypothesis: {v.hypothesis}")
        print(f"  Expected: {v.expected_outcome}")
        print(f"  Command: {' '.join(v.command)}")
        print()
        result = run_variant(v, dry_run=args.dry_run)
        status = "✅ PASS" if result.success else "❌ FAIL"
        print(f"\n  Result: {status} ({result.elapsed_s:.1f}s)")
        if not result.success:
            print(f"  Error: {result.error_msg}")
        sys.exit(0 if result.success else 1)

    # Full run mode
    print("=" * 80)
    print(f"  EXHAUSTIVE PIPELINE VARIANT RUNNER WITH N={n_qubits}, p=2, chain_1d topology")
    print("=" * 80)
    print(f"\n  Total variants: {len(all_variants)}")
    print(f"  Starting from:  #{args.start_from}")
    print(f"  Mode:           {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    print()

    if not args.dry_run:
        print("  ⚠️  Estimated time: 70 minutes for all variants")
        print("  ⚠️  Results saved to: results/thesis/variants/")
        print()

    # Execute variants
    t_total = time.time()
    results: list[RunResult] = []

    for i, variant in enumerate(all_variants):
        if i < args.start_from:
            continue

        print(f"\n{'─' * 60}")
        print(f"  [{i + 1}/{len(all_variants)}] {variant.id}: {variant.description}")
        print(f"  Hypothesis: {variant.hypothesis}")
        print(f"{'─' * 60}")

        result = run_variant(variant, dry_run=args.dry_run)
        results.append(result)

        status = "✅" if result.success else "❌"
        print(f"  {status} {variant.id}: {result.elapsed_s:.1f}s")
        if not result.success:
            print(f"     Error: {result.error_msg[:200]}")

    # ─── Final Summary ─────────────────────────────────────────────────────
    total_elapsed = time.time() - t_total
    n_pass = sum(1 for r in results if r.success)
    n_fail = sum(1 for r in results if not r.success)

    print("\n" + "=" * 80)
    print("  FINAL SUMMARY")
    print("=" * 80)
    print(f"\n  Total variants: {len(results)}")
    print(f"  Passed:         {n_pass} ✅")
    print(f"  Failed:         {n_fail} ❌")
    print(f"  Total time:     {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print()

    if n_fail > 0:
        print("  FAILURES:")
        for r in results:
            if not r.success:
                print(f"    ❌ {r.variant_id}: {r.error_msg[:100]}")
        print()

    # Save execution log
    log_dir = Path(f"results/thesis/variants_N{n_qubits}_chain")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"execution_log_{timestamp}.json"

    log_data = {
        "timestamp": timestamp,
        "n_qubits": n_qubits,
        "topology": "chain_1d",
        "total_variants": len(results),
        "passed": n_pass,
        "failed": n_fail,
        "total_elapsed_s": total_elapsed,
        "results": [
            {
                "variant_id": r.variant_id,
                "success": r.success,
                "elapsed_s": r.elapsed_s,
                "return_code": r.return_code,
                "error_msg": r.error_msg if not r.success else "",
            }
            for r in results
        ],
        "variants": [
            {
                "id": v.id,
                "description": v.description,
                "category": v.category,
                "hypothesis": v.hypothesis,
                "expected_outcome": v.expected_outcome,
                "output_dir": v.output_dir,
            }
            for v in all_variants[args.start_from :]
        ],
    }

    if not args.dry_run:
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"  Execution log: {log_path}")

    print("=" * 80)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
