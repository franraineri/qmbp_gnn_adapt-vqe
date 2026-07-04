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

import sys

from qmbp_simulation.framework.variant_runner import PipelineVariant, run_variant_script

# ═══════════════════════════════════════════════════════════════════════════
# Default system size (overridable via --n-qubits CLI argument)
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_N_QUBITS = 10


# ═══════════════════════════════════════════════════════════════════════════
# Variant definitions
# ═══════════════════════════════════════════════════════════════════════════


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
    for seed_val in DEFAULT_SEEDS:
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
    for seed_val in DEFAULT_SEEDS:
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
# Entry point — delegates to framework's shared variant runner
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    run_variant_script(
        topology="chain_1d",
        default_n_qubits=DEFAULT_N_QUBITS,
        build_noiseless=build_noiseless_variants,
        build_noisy=build_noisy_variants,
        build_extended=build_extended_variants,
        timeout=1500,
    )
