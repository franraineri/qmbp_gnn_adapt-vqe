#!/usr/bin/env python3
"""Exhaustive pipeline variant runner — N=10, p=2, TRIANGULAR topology.

Tests the framework's generalization to a 2D frustrated topology.
The triangular lattice has coordination number up to 6, which means:
  - Highest entanglement among supported topologies → VQE needs more restarts
  - Geometric frustration → richer energy landscape, harder optimization
  - Tests GNN's ability to learn on highly-connected graphs
  - Physically relevant: triangular antiferromagnets exhibit exotic phases

Key differences from chain_1d / ladder at N=10:
  - Triangular has ~3N/2 bonds (vs N-1 for chain, ~3N/2 for ladder)
  - Higher coordination → valid regime shifts to higher h
  - Frustration effects may create local minima in VQE landscape
  - MPNN must handle higher-degree nodes (edge features critical)

Usage:
    python scripts/run_thesis_variants-triangular.py --list
    python scripts/run_thesis_variants-triangular.py --dry-run
    python scripts/run_thesis_variants-triangular.py
    python scripts/run_thesis_variants-triangular.py --noiseless-only
    python scripts/run_thesis_variants-triangular.py --extended-only
    python scripts/run_thesis_variants-triangular.py --variant 0
    python scripts/run_thesis_variants-triangular.py --n-qubits 6
"""

from __future__ import annotations

import sys

from qmbp_simulation.framework.variant_runner import PipelineVariant, run_variant_script

# ═══════════════════════════════════════════════════════════════════════════
# Constants for this topology variant
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_N_QUBITS = 10
P_LAYERS = 2
TOPOLOGY = "triangular"
DEFAULT_SEED = 43  # Best-performing seed (from project-status validation)
# Triangular lattice at N=10 has the highest connectivity among supported
# topologies. More ZZ bonds → stronger coupling → paramagnetic phase requires
# higher h to dominate. Expected valid regime: h≥3.0 (shifted +1.0 vs ladder).
# We use h≥3.0 as the safe range and probe the boundary at h=2.5-3.0.
BASE_H_VALUES = ["5.0", "4.5", "4.0", "3.5", "3.0"]
EXTENDED_H_VALUES = ["5.0", "4.5", "4.0", "3.75", "3.5", "3.25", "3.0"]
DENSE_H_VALUES = [
    "5.0",
    "4.75",
    "4.5",
    "4.25",
    "4.0",
    "3.75",
    "3.5",
    "3.25",
    "3.0",
]


# ═══════════════════════════════════════════════════════════════════════════
# Variant definitions
# ═══════════════════════════════════════════════════════════════════════════


def build_noiseless_variants(n_qubits: int = DEFAULT_N_QUBITS) -> list[PipelineVariant]:
    """Build noiseless pipeline variants for triangular topology."""
    variants = []
    python = sys.executable
    output_base = f"results/thesis/variants_N{n_qubits}_{TOPOLOGY}"
    seed_args = ["--seed", str(DEFAULT_SEED)]

    # ─── Group A: VQE Restart Sensitivity (Triangular) ─────────────────────
    # Triangular has frustration → more local minima → likely needs ≥7 restarts
    for n_restarts in [1, 3, 5, 7]:
        variants.append(
            PipelineVariant(
                id=f"NL-A{n_restarts}",
                description=f"VQE restarts={n_restarts} (triangular, hidden=128)",
                category="noiseless",
                command=[
                    python,
                    "scripts/run_pipeline.py",
                    "--n-qubits",
                    str(n_qubits),
                    "--p",
                    str(P_LAYERS),
                    "--topology",
                    TOPOLOGY,
                    "--n-restarts",
                    str(n_restarts),
                    "--maxiter",
                    "1000",
                    "--h-values",
                    *BASE_H_VALUES,
                    "--h-test",
                    "3.5",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    *seed_args,
                    "--output-dir",
                    f"{output_base}/nl_restarts_{n_restarts}",
                    "--verbose",
                ],
                hypothesis=f"VQE with {n_restarts} restart(s) on triangular converges to ΔE/gap < 5%",
                expected_outcome="Frustration creates local minima — likely needs ≥5 restarts. "
                "1 restart almost certainly fails.",
                output_dir=f"{output_base}/nl_restarts_{n_restarts}",
            )
        )

    # ─── Group B: MPNN Hidden Dimension (Triangular needs more capacity?) ──
    # Higher connectivity → richer graph features → may need larger hidden dim
    for hidden_dim in [64, 128, 256]:
        variants.append(
            PipelineVariant(
                id=f"NL-B{hidden_dim}",
                description=f"MPNN hidden_dim={hidden_dim} (triangular, 7 restarts)",
                category="noiseless",
                command=[
                    python,
                    "scripts/run_pipeline.py",
                    "--n-qubits",
                    str(n_qubits),
                    "--p",
                    str(P_LAYERS),
                    "--topology",
                    TOPOLOGY,
                    "--n-restarts",
                    "7",
                    "--maxiter",
                    "1000",
                    "--h-values",
                    *BASE_H_VALUES,
                    "--h-test",
                    "3.5",
                    "--hidden-dim",
                    str(hidden_dim),
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    *seed_args,
                    "--output-dir",
                    f"{output_base}/nl_hidden_{hidden_dim}",
                    "--verbose",
                ],
                hypothesis=f"MPNN hidden_dim={hidden_dim} on triangular achieves ΔE/gap < 5%",
                expected_outcome="h=128 likely sufficient (GINConv is expressive). "
                "h=256 tests if extra capacity helps on frustrated lattice.",
                output_dir=f"{output_base}/nl_hidden_{hidden_dim}",
            )
        )

    # ─── Group C: h-Grid Density (Triangular) ───────────────────────────────
    # Frustrated lattice → steeper energy landscape → may need denser grid
    h_grids = {
        "sparse5": BASE_H_VALUES,
        "standard7": EXTENDED_H_VALUES,
        "dense9": DENSE_H_VALUES,
    }
    for grid_name, h_vals in h_grids.items():
        variants.append(
            PipelineVariant(
                id=f"NL-C-{grid_name}",
                description=f"h-grid: {grid_name} ({len(h_vals)} pts, triangular)",
                category="noiseless",
                command=[
                    python,
                    "scripts/run_pipeline.py",
                    "--n-qubits",
                    str(n_qubits),
                    "--p",
                    str(P_LAYERS),
                    "--topology",
                    TOPOLOGY,
                    "--n-restarts",
                    "7",
                    "--maxiter",
                    "1000",
                    "--h-values",
                    *h_vals,
                    "--h-test",
                    "3.5",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    *seed_args,
                    "--output-dir",
                    f"{output_base}/nl_grid_{grid_name}",
                    "--verbose",
                ],
                hypothesis=f"Grid '{grid_name}' sufficient for triangular MPNN training",
                expected_outcome="Dense grid should pass; sparse5 may fail on triangular "
                "(frustrated landscape needs more training points for smooth interpolation).",
                output_dir=f"{output_base}/nl_grid_{grid_name}",
            )
        )

    # ─── Group D: h-Test Points (Triangular) ──────────────────────────────
    # Test generalization at different points in the triangular valid regime
    h_test_configs = {
        "safe": (["3.5"], "Safe point (deep in paramagnetic phase)"),
        "boundary": (["3.0"], "Boundary of expected valid regime"),
        "multi": (["4.5", "3.75", "3.25"], "Multiple points across regime"),
    }
    for test_name, (h_tests, desc) in h_test_configs.items():
        variants.append(
            PipelineVariant(
                id=f"NL-D-{test_name}",
                description=f"h_test: {desc} (triangular)",
                category="noiseless",
                command=[
                    python,
                    "scripts/run_pipeline.py",
                    "--n-qubits",
                    str(n_qubits),
                    "--p",
                    str(P_LAYERS),
                    "--topology",
                    TOPOLOGY,
                    "--n-restarts",
                    "7",
                    "--maxiter",
                    "1000",
                    "--h-values",
                    *EXTENDED_H_VALUES,
                    "--h-test",
                    *h_tests,
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    *seed_args,
                    "--output-dir",
                    f"{output_base}/nl_htest_{test_name}",
                    "--verbose",
                ],
                hypothesis=f"MPNN generalizes on triangular to h_test={h_tests}",
                expected_outcome="Safe and multi should pass; boundary (h=3.0) may be marginal "
                "due to frustration effects near the valid regime edge.",
                output_dir=f"{output_base}/nl_htest_{test_name}",
            )
        )

    # ─── Group E: Seed Robustness (Triangular) ──────────────────────────────
    # Frustration may break seed independence (multiple degenerate minima)
    for seed_val in DEFAULT_SEEDS:
        variants.append(
            PipelineVariant(
                id=f"NL-E-seed{seed_val}",
                description=f"Seed={seed_val} (triangular, reproducibility)",
                category="noiseless",
                command=[
                    python,
                    "scripts/run_pipeline.py",
                    "--n-qubits",
                    str(n_qubits),
                    "--p",
                    str(P_LAYERS),
                    "--topology",
                    TOPOLOGY,
                    "--n-restarts",
                    "7",
                    "--maxiter",
                    "1000",
                    "--h-values",
                    *BASE_H_VALUES,
                    "--h-test",
                    "3.5",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--seed",
                    str(seed_val),
                    "--output-dir",
                    f"{output_base}/nl_seed_{seed_val}",
                    "--verbose",
                ],
                hypothesis=f"Triangular pipeline with seed={seed_val} is reproducible",
                expected_outcome="May show higher variance than chain/ladder due to frustration. "
                "Tests if 7 restarts is sufficient to overcome seed dependence.",
                output_dir=f"{output_base}/nl_seed_{seed_val}",
            )
        )

    # ─── Group F: Triangular vs Chain/Ladder comparison ────────────────────
    # Run chain_1d and ladder with same h-range for direct comparison
    for ref_topology in ["chain_1d", "ladder"]:
        variants.append(
            PipelineVariant(
                id=f"NL-F-{ref_topology.replace('_', '')}",
                description=f"{ref_topology} baseline (same h-range for comparison)",
                category="noiseless",
                command=[
                    python,
                    "scripts/run_pipeline.py",
                    "--n-qubits",
                    str(n_qubits),
                    "--p",
                    str(P_LAYERS),
                    "--topology",
                    ref_topology,
                    "--n-restarts",
                    "7",
                    "--maxiter",
                    "1000",
                    "--h-values",
                    *BASE_H_VALUES,
                    "--h-test",
                    "3.5",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    *seed_args,
                    "--output-dir",
                    f"{output_base}/nl_{ref_topology}_baseline",
                    "--verbose",
                ],
                hypothesis=f"{ref_topology} baseline at same h-range for topology comparison",
                expected_outcome=f"PASS — {ref_topology} at h≥3.0 is well within valid regime. "
                "Provides reference for triangular performance delta.",
                output_dir=f"{output_base}/nl_{ref_topology}_baseline",
            )
        )

    # ─── Group G: p=1 on Triangular ───────────────────────────────────────
    # p=1 on triangular: fewer parameters but highest connectivity
    variants.append(
        PipelineVariant(
            id="NL-G-p1-tri",
            description="p=1 layer on triangular (2 params, highest connectivity)",
            category="noiseless",
            command=[
                python,
                "scripts/run_pipeline.py",
                "--n-qubits",
                str(n_qubits),
                "--p",
                "1",
                "--topology",
                TOPOLOGY,
                "--n-restarts",
                "7",
                "--maxiter",
                "1000",
                "--h-values",
                "5.0",
                "4.5",
                "4.0",
                "3.5",
                "--h-test",
                "4.0",
                "--hidden-dim",
                "128",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                *seed_args,
                "--output-dir",
                f"{output_base}/nl_p1_triangular",
                "--verbose",
            ],
            hypothesis="p=1 HVA on triangular works at h≥3.5 (narrower regime than p=2)",
            expected_outcome="Valid regime likely shifts up significantly vs p=2 due to "
            "fewer params + high connectivity. Tests p=1 viability on 2D lattices.",
            output_dir=f"{output_base}/nl_p1_triangular",
        )
    )

    # ─── Group H: MPNN Training Epochs (Triangular) ───────────────────────
    # Frustrated landscape may need more epochs to learn
    for n_epochs in [4000, 6000, 8000]:
        variants.append(
            PipelineVariant(
                id=f"NL-H-ep{n_epochs}",
                description=f"MPNN epochs={n_epochs} (triangular, patience=500)",
                category="noiseless",
                command=[
                    python,
                    "scripts/run_pipeline.py",
                    "--n-qubits",
                    str(n_qubits),
                    "--p",
                    str(P_LAYERS),
                    "--topology",
                    TOPOLOGY,
                    "--n-restarts",
                    "7",
                    "--maxiter",
                    "1000",
                    "--h-values",
                    *BASE_H_VALUES,
                    "--h-test",
                    "3.5",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    str(n_epochs),
                    "--patience",
                    "500",
                    *seed_args,
                    "--output-dir",
                    f"{output_base}/nl_epochs_{n_epochs}",
                    "--verbose",
                ],
                hypothesis=f"MPNN with {n_epochs} epochs on triangular converges sufficiently",
                expected_outcome="6000 should be sufficient (patience=500 handles early stopping). "
                "8000 tests if frustrated landscape benefits from longer training.",
                output_dir=f"{output_base}/nl_epochs_{n_epochs}",
            )
        )

    return variants


def build_noisy_variants(n_qubits: int = DEFAULT_N_QUBITS) -> list[PipelineVariant]:
    """Build noisy pipeline variants for triangular topology."""
    variants = []
    python = sys.executable
    output_base = f"results/thesis/variants_N{n_qubits}_{TOPOLOGY}"

    # ─── Group NA: Shot Count on Triangular ────────────────────────────────
    # More bonds → more Pauli terms → needs more shots for convergence
    for shots in [8192, 16384, 32768]:
        variants.append(
            PipelineVariant(
                id=f"NY-A-shots{shots}",
                description=f"Noisy triangular: shots={shots} (3 layouts)",
                category="noisy",
                command=[
                    python,
                    "scripts/run_noisy_pipeline.py",
                    "--n-qubits",
                    str(n_qubits),
                    "--p",
                    str(P_LAYERS),
                    "--topology",
                    TOPOLOGY,
                    "--h-values",
                    "5.0",
                    "4.0",
                    "3.5",
                    "--n-layouts",
                    "3",
                    "--shots",
                    str(shots),
                    "--seed",
                    "42",
                    "--output-dir",
                    f"{output_base}/ny_shots_{shots}",
                ],
                hypothesis=f"ZNE on triangular with {shots} shots achieves R² > 0.8",
                expected_outcome="N=10 triangular has the most CX gates → ZNE almost certainly "
                "fails. Known: ZNE fails at N=10 chain_1d (R²<0.05). "
                "Triangular has even more gates → expect worse failure.",
                output_dir=f"{output_base}/ny_shots_{shots}",
            )
        )

    # ─── Group NB: Layout Count on Triangular ─────────────────────────────
    for n_layouts in [3, 5, 7]:
        variants.append(
            PipelineVariant(
                id=f"NY-B-lay{n_layouts}",
                description=f"Noisy triangular: n_layouts={n_layouts} (16384 shots)",
                category="noisy",
                command=[
                    python,
                    "scripts/run_noisy_pipeline.py",
                    "--n-qubits",
                    str(n_qubits),
                    "--p",
                    str(P_LAYERS),
                    "--topology",
                    TOPOLOGY,
                    "--h-values",
                    "5.0",
                    "4.0",
                    "3.5",
                    "--n-layouts",
                    str(n_layouts),
                    "--shots",
                    "16384",
                    "--seed",
                    "42",
                    "--output-dir",
                    f"{output_base}/ny_layouts_{n_layouts}",
                ],
                hypothesis=f"ZNE with {n_layouts} layouts on triangular",
                expected_outcome="Expected to FAIL — N=10 ZNE fails regardless of layout count "
                "(Tsubouchi et al. 2023: cost grows exp(depth×qubits)). "
                "Triangular makes it worse. Documents the failure mode.",
                output_dir=f"{output_base}/ny_layouts_{n_layouts}",
            )
        )

    # ─── Group NC: Seed Robustness (Noisy Triangular) ─────────────────────
    for seed_val in DEFAULT_SEEDS:
        variants.append(
            PipelineVariant(
                id=f"NY-C-seed{seed_val}",
                description=f"Noisy triangular: seed={seed_val} (3 layouts, 16384 shots)",
                category="noisy",
                command=[
                    python,
                    "scripts/run_noisy_pipeline.py",
                    "--n-qubits",
                    str(n_qubits),
                    "--p",
                    str(P_LAYERS),
                    "--topology",
                    TOPOLOGY,
                    "--h-values",
                    "5.0",
                    "4.0",
                    "3.5",
                    "--n-layouts",
                    "3",
                    "--shots",
                    "16384",
                    "--seed",
                    str(seed_val),
                    "--output-dir",
                    f"{output_base}/ny_seed_{seed_val}",
                ],
                hypothesis=f"Noisy triangular seed={seed_val} — consistent failure mode",
                expected_outcome="All seeds should show same failure pattern (R²<0.1). "
                "Confirms failure is systematic, not stochastic.",
                output_dir=f"{output_base}/ny_seed_{seed_val}",
            )
        )

    return variants


def build_extended_variants(n_qubits: int = DEFAULT_N_QUBITS) -> list[PipelineVariant]:
    """Extended variants pushing triangular topology limits."""
    variants = []
    python = sys.executable
    output_base = f"results/thesis/variants_N{n_qubits}_{TOPOLOGY}"
    seed_args = ["--seed", str(DEFAULT_SEED)]

    # ─── EXT-1: Triangular at very high h (topology-independent regime) ───
    # At very high h, ZZ coupling is negligible → topology shouldn't matter.
    # Tests if triangular converges to same results as chain/ladder at h>>J.
    variants.append(
        PipelineVariant(
            id="EXT-1-high-h",
            description="Triangular at very high h=[5.0, 4.5, 4.0, 3.5, 3.0] (topology-independent)",
            category="noiseless",
            command=[
                python,
                "scripts/run_pipeline.py",
                "--n-qubits",
                str(n_qubits),
                "--p",
                str(P_LAYERS),
                "--topology",
                TOPOLOGY,
                "--n-restarts",
                "5",
                "--maxiter",
                "1000",
                "--h-values",
                *BASE_H_VALUES,
                "--h-test",
                "4.25",
                "3.75",
                "--hidden-dim",
                "128",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                *seed_args,
                "--output-dir",
                f"{output_base}/ext_high_h",
                "--verbose",
            ],
            hypothesis="At h>>J, triangular and chain give identical results (ZZ negligible)",
            expected_outcome="PASS with low ΔE/gap (<5%). Confirms topology-independence "
            "at high field. h_test points are interpolations between training points.",
            output_dir=f"{output_base}/ext_high_h",
        )
    )

    # ─── EXT-2: Dense grid near valid regime boundary ─────────────────────
    # Probe where the triangular valid regime actually starts.
    # Hypothesis: h_valid ≈ 3.0 for triangular (vs 2.0 for ladder, 1.5 for chain).
    variants.append(
        PipelineVariant(
            id="EXT-2-near-boundary",
            description="Triangular near valid boundary: dense grid h=[4.0→2.5] (boundary probe)",
            category="noiseless",
            command=[
                python,
                "scripts/run_pipeline.py",
                "--n-qubits",
                str(n_qubits),
                "--p",
                str(P_LAYERS),
                "--topology",
                TOPOLOGY,
                "--n-restarts",
                "7",
                "--maxiter",
                "1000",
                "--h-values",
                "4.0",
                "3.75",
                "3.5",
                "3.25",
                "3.0",
                "2.75",
                "2.5",
                "--h-test",
                "3.0",
                "2.75",
                "2.5",
                "--hidden-dim",
                "128",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                *seed_args,
                "--output-dir",
                f"{output_base}/ext_near_boundary",
                "--verbose",
            ],
            hypothesis="Triangular valid regime boundary is at h≈3.0",
            expected_outcome="h≥3.5 should pass; h=3.0 marginal; h=2.5 likely fails. "
            "Maps the valid regime boundary for triangular topology.",
            output_dir=f"{output_base}/ext_near_boundary",
        )
    )

    # ─── EXT-3: Kagome topology comparison ────────────────────────────────
    # Kagome is another frustrated 2D lattice (corner-sharing triangles).
    # Even more frustrated than triangular → hardest test of all.
    variants.append(
        PipelineVariant(
            id="EXT-3-kagome",
            description="Kagome topology N=10 (corner-sharing triangles, maximum frustration)",
            category="noiseless",
            command=[
                python,
                "scripts/run_pipeline.py",
                "--n-qubits",
                str(n_qubits),
                "--p",
                str(P_LAYERS),
                "--topology",
                "kagome",
                "--n-restarts",
                "7",
                "--maxiter",
                "1000",
                "--h-values",
                "6.0",
                "5.0",
                "4.5",
                "4.0",
                "--h-test",
                "4.5",
                "--hidden-dim",
                "128",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                *seed_args,
                "--output-dir",
                f"{output_base}/ext_kagome",
                "--verbose",
            ],
            hypothesis="Framework works on kagome lattice at high h",
            expected_outcome="Needs h≥4.0+ for valid regime (maximum frustration). "
            "Tests true topology-agnosticism of the GNN approach on the hardest lattice.",
            output_dir=f"{output_base}/ext_kagome",
        )
    )

    # ─── EXT-4: Deep GNN on Triangular ─────────────────────────────────────
    # Test if more GNN layers help on triangular (deeper message passing
    # for higher-diameter graph). Triangular N=10 has diameter ~3-4 hops.
    variants.append(
        PipelineVariant(
            id="EXT-4-deep-gnn",
            description="Triangular with 5 GNN layers (deeper message passing)",
            category="noiseless",
            command=[
                python,
                "scripts/run_pipeline.py",
                "--n-qubits",
                str(n_qubits),
                "--p",
                str(P_LAYERS),
                "--topology",
                TOPOLOGY,
                "--n-restarts",
                "7",
                "--maxiter",
                "1000",
                "--h-values",
                *EXTENDED_H_VALUES,
                "--h-test",
                "3.5",
                "--hidden-dim",
                "128",
                "--n-layers",
                "5",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                *seed_args,
                "--output-dir",
                f"{output_base}/ext_deep_gnn",
                "--verbose",
            ],
            hypothesis="5 GNN layers (vs default 3) improves prediction on triangular",
            expected_outcome="Marginal improvement expected — triangular diameter is small (~3 hops), "
            "so 3 layers already covers full graph. Tests over-smoothing risk.",
            output_dir=f"{output_base}/ext_deep_gnn",
        )
    )

    # ─── EXT-5: Noisy triangular with p=1 (fewer CX gates) ───────────────
    # p=1 on triangular has fewer CX gates → ZNE might work where p=2 fails
    variants.append(
        PipelineVariant(
            id="EXT-5-noisy-p1",
            description="Noisy triangular p=1 (fewer CX → ZNE may succeed where p=2 fails)",
            category="noisy",
            command=[
                python,
                "scripts/run_noisy_pipeline.py",
                "--n-qubits",
                str(n_qubits),
                "--p",
                "1",
                "--topology",
                TOPOLOGY,
                "--h-values",
                "5.0",
                "4.5",
                "4.0",
                "--n-layouts",
                "3",
                "--shots",
                "16384",
                "--seed",
                "42",
                "--output-dir",
                f"{output_base}/ext_noisy_p1",
            ],
            hypothesis="p=1 triangular has ~50% fewer CX gates → ZNE may work at N=10",
            expected_outcome="Even with p=1, triangular has many bonds → CX count still high. "
            "Likely fails but quantifies the CX-budget hypothesis for frustrated lattices.",
            output_dir=f"{output_base}/ext_noisy_p1",
        )
    )

    # ─── EXT-6: Small system (N=6) triangular with optimal config ──────────
    # N=6 triangular with the proven-optimal config (hidden=128, standard7 grid)
    # to establish the N=6 baseline for this topology.
    variants.append(
        PipelineVariant(
            id="EXT-6-N6-opt",
            description="Triangular N=6 optimal config (hidden=128, 7pts, 7 restarts)",
            category="noiseless",
            command=[
                python,
                "scripts/run_pipeline.py",
                "--n-qubits",
                "6",
                "--p",
                str(P_LAYERS),
                "--topology",
                TOPOLOGY,
                "--n-restarts",
                "7",
                "--maxiter",
                "1000",
                "--h-values",
                *EXTENDED_H_VALUES,
                "--h-test",
                "3.5",
                "3.25",
                "--hidden-dim",
                "128",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                *seed_args,
                "--output-dir",
                f"{output_base}/ext_N6_optimal",
                "--verbose",
            ],
            hypothesis="Triangular N=6 with optimal config achieves ΔE/gap < 5%",
            expected_outcome="PASS — uses proven-optimal parameters (hidden=128, 7pts, 7 restarts). "
            "Establishes the N=6 triangular baseline with best config.",
            output_dir=f"{output_base}/ext_N6_optimal",
        )
    )

    return variants


# ═══════════════════════════════════════════════════════════════════════════
# Entry point — delegates to framework's shared variant runner
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    run_variant_script(
        topology=TOPOLOGY,
        default_n_qubits=DEFAULT_N_QUBITS,
        build_noiseless=build_noiseless_variants,
        build_noisy=build_noisy_variants,
        build_extended=build_extended_variants,
        timeout=1200,
    )
