#!/usr/bin/env python3
"""Exhaustive pipeline variant runner — N=10, p=2, LADDER topology.

Tests the framework's generalization to a quasi-2D topology (two-leg ladder).
The ladder has coordination number 3 (vs 2 for chain_1d), which means:
  - Higher entanglement → VQE may need more restarts
  - Different graph structure → tests GNN's topology-agnostic learning
  - Physically relevant: spin ladders exhibit richer phase behavior

Key differences from chain_1d at N=10:
  - Ladder has 5 rungs + 2 legs of length 5 = 13 bonds (vs 9 for chain)
  - Higher connectivity may shift the valid regime boundary
  - MPNN must learn from a different graph structure (edge features matter more)

Usage:
    python scripts/run_thesis_variants-ladder.py --list
    python scripts/run_thesis_variants-ladder.py --dry-run
    python scripts/run_thesis_variants-ladder.py
    python scripts/run_thesis_variants-ladder.py --noiseless-only
    python scripts/run_thesis_variants-ladder.py --extended-only
    python scripts/run_thesis_variants-ladder.py --variant 0
"""

from __future__ import annotations

import sys

from qmbp_simulation.framework.variant_runner import PipelineVariant, run_variant_script

# ═══════════════════════════════════════════════════════════════════════════
# Constants for this topology variant
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_N_QUBITS = 10
P_LAYERS = 2
TOPOLOGY = "ladder"
# Ladder at N=10 may have a different valid regime than chain_1d.
# chain_1d N=10: valid at h≥1.5. Ladder has more bonds → stronger ZZ →
# the paramagnetic phase may require higher h to dominate.
# From testing: ladder N=10 p=2 achieves fid≥0.93 only at h≥2.0 (with 5 restarts).
# The valid regime is shifted UP by ~0.5 compared to chain_1d.
# We use h≥2.0 as the safe range and probe the boundary at h=1.75-2.0.
BASE_H_VALUES = ["4.0", "3.5", "3.0", "2.5", "2.0"]
EXTENDED_H_VALUES = ["4.0", "3.5", "3.0", "2.75", "2.5", "2.25", "2.0"]
DENSE_H_VALUES = [
    "4.0",
    "3.75",
    "3.5",
    "3.25",
    "3.0",
    "2.75",
    "2.5",
    "2.25",
    "2.0",
]


# ═══════════════════════════════════════════════════════════════════════════
# Variant definitions
# ═══════════════════════════════════════════════════════════════════════════


def build_noiseless_variants(n_qubits: int = DEFAULT_N_QUBITS) -> list[PipelineVariant]:
    """Build noiseless pipeline variants for ladder topology."""
    variants = []
    python = sys.executable
    output_base = f"results/thesis/variants_N{n_qubits}_{TOPOLOGY}"

    # ─── Group A: VQE Restart Sensitivity (Ladder) ─────────────────────────
    # Ladder has higher connectivity → may need more restarts
    for n_restarts in [1, 3, 5, 7]:
        variants.append(
            PipelineVariant(
                id=f"NL-A{n_restarts}",
                description=f"VQE restarts={n_restarts} (ladder, hidden=128)",
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
                    "2.5",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--output-dir",
                    f"{output_base}/nl_restarts_{n_restarts}",
                    "--verbose",
                ],
                hypothesis=f"VQE with {n_restarts} restart(s) on ladder converges to ΔE/gap < 5%",
                expected_outcome="Ladder may need ≥5 restarts due to higher connectivity. "
                "1 restart likely fails.",
                output_dir=f"{output_base}/nl_restarts_{n_restarts}",
            )
        )

    # ─── Group B: MPNN Hidden Dimension (Ladder needs more capacity?) ──────
    # Ladder graph is more complex → MPNN may need larger hidden dim
    for hidden_dim in [64, 128, 256]:
        variants.append(
            PipelineVariant(
                id=f"NL-B{hidden_dim}",
                description=f"MPNN hidden_dim={hidden_dim} (ladder, 5 restarts)",
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
                    "2.5",
                    "--hidden-dim",
                    str(hidden_dim),
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--output-dir",
                    f"{output_base}/nl_hidden_{hidden_dim}",
                    "--verbose",
                ],
                hypothesis=f"MPNN hidden_dim={hidden_dim} on ladder achieves ΔE/gap < 5%",
                expected_outcome="h=128 is optimal for N=10 (from project-status). "
                "h=64 may underfit on ladder due to richer graph structure.",
                output_dir=f"{output_base}/nl_hidden_{hidden_dim}",
            )
        )

    # ─── Group C: h-Grid Density (Ladder) ─────────────────────────────────
    # More bonds → steeper energy landscape → may need denser grid
    h_grids = {
        "sparse5": BASE_H_VALUES,
        "standard7": EXTENDED_H_VALUES,
        "dense9": DENSE_H_VALUES,
    }
    for grid_name, h_vals in h_grids.items():
        variants.append(
            PipelineVariant(
                id=f"NL-C-{grid_name}",
                description=f"h-grid: {grid_name} ({len(h_vals)} pts, ladder)",
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
                    *h_vals,
                    "--h-test",
                    "2.5",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--output-dir",
                    f"{output_base}/nl_grid_{grid_name}",
                    "--verbose",
                ],
                hypothesis=f"Grid '{grid_name}' sufficient for ladder MPNN training",
                expected_outcome="Dense grid should pass; sparse3 may fail on ladder "
                "(steeper parameter landscape needs more training points).",
                output_dir=f"{output_base}/nl_grid_{grid_name}",
            )
        )

    # ─── Group D: h-Test Points (Ladder) ──────────────────────────────────
    # Test generalization at different points in the ladder's valid regime
    h_test_configs = {
        "safe": (["2.5"], "Safe point (deep in paramagnetic phase)"),
        "boundary": (["2.0"], "Boundary of expected valid regime"),
        "multi": (["3.5", "2.75", "2.25"], "Multiple points across regime"),
    }
    for test_name, (h_tests, desc) in h_test_configs.items():
        variants.append(
            PipelineVariant(
                id=f"NL-D-{test_name}",
                description=f"h_test: {desc} (ladder)",
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
                    *EXTENDED_H_VALUES,
                    "--h-test",
                    *h_tests,
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--output-dir",
                    f"{output_base}/nl_htest_{test_name}",
                    "--verbose",
                ],
                hypothesis=f"MPNN generalizes on ladder to h_test={h_tests}",
                expected_outcome="Safe and multi should pass; boundary may be marginal.",
                output_dir=f"{output_base}/nl_htest_{test_name}",
            )
        )

    # ─── Group E: Seed Robustness (Ladder) ─────────────────────────────────
    for seed_val in DEFAULT_SEEDS:
        variants.append(
            PipelineVariant(
                id=f"NL-E-seed{seed_val}",
                description=f"Seed={seed_val} (ladder, reproducibility)",
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
                    "2.5",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--output-dir",
                    f"{output_base}/nl_seed_{seed_val}",
                    "--verbose",
                ],
                hypothesis=f"Ladder pipeline with seed={seed_val} is reproducible",
                expected_outcome="All seeds pass — topology doesn't affect seed independence.",
                output_dir=f"{output_base}/nl_seed_{seed_val}",
            )
        )

    # ─── Group F: Ladder vs Chain comparison (same config) ─────────────────
    # Run the EXACT same config on chain_1d for direct comparison
    variants.append(
        PipelineVariant(
            id="NL-F-chain-baseline",
            description="Chain_1d baseline (same config as ladder for comparison)",
            category="noiseless",
            command=[
                python,
                "scripts/run_pipeline.py",
                "--n-qubits",
                str(n_qubits),
                "--p",
                str(P_LAYERS),
                "--topology",
                "chain_1d",
                "--n-restarts",
                "5",
                "--maxiter",
                "1000",
                "--h-values",
                *BASE_H_VALUES,
                "--h-test",
                "2.5",
                "--hidden-dim",
                "128",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                "--output-dir",
                f"{output_base}/nl_chain_baseline",
                "--verbose",
            ],
            hypothesis="Chain_1d baseline for direct topology comparison",
            expected_outcome="PASS — chain_1d at N=10 is well-validated. "
            "Provides reference for ladder performance delta.",
            output_dir=f"{output_base}/nl_chain_baseline",
        )
    )

    # ─── Group G: Periodic Boundary Conditions (Ladder) ────────────────────
    # Periodic ladder = cylinder topology (more bonds, higher entanglement)
    variants.append(
        PipelineVariant(
            id="NL-G-periodic",
            description="Periodic ladder (cylinder topology, max connectivity)",
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
                "--periodic",
                "--n-restarts",
                "7",
                "--maxiter",
                "1000",
                "--h-values",
                "4.0",
                "3.5",
                "3.0",
                "2.5",
                "--h-test",
                "3.0",
                "--hidden-dim",
                "128",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                "--output-dir",
                f"{output_base}/nl_periodic",
                "--verbose",
            ],
            hypothesis="Periodic ladder (cylinder) works with 7 restarts at h≥2.5",
            expected_outcome="May need even higher h threshold due to periodic bonds. "
            "Tests framework's handling of periodic boundary conditions.",
            output_dir=f"{output_base}/nl_periodic",
        )
    )

    # ─── Group H: p=1 on Ladder ───────────────────────────────────────────
    # p=1 on ladder: fewer parameters but higher connectivity
    variants.append(
        PipelineVariant(
            id="NL-H-p1-ladder",
            description="p=1 layer on ladder (2 params, higher connectivity)",
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
                "5",
                "--maxiter",
                "1000",
                "--h-values",
                "4.0",
                "3.5",
                "3.0",
                "2.5",
                "--h-test",
                "3.0",
                "--hidden-dim",
                "128",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                "--output-dir",
                f"{output_base}/nl_p1_ladder",
                "--verbose",
            ],
            hypothesis="p=1 HVA on ladder works at h≥2.5 (narrower regime than p=2)",
            expected_outcome="Valid regime likely shifts up vs p=2 due to fewer params. "
            "Tests if p=1 is viable on higher-connectivity topologies.",
            output_dir=f"{output_base}/nl_p1_ladder",
        )
    )

    return variants


def build_noisy_variants(n_qubits: int = DEFAULT_N_QUBITS) -> list[PipelineVariant]:
    """Build noisy pipeline variants for ladder topology."""
    variants = []
    python = sys.executable
    output_base = f"results/thesis/variants_N{n_qubits}_{TOPOLOGY}"

    # ─── Group NA: Shot Count on Ladder ────────────────────────────────────
    # More bonds → more Pauli terms → may need more shots for convergence
    for shots in [8192, 16384, 32768]:
        variants.append(
            PipelineVariant(
                id=f"NY-A-shots{shots}",
                description=f"Noisy ladder: shots={shots} (3 layouts)",
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
                    "4.0",
                    "3.0",
                    "2.5",
                    "--n-layouts",
                    "3",
                    "--shots",
                    str(shots),
                    "--seed",
                    "42",
                    "--output-dir",
                    f"{output_base}/ny_shots_{shots}",
                ],
                hypothesis=f"ZNE on ladder with {shots} shots achieves R² > 0.8",
                expected_outcome="N=10 ladder has more CX gates → ZNE may struggle. "
                "Known: ZNE fails at N=10 chain_1d (R²<0.05). "
                "Ladder has even more gates → expect failure.",
                output_dir=f"{output_base}/ny_shots_{shots}",
            )
        )

    # ─── Group NB: Layout Count on Ladder ──────────────────────────────────
    for n_layouts in [3, 5, 7]:
        variants.append(
            PipelineVariant(
                id=f"NY-B-lay{n_layouts}",
                description=f"Noisy ladder: n_layouts={n_layouts} (16384 shots)",
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
                    "4.0",
                    "3.0",
                    "2.5",
                    "--n-layouts",
                    str(n_layouts),
                    "--shots",
                    "16384",
                    "--seed",
                    "42",
                    "--output-dir",
                    f"{output_base}/ny_layouts_{n_layouts}",
                ],
                hypothesis=f"ZNE with {n_layouts} layouts on ladder",
                expected_outcome="Expected to FAIL — N=10 ZNE is known to fail regardless "
                "of layout count (Tsubouchi et al. 2023: cost grows exp(depth×qubits)). "
                "Ladder makes it worse. Documents the failure mode.",
                output_dir=f"{output_base}/ny_layouts_{n_layouts}",
            )
        )

    # ─── Group NC: Seed Robustness (Noisy Ladder) ──────────────────────────
    for seed_val in DEFAULT_SEEDS:
        variants.append(
            PipelineVariant(
                id=f"NY-C-seed{seed_val}",
                description=f"Noisy ladder: seed={seed_val} (3 layouts, 16384 shots)",
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
                    "4.0",
                    "3.0",
                    "2.5",
                    "--n-layouts",
                    "3",
                    "--shots",
                    "16384",
                    "--seed",
                    str(seed_val),
                    "--output-dir",
                    f"{output_base}/ny_seed_{seed_val}",
                ],
                hypothesis=f"Noisy ladder seed={seed_val} — consistent failure mode",
                expected_outcome="All seeds should show same failure pattern (R²<0.1). "
                "Confirms failure is systematic, not stochastic.",
                output_dir=f"{output_base}/ny_seed_{seed_val}",
            )
        )

    return variants


def build_extended_variants(n_qubits: int = DEFAULT_N_QUBITS) -> list[PipelineVariant]:
    """Extended variants pushing ladder topology limits."""
    variants = []
    python = sys.executable
    output_base = f"results/thesis/variants_N{n_qubits}_{TOPOLOGY}"

    # ─── EXT-1: Ladder with very high h (deep paramagnetic) ───────────────
    # At very high h, ZZ coupling is negligible → topology shouldn't matter.
    # Tests if ladder converges to same results as chain at h>>1.
    variants.append(
        PipelineVariant(
            id="EXT-1-high-h",
            description="Ladder at very high h=[3.0, 2.5, 2.0] (topology-independent regime)",
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
                "3",
                "--maxiter",
                "1000",
                "--h-values",
                "3.0",
                "2.5",
                "2.0",
                "--h-test",
                "2.5",
                "--hidden-dim",
                "128",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                "--output-dir",
                f"{output_base}/ext_high_h",
                "--verbose",
            ],
            hypothesis="At h>>J, ladder and chain give identical results (ZZ negligible)",
            expected_outcome="PASS with very low ΔE/gap (<0.1%). Confirms topology-independence "
            "at high field. Useful as sanity check.",
            output_dir=f"{output_base}/ext_high_h",
        )
    )

    # ─── EXT-2: Ladder with dense grid near phase transition ──────────────
    # The ladder has a different critical point than chain_1d.
    # For 2-leg ladder: h_c ≈ 1.0 (same universality class as chain).
    # But finite-size effects differ → test near h=2.0-3.0 (valid regime boundary).
    variants.append(
        PipelineVariant(
            id="EXT-2-near-hc",
            description="Ladder near valid boundary: dense grid h=[3.0→2.0] (boundary region)",
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
                "3.0",
                "2.75",
                "2.5",
                "2.25",
                "2.0",
                "--h-test",
                "2.25",
                "2.0",
                "--hidden-dim",
                "128",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                "--output-dir",
                f"{output_base}/ext_near_hc",
                "--verbose",
            ],
            hypothesis="Ladder pipeline works near valid boundary with 7 restarts",
            expected_outcome="h≥2.5 should pass; h=2.0 is marginal (fid≈0.93 boundary). "
            "Maps the valid regime boundary for ladder topology.",
            output_dir=f"{output_base}/ext_near_hc",
        )
    )

    # ─── EXT-3: Triangular topology comparison ────────────────────────────
    # Triangular has even higher connectivity (coord=4-6) → hardest test
    variants.append(
        PipelineVariant(
            id="EXT-3-triangular",
            description="Triangular topology N=10 (highest connectivity, hardest test)",
            category="noiseless",
            command=[
                python,
                "scripts/run_pipeline.py",
                "--n-qubits",
                str(n_qubits),
                "--p",
                str(P_LAYERS),
                "--topology",
                "triangular",
                "--n-restarts",
                "7",
                "--maxiter",
                "1000",
                "--h-values",
                "5.0",
                "4.0",
                "3.5",
                "3.0",
                "--h-test",
                "3.5",
                "--hidden-dim",
                "128",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                "--output-dir",
                f"{output_base}/ext_triangular",
                "--verbose",
            ],
            hypothesis="Framework works on triangular lattice at high h",
            expected_outcome="Needs h≥3.0+ for valid regime (many more ZZ bonds). "
            "Tests true topology-agnosticism of the GNN approach.",
            output_dir=f"{output_base}/ext_triangular",
        )
    )

    # ─── EXT-4: Ladder with MPNN architecture sweep ───────────────────────
    # Test if more GNN layers help on ladder (deeper message passing)
    variants.append(
        PipelineVariant(
            id="EXT-4-deep-gnn",
            description="Ladder with 5 GNN layers (deeper message passing for ladder)",
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
                *EXTENDED_H_VALUES,
                "--h-test",
                "2.5",
                "--hidden-dim",
                "128",
                "--n-layers",
                "5",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                "--output-dir",
                f"{output_base}/ext_deep_gnn",
                "--verbose",
            ],
            hypothesis="5 GNN layers (vs default 3) improves prediction on ladder",
            expected_outcome="Marginal improvement expected — ladder diameter is small (5 hops), "
            "so 3 layers already covers full graph. Tests over-smoothing risk.",
            output_dir=f"{output_base}/ext_deep_gnn",
        )
    )

    # ─── EXT-5: Noisy ladder with p=1 (fewer CX gates) ───────────────────
    # p=1 on ladder has fewer CX gates → ZNE might work where p=2 fails
    variants.append(
        PipelineVariant(
            id="EXT-5-noisy-p1",
            description="Noisy ladder p=1 (fewer CX → ZNE may succeed where p=2 fails)",
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
                "4.0",
                "3.5",
                "3.0",
                "--n-layouts",
                "3",
                "--shots",
                "16384",
                "--seed",
                "42",
                "--output-dir",
                f"{output_base}/ext_noisy_p1",
            ],
            hypothesis="p=1 ladder has ~50% fewer CX gates → ZNE may work at N=10",
            expected_outcome="If p=1 N=10 ladder CX count ≈ p=2 N=6 chain (≈18 CX), "
            "ZNE should work. Tests the CX-budget hypothesis for ZNE viability.",
            output_dir=f"{output_base}/ext_noisy_p1",
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
        timeout=1600,
    )
