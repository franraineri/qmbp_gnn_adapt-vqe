#!/usr/bin/env python3
"""Exhaustive pipeline variant runner — N=10, p=2, HEAVY-HEX topology.

Tests the framework on IBM's native hardware topology. The heavy-hex lattice
is the coupling map of IBM Eagle/Heron/Torino processors (127-133 qubits).
Using this topology means the HVA circuit maps directly to hardware without
SWAP routing overhead — critical for hardware deployment.

Key properties of heavy-hex:
  - Coordination number z=3 (max) — between chain (z=2) and ladder (z=3)
  - Backbone + bridge structure: linear chain with branches every 2 sites
  - No geometric frustration (bipartite graph) — similar to ladder
  - Directly maps to IBM Torino coupling map (no SWAP gates needed)
  - Expected valid regime: similar to ladder (h≥1.5 for p=2, h≥2.5 for p=1)

Thesis value:
  - Demonstrates framework adapts to real hardware graph topology
  - Enables zero-SWAP hardware deployment (circuit depth = HVA depth)
  - Bridges simulation results to hardware experiments

Usage:
    python scripts/experiment_runners/run_thesis_variants-heavy_hex.py --list
    python scripts/experiment_runners/run_thesis_variants-heavy_hex.py --dry-run
    python scripts/experiment_runners/run_thesis_variants-heavy_hex.py
    python scripts/experiment_runners/run_thesis_variants-heavy_hex.py --noiseless-only
    python scripts/experiment_runners/run_thesis_variants-heavy_hex.py --variant 0
    python scripts/experiment_runners/run_thesis_variants-heavy_hex.py --n-qubits 6
"""

from __future__ import annotations

import sys

from qmbp_simulation.framework.variant_runner import PipelineVariant, run_variant_script

# ═══════════════════════════════════════════════════════════════════════════
# Constants for this topology variant
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_N_QUBITS = 10
P_LAYERS = 2
TOPOLOGY = "heavy_hex"
DEFAULT_SEED = 43

# Heavy-hex has z=3 (similar to ladder). Expected valid regime similar to ladder:
# p=2: h≥1.5 (like ladder N=10), p=1: h≥2.5 (estimated, between chain and ladder)
# Training grid covers h=2.0 to 4.0 (safe paramagnetic regime)
# NOTE: h_test must NOT be in h_values (tests generalization, not memorization)
BASE_H_VALUES = ["4.0", "3.5", "3.0", "2.5", "2.0"]
EXTENDED_H_VALUES = ["4.0", "3.75", "3.5", "3.25", "3.0", "2.75", "2.5", "2.25", "2.0"]

# Unseen test points (NOT in any training grid)
H_TEST_SAFE = "2.75"  # Interpolation between 3.0 and 2.5 (in BASE, not EXTENDED)
H_TEST_P1 = "3.25"  # Interpolation between 3.5 and 3.0 (for p=1)
H_TEST_BOUNDARY = "2.125"  # Between 2.25 and 2.0 in EXTENDED (boundary probe)
H_TEST_D_SAFE = "3.125"  # Between 3.25 and 3.0 in EXTENDED (for Group D)
H_TEST_D_BOUNDARY = "2.375"  # Between 2.5 and 2.25 in EXTENDED (for Group D)

PIPELINE_SCRIPT = "scripts/experiment_runners/experiment_run_helpers/run_pipeline.py"
NOISY_SCRIPT = "scripts/experiment_runners/experiment_run_helpers/run_noisy_pipeline.py"


# ═══════════════════════════════════════════════════════════════════════════
# Variant definitions
# ═══════════════════════════════════════════════════════════════════════════


def build_noiseless_variants(n_qubits: int = DEFAULT_N_QUBITS) -> list[PipelineVariant]:
    """Build noiseless pipeline variants for heavy-hex topology."""
    variants = []
    python = sys.executable
    output_base = f"results/thesis/variants_N{n_qubits}_{TOPOLOGY}"
    seed_args = ["--seed", str(DEFAULT_SEED)]

    # ─── Group A: VQE Restart Sensitivity ──────────────────────────────────
    # Heavy-hex is bipartite (no frustration) → expect benign landscape like ladder
    for n_restarts in [1, 3, 5]:
        variants.append(
            PipelineVariant(
                id=f"NL-A{n_restarts}",
                description=f"VQE restarts={n_restarts} (heavy_hex, hidden=128)",
                category="noiseless",
                command=[
                    python,
                    PIPELINE_SCRIPT,
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
                    H_TEST_SAFE,
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
                hypothesis=(
                    f"VQE with {n_restarts} restart(s) on heavy_hex converges to ΔE/gap < 5%. "
                    "Heavy-hex is bipartite → expect benign landscape (like ladder)."
                ),
                expected_outcome=(
                    "Even 1 restart should work (no frustration). "
                    "Validates that heavy-hex landscape is as benign as ladder."
                ),
                output_dir=f"{output_base}/nl_restarts_{n_restarts}",
            )
        )

    # ─── Group B: MPNN Hidden Dimension ────────────────────────────────────
    for hidden_dim in [64, 128]:
        variants.append(
            PipelineVariant(
                id=f"NL-B{hidden_dim}",
                description=f"MPNN hidden_dim={hidden_dim} (heavy_hex, 5 restarts)",
                category="noiseless",
                command=[
                    python,
                    PIPELINE_SCRIPT,
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
                    H_TEST_SAFE,
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
                hypothesis=f"MPNN hidden_dim={hidden_dim} on heavy_hex achieves ΔE/gap < 5%",
                expected_outcome=(
                    "h=128 should be sufficient (z=3, similar to ladder). "
                    "h=64 tests if smaller model works on this simpler graph."
                ),
                output_dir=f"{output_base}/nl_hidden_{hidden_dim}",
            )
        )

    # ─── Group C: Seed Robustness ──────────────────────────────────────────
    for seed_val in DEFAULT_SEEDS:
        variants.append(
            PipelineVariant(
                id=f"NL-C-seed{seed_val}",
                description=f"Seed={seed_val} (heavy_hex, reproducibility)",
                category="noiseless",
                command=[
                    python,
                    PIPELINE_SCRIPT,
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
                    H_TEST_SAFE,
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
                hypothesis=f"Heavy-hex pipeline with seed={seed_val} is reproducible",
                expected_outcome=(
                    "Should be seed-independent (bipartite, no frustration). "
                    "Expect std < 0.01 across seeds (like chain_1d)."
                ),
                output_dir=f"{output_base}/nl_seed_{seed_val}",
            )
        )

    # ─── Group D: h-Test Points ────────────────────────────────────────────
    h_test_configs = {
        "safe": ([H_TEST_D_SAFE], "Safe point (deep in paramagnetic, unseen)"),
        "boundary": ([H_TEST_D_BOUNDARY], "Near expected valid regime boundary"),
        "multi": ([H_TEST_D_SAFE, H_TEST_D_BOUNDARY], "Multiple unseen points across regime"),
    }
    for test_name, (h_tests, desc) in h_test_configs.items():
        variants.append(
            PipelineVariant(
                id=f"NL-D-{test_name}",
                description=f"h_test: {desc} (heavy_hex)",
                category="noiseless",
                command=[
                    python,
                    PIPELINE_SCRIPT,
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
                    *seed_args,
                    "--output-dir",
                    f"{output_base}/nl_htest_{test_name}",
                    "--verbose",
                ],
                hypothesis=f"MPNN generalizes on heavy_hex to h_test={h_tests}",
                expected_outcome=(
                    "Safe should pass. Boundary tests where the valid regime "
                    "starts for heavy-hex (expected h≥1.5 for p=2)."
                ),
                output_dir=f"{output_base}/nl_htest_{test_name}",
            )
        )

    # ─── Group E: p=1 on Heavy-Hex ────────────────────────────────────────
    # p=1 on heavy-hex: the hardware deployment candidate
    for seed_val in DEFAULT_SEEDS:
        variants.append(
            PipelineVariant(
                id=f"NL-E-p1-s{seed_val}",
                description=f"p=1 heavy_hex seed={seed_val} (hardware deployment candidate)",
                category="noiseless",
                command=[
                    python,
                    PIPELINE_SCRIPT,
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
                    H_TEST_P1,
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--seed",
                    str(seed_val),
                    "--output-dir",
                    f"{output_base}/nl_p1_seed{seed_val}",
                    "--verbose",
                ],
                hypothesis=(
                    f"p=1 HVA on heavy_hex N={n_qubits} seed={seed_val} achieves ΔE/gap < 5%. "
                    "This is the target config for IBM Torino deployment (zero SWAP overhead)."
                ),
                expected_outcome=(
                    "PASS expected — heavy-hex is bipartite (no frustration), "
                    "p=1 should work at h≥2.5 (similar to ladder p=1 boundary)."
                ),
                output_dir=f"{output_base}/nl_p1_seed{seed_val}",
            )
        )

    return variants


def build_noisy_variants(n_qubits: int = DEFAULT_N_QUBITS) -> list[PipelineVariant]:
    """Build noisy pipeline variants for heavy-hex topology."""
    variants = []
    python = sys.executable
    output_base = f"results/thesis/variants_N{n_qubits}_{TOPOLOGY}"

    # ─── p=1 ZNE on Heavy-Hex (the hardware deployment test) ──────────────
    # This is THE key experiment: p=1 on heavy-hex with ZNE.
    # If this works, we can deploy directly on IBM Torino without SWAP routing.
    for seed_val in DEFAULT_SEEDS:
        variants.append(
            PipelineVariant(
                id=f"NY-A-p1-zne-s{seed_val}",
                description=f"p=1 heavy_hex ZNE seed={seed_val} (hardware deployment validation)",
                category="noisy",
                command=[
                    python,
                    NOISY_SCRIPT,
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
                    str(seed_val),
                    "--output-dir",
                    f"{output_base}/ny_p1_zne_seed{seed_val}",
                    "--verbose",
                    "--n-restarts",
                    "5",
                ],
                hypothesis=(
                    f"p=1 heavy_hex ZNE with seed={seed_val} achieves positive gain. "
                    "CX count for p=1 heavy_hex N=10 = 18 (at threshold). "
                    "Key advantage: zero SWAP overhead on IBM Torino hardware."
                ),
                expected_outcome=(
                    "PASS expected — p=1 heavy_hex has same CX count as chain_1d p=1 "
                    "(both 18 CX), and chain_1d ZNE works (+46% gain). "
                    "The real advantage is zero SWAP routing on hardware."
                ),
                output_dir=f"{output_base}/ny_p1_zne_seed{seed_val}",
            )
        )

    # ─── p=2 ZNE on Heavy-Hex (expected to fail, documents boundary) ─────
    variants.append(
        PipelineVariant(
            id="NY-B-p2-zne",
            description="p=2 heavy_hex ZNE (expected failure — documents CX boundary)",
            category="noisy",
            command=[
                python,
                NOISY_SCRIPT,
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
                "42",
                "--output-dir",
                f"{output_base}/ny_p2_zne",
                "--verbose",
                "--n-restarts",
                "5",
            ],
            hypothesis="p=2 heavy_hex ZNE fails (CX count > 18 threshold)",
            expected_outcome=(
                "FAIL expected — p=2 N=10 on any topology exceeds CX threshold. "
                "Documents that heavy-hex doesn't magically fix the ZNE boundary."
            ),
            output_dir=f"{output_base}/ny_p2_zne",
        )
    )

    return variants


def build_extended_variants(n_qubits: int = DEFAULT_N_QUBITS) -> list[PipelineVariant]:
    """Extended variants for heavy-hex topology."""
    variants = []
    python = sys.executable
    output_base = f"results/thesis/variants_N{n_qubits}_{TOPOLOGY}"
    seed_args = ["--seed", str(DEFAULT_SEED)]

    # ─── EXT-1: Cross-topology comparison at same h-range ─────────────────
    # Run chain_1d and ladder with same config for direct comparison
    for ref_topo in ["chain_1d", "ladder"]:
        variants.append(
            PipelineVariant(
                id=f"EXT-1-{ref_topo.replace('_', '')}",
                description=f"{ref_topo} baseline (same h-range for heavy_hex comparison)",
                category="extended",
                command=[
                    python,
                    PIPELINE_SCRIPT,
                    "--n-qubits",
                    str(n_qubits),
                    "--p",
                    str(P_LAYERS),
                    "--topology",
                    ref_topo,
                    "--n-restarts",
                    "5",
                    "--maxiter",
                    "1000",
                    "--h-values",
                    *BASE_H_VALUES,
                    "--h-test",
                    H_TEST_SAFE,
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    *seed_args,
                    "--output-dir",
                    f"{output_base}/ext_{ref_topo}_baseline",
                    "--verbose",
                ],
                hypothesis=f"{ref_topo} at same h-range as heavy_hex for comparison",
                expected_outcome="PASS — provides reference for heavy_hex performance delta.",
                output_dir=f"{output_base}/ext_{ref_topo}_baseline",
            )
        )

    # ─── EXT-2: N=6 heavy-hex baseline ───────────────────────────────────
    variants.append(
        PipelineVariant(
            id="EXT-2-N6",
            description="Heavy-hex N=6 baseline (smaller system validation)",
            category="extended",
            command=[
                python,
                PIPELINE_SCRIPT,
                "--n-qubits",
                "6",
                "--p",
                str(P_LAYERS),
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
                "2.0",
                "1.5",
                "--h-test",
                "2.25",
                "--hidden-dim",
                "128",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                *seed_args,
                "--output-dir",
                f"{output_base}/ext_N6_baseline",
                "--verbose",
            ],
            hypothesis="Heavy-hex N=6 achieves ΔE/gap < 5% at h_test=2.25 (unseen)",
            expected_outcome="PASS — N=6 is easy for all topologies. Establishes baseline.",
            output_dir=f"{output_base}/ext_N6_baseline",
        )
    )

    return variants


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Entry point for heavy-hex variant runner."""
    run_variant_script(
        topology=TOPOLOGY,
        default_n_qubits=DEFAULT_N_QUBITS,
        build_noiseless=build_noiseless_variants,
        build_noisy=build_noisy_variants,
        build_extended=build_extended_variants,
        timeout=600,
    )


if __name__ == "__main__":
    main()
