#!/usr/bin/env python3
"""Exhaustive pipeline variant runner — Heisenberg XXZ model experiments.

Tests the GNN-HVA framework on the Heisenberg XXZ model across multiple
anisotropy values (Δ), topologies, and system sizes. This is a systematic
exploration of model-agnostic pipeline capabilities.

Key physics of Heisenberg XXZ:
  - H = J(XX + YY + Δ·ZZ) - h·Z on each bond
  - Δ=0: XY model (easy-plane), Δ=1: isotropic Heisenberg, Δ>1: Ising-like
  - 4 variational parameters per HVA layer (θ_xx, θ_yy, θ_zz, θ_z)
  - Néel initial state (|010101...⟩) — breaks SU(2) symmetry
  - Higher entanglement than TFIM → HVA p=2 has limited expressibility

Expected results:
  - Max fidelity ~22% at N=6 p=2 for isotropic Heisenberg (Δ=1.0)
  - XY model (Δ=0) may have slightly better fidelity (lower entanglement)
  - Valid regime (if any) only at very high h (paramagnetic limit)
  - Negative results are scientifically valuable — document HVA limitations

Thesis value:
  - Demonstrates framework is model-agnostic (same pipeline, different physics)
  - Quantifies HVA expressibility limits via entanglement analysis
  - Provides negative result catalog for future ansatz design
  - Compares TFIM vs Heisenberg to isolate model-specific effects

Usage:
    python scripts/experiment_runners/run_thesis_variants-heisenberg.py --list
    python scripts/experiment_runners/run_thesis_variants-heisenberg.py --dry-run
    python scripts/experiment_runners/run_thesis_variants-heisenberg.py
    python scripts/experiment_runners/run_thesis_variants-heisenberg.py --noiseless-only
    python scripts/experiment_runners/run_thesis_variants-heisenberg.py --variant 0
    python scripts/experiment_runners/run_thesis_variants-heisenberg.py --n-qubits 6
    python scripts/experiment_runners/run_thesis_variants-heisenberg.py --extended-only
"""

from __future__ import annotations

import sys

from qmbp_simulation.framework.variant_runner import PipelineVariant, run_variant_script

# ═══════════════════════════════════════════════════════════════════════════
# Constants for Heisenberg model experiments
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_N_QUBITS = 6
P_LAYERS = 2
TOPOLOGY = "chain_1d"
DEFAULT_SEED = 42

# Heisenberg requires more restarts (4D landscape, 8 params at p=2)
# VQE defaults from ModelSpec: n_restarts=10, restart_sigma=0.5, maxiter=1500
DEFAULT_N_RESTARTS = 10
DEFAULT_MAXITER = 1500

# Delta values to explore (anisotropy parameter)
DELTA_ISOTROPIC = 1.0  # Standard Heisenberg
DELTA_XY = 0.0  # XY model (no ZZ term)
DELTA_INTERMEDIATE = 0.5  # Between XY and isotropic
DELTA_ISING_LIKE = 1.5  # Enhanced ZZ coupling

# h-values for Heisenberg: need higher h to reach paramagnetic regime
# (Heisenberg has stronger correlations than TFIM at same h)
BASE_H_VALUES = ["4.0", "3.5", "3.0", "2.5", "2.0"]
EXTENDED_H_VALUES = [
    "4.0",
    "3.75",
    "3.5",
    "3.25",
    "3.0",
    "2.75",
    "2.5",
    "2.25",
    "2.0",
    "1.75",
    "1.5",
]
DEEP_H_VALUES = ["4.0", "3.5", "3.0", "2.5", "2.0", "1.5", "1.0", "0.5"]

# Unseen test points (NOT in any training grid)
H_TEST_SAFE = "3.25"  # Deep paramagnetic (interpolation in BASE)
H_TEST_BOUNDARY = "2.25"  # Near expected valid regime boundary
H_TEST_DEEP = "1.75"  # Deep in correlated regime (likely fails)

PIPELINE_SCRIPT = "scripts/experiment_runners/_deprecated/experiment_run_helpers/run_heisenberg_pipeline.py"  # TODO: migrate to noiseless runner --model heisenberg


# ═══════════════════════════════════════════════════════════════════════════
# Variant definitions
# ═══════════════════════════════════════════════════════════════════════════


def build_noiseless_variants(n_qubits: int = DEFAULT_N_QUBITS) -> list[PipelineVariant]:
    """Build noiseless pipeline variants for Heisenberg XXZ model."""
    variants = []
    python = sys.executable
    output_base = f"results/thesis/variants_N{n_qubits}_heisenberg"
    seed_args = ["--seed", str(DEFAULT_SEED)]

    # ─── Group A: Anisotropy Sweep (Δ variation) ───────────────────────────
    # Core experiment: how does Δ affect HVA expressibility?
    delta_configs = {
        "xy": (DELTA_XY, "XY model (Δ=0, no ZZ term)"),
        "intermediate": (DELTA_INTERMEDIATE, "Intermediate anisotropy (Δ=0.5)"),
        "isotropic": (DELTA_ISOTROPIC, "Isotropic Heisenberg (Δ=1.0)"),
        "ising_like": (DELTA_ISING_LIKE, "Ising-like anisotropy (Δ=1.5)"),
    }
    for name, (delta, desc) in delta_configs.items():
        variants.append(
            PipelineVariant(
                id=f"NL-A-{name}",
                description=f"Δ={delta}: {desc}",
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
                    "--model",
                    "heisenberg",
                    "--delta",
                    str(delta),
                    "--n-restarts",
                    str(DEFAULT_N_RESTARTS),
                    "--maxiter",
                    str(DEFAULT_MAXITER),
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
                    f"{output_base}/nl_delta_{name}",
                    "--verbose",
                ],
                hypothesis=(
                    f"Heisenberg XXZ with Δ={delta} at N={n_qubits} p=2: "
                    f"explore HVA expressibility. XY (Δ=0) expected to have "
                    f"highest fidelity; isotropic (Δ=1) lowest (~22%)."
                ),
                expected_outcome=(
                    "Likely negative result for Δ≥0.5 (max fidelity < 60%). "
                    "XY model may reach ~40-50% at high h. Documents "
                    "expressibility as function of anisotropy."
                ),
                output_dir=f"{output_base}/nl_delta_{name}",
            )
        )

    # ─── Group B: Seed Robustness (isotropic Heisenberg) ─────────────────
    # Verify results are seed-independent for the base case
    for seed_val in DEFAULT_SEEDS:
        variants.append(
            PipelineVariant(
                id=f"NL-B-seed{seed_val}",
                description=f"Heisenberg Δ=1.0 seed={seed_val} (reproducibility)",
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
                    "--model",
                    "heisenberg",
                    "--delta",
                    "1.0",
                    "--n-restarts",
                    str(DEFAULT_N_RESTARTS),
                    "--maxiter",
                    str(DEFAULT_MAXITER),
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
                hypothesis=(
                    f"Heisenberg pipeline with seed={seed_val} produces consistent results. "
                    "4D landscape may have multiple local minima → higher seed variance than TFIM."
                ),
                expected_outcome=(
                    "Expect higher variance than TFIM (4D landscape vs 2D). "
                    "If max fidelity varies >10% across seeds, landscape is rugged."
                ),
                output_dir=f"{output_base}/nl_seed_{seed_val}",
            )
        )

    # ─── Group C: VQE Restart Sensitivity ──────────────────────────────────
    # Heisenberg has 4D landscape → may need more restarts than TFIM
    for n_restarts in [5, 10, 15, 20]:
        variants.append(
            PipelineVariant(
                id=f"NL-C-r{n_restarts}",
                description=f"VQE restarts={n_restarts} (Heisenberg Δ=1.0)",
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
                    "--model",
                    "heisenberg",
                    "--delta",
                    "1.0",
                    "--n-restarts",
                    str(n_restarts),
                    "--maxiter",
                    str(DEFAULT_MAXITER),
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
                    f"VQE with {n_restarts} restarts on Heisenberg 4D landscape. "
                    "More restarts needed than TFIM (4 params vs 2 params per layer)."
                ),
                expected_outcome=(
                    "Fidelity should plateau after ~10 restarts. "
                    "If 20 restarts ≈ 10 restarts, landscape has few local minima."
                ),
                output_dir=f"{output_base}/nl_restarts_{n_restarts}",
            )
        )

    # ─── Group D: Deep h-Sweep (regime discovery) ───────────────────────
    # Sweep from h=4.0 down to h=0.5 to find where fidelity drops
    for delta, label in [(0.0, "xy"), (1.0, "isotropic")]:
        variants.append(
            PipelineVariant(
                id=f"NL-D-deep-{label}",
                description=f"Deep h-sweep Δ={delta} (regime discovery, h=4→0.5)",
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
                    "--model",
                    "heisenberg",
                    "--delta",
                    str(delta),
                    "--n-restarts",
                    str(DEFAULT_N_RESTARTS),
                    "--maxiter",
                    str(DEFAULT_MAXITER),
                    "--h-values",
                    *DEEP_H_VALUES,
                    "--h-test",
                    H_TEST_DEEP,
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    *seed_args,
                    "--output-dir",
                    f"{output_base}/nl_deep_{label}",
                    "--verbose",
                ],
                hypothesis=(
                    f"Deep sweep for Δ={delta}: identify h_min where fidelity ≥ 0.60. "
                    "Expect fidelity to drop sharply below h≈2.0 (entanglement transition)."
                ),
                expected_outcome=(
                    "Maps the full fidelity landscape. Likely no valid regime at standard "
                    "threshold (0.93). May find partial regime at relaxed threshold (0.60)."
                ),
                output_dir=f"{output_base}/nl_deep_{label}",
            )
        )

    # ─── Group E: Topology Comparison ──────────────────────────────────────
    # Same Heisenberg model on different topologies
    for topo in ["chain_1d", "ladder", "triangular"]:
        variants.append(
            PipelineVariant(
                id=f"NL-E-{topo.replace('_', '')}",
                description=f"Heisenberg Δ=1.0 on {topo} (topology comparison)",
                category="noiseless",
                command=[
                    python,
                    PIPELINE_SCRIPT,
                    "--n-qubits",
                    str(n_qubits),
                    "--p",
                    str(P_LAYERS),
                    "--topology",
                    topo,
                    "--model",
                    "heisenberg",
                    "--delta",
                    "1.0",
                    "--n-restarts",
                    str(DEFAULT_N_RESTARTS),
                    "--maxiter",
                    str(DEFAULT_MAXITER),
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
                    f"{output_base}/nl_topo_{topo}",
                    "--verbose",
                ],
                hypothesis=(
                    f"Heisenberg on {topo}: more edges → more entanglement → lower fidelity. "
                    "Triangular (frustrated) expected worst; chain_1d best."
                ),
                expected_outcome=(
                    "chain_1d > ladder > triangular in max fidelity. "
                    "Frustration (triangular) compounds the expressibility problem."
                ),
                output_dir=f"{output_base}/nl_topo_{topo}",
            )
        )

    return variants


def build_noisy_variants(n_qubits: int = DEFAULT_N_QUBITS) -> list[PipelineVariant]:
    """Build noisy pipeline variants for Heisenberg model.

    Note: Heisenberg at p=2 has 3×|E| CX gates per layer (vs |E| for TFIM).
    For N=6 chain_1d: 3×5 = 15 CX per layer, 30 total at p=2.
    This exceeds the ZNE threshold (~18 CX) → ZNE expected to fail.
    Noisy variants are included to document this limitation.
    """
    # Heisenberg p=2 at N=6 already has 30 CX gates (chain_1d, 5 edges × 3 × 2 layers)
    # This is well above the ZNE threshold of ~18 CX.
    # No noisy variants are scientifically justified — ZNE will fail.
    # Document this as a known limitation rather than wasting compute.
    return []


def build_extended_variants(n_qubits: int = DEFAULT_N_QUBITS) -> list[PipelineVariant]:
    """Extended variants for deeper Heisenberg analysis."""
    variants = []
    python = sys.executable
    output_base = f"results/thesis/variants_N{n_qubits}_heisenberg"
    seed_args = ["--seed", str(DEFAULT_SEED)]

    # ─── EXT-1: TFIM Baseline (same h-range for direct comparison) ────────
    variants.append(
        PipelineVariant(
            id="EXT-1-tfim-baseline",
            description="TFIM baseline (same h-range for Heisenberg comparison)",
            category="extended",
            command=[
                python,
                PIPELINE_SCRIPT,
                "--n-qubits",
                str(n_qubits),
                "--p",
                str(P_LAYERS),
                "--topology",
                TOPOLOGY,
                "--model",
                "tfim",
                "--n-restarts",
                "5",
                "--maxiter",
                "1000",
                "--h-values",
                *BASE_H_VALUES,
                "--h-test",
                H_TEST_SAFE,
                "--hidden-dim",
                "64",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                *seed_args,
                "--output-dir",
                f"{output_base}/ext_tfim_baseline",
                "--verbose",
            ],
            hypothesis="TFIM at same h-range as Heisenberg for direct comparison",
            expected_outcome=(
                "PASS (ΔE/gap < 5%) — TFIM is easy at these h-values. "
                "Provides reference for quantifying Heisenberg difficulty."
            ),
            output_dir=f"{output_base}/ext_tfim_baseline",
        )
    )

    # ─── EXT-2: XY Model with Extended h-Range ───────────────────────────
    # XY model (Δ=0) has lower entanglement → may have a valid regime
    variants.append(
        PipelineVariant(
            id="EXT-2-xy-extended",
            description="XY model (Δ=0) extended h-range (searching for valid regime)",
            category="extended",
            command=[
                python,
                PIPELINE_SCRIPT,
                "--n-qubits",
                str(n_qubits),
                "--p",
                str(P_LAYERS),
                "--topology",
                TOPOLOGY,
                "--model",
                "xy",
                "--n-restarts",
                str(DEFAULT_N_RESTARTS),
                "--maxiter",
                str(DEFAULT_MAXITER),
                "--h-values",
                *EXTENDED_H_VALUES,
                "--h-test",
                H_TEST_BOUNDARY,
                "--hidden-dim",
                "128",
                "--n-epochs",
                "6000",
                "--patience",
                "500",
                *seed_args,
                "--output-dir",
                f"{output_base}/ext_xy_extended",
                "--verbose",
            ],
            hypothesis=(
                "XY model with extended h-range may find a valid regime at high h. "
                "Δ=0 removes ZZ correlations → lower entanglement → better HVA fit."
            ),
            expected_outcome=(
                "Best chance of finding a valid regime among non-TFIM models. "
                "If XY fails, all XXZ models fail at p=2."
            ),
            output_dir=f"{output_base}/ext_xy_extended",
        )
    )

    # ─── EXT-3: Continuous Δ Sweep (fine-grained anisotropy) ────────────
    # Map fidelity as a function of Δ at fixed h-range (deep paramagnetic)
    # Use 5 h-values minimum for meaningful VQE statistics
    for delta in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        variants.append(
            PipelineVariant(
                id=f"EXT-3-delta{delta:.2f}",
                description=f"Δ={delta:.2f} at h=4→2 (anisotropy-fidelity mapping)",
                category="extended",
                command=[
                    python,
                    PIPELINE_SCRIPT,
                    "--n-qubits",
                    str(n_qubits),
                    "--p",
                    str(P_LAYERS),
                    "--topology",
                    TOPOLOGY,
                    "--model",
                    "heisenberg",
                    "--delta",
                    str(delta),
                    "--n-restarts",
                    str(DEFAULT_N_RESTARTS),
                    "--maxiter",
                    str(DEFAULT_MAXITER),
                    "--h-values",
                    *BASE_H_VALUES,
                    "--h-test",
                    H_TEST_SAFE,
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "300",
                    *seed_args,
                    "--output-dir",
                    f"{output_base}/ext_delta_sweep_{delta:.2f}",
                    "--verbose",
                ],
                hypothesis=(
                    f"Δ={delta:.2f} at high h: map fidelity vs anisotropy. "
                    "Expect monotonic decrease in fidelity as Δ increases."
                ),
                expected_outcome=(
                    "Produces fidelity(Δ) curve. Quantifies how anisotropy "
                    "degrades HVA expressibility at fixed field strength."
                ),
                output_dir=f"{output_base}/ext_delta_sweep_{delta:.2f}",
            )
        )

    # ─── EXT-4: Multi-seed XY on Ladder (best-case scenario) ─────────────
    # XY + ladder: if anything works for non-TFIM, this is it
    for seed_val in DEFAULT_SEEDS:
        variants.append(
            PipelineVariant(
                id=f"EXT-4-xy-ladder-s{seed_val}",
                description=f"XY on ladder seed={seed_val} (best-case non-TFIM)",
                category="extended",
                command=[
                    python,
                    PIPELINE_SCRIPT,
                    "--n-qubits",
                    str(n_qubits),
                    "--p",
                    str(P_LAYERS),
                    "--topology",
                    "ladder",
                    "--model",
                    "xy",
                    "--n-restarts",
                    str(DEFAULT_N_RESTARTS),
                    "--maxiter",
                    str(DEFAULT_MAXITER),
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
                    f"{output_base}/ext_xy_ladder_seed{seed_val}",
                    "--verbose",
                ],
                hypothesis=(
                    f"XY model on ladder with seed={seed_val}: "
                    "ladder has more edges but XY has lower entanglement. "
                    "Tests if reduced anisotropy compensates for topology complexity."
                ),
                expected_outcome=(
                    "Likely negative result but documents the interaction "
                    "between topology complexity and model anisotropy."
                ),
                output_dir=f"{output_base}/ext_xy_ladder_seed{seed_val}",
            )
        )

    # ─── EXT-5: p=1 Heisenberg (reduced depth) ───────────────────────────
    # p=1 has only 4 params total — even more limited but faster
    variants.append(
        PipelineVariant(
            id="EXT-5-p1-heisenberg",
            description="Heisenberg p=1 (4 params, reduced expressibility)",
            category="extended",
            command=[
                python,
                PIPELINE_SCRIPT,
                "--n-qubits",
                str(n_qubits),
                "--p",
                "1",
                "--topology",
                TOPOLOGY,
                "--model",
                "heisenberg",
                "--delta",
                "1.0",
                "--n-restarts",
                str(DEFAULT_N_RESTARTS),
                "--maxiter",
                str(DEFAULT_MAXITER),
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
                f"{output_base}/ext_p1_heisenberg",
                "--verbose",
            ],
            hypothesis=(
                "Heisenberg p=1 (4 params): even less expressible than p=2. "
                "Documents the floor of HVA capability for this model."
            ),
            expected_outcome=(
                "Worse than p=2 (which already fails). Establishes that "
                "the problem is fundamental, not just insufficient depth."
            ),
            output_dir=f"{output_base}/ext_p1_heisenberg",
        )
    )

    return variants


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Entry point for Heisenberg variant runner."""
    run_variant_script(
        topology="heisenberg",  # Used as label in logs/output paths
        default_n_qubits=DEFAULT_N_QUBITS,
        build_noiseless=build_noiseless_variants,
        build_noisy=build_noisy_variants,
        build_extended=build_extended_variants,
        timeout=900,  # 15 min per variant (Heisenberg VQE is slower: 10 restarts × 1500 iter)
    )


if __name__ == "__main__":
    main()
