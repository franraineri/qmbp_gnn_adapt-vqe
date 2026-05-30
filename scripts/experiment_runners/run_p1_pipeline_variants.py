#!/usr/bin/env python3
"""p=1 pipeline variant runner for thesis validation.

Executes the p=1 noiseless pipeline across 3 topologies × 3 seeds to produce
a "p=1 Pipeline Performance" table comparable to the p=2 Table 5.4.

Addresses verified gaps from analysis/scan_coverage.py:
  - chain_1d N=10: h_test was outside valid regime (need h≥1.9)
  - ladder N=10: only 1 run without seed, needs 3 seeds
  - triangular N=10: failed with 7 restarts (restart paradox), use 5

Valid regime boundaries (p=1):
  - chain_1d N=10: h ≥ 1.9
  - ladder N=10: h ≥ 2.0
  - triangular N=10: h ≥ 3.5

Usage:
    # Run all p=1 variants (3 topologies × 3 seeds = 9 runs, ~14 min)
    python scripts/experiment_runners/run_p1_pipeline_variants.py

    # Dry run (show commands without executing)
    python scripts/experiment_runners/run_p1_pipeline_variants.py --dry-run

    # Run only a specific variant
    python scripts/experiment_runners/run_p1_pipeline_variants.py --variant 0

    # List all variants
    python scripts/experiment_runners/run_p1_pipeline_variants.py --list
"""

from __future__ import annotations

import sys

from qmbp_simulation.framework.variant_runner import PipelineVariant, run_variant_script

DEFAULT_N_QUBITS = 10
SEEDS = [42, 43, 44]

# Pipeline script path (relative to project root)
PIPELINE_SCRIPT = "scripts/experiment_runners/experiment_run_helpers_CHECK/run_pipeline.py"


def build_noiseless_variants(n_qubits: int = DEFAULT_N_QUBITS) -> list[PipelineVariant]:
    """Build p=1 noiseless pipeline variants for all topologies × seeds."""
    variants: list[PipelineVariant] = []
    python = sys.executable
    n = str(n_qubits)
    out_base = f"results/thesis/p1_variants_N{n_qubits}"

    # ─── chain_1d N=10 p=1 ─────────────────────────────────────────────────
    # Valid regime: h ≥ 1.9
    # Training: h=[4.0, 3.5, 3.0, 2.5, 2.0] (all well within valid regime)
    # Test: h_test=2.25 (unseen, safely inside valid regime h≥1.9)
    for seed in SEEDS:
        variants.append(
            PipelineVariant(
                id=f"P1-chain-s{seed}",
                description=f"p=1 chain_1d N={n_qubits} seed={seed} (h_test=2.25)",
                category="noiseless",
                command=[
                    python,
                    PIPELINE_SCRIPT,
                    "--n-qubits",
                    n,
                    "--topology",
                    "chain_1d",
                    "--p",
                    "1",
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
                    "--h-test",
                    "2.25",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--seed",
                    str(seed),
                    "--output-dir",
                    f"{out_base}/chain_1d_seed{seed}",
                    "--verbose",
                ],
                hypothesis=(
                    f"p=1 HVA on chain_1d N={n_qubits} achieves ΔE/gap < 5% "
                    f"at h_test=2.25 (within valid regime h≥1.9)"
                ),
                expected_outcome="PASS — landscape is trivial (1 effective param)",
                output_dir=f"{out_base}/chain_1d_seed{seed}",
            )
        )

    # ─── ladder N=10 p=1 ───────────────────────────────────────────────────
    # Valid regime: h ≥ 2.0
    # Training: h=[4.0, 3.5, 3.0, 2.5] (all within valid regime)
    # Test: h_test=2.75 (unseen, safely inside valid regime h≥2.0)
    for seed in SEEDS:
        variants.append(
            PipelineVariant(
                id=f"P1-ladder-s{seed}",
                description=f"p=1 ladder N={n_qubits} seed={seed} (h_test=2.75)",
                category="noiseless",
                command=[
                    python,
                    PIPELINE_SCRIPT,
                    "--n-qubits",
                    n,
                    "--topology",
                    "ladder",
                    "--p",
                    "1",
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
                    "2.75",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--seed",
                    str(seed),
                    "--output-dir",
                    f"{out_base}/ladder_seed{seed}",
                    "--verbose",
                ],
                hypothesis=(
                    f"p=1 HVA on ladder N={n_qubits} achieves ΔE/gap < 5% "
                    f"at h_test=2.75 (within valid regime h≥2.0)"
                ),
                expected_outcome="PASS — existing run passes at 0.036",
                output_dir=f"{out_base}/ladder_seed{seed}",
            )
        )

    # ─── triangular N=10 p=1 ──────────────────────────────────────────────
    # Valid regime: h ≥ 3.5
    # Training: h=[5.0, 4.5, 4.0, 3.5] (all within valid regime)
    # Test: h_test=4.25 (unseen, safely inside valid regime h≥3.5)
    # NOTE: Using 5 restarts (not 7!) to avoid restart paradox
    # Previous run with 7 restarts had θ_smooth=1.57, gen_gap=0.012 → FAIL
    for seed in SEEDS:
        # seed=43 already exists but failed with 7 restarts — re-run with 5
        variants.append(
            PipelineVariant(
                id=f"P1-tri-s{seed}",
                description=f"p=1 triangular N={n_qubits} seed={seed} (h_test=4.25, 5rst)",
                category="noiseless",
                command=[
                    python,
                    PIPELINE_SCRIPT,
                    "--n-qubits",
                    n,
                    "--topology",
                    "triangular",
                    "--p",
                    "1",
                    "--n-restarts",
                    "5",
                    "--maxiter",
                    "1000",
                    "--h-values",
                    "5.0",
                    "4.5",
                    "4.0",
                    "3.5",
                    "--h-test",
                    "4.25",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--seed",
                    str(seed),
                    "--output-dir",
                    f"{out_base}/triangular_seed{seed}",
                    "--verbose",
                ],
                hypothesis=(
                    f"p=1 HVA on triangular N={n_qubits} achieves ΔE/gap < 5% "
                    f"at h_test=4.25 with 5 restarts (avoids restart paradox)"
                ),
                expected_outcome=(
                    "PASS expected — previous failure was restart paradox (7rst), not physics limit"
                ),
                output_dir=f"{out_base}/triangular_seed{seed}",
            )
        )

    return variants


def build_noisy_variants(n_qubits: int = DEFAULT_N_QUBITS) -> list[PipelineVariant]:
    """No noisy variants needed — p=1 ZNE already confirmed (9 runs, +49% gain)."""
    return []


def build_extended_variants(n_qubits: int = DEFAULT_N_QUBITS) -> list[PipelineVariant]:
    """No extended variants needed for this focused run."""
    return []


def main() -> None:
    """Entry point."""
    run_variant_script(
        topology="multi",  # Multi-topology run
        default_n_qubits=DEFAULT_N_QUBITS,
        build_noiseless=build_noiseless_variants,
        build_noisy=build_noisy_variants,
        build_extended=build_extended_variants,
        timeout=1300,  # 10 min per variant (generous)
    )


if __name__ == "__main__":
    main()
