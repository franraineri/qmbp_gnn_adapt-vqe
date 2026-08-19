#!/usr/bin/env python3
"""Verification Plan — Systematic validation of thesis findings.

This script implements a structured, modular, and repeatable verification
plan for all claims in 08_lessons_learned.md and 10_key_findings_corrected.md.

The plan is organized into 3 priority tiers:

  TIER 1 (HIGH) — Correct existing claims with wrong h_test or missing seeds
    V1: p=1 ladder N=6, seeds 42/43/44, h_test=3.0
    V2: p=1 triangular N=6, seeds 42/43/44, h_test=4.5
    V3: p=1 ladder N=10, h_test=3.0 (boundary verification)

  TIER 2 (MEDIUM) — Strengthen findings with additional evidence
    V4: p=1 chain_1d N=10, h_test=2.0 (boundary verification)
    V5: p=1 triangular N=10, h_test=3.75, seeds 43/44 (seed-specificity check)

  TIER 3 (LOW) — Nice-to-have for completeness
    V6: p=1 chain_1d N=6, seeds 42/43/44, h_test=1.75

Each variant has:
  - A hypothesis being tested
  - Expected outcome based on existing knowledge
  - Clear pass/fail criteria
  - Traceability to the claim it verifies

Estimated total time: ~15 min (all tiers)
Estimated Tier 1 only: ~8 min

Usage:
    # List all verification variants
    .venv/bin/python scripts/experiment_runners/run_verification_plan.py --list

    # Run Tier 1 only (highest priority)
    .venv/bin/python scripts/experiment_runners/run_verification_plan.py --noiseless-only

    # Run Tier 2 (medium priority)
    .venv/bin/python scripts/experiment_runners/run_verification_plan.py --noisy-only

    # Run Tier 3 (low priority)
    .venv/bin/python scripts/experiment_runners/run_verification_plan.py --extended-only

    # Run everything
    .venv/bin/python scripts/experiment_runners/run_verification_plan.py

    # Dry run (show commands without executing)
    .venv/bin/python scripts/experiment_runners/run_verification_plan.py --dry-run

    # Run a specific variant by index
    .venv/bin/python scripts/experiment_runners/run_verification_plan.py --variant 0

    # After execution, verify results:
    python analysis/scan_coverage.py --discover --p 1 --extended
"""

from __future__ import annotations

import sys

from qmbp_simulation.framework.variant_runner import PipelineVariant, run_variant_script
from qmbp_simulation.framework.preflight import P1_VALID_REGIME
from qmbp_simulation.models.constants import DEFAULT_SEEDS

# ─── Configuration ───────────────────────────────────────────────────────────

PIPELINE_SCRIPT = "scripts/experiment_runners/noiseless/run_noiseless_pipeline.py"
OUTPUT_BASE = "results/thesis/verification_r1"
SEEDS = DEFAULT_SEEDS


# ─── Tier 1: HIGH priority — Correct claims ─────────────────────────────────


def build_tier1_variants(n_qubits: int) -> list[PipelineVariant]:
    """Tier 1: Correct existing claims with wrong h_test or missing seeds.

    These runs fill critical gaps that prevent us from making definitive
    statements about p=1 performance at N=6.

    Verifies claims from 08_summary §1.1 and 10_key_findings §6:
    - "Framework is topology-agnostic" (needs p=1 N=6 data for all topos)
    - "p=1 pipeline funciona a N=10" (needs ladder boundary confirmation)
    """
    variants: list[PipelineVariant] = []
    python = sys.executable

    # ─── V1: p=1 ladder N=6, 3 seeds ─────────────────────────────────────
    # Claim: "Framework is topology-agnostic"
    # Problem: Only 2 runs without seed, one PASS (0.015) one FAIL (0.153)
    # Cannot determine if ladder N=6 p=1 is reproducible
    for seed in SEEDS:
        variants.append(
            PipelineVariant(
                id=f"V1-ladder-N6-s{seed}",
                description=f"p=1 ladder N=6 seed={seed} h_test=3.0 (reproducibility)",
                category="noiseless",
                command=[
                    python,
                    PIPELINE_SCRIPT,
                    "--n-qubits",
                    "6",
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
                    "3.0",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--seed",
                    str(seed),
                    "--output-dir",
                    f"{OUTPUT_BASE}/v1_ladder_N6_seed{seed}",
                    "--verbose",
                ],
                hypothesis=(
                    "p=1 HVA on ladder N=6 achieves ΔE/gap < 5% at h_test=3.0 "
                    "reproducibly across 3 seeds (h_test=3.0 is well inside "
                    "valid regime h≥2.0 for ladder N=6)"
                ),
                expected_outcome=(
                    "PASS — existing run at h_test=3.0 passes (0.015). "
                    "If 2/3 seeds pass → ladder N=6 p=1 is viable. "
                    "If 0-1/3 pass → ladder N=6 p=1 is seed-dependent."
                ),
                output_dir=f"{OUTPUT_BASE}/v1_ladder_N6_seed{seed}",
            )
        )

    # ─── V2: p=1 triangular N=6, 3 seeds ─────────────────────────────────
    # Claim: "Framework is topology-agnostic"
    # Problem: Only 1 run, FAIL (0.193) at h_test=4.0
    # h_test=4.0 is inside valid regime (h≥3.0) but only barely above
    # the shifted boundary. Try h_test=4.5 (deeper in valid regime).
    for seed in SEEDS:
        variants.append(
            PipelineVariant(
                id=f"V2-tri-N6-s{seed}",
                description=f"p=1 triangular N=6 seed={seed} h_test=4.5 (deeper valid regime)",
                category="noiseless",
                command=[
                    python,
                    PIPELINE_SCRIPT,
                    "--n-qubits",
                    "6",
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
                    "4.5",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--seed",
                    str(seed),
                    "--output-dir",
                    f"{OUTPUT_BASE}/v2_triangular_N6_seed{seed}",
                    "--verbose",
                ],
                hypothesis=(
                    "p=1 HVA on triangular N=6 achieves ΔE/gap < 5% at h_test=4.5 "
                    "(deeper in valid regime than the failed h_test=4.0 run). "
                    "The previous failure may have been a boundary effect."
                ),
                expected_outcome=(
                    "PASS expected — triangular N=10 passes at h_test=4.25 (3/3 seeds). "
                    "N=6 should be easier. If FAIL → triangular N=6 p=1 has a "
                    "genuine expressibility issue."
                ),
                output_dir=f"{OUTPUT_BASE}/v2_triangular_N6_seed{seed}",
            )
        )

    # ─── V3: p=1 ladder N=10, h_test=3.0 (boundary) ─────────────────────
    # Claim from 10_key_findings §6: "p=1 pipeline funciona a N=10"
    # Problem: R1 at h=2.75 → catastrophic (11.06, 8.75). R2 at h=3.25 → PASS.
    # Need to verify the exact boundary: does h=3.0 pass or fail?
    for seed in SEEDS:
        variants.append(
            PipelineVariant(
                id=f"V3-ladder-N10-s{seed}",
                description=f"p=1 ladder N=10 seed={seed} h_test=3.0 (boundary test)",
                category="noiseless",
                command=[
                    python,
                    PIPELINE_SCRIPT,
                    "--n-qubits",
                    "10",
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
                    "3.0",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--seed",
                    str(seed),
                    "--output-dir",
                    f"{OUTPUT_BASE}/v3_ladder_N10_seed{seed}",
                    "--verbose",
                ],
                hypothesis=(
                    "p=1 ladder N=10 at h_test=3.0 is at the boundary of the "
                    "valid regime. R1 failed at 2.75, R2 passed at 3.25. "
                    "h=3.0 determines if the boundary is h≥3.0 or h≥3.25."
                ),
                expected_outcome=(
                    "MARGINAL or PASS — h=3.0 is in the training set so this "
                    "tests interpolation. If FAIL → valid regime is h≥3.25. "
                    "If PASS → valid regime is h≥3.0 (update P1_VALID_REGIME)."
                ),
                output_dir=f"{OUTPUT_BASE}/v3_ladder_N10_seed{seed}",
            )
        )

    return variants


# ─── Tier 2: MEDIUM priority — Strengthen findings ──────────────────────────


def build_tier2_variants(n_qubits: int) -> list[PipelineVariant]:
    """Tier 2: Strengthen existing findings with additional evidence.

    These runs don't correct errors but add confidence to claims
    that currently rest on limited data.

    Verifies:
    - Valid regime boundary for chain_1d p=1 N=10 (claim: h≥1.9)
    - Triangular N=10 p=1 failure at h=3.75 is seed-specific (not systematic)
    """
    variants: list[PipelineVariant] = []
    python = sys.executable

    # ─── V4: p=1 chain_1d N=10, h_test=2.0 (boundary verification) ──────
    # Claim: "Valid regime for chain_1d p=1 N=10 is h≥1.9"
    # Evidence: h=1.85 passes (0.049), h=1.75 is marginal (0.071)
    # h=2.0 should definitively pass if boundary is h≥1.9
    for seed in SEEDS:
        variants.append(
            PipelineVariant(
                id=f"V4-chain-N10-s{seed}",
                description=f"p=1 chain_1d N=10 seed={seed} h_test=2.0 (boundary confirm)",
                category="noisy",  # Using "noisy" category for Tier 2 filtering
                command=[
                    python,
                    PIPELINE_SCRIPT,
                    "--n-qubits",
                    "10",
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
                    "2.0",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--seed",
                    str(seed),
                    "--output-dir",
                    f"{OUTPUT_BASE}/v4_chain_N10_boundary_seed{seed}",
                    "--verbose",
                ],
                hypothesis=(
                    "p=1 chain_1d N=10 at h_test=2.0 passes (ΔE/gap < 5%). "
                    "This confirms the valid regime boundary is h≥1.9 as claimed "
                    "in binnacle-p1-scaling.md. h=2.0 is just above the boundary."
                ),
                expected_outcome=(
                    "PASS — h=2.0 is above the claimed boundary h≥1.9. "
                    "Existing data: h=1.85 passes (0.049), h=2.25 passes (seed 42). "
                    "If FAIL → boundary is actually h≥2.25 (needs correction)."
                ),
                output_dir=f"{OUTPUT_BASE}/v4_chain_N10_boundary_seed{seed}",
            )
        )

    # ─── V5: p=1 triangular N=10, h_test=3.75, seeds 43/44 ──────────────
    # Claim: "comp5_tri_multi_htest failure at h=3.75 is catastrophic"
    # Problem: Only seed=42 tested. ΔE/gap=13.58 at h=3.75 (but 0.115 at 4.25)
    # Need to determine: is h=3.75 always catastrophic, or seed-specific?
    for seed in [43, 44]:
        variants.append(
            PipelineVariant(
                id=f"V5-tri-N10-h375-s{seed}",
                description=f"p=1 triangular N=10 seed={seed} h_test=3.75 (seed-specificity)",
                category="noisy",  # Using "noisy" category for Tier 2 filtering
                command=[
                    python,
                    PIPELINE_SCRIPT,
                    "--n-qubits",
                    "10",
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
                    "3.75",
                    "--hidden-dim",
                    "128",
                    "--n-epochs",
                    "6000",
                    "--patience",
                    "500",
                    "--seed",
                    str(seed),
                    "--output-dir",
                    f"{OUTPUT_BASE}/v5_tri_N10_h375_seed{seed}",
                    "--verbose",
                ],
                hypothesis=(
                    f"p=1 triangular N=10 at h_test=3.75 with seed={seed} "
                    "determines if the catastrophic failure (ΔE/gap=13.58) at "
                    "seed=42 is seed-specific or systematic. h=3.75 is at the "
                    "boundary of the valid regime (h≥3.5)."
                ),
                expected_outcome=(
                    "FAIL expected — h=3.75 is barely inside valid regime. "
                    "If PASS → seed=42 failure was a chain break (seed-specific). "
                    "If FAIL → h=3.75 is genuinely outside the effective valid regime "
                    "and boundary should be raised to h≥4.0."
                ),
                output_dir=f"{OUTPUT_BASE}/v5_tri_N10_h375_seed{seed}",
            )
        )

    return variants


# ─── Tier 3: LOW priority — Completeness ────────────────────────────────────


def build_tier3_variants(n_qubits: int) -> list[PipelineVariant]:
    """Tier 3: Nice-to-have for completeness.

    These runs don't affect thesis claims but fill minor gaps
    in the coverage matrix.

    Verifies:
    - p=1 chain_1d N=6 reproducibility (currently no seeds recorded)
    """
    variants: list[PipelineVariant] = []
    python = sys.executable

    # ─── V6: p=1 chain_1d N=6, 3 seeds ──────────────────────────────────
    # Claim: "chain_1d is seed-independent"
    # Problem: p=1 N=6 has 4 runs but none with recorded seeds
    # Not critical (N=6 p=1 is not hardware target) but completes the matrix
    for seed in SEEDS:
        variants.append(
            PipelineVariant(
                id=f"V6-chain-N6-s{seed}",
                description=f"p=1 chain_1d N=6 seed={seed} h_test=1.75 (completeness)",
                category="extended",
                command=[
                    python,
                    PIPELINE_SCRIPT,
                    "--n-qubits",
                    "6",
                    "--topology",
                    "chain_1d",
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
                    "--seed",
                    str(seed),
                    "--output-dir",
                    f"{OUTPUT_BASE}/v6_chain_N6_seed{seed}",
                    "--verbose",
                ],
                hypothesis=(
                    "p=1 chain_1d N=6 is seed-independent at h_test=1.75 "
                    "(existing seedless run passes with 0.029). "
                    "Confirms chain_1d reproducibility extends to p=1."
                ),
                expected_outcome=(
                    "PASS 3/3 — chain_1d is the most stable topology. "
                    "p=2 chain_1d N=6 has std=0.004 across seeds. "
                    "p=1 should be equally stable."
                ),
                output_dir=f"{OUTPUT_BASE}/v6_chain_N6_seed{seed}",
            )
        )

    return variants


# ─── Main entry point ────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the verification plan runner.

    Tier mapping to CLI flags:
      --noiseless-only  → Tier 1 (HIGH priority, 9 runs, ~8 min)
      --noisy-only      → Tier 2 (MEDIUM priority, 5 runs, ~5 min)
      --extended-only   → Tier 3 (LOW priority, 3 runs, ~2 min)
      (no flag)         → All tiers (17 runs, ~15 min)
    """
    run_variant_script(
        topology="verification",
        default_n_qubits=10,  # Not used (each variant specifies its own N)
        build_noiseless=build_tier1_variants,
        build_noisy=build_tier2_variants,
        build_extended=build_tier3_variants,
        timeout=1000,  # 10 min per variant (generous for N≤10)
    )


if __name__ == "__main__":
    main()
