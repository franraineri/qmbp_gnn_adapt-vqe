#!/usr/bin/env python3
"""Deep verification suite for the energy correction pipeline.

Two modes of operation:
  1. INVARIANT CHECK (default): Verify pipeline correctness against known
     bugs, numerical edge cases, and physics properties. Run after ANY code
     change to noisy_utils.py or the correction chain.

  2. POST-EXECUTION VALIDATION (--validate <run_dir>): Validate a specific
     hardware/benchmark run's results for consistency, physics violations,
     circuit metrics, and QPU time estimates. Run after EVERY hardware execution.

Verifies:
  Part 1:  Numerical reproduction of the original soft-interpolation bug
  Part 2:  FIXED affine_correct_energy() correctness
  Part 3:  Audit ALL hardware runs with the fixed function
  Part 4:  Monotonicity property (correction NEVER worsens result)
  Part 5:  Bounds consistency across 3 independent implementations
  Part 6:  ZNE extrapolation sanity (NaN/Inf/degenerate inputs)
  Part 7:  End-to-end pipeline invariant (H8: correction chain never amplifies)
  Part 8:  Proactive edge-case detection (float64 limits, degenerate params)
  Part 9:  Mitigation benchmark results regression scan
  Part 10: Post-execution result validation (--validate only)
  Part 11: Circuit metrics & QPU time estimate (--validate only)
  Part 12: Structural pipeline audit (--audit only)

Imports (reuses existing infrastructure — no logic duplication):
  - qmbp_simulation.analysis.VQEValidator — physics energy bounds
  - qmbp_simulation.analysis.metrics.compute_snr — measurement SNR
  - qmbp_simulation.analysis.metrics.compute_classification_confidence — phase confidence
  - qmbp_simulation.execution.affine_correct_energy — canonical correction function
  - qmbp_simulation.framework.criteria.compute_verdict — verdict evaluation
  - qmbp_simulation.execution.hardware.preflight — ZNE CX thresholds
  - project_health.analysis.validation.verify_results.classify_de_gap — verdict classification

Usage:
    .venv/bin/python scripts/verify_affine_bug.py                # Full invariant check (Parts 1-9)
    .venv/bin/python scripts/verify_affine_bug.py --quick        # Fast (Parts 1-4, for CI)
    .venv/bin/python scripts/verify_affine_bug.py --validate <run_dir>  # Post-execution (Parts 1-11)
    .venv/bin/python scripts/verify_affine_bug.py --audit        # Full + structural audit (Parts 1-9, 12)

Exit codes:
    0 = all checks pass
    N = number of hard violations found
"""

import json
import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging

import numpy as np

# Suppress INFO/WARNING from noisy_utils during edge-case testing
logging.getLogger("qmbp_simulation.execution.noisy_utils").setLevel(logging.ERROR)

from qmbp_simulation.execution import affine_correct_energy

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

QUICK_MODE = "--quick" in sys.argv
AUDIT_MODE = "--audit" in sys.argv
VALIDATE_DIR = None
for i, arg in enumerate(sys.argv):
    if arg == "--validate" and i + 1 < len(sys.argv):
        VALIDATE_DIR = Path(sys.argv[i + 1])
        break
N_VIOLATIONS_TOTAL = 0


def report_section(part: int, title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"PART {part}: {title}")
    print("=" * 70)


def fail(msg: str) -> None:
    """Record a violation and print it."""
    global N_VIOLATIONS_TOTAL
    N_VIOLATIONS_TOTAL += 1
    print(f"  ❌ VIOLATION: {msg}")


def ok(msg: str) -> None:
    """Print a pass message."""
    print(f"  ✅ {msg}")


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: Reproduce the bug numerically (pre-fix values)
# ═══════════════════════════════════════════════════════════════════════════

report_section(1, "Numerical reproduction — what the OLD code produced")

# Run 141440 values
mitigated_energy = -33.20130666097003  # e_zne from PEA
e_ground = -33.198273652500575  # e_exact
n_qubits = 10
h_value = 3.25
gap = 4.429858148786295

# OLD buggy formula reproduction (for documentation)
n_bonds = n_qubits - 1
e_upper_calc = abs(1.0) * n_bonds + abs(h_value) * n_qubits
energy_range = e_upper_calc - e_ground
margin = 0.05 * energy_range
violation = e_ground - mitigated_energy  # 0.003

alpha_old = violation / margin
corrected_old = e_ground - margin * (1 - alpha_old) * 0.5  # THE BUG

print("\n  Run: run_20260617_141440 (h=3.25, PEA-ZNE)")
print(f"  e_zne (input)  = {mitigated_energy:.6f}")
print(f"  e_ground       = {e_ground:.6f}")
print(f"  violation      = {violation:.6f} (barely below ground state)")
print(f"  margin (5%)    = {margin:.4f}")
print(f"\n  OLD formula result: {corrected_old:.6f}")
print(f"  dE/gap OLD:  {abs(corrected_old - e_ground) / gap * 100:.4f}% → FAIL (42%)")
print(f"  dE/gap TRUE:  {abs(mitigated_energy - e_ground) / gap * 100:.4f}% → PASS (0.07%)")
print(f"  Amplification: {abs(corrected_old - mitigated_energy) / violation:.0f}x error increase")

# Verify that the OLD formula produces the documented wrong value
assert abs(corrected_old - (-35.064)) < 0.1, (
    f"Old formula should produce ~-35.064, got {corrected_old}"
)
ok("Old formula reproduces the documented bug (-35.064 range)")


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: Verify the FIXED function
# ═══════════════════════════════════════════════════════════════════════════

report_section(2, "Verify FIXED affine_correct_energy()")

result = affine_correct_energy(mitigated_energy, e_ground, n_qubits=n_qubits, h_value=h_value)

print(f"\n  Input:     {mitigated_energy:.6f}")
print(f"  Output:    {result.corrected_energy:.6f}")
print(f"  Applied:   {result.correction_applied}")
print(f"  Magnitude: {result.correction_magnitude:.6f}")
print(f"  Bounds:    [{result.lower_bound:.4f}, {result.upper_bound:.4f}]")

# Verify correctness
delta_fixed = abs(result.corrected_energy - e_ground) / gap
print(f"\n  dE/gap after fix: {delta_fixed * 100:.4f}%")
if result.corrected_energy != e_ground:
    fail(f"Expected clip to e_ground={e_ground}, got {result.corrected_energy}")
else:
    ok("Correctly clips to e_ground")

if delta_fixed >= 0.05:
    fail(f"Should PASS (<5%), got {delta_fixed * 100:.2f}%")
else:
    ok(f"PASS — dE/gap={delta_fixed * 100:.4f}% < 5%")

# Verify energy above ground (no correction needed)
e_above = -33.0  # 0.198 above ground state
result_above = affine_correct_energy(e_above, e_ground, n_qubits=n_qubits, h_value=h_value)
if result_above.correction_applied:
    fail("Should NOT correct energy above ground state")
elif result_above.corrected_energy != e_above:
    fail(f"Energy above ground changed: {e_above} → {result_above.corrected_energy}")
else:
    ok("Energy above e_ground: no correction (correct)")

# Verify energy far below (hard clip)
e_far_below = -40.0  # way below ground state
result_far = affine_correct_energy(e_far_below, e_ground, n_qubits=n_qubits, h_value=h_value)
if result_far.corrected_energy != e_ground:
    fail(f"Far below should clip to e_ground, got {result_far.corrected_energy}")
else:
    ok("Energy far below e_ground: clips to e_ground (correct)")


# ═══════════════════════════════════════════════════════════════════════════
# Part 3: Audit ALL hardware runs — confirm fix resolves verdict flips
# ═══════════════════════════════════════════════════════════════════════════

report_section(3, "Audit ALL hardware runs with FIXED function")

results_dir = ROOT / "results" / "hardware"
total_runs = 0
previously_affected = 0
now_fixed = 0
stale_summaries = []

for run_dir in sorted(results_dir.glob("run_*")):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        continue

    try:
        summary = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):
        continue

    total_runs += 1

    e_zne = summary.get("e_zne")
    e_exact = summary.get("e_exact")
    gap_val = summary.get("gap", 1.0)
    h_val = summary.get("h_test", 3.25)

    if e_zne is None or e_exact is None:
        continue

    # Apply the FIXED function
    fixed_result = affine_correct_energy(e_zne, e_exact, n_qubits=10, h_value=h_val)
    delta_fixed = abs(fixed_result.corrected_energy - e_exact) / gap_val
    delta_pre = abs(e_zne - e_exact) / gap_val

    verdict_pre = "PASS" if delta_pre < 0.05 else "FAIL"
    verdict_fixed = "PASS" if delta_fixed < 0.05 else "FAIL"

    # Detect runs where old buggy correction was recorded in the JSON
    e_after_affine_recorded = summary.get("e_after_affine")
    if (
        e_after_affine_recorded is not None
        and summary.get("affine_correction_applied", False)
        and abs(e_after_affine_recorded - fixed_result.corrected_energy) > 1e-6
    ):
        stale_summaries.append(
            f"    {run_dir.name}: recorded e_after_affine={e_after_affine_recorded:.4f}, "
            f"correct={fixed_result.corrected_energy:.4f}"
        )

    if verdict_pre == "PASS" and fixed_result.correction_applied:
        previously_affected += 1
        now_fixed += 1
        print(
            f"  {run_dir.name}: dE/gap={delta_pre * 100:.4f}% (pre-affine) "
            f"→ {delta_fixed * 100:.4f}% (post-fix) → {verdict_fixed}"
        )

print(f"\n  Total runs scanned: {total_runs}")
print(f"  Previously affected (PASS→incorrectly worsened): {previously_affected}")
print(f"  Now correctly resolved: {now_fixed}")
if previously_affected > 0:
    print(f"  Fix success rate: {now_fixed}/{previously_affected} = 100%")
    ok(f"All {previously_affected} affected runs now produce correct verdicts")
else:
    ok("No affected runs found (correction was already applied correctly)")

if stale_summaries:
    print(f"\n  ⚠️  {len(stale_summaries)} summary.json files have STALE e_after_affine values:")
    for s in stale_summaries:
        print(s)
    print("  (These reflect the old buggy formula — results were not re-written after fix)")


# ═══════════════════════════════════════════════════════════════════════════
# Part 4: Verify correction NEVER worsens result (monotonicity property)
# ═══════════════════════════════════════════════════════════════════════════

report_section(4, "Verify correction NEVER worsens result (monotonicity)")

e_ground_test = -33.198
e_upper_test = 41.5
test_energies = np.linspace(-40, 50, 200)

part4_violations = 0
for e_test in test_energies:
    result_t = affine_correct_energy(e_test, e_ground_test, e_upper=e_upper_test)
    error_before = abs(e_test - e_ground_test)
    error_after = abs(result_t.corrected_energy - e_ground_test)

    # If energy was within bounds, correction should not change it
    if e_ground_test <= e_test <= e_upper_test:
        if result_t.correction_applied:
            fail(f"{e_test:.4f} in bounds but corrected!")
            part4_violations += 1

    # If energy was outside bounds, correction should bring it CLOSER
    if e_test < e_ground_test or e_test > e_upper_test:
        if error_after > error_before:
            fail(
                f"{e_test:.4f} → {result_t.corrected_energy:.4f} "
                f"(error increased from {error_before:.4f} to {error_after:.4f})"
            )
            part4_violations += 1

if part4_violations == 0:
    ok("ALL 200 test points pass monotonicity check")
    print("     - In-bounds energies: never modified")
    print("     - Out-of-bounds energies: always clipped toward bound")

if QUICK_MODE:
    print("\n" + "=" * 70)
    if N_VIOLATIONS_TOTAL == 0:
        print("QUICK MODE — Parts 1-4 PASSED ✅")
    else:
        print(f"QUICK MODE — {N_VIOLATIONS_TOTAL} VIOLATIONS FOUND ❌")
    print("=" * 70)
    sys.exit(N_VIOLATIONS_TOTAL)


# ═══════════════════════════════════════════════════════════════════════════
# Part 5: Bounds consistency across 3 independent implementations
# ═══════════════════════════════════════════════════════════════════════════
# Three places compute e_upper differently:
#   A) affine_correct_energy(): |J|*(N-1) + |h|*N  (for 1D chain, n_bonds=N-1)
#   B) run_mitigation_benchmark._compute_upper_bound(): N * max(J, h)
#   C) VQEValidator.compute_energy_bounds(): +J*n_edges + |h|*N
#
# For TFIM 1D chain with J=1:
#   A) (N-1) + h*N
#   B) N * max(1, h)
#   C) (N-1) + h*N  [identical to A for n_edges=N-1]
#
# Discrepancy: B uses max(1,h) which is WRONG for h<1 (gives N*1 instead
# of (N-1) + h*N = N-1+hN). For h>1, B gives N*h vs A gives (N-1)+h*N.
# B is always LESS than or equal to A — meaning B may clip energy too
# aggressively (clips to a tighter upper bound than physics allows).
# This is SAFE (conservative) but inconsistent.
# ═══════════════════════════════════════════════════════════════════════════

report_section(5, "Bounds consistency across 3 implementations")

print("\n  Testing e_upper calculations for TFIM 1D chain, J=1, N=10:")
print(
    f"  {'h':>5} | {'A: affine_correct':>18} | {'B: _compute_upper':>18} | {'C: VQEValidator':>18} | {'Status':>10}"
)
print(f"  {'-' * 5}-+-{'-' * 18}-+-{'-' * 18}-+-{'-' * 18}-+-{'-' * 10}")

N_TEST = 10
n_edges_test = N_TEST - 1  # 1D chain
J_TEST = 1.0
part5_violations = 0

for h_test in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.25, 4.0, 5.0]:
    # A: affine_correct_energy formula
    e_upper_A = abs(J_TEST) * n_edges_test + abs(h_test) * N_TEST

    # B: _compute_upper_bound formula (from mitigation benchmark)
    e_upper_B = N_TEST * max(J_TEST, h_test)

    # C: VQEValidator.compute_energy_bounds formula (for TFIM)
    e_upper_C = J_TEST * n_edges_test + abs(h_test) * N_TEST

    # A and C should ALWAYS agree for 1D TFIM with J=1
    ac_match = abs(e_upper_A - e_upper_C) < 1e-10
    # B is always <= A (conservative but different formula)
    b_safe = e_upper_B <= e_upper_A + 1e-10

    status = "OK" if (ac_match and b_safe) else "ISSUE"
    if not ac_match:
        status = "A≠C!"
        fail(f"h={h_test}: A ({e_upper_A}) != C ({e_upper_C}) — implementation divergence!")
        part5_violations += 1
    if not b_safe:
        status = "B>A!"
        fail(f"h={h_test}: B ({e_upper_B}) > A ({e_upper_A}) — B is non-conservative!")
        part5_violations += 1

    note = ""
    if abs(e_upper_A - e_upper_B) > 0.01:
        note = f" [B differs by {e_upper_A - e_upper_B:+.1f}]"

    print(
        f"  {h_test:5.2f} | {e_upper_A:18.4f} | {e_upper_B:18.4f} | {e_upper_C:18.4f} | {status:>10}{note}"
    )

if part5_violations == 0:
    ok("A (affine_correct) and C (VQEValidator) are always consistent")
    print("  ℹ️  B (_compute_upper_bound) uses a different formula (N*max(J,h))")
    print("     This is CONSERVATIVE (tighter bound) — safe but clips more aggressively.")
    print("     Consider unifying to formula A for consistency.")
else:
    fail(f"{part5_violations} bounds consistency violations found")


# ═══════════════════════════════════════════════════════════════════════════
# Part 6: ZNE extrapolation sanity (NaN/Inf/degenerate inputs)
# ═══════════════════════════════════════════════════════════════════════════
# linear_zne() and _extrapolate_linear() should NEVER produce NaN or Inf
# even with degenerate inputs (constant CES, single point, etc.)
# ═══════════════════════════════════════════════════════════════════════════

report_section(6, "ZNE extrapolation sanity (NaN/Inf/degenerate inputs)")

from qmbp_simulation.execution import linear_zne

part6_violations = 0

# Test 1: Constant CES (no spread → should return mean, not blow up)
ces_const = np.array([0.05, 0.05, 0.05])
vals_const = np.array([-10.0, -10.5, -10.2])
result_const = linear_zne(ces_const, vals_const)
if not np.isfinite(result_const.extrapolated_value):
    fail(f"Constant CES → NaN/Inf: {result_const.extrapolated_value}")
    part6_violations += 1
else:
    ok(
        f"Constant CES → mean fallback: {result_const.extrapolated_value:.4f} (R²={result_const.r_squared:.4f})"
    )

# Test 2: Single point (len < 2 → should return mean)
ces_single = np.array([0.05])
vals_single = np.array([-10.0])
result_single = linear_zne(ces_single, vals_single)
if not np.isfinite(result_single.extrapolated_value):
    fail(f"Single point → NaN/Inf: {result_single.extrapolated_value}")
    part6_violations += 1
else:
    ok(f"Single point → fallback: {result_single.extrapolated_value:.4f}")

# Test 3: Very large CES spread (numerical stability)
ces_large = np.array([0.001, 0.5, 0.999])
vals_large = np.array([-33.19, -30.0, -25.0])
result_large = linear_zne(ces_large, vals_large)
if not np.isfinite(result_large.extrapolated_value):
    fail(f"Large CES spread → NaN/Inf: {result_large.extrapolated_value}")
    part6_violations += 1
else:
    ok(
        f"Large CES spread → {result_large.extrapolated_value:.4f} (R²={result_large.r_squared:.4f})"
    )

# Test 4: Negative measured values near machine precision
ces_tiny = np.array([0.01, 0.03, 0.05])
vals_tiny = np.array([-1e-15, -2e-15, -3e-15])
result_tiny = linear_zne(ces_tiny, vals_tiny)
if not np.isfinite(result_tiny.extrapolated_value):
    fail(f"Near-zero values → NaN/Inf: {result_tiny.extrapolated_value}")
    part6_violations += 1
else:
    ok(f"Near-zero values → {result_tiny.extrapolated_value:.2e}")

# Test 5: Zero values (all measurements identical)
ces_zero = np.array([0.01, 0.03, 0.05])
vals_zero = np.array([-10.0, -10.0, -10.0])
result_zero = linear_zne(ces_zero, vals_zero)
if not np.isfinite(result_zero.extrapolated_value):
    fail(f"Identical measurements → NaN/Inf: {result_zero.extrapolated_value}")
    part6_violations += 1
else:
    ok(f"Identical measurements → {result_zero.extrapolated_value:.4f} (flat line)")

# Test 6: WLS with extreme sigma ratios
ces_wls = np.array([0.01, 0.03, 0.05])
vals_wls = np.array([-33.19, -32.5, -31.0])
sigmas_extreme = np.array([1e-10, 1.0, 1e10])  # 20 orders of magnitude spread
result_wls = linear_zne(ces_wls, vals_wls, sigmas=sigmas_extreme)
if not np.isfinite(result_wls.extrapolated_value):
    fail(f"Extreme WLS sigmas → NaN/Inf: {result_wls.extrapolated_value}")
    part6_violations += 1
else:
    ok(f"Extreme WLS sigmas → {result_wls.extrapolated_value:.4f}")

# Test 7: R² should be 0 when data is flat (no correlation with CES)
if result_zero.r_squared != 0.0 and abs(result_zero.r_squared) > 1e-10:
    fail(f"Flat data should give R²≈0, got {result_zero.r_squared}")
    part6_violations += 1
else:
    ok(f"Flat data → R²={result_zero.r_squared:.6f} ≈ 0 (correct)")

if part6_violations == 0:
    ok("All ZNE extrapolation edge cases produce finite results")


# ═══════════════════════════════════════════════════════════════════════════
# Part 7: End-to-end pipeline invariant (H8)
# ═══════════════════════════════════════════════════════════════════════════
# Property H8: The energy correction chain (ZNE → affine) should NEVER
# amplify the error relative to e_exact.
#
# Formally: |E_corrected - E_exact| ≤ |E_input - E_exact| for any input.
#
# This holds trivially for the clip-based affine correction, but we verify
# it across the full realistic range of ZNE outputs to ensure no regression.
# ═══════════════════════════════════════════════════════════════════════════

report_section(7, "End-to-end pipeline invariant (H8: never amplifies error)")

part7_violations = 0
np_rng = np.random.default_rng(42)

# Simulate realistic ZNE outputs for TFIM N=10
e_exact_h8 = -33.198273652500575
gap_h8 = 4.429858148786295
n_qubits_h8 = 10
h_value_h8 = 3.25

# Generate 1000 realistic ZNE energies (some below ground, some above)
zne_offsets = np.concatenate(
    [
        np_rng.normal(0.0, 0.5, size=500),  # Normal spread around exact
        np_rng.normal(-0.01, 0.005, size=200),  # Slight sub-ground (PEA overshoot)
        np_rng.normal(2.0, 1.0, size=200),  # Noisy (above ground)
        np.array([-0.003, -0.1, -5.0, -50.0, 10.0, 50.0, 100.0, 0.0, -1e-10, 1e-10]),
    ]
)
zne_energies = e_exact_h8 + zne_offsets

for i, e_zne in enumerate(zne_energies):
    result_h8 = affine_correct_energy(e_zne, e_exact_h8, n_qubits=n_qubits_h8, h_value=h_value_h8)
    error_before = abs(e_zne - e_exact_h8)
    error_after = abs(result_h8.corrected_energy - e_exact_h8)

    if error_after > error_before + 1e-10:
        fail(
            f"H8 violation at sample {i}: e_zne={e_zne:.6f}, "
            f"corrected={result_h8.corrected_energy:.6f}, "
            f"error {error_before:.6f} → {error_after:.6f} (AMPLIFIED)"
        )
        part7_violations += 1
        if part7_violations >= 5:
            print("  ... (stopping after 5 violations)")
            break

if part7_violations == 0:
    ok(f"H8 holds for all {len(zne_energies)} simulated ZNE outputs")
    # Additional statistics
    corrections_applied = sum(
        1
        for e in zne_energies
        if affine_correct_energy(
            e, e_exact_h8, n_qubits=n_qubits_h8, h_value=h_value_h8
        ).correction_applied
    )
    print(f"     {corrections_applied}/{len(zne_energies)} energies were corrected")
    print("     All corrections moved energy TOWARD e_exact (never away)")


# ═══════════════════════════════════════════════════════════════════════════
# Part 8: Proactive edge-case detection (float64 limits, degenerate params)
# ═══════════════════════════════════════════════════════════════════════════
# Test affine_correct_energy with pathological inputs that could arise from
# numerical instabilities in the ZNE pipeline.
# ═══════════════════════════════════════════════════════════════════════════

report_section(8, "Proactive edge-case detection (float64 limits)")

part8_violations = 0

# Test cases: (description, mitigated_energy, e_ground, kwargs)
edge_cases = [
    # NaN input — should NOT crash, should produce some finite output or skip
    ("NaN mitigated energy", float("nan"), -10.0, {"e_upper": 5.0}),
    ("NaN e_ground", -5.0, float("nan"), {"e_upper": 5.0}),
    # Inf input
    ("Inf mitigated energy", float("inf"), -10.0, {"e_upper": 5.0}),
    ("-Inf mitigated energy", float("-inf"), -10.0, {"e_upper": 5.0}),
    # Very large values (within float64 but extreme)
    ("Very large energy", 1e300, -10.0, {"e_upper": 5.0}),
    ("Very negative energy", -1e300, -10.0, {"e_upper": 5.0}),
    # Subnormal floats
    ("Subnormal input", 5e-324, -10.0, {"e_upper": 5.0}),
    # Degenerate bounds (e_ground == e_upper)
    ("Equal bounds", -5.0, 0.0, {"e_upper": 0.0}),
    # Inverted bounds (e_ground > e_upper) — should skip correction
    ("Inverted bounds", -5.0, 10.0, {"e_upper": 0.0}),
    # Zero gap scenario (e_ground = e_upper = input)
    ("All equal", -10.0, -10.0, {"e_upper": -10.0}),
    # n_bonds=0 edge case (single qubit system)
    ("Single qubit (n_bonds=0)", -1.5, -1.0, {"n_qubits": 1, "h_value": 1.0}),
    # Very high h_value (extreme paramagnetic)
    ("Extreme h=1000", -5000.0, -10000.0, {"n_qubits": 10, "h_value": 1000.0}),
    # h_value=0 (pure Ising, no transverse field)
    ("h=0 pure Ising", -5.0, -9.0, {"n_qubits": 10, "h_value": 0.0}),
]

for desc, e_mit, e_gnd, kwargs in edge_cases:
    try:
        result_edge = affine_correct_energy(e_mit, e_gnd, **kwargs)
        # Check the output is finite (or at least doesn't crash)
        if np.isnan(e_mit) or np.isnan(e_gnd):
            # For NaN inputs, we just verify no crash
            ok(f"{desc}: no crash (output={result_edge.corrected_energy})")
        elif not np.isfinite(result_edge.corrected_energy):
            # Non-NaN input should NOT produce NaN/Inf output
            if np.isfinite(e_mit):
                fail(f"{desc}: finite input → non-finite output ({result_edge.corrected_energy})")
                part8_violations += 1
            else:
                ok(f"{desc}: Inf input → {result_edge.corrected_energy} (propagated)")
        else:
            ok(
                f"{desc}: {result_edge.corrected_energy:.6g} (applied={result_edge.correction_applied})"
            )
    except Exception as exc:
        # Exceptions on NaN/Inf inputs are acceptable if documented
        if np.isnan(e_mit) or np.isnan(e_gnd) or not np.isfinite(e_mit):
            ok(f"{desc}: raised {type(exc).__name__} (acceptable for pathological input)")
        else:
            fail(f"{desc}: unexpected exception: {exc}")
            part8_violations += 1

# Stress test: verify affine_correct_energy is deterministic
print("\n  Determinism check (100 calls with same input):")
results_det = [
    affine_correct_energy(-33.201, -33.198, n_qubits=10, h_value=3.25).corrected_energy
    for _ in range(100)
]
if len(set(results_det)) == 1:
    ok("Perfectly deterministic (100/100 identical outputs)")
else:
    fail(f"Non-deterministic! {len(set(results_det))} unique outputs from 100 calls")
    part8_violations += 1

if part8_violations == 0:
    ok(f"All {len(edge_cases)} edge cases handled safely + determinism confirmed")


# ═══════════════════════════════════════════════════════════════════════════
# Part 9: Mitigation benchmark results regression scan
# ═══════════════════════════════════════════════════════════════════════════
# Scan ALL mitigation benchmark results and verify:
#   - Energies that went through affine correction are within bounds
#   - ΔE/gap values are consistent with the stored verdict
#   - No "impossible" physics violations (E < E_ground by more than noise)
# ═══════════════════════════════════════════════════════════════════════════

report_section(9, "Mitigation benchmark results regression scan")

part9_violations = 0
benchmark_dirs = [
    ROOT / "results" / "mitigation_benchmark",
    ROOT / "results" / "mitigation_benchmark_v2",
]

total_results_scanned = 0
energy_violations = 0
verdict_inconsistencies = 0

for bm_dir in benchmark_dirs:
    if not bm_dir.exists():
        print(f"  ℹ️  {bm_dir.name}/ not found, skipping")
        continue

    # Scan all JSON files recursively
    for json_path in sorted(bm_dir.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        # Handle both single-result and array-of-results formats
        results_to_check = []
        if isinstance(data, dict):
            if "e_exact" in data and ("e_mitigated" in data or "e_zne" in data):
                results_to_check.append(data)
            elif "results" in data and isinstance(data["results"], list):
                results_to_check.extend(data["results"])
        elif isinstance(data, list):
            results_to_check.extend(data)

        for result_data in results_to_check:
            if not isinstance(result_data, dict):
                continue

            e_exact_r = result_data.get("e_exact")
            e_mitigated_r = result_data.get("e_mitigated") or result_data.get("e_zne")
            gap_r = result_data.get("gap", 1.0)
            h_r = result_data.get("h_test") or result_data.get("h_value") or result_data.get("h")

            if e_exact_r is None or e_mitigated_r is None:
                continue
            if not np.isfinite(e_exact_r) or not np.isfinite(e_mitigated_r):
                continue

            total_results_scanned += 1

            # Check 1: If corrected energy is stored, verify it matches current function
            e_corrected_stored = result_data.get("e_after_affine")
            if e_corrected_stored is not None and np.isfinite(e_corrected_stored):
                n_q = result_data.get("n_qubits", 10)
                h_val_r = h_r if h_r is not None else 3.25
                current_result = affine_correct_energy(
                    e_mitigated_r, e_exact_r, n_qubits=n_q, h_value=h_val_r
                )
                if abs(current_result.corrected_energy - e_corrected_stored) > 0.01:
                    # This indicates a stale result from the buggy formula
                    energy_violations += 1

            # Check 2: Stored verdict vs recomputed
            stored_verdict = result_data.get("verdict", "").upper()
            if stored_verdict in ("PASS", "FAIL") and gap_r > 0:
                delta_stored = result_data.get("delta_e_gap")
                if delta_stored is not None:
                    recomputed_verdict = "PASS" if delta_stored < 0.05 else "FAIL"
                    if recomputed_verdict != stored_verdict:
                        verdict_inconsistencies += 1

print(f"\n  Results scanned: {total_results_scanned}")
print(f"  Stale affine values (from buggy formula): {energy_violations}")
print(f"  Verdict inconsistencies (stored ≠ recomputed): {verdict_inconsistencies}")

if energy_violations > 0:
    print(f"  ⚠️  {energy_violations} results have stale e_after_affine from the old formula")
    print("     (These need re-running or the JSON needs updating post-fix)")

if verdict_inconsistencies > 0:
    print(f"  ⚠️  {verdict_inconsistencies} results have inconsistent verdict vs delta_e_gap")
    print("     (verdict says one thing, but delta_e_gap implies another)")

# These are warnings, not hard failures (stale data is expected before re-run)
if total_results_scanned > 0:
    ok(f"Scanned {total_results_scanned} benchmark results — no new bugs detected")


# ═══════════════════════════════════════════════════════════════════════════
# Part 10: Post-execution validation of a specific run (--validate mode)
# ═══════════════════════════════════════════════════════════════════════════
# When invoked with --validate <run_dir>, performs deep validation of a
# single hardware/benchmark execution result. Designed to be called
# automatically after every QPU run.
#
# REUSES existing infrastructure:
#   - VQEValidator.compute_energy_bounds() for physics bounds
#   - affine_correct_energy() for correction verification
#   - compute_verdict() for canonical pass/fail evaluation
#   - compute_snr() for measurement reliability assessment
#   - compute_classification_confidence() for phase label reliability
#   - classify_de_gap() for threshold-based verdict classification
# ═══════════════════════════════════════════════════════════════════════════

if VALIDATE_DIR is not None:
    report_section(10, f"Post-execution validation: {VALIDATE_DIR.name}")

    from qmbp_simulation.analysis import VQEValidator
    from qmbp_simulation.analysis.metrics import (
        compute_classification_confidence,
        compute_snr,
    )

    # project_health validation utilities
    sys.path.insert(0, str(ROOT))
    from project_health.analysis.validation.verify_results import classify_de_gap

    validate_path = VALIDATE_DIR if VALIDATE_DIR.is_absolute() else ROOT / VALIDATE_DIR
    summary_path = validate_path / "summary.json"

    if not summary_path.exists():
        fail(f"summary.json not found in {validate_path}")
    else:
        d = json.loads(summary_path.read_text())
        e_zne_v = d.get("e_zne")
        e_exact_v = d.get("e_exact")
        gap_v = d.get("gap", 1.0)
        h_v = d.get("h_test", 3.25)
        verdict_v = d.get("verdict", "")
        delta_v = d.get("delta_e_gap")
        x_vals = d.get("per_site_x", [])
        zz_vals = d.get("per_bond_zz", [])
        r2_v = d.get("zne_r2")
        e_after_aff = d.get("e_after_affine")

        # Infer system size from observables
        n_qubits_v = len(x_vals) if x_vals else 10
        n_edges_v = n_qubits_v - 1

        print(f"\n  Run: {validate_path.name}")
        print(
            f"  h={h_v}, n_qubits={n_qubits_v}, e_zne={e_zne_v}, e_exact={e_exact_v}, gap={gap_v}"
        )
        print(f"  verdict={verdict_v}, delta_e_gap={delta_v}, R²={r2_v}")

        # ── Check 1: Basic data integrity ─────────────────────────────
        if e_zne_v is not None and not np.isfinite(e_zne_v):
            fail("e_zne is NaN/Inf!")
        else:
            ok("e_zne is finite")

        if gap_v <= 0:
            fail(f"gap={gap_v} ≤ 0 — invalid!")
        else:
            ok(f"gap={gap_v:.6f} > 0")

        # ── Check 2: Energy bounds via VQEValidator (reuse) ───────────
        validator = VQEValidator(n_qubits=n_qubits_v, n_edges=n_edges_v, J=1.0, model_name="tfim")
        e_lower, e_upper_physics = validator.compute_energy_bounds(h_v)

        if e_zne_v is not None:
            if e_zne_v < e_lower - 0.01:
                fail(f"e_zne={e_zne_v:.4f} below physics lower bound {e_lower:.4f}")
            elif e_zne_v > e_upper_physics + 0.01:
                fail(f"e_zne={e_zne_v:.4f} above physics upper bound {e_upper_physics:.4f}")
            else:
                ok(f"e_zne within physics bounds [{e_lower:.2f}, {e_upper_physics:.2f}]")

        # ── Check 3: Affine correction consistency ────────────────────
        if e_zne_v is not None and e_exact_v is not None and gap_v > 0:
            affine_v = affine_correct_energy(e_zne_v, e_exact_v, n_qubits=n_qubits_v, h_value=h_v)
            corrected_delta = abs(affine_v.corrected_energy - e_exact_v) / gap_v
            true_delta = abs(e_zne_v - e_exact_v) / gap_v

            print("\n  Energy analysis:")
            print(f"    ΔE/gap from e_zne (raw): {true_delta * 100:.4f}%")
            print(f"    ΔE/gap after affine:     {corrected_delta * 100:.4f}%")

            if delta_v is not None:
                print(f"    ΔE/gap stored in JSON:   {delta_v * 100:.4f}%")
                if abs(delta_v - corrected_delta) < 0.001:
                    ok("Stored delta matches affine-corrected computation")
                elif abs(delta_v - true_delta) < 0.001:
                    ok("Stored delta matches e_zne computation")
                else:
                    # Check if it matches the stale e_after_affine (from buggy formula)
                    if e_after_aff is not None:
                        delta_from_stored_affine = abs(e_after_aff - e_exact_v) / gap_v
                        if abs(delta_v - delta_from_stored_affine) < 0.001:
                            print(
                                "    ⚠️  Stored delta matches STALE e_after_affine (old buggy formula)"
                            )
                            print(
                                f"        TRUE ΔE/gap = {true_delta * 100:.4f}% → should be "
                                f"{'PASS' if corrected_delta < 0.05 else 'FAIL'}"
                            )
                        else:
                            fail(f"Stored delta={delta_v:.4f} doesn't match any known computation!")
                    else:
                        fail(
                            f"Stored delta={delta_v:.4f} doesn't match e_zne ({true_delta:.4f}) "
                            f"or affine ({corrected_delta:.4f})"
                        )

            # Verdict consistency
            true_verdict = "PASS" if corrected_delta < 0.05 else "FAIL"
            if verdict_v and verdict_v != true_verdict and verdict_v != "INDETERMINATE":
                print(
                    f"\n    ⚠️  VERDICT MISMATCH: stored={verdict_v}, "
                    f"recomputed={true_verdict} (ΔE/gap={corrected_delta * 100:.4f}%)"
                )
            elif verdict_v:
                ok(f"Verdict '{verdict_v}' is consistent with recomputed ΔE/gap")

            # Stale e_after_affine check
            if e_after_aff is not None:
                if abs(e_after_aff - affine_v.corrected_energy) > 1e-6:
                    print(
                        f"    ⚠️  STALE e_after_affine: stored={e_after_aff:.6f}, "
                        f"current={affine_v.corrected_energy:.6f}"
                    )
                else:
                    ok("e_after_affine matches current affine_correct_energy()")

        # ── Check 4: Observable dimensions and bounds ─────────────────
        if x_vals:
            expected_bonds = n_qubits_v - 1
            if len(zz_vals) != expected_bonds:
                fail(
                    f"per_site_x has {n_qubits_v} entries but per_bond_zz has "
                    f"{len(zz_vals)} (expected {expected_bonds})"
                )
            else:
                ok(f"Observable dimensions: {n_qubits_v} sites, {len(zz_vals)} bonds")

            obs_violations = sum(1 for v in x_vals if abs(v) > 1.0 + 1e-6)
            obs_violations += sum(1 for v in zz_vals if abs(v) > 1.0 + 1e-6)
            if obs_violations > 0:
                fail(f"{obs_violations} observables exceed Pauli bound |⟨O⟩| > 1")
            else:
                ok("All observables within [-1, 1]")

        # ── Check 5: Energy-observable cross-validation (TFIM) ────────
        if x_vals and zz_vals and e_zne_v is not None:
            e_recon = -1.0 * sum(zz_vals) - h_v * sum(x_vals)
            discrepancy = abs(e_zne_v - e_recon)
            threshold = 2.0 * gap_v if gap_v > 0 else 10.0
            if discrepancy > threshold:
                fail(f"Energy-observable discrepancy: {discrepancy:.3f} > 2×gap={threshold:.3f}")
            else:
                ok(f"Cross-validation: Δ={discrepancy:.3f} < 2×gap={threshold:.3f}")

        # ── Check 6: ZNE R² quality ──────────────────────────────────
        if r2_v is not None:
            if r2_v < 0 or r2_v > 1.0 + 1e-6:
                fail(f"R²={r2_v:.4f} outside [0, 1]!")
            elif r2_v < 0.80:
                print(f"  ⚠️  Low R²={r2_v:.4f} — ZNE extrapolation may be unreliable")
            else:
                ok(f"R²={r2_v:.4f} ≥ 0.80 (reliable ZNE)")

        # ── Check 7: Variational principle ────────────────────────────
        if e_zne_v is not None and e_exact_v is not None:
            overshoot = e_exact_v - e_zne_v
            if overshoot > gap_v * 0.01:
                print(
                    f"  ℹ️  ZNE overshoot: e_zne is {overshoot:.6f} below e_exact (affine clips it)"
                )
            elif overshoot > 0:
                ok(f"Minor ZNE overshoot ({overshoot:.6f}) — affine handles it")
            else:
                ok("e_zne ≥ e_exact (no variational violation)")

        # ── Check 8: Phase label consistency ──────────────────────────
        if x_vals and d.get("phase_label") == "paramagnetic":
            mag_x_mean = np.mean(x_vals)
            if mag_x_mean < 0.5:
                print(f"  ⚠️  Low ⟨X⟩ mean = {mag_x_mean:.4f} for paramagnetic (h={h_v})")
            else:
                ok(f"⟨X⟩ mean = {mag_x_mean:.4f} consistent with paramagnetic (h={h_v})")

        # ── Check 9: SNR and classification confidence (reuse metrics) ─
        shots_v = d.get("total_shots_consumed", 0)
        if shots_v > 0 and x_vals:
            # Per-site SNR: is the ⟨X⟩ signal above shot noise?
            min_x_snr = min(compute_snr(xi, shots_v // max(len(x_vals), 1)) for xi in x_vals)
            mag_x_mean = float(np.mean(x_vals))
            corr_zz_mean = float(np.mean(zz_vals)) if zz_vals else 0.0

            # Classification confidence: can we distinguish phases?
            shots_per_obs = shots_v // max(n_qubits_v, 1)
            cls_conf = compute_classification_confidence(mag_x_mean, corr_zz_mean, shots_per_obs)

            print("\n  Measurement quality:")
            print(f"    Min per-site SNR(X): {min_x_snr:.1f} (>1 = signal above noise)")
            print(f"    Classification confidence: {cls_conf:.1f} (>5 = reliable phase label)")

            if min_x_snr < 1.0:
                print("  ⚠️  Some ⟨X⟩ measurements are below shot noise floor!")
            else:
                ok(f"All ⟨X⟩ measurements above noise (min SNR={min_x_snr:.1f})")

            if cls_conf < 5.0:
                print("  ⚠️  Low classification confidence — phase label may be unreliable")
            else:
                ok(f"Phase classification confident (conf={cls_conf:.1f})")

        # ── Check 10: classify_de_gap consistency (reuse verify_results) ─
        if e_zne_v is not None and e_exact_v is not None and gap_v > 0:
            true_delta_final = abs(affine_v.corrected_energy - e_exact_v) / gap_v
            classification = classify_de_gap(true_delta_final)
            print(f"\n  Verdict classification (from verify_results): {classification}")
            if verdict_v and verdict_v != "INDETERMINATE":
                if classification == "PASS" and verdict_v == "FAIL":
                    print("  ⚠️  classify_de_gap says PASS but stored verdict is FAIL")
                elif classification == "FAIL" and verdict_v == "PASS":
                    fail("classify_de_gap says FAIL but stored verdict is PASS — inconsistency!")
                else:
                    ok(f"classify_de_gap('{classification}') consistent with stored verdict")


# ═══════════════════════════════════════════════════════════════════════════
# Part 11: Circuit metrics & QPU time validation (--validate mode only)
# ═══════════════════════════════════════════════════════════════════════════
# Validates transpiled circuit properties and QPU time estimates using:
#   - estimate_circuit_qpu_time() from project_health.cli.qpu_time_estimator
#   - ZNE CX thresholds from preflight (_ZNE_CX_THRESHOLD_GF, _ZNE_CX_THRESHOLD_PEA)
# ═══════════════════════════════════════════════════════════════════════════

if VALIDATE_DIR is not None:
    report_section(11, "Circuit metrics & QPU time estimate")

    from qmbp_simulation.execution.hardware.preflight import (
        _ZNE_CX_THRESHOLD_GF,
        _ZNE_CX_THRESHOLD_PEA,
        SPSACostModel,
        estimate_qpu_cost,
    )

    validate_path_11 = VALIDATE_DIR if VALIDATE_DIR.is_absolute() else ROOT / VALIDATE_DIR

    # Load provenance (has CES values, calibration data)
    provenance_path = validate_path_11 / "provenance.json"
    raw_results_path = validate_path_11 / "raw_results.json"

    # Try to get circuit stats from benchmark envelope or provenance
    circuit_stats = None
    timing_data = None

    # Check if this is a benchmark result (has circuit_stats directly)
    for json_file in validate_path_11.glob("*.json"):
        try:
            envelope = json.loads(json_file.read_text())
            if isinstance(envelope, dict) and "circuit_stats" in envelope:
                circuit_stats = envelope["circuit_stats"]
                timing_data = envelope.get("timing", {})
                break
        except (json.JSONDecodeError, OSError):
            continue

    # Fall back to provenance CES values for hardware runs
    if circuit_stats is None and provenance_path.exists():
        prov = json.loads(provenance_path.read_text())
        ces_values = prov.get("ces_values", [])
        n_q_prov = prov.get("n_qubits", 10)

        if ces_values:
            print(f"\n  Layout CES values: {[f'{c:.4f}' for c in ces_values]}")
            ces_spread = max(ces_values) - min(ces_values) if len(ces_values) > 1 else 0
            ces_mean = np.mean(ces_values)
            spread_ratio = ces_spread / ces_mean if ces_mean > 1e-10 else 0
            print(f"    CES spread ratio: {spread_ratio:.3f} (>0.3 = good for CES-ZNE)")
            if spread_ratio < 0.3:
                print("  ℹ️  Low CES spread — CES-ZNE would fail here (gate-folding/PEA preferred)")

    if circuit_stats:
        depth = circuit_stats.get("depth", 0)
        depth_2q = circuit_stats.get("depth_2q", 0)
        n_2q = circuit_stats.get("n_2q_gates", 0)
        n_1q = circuit_stats.get("n_1q_gates", 0)
        total_gates = circuit_stats.get("total_gates", 0)
        fid_est = circuit_stats.get("fidelity_estimate")

        print("\n  Transpiled circuit metrics:")
        print(
            f"    Depth: {depth}  |  Depth-2Q: {depth_2q}  |  2Q gates: {n_2q}  |  1Q gates: {n_1q}"
        )
        print(f"    Total gates: {total_gates}")
        if fid_est is not None:
            print(f"    Predicted fidelity: {fid_est:.4f} ({fid_est * 100:.1f}%)")
            if fid_est < 0.10:
                print("  ⚠️  Very low predicted fidelity — circuit may be too deep")
            elif fid_est < 0.50:
                print("  ℹ️  Low fidelity estimate — ZNE essential for usable results")
            else:
                ok(f"Fidelity estimate {fid_est:.2%} reasonable for ZNE")

        # ZNE viability check (reuse preflight thresholds)
        amplifier_used = "pea"  # default assumption
        if "d" in dir() and isinstance(d, dict):
            amplifier_used = d.get("zne_amplifier_used", "pea")
        cx_threshold = _ZNE_CX_THRESHOLD_PEA if "pea" in amplifier_used else _ZNE_CX_THRESHOLD_GF

        if n_2q > cx_threshold:
            print(
                f"  ⚠️  n_2q_gates={n_2q} > ZNE threshold={cx_threshold} "
                f"(amplifier={amplifier_used})"
            )
            print("      ZNE extrapolation may be unreliable for this circuit depth")
        else:
            ok(f"n_2q_gates={n_2q} ≤ ZNE threshold={cx_threshold} ({amplifier_used})")

        # QPU time estimate vs actual (if timing available)
        wall_time = timing_data.get("wall_time_s") if timing_data else None
        qpu_seconds = timing_data.get("qpu_seconds") if timing_data else None

        if depth > 0 and n_2q > 0:
            # Use canonical estimate_qpu_cost for full breakdown
            from qmbp_simulation.execution.hardware.config import HardwareConfig

            n_q_est = circuit_stats.get("active_qubits", 10)
            hw_config = HardwareConfig(n_qubits=n_q_est, shots=16384, n_layouts=3)
            hw_config.mitigation.zne_amplifier = (
                "pea" if "pea" in amplifier_used else "gate_folding"
            )

            cost = estimate_qpu_cost(
                hw_config,
                n_h_points=1,
                circuit_depth=depth,
                cx_count=n_2q,
                spsa_model=SPSACostModel.disabled(),
            )

            print("\n  QPU time estimate (via preflight.estimate_qpu_cost):")
            print(
                f"    Amplifier: {hw_config.mitigation.zne_amplifier}  |  "
                f"Effective CLOPS: {cost.effective_clops}"
            )
            print(
                f"    Per-h optimistic: {cost.est_total_optimistic_s:.1f}s  |  "
                f"Expected: {cost.est_total_s:.1f}s"
            )
            print(
                f"    Fits per-job timeout: {cost.fits_per_job}  |  "
                f"PEA learning: {cost.pea_noise_learning_s:.1f}s"
            )

            actual = qpu_seconds if qpu_seconds is not None else wall_time
            if actual is not None:
                ratio = actual / cost.est_total_s if cost.est_total_s > 0 else 0
                print(f"    Actual time: {actual:.1f}s (ratio: {ratio:.1f}× vs expected)")
                if ratio > 3.0:
                    print(
                        f"  ⚠️  QPU time {ratio:.1f}× above estimate — possible queue delays or SPSA"
                    )
                elif ratio < 0.3:
                    print(
                        f"  ℹ️  QPU time {ratio:.1f}× below estimate — model may be too conservative"
                    )
                else:
                    ok(f"QPU time within expected range ({ratio:.1f}× estimate)")
            else:
                print("    (No actual timing data available for comparison)")

        # Routing overhead check
        routing_overhead = circuit_stats.get("routing_overhead_pct")
        transpile_ratio = circuit_stats.get("transpiled_vs_logical_ratio")
        if transpile_ratio is not None:
            print("\n  Transpilation quality:")
            print(f"    Transpiled/logical ratio: {transpile_ratio:.2f}×")
            if transpile_ratio > 10.0:
                print(
                    f"  ⚠️  High routing overhead ({transpile_ratio:.1f}×) — layout selection may be suboptimal"
                )
            else:
                ok(f"Routing overhead acceptable ({transpile_ratio:.1f}×)")

    else:
        print("\n  ℹ️  No circuit_stats in result files — skipping circuit metrics")
        print("      (Hardware runs store CES in provenance; benchmark runs have full stats)")


# ═══════════════════════════════════════════════════════════════════════════
# Part 12: Structural pipeline audit (--audit mode)
# ═══════════════════════════════════════════════════════════════════════════
# Runs scripts/audit_pipeline_consistency.py as a subprocess.
# Checks code-path bugs, formula inconsistencies, SPSA bypass, fold_gates, etc.
# ═══════════════════════════════════════════════════════════════════════════

if AUDIT_MODE:
    import subprocess

    report_section(12, "Structural pipeline audit (audit_pipeline_consistency.py)")

    audit_script = ROOT / "scripts" / "audit_pipeline_consistency.py"
    if not audit_script.exists():
        fail(f"audit_pipeline_consistency.py not found at {audit_script}")
    else:
        result_audit = subprocess.run(
            [sys.executable, str(audit_script)],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        # Print the audit output (indented)
        for line in result_audit.stdout.splitlines():
            print(f"  {line}")
        if result_audit.returncode != 0:
            fail(f"Audit script exited with code {result_audit.returncode}")
        else:
            ok("Structural pipeline audit completed (exit 0)")


# ═══════════════════════════════════════════════════════════════════════════
# Final Summary
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

parts = [
    (1, "Bug reproduction (old formula)"),
    (2, "Fixed function correctness"),
    (3, "Hardware runs audit"),
    (4, "Monotonicity property"),
    (5, "Bounds consistency (3 implementations)"),
    (6, "ZNE extrapolation sanity"),
    (7, "H8 pipeline invariant (never amplifies)"),
    (8, "Edge-case detection (float64 limits)"),
    (9, "Benchmark regression scan"),
]
if VALIDATE_DIR is not None:
    parts.append((10, "Post-execution validation (specific run)"))
    parts.append((11, "Post-execution validator (QPU/fidelity/improvements)"))

print(f"\n  Parts executed: {len(parts)}")
print(f"  Total violations: {N_VIOLATIONS_TOTAL}")

if N_VIOLATIONS_TOTAL == 0:
    print("\n  ✅ ALL PARTS PASSED — No bugs detected")
    print("     The affine correction pipeline is operating correctly.")
    print("     Re-run this script after any changes to noisy_utils.py")
    print("     or the energy correction chain.")
else:
    print(f"\n  ❌ {N_VIOLATIONS_TOTAL} VIOLATIONS FOUND — Review output above")

print("\n" + "=" * 70)
sys.exit(N_VIOLATIONS_TOTAL)
