# Hardware Rehearsal Findings — 2026-06-03

## Execution Summary

| Section | Status | Time | Key Finding |
|---------|--------|------|-------------|
| 1. MPNN Prediction Quality | ✅ PASS | 15.4s | ΔE/gap ≤ 0.8% noiseless (in-regime) |
| 2. End-to-End Noisy Pipeline | ❌ FAIL | 193.6s | ZNE gain only 16% (CES outlier) |
| 3. Observable SNR & Classification | ✅ PASS | 4.3s | 100% correct, SNR ≫ 1 |
| 4. Layout Stability | ❌ FAIL | 29.8s | std(ΔE/gap) > 2% across trials |
| 5. Shot Noise Sensitivity | ✅ PASS | 20.5s | std = 0.012 < 3% |

**Total time**: 264s (~4.4 min)

---

## Critical Finding: CES Outlier in Layout Selection

### Problem

`select_layouts_by_circuit_ces()` chose 3 layouts with CES values:
- Layout 0: CES = 0.151 (excellent)
- Layout 1: CES = **14.4–15.3** (catastrophic — 100× worse)
- Layout 2: CES = 0.211 (excellent)

The second layout introduces so much noise that ZNE extrapolation produces
a result far from the true energy. ZNE gain was only +16% (vs +63% in
previous validated runs that used uniform-CES layouts).

### Root Cause

The `select_layouts_by_circuit_ces()` function selects layouts to maximize
CES spread (for better extrapolation leverage), but this strategy backfires
when one layout has catastrophically high CES. The linear extrapolation
assumes noise is perturbative — at CES=14.4, the noise is non-perturbative
and the linear model breaks down.

### Solution

Use `select_layouts_low_ces()` with `max_ces` filter instead:

```python
from qmbp_simulation.execution import select_layouts_low_ces

# CORRECT: filter out high-CES layouts
layout_sel = select_layouts_low_ces(
    bound_circuit, fake_backend, candidates,
    n_select=3, max_ces=0.5  # Only accept CES < 0.5
)
```

This matches the HARDWARE_DEPLOYMENT_SPEC §11 risk mitigation:
> "select_layouts_low_ces(bound_circuit, backend, candidates, n_select=3, max_ces=0.5)"

### Impact on Hardware Deployment

**MUST fix before hardware execution.** The current script uses
`select_layouts_by_circuit_ces` (spread-maximizing) which can select
catastrophic layouts. Switch to `select_layouts_low_ces` (filter-based).

---

## Section-by-Section Analysis

### Section 1: MPNN Prediction Quality ✅

| h_test | ΔE/gap (noiseless) | Status |
|--------|:------------------:|--------|
| 4.0 | 0.22% | ✅ Pass |
| 3.25 | 0.54% | ✅ Pass |
| 3.0 | 0.77% | ✅ Pass |
| 2.5 | 9.46% | Expected fail (below regime) |

**Conclusion**: MPNN predictions are excellent in-regime. The warm-start
strategy is validated: no VQE refinement needed on hardware for h ≥ 3.0.

### Section 2: End-to-End Noisy Pipeline ❌

| h_test | ΔE/gap(noiseless) | ΔE/gap(noisy) | ΔE/gap(ZNE) | R² | Gain |
|--------|:-----------------:|:-------------:|:-----------:|:--:|:----:|
| 4.0 | 0.22% | 103.5% | 86.4% | 0.996 | +16% |
| 3.25 | 0.54% | 114.0% | 95.6% | 0.997 | +16% |
| 3.0 | 0.77% | 119.0% | 100.2% | 0.996 | +16% |
| 2.5 | 9.46% | — | — | — | — |

**Diagnosis**: R² is excellent (>0.99) but ZNE gain is only 16% because
the CES outlier (14.4) distorts the linear fit. Previous validated runs
achieved +63% gain with uniform CES values (0.15–0.45 range).

### Section 3: Observable SNR & Classification ✅

All in-regime points correctly classified as "paramagnetic". At h=3.25:
- ⟨X⟩ = -0.965 (SNR = 123.2σ — extremely strong signal)
- ⟨ZZ⟩ = 0.041 (SNR = 5.3σ)
- Confidence = 118σ (classification is unambiguous)

**Conclusion**: Phase classification will work on hardware even with
significant noise. The ⟨X⟩ signal at h≥3.0 is 120× above the noise floor.

### Section 4: Layout Stability ❌

5 independent layout selections at h=3.25:

| Trial | ΔE/gap(ZNE) | R² | CES values |
|-------|:-----------:|:--:|------------|
| 0 | varies | 0.99+ | Includes outlier CES |
| 1-4 | varies | 0.99+ | Different outlier patterns |

std(ΔE/gap) > 2% because the outlier-CES layout selection is non-deterministic.

**Fix**: Use `select_layouts_low_ces(max_ces=0.5)` → eliminates outliers.

### Section 5: Shot Noise Sensitivity ✅

At h=3.25, fixed layout, 10 repetitions:
- Mean energy: -10.930707 ± 0.054172
- Mean ΔE/gap: 5.027 ± 0.012
- Expected ZNE std: ±0.007

**Conclusion**: Shot noise alone (std=1.2%) is well below the 3% threshold.
The noise is dominated by gate errors (systematic, layout-dependent), not
statistical shot noise. This confirms 16384 shots is sufficient.

---

## Action Items for Hardware Deployment

### Must Fix (Blocking)

1. **Switch layout selection to `select_layouts_low_ces(max_ces=0.5)`**
   - Current: `select_layouts_by_circuit_ces` (spread-maximizing, allows CES outliers)
   - Correct: `select_layouts_low_ces` (filters out CES > 0.5)
   - Location: `scripts/run_hardware_rehearsal.py` Section 2 + hardware deployment script

2. **Re-run rehearsal after fix to validate ZNE gain returns to ~60%**

### Should Fix (Recommended)

3. **Add CES sanity check before ZNE**: if max(CES) > 5×min(CES), warn and
   optionally discard the outlier layout.

4. **Log CES values in provenance.json** for post-hoc diagnosis of hardware
   results (already in spec §13, ensure implementation matches).

### Good to Know (No action needed)

5. ΔE/gap at h=2.5 (below regime) is 9.5% — confirms the valid regime boundary.
6. Shot noise contribution is negligible (std=1.2%) compared to gate noise (~100%).
7. Phase classification works even at ΔE/gap~100% because ⟨X⟩ signal is dominant.

---

## Interpretation for Thesis

The rehearsal demonstrates:

1. **The pipeline works** (MPNN → transpile → measure → classify): all logic is correct
2. **Layout selection is the critical vulnerability**: a single bad layout can
   ruin ZNE extrapolation, reducing gain from 63% to 16%
3. **The fix is simple and known**: use `max_ces` filtering (already in the spec)
4. **Phase classification is robust**: even with poor energy estimation, the
   phase label is correct because ⟨X⟩ >> ⟨ZZ⟩ at h≥3.0

This validates the overall strategy while identifying the one parameter
(layout selection threshold) that must be correct for hardware success.


---

## Second Run: After `select_layouts_low_ces` Fix

### Results

| Section | Status | Observation |
|---------|--------|-------------|
| 2. Noisy Pipeline | ❌ FAIL | CES values too uniform (0.148–0.172), R²=0.04 |
| 4. Layout Stability | ❌ FAIL | Same issue: no ZNE leverage |

### Diagnosis: ZNE Requires CES Spread

With `select_layouts_low_ces`, all layouts have similarly low CES (~0.15).
This eliminates outliers but removes the CES spread needed for linear
extrapolation. The fit has R²≈0.04 because all 3 points are at the same
x-value (CES) — there's no slope to measure.

This reveals a fundamental tension:
- **High CES spread** → good extrapolation leverage, but outlier layouts break linearity
- **Low uniform CES** → all layouts in perturbative regime, but no extrapolation possible

### Root Cause: heavy_hex + p=1 Has Uniform Noise

For N=10 p=1 on heavy_hex:
- Only 18 CZ gates (very shallow)
- Heavy-hex has regular connectivity → layouts map to similar qubit regions
- All good layouts have CES ≈ 0.15 (there aren't enough BAD-but-not-catastrophic layouts)

This is different from chain_1d where qubit quality varies more, providing
natural CES spread in the [0.1, 0.4] range.

### Implication for Hardware Deployment

**ZNE via inhomogeneous CES may not be the right strategy for heavy_hex.**

Alternative approaches for hardware:
1. **Gate-folding ZNE**: Multiply CX gates by factors [1, 3, 5] instead of using
   different layouts. This creates artificial noise amplification for extrapolation.
   Use IBM's built-in `options.resilience.zne_mitigation = True`.

2. **Layout averaging** (no extrapolation): Average energies across 3 low-CES
   layouts without extrapolation. This reduces shot-noise variance by √3 without
   requiring CES spread.

3. **Mixed strategy**: Use 2 low-CES layouts + gate folding at 3× on one of them.

### Updated Recommendation

For IBM Heron heavy_hex N=10 p=1:
- **Primary**: Enable IBM's built-in ZNE (`options.resilience.zne_mitigation = True`)
  which uses gate folding (noise factors [1, 3, 5]). This works regardless of
  layout CES uniformity.
- **Fallback**: If gate-folding R² < 0.90, switch to PEA amplifier
  (`options.resilience.zne.amplifier = "pea"`). PEA learns the actual noise model
  and amplifies probabilistically (~50% extra QPU overhead).
  Implemented locally via `run_pea_zne()` in `noisy_utils.py`.
  CLI: `--zne-amplifier pea`.
- **Combine with**: 3 low-CES layouts for statistical averaging.
- **Monitor**: If R² < 0.8 on hardware, fall back to simple averaging.

This supersedes the HARDWARE_DEPLOYMENT_SPEC §5 Layer 4 recommendation.
The spec should be updated before hardware execution.
