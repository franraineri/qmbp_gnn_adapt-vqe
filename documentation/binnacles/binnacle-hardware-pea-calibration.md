# Binnacle: Hardware PEA Calibration Study

**Started**: 2026-06-14
**Status**: In Progress
**Backend**: IBM Kingston (156Q Eagle r3, heavy_hex)
**Circuit**: TFIM N=10, p=1, 9 RZZ gates, h=4.0

## Objective

Determine the minimum PEA (Probabilistic Error Amplification) noise learning
budget that achieves ΔE/gap < 5% on real quantum hardware with elevated
calibration error (mean 2Q error ~3.4%).

## Background

- PEA learns a Pauli-Lindblad noise model via randomized circuits before
  amplifying noise for ZNE extrapolation (Kim et al., Nature 618, 2023).
- IBM default budget: 32 randomizations × 128 shots = 4,096 learning shots.
- At 3.4% mean 2Q error (Kingston 2026-06-14 evening), default budget proved
  insufficient → ΔE/gap = 32.5% (target: <5%).
- Hypothesis: increasing learning budget gives PEA more data to accurately
  fit the noise model → better extrapolation → lower ΔE/gap.

## Hardware Conditions (2026-06-14, 19:00-20:00 UTC-3)

| Metric | Value |
|--------|-------|
| Mean 2Q error (chip-wide) | 3.36% |
| Mean readout error | 1.96% |
| Min T1 | 6.5 μs (isolated TLS defect) |
| P5 T1 | 125.9 μs (most qubits healthy) |
| Layout CES | 0.050 |
| Queue | 0 jobs (no wait) |

## Results

### EXP-1: IBM Default (32×128, 1 layout)

- **Config**: num_rand=32, shots/rand=128, noise_factors=IBM default, 1 layout
- **Learning budget**: 4,096 shots
- **Wall-clock**: 746.7s (~12.4 min)
- **QPU time**: 284s
- **Result**: E_ZNE = -38.641, E_exact = -40.566
- **ΔE/gap = 32.5%** ❌ FAIL
- **Analysis**: PEA learned an inaccurate noise model. Error = 1.92 units (raw
  error without ZNE would be ~9.7 units → PEA DID help, reducing error by ~80%,
  but not enough for 5% ΔE/gap target).
- **Ref**: `results/hardware/tier0_calibration/pea_32x128/`

### EXP-2: Aggressive (64×256, 3 layouts, 4 noise factors)

- **Config**: num_rand=64, shots/rand=256, noise_factors=[1, 1.5, 2, 3], 3 layouts
- **Learning budget**: 65,536 shots (16× IBM default)
- **Wall-clock**: >15 min (cancelled by user at ~16 min)
- **Result**: Jobs still "In progress" when cancelled. No energy result.
- **Analysis**: 3 layouts × 4 noise factors × 64×256 learning = too much QPU time.
  Each Batch contains 3 jobs (one per layout), each with independent PEA learning.
  Total: 3 × 65K = ~196K learning circuits. Impractical for iterative experimentation.
- **Ref**: `results/hardware/tier0_calibration/pea_64x256/`

### EXP-3: Balanced (48×192, 1 layout, 3 noise factors) — PENDING

- **Config**: num_rand=48, shots/rand=192, noise_factors=[1, 1.5, 3], 1 layout
- **Learning budget**: 9,216 shots (~2.3× IBM default)
- **Estimated wall-clock**: ~6-8 min (1 job, moderate learning)
- **Rationale**: 2.3× default should capture noise model better than 1× but
  won't timeout. Single layout avoids 3× multiplication. 3 noise factors
  (dropping factor 2) saves 25% PEA time.
- **Status**: Not yet executed.
- **Ref**: `results/hardware/tier0_calibration/pea_48x192/`

### EXP-4: IBM Default + 3 layouts (32×128, 3 layouts) — PENDING

- **Config**: num_rand=32, shots/rand=128, noise_factors=IBM default, 3 layouts
- **Learning budget**: 4,096 shots per layout (shared via Batch)
- **Estimated wall-clock**: ~8-10 min (3 jobs, default learning)
- **Rationale**: Test if layout averaging (√3 variance reduction) compensates
  for weak noise model. If yes → the issue is variance, not bias.
  If no → the issue is PEA model quality (bias), confirming need for more learning.
- **Status**: Not yet executed.
- **Ref**: `results/hardware/tier0_calibration/pea_32x128_3layout/`

## Decision Framework

After all experiments complete:

| If EXP-3 passes | If EXP-4 passes | Conclusion |
|:---:|:---:|:---|
| ✅ | — | Use 48×192 as default (balanced QPU cost vs accuracy) |
| ❌ | ✅ | Issue is variance → keep IBM default learning, use 3 layouts |
| ❌ | ❌ | Need both: 48×192 + 3 layouts (or better calibration window) |
| ✅ | ✅ | Both work → prefer 48×192 single layout (simpler, fewer jobs) |

## Deployment Impact

Once the optimal config is identified:
- Update `MitigationOptions` defaults in `src/qmbp_simulation/execution/backends.py`
- Update `build_hardware_config()` in deployment script
- All Tiers (1-3) use the validated config
- Document in `project-status.md` under Key Constraints

## References

- Kim et al., Nature 618, pp. 500-505 (2023) — PEA original paper
- QESEM, arXiv:2508.10997 — extended PEA validation
- Project finding: `documentation/analysis/11_hardware_rehearsal_findings.md`

---

## Addendum 2026-06-15: κ-Based Deployment Optimization

### Finding: κ Anti-correlates with Noise Sensitivity

From MPNN Eval Suite Section 19 (binnacle-mpnn-eval-suite.md):
- Pearson r(κ, ΔE_noise) = **-0.835** at N=6 chain_1d (|r| = 0.74-0.85 across grid)
- Low κ → h near h_c → HIGH hardware risk (small gap amplifies θ errors)
- High κ → deep paramagnetic regime → LOW risk

### κ Go/No-Go Integration (2026-06-15)

New functions added to `run_ibm_deployment.py`:

```python
kappa_per_h = compute_kappa_per_h(params_per_h, lattice)  # zero QPU cost
recommendations = kappa_go_no_go(kappa_per_h, shots_base=16384)
# Per-h: risk_level, n_layouts, shots, spsa_recommended
```

**Decision table (N=6 calibration, to be validated at N=10):**

| κ | Risk | n_layouts | Shots | SPSA |
|---|------|-----------|-------|------|
| ≥ 50 | LOW | 1 | 16K | No |
| 45-50 | MEDIUM | 3 | 16K | No |
| < 45 | HIGH | 3 | 32K | Recommended |

**QPU impact:** For TIER_1_H=[4.0, 3.75, 3.5, 3.25] at N=10, all h-points are
likely in the LOW/MEDIUM regime (far from h_c≈1.0). Expected QPU savings: ~40%
vs baseline (use 1 layout for all instead of 3).

### Pending Validation

- S19 needs to be run at N=10 heavy_hex to calibrate κ thresholds for the
  production deployment config. Current thresholds (κ<45=high) are from N=6
  chain_1d and may need adjustment.
- Ref: `binnacle-mpnn-eval-suite.md` § S19


### Addendum 2026-06-15: κ Thresholds — heavy_hex N=10 Invalidated

From S19 heavy_hex N=10 p=1 (binnacle-mpnn-eval-suite.md):
- κ range: [111, 174] — 3x higher than chain_1d [41, 53]
- Pearson r(κ, ΔE_noise) = -0.52 (|r| < 0.70 → WEAK correlation)

**Conclusion:** The κ-based go/no-go thresholds (κ<45=HIGH risk) are ONLY
valid for **chain_1d** topology. For **heavy_hex**, κ does NOT predict noise
sensitivity reliably because routing SWAPs introduce incoherent noise that is
independent of the VQE landscape curvature.

**Decision for IBM Heron deployment:**
- Do NOT use κ thresholds for heavy_hex go/no-go
- Use V2 hardware rehearsal pass/fail (sections 1-9) as the deployment gate
- κ is logged in per_h_results for reference but should not drive shots/layouts
  decisions for heavy_hex

**κ is still computed** in Tier 0, 1, 2 to track landscape quality — but
`kappa_go_no_go()` recommendations should be treated as informational-only for
heavy_hex until topology-specific thresholds are calibrated.
