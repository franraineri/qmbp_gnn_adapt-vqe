# PoC Results — Numerical Baselines

> Use these numbers to evaluate whether a change improved or regressed the pipeline.
> Updated after 28 benchmark runs across 7 configurations.

## Best Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| VQE restarts | 5 | 7 gives no improvement; 3 is worse |
| VQE maxiter | 1000 | 1500 gives no improvement |
| MPNN hidden | 64 | 128 overfits; 32 underfits |
| MPNN layers | 3 | 4 overfits; 2 underfits |
| MPNN epochs | 6000 | 8000 is wasteful; 4000 is borderline |
| MPNN lr | 1e-3 | 5e-4 causes ΔE/gap instability; 3e-3 causes failures |
| MPNN patience | 150 | 80 is too aggressive |
| Fid threshold | 0.93 | 0.90 adds noisy data; 0.95 removes too much |

## Expected Checklist by Test Point

| h_test | Checklist | ΔE/gap | ⟨X⟩ | ⟨ZZ⟩ | ΔE | Fidelity | ADAPT |
|--------|-----------|--------|-----|------|-----|----------|-------|
| 1.25 | 2–3/6 | ✅ 3.5% | ⚠️ ~1e-2 | ❌ ~2.5e-2 | ❌ ~3.5e-2 | ❌ 0.991 | ✅ |
| 1.4 | 4–5/6 | ✅ 1.9% | ✅ 5e-3 | ❌ ~1.5e-2 | ❌ ~1.7e-2 | ✅ 0.995 | ✅ |
| 1.5 | **5/6** | ✅ 1.4% | ✅ 2.6e-3 | ✅ 5e-3 | ❌ ~1.4e-2 | ✅ 0.997 | ✅ |

## What Each Metric Means

1. **ΔE/gap < 5%** (primary) — Does the pipeline resolve the physics? Always passes.
2. **⟨X⟩, ⟨ZZ⟩ errors < 1e-2** — Can we characterize the phase from observables? Passes at h≥1.4.
3. **ΔE < 1e-2** (aspirational) — Absolute energy accuracy. Never passes — bounded by HVA p=2 ceiling.
4. **Fidelity ≥ 99.5%** — State overlap with exact ground state. Passes at h≥1.4.
5. **ADAPT iterations ≤ 2** — Circuit depth compliance. Always passes.

## V4 Baseline (for comparison)

| Metric | V4/Py3.9 (h=1.25) | V4/Py3.12 (h=1.25) |
|--------|-------------------|---------------------|
| Checklist | 4/6 | 2/6 |
| ΔE/gap | 4.37% | ~4.3% |
| Note | Seed-sensitive — 4/6 was lucky | 2/6 is the reproducible baseline |

## Hyperparameter Sensitivity (from 28 runs)

| Change | Effect |
|--------|--------|
| 3→5 VQE restarts | ⟨X⟩ error drops 30% (highest impact) |
| 5→7 VQE restarts | No improvement |
| 4000→6000 MPNN epochs | Marginal improvement |
| 6000→8000 MPNN epochs | No improvement |
| hidden 64→128 | Overfits (worse) |
| hidden 64→32 | Underfits (worse) |
| lr 1e-3→5e-4 | ΔE/gap becomes unstable |
| lr 1e-3→3e-3 | ΔE/gap exceeds 5% on some seeds |
| fid 0.93→0.90 | More noisy data, worse results |
| fid 0.93→0.95 | Too few training points, worse results |
| maxiter 1000→1500 | No improvement |
