# Cross-Topology Unified Report

**Generated**: 2026-08-19 17:27 UTC
**Criterion**: `pass_rate_dual` (ΔE/gap < 5% AND |ΔE| < 0.10)
**Model**: TFIM bond-resolved, HVA p=1

> All quality metrics use the dual criterion exclusively.
> `summary.pass_rate` in runner JSONs = execution health (sections completed), NOT quality.

---

<!-- AUTO-GENERATED-BEGIN:scorecard -->
## 1. Scorecard

| Topology | N_max (dual≥70%) | Best pass_dual | Zoo model | Training pts | Data quality | Extrapolation | h_frontier |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| chain_1d | 20 | 99% | ⚠️ 36% | 693 | ⚠️ 78% | N=200 ❌0% | 1.56 |
| heavy_hex | 16 | 100% | ⚠️ 64% | 536 | ⚠️ 78% | N=40 ✅ | 0.96 |
| ladder | 6 | 73% | ❌ 18% | 591 | ⚠️ 43% | N=40 ❌0% | 1.85 |
| square | 8 | 85% | ❌ 22% | 481 | ⚠️ 74% | N=30 ❌0% | 1.84 |
| triangular | 4 | 100% | ❌ 25% | 407 | ⚠️ 57% | N=24 ❌0% | 0.50 |
<!-- AUTO-GENERATED-END:scorecard -->

<!-- AUTO-GENERATED-BEGIN:scaling -->
## 2. Scaling: pass_rate_dual per (Topology, N)

| Topology | N=4 | N=6 | N=8 | N=10 | N=12 | N=14 | N=16 | N=20 | N=26 | N=30 |
|----------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chain_1d | — | **99%** ✅ | **96%** ✅ | **85%** ✅ | **91%** ✅ | — | 76% ⚠️ | 80% ⚠️ | 67% ⚠️ | 29% ❌ |
| heavy_hex | **94%** ✅ | **100%** ✅ | — | **90%** ✅ | **87%** ✅ | — | **85%** ✅ | 60% ⚠️ | 0% | 33% ❌ |
| ladder | 73% ⚠️ | 72% ⚠️ | 51% ⚠️ | 33% ❌ | 48% ❌ | 27% ❌ | 25% ❌ | 23% ❌ | — | — |
| square | **85%** ✅ | **85%** ✅ | **81%** ✅ | 62% ⚠️ | 46% ❌ | 54% ⚠️ | 0% | — | — | — |
| triangular | 72% ⚠️ | 60% ⚠️ | 9% ❌ | 17% ❌ | 7% ❌ | — | — | — | — | — |

Legend: ✅ ≥80% | ⚠️ 50-79% | ❌ <50% | — no data
<!-- AUTO-GENERATED-END:scaling -->

<!-- AUTO-GENERATED-BEGIN:masking -->
## 3. Gap Masking Severity

Configs where single-criterion inflates by >10pp vs dual:

| Topology | N | pass@5% | pass@dual | Inflation |
|----------|---|---------|-----------|-----------|
| heavy_hex | 26 | 100% | 0% | +100% |
| square | 16 | 100% | 0% | +100% |
| chain_1d | 30 | 100% | 29% | +71% |
| heavy_hex | 30 | 100% | 33% | +67% |
| ladder | 12 | 94% | 48% | +46% |
| ladder | 10 | 76% | 33% | +43% |
| ladder | 16 | 66% | 25% | +42% |
| heavy_hex | 20 | 95% | 60% | +35% |
| chain_1d | 26 | 100% | 67% | +33% |
| ladder | 20 | 54% | 23% | +31% |
| ladder | 8 | 79% | 51% | +28% |
| triangular | 12 | 33% | 7% | +26% |
| chain_1d | 16 | 100% | 76% | +24% |
| square | 12 | 68% | 46% | +22% |
| ladder | 14 | 48% | 27% | +21% |
| ... | | | | *(6 more)* |
<!-- AUTO-GENERATED-END:masking -->

<!-- AUTO-GENERATED-BEGIN:extrapolation -->
## 4. Large-N Extrapolation (Zero-Shot)

MPNN predictions at N >> training data. Only dual criterion reported.

| Topology | N | h-range | Pts | pass_dual | |ΔE|/N | ΔE/gap (mean) |
|----------|---|---------|-----|-----------|--------|---------------|
| chain_1d | 16 | [3.5, 5.0] | 6 | 2/6 ⚠️ | 8.95e-03 | 0.0206 |
| chain_1d | 20 | [2.5, 5.0] | 7 | 6/7 ✅ | 1.81e-02 | 0.0678 |
| chain_1d | 30 | [2.5, 5.5] | 28 | 8/28 ❌ | 7.49e-03 | 0.0380 |
| chain_1d | 40 | [2.5, 5.5] | 24 | 10/24 ⚠️ | 5.90e-03 | 0.0408 |
| chain_1d | 60 | [2.5, 5.5] | 22 | 5/22 ❌ | 7.45e-03 | 0.0725 |
| chain_1d | 100 | [2.5, 5.5] | 19 | 2/19 ❌ | 7.98e-03 | 0.1369 |
| chain_1d | 150 | [4.0, 5.0] | 3 | 0/3 ❌ | 3.58e-02 | 0.7841 |
| chain_1d | 200 | [4.0, 5.0] | 3 | 0/3 ❌ | 3.59e-02 | 1.0472 |
| heavy_hex | 20 | [2.0, 5.0] | 37 | 19/37 ⚠️ | 1.12e-02 | 0.3341 |
| heavy_hex | 30 | [2.0, 5.0] | 27 | 14/27 ⚠️ | 1.84e-02 | 0.9713 |
| heavy_hex | 40 | [2.5, 4.5] | 6 | 5/6 ✅ | 1.79e-03 | 0.0390 |
| ladder | 20 | [2.5, 5.5] | 24 | 6/24 ❌ | 7.10e-03 | 0.2582 |
| ladder | 26 | [2.5, 5.0] | 14 | 2/14 ❌ | 8.95e-03 | 0.5173 |
| ladder | 30 | [2.5, 5.5] | 14 | 3/14 ❌ | 7.44e-03 | 0.5130 |
| ladder | 40 | [2.5, 5.0] | 6 | 0/6 ❌ | 9.88e-03 | 1.5579 |
| square | 16 | [2.5, 5.0] | 26 | 4/26 ❌ | 1.64e-02 | 0.0809 |
| square | 20 | [2.5, 5.0] | 26 | 1/26 ❌ | 1.74e-02 | 0.7903 |
| square | 30 | [2.5, 4.5] | 12 | 0/12 ❌ | 2.89e-02 | 2.5784 |
| triangular | 12 | [2.5, 5.0] | 10 | 0/10 ❌ | 7.33e-02 | 1.7850 |
| triangular | 16 | [2.5, 5.0] | 10 | 0/10 ❌ | 1.63e-01 | 28.7030 |
| triangular | 24 | [2.5, 5.0] | 10 | 0/10 ❌ | 2.56e-01 | 23.4736 |
<!-- AUTO-GENERATED-END:extrapolation -->

<!-- AUTO-GENERATED-BEGIN:data_quality -->
## 5. Training Data Quality

| Topology | NPZ files | Total pts | Verified | Approx | Unverified | Quality |
|----------|-----------|-----------|----------|--------|------------|---------|
| chain_1d | 9 | 693 | 543 (78%) | 105 | 45 | ⚠️ |
| heavy_hex | 8 | 536 | 420 (78%) | 70 | 46 | ⚠️ |
| ladder | 8 | 591 | 257 (43%) | 250 | 84 | ⚠️ |
| square | 7 | 481 | 357 (74%) | 82 | 42 | ⚠️ |
| triangular | 6 | 407 | 235 (57%) | 96 | 76 | ⚠️ |
<!-- AUTO-GENERATED-END:data_quality -->

<!-- AUTO-GENERATED-BEGIN:failure_modes -->
## 6. Failure Mode Classification

| Topology | Mode | Evidence | Implication |
|----------|------|----------|-------------|
| chain_1d | ✅ healthy | best_dual=99% | Pipeline works correctly |
| heavy_hex | ✅ healthy | best_dual=100% | Pipeline works correctly |
| ladder | 🔵 gap_masking | 7 configs, severity=32% | Model works; |ΔE|>0.10 from N×ε (expected) |
| square | 🔵 gap_masking | 4 configs, severity=39% | Model works; |ΔE|>0.10 from N×ε (expected) |
| triangular | ✅ healthy | best_dual=100% | Pipeline works correctly |

*Run `--deep` analyzer for full Tests A-L breakdown.*
<!-- AUTO-GENERATED-END:failure_modes -->

<!-- AUTO-GENERATED-BEGIN:actions -->
## 7. Recommended Actions

| Priority | Topology | Action |
|:---:|----------|--------|
| 1 | ladder | 🔴 Re-train UnifiedMPNN (current pass_dual=18%) |
| 1 | square | 🔴 Re-train UnifiedMPNN (current pass_dual=22%) |
| 1 | triangular | 🔴 Re-train UnifiedMPNN (current pass_dual=25%) |
| 4 | heavy_hex | 🟢 Expand to N=20: good candidate (pass_dual=100%) |
| 4 | square | 🟢 Expand to N=12: good candidate (pass_dual=85%) |
| 4 | triangular | 🟢 Expand to N=8: good candidate (pass_dual=100%) |
<!-- AUTO-GENERATED-END:actions -->
