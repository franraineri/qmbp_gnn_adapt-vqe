# Cross-Topology Unified Report

**Generated**: 2026-08-11 15:25 UTC
**Criterion**: `pass_rate_dual` (ΔE/gap < 5% AND |ΔE| < 0.10)
**Model**: TFIM bond-resolved, HVA p=1

> All quality metrics use the dual criterion exclusively.
> `summary.pass_rate` in runner JSONs = execution health (sections completed), NOT quality.

---

<!-- AUTO-GENERATED-BEGIN:scorecard -->
## 1. Scorecard

| Topology | N_max (dual≥70%) | Best pass_dual | Zoo model | Training pts | Data quality | Extrapolation | h_frontier |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| chain_1d | 20 | 100% | ⚠️ 48% | 168 | ✅ 83% | N=100 ❌0% | 1.56 |
| heavy_hex | 16 | 95% | ✅ 88% | 217 | ⚠️ 76% | N=30 ❌0% | 0.96 |
| ladder | — | 67% | ❌ 0% | 372 | ❌ 25% | N=30 ❌0% | 1.85 |
| square | 8 | 85% | ❌ 0% | 261 | ⚠️ 58% | N=30 ❌0% | 1.84 |
| triangular | 4 | 72% | ❌ 0% | 275 | ❌ 38% | N=12 ❌9% | 1.98 |
<!-- AUTO-GENERATED-END:scorecard -->

<!-- AUTO-GENERATED-BEGIN:scaling -->
## 2. Scaling: pass_rate_dual per (Topology, N)

| Topology | N=4 | N=6 | N=8 | N=10 | N=12 | N=14 | N=16 | N=20 |
|----------|---:|---:|---:|---:|---:|---:|---:|---:|
| chain_1d | — | **98%** ✅ | **100%** ✅ | **89%** ✅ | **92%** ✅ | — | — | **100%** ✅ |
| heavy_hex | 67% ⚠️ | 71% ⚠️ | — | **91%** ✅ | 75% ⚠️ | — | **95%** ✅ | — |
| ladder | 23% ❌ | 67% ⚠️ | 47% ❌ | 10% ❌ | 41% ❌ | 0% | 8% ❌ | 0% |
| square | **85%** ✅ | 73% ⚠️ | **84%** ✅ | 68% ⚠️ | 48% ❌ | 38% ❌ | 9% ❌ | 0% |
| triangular | 72% ⚠️ | 58% ⚠️ | 29% ❌ | 22% ❌ | 20% ❌ | 0% | 0% | — |

Legend: ✅ ≥80% | ⚠️ 50-79% | ❌ <50% | — no data
<!-- AUTO-GENERATED-END:scaling -->

<!-- AUTO-GENERATED-BEGIN:masking -->
## 3. Gap Masking Severity

Configs where single-criterion inflates by >10pp vs dual:

| Topology | N | pass@5% | pass@dual | Inflation |
|----------|---|---------|-----------|-----------|
| square | 20 | 100% | 0% | +100% |
| ladder | 16 | 100% | 8% | +92% |
| ladder | 14 | 66% | 0% | +66% |
| square | 14 | 100% | 38% | +62% |
| ladder | 10 | 71% | 10% | +61% |
| ladder | 12 | 100% | 41% | +59% |
| triangular | 8 | 86% | 29% | +57% |
| triangular | 10 | 70% | 22% | +48% |
| chain_1d | 15 | 100% | 59% | +41% |
| triangular | 12 | 60% | 20% | +40% |
| ladder | 8 | 86% | 47% | +39% |
| square | 16 | 44% | 9% | +34% |
| square | 12 | 76% | 48% | +28% |
| ladder | 6 | 84% | 67% | +18% |
| heavy_hex | 6 | 88% | 71% | +17% |
| ... | | | | *(5 more)* |
<!-- AUTO-GENERATED-END:masking -->

<!-- AUTO-GENERATED-BEGIN:extrapolation -->
## 4. Large-N Extrapolation (Zero-Shot)

MPNN predictions at N >> training data. Only dual criterion reported.

| Topology | N | h-range | Pts | pass_dual | |ΔE|/N | ΔE/gap (mean) |
|----------|---|---------|-----|-----------|--------|---------------|
| chain_1d | 10 | [3.5, 5.0] | 8 | 6/8 ✅ | 8.11e-03 | 0.0117 |
| chain_1d | 16 | [3.5, 5.0] | 6 | 2/6 ⚠️ | 8.95e-03 | 0.0206 |
| chain_1d | 20 | [3.5, 5.0] | 6 | 2/6 ⚠️ | 9.10e-03 | 0.0263 |
| chain_1d | 30 | [2.5, 5.5] | 18 | 1/18 ❌ | 9.40e-03 | 0.0482 |
| chain_1d | 40 | [2.5, 5.5] | 12 | 0/12 ❌ | 8.55e-03 | 0.0617 |
| chain_1d | 60 | [2.5, 5.5] | 12 | 0/12 ❌ | 1.10e-02 | 0.1114 |
| chain_1d | 100 | [3.5, 5.5] | 12 | 0/12 ❌ | 9.87e-03 | 0.1347 |
| heavy_hex | 10 | [3.5, 5.0] | 8 | 8/8 ✅ | 3.08e-03 | 0.0046 |
| heavy_hex | 16 | [3.0, 5.0] | 15 | 9/15 ⚠️ | 6.31e-03 | 0.0173 |
| heavy_hex | 20 | [2.5, 5.0] | 23 | 0/23 ❌ | 1.33e-02 | 0.4054 |
| heavy_hex | 30 | [2.5, 4.5] | 9 | 0/9 ❌ | 5.59e-02 | 2.8969 |
| ladder | 10 | [3.5, 5.0] | 6 | 5/6 ✅ | 6.37e-03 | 0.0130 |
| ladder | 16 | [3.5, 5.5] | 12 | 7/12 ⚠️ | 6.11e-03 | 0.0196 |
| ladder | 20 | [3.5, 5.5] | 12 | 1/12 ❌ | 5.56e-03 | 0.2638 |
| ladder | 30 | [2.5, 5.0] | 3 | 0/3 ❌ | 2.11e-02 | 3.0236 |
| square | 10 | [3.5, 5.0] | 6 | 0/6 ❌ | 2.69e-02 | 0.0509 |
| square | 16 | [3.0, 5.0] | 15 | 0/15 ❌ | 2.09e-02 | 0.0784 |
| square | 20 | [3.0, 5.0] | 15 | 0/15 ❌ | 2.19e-02 | 1.0411 |
| square | 30 | [3.5, 4.5] | 3 | 0/3 ❌ | 1.04e-02 | 1.4906 |
| triangular | 10 | [3.5, 5.5] | 11 | 5/11 ⚠️ | 1.27e-02 | 0.0366 |
| triangular | 12 | [3.5, 5.5] | 11 | 1/11 ❌ | 1.92e-02 | 0.0840 |
<!-- AUTO-GENERATED-END:extrapolation -->

<!-- AUTO-GENERATED-BEGIN:data_quality -->
## 5. Training Data Quality

| Topology | NPZ files | Total pts | Verified | Approx | Unverified | Quality |
|----------|-----------|-----------|----------|--------|------------|---------|
| chain_1d | 6 | 168 | 141 (83%) | 27 | 0 | ✅ |
| heavy_hex | 5 | 217 | 166 (76%) | 48 | 3 | ⚠️ |
| ladder | 10 | 372 | 95 (25%) | 238 | 39 | ❌ |
| square | 8 | 261 | 153 (58%) | 82 | 26 | ⚠️ |
| triangular | 9 | 275 | 107 (38%) | 98 | 70 | ❌ |
<!-- AUTO-GENERATED-END:data_quality -->

<!-- AUTO-GENERATED-BEGIN:actions -->
## 6. Recommended Actions

| Priority | Topology | Action |
|:---:|----------|--------|
| 1 | ladder | 🔴 Re-train UnifiedMPNN (current pass_dual=0%) |
| 1 | square | 🔴 Re-train UnifiedMPNN (current pass_dual=0%) |
| 1 | triangular | 🔴 Re-train UnifiedMPNN (current pass_dual=0%) |
| 2 | ladder | ⚠️ Run iterative-improve to increase verified% (currently 25%) |
| 2 | triangular | ⚠️ Run iterative-improve to increase verified% (currently 38%) |
| 4 | heavy_hex | 🟢 Expand to N=20: good candidate (pass_dual=95%) |
| 4 | square | 🟢 Expand to N=12: good candidate (pass_dual=85%) |
<!-- AUTO-GENERATED-END:actions -->
