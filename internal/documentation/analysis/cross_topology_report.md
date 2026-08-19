# Cross-Topology Unified Report

**Generated**: 2026-08-18 19:19 UTC
**Criterion**: `pass_rate_dual` (ΔE/gap < 5% AND |ΔE| < 0.10)
**Model**: TFIM bond-resolved, HVA p=1

> All quality metrics use the dual criterion exclusively.
> `summary.pass_rate` in runner JSONs = execution health (sections completed), NOT quality.

---

<!-- AUTO-GENERATED-BEGIN:scorecard -->
## 1. Scorecard

| Topology | N_max (dual≥70%) | Best pass_dual | Zoo model | Training pts | Data quality | Extrapolation | h_frontier |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| chain_1d | 20 | 99% | ⚠️ 69% | 606 | ✅ 89% | N=200 ❌0% | 1.56 |
| heavy_hex | 16 | 100% | ⚠️ 62% | 441 | ✅ 90% | N=40 ✅ | 0.96 |
| ladder | 6 | 73% | ❌ 18% | 519 | ⚠️ 46% | N=40 ❌0% | 1.85 |
| square | 8 | 85% | ⚠️ 33% | 435 | ⚠️ 77% | N=30 ❌0% | 1.84 |
| triangular | 4 | 100% | ❌ 25% | 383 | ⚠️ 56% | N=24 ❌0% | 0.50 |
<!-- AUTO-GENERATED-END:scorecard -->

<!-- AUTO-GENERATED-BEGIN:scaling -->
## 2. Scaling: pass_rate_dual per (Topology, N)

| Topology | N=4 | N=6 | N=8 | N=10 | N=12 | N=14 | N=16 | N=20 |
|----------|---:|---:|---:|---:|---:|---:|---:|---:|
| chain_1d | — | **99%** ✅ | **96%** ✅ | **92%** ✅ | **91%** ✅ | — | — | 78% ⚠️ |
| heavy_hex | **94%** ✅ | **100%** ✅ | — | **91%** ✅ | **87%** ✅ | — | **87%** ✅ | — |
| ladder | 73% ⚠️ | 72% ⚠️ | 50% ⚠️ | 24% ❌ | 48% ❌ | 27% ❌ | 28% ❌ | 36% ❌ |
| square | **85%** ✅ | **85%** ✅ | **81%** ✅ | 61% ⚠️ | 63% ⚠️ | 54% ⚠️ | — | — |
| triangular | 72% ⚠️ | 58% ⚠️ | 12% ❌ | 17% ❌ | 7% ❌ | — | — | — |

Legend: ✅ ≥80% | ⚠️ 50-79% | ❌ <50% | — no data
<!-- AUTO-GENERATED-END:scaling -->

<!-- AUTO-GENERATED-BEGIN:masking -->
## 3. Gap Masking Severity

Configs where single-criterion inflates by >10pp vs dual:

| Topology | N | pass@5% | pass@dual | Inflation |
|----------|---|---------|-----------|-----------|
| ladder | 10 | 73% | 24% | +50% |
| ladder | 12 | 94% | 48% | +46% |
| ladder | 16 | 72% | 28% | +43% |
| ladder | 8 | 79% | 50% | +29% |
| triangular | 8 | 40% | 12% | +28% |
| triangular | 12 | 33% | 7% | +26% |
| square | 12 | 87% | 63% | +23% |
| ladder | 14 | 48% | 27% | +21% |
| square | 14 | 73% | 54% | +19% |
| triangular | 10 | 35% | 17% | +19% |
| ladder | 6 | 87% | 72% | +14% |
| square | 10 | 72% | 61% | +12% |
| triangular | 6 | 69% | 58% | +11% |
<!-- AUTO-GENERATED-END:masking -->

<!-- AUTO-GENERATED-BEGIN:extrapolation -->
## 4. Large-N Extrapolation (Zero-Shot)

MPNN predictions at N >> training data. Only dual criterion reported.

| Topology | N | h-range | Pts | pass_dual | |ΔE|/N | ΔE/gap (mean) |
|----------|---|---------|-----|-----------|--------|---------------|
| chain_1d | 16 | [3.5, 5.0] | 6 | 2/6 ⚠️ | 8.95e-03 | 0.0206 |
| chain_1d | 20 | [2.5, 5.0] | 6 | 6/6 ✅ | 1.79e-03 | 0.0090 |
| chain_1d | 30 | [2.5, 5.5] | 20 | 6/20 ⚠️ | 7.71e-03 | 0.0358 |
| chain_1d | 40 | [2.5, 5.5] | 16 | 4/16 ❌ | 7.35e-03 | 0.0480 |
| chain_1d | 60 | [2.5, 5.5] | 16 | 3/16 ❌ | 9.23e-03 | 0.0862 |
| chain_1d | 100 | [3.5, 5.5] | 12 | 2/12 ❌ | 8.15e-03 | 0.1111 |
| chain_1d | 150 | [4.0, 5.0] | 3 | 0/3 ❌ | 3.58e-02 | 0.7841 |
| chain_1d | 200 | [4.0, 5.0] | 3 | 0/3 ❌ | 3.59e-02 | 1.0472 |
| heavy_hex | 20 | [2.5, 5.0] | 25 | 9/25 ⚠️ | 8.89e-03 | 0.1946 |
| heavy_hex | 30 | [2.5, 4.5] | 14 | 5/14 ⚠️ | 2.19e-02 | 1.2121 |
| heavy_hex | 40 | [2.5, 4.5] | 6 | 5/6 ✅ | 1.79e-03 | 0.0390 |
| ladder | 20 | [2.5, 5.5] | 16 | 3/16 ❌ | 6.65e-03 | 0.2666 |
| ladder | 26 | [2.5, 5.0] | 6 | 1/6 ❌ | 9.81e-03 | 0.6711 |
| ladder | 30 | [2.5, 5.5] | 14 | 3/14 ❌ | 7.44e-03 | 0.5130 |
| ladder | 40 | [2.5, 5.0] | 6 | 0/6 ❌ | 9.88e-03 | 1.5579 |
| square | 16 | [2.5, 5.0] | 18 | 1/18 ❌ | 1.89e-02 | 0.0904 |
| square | 20 | [2.5, 5.0] | 18 | 0/18 ❌ | 2.01e-02 | 0.9344 |
| square | 30 | [2.5, 4.5] | 12 | 0/12 ❌ | 2.89e-02 | 2.5784 |
| triangular | 12 | [2.5, 5.0] | 10 | 0/10 ❌ | 7.33e-02 | 1.7850 |
| triangular | 16 | [2.5, 5.0] | 10 | 0/10 ❌ | 1.63e-01 | 28.7030 |
| triangular | 24 | [2.5, 5.0] | 10 | 0/10 ❌ | 2.56e-01 | 23.4736 |
<!-- AUTO-GENERATED-END:extrapolation -->

<!-- AUTO-GENERATED-BEGIN:data_quality -->
## 5. Training Data Quality

| Topology | NPZ files | Total pts | Verified | Approx | Unverified | Quality |
|----------|-----------|-----------|----------|--------|------------|---------|
| chain_1d | 6 | 606 | 544 (89%) | 45 | 17 | ✅ |
| heavy_hex | 5 | 441 | 400 (90%) | 21 | 20 | ✅ |
| ladder | 8 | 519 | 239 (46%) | 224 | 56 | ⚠️ |
| square | 6 | 435 | 337 (77%) | 70 | 28 | ⚠️ |
| triangular | 6 | 383 | 217 (56%) | 91 | 75 | ⚠️ |
<!-- AUTO-GENERATED-END:data_quality -->

<!-- AUTO-GENERATED-BEGIN:failure_modes -->
## 6. Failure Mode Classification

| Topology | Mode | Evidence | Implication |
|----------|------|----------|-------------|
| chain_1d | ✅ healthy | best_dual=99% | Pipeline works correctly |
| heavy_hex | ✅ healthy | best_dual=100% | Pipeline works correctly |
| ladder | 🔵 gap_masking | 6 configs, severity=34% | Model works; |ΔE|>0.10 from N×ε (expected) |
| square | 🟡 partial | best_dual=85%, 3 masked | Partially working; focus on viable h-range |
| triangular | ✅ healthy | best_dual=100% | Pipeline works correctly |

*Run `--deep` analyzer for full Tests A-L breakdown.*
<!-- AUTO-GENERATED-END:failure_modes -->

<!-- AUTO-GENERATED-BEGIN:actions -->
## 7. Recommended Actions

| Priority | Topology | Action |
|:---:|----------|--------|
| 1 | ladder | 🔴 Re-train UnifiedMPNN (current pass_dual=18%) |
| 1 | triangular | 🔴 Re-train UnifiedMPNN (current pass_dual=25%) |
| 4 | heavy_hex | 🟢 Expand to N=20: good candidate (pass_dual=100%) |
| 4 | square | 🟢 Expand to N=12: good candidate (pass_dual=85%) |
| 4 | triangular | 🟢 Expand to N=8: good candidate (pass_dual=100%) |
<!-- AUTO-GENERATED-END:actions -->
