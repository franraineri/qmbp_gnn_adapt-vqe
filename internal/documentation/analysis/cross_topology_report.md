# Cross-Topology Unified Report

**Generated**: 2026-08-29 00:08 UTC
**Criterion**: `pass_rate_dual` (ΔE/gap < 5% AND |ΔE| < 0.10)
**Model**: TFIM bond-resolved, HVA p=1

> All quality metrics use the dual criterion exclusively.
> `summary.pass_rate` in runner JSONs = execution health (sections completed), NOT quality.

---

<!-- AUTO-GENERATED-BEGIN:scorecard -->
## 1. Scorecard

| Topology | N_max (dual≥70%) | Best pass_dual | Zoo model | Training pts | Data quality | Extrapolation | h_frontier | h≤h_c |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| chain_1d | 15 | 100% | ✅ 71% | 1050 | ⚠️ 79% | N=200 ❌0% | 1.28 | ✓ |
| heavy_hex | 40 | 100% | ❌ 10% | 1224 | ✅ 88% | N=60 ❌17% | 0.96 | ✓ |
| ladder | 6 | 73% | ⚠️ 45% | 624 | ⚠️ 41% | N=40 ❌0% | 1.85 | — |
| square | 8 | 85% | ⚠️ 33% | 507 | ⚠️ 71% | N=30 ❌0% | 1.84 | — |
| triangular | 4 | 100% | ❌ 25% | 407 | ⚠️ 57% | N=24 ❌0% | 0.50 | ✓ |
<!-- AUTO-GENERATED-END:scorecard -->

<!-- AUTO-GENERATED-BEGIN:scaling -->
## 2. Scaling: pass_rate_dual per (Topology, N)

| Topology | N=4 | N=6 | N=8 | N=10 | N=12 | N=14 | N=16 | N=20 | N=26 | N=30 | N=40 |
|----------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chain_1d | 49% ❌ | **84%** ✅ | 75% ⚠️ | 66% ⚠️ | 58% ⚠️ | 14% ❌ | 39% ❌ | 64% ⚠️ | 67% ⚠️ | 29% ❌ | 42% ❌ |
| heavy_hex | **94%** ✅ | **100%** ✅ | 55% ⚠️ | **90%** ✅ | **88%** ✅ | 36% ❌ | 72% ⚠️ | 26% ❌ | 0% | 52% ⚠️ | **83%** ✅ |
| ladder | 73% ⚠️ | 72% ⚠️ | 53% ⚠️ | 37% ❌ | 48% ❌ | 27% ❌ | 25% ❌ | 23% ❌ | 12% ❌ | 20% ❌ | 0% |
| square | **85%** ✅ | **85%** ✅ | **81%** ✅ | 63% ⚠️ | 47% ❌ | 54% ⚠️ | 15% ❌ | 0% | — | — | — |
| triangular | 72% ⚠️ | 60% ⚠️ | 9% ❌ | 17% ❌ | 7% ❌ | — | — | — | — | — | — |

Legend: ✅ ≥80% | ⚠️ 50-79% | ❌ <50% | — no data
<!-- AUTO-GENERATED-END:scaling -->

<!-- AUTO-GENERATED-BEGIN:masking -->
## 3. Gap Masking Severity

Configs where single-criterion inflates by >10pp vs dual:

| Topology | N | pass@5% | pass@dual | Inflation |
|----------|---|---------|-----------|-----------|
| heavy_hex | 26 | 100% | 0% | +100% |
| ladder | 40 | 50% | 0% | +50% |
| chain_1d | 30 | 75% | 29% | +46% |
| ladder | 12 | 94% | 48% | +46% |
| ladder | 16 | 66% | 25% | +42% |
| ladder | 10 | 77% | 37% | +40% |
| chain_1d | 26 | 100% | 67% | +33% |
| ladder | 20 | 54% | 23% | +31% |
| chain_1d | 40 | 71% | 42% | +29% |
| square | 16 | 42% | 15% | +27% |
| triangular | 12 | 33% | 7% | +26% |
| ladder | 8 | 79% | 53% | +26% |
| ladder | 26 | 38% | 12% | +25% |
| chain_1d | 60 | 45% | 23% | +23% |
| ladder | 14 | 48% | 27% | +21% |
| ... | | | | *(11 more)* |
<!-- AUTO-GENERATED-END:masking -->

<!-- AUTO-GENERATED-BEGIN:extrapolation -->
## 4. Large-N Extrapolation (Zero-Shot)

MPNN predictions at N >> training data. Only dual criterion reported.

| Topology | N | h-range | Pts | pass_dual | |ΔE| | ΔE/gap (mean) |
|----------|---|---------|-----|-----------|------|---------------|
| chain_1d | 10 | [0.5, 3.0] | 41 | 4/41 ❌ | 0.694 | 74.0910 |
| chain_1d | 12 | [2.5, 3.0] | 2 | 0/2 ❌ | 0.170 | 0.0504 |
| chain_1d | 16 | [2.5, 5.0] | 10 | 2/10 ❌ | 0.335 | 0.0575 |
| chain_1d | 20 | [0.5, 5.0] | 36 | 6/36 ❌ | 1.186 | 2.1030 |
| chain_1d | 30 | [2.5, 5.5] | 28 | 8/28 ❌ | 0.213 | 0.0351 |
| chain_1d | 40 | [2.5, 5.5] | 24 | 10/24 ⚠️ | 0.236 | 0.0405 |
| chain_1d | 60 | [2.5, 5.5] | 22 | 5/22 ❌ | 0.447 | 0.0725 |
| chain_1d | 80 | [2.5, 5.0] | 8 | 0/8 ❌ | 0.708 | 0.1658 |
| chain_1d | 100 | [2.5, 5.5] | 19 | 2/19 ❌ | 0.798 | 0.1369 |
| chain_1d | 150 | [4.0, 5.0] | 3 | 0/3 ❌ | 5.376 | 0.7841 |
| chain_1d | 200 | [4.0, 5.0] | 3 | 0/3 ❌ | 7.180 | 1.0472 |
| heavy_hex | 8 | [2.5, 5.0] | 47 | 25/47 ⚠️ | 0.187 | 0.0302 |
| heavy_hex | 10 | [2.5, 5.0] | 57 | 47/57 ✅ | 0.079 | 0.0143 |
| heavy_hex | 12 | [2.5, 5.0] | 25 | 0/25 ❌ | 0.608 | 0.1251 |
| heavy_hex | 14 | [2.5, 5.0] | 54 | 12/54 ❌ | 8.150 | 1.9993 |
| heavy_hex | 16 | [2.5, 5.0] | 47 | 6/47 ❌ | 0.225 | 0.0480 |
| heavy_hex | 18 | [2.5, 5.0] | 43 | 3/43 ❌ | 1.131 | 0.2323 |
| heavy_hex | 20 | [2.0, 5.0] | 96 | 31/96 ⚠️ | 0.292 | 0.1771 |
| heavy_hex | 21 | [2.5, 5.0] | 14 | 0/14 ❌ | 0.932 | 0.3410 |
| heavy_hex | 22 | [2.5, 5.0] | 31 | 6/31 ❌ | 0.560 | 0.2026 |
| heavy_hex | 24 | [2.5, 5.0] | 47 | 11/47 ❌ | 0.803 | 0.2864 |
| heavy_hex | 26 | [2.5, 5.0] | 27 | 11/27 ⚠️ | 0.689 | 0.2507 |
| heavy_hex | 30 | [2.0, 5.0] | 51 | 33/51 ⚠️ | 0.338 | 0.3369 |
| heavy_hex | 32 | [2.5, 5.0] | 10 | 6/10 ⚠️ | 0.648 | 0.3840 |
| heavy_hex | 40 | [2.5, 5.0] | 33 | 13/33 ⚠️ | 0.581 | 0.2009 |
| heavy_hex | 50 | [2.5, 5.0] | 6 | 2/6 ⚠️ | 0.919 | 0.2856 |
| heavy_hex | 60 | [2.5, 5.0] | 6 | 1/6 ❌ | 1.753 | 0.4976 |
| ladder | 16 | [2.5, 5.0] | 6 | 0/6 ❌ | 0.224 | 0.0784 |
| ladder | 20 | [2.5, 5.5] | 24 | 6/24 ❌ | 0.145 | 0.2592 |
| ladder | 26 | [2.5, 5.0] | 14 | 2/14 ❌ | 0.233 | 0.5173 |
| ladder | 30 | [2.5, 5.5] | 14 | 3/14 ❌ | 0.223 | 0.5130 |
| ladder | 40 | [2.5, 5.0] | 6 | 0/6 ❌ | 0.395 | 1.5579 |
| square | 16 | [2.5, 5.0] | 26 | 4/26 ❌ | 0.264 | 0.0813 |
| square | 20 | [2.5, 5.0] | 26 | 1/26 ❌ | 0.351 | 0.7914 |
| square | 30 | [2.5, 5.0] | 13 | 0/13 ❌ | 0.890 | 2.4057 |
| triangular | 12 | [2.5, 5.0] | 10 | 0/10 ❌ | 0.879 | 1.7850 |
| triangular | 16 | [2.5, 5.0] | 10 | 0/10 ❌ | 2.603 | 28.7030 |
| triangular | 24 | [2.5, 5.0] | 10 | 0/10 ❌ | 6.145 | 23.4736 |
<!-- AUTO-GENERATED-END:extrapolation -->

<!-- AUTO-GENERATED-BEGIN:data_quality -->
## 5. Training Data Quality

| Topology | NPZ files | Total pts | Verified | Approx | Unverified | Quality |
|----------|-----------|-----------|----------|--------|------------|---------|
| chain_1d | 14 | 1050 | 839 (79%) | 144 | 67 | ⚠️ |
| heavy_hex | 23 | 1224 | 1080 (88%) | 82 | 62 | ✅ |
| ladder | 11 | 624 | 261 (41%) | 259 | 104 | ⚠️ |
| square | 8 | 507 | 364 (71%) | 96 | 47 | ⚠️ |
| triangular | 6 | 407 | 235 (57%) | 96 | 76 | ⚠️ |
<!-- AUTO-GENERATED-END:data_quality -->

<!-- AUTO-GENERATED-BEGIN:failure_modes -->
## 6. Failure Mode Classification

| Topology | Mode | Evidence | Implication |
|----------|------|----------|-------------|
| chain_1d | ✅ healthy | best_dual=100% | Pipeline works correctly |
| heavy_hex | ✅ healthy | best_dual=100% | Pipeline works correctly |
| ladder | 🔵 gap_masking | 10 configs, severity=32% | Model works; |ΔE|>0.10 from N×ε (expected) |
| square | 🟡 partial | best_dual=85%, 5 masked | Partially working; focus on viable h-range |
| triangular | ✅ healthy | best_dual=100% | Pipeline works correctly |

*Run `--deep` analyzer for full Tests A-L breakdown.*
<!-- AUTO-GENERATED-END:failure_modes -->

<!-- AUTO-GENERATED-BEGIN:actions -->
## 7. Recommended Actions

| Priority | Topology | Action |
|:---:|----------|--------|
| 1 | heavy_hex | 🔴 Re-train UnifiedMPNN (current pass_dual=10%) |
| 1 | triangular | 🔴 Re-train UnifiedMPNN (current pass_dual=25%) |
| 4 | chain_1d | 🟢 Expand to N=19: good candidate (pass_dual=100%) |
| 4 | square | 🟢 Expand to N=12: good candidate (pass_dual=85%) |
| 4 | triangular | 🟢 Expand to N=8: good candidate (pass_dual=100%) |
<!-- AUTO-GENERATED-END:actions -->
