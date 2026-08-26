# Cross-Topology Unified Report

**Generated**: 2026-08-26 15:40 UTC
**Criterion**: `pass_rate_dual` (ΔE/gap < 5% AND |ΔE| < 0.10)
**Model**: TFIM bond-resolved, HVA p=1

> All quality metrics use the dual criterion exclusively.
> `summary.pass_rate` in runner JSONs = execution health (sections completed), NOT quality.

---

<!-- AUTO-GENERATED-BEGIN:scorecard -->
## 1. Scorecard

| Topology | N_max (dual≥70%) | Best pass_dual | Zoo model | Training pts | Data quality | Extrapolation | h_frontier | h≤h_c |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| chain_1d | 20 | 100% | ✅ 71% | 772 | ⚠️ 72% | — | 2.50 | — |
| heavy_hex | 40 | 100% | ❌ 10% | 1194 | ✅ 87% | — | 0.96 | ✓ |
| ladder | 6 | 73% | ⚠️ 45% | 624 | ⚠️ 41% | — | 1.85 | — |
| square | 8 | 85% | ⚠️ 33% | 507 | ⚠️ 71% | — | 1.84 | — |
| triangular | 4 | 100% | ❌ 25% | 407 | ⚠️ 57% | — | 0.50 | ✓ |
<!-- AUTO-GENERATED-END:scorecard -->

<!-- AUTO-GENERATED-BEGIN:scaling -->
## 2. Scaling: pass_rate_dual per (Topology, N)

| Topology | N=4 | N=6 | N=8 | N=10 | N=12 | N=14 | N=16 | N=20 | N=26 | N=30 | N=40 |
|----------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chain_1d | **100%** ✅ | **99%** ✅ | **96%** ✅ | **85%** ✅ | **91%** ✅ | **80%** ✅ | 73% ⚠️ | 78% ⚠️ | 67% ⚠️ | 29% ❌ | 42% ❌ |
| heavy_hex | **94%** ✅ | **100%** ✅ | 55% ⚠️ | **91%** ✅ | **88%** ✅ | 36% ❌ | 72% ⚠️ | 26% ❌ | 0% | 52% ⚠️ | **83%** ✅ |
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
| chain_1d | 16 | 100% | 73% | +27% |
| square | 16 | 42% | 15% | +27% |
| triangular | 12 | 33% | 7% | +26% |
| ladder | 8 | 79% | 53% | +26% |
| ladder | 26 | 38% | 12% | +25% |
| chain_1d | 60 | 45% | 23% | +23% |
| ... | | | | *(13 more)* |
<!-- AUTO-GENERATED-END:masking -->

<!-- AUTO-GENERATED-BEGIN:extrapolation -->
## 4. Large-N Extrapolation (Zero-Shot)

MPNN predictions at N >> training data. Only dual criterion reported.

*No extrapolation data available yet.*
<!-- AUTO-GENERATED-END:extrapolation -->

<!-- AUTO-GENERATED-BEGIN:data_quality -->
## 5. Training Data Quality

| Topology | NPZ files | Total pts | Verified | Approx | Unverified | Quality |
|----------|-----------|-----------|----------|--------|------------|---------|
| chain_1d | 13 | 772 | 560 (72%) | 144 | 68 | ⚠️ |
| heavy_hex | 23 | 1194 | 1050 (87%) | 82 | 62 | ✅ |
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
| 3 | chain_1d | ℹ️ Run large-N extrapolation to validate scaling |
| 3 | heavy_hex | ℹ️ Run large-N extrapolation to validate scaling |
| 3 | ladder | ℹ️ Run large-N extrapolation to validate scaling |
| 3 | square | ℹ️ Run large-N extrapolation to validate scaling |
| 3 | triangular | ℹ️ Run large-N extrapolation to validate scaling |
| 4 | square | 🟢 Expand to N=12: good candidate (pass_dual=85%) |
| 4 | triangular | 🟢 Expand to N=8: good candidate (pass_dual=100%) |
<!-- AUTO-GENERATED-END:actions -->
