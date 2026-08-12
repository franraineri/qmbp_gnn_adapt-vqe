# Analysis Rules — Energy Metrics

## Unified Quality Metric: `pass_rate_dual` (CANONICAL)

**All pass_rate values in the project (zoo, runners, reports) use the dual criterion:**

```
pass = (ΔE/gap < 0.05) AND (|ΔE| < 0.10)
```

This is the ONLY quality metric. Constants live in `analysis/metrics.py`:
- `DE_GAP_THRESHOLD = 0.05`
- `MAX_ABS_ERROR = 0.10`

**Design choice**: `|ΔE| < 0.10` is an absolute cap, NOT per-site. At large N,
`|ΔE| = N × (error per site)` grows linearly → large-N points intentionally
"fail" dual criterion. This is honest: the pipeline's precision ceiling IS
`|ΔE| < 0.10` regardless of N. Per-site error (`|ΔE|/N`) is reported as a
separate informational metric for extensive scaling analysis.

**Deprecated**: `pass_rate_5pct` (ΔE/gap only) — never use this alone for quality claims.
**Different concept**: `summary.pass_rate` in runner results = sections completed / total sections (execution health, NOT quality).

## Sources of Truth for Thesis (data hierarchy)

When writing the thesis, use these sources in this priority order:

| Priority | Source | What it provides | Metric version |
|:---:|--------|-----------------|----------------|
| 1 | `data/multi_n_training/*.npz` | Raw per-h data: e_vqe, e_exact, gaps, quality_tier | Always retrocomputable |
| 2 | `data/model_quality_dashboard.json` | Aggregated: pass_rate_dual per (topo, N) | Dual (regenerable) |
| 3 | `data/large_n_extrapolation/*.npz` | Extrapolation raw data | Always retrocomputable |
| 4 | `results/experiments/exp_large_n_extrap/` | Speedup comparisons, per-N summaries | Has pass_rate_dual |
| 5 | `data/model_zoo/manifest.json` | Zoo pass_rate (= dual since Aug 11 2026) | Dual after 2026-08-11 |
| 6 | `internal/documentation/analysis/cross_topology_report.md` | Unified report | Dual (auto-generated) |

**NOT a source of truth for quality**:
- `.result_index.json` — mixed metric versions, 60% aggregate-only
- `results/experiments/exp_accel_cross_n/` — no per-h energy data in JSONs
- Any run without `summary.metric_version == "dual_v1"` — legacy single criterion

**Rule**: All runs from 2026-08-11 onward carry `"metric_version": "dual_v1"` in their
summary. When filtering historical results, check this field. Runs without it used
the single criterion and their `pass_rate` is NOT comparable with current results.

## Always report both ΔE/gap AND |ΔE| absolute

When analyzing VQE or MPNN deployment results:

1. **ΔE/gap** (normalized by spectral gap) — primary pass/fail criterion (<5%)
2. **|ΔE|** (absolute energy error) — physical accuracy measure
3. **|ΔE|/N** (error per site) — extensive scaling diagnostic

Flag cases where ΔE/gap < 5% but |ΔE| > 1.0 — the gap is masking a large error.

Rationale: at large h, gap ≈ 2h → ΔE/gap shrinks artificially while |ΔE| may be significant.

## Gap Masking Problem (CRITICAL — discovered 2026-08-09)

**The dual criterion is MANDATORY for cross-N and scaling claims:**

A point only passes if BOTH hold:
- ΔE/gap < 5%  **AND**
- |ΔE| < 0.10

Without the dual criterion, large gaps at high h artificially inflate pass rates:
- ladder N=14: 80% → 0% under dual criterion (ALL gap-masked)
- ladder N=16: 100% → 0% (ALL gap-masked)
- square N=20: 100% → 0% (ALL gap-masked)
- heavy_hex N=16: 97% → 94% (genuine — minimal masking)

**Honest N_max_viable (dual criterion):**
| Topology | N_max_viable |
|----------|:---:|
| chain_1d | 20 |
| ladder | 10 (not 16!) |
| square | 10 (not 12!) |
| heavy_hex | 16 ✅ (genuine) |
| triangular | 4 (not 10!) |

Heavy_hex is the ONLY topology where cross-N genuinely works at N=16.
This is a publishable methodological finding about reporting metrics in VQE.

Use constants from `analysis/metrics.py`: `DE_GAP_THRESHOLD`, `MAX_ABS_ERROR`.

## Theta canonicalization is mandatory

All θ_opt data from VQE MUST be canonicalized before MPNN training:
- `from qmbp_simulation.utils import canonicalize_theta` (period=π, wraps + Z₂)
- `from qmbp_simulation.utils import filter_consistent_theta` (MAD outlier detection)
- Both are integrated in `build_graph_dataset` as mandatory steps

## Analysis script location

- Phase3 MPNN analysis: `scripts/analysis/analyze_all_phase3.py`
- Frontier/boundary mapping: `scripts/analysis/compute_h_frontier.py`
- All-topology frontier: `scripts/analysis/compute_h_frontier_all.py`
- Coverage gaps: `scripts/analysis/check_matrix_gaps.py`
- Cross-N analyzer: `project_health/analysis/accelerated_cross_n_analyzer.py`
- Data stores inspection: `scripts/maintenance/inspect_data_stores.py`
- Cross-N coverage update: `scripts/maintenance/update_cross_n_coverage.py`
- θ variation diagnostic: `scripts/analysis/theta_derivative_analysis.py`
- HVA periodicity verification: `scripts/analysis/verify_hva_periodicity.py`
