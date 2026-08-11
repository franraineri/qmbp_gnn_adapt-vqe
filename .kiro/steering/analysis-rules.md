# Analysis Rules — Energy Metrics

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
