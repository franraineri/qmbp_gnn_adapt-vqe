# Analysis Rules — Energy Metrics

## Always report both ΔE/gap AND |ΔE| absolute

When analyzing VQE or MPNN deployment results:

1. **ΔE/gap** (normalized by spectral gap) — primary pass/fail criterion (<5%)
2. **|ΔE|** (absolute energy error) — physical accuracy measure
3. **|ΔE|/N** (error per site) — extensive scaling diagnostic

Flag cases where ΔE/gap < 5% but |ΔE| > 1.0 — the gap is masking a large error.

Rationale: at large h, gap ≈ 2h → ΔE/gap shrinks artificially while |ΔE| may be significant.

## Theta canonicalization is mandatory

All θ_opt data from VQE MUST be canonicalized before MPNN training:
- `from qmbp_simulation.utils import canonicalize_theta` (period=π, wraps + Z₂)
- `from qmbp_simulation.utils import filter_consistent_theta` (MAD outlier detection)
- Both are integrated in `build_graph_dataset` as mandatory steps

## Analysis script location

- Phase3 MPNN analysis: `scripts/analysis/analyze_all_phase3.py`
- Frontier/boundary mapping: `scripts/analysis/extract_frontier_data.py`
- θ variation diagnostic: `scripts/analysis/analyze_theta_variation.py`
- HVA periodicity verification: `scripts/analysis/verify_hva_periodicity.py`
