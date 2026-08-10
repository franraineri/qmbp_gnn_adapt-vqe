---
inclusion: fileMatch
fileMatchPattern: "project_health/**,scripts/maintenance/**,data/**,**/inspect_*,**/coverage*,**/quality_predictor*,**/model_quality_dashboard*"
---

# Data Quality & Model Analysis Patterns

## Model Quality Dashboard (`data/model_quality_dashboard.json`)

Auto-generated after every runner completes. Single file, always overwritten.
Source: `qmbp_simulation.analysis.metrics.generate_model_quality_dashboard()`

### When to use it
- **QualityPredictor preflight**: read h_frontier as fresh signal (Priority 1 over heuristic)
- **Coverage gap detection**: configs with pass_rate < 50% need iterative improvement
- **Preset generation**: only generate presets for configs with dashboard pass_rate > 50%
- **Budget estimation**: use pass_rate to predict how many points will need VQE refinement

### Key fields per config entry
```python
{
    "topology": str,         # e.g. "ladder"
    "n_qubits": int,         # System size
    "p_layers": int,         # HVA depth
    "n_params": int,         # Total variational parameters (distinguishes bond-resolved from global)
    "n_edges": int,          # Topology-dependent edge count
    "model": str,            # e.g. "tfim_bond_resolved"
    "h_frontier": float,     # h where ΔE/gap crosses 5% (empirical boundary)
    "pass_rate_5pct": float, # Fraction with ΔE/gap < 5%
    "pass_rate_10pct": float,# Fraction with ΔE/gap < 10%
    "best_de_gap": float,    # Best achieved (lower = better)
    "worst_de_gap": float,   # Worst point (higher = harder)
    "n_below_frontier": int, # Points that need refinement
    "zoo_model_available": bool,  # Cross-check with model zoo
    "zoo_pass_rate": float | None, # Zoo model quality (None if unevaluated)
    # Cross-N transfer quality
    "cross_n_transfers": list[dict], # All transfer records for this target_n
    "cross_n_best_source": dict | None, # Best train_n → {train_n, pass_rate_10pct, mean_de_gap}
}
```

### Topology-level summary (`topology_summary` key)
```python
{
    "topology_name": {
        "n_values": list[int],       # All N with training data
        "n_max_viable": int | None,  # Largest N with pass_rate_10pct > 50%
        "n_configs": int,            # Number of NPZ configs
        "best_pass_rate_5pct": float,# Best pass rate across all N
        "best_n": int,               # N with best pass rate
        "cross_n_best_source_for_largest": dict | None,  # Best source for predicting largest N
    }
}
```

### Bond-resolved vs Global HVA distinction
- **Bond-resolved** (n_params > 2 * p_layers): per-edge θ_zz + per-site θ_x → many parameters
- **Global** (n_params = 2 * p_layers): single θ_zz, single θ_x → few parameters
- This matters for VQE convergence difficulty and MPNN training data requirements

### Cross-checks available
1. `zoo_pass_rate` vs `pass_rate_5pct`: divergence = model or data stale
2. `n_below_frontier` > 0: actionable — these points need VQE refinement
3. `zoo_model_available = false`: config needs model training first

## Refinement Priority Scoring

Use `compute_refinement_priority()` from `analysis/metrics.py` to decide which
h-points are worth spending VQE compute on.

```python
from qmbp_simulation.analysis.metrics import compute_refinement_priority

priority, should_skip, reason = compute_refinement_priority(
    de_gap=0.08, abs_error=0.15, gap=2.0, n_params=30,
    e_prev=-10.0, e_pred=-10.5, n_prev_attempts=1,
)
# priority ∈ [0, 1], should_skip: bool, reason: str
```

Factors: proximity to threshold, gap feasibility, n_params discount,
MPNN improvement signal, stale attempt count.

## H-Frontier Computation

```python
from qmbp_simulation.analysis.metrics import compute_h_frontier, compute_h_frontier_from_npz

# From arrays
frontier = compute_h_frontier(h_values, de_gaps, threshold=0.05)

# From NPZ file
result = compute_h_frontier_from_npz("data/multi_n_training/ladder_N10_p1.npz")
# Returns: {h_frontier, n_points, pass_rate, h_range, mean_de_gap, mean_abs_error}
```

## Coverage Gap Detection

`project_health/core/coverage.py` now detects `GapType.LOW_PASS_RATE` from the dashboard.
Configs with pass_rate < 50% and ≥ 5 data points get flagged with actionable recommendations.

## Thresholds (from `analysis/metrics.py` constants)
- `DE_GAP_THRESHOLD = 0.05` — primary pass/fail
- `MAX_ABS_ERROR = 0.10` — absolute error cap
- `MIN_FIDELITY = 0.97` — state overlap minimum
- Always import these constants; never hardcode 0.05/0.10/0.97 in new code.
