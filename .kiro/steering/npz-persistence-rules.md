---
inclusion: fileMatch
fileMatchPattern: "**/result_io*,**/multi_n_aggregator*,**/dataset_io*,**/run_accelerated*,**/run_bond_resolved*,**/unified_graph*,**/mpnn.py,**/test_theta_npz*,**/test_multi_n_aggregator*,**/test_iterative*"
---

# NPZ Persistence Rules (dtype safety + anti-regression)

## Critical: dtype=object Prevention

ALL theta data loaded from NPZ files MUST be explicitly cast to `float64` before:
- Calling `np.isfinite()` (fails with `ufunc 'isfinite' not supported`)
- Passing to `torch.tensor()` (fails with `can't convert np.ndarray of type numpy.object_`)
- Using in numpy arithmetic (`+`, `-`, `/`, `np.abs`, etc.)

### Canonical Loading Pattern
```python
# ✅ CORRECT — explicit dtype cast
raw = np.load(npz_path, allow_pickle=True)
h_values = np.asarray(raw["h_values"], dtype=np.float64)
e_exact = np.asarray(raw["e_exact"], dtype=np.float64)
# Per-row theta (may be dtype=object in legacy files)
theta_i = np.asarray(raw["theta_opt"][i], dtype=np.float64)

# ❌ WRONG — raw access without cast
theta_i = raw["theta_opt"][i]  # May be object dtype!
np.isfinite(theta_i)  # CRASHES if object
torch.tensor(theta_i)  # CRASHES if object
```

### Canonical Save Pattern
```python
# ✅ CORRECT — use upsert_theta_npz for ALL theta NPZ writes
from qmbp_simulation.framework.result_io import upsert_theta_npz
n_upd, n_add = upsert_theta_npz(
    npz_path,
    h_new=h_arr,
    theta_new=theta_arr,  # Must be float64, shape (n, n_params)
    e_vqe_new=e_arr,
    e_exact_new=e_exact_arr,
    gaps_new=gaps_arr,
    method_new=["vqe_refined"],  # Distinguishable source label
)

# ❌ WRONG — raw np.savez (no anti-regression, no validation, no atomic write)
np.savez(npz_path, theta_opt=theta, ...)
```

## Anti-Regression Rule

`upsert_theta_npz` ONLY updates an existing h-point if new energy is **strictly lower**.
This prevents overwriting good VQE-converged data with worse MPNN predictions.

## Method Labels (distinguishable data sources)

Every persisted theta MUST have a method label:
- `"vqe_full"` — full VQE optimization from scratch
- `"vqe_refined"` — VQE warm-start refinement (iterative improvement)
- `"mpnn_pred"` — direct MPNN prediction that passed ΔE/gap < 5%
- `"mpnn_direct"` — MPNN prediction (not yet validated)
- `"unknown"` — legacy data without provenance

## Immediate Persistence (crash safety)

Every computed improvement MUST be persisted immediately:
```python
# ✅ CORRECT — persist per-point as soon as computed
for h in h_values:
    theta_opt, e_vqe = run_vqe(h)
    upsert_theta_npz(npz_path, ...)  # Immediate

# ❌ WRONG — accumulate, persist at end (lost on crash)
results = []
for h in h_values:
    results.append(run_vqe(h))
np.savez(npz_path, ...)  # Only at end
```

## load_theta_npz (canonical loader)

Use `load_theta_npz()` from `result_io` instead of raw `np.load()`:
- Handles both legacy `dtype=object` and modern `float64`
- Filters NaN/Inf entries automatically
- Returns consistent dict with keys: `h_values, theta_opt, e_vqe, e_exact, gaps, de_gaps, method`

## MultiNAggregator Integration

The `MultiNAggregator.scan()` method loads all NPZ files for a topology and:
1. Converts ALL arrays to `float64` at load time
2. Computes `de_gaps` on-the-fly if missing from NPZ
3. The `build_combined_dataset()` method passes `float64` theta to torch graph builders

Never access `agg._data_by_n` theta without ensuring `float64` conversion already happened.
