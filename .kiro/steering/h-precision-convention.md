inclusion: always

# h-Value Precision Convention (MANDATORY)

## Rule

All h-values used as **cache keys, dictionary keys, or file identifiers** MUST use
exactly **2 decimal places** (`:.2f`). Never 3, 4, 5, or 6 decimals.

```python
# ✅ CORRECT
key = f"{topo}|{n}|{model}|{h:.2f}"     # "chain_1d|10|tfim|2.50"
h_rounded = round(h, 2)                   # 2.50, not 2.500000

# ❌ WRONG — NEVER DO THIS
key = f"{topo}|{n}|{model}|{h:.6f}"     # "chain_1d|10|tfim|2.500000"
key = f"{topo}|{n}|{model}|{h:.4f}"     # "chain_1d|10|tfim|2.5000"
key = f"...{float(h)}"                    # Unpredictable precision
```

## Why

1. GT cache uses `:.2f` in `_make_key()` → keys look like `chain_1d|10|tfim_bond_resolved|2.50`
2. NPZ h_values are generated with `np.linspace` or `np.arange(step=0.05)` → always round at 2 decimals
3. Mismatched precision causes **phantom cache misses** and **false staleness reports**
4. Physics: h increments are never finer than 0.05 in our grid — 2 decimals is sufficient

## Where This Applies

| Module | Key format |
|--------|-----------|
| `GroundTruthCache._make_key()` | `:.2f` ✅ |
| `EvalCache.get/put_ground_truth()` | `:.2f` ✅ |
| `validate_gt_npz_coherence()` | `:.2f` ✅ |
| `training_intelligence._check_gt_coherence_for_topology()` | `:.2f` ✅ |
| Any new code that creates h-based keys | MUST use `:.2f` |

## h-Values in Data (NPZ arrays)

h-values stored in NPZ `h_values` arrays retain full float64 precision for computation.
The `:.2f` rule only applies to **key construction for lookups and matching**.

When comparing h-values numerically (not as strings), use tolerance:
```python
# ✅ CORRECT — numeric comparison with tolerance
if abs(h_npz - h_gt) < 0.005:  # half of 0.01 step
    # match

# ❌ WRONG — exact float comparison
if h_npz == h_gt:  # floating point equality is unreliable
```
