# h_frontier Matrix — HVA Expressibility Boundary

## Complete Matrix: h_frontier(N, p) for TFIM chain_1d

Definition: the exact h-value where ΔE/gap crosses 5%, via linear interpolation
between last passing and first failing point. Median across seeds.

Model: TFIM chain_1d | Backend: MPS deterministic χ=64 | Optimizer: L-BFGS-B adaptive restarts

```
     N    p=1     p=2     p=3     p=4
    20   2.222   1.616   1.423   1.357
    30   2.437   1.697   1.502   1.318
    40   2.612   1.798   1.613   1.417
    50   2.747   1.859   1.758   1.390
    60   2.865   1.905   1.700   1.524
    70   2.970     —       —       —
    80   3.074   1.999   1.669   1.507
   100   3.231   2.084   1.661     —
   120   3.368   2.114   1.730     —
   250   4.000     —       —       —
```

## Fits (N ≥ 20)

```
p=1: h_frontier = 2.355 + 0.00729·N   (R² = 0.912, 10 points, includes N=250)
p=2: h_frontier = 1.574 + 0.00496·N   (R² = 0.950, 8 points)
p=3: h_frontier ≈ 1.6 ± 0.1           (quasi-constant, R² = 0.46 — linear model inappropriate)
p=4: h_frontier ≈ 1.4 ± 0.1           (quasi-constant, R² = 0.73)
```

## Distance from h_c = 1.0

```
p=1: mean distance = 1.95  (grows sublinearly with N)
p=2: mean distance = 0.88  (grows slowly)
p=3: mean distance = 0.63  (quasi-constant)
p=4: mean distance = 0.42  (quasi-constant, closest to h_c)
```

## Anomalies — Status

### 1. N=70, N=250 auto h-grid artifact — RESOLVED ✅

Re-run with explicit h-min covering the frontier. Results:
- N=70: h_frontier = 2.97 (fits perfectly between N=60: 2.87 and N=80: 3.07)
- N=250: h_frontier = 4.00 (confirms sublinear growth at large N)

### 2. p=3 non-monotonic — CHARACTERIZED, NOT FIXABLE ⚠️

Re-run with 5 seeds × 12 restarts did NOT reduce the variance (std remains ~0.10-0.13).
This is intrinsic to the p=3 landscape (6 params, multiple local minima of comparable quality).

Resolution: For p≥3, report h_frontier as a RANGE [1.5, 1.8] rather than a point estimate.
The linear fit is inappropriate (R²=0.46). The frontier is genuinely quasi-constant at ~1.6.

### 3. p=1 sublinear growth at N>100

The fit R² dropped from 0.964 (N=20-120) to 0.912 (N=20-250) because N=250 (h=4.00)
is below the linear extrapolation (h=4.18). The growth saturates at large N.
A better model for p=1 may be: `h_frontier = h_∞ - A·exp(-B·N)` (saturating exponential).

## Extended Matrices (N=10, p=2-8)

### TFIM / chain_1d (N=10)

```
     N    p=2    p=3    p=4    p=5    p=6    p=7    p=8
    10   1.38   1.20   1.11   1.06   1.02   1.02   1.07
```

Converges to h_c=1.0 at p≥5.

### TFIM Longitudinal / multiple topologies (N=10)

```
Topology     p=2    p=3    p=4    p=5    p=6    p=7    p=8
chain_1d    1.38   1.32   1.18   1.17   1.16   1.09   1.11
heavy_hex   1.51   1.36   1.31   1.36   1.33   1.12   1.27
ladder      2.57   1.97   1.84   1.87   1.67   1.78   1.83
square      2.24   2.07   1.88   1.90   1.74   1.73   1.68
kagome      2.05   1.93   1.80   1.68   1.74   1.67   1.49
triangular  3.41    —    2.72   2.44   2.44   2.24   2.20
```

Topology ordering (easiest → hardest): chain_1d < heavy_hex < kagome < ladder ≈ square < triangular

### TFIM Frustrated / chain_1d (N=10)

```
     N    p=2    p=3    p=4    p=5    p=6    p=7    p=8
    10   1.23   1.11   1.11   1.00   0.88   0.88   0.87
```

Frustration LOWERS the boundary below h_c at p≥5 (NNN coupling adds expressivity per layer).

### Heisenberg Transverse / chain_1d (N=10)

```
     N    p=2    p=3    p=4    p=5    p=6    p=7    p=8
    10   3.67   3.67   3.64   3.59   3.60   3.50   3.50
```

h_frontier ≈ 3.5 CONSTANT regardless of p. HVA cannot express Heisenberg ground state
at any depth p≤8. Confirms expressibility limit (needs p∝N, cite Wiersema2020).

## Scripts

- Matrix (chain_1d, p=1-4, N=20-250): `python scripts/analysis/compute_h_frontier.py`
- All topologies/models: `python scripts/analysis/compute_h_frontier_all.py`
- Gap check: `python scripts/analysis/check_matrix_gaps.py`
- Fill gaps batch: `./run_fill_matrix_gaps.sh`
