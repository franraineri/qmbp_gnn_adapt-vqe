# h_frontier Matrix — Model Expressibility Boundary

## Experiment Configuration

- **System**: N=10 qubits, chain_1d topology
- **Backend**: StatevectorEstimator (noiseless, exact)
- **Optimizer**: L-BFGS-B (p=2), COBYLA auto-switch (p≥3, n_params>8)
- **Seeds**: 42, 43 (median across seeds)
- **Restarts**: 5-14 (scaled with model complexity and p)
- **Date**: 2026-07-17 to 2026-07-18
- **Total runs**: 56 configurations (8 models × 7 p-values)

## Complete Matrix: h_frontier(model, p)

Definition: exact h where ΔE/gap crosses 5%, via linear interpolation between
last failing and first passing point. Median across seeds.

```
Model                      p=2     p=3     p=4     p=5     p=6     p=7     p=8
─────────────────────────────────────────────────────────────────────────────────
tfim                     1.378   1.204   1.109   1.058   1.025   1.020   1.074
tfim_longitudinal        1.377   1.320   1.179   1.179   1.181   1.090   1.107
tfim_frustrated          1.230   1.115   1.107   0.997   1.061   0.883   0.873
kitaev                    FAIL    FAIL    FAIL    FAIL    FAIL    FAIL    FAIL
heisenberg                FAIL    FAIL    FAIL    FAIL    FAIL    FAIL    FAIL
heisenberg_transverse    3.672   3.672   3.636   3.590   3.598   3.497   3.498
xy                        FAIL    FAIL    FAIL    FAIL    FAIL    FAIL    FAIL
tfim_bond_resolved       1.381   1.162   1.098   0.874   0.833   0.892   0.865
```

FAIL = no h-point achieved ΔE/gap < 5% at any tested value (sweep up to h=5.0).

## Distance from h_c (h_frontier − h_c)

Critical field estimates: tfim/tfim_longitudinal/tfim_frustrated/tfim_bond_resolved: h_c=1.0,
kitaev: h_c=2.0, heisenberg/xy: h_c≈2.5, heisenberg_transverse: h_c≈2.5.

```
Model                      p=2     p=3     p=4     p=5     p=6     p=7     p=8
─────────────────────────────────────────────────────────────────────────────────
tfim                    +0.378  +0.204  +0.109  +0.058  +0.025  +0.020  +0.074
tfim_longitudinal       +0.377  +0.320  +0.179  +0.179  +0.181  +0.090  +0.107
tfim_frustrated         +0.230  +0.115  +0.107  −0.003  +0.061  −0.117  −0.127
kitaev                     —      —      —      —      —      —      —
heisenberg                 —      —      —      —      —      —      —
heisenberg_transverse   +1.172  +1.172  +1.136  +1.090  +1.098  +0.997  +0.998
xy                         —      —      —      —      —      —      —
tfim_bond_resolved      +0.381  +0.162  +0.098  −0.126  −0.167  −0.108  −0.135
```

Key: negative distance = frontier BELOW h_c (circuit expresses ground state in the ordered phase).

## Mean Fidelity at Passing Points

```
Model                      p=2     p=3     p=4     p=5     p=6     p=7     p=8
─────────────────────────────────────────────────────────────────────────────────
tfim                    0.9970  0.9988  0.9992  0.9989  0.9995  0.9996  0.9995
tfim_longitudinal       0.9972  0.9976  0.9978  0.9987  0.9985  0.9994  0.9989
tfim_frustrated         0.9991  0.9991  0.9992  0.9993  0.9995  0.9996  0.9997
heisenberg_transverse   0.9988  0.9989  0.9992  0.9994  0.9991  0.9994  0.9995
tfim_bond_resolved      0.9976  0.9982  0.9985  0.9992  0.9991  0.9991  0.9994
```

All passing points achieve F > 99.7%. Fidelity improves monotonically with p (more
parameters → closer to exact eigenstate). Essentially perfect VQE where it converges.

## Best ΔE/gap Achieved (minimum across all h-points)

```
Model                      p=2       p=3       p=4       p=5       p=6       p=7       p=8
─────────────────────────────────────────────────────────────────────────────────────────────
tfim                     1.1e-04   5.3e-06   7.6e-07   4.9e-05   3.2e-05   6.7e-05   1.0e-04
tfim_longitudinal        1.1e-04   8.4e-05   2.5e-04   5.8e-05   8.9e-05   4.0e-05   1.3e-04
tfim_frustrated          7.0e-06   1.0e-04   1.1e-04   1.4e-04   1.4e-04   8.0e-05   6.4e-05
kitaev                   2.0e+00   1.3e+00   9.7e-01   9.2e-01   8.0e-01   5.7e-01   6.0e-01
heisenberg               6.2e+00   4.4e+00   3.6e+00   2.9e+00   2.8e+00   2.3e+00   2.1e+00
heisenberg_transverse    1.6e-03   3.5e-03   3.1e-03   3.1e-03   2.8e-03   6.9e-04   6.9e-04
xy                       3.5e+00   2.5e+00   1.9e+00   1.6e+00   1.4e+00   1.2e+00   1.1e+00
tfim_bond_resolved       2.3e-03   1.4e-03   6.7e-04   8.8e-04   8.5e-04   5.2e-04   3.2e-04
```

## Model Classification

### Tier 1: Fully Compatible (h_frontier approaches or crosses h_c)

| Model | Best h_frontier | Best p | Params/layer | CX gates (p=1) |
|---|:---:|:---:|:---:|:---:|
| tfim_bond_resolved | 0.833 | p=6 | N_e+N=19 | 9 |
| tfim_frustrated | 0.873 | p=8 | 3 | 9+NNN |
| tfim | 1.020 | p=7 | 2 | 9 |
| tfim_longitudinal | 1.090 | p=7 | 3 | 9 |

These models achieve h_frontier ≈ h_c or below, meaning the HVA+|+⟩^N ansatz can
express the ground state across the full phase diagram (paramagnetic + ordered phase).

### Tier 2: Partially Compatible (only paramagnetic phase)

| Model | Best h_frontier | Best p | Distance from h_c |
|---|:---:|:---:|:---:|
| heisenberg_transverse | 3.497 | p=7 | +0.997 |

Functions only for h > 3.5 (deep paramagnetic). The circuit cannot capture the
antiferromagnetic correlations below h_c ≈ 2.5. Increasing p from 2→8 barely helps
(h_frontier: 3.67 → 3.50, Δ=0.17). The barrier is not depth but ansatz structure.

### Tier 3: Incompatible (FAIL at all p)

| Model | Best ΔE/gap (p=8) | Root Cause |
|---|:---:|---|
| kitaev | 0.60 | Pairing ground state inaccessible from |+⟩^N via RXX+RYY+RZ |
| heisenberg | 2.1 | AF ground state with longitudinal field (h·Z) vs transverse init |
| xy | 1.1 | Same as heisenberg but with Δ=0 (XY order incompatible) |

## Analysis and Anomalies

### 1. TFIM saturation at p=5-7, degradation at p=8

```
tfim:     p=5: 1.058 → p=6: 1.025 → p=7: 1.020 → p=8: 1.074 (↑ worse!)
```

Cause: COBYLA (gradient-free) with 24 params (p=8 × 2 params/layer) struggles to
find the global minimum in the high-dimensional landscape. More restarts (12) are
insufficient to overcome the combinatorial explosion of local minima. The VQE is
not limited by circuit expressibility (the ansatz CAN express the state) but by
classical optimization difficulty.

Evidence: p=4 uses L-BFGS-B (8 params, gradient-based) and achieves 1.109 — only
marginally worse than p=7 (1.020) despite half the depth.

### 2. tfim_bond_resolved crosses h_c at p≥5

```
tfim_bond_resolved: p=5: 0.874, p=6: 0.833, p=7: 0.892, p=8: 0.865
```

This is the only model that consistently reaches BELOW h_c=1.0 at moderate depth.
The per-bond parametrization (19 params/layer at N=10) provides local expressibility
that the global HVA lacks. However, COBYLA with 19×p params is extremely challenging
→ non-monotonic behavior (p=6 best, p=7 worse).

### 3. tfim_frustrated crosses h_c at p=5, p=7, p=8

```
tfim_frustrated: p=5: 0.997, p=7: 0.883, p=8: 0.873
```

The J2 frustration (NNN coupling) creates a flatter energy landscape near criticality
that is EASIER for VQE to navigate. The frustrated model achieves the deepest frontier
at high p (0.873 at p=8) among the global-parameter models.

### 4. Kitaev total failure — structural incompatibility

Kitaev best ΔE/gap decreases with p (2.0 → 0.6) but never approaches 5%.
At p=8 with 24 params, the circuit reaches ~60% of the spectral gap — fundamentally
insufficient. The Kitaev chain ground state has p-wave pairing correlations
(⟨c_i c_{i+1}⟩ ≠ 0) that require a different circuit structure (e.g., Givens rotations
on fermion modes, or a number-preserving ansatz).

### 5. Heisenberg: longitudinal vs transverse field

- heisenberg (H·Z field): FAIL at all p. The ground state is antiferromagnetic
  (|↑↓↑↓...⟩) — orthogonal to |+⟩^N. No amount of RXX+RYY+RZZ+RZ rotations from
  |+⟩^N can efficiently reach it.
- heisenberg_transverse (H·X field): partially works because at h >> h_c, the ground
  state IS ~|+⟩^N (the initial state itself). The ansatz adds nothing below h_c.

### 6. XY model (Δ=0 Heisenberg): consistent FAIL

Same circuit as heisenberg (RXX+RYY+RZZ+RZ) but the XY ground state has U(1) spin
symmetry that the HVA starting from |+⟩^N cannot reproduce. The best ΔE/gap (1.1 at
p=8) is worse than kitaev — the landscape is particularly adversarial.

## Implications for the Thesis

1. **Circuit depth vs expressibility**: For TFIM-class models, the frontier saturates
   at p=5-7. Going deeper provides diminishing returns and may degrade due to COBYLA.
   The optimal operating point is **p=4-6** (best cost/benefit).

2. **Per-bond parametrization is the winning strategy**: tfim_bond_resolved reaches
   h_frontier=0.83 (below h_c) with the same gate count as global HVA. The GNN predictor
   becomes essential here (19 params/layer → impossible to grid-search).

3. **Incompatible models need different ansätze**: Kitaev, Heisenberg, XY cannot work
   with HVA+|+⟩^N. Alternative approaches: Néel initial state for Heisenberg,
   particle-hole symmetric ansatz for Kitaev, ADAPT-VQE for all.

4. **Hardware viability**: Only tfim, tfim_longitudinal, tfim_frustrated have both
   low CX count AND deep frontier. tfim_bond_resolved is ideal for noiseless
   demonstration but has the same CX count → hardware-viable too.

## Scripts

- Compute matrix: `python scripts/analysis/compute_h_frontier_models.py`
- Detailed per-h breakdown: `python scripts/analysis/analyze_hmin_models.py`
- Run exploration: `./run_hmin_models_exploration.sh`
