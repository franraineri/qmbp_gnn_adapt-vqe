---
inclusion: manual
---

# Error Budget Decomposition

## Principle

Total VQE error at each h-point decomposes into independent additive components:

```
ΔE_total = ΔE_ansatz + ΔE_optimizer + ΔE_simulator + ΔE_noise
```

Where:
- **ΔE_ansatz** = E(θ_opt_exact) − E₀ — expressibility limit (HVA p≤2 cannot reach GS)
- **ΔE_optimizer** = E(θ_found) − E(θ_opt_exact) — convergence gap (restarts, maxiter)
- **ΔE_simulator** = E_simulator(θ) − E_true(θ) — numerical precision (~0 for MPS/SV)
- **ΔE_noise** = E_noisy(θ) − E_noiseless(θ) — only on hardware/noisy runs

## Noiseless Regime (our primary focus)

For noiseless runs: `ΔE_simulator ≈ 0` and `ΔE_noise = 0`, so:

```
ΔE_total ≈ ΔE_ansatz + ΔE_optimizer
```

### Measuring ΔE_ansatz (ansatz expressibility limit)

To isolate ΔE_ansatz: run VQE with `maxiter=∞` and many restarts (or use
DMRG restricted to the ansatz subspace). This is expensive but can be done
once per (topology, N, p, h) configuration.

Practical proxy: if the best VQE run across all seeds/restarts has identical
energy to the next-best, ansatz saturation is likely (ΔE_optimizer → 0).

### Measuring ΔE_optimizer (convergence gap)

```
ΔE_optimizer = E(θ_best_run) − E(θ_global_optimum_in_ansatz)
```

Indicators of non-zero ΔE_optimizer:
- High variance across restarts at same h-point
- Energy improves with more restarts
- Ascending pass finds lower energies than descending

### Measuring ΔE_simulator (chi truncation for MPS)

Verified via `--verify-chi`: compare E(χ) vs E(2χ).
- If |ΔE| < 1e-10: chi is sufficient, ΔE_simulator = 0
- If |ΔE| > 1e-10: chi truncation contributes measurable error

## When Each Dominates

| Regime | Dominant Component | Diagnostic |
|--------|-------------------|------------|
| h near h_c (critical) | ΔE_ansatz | p=2 insufficient, ceiling at ΔE/gap ≈ 2-5% |
| h >> h_c (paramagnetic) | Neither (converges easily) | ΔE_total ≈ 0 |
| N > 16, 2D topology | ΔE_simulator | Chi-convergence test needed |
| Low maxiter / 1 restart | ΔE_optimizer | Increase restarts, check variance |
| Hardware / FakeTorino | ΔE_noise | ZNE mitigation, SNR analysis |

## Recording in Results

The `variational_violation` field (per-point) flags cases where E_vqe < E_exact − ε,
which indicates either:
1. Numerical noise in the simulator (ΔE_simulator < 0, rare)
2. Incorrect E_exact reference (DMRG not converged)

The `simulation_diagnostics` block in each result documents:
- `backend_type`: which simulator generated the data
- `method_exact`: whether the method is numerically exact
- `chi_max`: MPS bond dimension (if applicable)
- `chi_sufficiency_warning`: 2D topology alert

## Usage in Thesis

When presenting results, decompose observed ΔE/gap:
1. Report total ΔE/gap (what we measure)
2. Attribute to ansatz limit (known from p=2 + h_c proximity)
3. Attribute remainder to optimizer (reducible with more compute)
4. Confirm simulator precision is negligible (chi-convergence or exact SV)
