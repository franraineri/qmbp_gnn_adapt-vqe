---
inclusion: fileMatch
fileMatchPattern: "scripts/experiment_runners/noiseless/*.py,scripts/experiment_runners/scaling/*.py,scripts/experiment_runners/bond_resolved/*.py"
---

# Noiseless & Scaling Runner Patterns (ALWAYS ENFORCE)

## VQE Training Data Generation (Phase 1+2 sections)

All sections that generate VQE training data MUST include:

1. **Backend selection**: Use `select_backend(n, for_vqe_loop=True)` to auto-select
   MPS for N>10 (StatevectorEstimator is O(2^N), too slow for iterative VQE at N≥12).

2. **Bidirectional sweep**: After descending warm-start pass, run ascending pass
   keeping better energies. Typically improves 30-50% of points near criticality.

3. **Per-point ΔE/gap**: Compute and store `de_gaps` array for quality gating in Phase 3.

4. **Variational principle check**: Call `self.check_variational_principle(vqe_energies, exact_energies)`.
   Log warning if violations > 0 (indicates numerical noise or DMRG issues).

5. **Theta smoothness**: Call `self.compute_theta_smoothness(theta_array)`.
   Log the value. If >1.0, theta alignment may be needed before MPNN training.

6. **Theta canonicalization**: For sign-symmetric models (TFIM), canonicalize θ
   to enforce consistent sign convention (e.g., θ_x > 0).

7. **Energy variance** (automatic): `VQEOptimizer.optimize()` computes `energy_variance`
   per-point. Store it in the per-point result dict. Use for:
   - Detecting fragile passes (ΔE/gap<5% but Var>0.5 → hardware-vulnerable)
   - Identifying the ansatz expressibility boundary (Var spikes at h→h_c)
   - Go/No-Go criterion for hardware deployment (Var(θ_pred) < 0.2 required)

## MPNN Training (Phase 3 sections)

All sections that train MPNN MUST include:

1. **Quality gate**: Use `build_graph_dataset(..., de_gaps=de_gaps, de_gap_threshold=0.20)`
   or manually filter points with ΔE/gap > 20%. Never train on unconverged VQE data.

2. **Model capacity auto-scaling**: Call `self.select_mpnn_hidden_dim(n_graphs, theta_dim)`
   to prevent overparameterization. Floor at hidden_dim=32.

3. **Data/params ratio logging**: Log the ratio explicitly. If < 0.01, warn.

4. **norm_type="none" for cross-N**: Mandatory when training on mixed system sizes.
   BatchNorm destroys cross-N generalization (18.5% error vs 0.13%).

5. **MSE quality guard**: If final_mse > 0.5 after training, abort deployment
   (predictions would be unreliable). Log actionable advice.

## Deployment/Evaluation (Phase 4 sections)

All sections that evaluate MPNN predictions MUST include:

1. **θ_pred bounds check**: Warn if |θ_pred| > 2π (indicates poorly trained model).

2. **CrossNValidator integration**: After evaluation, run `CrossNValidator.validate_prediction()`
   for L1 (energy) and L2 (bounds/NaN) checks. L3 (LOO-CV) is optional (expensive).

3. **Warm vs Cold comparison** (when applicable): Use 1 restart for warm-start
   vs full n_restarts for cold-start to properly measure warm-start advantage.

## Backend Selection Rules

```python
# For single evaluations or ground truth:
backend = select_backend(n_qubits)  # N≤15 → Statevector, N>15 → MPS

# For VQE optimization loops (iterative, many evals):
backend = select_backend(n_qubits, for_vqe_loop=True)  # N≤10 → Statevector, N>10 → MPS
```

## Reusable Utility Methods (from ValidationRunner)

```python
# All available via self.* in any runner subclass:
metrics = self.compute_vqe_quality_metrics(vqe_energies, exact_energies, gaps)
smoothness = self.compute_theta_smoothness(theta_array)
hidden_dim = self.select_mpnn_hidden_dim(n_graphs, theta_dim, max_hidden=128)
n_violations = self.check_variational_principle(vqe_energies, exact_energies)
```

## h-grid Rules

- Default `h_min=1.0` for p=1 TFIM (h<1.0 is outside valid regime).
- Use `generate_nonuniform_h_grid(h_critical=1.0)` for denser sampling near transition.
- For cross-N: same h-grid for all training sizes (enables interpolation).
