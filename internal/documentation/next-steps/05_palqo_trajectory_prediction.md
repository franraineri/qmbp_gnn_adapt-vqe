# Integration Plan 05: PALQO Trajectory Prediction (Phase 2 Speedup)

**Paper:** Huang et al. (2025) — PALQO: Physics-informed Model for Accelerating Large-scale QO  
**arXiv:** 2509.20733 (NeurIPS 2025)  
**Code:** ❌ No public repository  
**Priority:** HIGH impact but HIGH effort (2 weeks, 80-90% Phase 2 speedup)

## What It Does

PALQO models VQE optimization dynamics as a nonlinear PDE:
  ∂θ/∂t = f(θ, ∇E(θ))

A Physics-Informed Neural Network (PINN) learns this dynamical system from a few
initial optimization trajectories. Given θ at iteration t=10, the PINN predicts
θ_final without running the remaining 90-190 iterations.

Key insight: VQE optimization follows smooth trajectories governed by the energy
landscape geometry. A PINN that captures this geometry can extrapolate convergence.

## Viability Assessment

| Criterion | Status |
|-----------|--------|
| Compatible with our pipeline? | ✅ VQEOptimizer already logs trajectories |
| Requires new dependencies? | ❌ PyTorch only |
| Reuses existing modules? | ✅ `VQEOptimizer`, `OptimizationCallback`, trajectory data |
| Addresses a real problem? | ✅ Phase 2 is 80-95% of pipeline compute time |
| Data available? | ⚠️ Need to collect trajectories (currently only log final θ) |
| Publishable? | ✅ "Phase 2 acceleration from 200→20 iters" is strong claim |

## How To Integrate

### What It Proves

That VQE optimization trajectories for HVA on TFIM are sufficiently smooth and
predictable that a neural network can extrapolate convergence from partial data,
reducing Phase 2 cost by 80-90%.

### Conditions Where It Makes Sense

- **Models:** `tfim`, `tfim_longitudinal` (smooth landscapes, L-BFGS-B converges monotonically)
- **Topologies:** ALL (trajectory structure is universal for smooth landscapes)
- **N:** 10-100 (larger N = more iterations needed = more value from acceleration)
- **p:** 1-4 (all tested depths)
- **Optimizer:** L-BFGS-B only (COBYLA trajectories are less smooth/predictable)

### When NOT to Use

- `heisenberg`, `kitaev`: HVA incompatible (VQE doesn't converge → no trajectory to learn)
- Near h_c with p=1: landscape has saddle points → trajectory is non-monotone
- First run (no trajectory data): must collect at least 10-20 full trajectories first
- `tfim_bond_resolved` with COBYLA: stochastic steps → PDE model breaks down

### Integration Architecture

```
src/qmbp_simulation/
└── optimizers/
    ├── vqe.py                    # ✅ EXISTS: VQEOptimizer, OptimizationCallback
    ├── trajectory_predictor.py   # NEW: PINN for trajectory extrapolation
    └── accelerated_vqe.py        # NEW: Wrapper combining VQE + PINN
```

### Modules to Reuse

| Module | Usage |
|--------|-------|
| `optimizers.vqe.VQEOptimizer` | Source of trajectories + warm-start logic |
| `optimizers.vqe.OptimizationCallback` | Already logs θ per iteration! |
| `models.VQEResult.trajectory` | `OptimizationTrajectory` dataclass (energies, params) |
| `pipeline.runner.PipelineRunner.run_phase2` | Integration point |
| `execution.NoiselessBackend` | Single energy eval for validation |

### Architecture Design

```python
class TrajectoryPINN(nn.Module):
    """Physics-Informed NN for VQE trajectory extrapolation.

    Input: θ(t=0..T_init), E(t=0..T_init), ∇E estimates
    Output: θ(t=T_final) prediction

    Physics loss: ∂θ/∂t should follow gradient descent dynamics
    Data loss: MSE on known trajectory points
    """

    def __init__(self, theta_dim: int, hidden: int = 64, n_layers: int = 3):
        self.net = nn.Sequential(
            nn.Linear(theta_dim + 2, hidden),  # +2 for (t, E(t))
            nn.Tanh(),
            *[nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh()) for _ in range(n_layers-1)],
            nn.Linear(hidden, theta_dim)
        )

    def forward(self, theta_t, t, energy_t):
        """Predict θ at next timestep."""
        x = torch.cat([theta_t, t.unsqueeze(-1), energy_t.unsqueeze(-1)], dim=-1)
        return self.net(x)


class AcceleratedVQE:
    """VQE with PINN-based trajectory extrapolation.

    Strategy:
    1. Run VQE for T_init iterations (default: 10-20)
    2. Feed partial trajectory to trained PINN
    3. PINN predicts θ_final
    4. Validate: evaluate E(θ_predicted) once
    5. If ΔE/gap < threshold: accept (save 80-90% iterations)
       Else: fall back to full VQE from θ_predicted as warm-start
    """
```

### Implementation Steps

1. **Modify `VQEOptimizer`** to save full trajectories (currently optional via callback):
   - Add `save_trajectory=True` flag to sweep methods
   - Store `OptimizationTrajectory` in VQEResult (already supported!)
   - Collect 20-50 complete trajectories across h-grid

2. **Create `optimizers/trajectory_predictor.py`** (~120 lines):
   - `TrajectoryPINN` model
   - Training: MSE on known trajectories + physics regularizer
   - Inference: predict θ_final from θ(0:T_init)

3. **Create `optimizers/accelerated_vqe.py`** (~80 lines):
   - `AcceleratedVQE` wrapper
   - Falls back to full VQE if PINN prediction fails validation
   - Logs acceleration statistics (iterations saved, fallback rate)

4. **Training data collection script** `scripts/collect_trajectories.py`:
   - Run Phase 2 with `enable_callbacks=True` on representative configs
   - Save trajectories to `results/trajectories/` (NPZ format)

5. **Benchmark**: Compare Phase 2 time with vs without PINN acceleration

### Expected Output

```json
{
  "model": "tfim_longitudinal",
  "topology": "chain_1d",
  "N": 40,
  "p": 2,
  "method": "accelerated_vqe",
  "mean_iters_full_vqe": 180,
  "mean_iters_with_pinn": 15,
  "acceleration_factor": 12.0,
  "fallback_rate": 0.10,
  "de_gap_full_vqe": 0.008,
  "de_gap_accelerated": 0.012,
  "phase2_time_full": "45 min",
  "phase2_time_accelerated": "5 min"
}
```

### Success Criterion

- Acceleration ≥ 5× on Phase 2 with < 5% quality loss → publishable
- Fallback rate < 20% (PINN succeeds most of the time)
- Works for N=20-100 (where Phase 2 is the bottleneck)

### Risks

- Need sufficient trajectory data (20-50 full runs) before PINN can be trained
- L-BFGS-B uses line search → iteration count varies unpredictably → T_init choice matters
- Physics loss assumes gradient descent dynamics, but L-BFGS-B uses quasi-Newton
  → may need to reformulate as θ(t+1) = g(θ(t), E(t), E(t-1)) without explicit PDE
- For tfim_bond_resolved with COBYLA: trajectory is noisy → PINN may not generalize
- Bootstrap problem: first campaign has no training data → collect in first pass, use in second
