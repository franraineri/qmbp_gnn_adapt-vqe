---
inclusion: fileMatch
fileMatchPattern: "experiments/**"
---

# Experiment Framework — Agent Guide

## Overview

The experiment framework (`src/qmbp_simulation/framework/` + `experiments/`) is a modular system for running
noiseless quantum simulation experiments. It provides a `BaseExperiment` lifecycle with reusable techniques
for VQE optimization, MPNN enhancement, landscape analysis, and scaling demonstrations.

## Architecture

```
src/qmbp_simulation/framework/    # Infrastructure (DO NOT modify without good reason)
├── base.py              # Abstract lifecycle: setup → run → analyze → report → save
├── config.py            # Typed configs with constraint validation
├── metrics.py           # ExperimentMetrics, WarmColdComparison
├── result_store.py      # Result loading, baseline comparison
└── logging.py           # StructuredLogger

experiments/             # Experiment scripts organized by category
├── helpers/             # Reusable building blocks (import into experiments)
│   ├── dypp.py, sign_equivariant.py, parameter_freezing.py
│   ├── analytical_init.py, physics_loss.py, hessian_restart.py
│   └── active_learning.py
├── optimization/        # B1, B2, B4, C3
├── scaling/             # A3
├── landscape/           # F1, F3
├── predictor/           # C1, D1, E3
├── hardware/            # Hardware-specific experiments
└── generalization/      # E4

scripts/
├── run_experiment.py    # CLI entry point
└── compare.py           # Cross-experiment comparison

results/experiments/     # Auto-generated JSON results
```

## How to Run Experiments

```bash
# Single experiment
python scripts/run_experiment.py --exp A3 --verbose

# Multiple experiments
python scripts/run_experiment.py --exp B1 B4 F3

# List available
python scripts/run_experiment.py --list

# Compare results
python scripts/compare.py --all
```

## How to Create a New Experiment

1. Create `experiments/<category>/exp_xx_name.py`:
   - Inherit from `BaseExperiment` (from `qmbp_simulation.framework`)
   - Implement `default_config()` classmethod returning `ExperimentConfig`
   - Implement `run_single(seed: int) -> list[ExperimentMetrics]`
2. Register in `experiments/<category>/__init__.py` EXPERIMENT_REGISTRY
3. If using a new technique, add module to `experiments/helpers/`

## How to Add a New Technique

1. Create `experiments/helpers/<technique_name>.py`
2. Import from `qmbp_simulation` (models, solvers, circuits, etc.)
3. Implement as a standalone function or class that can be called from any experiment
4. Export from `experiments/helpers/__init__.py`
5. Use in experiments via `from experiments.helpers import <technique>`

## Critical Constraints (ALWAYS ENFORCE)

- **p_layers ≤ 2** — `ExperimentConfig.validate()` raises ValueError if violated
- **Pure energy cost** — never hybrid/observable cost in VQE
- **Descending h-sweep** — always sweep h from high to low (warm-start)
- **Division by gap** — always use `max(gap, 1e-10)` to avoid division by zero
- **No modification to stable code** — use `qmbp_simulation` modules via imports only

## Key Patterns

### Computing ΔE/gap safely:
```python
de_gap = abs(energy - exact_energy) / max(gap, 1e-10)
```

### Getting exact solutions:
```python
sol = self.get_exact_solution(h)  # Returns dict with lattice, hamiltonian, exact
exact_energy = sol["exact"].energy
gap = sol["exact"].gap
```

### Warm-start vs cold-start comparison:
```python
comparison = self.run_warm_cold_comparison(
    h=h, seed=seed, warm_init=theta_warm,
    hamiltonian=H, exact_energy=e_exact, gap=gap,
)
```

### Evaluating energy (uses cached estimator):
```python
energy = self.evaluate_energy(params, hamiltonian)
```

### Structured event logging (auto-available via self.slog):
```python
# In run_single():
self.slog.log("vqe_start", seed=seed, h_value=h)
self.slog.start_timer("vqe_point")
# ... do VQE ...
self.slog.stop_timer("vqe_point", event_type="vqe_complete", seed=seed, h_value=h,
                     data={"energy": energy, "n_evals": result.nfev, "converged": result.success})
# Failures:
self.slog.log("vqe_failed", seed=seed, h_value=h, data={"reason": "maxiter reached"})
```

### Auto-registering a new experiment:
```python
from qmbp_simulation.framework import BaseExperiment, ExperimentConfig

class ExperimentX1(BaseExperiment):
    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(...)

    def run_single(self, seed: int) -> list:
        ...
```

### Running a descending VQE sweep (centralized helper):
```python
# Instead of reimplementing _run_vqe_sweep locally, use:
vqe_data = self.run_vqe_sweep(h_values, seed=42)
# Returns dict[float, np.ndarray] (h → θ_opt)

# With overrides:
vqe_data = self.run_vqe_sweep(
    h_values, seed=42,
    topology="ladder",      # cross-topology mode
    n_restarts=3,           # override config
    maxiter=500,
)
```

### Auto-preflight validation:
```python
# execute() now validates config BEFORE running (p≤2, descending h, data leakage, regime)
analysis = experiment.execute()  # auto-validates

# To skip (e.g., in tests):
analysis = experiment.execute(skip_preflight=True)
```

### Computing experiment verdicts:
```python
from qmbp_simulation.framework.criteria import compute_verdict

verdict, desc = compute_verdict("G1", summary_dict)
# verdict: "confirmed" | "rejected" | "failed"
```

## Currently Implemented Experiments

| ID | Name | Status |
|----|------|--------|
| A3 | Finite-size scaling law | ⚠️ SUPERSEDED (valid N=4-20 only, see H_EXPR_MATRIX) |
| B1 | Analytical initial guess | ✅ Executed (negative result) |
| B2 | TITAN parameter freezing | ✅ Executed |
| B4 | Hessian-guided restarts | ✅ Executed (N=6 + N=10) |
| C1 | Physics-informed loss | ✅ Executed (N=6 + N=10) |
| C3 | Sign canonicalization | ✅ Executed (3 runs) |
| D1 | Weight-space phase detection | ✅ Executed (N=6 + N=10 + regularized) |
| E4 | TFIM + longitudinal field | ✅ Executed (negative result) |
| F1 | DyPP extrapolation | ✅ Executed (negative result) |
| F3 | Landscape fluctuation | ✅ Executed (p=2 + p=1 comparison) |

## Planned but NOT Implemented (do not try to run)

E3 — technique module exists but experiment script is pending (active learning).
A1, A2, B3, D3, E1 — excluded from final plan (high effort or needs external libs).

## Result Format

Every experiment produces a JSON file in `results/exp_<id>/run_<timestamp>.json`:
```json
{
  "config": { /* full ExperimentConfig */ },
  "analysis": { "summary": { "mean_de_gap": ..., "pass_rate": ... }, ... },
  "results": { "42": [...], "43": [...], "44": [...] },
  "environment": { "python": "3.12", "qiskit": "1.4.2", ... }
}
```

## Documentation References

- Status: `.kiro/steering/project-status.md` (single source of truth for project state)
- Improvement techniques: `documentation/v8/analysis-improvement-techniques.md`
- Binnacles: `documentation/binnacles/binnacle-v8-experiments-initial.md`, `*-round1.md`, `*-round2.md`

---

## Executed Experiment Records (Complete Parameters & Results)

> All V8 experiments are **noiseless** (StatevectorEstimator, no shot noise, no hardware noise).
> Backend: `qiskit.primitives.StatevectorEstimator` — exact expectation values.
> Optimizer: `scipy.optimize.minimize(method="L-BFGS-B")` unless noted.
> All use HVA ansatz with |+⟩^N initial state, pure energy cost, descending h-sweep.

### F3: Landscape Fluctuation Analysis

**Execution date:** 2026-05-22
**Noise model:** None (noiseless, StatevectorEstimator)
**Time:** 3 seconds

| Parameter | Value |
|-----------|-------|
| N (qubits) | 6 |
| p (HVA layers) | 2 |
| n_params | 4 |
| Topology | chain_1d (open boundary) |
| J (coupling) | 1.0 |
| h_values | [0.5, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0] |
| Seeds | [42, 43, 44] |
| n_samples per (h, seed) | 100 |
| Parameter sampling | Uniform in [-π, π]^4 |
| Metric: fluctuation | Var(E) / E_mean² |
| Metric: fraction_near_gs | fraction of samples with E < E_exact + gap |

**Key numerical results:**
- Fluctuation range: [1.27, 5.26] (all > 1.0 → trainable everywhere)
- fraction_near_gs: 0.0 at h≤0.8, 0.003 at h=1.0, 0.053 at h=1.5, 0.077 at h=2.0
- Boundary predictor threshold: fraction_near_gs > 0.01 → h ≥ 1.1

**Conclusion:** No barren plateaus. Limit is expressibility, not trainability.
Novel finding: `fraction_near_gs` is a training-free boundary predictor.

---

### B1: Analytical Initial Guess Validation

**Execution date:** 2026-05-22
**Noise model:** None (noiseless, StatevectorEstimator)
**Time:** 17 seconds

| Parameter | Value |
|-----------|-------|
| N (qubits) | 6 |
| p (HVA layers) | 2 |
| n_params | 4 |
| Topology | chain_1d (open boundary) |
| J (coupling) | 1.0 |
| h_values tested | [1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0] |
| Seeds | [42, 43, 44] |
| VQE maxiter | 500 |
| VQE ftol | 1e-14 |
| VQE bounds | [-π, π] per parameter |
| Baseline: n_restarts | 5 (random init, best-of-5) |
| Analytical formula (p=2) | θ_zz1=0.5J/h, θ_x1=π/4·(1-0.5/h), θ_zz2=0.15J/h, θ_x2=0.3·θ_x1 |

**Key numerical results (mean over 3 seeds):**

| h | Analytical raw ΔE/gap | VQE(from analytical) ΔE/gap | VQE(from random) ΔE/gap | Iter savings |
|---|---|---|---|---|
| 1.25 | 2.178 | 0.149 | 0.038 | 46% |
| 1.50 | 1.301 | 0.631 | 0.019 | 96% |
| 2.00 | 0.713 | 0.275 | 0.002 | 97% |
| 3.00 | 0.336 | 0.098 | 0.000 | 97% |
| 4.00 | 0.195 | 0.001 | 0.000 | 86% |

**Conclusion:** Analytical init saves iterations (86-97%) but converges to worse basin.
Warm-start descending sweep is definitively superior.

---

### A3: Finite-Size Scaling Law ⚠️ SUPERSEDED

**Execution date:** 2026-05-22 | **Status:** Results valid for N=4-20 only. Extrapolation WRONG.

**Data (N=4-20, p=2, StatevectorEstimator):**

| N | h_min | N | h_min |
|---|---|---|---|
| 4 | 0.95 | 10 | 1.40 |
| 6 | 1.20 | 20 | 2.00 |
| 8 | 1.30 | | |

**Original fit:** `h_min = 1.0 + 0.0186·N^1.331` (R²=0.9998 on N=4-20 only).

**⚠️ SUPERSEDED by H_EXPR_MATRIX (MPS deterministic, N=20-250):**
- p=1: h = 2.36 + 0.0073·N (linear, R²=0.91)
- p=2: h = 1.57 + 0.005·N (linear, R²=0.95)
- p≥3: h ≈ 1.4-1.6 (constant, independent of N — area law)

The power law overestimates 1.9× at N=60, 2.7× at N=100. Do NOT use for experiment design.

---

### B4-lite: Hessian Analysis at VQE Minima

**Execution date:** 2026-05-22
**Noise model:** None (noiseless, StatevectorEstimator)
**Time:** ~30 seconds

| Parameter | Value |
|-----------|-------|
| N (qubits) | 6 |
| p (HVA layers) | 2 |
| n_params | 4 |
| Topology | chain_1d (open boundary) |
| J (coupling) | 1.0 |
| h_values | [2.0, 1.5, 1.25, 1.0] |
| Seed | 42 |
| VQE n_restarts | 5 |
| VQE maxiter | 300 |
| Hessian method | Central finite differences |
| Hessian epsilon | 5×10⁻³ |
| Hessian size | 4×4 matrix |

**Key numerical results:**

| h | ΔE/gap | Min type | Eigenvalues [λ₁,λ₂,λ₃,λ₄] | Condition # |
|---|---|---|---|---|
| 2.00 | 0.0025 | minimum | [0.1, 3.3, 132.6, 187.5] | 1399 |
| 1.50 | 0.0101 | minimum | [3.7, 9.8, 39.1, 133.5] | 36 |
| 1.25 | 0.0339 | minimum | [5.2, 11.6, 42.7, 116.8] | 23 |
| 1.00 | 0.1545 | minimum | [7.2, 13.9, 47.3, 100.7] | 14 |

**Conclusion:** All minima genuine (no saddle points). Condition number grows
100× from h=1.0 to h=2.0. Flat direction at large h explains analytical init failure.

**Note:** B4-lite was originally run as standalone script (now removed with archive cleanup).
Results are in terminal output and this steering file only (no JSON artifact).
To reproduce, use the framework:
```bash
python scripts/run_experiment.py --exp B4 --verbose
```

---

## Infrastructure Fixes Applied (2026-05-22)

These fixes were applied to stable code BEFORE running the experiments (now in `src/qmbp_simulation/`):

| Fix | Module | Change |
|-----|--------|--------|
| DMRG gap fallback | `solvers/classical.py` | gap=0 → `max(2|J-h|, 2π/N)` with warning |
| VQE convergence check | `optimizers/vqe.py` | Log warning when `result.success=False` |
| Dataset validation | `predictors/mpnn.py` | Raise ValueError if dataset < 3 points |
| Divergence threshold | `predictors/mpnn.py` | New param `divergence_threshold` (default 0.01) |

All 131 existing tests pass after these changes.

---

## Round 2 Results (2026-05-22 afternoon)

### B4 at N=10: Hessian Landscape Verification

**Time:** ~5 min | **Script:** `run_b4_n10.py`

| h | ΔE/gap | Type | Cond # (N=10) | Cond # (N=6) |
|---|--------|------|:-------------:|:------------:|
| 2.00 | 0.52% | minimum | 1294 | 1399 |
| 1.75 | 0.95% | minimum | 52 | — |
| 1.50 | 2.72% | minimum | 33 | 36 |
| 1.25 | 10.2% | minimum | 21 | 23 |
| 1.00 | 61.8% | minimum | 13 | 14 |

**Conclusion:** Saddle-free property confirmed at N=10. Condition numbers N-independent.

---

### F3 at p=1: Landscape Comparison

**Time:** ~10s | **Script:** `run_f3_p1.py`

- p=1 mean fluctuation: 1.38 (vs p=2: 1.99) — simpler landscape
- p=1 fraction_near_gs at h=2.0: 0.14 (vs p=2: 0.11) — easier random access
- Both have NO barren plateaus

---

### D1 Regularized: Robust Phase Detection

**Time:** ~3 min | **Script:** `run_d1_regularized.py`

| Variant | Mean peak | Std | Reliable? |
|---------|:---------:|:---:|:---------:|
| No reg | 1.47 | 0.90 | ❌ |
| **Dropout=0.1** | **0.61** | **0.13** | **✅** |
| EarlyStop@0.002 | 0.73 | 0.28 | ✅ |

**Conclusion:** Dropout=0.1 makes D1 robust (7× lower variance).

---

### C1 at N=10: Physics Loss Scaling

**Time:** 85s | **Script:** `run_c1_n10.py`

| h | Baseline ΔE/gap | Physics ΔE/gap | Improvement |
|---|:---:|:---:|:---:|
| 1.50 | 0.034 | 0.041 | -22% ❌ |
| 1.75 | 0.016 | 0.017 | -8% ❌ |
| 2.00 | 0.012 | 0.011 | +10% ✅ |

**Conclusion:** Physics loss HURTS at N=10 with valid-regime-only training (-12.3% overall).
Only helps when training includes invalid regime data (where MSE≠ΔE/gap decorrelation exists).
