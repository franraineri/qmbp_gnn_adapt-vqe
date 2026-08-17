---
inclusion: fileMatch
fileMatchPattern: "**/pipeline/**,**/runner.py,**/theta_validator*,**/vqe_validator*,**/diagnostics.py"
---

# Pipeline Context (invoke with #context-pipeline)

> Pre-digested context for PipelineRunner phases, validators, result schemas, and orchestration.

## Pipeline Phases

| Phase | Goal | Key class/function | Output |
|-------|------|-------------------|--------|
| 1 | Exact ground truth | `ClassicalSolver.solve()` / `run_exact_diag_sweep()` | E_exact, gap, observables |
| 2 | VQE θ_opt via warm-start | `VQEOptimizer.descending_sweep()` | θ_opt dataset + diagnostics |
| 3 | MPNN predictor training | `train_mpnn()` | Trained model, per-h MSE |
| 4 | Deployment (predict + evaluate) | `PipelineRunner.run_phase4()` | ΔE/gap, phase label |

## Core Orchestration

lattice = make_lattice("chain_1d", n_qubits=6, J=1.0, h=2.0)
config = VQEConfig(p_layers=2, n_restarts=5, maxiter=1000)
runner = PipelineRunner(lattice=lattice, config=config, verbose=True)
results = runner.run_full(
    h_values=h_values, h_test=[1.5],
    mpnn_config={"hidden_dim": 128, "n_epochs": 6000, "patience": 500},
)
# Keys: "phase1", "phase2", "phase3", "phase4", "diagnostics"
```

## Validation Auto-Integration

### VQE Validator (Phase 2 — automatic)
```python
from qmbp_simulation.analysis import VQEValidator
# Runs automatically after Phase 2. Checks:
# - Variational principle: E_vqe ≥ E_exact (always)
# - Energy bounds: E_vqe within [E_exact, E_exact + gap]
# - θ bounds: all parameters in [-π, π]
# - Convergence: optimizer terminated normally
# - Sweep quality: θ_smoothness < 1.0
# Results in diagnostics.vqe_validation
# CLI: --no-validate-vqe to disable, --strict-validation to abort on CRITICAL
```

validator = ThetaValidator.from_training_data(theta_opt_array, h_values)
report = validator.validate(theta_pred, level=4, circuit=qc, exact_state=psi)
report.passes()           # True if all checks pass
report.confidence_score   # [0, 1] weighted composite
# Levels: L1=bounds, L2=NaN, L3=interpolation, L4=fidelity, L5=gradient, L6=MC-Dropout, L7=sensitivity
# Default in runner: L1-L4 (zero extra circuit evals)
```

## Result JSON Schema (key fields)

```json
{
  "config": { "n_qubits": 6, "topology": "chain_1d", "p_layers": 2, ... },
  "phase4_results": [{
    "h_test": 1.5,
    "delta_e_over_gap": 0.017,   // PRIMARY METRIC (< 0.05 = pass)
    "phase_label": "paramagnetic"
  }],
  "diagnostics": {
    "phase2": { "theta_smoothness": 0.036, "convergence_rate": 1.0 },
    "phase3": { "generalization_gap": 1.96e-05 },
    "phase4": { "energy_decomposition": { "error_from_circuit": 0.0, "error_from_mpnn": 0.036 } },
    "vqe_validation": { ... },
    "theta_validation": [{ "level": 4, "passes": true, "confidence": 0.98 }]
  }
}
```

## Early-Stopping Rules

```
PRE-RUN:  h_test ≥ valid_regime_boundary + 0.5
Phase 2:  IF θ_smoothness > 1.0 → WARN (chain break, 45% of failures)
Phase 3:  IF generalization_gap > 0.01 → ABORT (MPNN overfit, 25% of failures)
```

## Key Quality Metrics

| Metric | Good | Suspect | Failure |
|--------|------|---------|---------|
| `delta_e_over_gap` | < 0.05 | 0.05-0.10 | > 0.10 |
| `theta_smoothness` | < 0.1 | 0.1-1.0 | > 1.0 |
| `generalization_gap` | < 1e-3 | 1e-3 to 0.01 | > 0.01 |
| `convergence_rate` | 1.0 | 0.8-1.0 | < 0.8 |
| `fidelity` (noiseless) | ≥ 0.93 | 0.80-0.93 | < 0.80 |

## DiagnosticCollector Pattern

```python
collector = DiagnosticCollector(verbose=True, save_dir=Path("results/"))
collector.record_vqe_point(h, n_iters, restart_energies, theta_opt, elapsed_s)
collector.record_mpnn_per_h_error(h_values, per_h_mse)
collector.record_deployment(h_test, result, per_layout_data)
result["diagnostics"] = collector.to_dict()
```

## DO NOT

- Use ascending h-sweep (breaks warm-start: θ_opt(h_{i}) seeds θ_opt(h_{i+1})).
- Use phase coupling in cost function (V5.x lesson — pure energy only in Phase 2).
- Train MPNN on points with fidelity < 0.93 (poisons model).
- Skip preflight before running variant scripts.
- Use `sys.path.insert()` — package is installed.

## Source Files

- #[[file:src/qmbp_simulation/pipeline/runner.py]]
- #[[file:src/qmbp_simulation/analysis/vqe_validator.py]]
- #[[file:src/qmbp_simulation/analysis/theta_validator.py]]
- #[[file:src/qmbp_simulation/analysis/diagnostics.py]]
- #[[file:src/qmbp_simulation/framework/result_io.py]]
- #[[file:src/qmbp_simulation/framework/preflight.py]]
- #[[file:.kiro/knowledge/result-schemas.md]]
- #[[file:.kiro/knowledge/workflow-recipes.md]]
