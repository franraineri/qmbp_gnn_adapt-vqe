# V8 Experiments — New Noiseless Simulation Suite

## Architecture Overview

```
scripts/experiments_v8/
├── README.md                    # This file
├── __init__.py
├── core/                        # Shared infrastructure
│   ├── __init__.py
│   ├── base_experiment.py       # Abstract base class (lifecycle management)
│   ├── config.py                # Typed experiment configuration dataclasses
│   ├── metrics.py               # V8Metrics, WarmColdComparison, ComparisonResult
│   ├── result_store.py          # Result storage + cross-experiment comparison
│   └── landscape.py             # Hessian, fluctuation, trajectory analysis
├── techniques/                  # Reusable technique implementations
│   ├── __init__.py
│   ├── analytical_init.py       # B1: Perturbation theory initialization
│   ├── parameter_freezing.py    # B2: TITAN-style freezing
│   ├── hessian_restart.py       # B4: Hessian-guided restarts
│   ├── physics_loss.py          # C1: Physics-informed MPNN loss
│   ├── sign_equivariant.py      # C3: Z₂ sign canonicalization
│   ├── active_learning.py       # E3: Acquisition functions + ensemble
│   └── dypp.py                  # F1: Dynamic parameter prediction
├── experiments/                  # Individual experiment scripts
│   ├── __init__.py              # Registry (only implemented experiments)
│   ├── exp_a3_scaling_law.py    # A3: Finite-size scaling
│   ├── exp_b1_analytical.py     # B1: Analytical init validation
│   ├── exp_b2_freezing.py       # B2: Parameter freezing
│   ├── exp_b4_hessian.py        # B4: Hessian restarts
│   ├── exp_c1_physics_loss.py   # C1: Physics-informed MPNN loss
│   ├── exp_c3_sign.py           # C3: Sign equivariance
│   ├── exp_d1_weight_space.py   # D1: Weight space phase detection
│   ├── exp_e4_longitudinal.py   # E4: TFIM + longitudinal field
│   ├── exp_f1_dypp.py           # F1: DyPP extrapolation
│   └── exp_f3_fluctuation.py    # F3: Landscape fluctuation
├── run_b4_n10.py                # Standalone: B4 Hessian at N=10
├── run_f3_p1.py                 # Standalone: F3 landscape p=1 vs p=2
├── run_d1_regularized.py        # Standalone: D1 with dropout regularization
├── run_c1_n10.py                # Standalone: C1 physics loss at N=10
├── results/                     # Auto-generated results (gitignored)
├── run_experiment.py            # Unified CLI runner
└── compare_results.py           # Cross-experiment comparison tool
```

## Design Principles

1. **Single entry point**: `python scripts/experiments_v8/run_experiment.py --exp A3`
2. **Composable techniques**: `techniques/` are reusable building blocks
3. **Automatic comparison**: every experiment auto-compares with V6.1 baseline
4. **Structured results**: JSON with schema validation, queryable via `compare_results.py`
5. **Crash recovery**: checkpoint after each seed, resume from last checkpoint
6. **Reproducibility**: full config serialized in results, seed pinned everywhere
7. **Constraint enforcement**: config validation raises on HVA p>2, missing seeds, etc.

## Quick Start

```bash
# List available experiments
python scripts/experiments_v8/run_experiment.py --list

# Run a single experiment
python scripts/experiments_v8/run_experiment.py --exp A3

# Run with verbose logging
python scripts/experiments_v8/run_experiment.py --exp B1 --verbose

# Override system size
python scripts/experiments_v8/run_experiment.py --exp B1 --n-qubits 20 --p 1

# Run multiple experiments
python scripts/experiments_v8/run_experiment.py --exp B1 B4 F3

# Compare results
python scripts/experiments_v8/compare_results.py --all
python scripts/experiments_v8/compare_results.py --category B
```

## Experiment Lifecycle

Every experiment follows the same lifecycle managed by `BaseExperiment`:

```python
class MyExperiment(BaseExperiment):
    @classmethod
    def default_config(cls) -> ExperimentConfig:
        """Define default configuration."""
        return ExperimentConfig(
            experiment_id="X1",
            category="X",
            description="My experiment",
            hypothesis="...",
            system=SystemConfig(n_qubits=6, h_values=[1.0, 1.5, 2.0]),
            seeds=[42, 43, 44],
        )

    def run_single(self, seed: int) -> list[V8Metrics]:
        """Execute for one seed. Return one V8Metrics per h-value."""
        results = []
        for h in self.config.system.h_values:
            sol = self.get_exact_solution(h)
            energy = self.evaluate_energy(params, sol["hamiltonian"])
            gap = sol["exact"].gap
            de_gap = abs(energy - sol["exact"].energy) / max(gap, 1e-10)
            results.append(V8Metrics(
                h_value=h, energy=energy,
                exact_energy=sol["exact"].energy,
                energy_error=abs(energy - sol["exact"].energy),
                gap=gap, relative_error=de_gap, seed=seed,
            ))
        return results
```

Lifecycle: `setup() → run() → analyze() → report() → save()`

The `execute()` method runs the full lifecycle in one call.

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `BaseExperiment` | `core/base_experiment.py` | Abstract base with lifecycle management |
| `ExperimentConfig` | `core/config.py` | Typed config with validation |
| `V8Metrics` | `core/metrics.py` | Per-point metrics with sanity checks |
| `WarmColdComparison` | `core/metrics.py` | Warm-start vs cold-start comparison |
| `ComparisonResult` | `core/metrics.py` | Cross-experiment comparison |
| `ResultStore` | `core/result_store.py` | Load/query/compare results |

## Constraints (enforced by config validation)

- HVA depth: p ≤ 2 (raises ValueError if violated)
- At least one seed required
- N=12 produces a warning (too slow for iteration)
- n_restarts > 10 produces a warning

## Result Storage

```
results/
├── exp_a3/
│   ├── run_20260522_110943.json    # Full result (config + analysis + metrics)
│   └── checkpoints/                # Auto-cleaned after success
├── exp_b1/
│   └── ...
└── baselines/                      # V6.1 reference results (cached)
    ├── baseline_n6_h1.5.json
    └── baseline_n10_h1.5.json
```

## Adding a New Experiment

1. Create `experiments/exp_xx_name.py` inheriting from `BaseExperiment`
2. Implement `default_config()` and `run_single(seed)`
3. Register in `experiments/__init__.py` EXPERIMENT_REGISTRY
4. If the experiment uses a new technique, add it to `techniques/`
5. Run: `python scripts/experiments_v8/run_experiment.py --exp XX --verbose`

## Related Documentation

- Status & results: `documentation/v8/STATUS.md`
- Improvement techniques: `documentation/v8/analysis-improvement-techniques.md`
- Binnacles: `documentation/binnacles/binnacle-v8-experiments-*.md`
- Framework guide: `.kiro/steering/v8-experiments.md`
