---
inclusion: fileMatch
fileMatchPattern: "scripts/run_*,scripts/experiment_runners/run_*"
---

# Runner Script Standards — Agent Guide

## Overview

All runner scripts (`scripts/run_*.py`, `scripts/experiment_runners/run_*.py`) MUST use the
standardized runner base classes from `src/qmbp_simulation/framework/runner_base.py`.

This ensures every execution is:
- **Pre-validated** (preflight catches config errors before wasting time).
- **Logged** (StructuredLogger + ProgressReporter for post-hoc analysis).
- **Saved** (standardized JSON to `results/experiments/exp_{id}/run_{ts}.json`).
- **Digest-compatible** (parseable by `project_health/digest/`, `compare.py`, `ResultStore`).
- **Exit-coded** (non-zero on failure for CI/automation).

## Four Runner Types

| Type | When to use | Base class |
|------|-------------|------------|
| `ValidationRunner` | Multi-section suites (VQE sweeps, cross-topology, ZNE, MPS) | Most common |
| `HardwareValidationRunner` | Hardware QPU execution (IBM Torino / FakeTorino) | Hardware deployment |
| `ExperimentRunner` | Wrapping a single `BaseExperiment` subclass | Simplest |
| `VariantPipelineRunner` | Running many pipeline variants (topology × seed × params) | Batch jobs |

## MANDATORY: Choose the Right Runner

When creating or modifying a `scripts/run_*.py` file:

1. **If it runs a single BaseExperiment** → Use `ExperimentRunner`.
2. **If it has multiple test sections with tables/metrics** → Use `ValidationRunner`.
3. **If it executes on QPU or FakeTorino** → Use `HardwareValidationRunner`.
4. **If it defines `PipelineVariant` lists for batch execution** → Use `VariantPipelineRunner`.

**NEVER create a new runner script without inheriting from one of these three classes.**

## Import Pattern

```python
#!/usr/bin/env python3
"""<Description of what this runner validates/executes>."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from qmbp_simulation.framework.runner_base import (
    ValidationRunner,  # or ExperimentRunner, VariantPipelineRunner
    Section,
    resolve_project_root,
)

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)
```

**Important**: Use `resolve_project_root(__file__)` instead of hardcoded `parent.parent`.
It works from any script depth by searching for `pyproject.toml` or `Makefile`.

## ValidationRunner — Required Implementation

```python
class MyRunner(ValidationRunner):
    # ── REQUIRED attributes ──
    runner_id = "e4b_hw_readiness"     # Unique ID (used in log files)
    experiment_id = "E4b"              # Maps to result_io naming + criteria.py
    description = "E4b Hardware ..."   # One-line summary
    hypothesis = "TFIM+long ..."       # Overall hypothesis

    # ── REQUIRED method ──
    def define_sections(self) -> list[Section]:
        return [
            Section(id=1, name="...", fn=self.section_1, hypothesis="..."),
            Section(id=2, name="...", fn=self.section_2, hypothesis="..."),
        ]

    # ── Section implementations ──
    def section_1(self) -> dict:
        # ... computation ...
        return {"metric": value, "pass": True}  # MUST return dict

    # ── OPTIONAL overrides ──
    def setup(self):
        """Heavy imports + shared objects (called AFTER preflight)."""
        from qmbp_simulation import HamiltonianBuilder
        self.builder = HamiltonianBuilder()
        # Declare cross-section shared state:
        self._calibrated_chi: int | None = None

    def build_config(self) -> dict:
        """Config for the result envelope JSON (digest-compatible)."""
        return {
            "experiment_id": self.experiment_id,
            "category": "E",
            "description": self.description,
            "hypothesis": self.hypothesis,
            "system": {"n_qubits": 6, "p_layers": 1, "topology": "chain_1d", "model": "tfim"},
            "seeds": [42, 43, 44],
        }

    def run_preflight(self) -> bool:
        """Custom preflight (call super() first)."""
        if not super().run_preflight():
            return False
        # custom checks...
        return True

    @classmethod
    def _add_custom_args(cls, parser):
        """Custom CLI args (accessible as self._args.<name>)."""
        parser.add_argument("--g-value", type=float, default=0.3)
```

## Section Method Contract

Every section function MUST:
1. **Return a `dict`** (not None, not a list, not a tuple).
2. **Include `"pass": bool`** to explicitly signal pass/fail.
3. **Log tables/metrics** via `logger.info()` for console output.
4. **Not catch its own exceptions** — let the framework handle error isolation.

If `"pass"` key is absent → framework assumes success.
If section returns None → warning logged, treated as success.
If an exception is raised → framework marks it as failure and continues.

## Built-in Utility Methods

ValidationRunner provides reusable physics utilities that avoid
duplicating VQE/exact-diag/fidelity/MPS logic across runners:

### `self.vqe_descending_sweep(...)`
Run a warm-start VQE sweep, returns `dict[float, np.ndarray]` (h → θ_opt).
- Automatically uses existing `self.noiseless`/`self.backend` if available.
- Dispatches to model registry (`get_model_spec`) for any Hamiltonian.
- Supports: `model="tfim"`, `"tfim_longitudinal"`, `"tfim_frustrated"`, etc.

```python
theta_map = self.vqe_descending_sweep(
    topology="chain_1d", n_qubits=10,
    h_values=[2.5, 2.0, 1.5], seed=42,
    p_layers=1, n_restarts=1, maxiter=500,
    model="tfim_longitudinal", model_kwargs={"g": 0.3},
)
```

### `self.exact_ground_state(topology, n_qubits, h, *, model, model_kwargs)`
Get exact energy and gap. Auto-dispatches: exact diag (N≤15) or DMRG (N>15).
Static method — can be called without `self`.

```python
e_exact, gap = self.exact_ground_state("heavy_hex", 10, h=3.25)
```

### `self.compute_fidelity(circuit, theta, exact_state)`
Compute |⟨ψ_exact|ψ_vqe⟩|². Static method.

```python
fid = self.compute_fidelity(circuit, theta_opt, ground_state_vector)
```

### `self.truncate_statevector_mps(psi, n_qubits, chi_max)`
Truncate |ψ⟩ to MPS with limited bond dimension. Static method.
Useful for noise proxy simulations (low chi ≈ decoherence).

```python
psi_trunc = self.truncate_statevector_mps(psi, n_qubits=10, chi_max=16)
energy_noisy = float(np.real(psi_trunc.conj() @ H_mat @ psi_trunc))
```

### MPNN Evaluation Helpers (9 methods, 2026-06-15)

All 9 methods are available on every `ValidationRunner` subclass (no imports needed).
They produce structured JSON-serializable dicts parseable by `mpnn_eval_analyzer.py`.

**Basic suite (sections 10-14):**
```python
# S10: Warm-start speedup benchmark
result = self.benchmark_mpnn_warmstart(topology, n_qubits, h_train, h_test, ...)

# S11: LOO cross-validation
result = self.mpnn_leave_one_out_cv(topology, n_qubits, h_train, ...)

# S12: Landscape quality (circuit vs ML error decomposition + curvature)
result = self.mpnn_landscape_quality(topology, n_qubits, h_train, h_test, ...)

# S13: Interpolation vs extrapolation
result = self.mpnn_interpolation_extrapolation(topology, n_qubits, h_train,
                                               h_interpolate, h_extrapolate, ...)
```

**Extended suite (sections 15-19):**
```python
# S15: Speedup scaling with N — CRITICAL: use p_layers_per_n={N: p} for hardware
result = self.mpnn_scaling_with_system_size(topology, [4,6,10], h_train, h_test,
    p_layers_per_n={10: 1}, ...)  # p=1 for N=10 (ZNE limit 18 CX)

# S16: Learning curve (sample efficiency)
result = self.mpnn_learning_curve(topology, n_qubits, h_pool, h_test, ...)

# S17: Zero-shot topology transfer
result = self.mpnn_topology_transfer(source_topology, target_topology, n_qubits, ...)

# S18: Multi-seed LOO robustness (min_train_size propagated to inner LOO)
result = self.mpnn_data_efficiency_vs_loo(topology, n_qubits, h_pool,
    n_seeds=3, min_train_size=3, ...)

# S19: κ-noise correlation (hardware risk proxy)
result = self.mpnn_curvature_noise_correlation(topology, n_qubits, h_grid,
    noise_levels=[0.01, 0.05, 0.10, 0.20], ...)
```

**Key constraints:**
- S17: chain→ladder transfer **FAILS** (ratio=200x) — GNN params are topology-specific
- S19: `|r|` is absolute value — negative correlation (high κ = low risk) is CORRECT
- S15: Always pass `p_layers_per_n` for N≥10 to respect ZNE 18-CX limit
- κ thresholds auto-calibrate via percentiles for non-chain topologies

### Backend Resolution (`self._resolve_backend()`)
Internally resolves which backend to use: `self.noiseless` → `self.backend` → new `NoiselessBackend()`.
Called automatically by `vqe_descending_sweep`. No need to call explicitly.

## Cross-Section Data Sharing

Sections execute sequentially. Share data via instance attributes:

```python
def setup(self):
    self._calibrated_chi: int | None = None  # Set by section 1
    self._phase2_data: dict | None = None    # Set by section 2

def section_calibration(self) -> dict:
    chi = self._compute_chi(...)
    self._calibrated_chi = chi  # Available to subsequent sections
    return {"chi": chi, "pass": True}

def section_scaling(self) -> dict:
    chi = self._calibrated_chi or 16  # Guard with fallback
    ...
```

**Rules:**
- Declare shared state in `setup()` (not `__init__`).
- Always provide a fallback value when reading (in case section was skipped via `--section`).
- Never depend on section execution order beyond what `define_sections()` guarantees.

## CLI Flags (Automatically Provided)

| Flag | Effect |
|------|--------|
| `--section 1 3` | Run only sections 1 and 3 |
| `--dry-run` | List sections without executing |
| `--stop-on-failure` | Abort after first section failure |
| `--skip-preflight` | Bypass preflight validation |
| `--verbose` / `-v` | DEBUG logging + full tracebacks |

## Result Saving & Digest Compatibility (Automatic)

Results are saved to:
```
results/experiments/exp_{experiment_id}/run_{YYYYMMDD_HHMMSS}.json
results/experiments/exp_{experiment_id}/log_{YYYYMMDD_HHMMSS}.json  (structured events)
```

The output is **dual-compatible** — parseable by result_io tools AND
digest/compare.py/ResultStore:

```json
{
  "timestamp": "...",
  "config": {
    "experiment_id": "E4b", "category": "E", "hypothesis": "...",
    "description": "...",
    "system": {"n_qubits": 6, "p_layers": 1, "topology": "chain_1d", "model": "tfim"},
    "seeds": [42, 43, 44]
  },
  "results": { "section_1": {...}, "section_2": {...} },
  "summary": { "pass_rate": 0.5, "n_sections": 2, "n_passed": 1, ... },
  "elapsed_s": 5.5,
  "metadata": { "python_version": "...", "qiskit_version": "..." },
  "analysis": {
    "experiment_id": "E4b", "n_seeds": 3,
    "summary": { "pass_rate": 0.5, ... }
  }
}
```

The `analysis` wrapper is what `project_health/digest/scanner.py` reads.
The `config.system` is what provides n_qubits/topology/model to the digest.
The `summary.pass_rate` is what `compute_verdict()` uses for experiment verdicts.

## Registering New Experiments in criteria.py

When creating a runner with a NEW `experiment_id`, you MUST register it in
`src/qmbp_simulation/framework/criteria.py`:

```python
EXPERIMENT_CRITERIA: dict[str, dict[str, Any]] = {
    ...
    "MPS_HW": {"metric": "pass_rate", "threshold": 0.80, "desc": "MPS chi-proxy matches hardware"},
}
```

Without this, `compute_verdict()` falls back to `mean_de_gap < 0.05` which
may not match your experiment's success criteria.

## ExperimentRunner Pattern

For wrapping existing `BaseExperiment` subclasses:

```python
class RunE4b(ExperimentRunner):
    runner_id = "run_e4b"

    def get_experiment_class(self):
        from experiments.generalization.exp_e4b import ExperimentE4b
        return ExperimentE4b
```

Provides: CLI (--n-qubits, --seeds, --topology), automatic preflight, exit codes,
import error handling, config validation.

## VariantPipelineRunner Pattern

For batch pipeline runs:

```python
class RunP1Variants(VariantPipelineRunner):
    runner_id = "p1_variants"
    topology = "multi"
    default_n_qubits = 10
    timeout = 1300

    def build_noiseless_variants(self, n_qubits):
        return [PipelineVariant(...), ...]
```

Provides: preflight on ALL variants before execution, --skip-preflight,
n_qubits synchronization between preflight and actual run.

## HardwareValidationRunner Pattern

For runners that execute on IBM Torino or FakeTorino:

```python
class HardwareDeployment(HardwareValidationRunner):
    runner_id = "hw_deploy_n10"
    experiment_id = "HW_DEPLOY"
    description = "IBM Torino deployment validation"
    hypothesis = "ΔE/gap<5% and correct phase at h=3.25"

    def setup(self):
        super().setup()  # Initializes self.hw_backend
        self.circuit = ...

    def define_sections(self):
        return [
            Section(id=1, name="Single-point", fn=self.section_single),
        ]

    def section_single(self) -> dict:
        result = self.hw_backend.run_deployment(
            self.circuit, self.H, self.params,
            h_value=3.25, e_exact=-12.5, gap=0.8,
        )
        return {"delta_e_gap": result.delta_e_gap, "pass": result.verdict == "PASS"}
```

Extends ValidationRunner with:
- `self.hw_backend` — HardwareBackend instance created in setup().
- Dual preflight: structural + QPU status/calibration (hardware mode only).
- Shared StructuredLogger between runner and hardware backend.
- Hardware output cross-reference in result envelope.
- CLI: `--mode hardware|fake_backend`, `--shots`, `--n-layouts`, `--n-qubits`, `--topology`.
- Always call `super().setup()` in subclass to retain hw_backend initialization.

## Anti-Patterns (NEVER DO)

```python
# ❌ Raw script without runner base
if __name__ == "__main__":
    section_1()
    section_2()
    save_results(...)

# ❌ Manual JSON saving (duplicates result_io)
with open("results/exp.json", "w") as f:
    json.dump(data, f)

# ❌ No preflight validation
# Running VQE without checking valid regime

# ❌ sys.exit(0) regardless of results
# Use exit code from runner.run()

# ❌ Catching all exceptions silently
try:
    section_1()
except:
    pass

# ❌ Reimplementing VQE sweep in each runner
# Use self.vqe_descending_sweep()

# ❌ Reimplementing exact diag dispatch
# Use self.exact_ground_state()

# ❌ Hardcoding parent.parent for project root
# Use resolve_project_root(__file__)
```

## Templates

Copy from `scripts/runner_templates/`:
- `template_validation_runner.py` — Multi-section validation.
- `template_experiment_runner.py` — BaseExperiment wrapper.
- `template_variant_runner.py` — Pipeline variant runner.

See `scripts/runner_templates/README.md` for full documentation.

## Entry Point

Always end with:
```python
if __name__ == "__main__":
    MyRunner.main()
```

`main()` handles: argument parsing → instantiation → execution → sys.exit(code).
