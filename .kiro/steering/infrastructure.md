# Infrastructure & Tooling Guide

## Architecture Overview

```
                    ResultIndex (single metadata cache)
                   ╱          |            ╲
          ResultStore    ValidationRunner    ResultScanner
          (queries)     (auto-refresh)      (health report)
                ╲            |            ╱
                 ──── load_result() ─────  (50MB guard, atomic writes)
                           |
                   load_results_from_dir()  (batch loading)
```

**Key principle**: One data layer (`ResultIndex`), one file-loading path (`load_result`),
shared base-class methods on `ValidationRunner`. Every new runner inherits checkpointing,
logging, physics setup, backend selection, and auto-refresh for free.

## Creating a New Runner (MANDATORY PATTERN)

Every new experiment runner MUST subclass `ValidationRunner` and use its infrastructure:

```python
from qmbp_simulation.framework.runner_base import ValidationRunner, Section

class MyNewRunner(ValidationRunner):
    runner_id = "my_experiment"
    experiment_id = "noiseless/tfim/heavy_hex"  # or use build_experiment_id()
    description = "Short description of what this validates"
    hypothesis = "The hypothesis being tested"

    def setup(self):
        # Standard physics objects (builder, solver, hva, make_lattice, etc.)
        self.setup_physics()
        # Auto-select backend (MPS for N>10 in VQE loops, statevector otherwise)
        self._backend = self.select_backend(self._args.n_qubits, for_vqe_loop=True)
        # Memory estimate warning for large N
        self.log_memory_estimate(self._args.n_qubits)

    def define_sections(self) -> list[Section]:
        return [
            Section(id=1, name="Ground Truth", fn=self.section_exact, hypothesis="..."),
            Section(id=2, name="VQE Sweep", fn=self.section_vqe, hypothesis="..."),
        ]

    def section_exact(self) -> dict:
        # Use self.builder, self.solver, self.make_lattice, self.get_model_spec
        ...
        return {"pass": True, "data": {...}}

    def section_vqe(self) -> dict:
        # Use self.save_checkpoint() for crash recovery
        for idx, h in enumerate(h_values):
            result = do_vqe(h)
            results.append(result)
            self.save_checkpoint("vqe_sweep", {"n_done": idx+1, "results": results})

        # Resume from checkpoint on restart
        cp = self.load_checkpoint("vqe_sweep")
        if cp:
            results = cp["results"]
            start_idx = cp["n_done"]

        # Cleanup on success
        self.cleanup_checkpoints("vqe_*")
        return {"pass": True, "data": {...}}

if __name__ == "__main__":
    MyNewRunner.main()
```

### What `ValidationRunner` provides automatically:

| Feature | How | Zero-effort? |
|---------|-----|:---:|
| Atomic JSON write (no corruption) | `save_experiment_result()` | ✅ |
| Auto-index update | After every save | ✅ |
| Auto-refresh project status | `ResultIndex.refresh_status()` after save | ✅ |
| Baseline comparison | Finds previous best, records improvement delta | ✅ |
| SIGTERM/Ctrl+C graceful shutdown | Saves partial results | ✅ |
| `--resume <file>` | Skips completed sections | ✅ |
| `--dry-run` | Lists sections without executing | ✅ |
| `--section N` | Run specific section only | ✅ |
| `--stop-on-failure` | Abort on first failed section | ✅ |
| Structured logging | `self.slog.log(event, data)` | ✅ |
| Crash recovery checkpoints | `self.save_checkpoint(label, data)` | Call in loop |
| Memory estimate logging | `self.log_memory_estimate(N)` | Call once |
| Backend auto-selection | `self.select_backend(N, for_vqe_loop=True)` | Call once |
| Physics object setup | `self.setup_physics()` | Call once |

### What `setup_physics()` initializes:

After calling `self.setup_physics()`, these are available:
- `self.builder` — `HamiltonianBuilder()`
- `self.solver` — `ClassicalSolver()`
- `self.hva` — `HVACircuitBuilder()`
- `self.make_lattice` — `make_lattice()` function
- `self.get_model_spec` — `get_model_spec()` function
- `self.noiseless` — `NoiselessBackend()` instance
- `self.NoiselessBackend` — class (for manual instantiation)
- `self.MPSBackend` — class (for large-N manual use)
- `self.VQEOptimizer` — class
- `self.VQEConfig` — class

## Result Storage Architecture

Results are saved to `results/experiments/` with hierarchical organization:

```
results/experiments/
├── exp_noiseless/{model}/{topology}/run_*.json   ← NEW (nested)
├── exp_noisy/{model}/{topology}/run_*.json       ← NEW (nested)
├── exp_hardware/{model}/{topology}/run_*.json    ← NEW (nested)
├── exp_noiseless_tfim_4/run_*.json               ← LEGACY (flat, still works)
└── .result_index.json                            ← Auto-maintained cache
```

### Key Modules

| Module | Purpose | Import |
|--------|---------|--------|
| `framework/result_io.py` | Save/load with atomic writes, 50MB guard, collision prevention | `from qmbp_simulation.framework import save_experiment_result, load_result, load_results_from_dir, build_experiment_id` |
| `framework/result_index.py` | Single metadata cache: query, stats, coverage, diagnose, refresh_status | `from qmbp_simulation.framework.result_index import ResultIndex` |
| `framework/result_store.py` | Higher-level queries with `.index` property → ResultIndex | `from qmbp_simulation.framework import ResultStore` |
| `framework/runner_base.py` | ValidationRunner base with all auto-features | `from qmbp_simulation.framework.runner_base import ValidationRunner, Section` |
| `framework/criteria.py` | EXPERIMENT_CRITERIA single source of truth for verdicts | `from qmbp_simulation.framework.criteria import compute_verdict, EXPERIMENT_CRITERIA` |

### Result Loading — ALWAYS use these (NEVER raw `json.load`)

```python
# Single file
from qmbp_simulation.framework import load_result
data = load_result(path)  # 50MB guard, empty-file check, proper error messages

# Batch (directory scan)
from qmbp_simulation.framework import load_results_from_dir
results = load_results_from_dir(Path("results/experiments/exp_noiseless_tfim_4"))
# Returns: list[tuple[Path, dict]] — sorted chronologically, skips corrupt files

# NEVER do this:
# with open(path) as f: data = json.load(f)  ← no size guard, silent corruption
```

### Result Envelope Schema (v2.0)

Every result JSON has this structure:
```json
{
  "schema_version": "2.0",
  "timestamp": "ISO format",
  "config": { "system": { "model", "topology", "n_qubits", "p_layers" }, "seeds", ... },
  "results": { "section_1": { "name", "success", "elapsed_s", "data", "error" }, ... },
  "summary": { "n_sections", "n_passed", "n_failed", "pass_rate", "all_passed" },
  "metadata": { "python_version", "qiskit_version", "torch_version", ... },
  "analysis": { "experiment_id", "summary", "per_section" },
  "baseline_ref": { "file", "pass_rate", "timestamp" },
  "improvement": { "pass_rate_delta", "is_improvement" }
}
```

### build_experiment_id() Convention

```python
from qmbp_simulation.framework.result_io import build_experiment_id

# Categories: "noiseless", "noisy", "hardware", "experiment"
eid = build_experiment_id("noiseless", "tfim", "heavy_hex")
# → "noiseless/tfim/heavy_hex"
# → saves to: results/experiments/exp_noiseless/tfim/heavy_hex/run_*.json
```

## CLI Tools

| Command | Purpose |
|---------|---------|
| `python -m project_health` | Full health report |
| `python -m project_health --diagnose` | Quick group-level failure diagnosis |
| `python -m project_health --diagnose --model tfim` | Filtered diagnosis |
| `python -m project_health --refresh-status` | Regenerate project-status.md |
| `python project_health/cli/query_index.py --stats` | Index statistics |
| `python project_health/cli/query_index.py --coverage` | Coverage matrix |
| `python project_health/cli/query_index.py --suggest` | Suggest next experiments |
| `python project_health/cli/query_index.py --regressions` | Detect regressions |
| `python project_health/cli/query_index.py --best --model tfim --n-qubits 20` | Best run for config |
| `python project_health/cli/inspect_noiseless_run.py --latest exp_noiseless_tfim_4` | Per-h-point breakdown |
| `python scripts/scan_new_runs.py --dirs exp_noiseless_tfim_4` | Deep per-run analysis |

## Runner Features (auto-applied to ALL ValidationRunner subclasses)

1. **`--resume <file>`** — Skip completed sections, restore VQE data
2. **KeyboardInterrupt** — Saves partial results (completed sections preserved)
3. **SIGTERM** — Graceful shutdown with save (for nohup/kill)
4. **Baseline comparison** — Auto-detects previous best from index
5. **Checkpoint streaming** — `save_checkpoint()` / `load_checkpoint()` for crash recovery
6. **Atomic write** — Prevents corrupt JSON on crash (temp + rename)
7. **Auto-index update** — New runs instantly queryable
8. **Auto-refresh status** — `.kiro/steering/project-status.md` always current
9. **Memory estimate** — `log_memory_estimate(N)` warns before OOM
10. **Backend auto-select** — `select_backend(N)` → MPS or Statevector

## NEVER patterns (anti-patterns to avoid)

```python
# ❌ NEVER: Manual json.load without error handling
with open(path) as f:
    data = json.load(f)
# ✅ INSTEAD: Use load_result(path)

# ❌ NEVER: Manual glob + parse loops
for f in sorted(exp_dir.rglob("run_*.json")):
    with open(f) as fh:
        data = json.load(fh)
# ✅ INSTEAD: Use load_results_from_dir(exp_dir)

# ❌ NEVER: Duplicate physics setup in every runner
from qmbp_simulation import HamiltonianBuilder, ClassicalSolver, ...
self.builder = HamiltonianBuilder()
self.solver = ClassicalSolver()
# ✅ INSTEAD: self.setup_physics()

# ❌ NEVER: Manual backend selection
if N <= 22:
    backend = NoiselessBackend()
else:
    backend = MPSBackend(...)
# ✅ INSTEAD: backend = self.select_backend(N, for_vqe_loop=True)

# ❌ NEVER: Bare except Exception: pass
try:
    something()
except Exception:
    pass  # Silent failure
# ✅ INSTEAD:
try:
    something()
except Exception as e:
    logger.debug("Context: %s", e)

# ❌ NEVER: Duplicate experiment criteria
CRITERIA = {"my_exp": {"metric": "pass_rate", ...}}
# ✅ INSTEAD: Add to framework/criteria.py EXPERIMENT_CRITERIA

# ❌ NEVER: json_default functions in scripts
def _json_default(obj): ...
# ✅ INSTEAD: from qmbp_simulation.utils.helpers import json_serialize, json_dump
```

## Physics Sanity Checks (auto-active in core modules)

- Variational principle: 3-tier escalation (benign noise → warning → CRITICAL)
- Fidelity bounds: clips to [0,1] with logging
- MPS normalization: detects χ truncation
- Lattice integrity: self-loops (error), duplicates (warning), isolated sites (warning)
- Gap degeneracy: warns when ΔE/gap is unreliable
- Observable bounds: |⟨O⟩| > 1 flagged as error

## Running Experiments

```bash
# Noiseless pipeline (primary)
.venv/bin/python scripts/experiment_runners/noiseless/run_noiseless_pipeline.py \
    --n-qubits 20 --p-layers 4 --topology heavy_hex --model tfim \
    --h-min 1.3 --h-max 3.0 --h-points 40 --maxiter 1000 --n-restarts 7

# Resume interrupted run
.venv/bin/python scripts/experiment_runners/noiseless/run_noiseless_pipeline.py \
    --resume results/experiments/exp_noiseless/tfim/heavy_hex/run_INTERRUPTED.json \
    --n-qubits 20 --p-layers 4 --topology heavy_hex --model tfim

# With nohup for long runs
nohup .venv/bin/python scripts/experiment_runners/noiseless/run_noiseless_pipeline.py \
    --n-qubits 20 --p-layers 4 --topology heavy_hex --model tfim \
    --h-min 1.25 --h-max 3.0 --h-points 40 > results/run_output.log 2>&1 &

# Quick health check after running
python -m project_health --diagnose --model tfim
```

## Data Flow (end-to-end)

```
Runner.run() → save_experiment_result()
    → atomic write (temp + rename)
    → ResultIndex.add_entry() (instant queryable)
    → ResultIndex.refresh_status() (Kiro context updated)
    → project-status.md regenerated

Later:
    python -m project_health --diagnose     → reads from ResultIndex (cached, fast)
    python -m project_health                → ResultScanner → full health report
    python scripts/scan_new_runs.py         → deep per-point analysis
```
