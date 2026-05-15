# Plan: Pipeline Observability & Diagnostic Metrics (V6.1)

## Objective

Improve experiment debugging, metric richness, and run resilience across the V6.1 pipeline. This adds structured logging, new diagnostic metrics, incremental saving, and a verbose mode — without modifying stable modules.

---

## Motivation

Current gaps identified from 40+ benchmark runs and 15 thesis executions:

1. **Lost work on crashes** — if Phase 3 fails, Phase 1-2 data (which took minutes) is lost
2. **Black-box VQE** — `enable_callbacks=False` means we can't see convergence behavior
3. **No per-h diagnostics** — global MSE hides that the MPNN struggles specifically at h∈[1.0, 1.3]
4. **Flat logging** — everything is `print()`, no way to filter errors from info
5. **Missing thesis-relevant metrics** — SNR, θ smoothness, generalization gap are computed internally but never recorded

---

## Architecture

### New Module: `src/poc/v6/diagnostics.py` (~200 lines)

A lightweight diagnostic collector that accumulates metrics during pipeline execution and serializes them. Does NOT modify any stable module — it observes from outside.

```python
class DiagnosticCollector:
    """Accumulates per-phase diagnostic metrics during pipeline execution."""

    def __init__(self, verbose: bool = False, save_dir: Path | None = None):
        ...

    # Phase 2 diagnostics
    def record_vqe_point(self, h: float, n_iters: int, restarts_energies: list[float],
                         theta_opt: np.ndarray, elapsed_s: float) -> None: ...

    # Phase 3 diagnostics  
    def record_mpnn_epoch(self, epoch: int, train_loss: float,
                          val_loss: float | None = None) -> None: ...
    def record_mpnn_per_h_error(self, h_values: np.ndarray,
                                 per_h_mse: np.ndarray) -> None: ...

    # Phase 4 diagnostics
    def record_deployment(self, h_test: float, result: DeployResultV61,
                          per_layout_data: dict | None = None) -> None: ...

    # Persistence
    def save_checkpoint(self, phase: str) -> Path: ...
    def save_final(self) -> Path: ...
    def to_dict(self) -> dict: ...
```

### Modifications to Active Scripts (NOT stable modules)

| File | Change | Lines |
|------|--------|-------|
| `scripts/run_v61_parametric.py` | Add `--verbose` flag, use DiagnosticCollector | ~40 |
| `scripts/run_v61_noisy.py` (new) | Built with diagnostics from the start | included |
| `src/poc/v6/analysis_utils.py` | Add `compute_snr()` and `compute_theta_smoothness()` helpers | ~30 |

### What We Do NOT Touch

- `config.py`, `hamiltonian_builder.py`, `classical_solver.py`, `hva_builder.py`, `pipeline_utils.py`, `vqe_optimizer.py` — all stable
- The VQE callbacks mechanism already exists in `VQEOptimizer` (just disabled by default) — we enable it via the existing `enable_callbacks=True` flag when `--verbose` is set

---

## New Metrics — By Phase

### Phase 2: VQE Diagnostics

| Metric | Computation | Cost | Value |
|--------|-------------|------|-------|
| **Iterations to convergence** per h-point | From VQE callback (already exists) | Zero (just enable) | Shows warm-start effectiveness |
| **Restart energy spread** | `std(E_restart_1, ..., E_restart_5)` | Zero (already computed) | Landscape ruggedness indicator |
| **Best-vs-worst restart gap** | `max(E) - min(E)` across restarts | Zero | Multi-start justification |
| **θ smoothness** | `max(|θ(h_i) - θ(h_{i-1})|)` across sweep | O(N_points) | Predicts MPNN learnability |
| **Per-h-point timing** | `time.time()` around each VQE call | Zero | Identifies stuck optimizations |

**Implementation**: The `VQEOptimizer` already supports `enable_callbacks=True` which records `OptimizationTrajectory`. We just need to:
1. Enable it when `--verbose` is set
2. Extract `n_iterations` and `final_energy` per restart from the trajectory
3. Record in DiagnosticCollector

### Phase 3: MPNN Diagnostics

| Metric | Computation | Cost | Value |
|--------|-------------|------|-------|
| **Per-h prediction error** | `|θ_pred(h) - θ_opt(h)|²` for each h in training set | O(N_train) | Identifies hard h-regions |
| **Generalization gap** | Hold out 2 points, compare train vs val MSE | ~5% more compute | Overfitting detection |
| **Per-parameter MSE** | Separate MSE for θ_zz vs θ_x | Zero (just split) | Which params are harder |
| **Loss curve (last 100 epochs)** | Already computed, just save it | Zero | Convergence quality |
| **Gradient norm during training** | `torch.nn.utils.clip_grad_norm_` value | Near-zero | Training stability |

**Implementation**: `train_mpnn()` already returns `loss_history`. We add:
1. Return `per_h_mse` array (compute after training, one forward pass)
2. Return `theta_zz_mse` and `theta_x_mse` separately
3. In verbose mode, save full loss curve (not just final MSE)

### Phase 4: Deployment Diagnostics

| Metric | Computation | Cost | Value |
|--------|-------------|------|-------|
| **Observable SNR** | `|⟨X⟩| / σ` where σ = 1/√shots | Zero | Measurement reliability |
| **Phase classification confidence** | `|⟨X⟩ - ⟨ZZ⟩| / σ` | Zero | Robustness of label |
| **Per-layout energy spread** | `std(E_layout_1, ..., E_layout_3)` | Zero (already in DeployResultV61) | ZNE diversity |
| **CES-energy Pearson r** | `np.corrcoef(ces_values, energies)` | Zero | Linear ZNE validity |
| **Energy error decomposition** | Compare `E_VQE(θ_opt)` vs `E_VQE(θ_pred)` vs `E_exact` | 1 extra estimator call | Separates MPNN error from circuit limit |

**Implementation**: Most are trivial computations on data already in `DeployResultV61`. The energy decomposition requires one extra `StatevectorEstimator` call with `θ_opt` (from Phase 2) to get `E_VQE_ceiling`.

---

## Incremental Saving

### Problem
A 6-minute N=10 run that crashes in Phase 4 loses all Phase 1-3 data.

### Solution
After each phase completes, write a checkpoint:

```
scripts/notebook_results/checkpoint_<run_id>_phase1.json
scripts/notebook_results/checkpoint_<run_id>_phase2.json
scripts/notebook_results/checkpoint_<run_id>_phase3.json
scripts/notebook_results/parametric_run_<timestamp>_<run_id>.json  ← final
```

On successful completion, delete checkpoints and write the final JSON. On crash, checkpoints remain for inspection.

### Resume (optional, low priority)
A `--resume <run_id>` flag that loads the last checkpoint and continues from there. Useful for N=10 runs where Phase 2 takes 3+ minutes.

---

## Verbose Mode (`--verbose` / `-v`)

### Behavior

| Without `--verbose` | With `--verbose` |
|---------------------|------------------|
| Phase summaries only | Per-h-point progress |
| Final MSE only | Per-epoch loss (last 100) |
| Checklist pass/fail | Full metric breakdown |
| ~50 lines output | ~200 lines output |
| No VQE callbacks | VQE convergence tracking |
| No per-layout data | Per-layout energies + CES |

### Logging Levels

```python
import logging

# Default (no flag): WARNING only
# --verbose: INFO level  
# --debug: DEBUG level (everything, including per-iteration VQE)

logger = logging.getLogger("gnn_hva")
```

| Level | What it shows |
|-------|---------------|
| DEBUG | Per-VQE-iteration energy, per-MPNN-epoch loss, per-layout raw values |
| INFO | Per-h-point summary, phase transitions, metric computations |
| WARNING | Borderline metrics (ΔE/gap 4-5%), low R², slow convergence |
| ERROR | Failures, NaN detection, assertion violations |

---

## Structured Output Enhancement

### Current JSON structure (keep as-is)
```json
{
  "config": {...},
  "phases": {
    "phase1": {"elapsed_s": ..., "n_points": ...},
    "phase2": {"elapsed_s": ..., "avg_fidelity": ...},
    "phase3": {"elapsed_s": ..., "final_mse": ...},
    "phase4": {"elapsed_s": ..., "delta_e_over_gap": ...}
  }
}
```

### New `diagnostics` section (added when `--verbose`)
```json
{
  "diagnostics": {
    "phase2": {
      "per_h_timing_s": [1.2, 1.1, 3.8, ...],
      "per_h_iterations": [45, 38, 120, ...],
      "per_h_restart_spread": [0.001, 0.003, 0.05, ...],
      "theta_smoothness": 0.42,
      "worst_convergence_h": 1.1
    },
    "phase3": {
      "per_h_mse": {"0.9": 1.2e-4, "1.0": 3.8e-3, ...},
      "theta_zz_mse": 1.5e-4,
      "theta_x_mse": 3.2e-4,
      "generalization_gap": 0.15,
      "loss_curve_last100": [0.0012, 0.0011, ...]
    },
    "phase4": {
      "snr_mag_x": 12.3,
      "snr_corr_zz": 8.7,
      "classification_confidence": 5.2,
      "per_layout_energies": [-9.12, -9.08, -9.15],
      "per_layout_ces": [0.032, 0.045, 0.058],
      "ces_energy_pearson_r": 0.94,
      "energy_decomposition": {
        "e_exact": -9.25,
        "e_vqe_ceiling": -9.22,
        "e_mpnn_predicted": -9.15,
        "error_from_circuit": 0.03,
        "error_from_mpnn": 0.07
      }
    }
  }
}
```

---

## Implementation Steps

### Step 1: Create `src/poc/v6/diagnostics.py` (~200 lines)

- `DiagnosticCollector` class with phase-specific recording methods
- Checkpoint save/load (JSON serialization with numpy handling)
- Verbose print helpers (formatted tables, progress bars)

### Step 2: Add helper functions to `analysis_utils.py` (~30 lines)

- `compute_snr(observable_value: float, shots: int) -> float`
- `compute_theta_smoothness(theta_array: np.ndarray) -> float`
- `compute_classification_confidence(mag_x: float, corr_zz: float, shots: int) -> float`
- `compute_energy_decomposition(e_exact, e_vqe_ceiling, e_predicted) -> dict`

### Step 3: Integrate into `run_v61_parametric.py` (~40 lines)

- Add `--verbose` / `-v` and `--debug` flags
- Instantiate `DiagnosticCollector` when verbose
- Enable VQE callbacks when verbose
- Save checkpoints after each phase
- Include `diagnostics` section in output JSON

### Step 4: Build into `run_v61_noisy.py` from the start

- The noisy simulation script (from the other spec) should use DiagnosticCollector natively
- Per-layout data is especially important for ZNE validation

---

## What This Does NOT Change

- No modifications to stable modules
- No changes to the pipeline logic (same results with or without `--verbose`)
- No new dependencies (uses only stdlib + numpy)
- Backward-compatible JSON output (diagnostics section is additive)
- Default behavior unchanged (no verbose = same output as today)

---

## Success Criteria

1. `run_v61_parametric.py --verbose` produces the extended diagnostics JSON
2. Crash at any phase preserves previous phase data via checkpoints
3. `--verbose` output identifies the specific h-value causing issues (not just "MSE is high")
4. Energy decomposition clearly separates "MPNN error" from "circuit expressibility limit"
5. SNR metric correctly predicts which hardware measurements will be unreliable

---

## Thesis Value

- **Table 4.x**: "Pipeline Diagnostic Summary" — shows convergence quality across phases
- **Figure 4.x**: Per-h prediction error curve (identifies critical region as the bottleneck)
- **Section 4.4**: Energy error decomposition proves the ceiling is physics, not pipeline
- **Hardware planning**: SNR predictions guide shot budget decisions before spending QPU credits
