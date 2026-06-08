---
inclusion: fileMatch
fileMatchPattern: "**/hardware/**,**/hardware_deployer*,scripts/run_hardware*"
---

# Hardware Deployment — Phase 4 Guidelines

## IBM Torino Target (133 qubits, Eagle r3)

### Connection
- Use `QiskitRuntimeService` with `channel="ibm_quantum_platform"`
- Token from `os.environ["IBM_KEY"]`, instance from `os.environ["IBM_INSTANCE_CRN"]`
- Backend: `"ibm_torino"` (or latest Eagle r3 processor)

### Circuit Preparation
- `generate_preset_pass_manager(backend=backend, optimization_level=2)`
- Apply layout to observables: `obs.apply_layout(isa_qc.layout)`
- Add dynamical decoupling: `PadDynamicalDecoupling` pass after transpilation

### Error Mitigation Stack (in order of application)
1. **Dynamical Decoupling** — free, always apply. Use optimized sequences if available.
2. **Pauli Twirling** — 32 randomizations × 128 shots. Converts coherent → stochastic noise.
3. **TREX** — twirled readout error extinction. Enable via EstimatorV2 options.
4. **ZNE** (configurable amplifier — select via `MitigationOptions.zne_amplifier`):
   - **PEA** (`"pea"`, default): Probabilistic Error Amplification. Learns noise model
     via Pauli-Lindblad fitting, then amplifies probabilistically. ~50% QPU overhead from
     noise learning phase. Validated +94.4% gain across all topologies (t=46.32, p<10⁻¹⁹).
     IBM Runtime handles it automatically via `options.resilience.zne.amplifier = "pea"`.
   - **Gate-folding** (`"gate_folding"`, fallback): Digital noise amplification U→U·U†·U at
     factors [1,3,5]. Simple, zero overhead, validated locally (R²>0.99 on chain_1d).
     May give low R² on heavy_hex p=1 shallow circuits.
   - **Adaptive** (`"adaptive"`): Tries gate-folding first; if R²<threshold (default 0.90),
     falls back to PEA. Best for unattended deployment where topology properties are uncertain.
     Configure threshold via `MitigationOptions.zne_r2_fallback_threshold` or `--zne-r2-threshold`.
   - **Inhomogeneous CES-ZNE** (deprecated for heavy_hex): Different layouts → different CES.
     Fails on heavy_hex due to uniform CES≈0.15. Only used as legacy path when `zne_enabled=False`.
5. **NN-enhanced extrapolation** — optional improvement:
   - After collecting ZNE data, fit 2-layer MLP instead of linear regression
   - `MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000)`

### Shot Budget
- Minimum: **8192 shots** (σ ≈ 1.1e-2, comparable to ⟨X⟩ signal)
- Recommended for N=10: **16384 shots** (σ ≈ 7.8e-3, below ⟨X⟩ signal of 8.4e-3)
- Use `EstimatorV2` `precision` parameter to control shot allocation

### Observable Grouping
- ⟨X_i⟩ observables: all commute (single measurement basis)
- ⟨Z_iZ_{i+1}⟩ observables: all commute (single measurement basis)
- Total: 2 circuit executions per noise level (not N+N-1 separate runs)

### Expected Hardware Behavior (from literature)
- Ground-state energies: reliably captured across full parameter space
- Magnetic order parameters: noise broadening near critical crossover
- Phase classification: correct away from h_c, "smeared" near transition
- Success criterion: ΔE/gap < 5% AND correct phase label — NOT fidelity ≥ 99.5%

### AdaptVQE on Hardware
- max_iterations = 2 (Mele et al. constraint)
- gradient_threshold = 1e-3
- If AlgorithmError at iteration 0 → ideal outcome (warm-start was optimal)
- Pauli pool: Hamiltonian terms only (ZZ bonds + X sites)
- Use COBYLA or SPSA optimizer (gradient-free, noise-robust)

---

## CRITICAL PITFALLS (learned from V6.1 implementation)

### EstimatorV2 Observable Return Types — NEVER FORGET
- `(circuit, single_SparsePauliOp)` → returns **SCALAR** (the weighted sum)
- `(circuit, [list_of_SparsePauliOps])` → returns **ARRAY** (one value per op)
- For per-site ⟨X_i⟩ or per-bond ⟨Z_iZ_j⟩: ALWAYS submit as a LIST of individual single-term operators
- For total energy: submit the full Hamiltonian as a single SparsePauliOp (scalar is what you want)
- This applies to BOTH StatevectorEstimator AND IBM Runtime EstimatorV2

### Energy Computation — Don't Reconstruct Manually
- WRONG: `energy = -J * np.sum(zz_vals) - h * np.sum(x_vals)` (error-prone, ignores per-bond J)
- RIGHT: Submit the full Hamiltonian as a PUB → get energy directly from Estimator
- The Estimator handles the coefficient weighting correctly for any Hamiltonian structure

### Inhomogeneous ZNE — Two Types of CES
- **Topology CES** (`_compute_subset_ces`): sum of edge errors in the qubit subset's connectivity. Fast heuristic for RANKING candidate layouts. Does NOT account for routing overhead.
- **Circuit CES** (`compute_ces(transpiled)`): sum of 2Q gate errors in the actual transpiled circuit. This is the TRUE noise axis for ZNE extrapolation.
- Use topology CES for selection, circuit CES for extrapolation. Never mix them.

### NNConv — Use Sum Aggregation
- `aggr="add"` not `"mean"` — mean loses node degree information (Xu et al. 2019)
- This matters for lattices where sites have different coordination numbers

### Calibration Timestamp — May Be None
- Modern IBM backends (Target API) don't always expose `backend.properties().last_update_date`
- Default to assuming FRESH calibration when timestamp unavailable
- The error rates themselves ARE accessible via `backend.target[op_name].get(qargs).error`

### Phase Classification — Use Magnitudes
- ⟨X⟩ ≥ 0 always for TFIM with |+⟩^N initial state
- ⟨ZZ⟩ ≤ 0 for our convention (H = -J*ZZ - h*X)
- Compare `|⟨X⟩|` vs `|⟨ZZ⟩|` for crossover criterion
- Return "indeterminate" when difference < σ = 1/√shots

### Layout Selection — Seed for Reproducibility
- BFS-based subset search uses random starting nodes
- Always use a seeded `random.Random(seed)` instance, not module-level `random.sample()`
- This ensures reproducible layout selection across runs

### No Libraries Exist For
- Inhomogeneous ZNE (Uvarov 2024) — must implement ourselves
- Layout selection on heavy-hex topology — must implement ourselves
- Weight gradient analysis (Hernandes 2025) — must implement ourselves
- Mitiq does gate-folding ZNE only (different paradigm, not applicable)
- PEA local simulation — implemented in `noisy_utils.py` (IBM Runtime handles it on hardware)
- DD/twirling/TREX are native to Qiskit Runtime (just set options, no custom code)

---

## Do NOT
- Measure global fidelity on hardware (requires exponential tomography)
- Use more than p=2 total HVA layers (including ADAPT additions)
- Use Primitives V1 or `backend.run()`
- Hardcode h_c = 1.0 for phase classification (use data-driven crossover)
- Submit multi-term SparsePauliOp when you need per-term values
- Use `random.sample()` without a seed in layout selection
- Use `aggr="mean"` in NNConv (use `"add"`)
- Manually reconstruct energy from observables (submit Hamiltonian PUB instead)

---

## Noisy Simulation Mode (FakeTorino)

### Overview
`mode="fake_backend"` exercises the full ZNE pipeline locally using `FakeTorino` +
`BackendEstimatorV2`. No IBM credentials required. Validates that PEA-ZNE reduces
errors before real QPU deployment.

### Backend & Estimator
- Backend: `FakeTorino` from `qiskit_ibm_runtime.fake_provider` (133 qubits, heavy-hex)
- Estimator: `BackendEstimatorV2` from `qiskit.primitives` (local simulation)
- Uses `default_precision = 1/sqrt(shots)` parameter

### ZNE Strategy (2026-06-06 — post-refactoring)
- **Primary**: PEA-ZNE via `run_pea_zne()` — validated +94.4% gain, R²=0.998
- **Fallback**: Gate-folding ZNE via `run_gate_folding_zne()` — +20.6% gain
- **Adaptive**: Auto-selects GF first, falls back to PEA if R² < threshold
- **CES-ZNE**: DEPRECATED on heavy_hex (uniform CES≈0.15 → R²≈0.04)

### Post-ZNE Correction Stack
Applied automatically by `run_deployment()`:
1. **GNN-QEM** (optional) — Only if model loaded AND amplifier ≠ PEA. Confidence-gated.
2. **Affine correction** (always) — Clips to [E_ground, E_upper]. Zero cost.

### What's Included (same as hardware)
- Layout selection via BFS on heavy-hex topology (`select_layouts_low_ces`)
- Transpilation with `generate_preset_pass_manager(optimization_level=2)`
- PEA noise amplification (learned from FakeTorino calibration data)
- Observable grouping (commuting Paulis — 2 circuit executions total)
- TLS calibration drift monitoring (between h-points in sweeps)

### What's Excluded (isolates ZNE contribution)
- No DD/twirling/TREX (no effect on local simulation)
- No IBM server-side ZNE (only available on real hardware)

### Usage Pattern
```python
from qmbp_simulation.execution.hardware import HardwareBackend, HardwareConfig

# Fake backend (local PEA-ZNE simulation)
config = HardwareConfig(mode="fake_backend", n_qubits=10, shots=16384)
backend = HardwareBackend(config=config)
energy = backend.evaluate(circuit, H, params)  # PEA-ZNE mitigated energy
result = backend.run_deployment(circuit, H, params, h, e_exact, gap)  # Full pipeline
```

### Rehearsal (mandatory before real QPU)
```bash
# Full rehearsal (9 sections, ~60s)
python scripts/experiment_runners/run_hardware_rehearsal_v2.py

# Quick check (sections 1-3 only)
python scripts/experiment_runners/run_hardware_rehearsal_v2.py --section 1 2 3
```

### Deployment Script (real QPU)
```bash
# Tiered deployment on IBM Torino
python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py --dry-run
python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py --tier 0
python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py  # Full auto
```

---

## Hardware Module Integration with Runner Framework

### Runner Pattern for Hardware Scripts

All `scripts/run_hardware*.py` MUST use `HardwareValidationRunner`:

```python
from qmbp_simulation.framework.runner_base import HardwareValidationRunner, Section

class MyHWRunner(HardwareValidationRunner):
    runner_id = "hw_deploy_n10"
    experiment_id = "HW_DEPLOY"
    ...
```

### Automatic Validations (enforced by framework)

1. **Structural preflight** — runner_id, hypothesis, sections (from ValidationRunner).
2. **QPU preflight** — backend status, calibration, topology (from HardwareBackend).
3. **Cost ceiling check** — `shots × n_layouts ≤ max_total_shots` (in preflight.py).
4. **Circuit ZNE check** — 2Q gate count ≤ 18 (in run_deployment, before submission).
5. **Input validation** — params shape, gap>0, finite values (in run_deployment).
6. **Timeout handling** — job.result() respects job_timeout_s (in submission.py).

### CLI Arguments (HardwareValidationRunner)

```bash
--mode hardware|fake_backend    # Execution mode
--shots 16384                   # Shots per circuit
--n-layouts 3                   # Number of low-CES layouts
--n-qubits 10                   # System size
--topology heavy_hex            # Lattice topology
--zne-amplifier gate_folding|pea|adaptive  # ZNE noise amplification strategy
--zne-noise-factors 1 3 5      # Noise amplification factors
--zne-r2-threshold 0.90        # R² threshold for adaptive fallback
```

### Dual Persistence

Results are saved in TWO locations:
- `results/experiments/exp_{id}/run_{ts}.json` — digest/compare.py compatible
- `results/hardware/{runner_id}/run_{ts}/` — full provenance + input_params.json

### Key Rule: NEVER Skip Preflight for Real Hardware

- `--skip-preflight` is available for FakeTorino debugging only.
- For `--mode hardware`, preflight is MANDATORY — it prevents wasting IBM credits on misconfigured runs.
- If preflight aborts, fix the underlying issue. Do not bypass.
