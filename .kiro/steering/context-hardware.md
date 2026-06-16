---
inclusion: fileMatch
fileMatchPattern: "**/hardware/**,**/hardware_deployer*,scripts/run_hardware*,**/ibm_*deployment*"
---

# Hardware Context (invoke with #context-hardware)

> Pre-digested context for IBM Kingston/Boston deployment, hardware backend, and QPU execution.

## Status: ACTIVE QPU DEPLOYMENT (ibm_kingston)

- **First real QPU result (2026-06-14)**: Tier 0 completed on ibm_kingston (Heron r2).
- E_ZNE = -38.64, E_exact = -40.57, **ΔE/gap = 32.5% (FAIL)** with IBM default PEA (32×128).
- Root cause: PEA learning budget (4K shots) insufficient for 3.4% mean 2Q error.
- PEA calibration study in progress: testing balanced (48×192) and default+3layout configs.
- **Bugs fixed (2026-06-14)**:
  - `EstimatorV2(backend=...)` → `EstimatorV2(mode=...)` (qiskit-ibm-runtime 0.47.0 API change)
  - `_aggregate_qpu_metrics` TypeError: `sum()` on ISO timestamp strings → type-safe extraction
  - Jobs submitted without Batch → CANCELLED by IBM. Now uses `Batch(backend=backend)` context.
  - Observables job outside Batch → hung indefinitely. Now uses its own Batch.
  - Preflight 2Q threshold: relaxed from 1% → abort at 5%, warn at 3% (large chips have degraded outlier qubits that layout selection avoids).

## IBM Kingston Specs (Heron r2)

- 156 qubits, Heron r2, heavy-hex topology.
- Backend name: `"ibm_kingston"` (override: `--backend ibm_boston` for r3).
- Native gates: **cz, rzz**, id, rx, rz, sx, x. **RZZ is native** for our HVA.
- 2Q error median: 1.95×10⁻³. **Observed chip-wide mean: 3.36% (evening 2026-06-14).**
- T1 median: 258.88 μs. Min observed: 6.5 μs (isolated TLS defect, p5=125.9 μs).
- Channel: `"ibm_quantum_platform"` (qiskit-ibm-runtime 0.47.0).
- CLOPS: 3750 effective (measured via job metrics).

## ibm_boston (Heron r3, 156Q) — Premium Only

- **EPLG (100q): 2.15×10⁻³** — best error rates in fleet.
- 57/176 2Q gates below 10⁻³ error.
- Same topology (heavy-hex) and native gates as Kingston.
- **Access: Premium/Flex/PAYG plans only** — NOT available on Open Plan.
- If available, prefer over Kingston for thesis data (lower error → better PEA model → better ZNE).

## PEA Configuration (2026-06-14 calibration study)

**Critical lesson**: IBM default PEA (32×128 = 4K learning shots) is INSUFFICIENT
on processors with elevated calibration (>2% mean 2Q error).

### PEA Presets (CLI: `--pea-config <preset>`)

| Preset | num_rand | shots/rand | noise_factors | n_layouts | Learning shots | Status |
|--------|:--------:|:----------:|:-------------:|:---------:|:--------------:|--------|
| `default` | 32 | 128 | IBM default | 1 | 4,096 | ❌ ΔE/gap=32.5% |
| `balanced` | 48 | 192 | [1,1.5,3] | 1 | 9,216 | Pending |
| `aggressive` | 64 | 256 | [1,1.5,2,3] | 3 | 65,536 | ❌ >17min (cancelled) |
| `default_3layout` | 32 | 128 | IBM default | 3 | 4,096 | Pending |

Results stored in: `results/hardware/tier0_calibration/<pea_tag>/`
Binnacle: `documentation/binnacles/binnacle-hardware-pea-calibration.md`

## Connection Pattern

```python
import os
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2, Batch
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=os.environ["IBM_KEY"],
    instance=os.environ["IBM_INSTANCE_CRN"],
)
backend = service.backend("ibm_kingston")

# CRITICAL: Use Batch for multi-job submissions (avoids CANCELLED jobs)
with Batch(backend=backend) as batch:
    estimator = EstimatorV2(mode=batch)  # NOTE: 'mode=' not 'backend='
    # Apply options AFTER construction:
    estimator.options.default_shots = 16384
    estimator.options.resilience.zne_mitigation = True
    estimator.options.resilience.zne.amplifier = "pea"
    # Submit multiple PUBs within the batch
    job = estimator.run([(isa_circuit, hamiltonian)])
```

### API Version Notes (qiskit-ibm-runtime 0.47.0)
- `EstimatorV2(mode=backend)` — NOT `backend=`. Changed in ~0.40.
- `Batch(backend=backend)` — first positional arg IS still `backend`.
- `job.metrics()["timestamps"]["running"]` returns ISO string, NOT seconds.
- `job.status()` may return string or enum — handle both.

## Transpilation

```python
pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
isa_qc = pm.run(circuit.assign_parameters(theta))
isa_obs = [obs.apply_layout(isa_qc.layout) for obs in observables]
```

- PauliEvolutionGate gives 11% less 2Q-depth (same n_2Q=34).
- Level 3 / Rustiq provide NO benefit for HVA.

## Error Mitigation Stack (applied in order)

1. **Dynamical Decoupling** (free, always): `XpXm` sequence.
2. **Pauli Twirling** (32 randomizations): coherent → stochastic noise.
3. **TREX**: twirled readout error extinction.
4. **ZNE**: PEA primary, gate-folding fallback. See #context-zne-mitigation.
5. **Affine correction** (always): clips to [E_ground, E_upper].

## EstimatorV2 Configuration

```python
# CRITICAL: Use mode= (not backend=) for qiskit-ibm-runtime >= 0.40
estimator = EstimatorV2(mode=backend)  # or EstimatorV2(mode=batch) inside Batch context
estimator.options.default_shots = 16384
estimator.options.dynamical_decoupling.enable = True
estimator.options.dynamical_decoupling.sequence_type = "XpXm"
estimator.options.twirling.enable_gates = True
estimator.options.twirling.enable_measure = True
estimator.options.twirling.num_randomizations = 64  # Increased from 32
estimator.options.resilience.measure_mitigation = True  # TREX
estimator.options.resilience.zne_mitigation = True
estimator.options.resilience.zne.amplifier = "pea"
estimator.options.resilience.zne.noise_factors = [1, 1.5, 2, 3]  # Better for short circuits
estimator.options.resilience.layer_noise_learning.num_randomizations = 64  # 4× IBM default
estimator.options.resilience.layer_noise_learning.shots_per_randomization = 256
```

## Shot Budget

| N | Minimum shots | Recommended | σ (energy) |
|---|--------------|-------------|------------|
| 6 | 8192 | 16384 | 7.8e-3 |
| 10 | 8192 | 16384 | 7.8e-3 |
| 40 | 16384 | 32768 | 5.5e-3 |

## Observable Grouping

- ⟨X_i⟩: all commute → 1 circuit execution.
- ⟨Z_iZ_{i+1}⟩: all commute → 1 circuit execution.
- Total: 2 circuit executions per noise level (not N+N-1 separate).

## Success Criteria (hardware)

- **ΔE/gap < 5%** AND **correct phase label** — NOT fidelity.
- Phase classification: compare |⟨X⟩| vs |⟨ZZ⟩| for crossover.
- Return "indeterminate" when difference < σ = 1/√shots.

## Deployment Tiers (auto-advancing)

| Tier | What | h-points | Purpose |
|------|------|----------|---------|
| 0 | Single-point sanity | h=4.0 | Connection + basic mitigation |
| 1 | 3-point sweep | h=4.0, 3.25, 2.5 | PEA-ZNE validation |
| 2 | Full sweep + MPNN | 5+ h-points | End-to-end pipeline |
| 3 | Cross-topology | heavy_hex + chain | Generalization claim |

## Hardware Config (p=1 heavy-hex N=10)

- 1 restart (p=1 has single basin).
- 3 layouts (low-CES, BFS selection).
- 16k shots.
- h_test ≥ 3.25 (valid regime boundary).
- SPSA optimizer: a=0.1, c=0.05, A=10.
- Seed-independent (std=0.0003).

## CRITICAL PITFALLS

1. **EstimatorV2 returns**: single SparsePauliOp → SCALAR. List → ARRAY.
2. **Don't reconstruct energy manually**: submit full H as PUB → get energy directly.
3. **CES types**: topology CES for selection, circuit CES for extrapolation. Never mix.
4. **Calibration may be None**: modern Target API doesn't always expose `last_update_date`.
5. **Layout selection needs seed**: use `random.Random(seed)`, not module-level.

## DO NOT

- Measure global fidelity on hardware (exponential tomography).
- Use Primitives V1 or `backend.run()`.
- Skip preflight for real hardware runs (wastes IBM credits).
- Use p=2 at N≥10 (36 CX exceeds ZNE threshold).
- Use CES-ZNE on heavy_hex (uniform CES≈0.15, no spread).
- Submit multi-term SparsePauliOp when you need per-term values.

## Source Files

- #[[file:src/qmbp_simulation/execution/hardware.py]]
- #[[file:scripts/experiment_runners/hardware/run_ibm_torino_deployment.py]]
- #[[file:scripts/experiment_runners/run_hardware_rehearsal_v2.py]]
- #[[file:documentation/analysis/11_hardware_rehearsal_findings.md]]
- #[[file:documentation/analysis/18_ibm_hardware_generations.md]]
- #[[file:.kiro/steering/hardware-deployment.md]]
- #[[file:.kiro/steering/hardware-checklist.md]]
- #[[file:HARDWARE_DEPLOYMENT_SPEC.md]]
