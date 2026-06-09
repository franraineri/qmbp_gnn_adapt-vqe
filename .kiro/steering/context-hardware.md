---
inclusion: fileMatch
fileMatchPattern: "**/hardware/**,**/hardware_deployer*,scripts/run_hardware*,**/ibm_torino*"
---

# Hardware Context (invoke with #context-hardware)

> Pre-digested context for IBM Torino deployment, hardware backend, and QPU execution.

## Status: READY FOR QPU

- Rehearsal V2 passed 3/3 after fixes (2026-06-06).
- Hardware module: credential passing, TLS drift monitoring, QPU metrics, EstimatorV2 options.
- Deployment script: `scripts/experiment_runners/hardware/run_ibm_torino_deployment.py`.
- Only IBM credentials needed (`IBM_KEY`, `IBM_INSTANCE_CRN`).

## IBM Torino Specs

- 133 qubits, Eagle r3, heavy-hex topology.
- Backend name: `"ibm_torino"`.
- Channel: `"ibm_quantum_platform"`.
- Calibration: accessible via `backend.target[op_name].get(qargs).error`.

## Connection Pattern

```python
import os
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=os.environ["IBM_KEY"],
    instance=os.environ["IBM_INSTANCE_CRN"],
)
backend = service.backend("ibm_torino")
```

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
estimator = EstimatorV2(mode=backend)
estimator.options.dynamical_decoupling.enable = True
estimator.options.dynamical_decoupling.sequence_type = "XpXm"
estimator.options.twirling.enable_gates = True
estimator.options.resilience.zne_mitigation = True
estimator.options.resilience.zne.noise_factors = [1, 2, 3]
estimator.options.resilience.zne.amplifier = "pea"
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
