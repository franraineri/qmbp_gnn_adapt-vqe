# Hardware Deployment Specification — IBM Torino QPU

**Fecha**: 2026-06-01
**Status**: Ready for execution (all local validation complete)
**Autor**: GNN-HVA Framework Team

---

## 1. Executive Summary

This document specifies the exact configuration for deploying the GNN-HVA pipeline
on IBM Torino quantum hardware. Every parameter has been validated through local
simulation (308+ noiseless runs, 90+ noisy runs, 17 V8 experiments).

**Target**: Demonstrate ΔE/gap < 5% on real quantum hardware for TFIM phase
characterization at N=10 qubits using the GNN-predicted warm-start.

---

## 2. Hardware Target

| Property | Value | Notes |
|----------|-------|-------|
| Backend | `ibm_torino` | Eagle r3, 133 qubits |
| Topology | Heavy-hex | z=3, bipartite |
| Native gate | ECR (2-qubit) | FakeTorino uses CZ |
| Connectivity | Fixed-frequency transmons | No tunable couplers |
| 2Q gate error | ~0.3-0.5% | Per calibration data |
| Readout error | ~1-3% | Mitigated by TREX |

**Alternative**: If IBM Heron r2 (`ibm_kingston`, 156q) is available, prefer it
for lower 2Q gate error (~0.1-0.2%). Same heavy-hex topology.

---

## 3. Circuit Configuration (Validated)

| Parameter | Value | Evidence |
|-----------|-------|----------|
| **Ansatz** | HVA p=1 | 3/3 seeds pass, ΔE/gap=0.56% |
| **N qubits** | 10 | Validated locally |
| **Topology** | heavy_hex | Zero SWAP overhead (CONFIRMED) |
| **Parameters** | 2 (θ_zz, θ_x) | From MPNN prediction |
| **Initial state** | \|+⟩^N | H gates on all qubits |
| **2Q gates** | 18 CZ/ECR | 9 edges × 2 (RZZ decomposition) |
| **Circuit depth** | 62 (transpiled) | From FakeTorino test |
| **SWAP gates** | 0 | **CONFIRMED** via transpilation test |

### Transpilation Verification

```
Original: 10 qubits, 2 params, 9 RZZ + 10 RX + 10 H
Transpiled (FakeTorino, opt_level=2): 18 CZ, 68 RZ, 57 SX, 9 X
SWAP gates: 0 ✅
```

---

## 4. Measurement Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Shots** | 16384 | EXT-4: 32k gives identical results (noise is layout-dominated) |
| **Precision** | ~0.0078 (1/√16384) | Below ⟨X⟩ signal (~8e-3 at N=10) |
| **Layouts** | 3 | EXT-3: 5 layouts gives only +3% marginal gain |
| **Observables** | Energy (full H) + per-site ⟨X_i⟩ + per-bond ⟨Z_iZ_j⟩ | 2 measurement bases |

### Observable Submission Pattern

```python
# Energy: submit full Hamiltonian as single PUB → scalar result
energy_pub = (isa_circuit, hamiltonian.apply_layout(isa_circuit.layout))

# Per-site observables: submit as LIST of single-term ops → array result
x_ops = [SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=N)
         for i in range(N)]
obs_pub = (isa_circuit, [op.apply_layout(isa_circuit.layout) for op in x_ops])
```

---

## 5. Error Mitigation Stack

Applied in this order (each layer builds on the previous):

### Layer 1: Dynamical Decoupling (free, always enable)
```python
options.dynamical_decoupling.enable = True
options.dynamical_decoupling.sequence_type = "XpXm"
```
- Suppresses idle-time decoherence
- No shot overhead
- XpXm is robust to pulse calibration errors

### Layer 2: Pauli Twirling (converts coherent → stochastic noise)
```python
options.twirling.enable_gates = True
options.twirling.enable_measure = True
options.twirling.num_randomizations = 32
options.twirling.shots_per_randomization = 512  # 32×512 = 16384 total
```
- Makes noise stochastic (prerequisite for ZNE/TREX to work correctly)
- 32 randomizations is standard (IBM default)

### Layer 3: TREX (readout error mitigation)
```python
options.resilience.measure_mitigation = True
```
- Twirled Readout Error eXtinction
- Corrects measurement bit-flip errors
- Minimal overhead (~1.5×)

### Layer 4: Inhomogeneous ZNE (our custom implementation)
```python
# NOT using IBM's built-in ZNE (gate folding)
# Instead: 3 different transpilation layouts → different CES → linear extrapolation
options.resilience.zne_mitigation = False  # Disable built-in

# Our ZNE: transpile with 3 different initial_layouts
for layout in selected_layouts:
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=2, initial_layout=layout
    )
    isa_circuits.append(pm.run(bound_circuit))
# Then: linear_fit(energies, ces_values) → extrapolate to CES=0
```

**Why not IBM's built-in ZNE?** IBM uses gate folding (noise factors [1,2,3]) which
increases circuit depth. Our inhomogeneous ZNE uses different qubit mappings at the
SAME depth — better for shallow circuits where depth increase is catastrophic.

---

## 6. VQE Strategy on Hardware

### Primary approach: NO VQE on hardware

The MPNN predicts θ_opt with ΔE/gap=0.56% locally. On hardware, we:
1. Use θ_opt directly from MPNN prediction
2. Evaluate energy at that point
3. Compare against exact (known from local simulation)

This avoids the cost and noise of hardware VQE entirely.

### Fallback: SPSA refinement (only if ΔE/gap > 5%)

```python
spsa_config = {
    "a": 0.1,
    "c": 0.05,
    "A": 10,  # A = 0.05 × n_iterations
    "n_iterations": 200,
    "alpha": 0.602,
    "gamma": 0.101,
}
```
- Source: V7-4A grid search (36 configs × 10 seeds)
- 2 circuit evaluations per iteration → 400 total evaluations
- At 16384 shots each → ~6.5M total shots for refinement
- Estimated time: ~10-15 min on hardware

**Rule**: Do NOT apply SPSA if initial ΔE/gap < 5% (V7-4B: refinement HURTS
good warm-starts by -146%).

---

## 7. Test Points

| # | h_test | Purpose | Expected ΔE/gap | Risk |
|---|--------|---------|-----------------|------|
| 1 | 4.0 | Smoke test (trivial) | <0.1% | None |
| 2 | 3.25 | **Primary thesis target** | ~0.6% | Low |
| 3 | 3.0 | Boundary probe | ~3% | Medium (at boundary) |
| 4 | 2.5 | Below valid regime | >10% (expected fail) | Documents limit |

**Execution order**: 1 → 2 → 3 → 4 (easiest first, validates pipeline before harder points)

---

## 8. Success Criteria

| Criterion | Threshold | Priority |
|-----------|-----------|----------|
| ΔE/gap < 5% at h=3.25 | PRIMARY | Must pass |
| Correct phase label (paramagnetic) | PRIMARY | Must pass |
| ZNE R² > 0.8 | SECONDARY | Should pass (R²=0.998 in simulation) |
| ZNE gain > 0% | SECONDARY | Should pass (+76% in simulation) |
| No "indeterminate" phase at h≥3.0 | TERTIARY | Expected |

---

## 9. Execution Protocol

### Pre-flight (before submitting jobs)
```bash
# 1. Verify credentials
export IBM_KEY=<your_token>
export IBM_INSTANCE_CRN=<your_crn>

# 2. Check backend status
python -c "
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(channel='ibm_quantum_platform')
backend = service.backend('ibm_torino')
print(f'Status: {backend.status()}')
print(f'Queue: {backend.status().pending_jobs} pending jobs')
"

# 3. Verify calibration freshness
python -c "
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(channel='ibm_quantum_platform')
backend = service.backend('ibm_torino')
# Check 2Q gate errors on our target qubits
target = backend.target
for gate_name in ['ecr', 'cz']:
    if gate_name in target.operation_names:
        errors = [target[gate_name][qargs].error for qargs in target.qargs
                  if gate_name in target.operation_names_for_qargs(qargs)]
        print(f'{gate_name}: mean error = {sum(errors)/len(errors):.4f}')
"
```

### Execution
```python
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
from qiskit_ibm_runtime.options import EstimatorOptions

service = QiskitRuntimeService(channel="ibm_quantum_platform")
backend = service.backend("ibm_torino")

options = EstimatorOptions()
options.dynamical_decoupling.enable = True
options.dynamical_decoupling.sequence_type = "XpXm"
options.twirling.enable_gates = True
options.twirling.enable_measure = True
options.twirling.num_randomizations = 32
options.twirling.shots_per_randomization = 512
options.resilience.measure_mitigation = True
options.resilience.zne_mitigation = False  # We do our own inhomogeneous ZNE

estimator = EstimatorV2(mode=backend, options=options)

# Submit PUBs for each layout
for layout_idx, isa_circuit in enumerate(isa_circuits):
    pub = (isa_circuit, hamiltonian_isa)
    job = estimator.run([pub])
    # Save job ID for provenance
    print(f"Layout {layout_idx}: job_id={job.job_id()}")
```

### Post-run analysis
```bash
# 1. Collect results and compute ZNE extrapolation
# 2. Compare against noiseless prediction
# 3. Log to binnacle
# 4. Update project-status.md with hardware results
```

---

## 10. Cost Estimate

| Component | Shots | Circuits | Total evaluations |
|-----------|-------|----------|-------------------|
| Energy (3 layouts × 1 h-point) | 16384 | 3 | 49,152 shots |
| Observables (3 layouts × 1 h-point) | 16384 | 3 | 49,152 shots |
| **Per h-point total** | | | **~98k shots** |
| **4 h-points** | | | **~393k shots** |
| SPSA refinement (if needed) | 16384 | 400 | ~6.5M shots |

Estimated wall-clock time (including queue):
- Without SPSA: ~20-30 min (4 h-points × 3 layouts × ~2 min each)
- With SPSA: +15 min

---

## 11. Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Queue congestion (>50 jobs) | Medium | Execute UTC 2-6 AM |
| Calibration drift mid-run | Low | All 4 h-points in single session |
| Layout maps to bad qubits | Medium | Use `select_layouts_low_ces()` with CES<3× median filter |
| ΔE/gap > 5% at h=3.25 | Low (0.6% in sim) | Activate SPSA refinement |
| ZNE R² < 0.8 | Very low (0.998 in sim) | Increase to 5 layouts |
| Connection timeout | Low | Retry after 30 min |
| Job fails/errors | Low | Save all job IDs, retry failed jobs |

---

## 12. Validation Evidence (Local Simulation)

| Claim | Evidence | Confidence |
|-------|----------|------------|
| p=1 heavy-hex passes | 3/3 seeds, ΔE/gap=0.006, std=0.0003 | ★★★ |
| ZNE works on heavy-hex | 3/3 seeds, gain=+63%, R²=0.998 | ★★★ |
| Zero SWAP transpilation | FakeTorino test: 0 SWAPs, 18 CZ | ★★★ |
| 16k shots sufficient | EXT-4: 32k identical to 16k | ★★★ |
| 3 layouts sufficient | EXT-3: 5 layouts +3% marginal only | ★★★ |
| 1 restart sufficient | EXT-6: ΔE/gap=0.006 with 1 restart | ★★★ |
| Valid regime h≥3.0 | EXT-5: h=2.625 fails catastrophically | ★★★ |
| p=2 unrescuable | EXT-7: 5 layouts still fails (-27%) | ★★★ |
| SPSA config optimal | V7-4A: 36×10 grid search | ★★★ |
| SPSA hurts good warm-start | V7-4B: -146% at h=2.0 | ★★★ |

---

## 13. Files & Scripts

| Purpose | File |
|---------|------|
| Heavy-hex variant runner | `scripts/experiment_runners/run_thesis_variants-heavy_hex.py` |
| Noisy pipeline (ZNE) | `scripts/experiment_runners/experiment_run_helpers_CHECK/run_noisy_pipeline.py` |
| Noiseless pipeline | `scripts/experiment_runners/experiment_run_helpers_CHECK/run_pipeline.py` |
| Hardware backend (stub) | `src/qmbp_simulation/execution/backends.py` → `HardwareBackend` |
| Layout selector | `src/qmbp_simulation/execution/` → `select_layouts_low_ces()` |
| Preflight checker | `src/qmbp_simulation/framework/preflight.py` |
| Coverage scanner | `analysis/scan_coverage.py` |
| Hardware deployment guide | `.kiro/steering/hardware-deployment.md` |
| Hardware checklist | `.kiro/steering/hardware-checklist.md` |
| Optimization reference | `.kiro/knowledge/optimization-hardware.md` |

---

## 14. What This Document Does NOT Cover

- IBM account setup and billing
- Qiskit Runtime session management (batching multiple jobs)
- Real-time calibration monitoring
- Post-hardware thesis writing
- Comparison with other quantum computing frameworks

These are operational concerns to be addressed during execution, not design decisions.
