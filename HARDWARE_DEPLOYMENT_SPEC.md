# Hardware Deployment Specification — IBM Torino QPU

**Version**: 1.1
**Fecha**: 2026-06-01
**Status**: Ready for execution (all local validation complete)
**Autor**: GNN-HVA Framework Team

> **Maintainability note**: This spec is the single source of truth for hardware
> deployment parameters. If any value here conflicts with code or steering files,
> this spec takes precedence. Update this document FIRST, then propagate to code.

---

## 1. Executive Summary

This document specifies the exact configuration for deploying the GNN-HVA pipeline
on IBM Torino quantum hardware. Every parameter has been validated through local
simulation (210+ pipeline runs across 5 topologies, 69+ noisy/ZNE runs, 21 formal experiments).

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

**Fallback backends** (if ibm_torino unavailable — verify names via `service.backends()`):
- `ibm_brisbane` (Eagle r3, 127q) — same topology, slightly older calibration
- `ibm_osaka` (Eagle r3, 127q) — same topology
- Any Eagle r3 or Heron r2 with heavy-hex topology works

**Calibration quality gate**: Do NOT proceed if mean 2Q gate error > 1.0% on target
qubits. Check at pre-flight time.

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
| **Circuit depth** | ~60-65 (transpiled) | Varies with calibration; measured 62 on FakeTorino |
| **SWAP gates** | 0 | **CONFIRMED** via transpilation test |

### Transpilation Verification

```
Original: 10 qubits, 2 params, 9 RZZ + 10 RX + 10 H
Transpiled (FakeTorino, opt_level=2): 18 CZ, 68 RZ, 57 SX, 9 X
SWAP gates: 0 ✅
Note: Exact gate counts may vary with Qiskit version and calibration data.
```

---

## 4. Measurement Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Shots** | 16384 | EXT-4: 32k gives identical results (noise is layout-dominated) |
| **Precision** | ~0.0078 (1/√16384) | Sufficient for phase discrimination near h_c (signal ~8e-3). At h=3.25 (primary target), ⟨X⟩≈0.97 — signal is 120× above noise floor. |
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
# Shot distribution across randomizations is handled internally by the runtime.
# Total shots controlled via options.default_shots (see §9).
```
- Makes noise stochastic (prerequisite for ZNE/TREX to work correctly)
- 32 randomizations is standard (IBM default)
- Runtime automatically divides total shots across randomizations

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

### Required Software Versions
```
qiskit>=1.4,<2
qiskit-ibm-runtime>=0.20
```
Pin exact versions at execution time and record in provenance metadata.

### Pre-flight (before submitting jobs)
```bash
# 1. Verify credentials
export IBM_KEY=<your_token>
export IBM_INSTANCE_CRN=<your_crn>

# 2. Check backend status and queue
python -c "
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(channel='ibm_quantum_platform')
backend = service.backend('ibm_torino')
print(f'Status: {backend.status()}')
print(f'Queue: {backend.status().pending_jobs} pending jobs')
# Abort if queue > 50 jobs (execute UTC 2-6 AM instead)
"

# 3. Verify calibration quality (GATE: abort if mean 2Q error > 1.0%)
python -c "
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(channel='ibm_quantum_platform')
backend = service.backend('ibm_torino')
target = backend.target
for gate_name in ['ecr', 'cz']:
    if gate_name in target.operation_names:
        errors = [target[gate_name][qargs].error for qargs in target.qargs
                  if gate_name in target.operation_names_for_qargs(qargs)]
        mean_err = sum(errors)/len(errors)
        print(f'{gate_name}: mean error = {mean_err:.4f}')
        if mean_err > 0.01:
            print('⚠️  ABORT: 2Q error exceeds 1.0% threshold. Wait for recalibration.')
"

# 4. Record Qiskit versions for provenance
python -c "
import qiskit; print(f'qiskit={qiskit.__version__}')
import qiskit_ibm_runtime; print(f'qiskit-ibm-runtime={qiskit_ibm_runtime.__version__}')
"
```

### Execution (using Session for batched scheduling)
```python
import json
from datetime import datetime, timezone
from pathlib import Path
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2, Session
from qiskit_ibm_runtime.options import EstimatorOptions

service = QiskitRuntimeService(channel="ibm_quantum_platform")
backend = service.backend("ibm_torino")

# --- Options ---
options = EstimatorOptions()
options.default_shots = 16384  # Total shots per PUB
options.dynamical_decoupling.enable = True
options.dynamical_decoupling.sequence_type = "XpXm"
options.twirling.enable_gates = True
options.twirling.enable_measure = True
options.twirling.num_randomizations = 32
options.resilience.measure_mitigation = True
options.resilience.zne_mitigation = False  # We do our own inhomogeneous ZNE

# --- Provenance metadata ---
provenance = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "backend": "ibm_torino",
    "qiskit_version": qiskit.__version__,
    "runtime_version": qiskit_ibm_runtime.__version__,
    "options": {
        "shots": 16384,
        "dd": "XpXm",
        "twirling_randomizations": 32,
        "trex": True,
        "zne_builtin": False,
    },
    "layouts_used": [],
    "job_ids": [],
}

# --- Submit within a Session (batched scheduling, priority access) ---
with Session(service=service, backend=backend) as session:
    estimator = EstimatorV2(mode=session, options=options)

    for layout_idx, isa_circuit in enumerate(isa_circuits):
        pub = (isa_circuit, hamiltonian_isa)
        job = estimator.run([pub])
        job_id = job.job_id()
        provenance["job_ids"].append(job_id)
        provenance["layouts_used"].append(selected_layouts[layout_idx])
        print(f"Layout {layout_idx}: job_id={job_id}")

# --- Save provenance ---
output_dir = Path("results/thesis/hardware_deployment")
output_dir.mkdir(parents=True, exist_ok=True)
provenance_path = output_dir / f"provenance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(provenance_path, "w") as f:
    json.dump(provenance, f, indent=2)
print(f"Provenance saved: {provenance_path}")
```

### Post-run analysis (fully reproducible from saved data)
```python
import json
import numpy as np
from pathlib import Path

output_dir = Path("results/thesis/hardware_deployment")
run_dir = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run_dir.mkdir(parents=True, exist_ok=True)

# 1. Retrieve results from all jobs
raw_data = {"energies_per_layout": [], "stds_per_layout": [], "ces_per_layout": []}
for i, job_id in enumerate(provenance["job_ids"]):
    job = service.job(job_id)
    job.wait_for_final_state(timeout=600)
    if job.status().name != "DONE":
        print(f"⚠️  Job {job_id} status: {job.status()} — skipping")
        continue
    result = job.result()
    ev = float(result[0].data.evs)
    std = float(result[0].data.stds)
    raw_data["energies_per_layout"].append(ev)
    raw_data["stds_per_layout"].append(std)
    # CES was computed during layout selection (save it)
    raw_data["ces_per_layout"].append(ces_values[i])

# 2. ZNE extrapolation (linear fit: E(CES) → E(0))
energies = np.array(raw_data["energies_per_layout"])
ces = np.array(raw_data["ces_per_layout"])
coeffs = np.polyfit(ces, energies, 1)
e_zne = float(coeffs[1])  # Extrapolated to CES=0
r2 = 1 - np.sum((energies - np.polyval(coeffs, ces))**2) / np.sum((energies - energies.mean())**2)
gain = (abs(energies[-1] - e_exact) - abs(e_zne - e_exact)) / abs(energies[-1] - e_exact)

zne_analysis = {
    "e_zne": e_zne, "r_squared": float(r2), "gain": float(gain),
    "slope": float(coeffs[0]), "intercept": float(coeffs[1]),
    "energies": energies.tolist(), "ces": ces.tolist(),
}

# 3. Compute ΔE/gap and phase label
delta_e_gap = abs(e_zne - e_exact) / gap
phase_label = "paramagnetic" if abs(mag_x_mean) > abs(corr_zz_mean) else "ordered"
passed = delta_e_gap < 0.05 and phase_label == "paramagnetic"

summary = {
    "h_test": h_test, "e_exact": e_exact, "e_zne": e_zne,
    "delta_e_gap": float(delta_e_gap), "phase_label": phase_label,
    "zne_r2": float(r2), "zne_gain": float(gain),
    "verdict": "PASS" if passed else "FAIL",
}

# 4. Save everything
with open(run_dir / "provenance.json", "w") as f:
    json.dump(provenance, f, indent=2)
with open(run_dir / "raw_results.json", "w") as f:
    json.dump(raw_data, f, indent=2)
with open(run_dir / "zne_analysis.json", "w") as f:
    json.dump(zne_analysis, f, indent=2)
with open(run_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print(f"  h={h_test}: ΔE/gap={delta_e_gap:.4f} [{summary['verdict']}]")
print(f"  ZNE: R²={r2:.4f}, gain={gain:+.1%}")
print(f"  Phase: {phase_label}")
print(f"  Saved to: {run_dir}")
print(f"{'='*60}")
```

### Timeout & Error Handling
- If a job does not reach DONE within 10 minutes: log job_id, skip, continue with remaining jobs.
- If >1 job fails: abort session, wait 30 min, retry with fresh session.
- Maximum retry attempts: 3 per session.
- **Cost ceiling**: Do not exceed 10M total shots (≈ SPSA + 4 h-points + retries). Abort if approaching.

---

## 10. Cost Estimate

| Component | Shots | Circuits | Total evaluations |
|-----------|-------|----------|-------------------|
| Energy (3 layouts × 1 h-point) | 16384 | 3 | 49,152 shots |
| Observables (3 layouts × 1 h-point) | 16384 | 3 | 49,152 shots |
| **Per h-point total** | | | **~98k shots** |
| **4 h-points** | | | **~393k shots** |
| SPSA refinement (if needed) | 16384 | 400 | ~6.5M shots |

Estimated wall-clock time (using Session mode):
- Without SPSA: ~10-15 min (Session eliminates per-job queue waits)
- With SPSA: +15 min
- Total budget ceiling: 10M shots maximum

---

## 11. Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Queue congestion (>50 jobs) | Medium | Execute UTC 2-6 AM; use Session mode |
| Calibration drift mid-run | Low | All 4 h-points in single Session |
| Layout maps to bad qubits | Medium | `select_layouts_low_ces(bound_circuit, backend, candidates, n_select=3, max_ces=0.5)` |
| ΔE/gap > 5% at h=3.25 | Low (0.6% in sim) | Activate SPSA refinement |
| ZNE R² < 0.8 | Very low (0.998 in sim) | Increase to 5 layouts |
| Connection timeout | Low | Retry after 30 min (max 3 retries) |
| Job fails/errors | Low | Save all job IDs, retry failed jobs within Session |
| Backend unavailable | Low | Fallback to ibm_brisbane or ibm_osaka |
| Cost overrun | Very low | Hard ceiling at 10M shots |

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

## 13. Provenance & Reproducibility

Every hardware run MUST save the following metadata:

| Field | Source | Purpose |
|-------|--------|---------|
| `job_ids` | Runtime API | Retrieve raw results later |
| `backend_name` | `backend.name` | Identify hardware |
| `calibration_timestamp` | `backend.target` properties | Detect calibration drift |
| `qiskit_version` | `qiskit.__version__` | Reproduce transpilation |
| `runtime_version` | `qiskit_ibm_runtime.__version__` | Reproduce options behavior |
| `layouts_used` | `select_layouts_low_ces()` output | Reproduce qubit mapping |
| `options_snapshot` | Full EstimatorOptions dict | Reproduce mitigation stack |
| `raw_evs` | `job.result()[0].data.evs` | Raw expectation values |
| `raw_stds` | `job.result()[0].data.stds` | Standard errors |
| `ces_values` | `compute_circuit_ces()` per layout | ZNE extrapolation axis |
| `execution_timestamp` | UTC ISO format | Temporal ordering |

**Storage**: `results/thesis/hardware_deployment/run_YYYYMMDD_HHMMSS/`
- `provenance.json` — all metadata above
- `raw_results.json` — per-layout raw EVs and stds
- `zne_analysis.json` — extrapolation results, R², gain
- `summary.json` — final ΔE/gap, phase label, pass/fail

---

## 14. Files & Scripts

| Purpose | File |
|---------|------|
| Heavy-hex variant runner | `scripts/experiment_runners/run_thesis_variants-heavy_hex.py` |
| Noisy pipeline (ZNE) | `scripts/experiment_runners/experiment_run_helpers_CHECK/run_noisy_pipeline.py` |
| Noiseless pipeline | `scripts/experiment_runners/experiment_run_helpers_CHECK/run_pipeline.py` |
| Hardware backend (stub) | `src/qmbp_simulation/execution/backends.py` → `HardwareBackend` |
| Layout selector | `src/qmbp_simulation/execution/noisy_utils.py` → `select_layouts_low_ces()` |
| CES computation | `src/qmbp_simulation/execution/noisy_utils.py` → `compute_circuit_ces()` |
| Preflight checker | `src/qmbp_simulation/framework/preflight.py` |
| Coverage scanner | `analysis/scan_coverage.py` |
| Hardware deployment guide | `.kiro/steering/hardware-deployment.md` |
| Hardware checklist | `.kiro/steering/hardware-checklist.md` |
| Optimization reference | `.kiro/knowledge/optimization-hardware.md` |

---

## 15. What This Document Does NOT Cover

- IBM account setup and billing
- Real-time calibration monitoring dashboards
- Post-hardware thesis writing
- Comparison with other quantum computing frameworks
- AdaptVQE on hardware (deprecated — MPNN prediction is sufficient, no VQE needed)

These are operational concerns to be addressed during execution, not design decisions.

---

## 16. Scalability Plan — Beyond the Initial Run

This spec targets the **minimum viable hardware demonstration** (N=10, p=1, 4 h-points).
The following extensions are pre-validated and can be executed by changing parameters only:

### Extension A: Additional h-points (same circuit)
- Add h_test ∈ {3.5, 3.75, 4.5} for denser coverage
- Same circuit, same layouts, same mitigation → just more PUBs per Session
- Cost: +98k shots per additional h-point

### Extension B: N=6 p=2 (validation baseline)
- Known to work in simulation (ΔE/gap < 1%, ZNE gain=+48.5%)
- Same heavy-hex topology, 18 CZ gates (same CX budget as N=10 p=1)
- Validates that simulation→hardware transfer is consistent across configs
- Change: `n_qubits=6, p_layers=2, h_test=[1.5, 1.75, 2.0]`

### Extension C: Multiple seeds (statistical robustness)
- Run h=3.25 with seeds 42, 43, 44 (different MPNN predictions)
- Validates seed-independence on hardware (confirmed locally: std=0.0003)
- Cost: 3× per h-point (3 different θ_opt values)

### Extension D: N=20 p=1 (scaling demonstration)
- Requires: MPS-based VQE for ground truth (already validated, chi=64)
- Circuit: 38 CZ gates (at ZNE boundary — may need 5 layouts)
- Valid regime: h≥2.25
- **Risk**: 38 CX may exceed ZNE perturbative regime. Test with 5 layouts first.
- Pre-validate locally with FakeTorino before submitting to real hardware.

### Extension E: Different backend (portability)
- All code uses `backend = service.backend(BACKEND_NAME)` — single variable change
- Layout selection adapts automatically (BFS on backend topology)
- Transpilation adapts automatically (pass manager uses backend target)
- Only constraint: backend must have heavy-hex topology with ≥10 connected qubits

### Scaling Invariants (must hold for ANY extension)
1. Always use `select_layouts_low_ces()` for layout selection (never manual)
2. Always save full provenance (§13) — no exceptions
3. Always verify R² > 0.8 before trusting ZNE results
4. Always compare against known E_exact (never trust hardware energy alone)
5. Never apply SPSA if initial ΔE/gap < 5%
6. Never exceed 18 CZ gates without first validating ZNE locally at that depth

---

## 17. Reproducibility Checklist

Before declaring a hardware run "complete", verify ALL of the following:

- [ ] `provenance.json` saved with all fields from §13
- [ ] `raw_results.json` contains per-layout EVs and stds
- [ ] `zne_analysis.json` contains R², gain, extrapolated energy
- [ ] `summary.json` contains ΔE/gap, phase label, pass/fail verdict
- [ ] Qiskit versions recorded match those used for local validation
- [ ] Backend calibration timestamp is within 24h of execution
- [ ] All job IDs are retrievable via `service.job(job_id)`
- [ ] ZNE R² > 0.8 (if not, document why and whether results are trustworthy)
- [ ] Results logged in `documentation/binnacles/binnacle-hardware-deployment.md`
- [ ] At least one h-point matches local simulation within 2× expected ΔE/gap

### To reproduce this run from scratch:
```bash
# 1. Install exact versions
pip install qiskit==<version_from_provenance> qiskit-ibm-runtime==<version_from_provenance>

# 2. Load provenance
python -c "
import json
prov = json.load(open('results/thesis/hardware_deployment/run_XXXXXXXX/provenance.json'))
print(f'Backend: {prov[\"backend\"]}')
print(f'Layouts: {prov[\"layouts_used\"]}')
print(f'Jobs: {prov[\"job_ids\"]}')
"

# 3. Retrieve raw results (jobs persist on IBM servers for 30 days)
python -c "
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(channel='ibm_quantum_platform')
job = service.job('<job_id_from_provenance>')
result = job.result()
print(result[0].data.evs)  # Raw expectation values
"

# 4. Recompute ZNE extrapolation from raw data (deterministic)
python -c "
import json, numpy as np
raw = json.load(open('results/thesis/hardware_deployment/run_XXXXXXXX/raw_results.json'))
energies = np.array(raw['energies_per_layout'])
ces = np.array(raw['ces_per_layout'])
# Linear fit: E(CES) = a*CES + b → E(0) = b
coeffs = np.polyfit(ces, energies, 1)
e_zne = coeffs[1]  # Extrapolated to CES=0
r2 = 1 - np.sum((energies - np.polyval(coeffs, ces))**2) / np.sum((energies - energies.mean())**2)
print(f'E_ZNE = {e_zne:.6f}, R² = {r2:.4f}')
"
```

---

## 18. Maintenance & Update Protocol

This spec should be updated when:
1. **Qiskit API changes** — verify options names, Session API, EstimatorV2 interface
2. **Backend retires** — update fallback list in §2
3. **New validation data** — if additional local runs change optimal parameters
4. **After first hardware run** — add "Lessons Learned" appendix with actual vs expected

### Version History
| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-06-01 | Initial spec (all local validation complete) |
| 1.1 | 2026-06-01 | Added: Session mode, provenance, scalability plan, reproducibility checklist, fallback backends, calibration gate, cost ceiling |
