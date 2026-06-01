# Hardware Execution Module — IBM Torino QPU

This module implements real quantum hardware execution via IBM Runtime,
integrated into the `ExecutionBackend` ABC pattern. It supports both
real IBM hardware and local FakeTorino simulation for validation.

## Architecture

```
execution/hardware/
├── backend.py       # HardwareBackend class (evaluate, run_deployment, run_h_sweep)
├── config.py        # HardwareConfig, SPSAConfig, HardwareRunResult
├── preflight.py     # Pre-execution checks (status, calibration, topology)
├── submission.py    # Job submission with retry logic + layout selection
├── observables.py   # Per-site observable construction and extraction
├── phase.py         # Phase classification (paramagnetic/ordered/indeterminate)
├── spsa.py          # Conditional SPSA refinement
└── persistence.py   # Result saving with full provenance
```

## Pipeline Flow

```
MPNN θ_opt → HardwareBackend.evaluate(circuit, H, params)
                │
                ├─ 1. Bind parameters to HVA circuit
                ├─ 2. Select 3 lowest-CES layouts (cached)
                ├─ 3. Transpile circuit to each layout
                ├─ 4. Submit all 3 energy PUBs (parallel)
                ├─ 5. Collect results + discard NaN/Inf
                ├─ 6. Linear ZNE extrapolation → E(CES=0)
                └─ return: float (ZNE-extrapolated energy)
```

For the full deployment pipeline with phase classification and persistence:

```
HardwareBackend.run_deployment(circuit, H, params, h, e_exact, gap)
                │
                ├─ 1. Preflight checks (status, calibration, topology)
                ├─ 2. evaluate() → ZNE energy
                ├─ 3. Measure per-site ⟨X_i⟩ and ⟨Z_iZ_j⟩
                ├─ 4. Classify phase: |⟨X⟩| vs |⟨ZZ⟩|
                ├─ 5. Compute ΔE/gap
                ├─ 6. Conditional SPSA (only if ΔE/gap > 5%)
                ├─ 7. Save all results + logs to disk
                └─ return: HardwareRunResult
```

## Quick Start

### 1. Local Validation (FakeTorino — no credentials needed)

```python
import numpy as np
from qmbp_simulation.execution.hardware import HardwareBackend, HardwareConfig
from qmbp_simulation.models import HamiltonianBuilder, make_lattice
from qmbp_simulation.circuits import HVACircuitBuilder

# Configure for local validation
config = HardwareConfig(
    mode="fake_backend",
    n_qubits=10,
    shots=16384,
    n_layouts=3,
    output_dir="results/hardware",
)

# Build circuit and Hamiltonian
lattice = make_lattice("heavy_hex", 10)
circuit, _ = HVACircuitBuilder().create(10, 1, lattice)
H = HamiltonianBuilder().build_tfim(lattice, h=3.25)

# MPNN-predicted parameters (example)
params = np.array([0.15, 0.78])  # θ_zz, θ_x from MPNN

# Evaluate (returns ZNE-extrapolated energy)
backend = HardwareBackend(config=config)
energy = backend.evaluate(circuit, H, params)
print(f"ZNE energy: {energy:.6f}")
```

### 2. Real Hardware Execution (IBM Torino)

```bash
# Set credentials
export IBM_KEY="your_ibm_quantum_api_token"
export IBM_INSTANCE_CRN="your_instance_crn"
```

```python
import numpy as np
from qmbp_simulation.execution.hardware import HardwareBackend, HardwareConfig
from qmbp_simulation.models import HamiltonianBuilder, make_lattice
from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.solvers import ClassicalSolver

# Configure for real hardware
config = HardwareConfig(
    mode="hardware",
    backend_name="ibm_torino",
    n_qubits=10,
    shots=16384,
    n_layouts=3,
    output_dir="results/hardware",
)

# Build circuit and Hamiltonian
lattice = make_lattice("heavy_hex", 10)
circuit, _ = HVACircuitBuilder().create(10, 1, lattice)

# Get exact reference values (from local simulation)
solver = ClassicalSolver()
h_value = 3.25
H = HamiltonianBuilder().build_tfim(lattice, h=h_value)
exact = solver.solve(H)
e_exact = exact.energy
gap = exact.gap

# MPNN-predicted parameters
params = np.array([0.15, 0.78])  # From trained MPNN

# Full deployment with phase classification and persistence
backend = HardwareBackend(config=config)
result = backend.run_deployment(
    circuit, H, params,
    h_value=h_value,
    e_exact=e_exact,
    gap=gap,
    expected_label="paramagnetic",
)

print(f"ΔE/gap: {result.delta_e_gap:.4f}")
print(f"Phase: {result.phase_label}")
print(f"Verdict: {result.verdict}")
print(f"ZNE R²: {result.zne_r2:.4f}")
```

### 3. Multi-h Sweep (4 test points in one session)

```python
# Define test points
h_values = [4.0, 3.25, 3.0, 2.5]

# Pre-compute exact values and MPNN predictions for each h
params_per_h = {}
e_exact_per_h = {}
gap_per_h = {}

for h in h_values:
    H_h = HamiltonianBuilder().build_tfim(lattice, h=h)
    exact = solver.solve(H_h)
    e_exact_per_h[h] = exact.energy
    gap_per_h[h] = exact.gap
    params_per_h[h] = mpnn.predict(h)  # Your trained MPNN

# Execute sweep (h=4.0 first as smoke test)
results = backend.run_h_sweep(
    circuit,
    hamiltonian_builder=lambda h: HamiltonianBuilder().build_tfim(lattice, h=h),
    h_values=h_values,
    params_per_h=params_per_h,
    e_exact_per_h=e_exact_per_h,
    gap_per_h=gap_per_h,
)

for r in results:
    print(f"h={r.h_value}: ΔE/gap={r.delta_e_gap:.4f} [{r.verdict}]")
```

## Preflight Checks

Before submitting jobs, run preflight to verify conditions:

```python
backend = HardwareBackend(config=config)
checks = backend.run_preflight()

if checks["abort"]:
    print(f"ABORT: {checks['abort_reason']}")
else:
    print(f"Backend operational: {checks.get('backend_operational', 'N/A')}")
    print(f"Queue depth: {checks.get('queue_pending_jobs', 'N/A')}")
    print(f"Mean 2Q error: {checks.get('mean_2q_error', 'N/A')}")
```

Preflight checks:
- Backend operational status
- Queue depth (warns if > 50 jobs)
- Mean 2-qubit gate error (aborts if > 1%)
- Topology connectivity (enough connected qubits)

## Error Mitigation Stack

Applied automatically in hardware mode:

| Layer | Technique | Effect |
|-------|-----------|--------|
| 1 | Dynamical Decoupling (XpXm) | Suppresses idle decoherence |
| 2 | Pauli Twirling (32 randomizations) | Converts coherent → stochastic noise |
| 3 | TREX | Mitigates readout errors |
| 4 | Inhomogeneous ZNE (3 layouts) | Extrapolates to zero noise |

In `fake_backend` mode, only layout selection + ZNE are applied (DD/twirling/TREX
have no effect on local simulation).

## Output Structure

Each run creates a timestamped directory:

```
results/hardware/run_20260615_032241/
├── config.json          # Full input configuration (reproducibility)
├── provenance.json      # Execution metadata (job_ids, versions, layouts, CES)
├── raw_results.json     # Per-layout raw EVs and stds (no rounding)
├── zne_analysis.json    # ZNE extrapolation (R², gain, slope, CES-E pairs)
├── summary.json         # ΔE/gap, phase label, verdict (PASS/FAIL)
└── execution_log.json   # Complete structured log (all events with timestamps)
```

For multi-h sweeps, an additional `sweep_summary.json` is created with consolidated results.

## Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| ΔE/gap at h=3.25 | < 5% (PRIMARY) |
| Phase label | Correct ("paramagnetic") |
| ZNE R² | > 0.8 |
| ZNE gain | > 0% |

## SPSA Fallback

SPSA refinement activates ONLY when ΔE/gap > 5%:
- Parameters: a=0.1, c=0.05, A=10, 200 iterations (validated V7-4A)
- Never worsens the result (returns better of initial vs refined)
- Respects 10M shot cost ceiling
- Skipped for good warm-starts (refinement hurts by -146% per V7-4B)

## Key Constraints

- HVA p=1 only for N≥10 (18 CZ gates, within ZNE perturbative regime)
- 16384 shots (validated: 32k gives identical results)
- 3 layouts with lowest CES (validated: 5 layouts gives only +3% marginal gain)
- Never apply SPSA if ΔE/gap ≤ 5%
- Cost ceiling: 10M total shots per execution run
- Execute during off-peak hours (UTC 2-6 AM) if queue > 50 jobs

## Testing

```bash
# Run hardware module tests (fast, uses FakeTorino)
python -m pytest tests/test_hardware_module.py -v

# Run full test suite
python -m pytest tests/ -m "not slow" -q
```
