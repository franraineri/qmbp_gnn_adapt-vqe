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
MPNN θ_opt → HardwareBackend.run_deployment(circuit, H, params, h, e_exact, gap)
                │
                ├─ 1. Input validation (param shape, gap>0, finite values)
                ├─ 2. Preflight checks (status, calibration, topology, cost ceiling)
                ├─ 3. Circuit ZNE check (2Q gate count ≤ 18 threshold)
                ├─ 4. Bind parameters to HVA circuit
                ├─ 5. Select 3 lowest-CES layouts (cached)
                ├─ 6. Transpile circuit to each layout
                ├─ 7. Submit all 3 energy PUBs (parallel, with retry)
                ├─ 8. Collect results + discard NaN/Inf (timeout-aware)
                ├─ 9. Linear ZNE extrapolation → E(CES=0)
                ├─ 10. Measure per-site ⟨X_i⟩ and ⟨Z_iZ_j⟩
                ├─ 11. Classify phase: |⟨X⟩| vs |⟨ZZ⟩|
                ├─ 12. Compute ΔE/gap
                ├─ 13. Conditional SPSA (only if ΔE/gap > 5%)
                ├─ 14. Save all results + params + logs to disk
                └─ return: HardwareRunResult
```

For simple energy evaluation only (no phase classification or persistence):

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
    print(f"Shots/eval: {checks.get('shots_per_eval', 'N/A')}")
```

Preflight checks (both modes):
- **Topology connectivity** — enough connected qubits for N
- **Cost ceiling** — `shots × n_layouts` does not exceed `max_total_shots`

Additional checks (hardware mode only):
- **Backend operational status** — QPU is active
- **Queue depth** — warns if > 50 pending jobs
- **Mean 2-qubit gate error** — aborts if > 1% (poor calibration)

### Circuit-Level Validation

Called automatically by `run_deployment()` before job submission:

```python
from qmbp_simulation.execution.hardware.preflight import validate_circuit_for_zne

result = validate_circuit_for_zne(circuit, config, logger)
if result["abort"]:
    print(f"ABORT: {result['abort_reason']}")
print(f"2Q gates: {result['two_qubit_gate_count']} (threshold: 18)")
```

- **Aborts** if 2Q gate count > 18 (ZNE non-perturbative regime)
- **Warns** if 2Q gate count > 80% of threshold (R² may degrade)
- Counts: CX, CZ, ECR, RZZ, RXX, RYY, CP gates

### Input Validation (run_deployment)

Before any QPU interaction, `run_deployment()` validates:
- `params.shape` matches `circuit.num_parameters`
- `gap > 0` (prevents division by zero in ΔE/gap)
- `params` contains no NaN/Inf (prevents invalid circuits)
- `e_exact` is finite (prevents meaningless verdicts)

All validations fail with `ValueError` — zero QPU cost on misconfiguration.

## Error Mitigation Stack

Applied automatically in hardware mode:

| Layer | Technique | Effect |
|-------|-----------|--------|
| 1 | Dynamical Decoupling (XpXm) | Suppresses idle decoherence |
| 2 | Pauli Twirling (32 randomizations) | Converts coherent → stochastic noise |
| 3 | TREX | Mitigates readout errors |
| 4 | ZNE (configurable amplifier) | Extrapolates to zero noise |

### ZNE Amplifier Selection

Two noise amplification strategies are available via `MitigationOptions.zne_amplifier`:

| Amplifier | Method | Pros | Cons |
|-----------|--------|------|------|
| `"pea"` (**recommended**) | Probabilistic Error Amplification | +94% gain, R²=0.998, topology-independent | ~50% QPU overhead from noise learning phase |
| `"gate_folding"` (fallback) | Digital: U → U·U†·U | Simple, zero overhead | R²=0.47 on heavy_hex p=1 (may fail on shallow circuits) |

**Validated strategy** (from ZNE_CROSS_TOPO, 2026-06-04):
1. **Primary**: Deploy with `pea` (validated +94.4% gain, 18/18 wins across 3 topologies, t=46.32, p<10⁻¹⁹)
2. **Fallback**: If `pea` unavailable or `qiskit-aer` not installed, use `gate_folding` (+20.6% gain)
3. IBM Runtime handles PEA's noise learning automatically on real hardware

> **Critical finding**: CES-based inhomogeneous ZNE (different layouts → extrapolate
> to CES=0) **fails on heavy_hex** because all good layouts have CES≈0.15 (no spread).
> The current `run_deployment()` uses this broken strategy and must be updated.
> See `documentation/analysis/13_hardware_zne_improvements.md` for the implementation plan.

**Recommended configuration:**

```python
from qmbp_simulation.execution.backends import MitigationOptions

# PEA (recommended primary — validated across chain_1d, heavy_hex, ladder)
mitigation = MitigationOptions(
    zne_enabled=True,
    zne_amplifier="pea",
    zne_noise_factors=[1, 3, 5],
    num_randomizations=32,
    shots_per_randomization=512,
)

# Gate-folding (fallback if PEA unavailable)
mitigation = MitigationOptions(
    zne_enabled=True,
    zne_amplifier="gate_folding",
    zne_noise_factors=[1, 3, 5],
)
```

**On real hardware**, PEA is controlled via EstimatorV2 options:
```json
{
  "resilience": {
    "zne_mitigation": true,
    "zne": {"amplifier": "pea", "noise_factors": [1, 3, 5]},
    "layer_noise_learning": {"num_randomizations": 32, "shots_per_randomization": 512}
  }
}
```

> **Note**: IBM Runtime returns already-mitigated energies when `zne_mitigation=True`.
> The client does NOT need to perform additional CES extrapolation.
> Multi-layout averaging across 3 low-CES layouts provides √3 variance reduction.
```

**For local simulation** (FakeTorino), PEA is simulated by:
1. Learning per-gate error rates from backend calibration data
2. Building amplified noise models (depolarizing × noise_factor)
3. Running the unmodified circuit through the amplified model

```python
from qmbp_simulation.execution import run_pea_zne, run_pea_zne_deployment

# Single-layout PEA-ZNE
pea_result = run_pea_zne(
    transpiled, H_mapped, fake_backend, config,
    noise_factors=(1, 3, 5), extrapolator="linear",
)
print(f"E={pea_result.extrapolated_value:.6f}, R²={pea_result.r_squared:.4f}")
print(f"Learned rates: {pea_result.learned_error_rates}")
```

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
├── input_params.json    # Exact parameters used (for reproduction)
└── execution_log.json   # Complete structured log (all events with timestamps)
```

For multi-h sweeps, an additional `sweep_summary.json` is created with consolidated results.

### Reproducibility

Every run saves `input_params.json` containing the exact parameter vector used:
```json
{"params": [0.15, 0.78], "n_params": 2}
```
This allows re-executing the identical point without retraining the MPNN.

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

## Integration with Runner Framework

For runner scripts that use hardware execution, use `HardwareValidationRunner`
from `qmbp_simulation.framework.runner_base`:

```python
from qmbp_simulation.framework.runner_base import HardwareValidationRunner, Section

class MyHWRunner(HardwareValidationRunner):
    runner_id = "hw_deploy"
    experiment_id = "HW_DEPLOY"
    description = "Hardware deployment validation"
    hypothesis = "ΔE/gap<5% at h=3.25 on IBM Torino"

    def setup(self):
        super().setup()  # Creates self.hw_backend
        # ... build circuit, H, params ...

    def define_sections(self):
        return [Section(id=1, name="Deploy", fn=self.section_deploy, hypothesis="...")]

    def section_deploy(self) -> dict:
        result = self.hw_backend.run_deployment(...)
        return {"delta_e_gap": result.delta_e_gap, "pass": result.verdict == "PASS"}
```

This provides:
- Dual preflight (structural + QPU)
- Shared StructuredLogger between runner and backend
- CLI: `--mode hardware|fake_backend`, `--shots`, `--n-layouts`, `--zne-amplifier gate_folding|pea`, `--zne-noise-factors`
- Result saved to both `results/experiments/` (digest-compatible) and `results/hardware/` (full provenance)
