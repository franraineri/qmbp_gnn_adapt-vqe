# Hardware Execution Module — IBM Torino QPU

This module implements real quantum hardware execution via IBM Runtime,
integrated into the `ExecutionBackend` ABC pattern. It supports both
real IBM hardware and local FakeTorino simulation for validation.

## Deployment Plan — How to Use This Toolset

### The Big Picture

The hardware deployment follows a **calibration-first, 3-phase protocol** that
minimizes QPU credit risk while maximizing confidence in the results:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE A: LOCAL VALIDATION (no QPU, no credentials)                     │
│  ─────────────────────────────────────────────────────────────────────  │
│  1. Cost estimate:   make hw-cost N=10 H=3                              │
│  2. Preflight:       make hw-preflight N=10                             │
│  3. Full rehearsal:  make hw-rehearsal          (9 sections, ~60s)      │
│  4. GO/NO-GO:        make hw-analyze            (must show 🟢 GO)       │
│                                                                         │
│  Output: Confidence that software pipeline works end-to-end.            │
├─────────────────────────────────────────────────────────────────────────┤
│  PHASE B: CALIBRATION RUN (~5 min QPU)                                  │
│  ─────────────────────────────────────────────────────────────────────  │
│  5. Calibration:     make hw-deploy-calibrate                           │
│                      (Tier 0: 1 circuit, measures T_one_job)            │
│                                                                         │
│  Output: Real T_one_job, budget recompute, GO/NO-GO for full sweep.     │
│  Decision: Team reviews budget → approves or adjusts config.            │
├─────────────────────────────────────────────────────────────────────────┤
│  PHASE C: FULL EXECUTION (~30-120 min QPU)                              │
│  ─────────────────────────────────────────────────────────────────────  │
│  6. Deployment:      make hw-deploy                                     │
│                      (Tier 0→1→2→3, --no-spsa safety, auto-advancing)   │
│                                                                         │
│  Output: Thesis Table 5.23 data (4 h-points × 3 seeds × 2 models).     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Quick Reference — All Commands

| Phase | Command | What it does | QPU time |
|-------|---------|-------------|:--------:|
| A | `make hw-cost N=10 H=3` | Estimate QPU budget (model-based) | 0 |
| A | `make hw-preflight N=10` | Check FakeTorino topology & calibration | 0 |
| A | `make hw-rehearsal` | Full 9-section validation on FakeTorino | 0 |
| A | `make hw-rehearsal-quick` | Only cost + circuit audit (~2s) | 0 |
| A | `make hw-analyze` | Parse latest results → GO/NO-GO | 0 |
| B | `make hw-deploy-dry` | Verify config + credentials (no QPU) | 0 |
| B | `make hw-deploy-calibrate` | Tier 0: measure T_one_job | ~5 min |
| C | `make hw-deploy` | Full deployment (--no-spsa safety) | ~30-120 min |

### Why Multiple h-Points?

We measure at h = [4.0, 3.25, 3.0, 2.5] because:
- **h=4.0**: Deep paramagnetic, trivial ground state — validates QPU connectivity
- **h=3.25**: Primary thesis target — where MPNN accuracy is validated
- **h=3.0**: Closer to crossover — tests ZNE robustness as the gap shrinks
- **h=2.5**: Near boundary of valid regime — demonstrates pipeline limits

The spectral gap decreases as h → h_c ≈ 1.0, amplifying any energy error in
the ΔE/gap metric. This gradient validates that our pipeline works not just at
the easiest point, but across the physically meaningful region.

### Key Safety Features

- **`--no-spsa`**: Disables SPSA refinement. On real hardware SPSA costs 400 min
  per h-point if triggered. Since our MPNN achieves ΔE/gap < 5% (rehearsal-validated),
  SPSA is unnecessary and the flag prevents budget blowouts.
- **Tier auto-advancement**: Each tier only proceeds if the previous one passes.
  If Tier 0 fails → everything stops (zero waste).
- **Wall-clock timing**: Every QPU call is timed. If any h-point exceeds 600s,
  a warning is logged (likely SPSA triggered or queue stalled).
- **Budget recompute**: After Tier 0, the script replaces the theoretical CLOPS
  estimate with the REAL measured T_one_job for accurate budget planning.

### What Gets Saved

Every execution saves comprehensive data for thesis analysis:

| Data | Where | For what |
|------|-------|----------|
| Energy (E_zne) | `execution_summary.json` | Primary result |
| Per-site ⟨X_i⟩ (10 values) | `per_h[].per_site_x` | Thesis figures |
| Per-bond ⟨ZZ_ij⟩ (9 values) | `per_h[].per_bond_zz` | Phase transition |
| ZNE R², gain, amplifier | `per_h[].zne_*` | Methodology validation |
| Phase label + confidence | `per_h[].phase_label/sigma` | Classification |
| Job IDs + layouts + CES | `per_h[].job_ids/layouts_used` | Reproducibility |
| T1/T2 at execution time | `per_h[].calibration_snapshot` | Correlation analysis |
| Transpiled depth/gates | `per_h[].transpiled_stats` | Circuit characterization |
| QPU seconds (IBM-charged) | `per_h[].qpu_metrics` | Budget tracking |
| Wall-clock per h-point | `per_h[].wall_clock_s` | Timing model validation |
| Affine/GNN-QEM corrections | `per_h[].affine_*/gnn_qem_*` | Post-processing audit |

Additionally, `HardwareBackend.run_deployment()` independently saves per-h-point
directories in `results/hardware/run_*/` with full provenance (config, raw per-layout
energies, ZNE analysis, input parameters, execution log).


## Architecture

```
execution/hardware/
├── __init__.py      # Public API exports (HardwareBackend, QPUThroughputProfile, etc.)
├── backend.py       # HardwareBackend class (evaluate, run_deployment, run_h_sweep)
├── config.py        # HardwareConfig, SPSAConfig, HardwareRunResult
├── preflight.py     # Pre-execution checks + QPU cost estimation + CLOPS model
├── submission.py    # Job submission with retry logic + layout selection
├── observables.py   # Per-site observable construction and extraction
├── phase.py         # Phase classification (paramagnetic/ordered/indeterminate)
├── spsa.py          # Conditional SPSA refinement
└── persistence.py   # Result saving with full provenance
```

### Key exports from `__init__.py`

```python
from qmbp_simulation.execution.hardware import (
    HardwareBackend, HardwareConfig, HardwareRunResult, SPSAConfig,
    # Cost estimation (composable, pluggable)
    QPUCostEstimate, QPUThroughputProfile, SPSACostModel,
    estimate_effective_clops, estimate_qpu_cost,
)
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
                ├─ 6. Mode-aware ZNE aggregation:
                │      ├─ hardware + zne_enabled → mean(per-layout energies)
                │      ├─ fake_backend + zne_enabled → local GF/PEA/adaptive ZNE
                │      └─ zne_disabled → CES-based linear extrapolation (legacy)
                └─ return: float (ZNE-mitigated energy)
```

### HardwareRunResult Fields (new in 2026-06-05)

| Field | Type | Description |
|-------|------|-------------|
| `mitigation_strategy` | `str` | Human-readable label: `"ibm_zne_layout_avg"`, `"pea_local"`, `"gate_folding_local"`, or `"ces_zne"` |
| `layout_std` | `float \| None` | Standard deviation across per-layout energies (variance metric for layout averaging) |
| `fallback_triggered` | `bool` | True if adaptive ZNE fell back from gate-folding to PEA due to low R² |

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
# Use create_pauli_evolution() for hardware — 6-10% lower total_depth
# (noiseless VQE training still uses create())
circuit, _ = HVACircuitBuilder().create_pauli_evolution(10, 1, lattice)
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
# Set credentials (REQUIRED — passed directly to QiskitRuntimeService)
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
# Use create_pauli_evolution() for hardware deployment (6-10% lower total_depth)
circuit, _ = HVACircuitBuilder().create_pauli_evolution(10, 1, lattice)

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
print(f"Verdict: {result.verdict} — {result.verdict_reason}")
print(f"ZNE R²: {result.zne_r2:.4f}")
print(f"Strategy: {result.mitigation_strategy}")
print(f"Layout std: {result.layout_std}")
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
    print(f"Mean readout error: {checks.get('mean_readout_error', 'N/A')}")
    print(f"Min T1: {checks.get('min_t1_us', 'N/A')} μs")
    print(f"Shots/eval: {checks.get('shots_per_eval', 'N/A')}")
```

Preflight checks (both modes):
- **Topology connectivity** — enough connected qubits for N
- **Cost ceiling** — `shots × n_layouts` does not exceed `max_total_shots`

Additional checks (hardware mode only):
- **Backend operational status** — QPU is active
- **Queue depth** — warns if > 50 pending jobs
- **Mean 2-qubit gate error** — aborts if > 1% (poor calibration)
- **Mean readout error** — warns if > 3% (TREX mitigates, but increases variance)
- **T1/T2 coherence** — aborts if min T1 < 50μs (decoherence dominates)
- **Native gate support** — warns if ECR not in native set (transpiler overhead)

### Circuit-Level Validation

Called automatically by `run_deployment()` before job submission:

```python
from qmbp_simulation.execution.hardware.preflight import validate_circuit_for_zne

result = validate_circuit_for_zne(circuit, config, logger)
if result["abort"]:
    print(f"ABORT: {result['abort_reason']}")
print(f"2Q gates: {result['two_qubit_gate_count']} (threshold: {result['zne_threshold']})")
```

Thresholds are **amplifier-aware** (updated 2026-06-05):
- **Gate-folding**: aborts if 2Q gates > 18 (folding triples depth → non-perturbative)
- **PEA / Adaptive**: aborts if 2Q gates > 50 (noise amplified via model, not depth)
- Warns at 80% of threshold (R² may degrade)

| Amplifier | 2Q Gate Threshold | Rationale |
|-----------|:-----------------:|-----------|
| `gate_folding` | 18 | Each gate folded 3×: 18 → 54 effective gates |
| `pea` | 50 | Noise model amplification, circuit unchanged |
| `adaptive` | 50 | Uses PEA threshold (falls back to PEA if GF R²<0.90) |

### QPU Cost Estimation

Depth-aware cost estimation with pluggable hardware profiles:

```python
from qmbp_simulation.execution.hardware import (
    estimate_qpu_cost, QPUThroughputProfile, SPSACostModel,
)

# Default (Torino, P(SPSA)=0.30)
est = estimate_qpu_cost(config, n_h_points=4)
print(f"Optimistic: {est.est_total_optimistic_s/60:.1f} min")
print(f"Expected:   {est.est_total_s/60:.1f} min")
print(f"Pessimistic: {est.est_total_pessimistic_s/60:.1f} min")

# Safe mode (no SPSA, recommended for hardware)
est_safe = estimate_qpu_cost(config, n_h_points=4,
    spsa_model=SPSACostModel.disabled())
print(f"Safe: {est_safe.est_total_optimistic_s/60:.1f} min")
```

### Input Validation (run_deployment)

Before any QPU interaction, `run_deployment()` validates:
- `params.shape` matches `circuit.num_parameters`
- `gap > 0` (prevents division by zero in ΔE/gap)
- `params` contains no NaN/Inf (prevents invalid circuits)
- `e_exact` is finite (prevents meaningless verdicts)

All validations fail with `ValueError` — zero QPU cost on misconfiguration.

## Transpilation Strategy

**optimization_level=2** with explicit `initial_layout` (BFS-selected, CES < 0.5).

### Explored options (2026-06-05, corrected 2026-06-15)

| Method | Total Depth | n_2Q | CES | Verdict |
|--------|:-----------:|:----:|:---:|---------|
| **PauliEvol + SABRE lvl2** ✅ | 82–90 | 34 | 0.1251 | **Best — in production** |
| Orig HVA + SABRE lvl2 | 89–90 | 34 | 0.1271 | Previous default |
| Orig HVA + SABRE lvl3 | ~88 | 34 | 0.1271 | No benefit |
| PauliEvol + Rustiq | — | 67 | — | **Counterproductive** |

> **Note (2026-06-15)**: The original 2026-06-05 report measured *2Q-depth* (27→24,
> −11%). The correct metric for hardware decoherence is *total_depth*, which differs
> by 6–10% at non-trivial theta values (Section 20 validation). On heavy_hex N=10 p=1,
> all ZZ bonds are non-overlapping, so 2Q-depth = 1 for both representations (already
> fully parallelized by the scheduler). The total_depth reduction (82–81 vs 89–90)
> reduces time-domain decoherence exposure on real hardware.

**Use `HVACircuitBuilder.create_pauli_evolution()`** for hardware deployment.
This is already the default in `run_ibm_torino_deployment.py` (Tiers 0, 1, 2).

> **Important**: `create_pauli_evolution()` uses coefficient `0.5` for ZZ and X
> operators so that `PauliEvolutionGate(H, time=2θ) = e^{-iθ·ZZ} = RZZ(2θ)`. Bug
> in original 2026-06-05 version (coefficient=1.0) produced wrong energies — fixed
> 2026-06-15. See `binnacle-pauli-evolution-transpilation.md` for full details.

Noiseless VQE training (StatevectorEstimator) still uses `create()` — there is
no transpilation in that path and no benefit from the PauliEvolutionGate form.

## Error Mitigation Stack

Applied automatically in hardware mode:

| Layer | Technique | Effect |
|-------|-----------|--------|
| 1 | Dynamical Decoupling (XpXm) | Suppresses idle decoherence |
| 2 | Pauli Twirling (32 randomizations) | Converts coherent → stochastic noise |
| 3 | TREX | Mitigates readout errors |
| 4 | ZNE (configurable amplifier) | Extrapolates to zero noise |

### ZNE Amplifier Selection

Three noise amplification strategies are available via `MitigationOptions.zne_amplifier`:

| Amplifier | Method | Pros | Cons |
|-----------|--------|------|------|
| `"pea"` (**recommended**) | Probabilistic Error Amplification | +94% gain, R²=0.998, topology-independent | ~50% QPU overhead from noise learning phase |
| `"gate_folding"` (fallback) | Digital: U → U·U†·U | Simple, zero overhead | R²=0.47 on heavy_hex p=1 (may fail on shallow circuits) |
| `"adaptive"` (auto) | GF first, PEA if R²<threshold | Best of both: cheap when GF works, reliable when it doesn't | Slightly more complex; 2× measurement in worst case |

**Adaptive mode** (new in 2026-06-05): Tries gate-folding first (zero overhead). If
the resulting R² < `zne_r2_fallback_threshold` (default 0.90), automatically falls back
to PEA. This is the recommended mode for unattended deployment where circuit/topology
properties are uncertain.

```python
from qmbp_simulation.execution import MitigationOptions

# Adaptive (recommended for new experiments)
mitigation = MitigationOptions(
    zne_enabled=True,
    zne_amplifier="adaptive",
    zne_r2_fallback_threshold=0.90,  # configurable
    zne_noise_factors=[1, 3, 5],
)
```

CLI usage:
```bash
python my_runner.py --zne-amplifier adaptive --zne-r2-threshold 0.85
```

**Validated strategy** (from ZNE_CROSS_TOPO, 2026-06-04):
1. **Primary**: Deploy with `pea` (validated +94.4% gain, 18/18 wins across 3 topologies, t=46.32, p<10⁻¹⁹)
2. **Fallback**: If `pea` unavailable or `qiskit-aer` not installed, use `gate_folding` (+20.6% gain)
3. IBM Runtime handles PEA's noise learning automatically on real hardware

> **Critical finding**: CES-based inhomogeneous ZNE (different layouts → extrapolate
> to CES=0) **fails on heavy_hex** because all good layouts have CES≈0.15 (no spread).
> This has been resolved (NM-1, 2026-06-05): `run_deployment()` now uses IBM server-side
> ZNE for hardware mode and local GF/PEA for fake_backend mode.
> See `documentation/analysis/13_hardware_zne_improvements.md` for the original plan.

**Recommended configuration:**

```python
from qmbp_simulation.execution.backends import MitigationOptions

# PEA (recommended primary — validated across chain_1d, heavy_hex, ladder)
mitigation = MitigationOptions(
    zne_enabled=True,
    zne_amplifier="pea",
    zne_noise_factors=[1, 3, 5],
    num_randomizations=32,
    shots_per_randomization=128,  # IBM LayerNoiseLearning default
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
    "layer_noise_learning": {"num_randomizations": 32, "shots_per_randomization": 128}
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

### QESEM-Style Tiered Mitigation (Literature Validation)

Our adaptive PEA→GF strategy is independently validated by the QESEM framework
(Aharonov, Lindner, et al., arXiv:2508.10997, Aug 2025):

> **Core principle**: Characterization-based methods (PEA, PEC, QESEM) always
> outperform heuristic methods (gate-folding) because they model the *actual*
> noise channel via Sparse Pauli-Lindblad fitting, not uniform noise scaling.

| Property | QESEM (127 qubits) | Our PEA (10 qubits) |
|----------|-------|-----------|
| Noise model | Sparse Pauli-Lindblad | Same (via `qiskit-aer`) |
| Mean gain vs heuristic | "Consistently higher accuracy" | +94.4% vs +20.6% (GF) |
| Tiered approach | PEC with reduced overhead | PEA primary, GF fallback |
| Validation device | IBM Heron (kicked TFIM) | FakeTorino (TFIM static) |

**Critical finding** (HW_REHEARSAL_V2 section 5, 2026-06-05):
Gate-folding R²=0.996 produced ΔE/gap=89.8% while PEA R²=0.998 produced
ΔE/gap=2.07% on the same circuit. High R² ≠ accuracy. The `run_adaptive_zne()`
default strategy was changed from `gf_primary` to `pea_primary` accordingly.

**Post-processing stack** (applied after ZNE, from 2025-2026 literature):
1. `affine_correct_energy()` — Physics bounds E ∈ [E₀, E_max] (arXiv:2604.16815)
2. `run_block_zne()` — Block-level folding for p≥2 (arXiv:2507.23314)
3. `check_calibration_drift()` — TLS stability monitoring (Nature Comms 2025)

See: `documentation/analysis/15_advanced_mitigation_techniques.md`

### GNN-QEM Integration (Optional Post-ZNE Correction)

When a trained GNN-QEM model is loaded via `backend.load_gnn_qem(path)`, the
deployment pipeline applies an additional correction after ZNE:

```
ZNE energy (E_zne)
  → GNN-QEM correction (E_zne + ΔE_GNN, if confidence ≥ threshold)
  → Affine correction (clip to [E_ground, E_upper])
  → Final energy used for verdict
```

**Critical constraint (validated 2026-06-06):** GNN-QEM only activates when
`zne_amplifier != "pea"`. PEA already removes structured noise → GNN-QEM on
PEA residuals REGRESSES (0/15 improvements, −31,000% mean regression). The
model is designed for GF-ZNE residuals where structured error remains.

**Circuit selection mode** (no E_noisy, Spearman ρ=0.945): The model can also
rank circuits by expected error difficulty WITHOUT executing them. This
enables layout selection and feasibility checks at zero QPU cost.

**Usage:**

```python
backend = HardwareBackend(config=config)
backend.load_gnn_qem("results/gnn_qem/model_vqe_realistic.pt")  # Optional

result = backend.run_deployment(circuit, H, params, h, e_exact, gap)
# GNN-QEM only activates if zne_amplifier != "pea" (e.g., gate_folding fallback)
# result.gnn_qem_applied: bool
# result.gnn_qem_delta_e: float or None
# result.gnn_qem_confidence: float or None
```

**Behavior:**
- If no model loaded (`_gnn_qem_model is None`): GNN step is skipped entirely
- If PEA is the ZNE amplifier: GNN step is skipped (PEA handles structured noise)
- If model confidence < threshold (default 0.8): correction NOT applied
- Affine correction ALWAYS runs (zero cost, clips unphysical extrapolations)

## Output Structure

Each run creates a timestamped directory:

```
results/hardware/run_20260615_032241/
├── config.json          # Full input configuration (reproducibility)
├── provenance.json      # Execution metadata (job_ids, versions, layouts, CES, qpu_seconds)
├── raw_results.json     # Per-layout raw EVs, stds, and QPU metrics (no rounding)
├── zne_analysis.json    # ZNE extrapolation (R², gain, slope, amplifier)
├── summary.json         # ΔE/gap, phase, verdict, verdict_reason, mitigation_strategy,
│                        # per_site_x, per_bond_zz, layout_std, GNN-QEM fields, affine fields
├── input_params.json    # Exact parameters used (for reproduction)
└── execution_log.json   # Complete structured log (all events with timestamps)
```

For multi-h sweeps, an additional `sweep_summary.json` is created with consolidated
results including per-h-point metadata, aggregate statistics, and pass rates.

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

## Limitations

### PEA Local Simulation Approximation

The local PEA simulation (`run_pea_zne()` with FakeTorino) uses an
**isotropic depolarizing** noise model to approximate IBM's full
Pauli-Lindblad error amplification:

| Aspect | Local Simulation | IBM Runtime (real hardware) |
|--------|------------------|---------------------------|
| Noise model | Depolarizing (1 rate per gate pair) | Full Pauli-Lindblad (15 generators per 2Q pair) |
| Amplification | Scale depolarizing rate × factor | Scale each Pauli channel independently |
| Accuracy vs FakeTorino | Exact (same noise type) | N/A |
| Accuracy vs real QPU | ±5-10% in extrapolated energy | Ground truth |
| Overhead | Zero (local sim) | ~50% extra QPU time (noise learning) |

**Implication for thesis**: Local PEA results (R², gain percentages)
are accurate for validating the *methodology* but the absolute energy
values may differ from real QPU results by up to 10%. The
comparative rankings (PEA > GF > raw) remain valid on hardware.

### CES-ZNE Failure on heavy_hex

Client-side CES-based inhomogeneous ZNE (extrapolating E(CES) → CES=0
across different layouts) **does not work on heavy_hex topology** because
all good layouts have nearly identical CES≈0.15 (no spread → R²≈0.04).
The current implementation uses IBM's server-side ZNE for hardware mode
and local GF/PEA for fake_backend mode to avoid this issue.

## Pre-Hardware Validation (Rehearsal V2)

Before committing QPU time, run the full 9-section rehearsal:

```bash
# Full rehearsal (all 9 sections, ~60s on FakeTorino N=10)
python scripts/hardware.py rehearsal --topology heavy_hex

# With optional backend preflight (Section 0)
python scripts/hardware.py rehearsal --run-preflight

# Quick check (cost + circuit audit only, ~2s)
python scripts/hardware.py rehearsal --section 8 9

# Or via make:
make hw-rehearsal
make hw-rehearsal-quick

# Analysis of results
python scripts/hardware.py analyze --all
make hw-analyze
```

### Rehearsal Sections

| # | Section | What it validates | Pass criterion |
|---|---------|-------------------|----------------|
| 0 | HW Preflight (optional) | Topology, cost ceiling, T1/T2, readout | No abort |
| 1 | MPNN Prediction | θ_pred quality (noiseless) | ΔE/gap < 5% all h_test |
| 2 | HardwareBackend Noisy | evaluate() + ZNE pipeline | ΔE/gap < 5% |
| 3 | Full run_deployment() | Complete verdict pipeline | verdict = PASS |
| 4 | Adaptive ZNE | GF→PEA fallback mechanism | Fields populated correctly |
| 5 | Amplifier Comparison | GF vs PEA cost/quality | At least one < 5% |
| 6 | Shot Noise | Measurement reproducibility | std(ΔE/gap) < 3% |
| 7 | Phase Classification | Observable SNR + labels | All correct, SNR > 1 |
| 8 | QPU Cost Estimation | Fits IBM time limits | time_per_h < max_execution_time |
| 9 | Circuit Depth Audit | 2Q gates ≤ ZNE threshold | All layouts viable |

### Rehearsal CLI Parameters (scalable configuration)

All hardcoded defaults can be overridden for different configurations:

```bash
python scripts/hardware.py rehearsal \
  --n-qubits 10 \
  --topology heavy_hex \
  --p-layers 1 \
  --zne-amplifier pea \
  --shots 16384 \
  --h-test 4.0 3.25 3.0 \
  --h-train 4.5 4.25 4.0 3.75 3.5 3.25 3.0 \
  --vqe-restarts 1 \
  --mpnn-epochs 6000 \
  --mpnn-hidden-dim 128 \
  --n-shots-reps 5 \
  --run-preflight
```

### GO/NO-GO Decision

After running all sections, `analyze_hw_rehearsal_v2.py` reports:
- **🟢 GO**: Sections 1, 2, 3, 7, 9 all pass → safe to submit to IBM Torino
- **🔴 NO-GO**: Any critical section fails → fix before QPU

The analyzer also checks for ZNE regression (mitigation making things worse)
and supports `--threshold` to tighten the pass criterion for thesis targets.

## Tiered QPU Deployment (Calibration-First Strategy)

For production hardware runs on IBM Torino, use the tiered deployment script.
The strategy follows Hamed's calibration-first protocol: measure T_one_job first,
then scale the budget with confidence.

### 2-Session Protocol

```
┌─────────────────────────────────────────────────────────────────┐
│  Session 1: CALIBRATION (~5 min QPU)                            │
│  ─────────────────────────────────────────────────────────────  │
│  Tier 0: 1 circuit + full mitigation → measures T_one_job       │
│  Output: "Full sweep will take X min based on measured time"    │
│  Action: Team reviews, approves budget                          │
├─────────────────────────────────────────────────────────────────┤
│  Session 2: EXECUTION (~30-120 min QPU)                         │
│  ─────────────────────────────────────────────────────────────  │
│  Tier 1: 4 h-points × 3 layouts × 3 ZNE factors (thesis data)  │
│  Tier 2: Same × 3 seeds (statistical robustness)                │
│  Tier 3: tfim_longitudinal at g=0.3 (model extensibility)       │
└─────────────────────────────────────────────────────────────────┘
```

### Usage

```bash
# Dry run (cost estimate + preflight, no QPU)
make hw-deploy-dry
python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py --dry-run

# Session 1: Calibration only (measures T_one_job)
make hw-deploy-calibrate
python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py --tier 0

# Session 2: Full deployment with SPSA disabled (recommended)
make hw-deploy
python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py --no-spsa

# Full deployment (all tiers, auto-advancing)
python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py

# Custom configuration
python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py \
  --shots 32768 --zne-amplifier adaptive --tier 1 2
```

### Key Safety Features

| Feature | Flag | Purpose |
|---------|------|---------|
| **SPSA kill-switch** | `--no-spsa` | Prevents 400-min budget blowout (200 iters × 2 evals × 60s/job) |
| **Per-h timing** | automatic | Logs wall-clock for each h-point, warns if > 600s |
| **Budget recompute** | after Tier 0 | Uses measured T_one_job to validate model estimate |
| **Auto-abort** | automatic | Stops tier progression if smoke test fails |
| **Budget ceiling** | 4h max | Warns if optimistic estimate exceeds 4 hours |

### Tier Details

| Tier | h-points | Seeds | Purpose | Pass criterion |
|------|----------|-------|---------|----------------|
| 0 | 1 (h=4.0) | 42 | Calibration, measure T_one_job | verdict=PASS |
| 1 | 4 (4.0, 3.25, 3.0, 2.5) | 42 | Primary thesis data | ≥3/4 PASS |
| 2 | 4 × 3 seeds | 42,43,44 | Statistical robustness | ≥75% pass rate |
| 3 | 1 (h=3.25) | 42 | tfim_longitudinal extensibility | verdict=PASS |

### Output Structure

Each deployment run creates:
```
results/hardware/run_YYYYMMDD_HHMMSS/
├── execution_summary.json   # Full metrics: timing, per-h results, budget recompute
├── config.json              # Reproducibility config
├── provenance.json          # Job IDs, layouts, CES values
├── raw_results.json         # Per-layout raw EVs
├── zne_analysis.json        # ZNE extrapolation details
└── summary.json             # ΔE/gap, phase, verdict per h-point
```

### QPU Budget Estimate (from measured T_one_job)

After Tier 0 measures T_one_job, the script prints:
```
T_total = PEA_learning + N_h × N_layouts × N_zne_factors × T_one_job
        = 60s + 4h × 3 layouts × 3 factors × 60s
        ≈ 37 min (optimistic, no SPSA)
```

If T_one_job ≈ 60s (typical for N=10, 16k shots, full mitigation):
- Tier 1 alone: ~37 min
- Tier 1+2: ~150 min
- Full (Tier 0-3): ~190 min

**Critical**: With `--no-spsa`, this is the hard ceiling. Without it,
SPSA on ONE h-point adds 400 min.

## QPU Cost Estimation

The cost estimation module provides depth-aware, composable budget planning:

```python
from qmbp_simulation.execution.hardware import (
    estimate_qpu_cost, QPUThroughputProfile, SPSACostModel, HardwareConfig,
)

# Default: IBM Torino, PEA amplifier, P(SPSA)=0.30
config = HardwareConfig(n_qubits=10, shots=16384, n_layouts=3)
est = estimate_qpu_cost(config, n_h_points=3)
print(f"Optimistic: {est.est_total_optimistic_s/60:.1f} min")
print(f"Expected:   {est.est_total_s/60:.1f} min")

# Compare backends
for profile_fn in [QPUThroughputProfile.ibm_torino,
                   QPUThroughputProfile.ibm_heron_r2,
                   QPUThroughputProfile.ibm_nighthawk]:
    p = profile_fn()
    e = estimate_qpu_cost(config, n_h_points=3, profile=p, spsa_model=SPSACostModel.disabled())
    print(f"{p.name}: {e.est_total_optimistic_s/60:.1f} min, CLOPS={e.effective_clops}")
```

### Depth-Aware CLOPS Model

CLOPS scales with circuit width and depth:
```
CLOPS(N, D) = base_clops × (ref_N / N)^α × (ref_D / D)^β
```

| N | Effective CLOPS (Torino) | Time/circuit (16k shots) |
|---|:---:|:---:|
| 6 | 3674 | 4.5s |
| 10 | 2500 | 6.6s |
| 20 | 1503 | 10.9s |
| 40+ | 1000 | 16.4s |

### SPSA Cost Models

| Model | P(trigger) | Use case |
|-------|:---:|---|
| `SPSACostModel.disabled()` | 0% | Validated MPNN, `--no-spsa` flag |
| `SPSACostModel()` | 30% | Default (good MPNN predictions) |
| `SPSACostModel.conservative()` | 50% | First deployment, untested config |
| `SPSACostModel.aggressive()` | 100% | Worst-case budget ceiling |

### CLI

```bash
# Quick estimate
python scripts/hardware.py cost -N 10 --h-points 3

# Compare backends
python scripts/hardware.py cost -N 10 --profile nighthawk --spsa disabled

# JSON output for scripting
python scripts/hardware.py cost -N 10 --json

# Make target
make hw-cost N=10 H=3 PROFILE=torino
```

## Unified Hardware CLI

All hardware operations are available through a single entry point:

```bash
python scripts/hardware.py <command> [options]
```

| Command | Purpose | Make target |
|---------|---------|-------------|
| `cost` | QPU time/shot budget estimate | `make hw-cost` |
| `preflight` | Backend checks (topology, T1/T2, readout) | `make hw-preflight` |
| `rehearsal` | Full 9-section validation on FakeTorino | `make hw-rehearsal` |
| `analyze` | Post-rehearsal GO/NO-GO verdict | `make hw-analyze` |

Deployment uses a separate script (requires IBM credentials):

| Target | Purpose |
|--------|---------|
| `make hw-deploy-dry` | Preflight + cost, no QPU |
| `make hw-deploy-calibrate` | Session 1: Tier 0 only |
| `make hw-deploy` | Session 2: Full sweep (--no-spsa) |

## Hardware Generations & Scaling

For a detailed comparison of IBM processor generations (Eagle → Heron → Nighthawk)
including EPLG benchmarks, fidelity estimates by system size, and N-scaling projections:

→ **`documentation/analysis/18_ibm_hardware_generations.md`**

Key takeaway:
- ibm_torino (Heron r1): N=50-60 viable with PEA-ZNE + fractional gates
- Nighthawk: N=100-120 viable (3× better EPLG, square lattice, T₁=350μs)
- Fractional gates (rzz nativo) eliminate CX decomposition → halve effective gate count

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
