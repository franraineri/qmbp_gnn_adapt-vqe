---
inclusion: fileMatch
fileMatchPattern: "**/execution/noisy_utils*,**/execution/hardware/**,**/execution/backends*,**/noise_zne*,**/run_pea*,**/run_zne*,**/run_gate_folding*,**/run_adaptive*,**/noisy*"
---

# Noisy Execution — Methodology & Invariants

> Auto-included when touching noisy simulation, ZNE, or hardware execution files.
> Covers methodology correctness, performance constraints, and API contracts.

## Module Architecture (execution/)

```
execution/
├── backends.py         ← ExecutionBackend ABC, NoiselessBackend, NoisyBackend, MPSBackend re-export
├── mps_backend.py      ← MPSBackend (N>22, deterministic default, χ=64)
├── noisy_utils.py      ← ALL ZNE strategies + layout selection + calibration
└── hardware/           ← IBM Torino deployment orchestration
    ├── backend.py      ← HardwareBackend (evaluate, run_deployment, run_h_sweep)
    ├── config.py       ← HardwareConfig, SPSAConfig, HardwareRunResult
    ├── preflight.py    ← validate_circuit_for_zne, run_preflight_checks
    ├── submission.py   ← submit_all_then_collect, build_estimator_options
    ├── persistence.py  ← save_run, save_sweep_summary
    ├── observables.py  ← build_per_site_observables, map_observables_to_layout
    ├── phase.py        ← classify_phase
    └── spsa.py         ← spsa_refinement
```

## Methodology Invariants (NEVER violate)

### 0. Noisy Simulation Scalability Limits (HARD CONSTRAINTS)

- **FakeTorino**: ONLY usable for N≤10. OOM at N≥20 (133-qubit Target metadata).
- **BackendEstimatorV2 + AerSimulator(statevector)**: N≤20 (2^N memory).
- **MPS noisy (AerSimulator method=mps + noise_model)**: N≤100+, BUT does NOT
  include SWAP routing overhead → native chain circuits show negligible noise.
- **PEA/ZNE local validation**: ONLY valid at N≤10 with FakeTorino (routing adds
  noise). At N≥20, PEA gain=0% because native circuits are too clean.
- **PEA N≥20 validation**: REQUIRES real QPU (where physical routing adds noise).
- **HardwareBackend rehearsal**: N=10 ONLY. Do NOT pass `--n-qubits 20`.

### 1. Reproducibility

Every noisy estimation MUST pass `seed_simulator` and `default_precision`:
```python
estimator = BackendEstimatorV2(backend=backend, options={
    "seed_simulator": config.seed_simulator + offset,
    "default_precision": config.precision,    # = 1/√shots
})
```
**Violation**: Using `BackendEstimatorV2` without options → non-reproducible, wrong shot count.

### 2. Circuit State Before ZNE

All ZNE functions (`run_pea_zne`, `run_gate_folding_zne`, `run_block_zne`) require:
- Circuit is **already transpiled** (ISA-compliant)
- Circuit is **parameter-bound** (`num_parameters == 0`)
- Observable is **layout-mapped** (`H.apply_layout(transpiled.layout)`)

**Violation**: Passing parameterized circuit → `ValueError`. Passing unmapped H → wrong energy.

### 3. Noise Factor Semantics

| Strategy | Valid factors | Meaning |
|----------|-------------|---------|
| PEA | Any float ≥1 | Depolarizing rate multiplier |
| Gate-folding | Odd integers ≥1 | Number of effective gate applications |
| Block-ZNE | Odd integers ≥1 | Same as GF, but per-layer |

**Violation**: Even integer for GF → `ValueError`. Factor <1 → non-physical.

### 4. CES Metric

- ALWAYS use **circuit CES** (post-transpilation): `compute_circuit_ces(transpiled, backend)`
- NEVER use topology CES (pre-transpilation, ignores SWAP routing)
- CES = Σ(2Q gate error rates) over all 2Q gates in the transpiled circuit

### 5. R² Does NOT Guarantee Accuracy

GF-ZNE can produce R²=0.996 with ΔE/gap=89.8% (observed in HW_REHEARSAL_V2).
High R² means the extrapolation is **consistent** (good fit), not **accurate**
(correct physics). PEA's characterization-based amplification is the only
reliable accuracy indicator.

### 6. PEA Local Simulation Approximation

Our local PEA uses **isotropic depolarizing** as noise model. Real IBM PEA uses
full Pauli-Lindblad with 15 generators per 2Q pair. Expect:
- Simulation gain: ~95% (depolarizing is exactly linear)
- Hardware gain: ~30-60% (non-depolarizing components don't extrapolate)
- R² on hardware: >0.80 (acceptable), not 0.998 as in simulation

### 7. Affine Correction Is Zero-Cost Insurance

`affine_correct_energy()` clips to [E_ground, E_upper]. Apply ALWAYS after ZNE.
0% overshoot in 102 records → never modifies correct energies. Never harmful.

### 8. GNN-QEM vs PEA Are Alternatives

- GNN-QEM: removes structured noise by learned correction
- PEA: removes structured noise by characterization-based extrapolation
- **After one removes structure, residual is unstructured shot noise**
- Combining them: GNN over-corrects (validated: 0% improvement post-PEA)
- Deploy: PEA primary → affine (always). GNN-QEM only if PEA unavailable.

## Performance Optimization (2026-06-10)

### PEA Noise Model Filtering

`run_pea_zne()` filters noise pairs to circuit-relevant qubits:
- FakeTorino has 300 qubit pairs; N=6 circuit uses only ~20 relevant pairs
- Speedup: 5-10× for noise model construction (bit-exact, validated)
- Implementation: `_filter_rates_to_circuit()` + `_get_circuit_qubits()`
- Pre-builds ALL noise models before the measurement loop

### BackendEstimatorV2 Does NOT Transpile

Critical finding: `BackendEstimatorV2.run()` only applies `Optimize1qGatesDecomposition`
(measurement rotations, <1ms). It does NOT re-transpile the circuit. The actual
cost is in `backend.run()` (Aer simulation), not in transpilation overhead.

### What Is Actually Expensive (profiled)

| Component | Time (N=6) | Optimization |
|-----------|-----------|-------------|
| `depolarizing_error()` ×300 pairs ×3 factors | 1.8s (63%) | Filter to ~20 pairs |
| AerSimulator shot-based execution ×3 | 0.3s (10%) | Cannot reduce (physics) |
| `AerSimulator.from_backend()` ×3 | 0.04s (1%) | Negligible |
| `_learn_noise_rates()` | 0.3ms | Negligible |

## API Contracts

### noisy_estimate() — Canonical Noisy Evaluation

Single correct way to call `BackendEstimatorV2`:
```python
energy = noisy_estimate(transpiled, H_mapped, backend, config, seed_offset=0)
```
Do NOT call BackendEstimatorV2 directly outside this wrapper (risk: missing seed/precision).

### run_pea_zne() — Primary ZNE Strategy

```python
result = run_pea_zne(transpiled, H_mapped, backend, config, noise_factors=(1, 3, 5))
# Returns: PEAResult with .extrapolated_value, .r_squared, .measured_values
```
- Internally: learns rates → filters to circuit → pre-builds models → measures → extrapolates
- Thread-safe: each call gets independent AerSimulator instances
- Seed independence: `seed_offset + i*100` per noise factor

### run_adaptive_zne() — Tiered Strategy

```python
result = run_adaptive_zne(transpiled, H_mapped, backend, config, strategy="pea_primary")
# strategy="pea_primary" (default): PEA first, GF fallback if PEA unavailable
# strategy="gf_primary" (legacy): GF first, PEA if R² < threshold
```

### affine_correct_energy() — Physics Post-Processing

```python
corrected = affine_correct_energy(e_zne, e_ground=e_exact, n_qubits=N, h_value=h)
# Always apply after ZNE. Zero cost. No false positives in 102 records.
```

## Hardware Backend Orchestration

The `HardwareBackend.run_deployment()` pipeline:
```
1. Preflight checks (gate count, backend status)
2. validate_circuit_for_zne() — abort if CX > threshold
3. Layout selection (cached per circuit structure)
4. Submit circuits to all layouts
5. ZNE aggregation (mode-aware: IBM server-side vs local)
6. Per-site observables (phase classification)
7. Post-ZNE corrections: GNN-QEM (optional) → affine (always)
8. SPSA refinement (if ΔE/gap > threshold)
9. R²-gated verdict: PASS/FAIL/INDETERMINATE/PARTIAL
10. Persistence (save_run with full metadata)
```

### Mode-Aware ZNE Aggregation

- **hardware mode**: IBM Runtime applies ZNE server-side. Each layout returns
  already-mitigated energy. We average for √n variance reduction.
- **fake_backend mode**: Local GF/PEA ZNE via noisy_utils functions.
  Adaptive strategy (pea_primary) used by default.

## Critical File References

- #[[file:src/qmbp_simulation/execution/noisy_utils.py]]
- #[[file:src/qmbp_simulation/execution/hardware/backend.py]]
- #[[file:src/qmbp_simulation/execution/hardware/preflight.py]]
- #[[file:documentation/binnacles/binnacle-gate-folding-zne.md]]
- #[[file:documentation/binnacles/binnacle-performance-optimizations.md]]
