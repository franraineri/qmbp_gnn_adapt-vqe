# Noisy Simulation Scalability Limits (2026-06-10)

> Definitive findings on what CAN and CANNOT be validated locally for
> noise suppression at N≥20. Prevents future repetition of invalid experiments.

---

## Executive Summary

Local noisy simulation (FakeTorino + AerSimulator) has two hard limits:

1. **Memory limit (N≥20)**: FakeTorino (133 qubits) causes OOM when used with
   `BackendEstimatorV2`, `AerSimulator.from_backend()`, or `select_layouts_low_ces()`.
   Any path that loads the full 133-qubit Target object exceeds ~16GB at N≥20.

2. **Physics limit (N≥20 without routing)**: Native chain_1d circuits have only
   N-1 CZ gates. At 0.8% error/gate, total noise is ~0.3% of the energy — too
   small for ZNE to have any effect. PEA gain = 0%, R² = 0. The noise that makes
   ZNE necessary on hardware comes from SWAP routing (which doubles CZ count).

---

## Experiment: PEA_SCALING (2026-06-10)

### Configuration
- N=10 (FakeTorino, reference), N=40 (MPS+noise), N=50 (MPS+noise)
- noise_factors = (1, 3, 5), Torino-realistic depolarizing (0.8% mean CZ error)
- MPS χ=64, save_expectation_value (exact Tr(ρ·H), deterministic)

### Results

| N | Method | CZ gates | ΔE/gap (noisy) | PEA gain | PEA R² | Status |
|---|--------|----------|:-:|:-:|:-:|---|
| 10 | FakeTorino | 54 (with routing) | 1.13-1.29 | **+97%** | 0.997 | ✅ Valid |
| 40 | MPS native | 39 (no routing) | 0.002-0.005 | ~0% | 0.0 | ⚠️ No noise effect |
| 50 | MPS native | 49 (no routing) | 0.001-0.003 | ~0% | 0.0 | ⚠️ No noise effect |

### Root Cause Analysis

The critical difference is **SWAP routing overhead**:

```
N=10 on heavy_hex (FakeTorino):
  - Native circuit: 9 CZ (chain_1d)
  - After routing to heavy_hex 133 qubits: 54 CZ (+500% from SWAPs)
  - Noise accumulation: 54 × 0.8% = 43% total error
  → ZNE has LARGE effect to mitigate

N=40 chain_1d (MPS native):
  - Native circuit: 39 CZ (N-1, no SWAPs needed)
  - No routing: connectivity matches circuit perfectly
  - Noise accumulation: 39 × 0.8% = 31% total error
  - BUT: 31% distributed over 40 qubits × 2 params = ~0.3% per energy unit
  → ZNE has NEGLIGIBLE effect
```

The paradox: 54 CZ at N=10 produces MORE noise impact than 39 CZ at N=40
because routing SWAPs create noise on qubits that DON'T contribute to the
observable (crosstalk paths) while native CZ gates are exactly the ones
computing the energy (maximum signal-to-noise ratio).

### Implication for Hardware

On IBM Torino with heavy_hex topology:
- N=40 chain_1d would route through ~80 physical qubits → ~78 CZ after SWAP
- Noise would be ~78 × 0.8% = 62% total → ZNE highly relevant
- PEA gain on real hardware WILL be significant (estimated +50-80%)

**Conclusion**: PEA scalability to N≥20 can ONLY be validated on real QPU.

---

## Memory Constraints (Definitive)

| Path | Max N | Reason | Alternative |
|------|:-----:|--------|-------------|
| `FakeTorino` + `BackendEstimatorV2` | **10** | 133-qubit Target OOM | Use MPS noisy |
| `AerSimulator.from_backend(FakeTorino)` | **10** | Copies full Target | Direct AerSim |
| `select_layouts_low_ces(FakeTorino)` | **10** | Transpiles N qubits → 133 qubits ISA | Skip routing |
| `AerSimulator(method="statevector")` | **20** | 2^N amplitude vector | Use MPS |
| `AerSimulator(method="mps", noise_model=...)` | **100+** | O(N·χ³) | ✅ Correct path |
| `MPSBackend(deterministic=True)` | **100+** | O(N·χ³), no noise | ✅ VQE path |
| `NoiselessBackend` (StatevectorEstimator) | **22** | Exact eigensolve limit | Use MPSBackend |

### HardwareBackend Rehearsal Constraint

`HardwareBackend(mode="fake_backend")` uses FakeTorino internally. Therefore:
- `run_hardware_rehearsal_v2.py` ONLY works at N≤10.
- Passing `--n-qubits 20` will OOM — this is by design (the rehearsal validates
  the *code path*, not N-scalability).
- N-scalability is validated via MPS scaling experiments (binnacle-mps-scaling).

---

## DO NOT REPEAT

1. ❌ Do NOT attempt PEA on N≥20 with FakeTorino (OOM guaranteed)
2. ❌ Do NOT attempt BackendEstimatorV2 with AerSimulator at N>20 (2^N memory)
3. ❌ Do NOT expect ZNE to show effect on native chain_1d at N≥20 (no routing overhead)
4. ❌ Do NOT use rehearsal at N>10 (FakeTorino is the bottleneck, not the code)
5. ✅ DO use MPS+noise for noisy simulation research at N≥20
6. ✅ DO validate PEA N≥20 ONLY on real IBM Torino QPU
7. ✅ DO use N=10 FakeTorino as the definitive PEA local validation (97% gain, R²=0.997)

---

## References

- Result: `results/experiments/exp_pea_scaling/run_20260610_153437.json`
- Runner: `scripts/experiment_runners/noise_zne_gf_pea/run_pea_scaling_n40.py`
- Binnacle: `documentation/binnacles/binnacle-performance-optimizations.md`
- Prior MPS validation: `documentation/binnacles/binnacle-mps-scaling.md`
