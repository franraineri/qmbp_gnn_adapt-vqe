---
inclusion: fileMatch
fileMatchPattern: "**/noisy_utils*,**/zne*,**/mitigation*,**/gate_folding*,**/binnacle-gate-folding*"
---

# ZNE Mitigation Context (invoke with #context-zne-mitigation)

> Pre-digested context for all error mitigation strategies: PEA, gate-folding, adaptive, affine.

## Strategy Hierarchy (deployment order)

```
1. PEA-ZNE (primary)     → +94.4% gain, R²=0.998, 18/18 wins vs GF
2. Gate-folding (fallback) → +20.6% gain, R²>0.99, always positive
3. Affine correction (always) → clips to [E_ground, E_upper], zero cost
4. GNN-QEM (only if PEA unavailable) → +72% zero-shot, but NOT after PEA
```

## PEA (Probabilistic Error Amplification)

- Learns noise model via Pauli-Lindblad fitting, amplifies probabilistically.
- ~50% extra QPU overhead (noise learning phase). Justified by 4.6× gain vs GF.
- IBM Runtime: `options.resilience.zne.amplifier = "pea"` (handled server-side).
- Local simulation: `run_pea_zne()` in `execution/noisy_utils.py`.
- Cross-validated: chain_1d (+97%), ladder (+91%), heavy_hex (+98%), triangular (+97%).

## Gate-Folding ZNE

- Digital amplification: U → U·U†·U at factors [1, 3, 5].
- Simple, zero overhead, validated locally (R²>0.99 on chain_1d).
- **Fails on heavy_hex p=1 shallow circuits** (R²=0.47 — depth≤3 insufficient).
- Use only as fallback or when `qiskit-aer` not available.

## Block-Level ZNE (for p≥2)

- `run_block_zne()` folds only 1 HVA layer → shallower folded depth.
- Better linearity assumption than full-circuit folding.
- Ref: arXiv:2507.23314.

## Adaptive Strategy

```python
from qmbp_simulation.execution import MitigationOptions

# Default: PEA primary
opts = MitigationOptions(zne_amplifier="pea")

# Adaptive: tries GF first, falls back to PEA if R²<0.90
opts = MitigationOptions(zne_amplifier="adaptive", zne_r2_fallback_threshold=0.90)

# Legacy: gate-folding only
opts = MitigationOptions(zne_amplifier="gate_folding")
```

## Affine Correction

```python
from qmbp_simulation.execution import affine_correct_energy
# Clips ZNE energy to [E_ground, E_upper]. Zero cost. Always apply.
# 0% overshoot in 102 ZNE records (safety net confirmed).
e_corrected = affine_correct_energy(e_zne, e_ground, e_upper)
```

## CX Budget Rule

| Config | CX gates | ZNE viable? |
|--------|----------|-------------|
| p=1 N=6 | 10 | ✅ |
| p=1 N=10 | 18 | ✅ (threshold) |
| p=2 N=6 | 18 | ✅ (barely) |
| p=2 N=10 | 36 | ❌ too deep |
| p=1 N=20 | 38 | ❌ but PEA might handle |
| p=1 N=40 | 78 | ✅ PEA viable |
| p=1 N=50 | 98 | ✅ PEA viable |

**Rule**: ~18 CX is the GF-ZNE threshold. PEA handles deeper circuits (validated to 98 CX).

## CES-ZNE (DEPRECATED on heavy_hex)

- Different layouts → different CES → linear extrapolation.
- **Fails**: All good heavy_hex layouts have CES≈0.15 (no spread for extrapolation).
- Only use for legacy compatibility or topologies with CES diversity.

## Hardware Integration

```python
# IBM Runtime (real QPU)
estimator.options.resilience.zne_mitigation = True
estimator.options.resilience.zne.noise_factors = [1, 2, 3]
estimator.options.resilience.zne.amplifier = "pea"  # or "gate_folding"

# Local simulation (FakeTorino)
from qmbp_simulation.execution.hardware import HardwareBackend, HardwareConfig
config = HardwareConfig(mode="fake_backend", n_qubits=10, shots=16384)
backend = HardwareBackend(config=config)  # PEA-ZNE by default
```

## Key Results

| Experiment | Strategy | Gain | R² | Stat sig |
|-----------|----------|------|-----|----------|
| ZNE_CROSS_TOPO | PEA | +94.4% | 0.998 | t=46.32, p<10⁻¹⁹ |
| GF_ZNE_CMP | Gate-folding | +12% | >0.99 | wins 9/12 vs CES |
| PEA_TRIANGULAR | PEA | +96.8% | — | t=111.22, 9/9 wins |
| PEA_PIPELINE | PEA + MPNN θ | +81% | — | works with pred error |

## DO NOT

- Use CES-ZNE on heavy_hex (R²=0.04, gain=0%).
- Apply GNN-QEM after PEA (over-corrects residuals).
- Use GF-ZNE R²>0.99 as proof of accuracy (89.8% ΔE/gap observed with "good" R²).
- Use gate-folding on p=1 heavy_hex (depth≤3 → meaningless extrapolation).
- Skip affine correction (zero cost, zero risk).
- Use noise_factors beyond [1,3,5] for GF (diminishing returns, more circuits).

## TLS Drift Monitoring (hardware runs)

```python
from qmbp_simulation.execution import take_calibration_snapshot, check_calibration_drift
# Take snapshot before run, check between h-points
# Abort if T1 drift > 20%
snapshot = take_calibration_snapshot(backend)
drift = check_calibration_drift(snapshot, backend)
if drift.t1_relative_change > 0.20:
    raise CalibrationDriftError(...)
```

## Source Files

- #[[file:src/qmbp_simulation/execution/noisy_utils.py]]
- #[[file:src/qmbp_simulation/execution/hardware.py]]
- #[[file:documentation/binnacles/binnacle-gate-folding-zne.md]]
- #[[file:documentation/analysis/13_hardware_zne_improvements.md]]
- #[[file:documentation/analysis/15_advanced_mitigation_techniques.md]]
- #[[file:results/experiments/exp_zne_cross_topo/]]
