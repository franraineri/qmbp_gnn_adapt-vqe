# Transpiler Exploration — Findings (2026-06-05)

## Objective

Evaluate whether advanced transpilation options can reduce circuit depth/noise
for HVA p=1 N=10 heavy_hex hardware deployment.

## Methods Tested

1. `optimization_level` 0, 1, 2, 3 (standard Qiskit preset pass managers)
2. `approximation_degree` < 1.0 at level 3 (KAK approximate synthesis)
3. `PauliEvolutionGate` circuit representation (exposes commutativity)
4. Rustiq plugin (`HLSConfig` with `nshuffles=400`)
5. Noise suppression stack (DD + Twirling + TREX + PEA-ZNE)

## Key Results

### Transpiler Levels (same layout, Orig HVA)

| Level | Total Depth | 2Q Gates | CES | Compile Time |
|-------|:-----------:|:--------:|:---:|:------------:|
| 0 | 127 | 36 | 0.1334 | 0.01s |
| 1 | 105 | 36 | 0.1334 | 0.02s |
| **2** | **80** | **34** | **0.1251** | 0.06s |
| 3 | 89 | 34 | 0.1271 | 0.17s |

**Level 3 provides zero benefit.** KAK resynthesis targets multi-gate 2Q
unitary blocks. HVA uses individual RZZ gates — nothing to consolidate.

### PauliEvolutionGate Representation (same CES-optimized layout)

| Method | Total Depth | 2Q-Depth | n_2Q | CES |
|--------|:-----------:|:--------:|:----:|:---:|
| Orig HVA + level 2 (current) | 88 | 27 | 34 | 0.1271 |
| **PauliEvol + level 2** | **81** | **24** | 34 | **0.1251** |

**11% reduction in 2Q-depth** — free improvement, no gate count change.
The transpiler recognizes that ZZ rotations within a layer commute and
schedules them in parallel.

### Rustiq Plugin

| Method | 2Q-Depth | n_2Q |
|--------|:--------:|:----:|
| PauliEvol + Rustiq | 50 | 67 |

**Counterproductive.** Rustiq is designed for dense Pauli networks (e.g.
molecular Hamiltonians with many non-commuting terms). Our HVA has only
9 ZZ + 10 X terms already grouped into commuting layers. Rustiq's
resynthesis produces MORE gates.

## Implementation

Added `HVACircuitBuilder.create_pauli_evolution()` to `src/qmbp_simulation/circuits/hva.py`.
Uses `PauliEvolutionGate(H_zz, time=2*theta_zz)` + `PauliEvolutionGate(H_x, time=2*theta_x)`
instead of explicit RZZ/RX loops. Functionally identical (same energy to 1e-10).

## Noise Suppression Stack (unchanged)

The full stack is correctly configured:
- **DD** (XpXm): suppresses idle coherent errors
- **Twirling** (32 randomizations): coherent → stochastic
- **TREX**: readout error mitigation
- **PEA-ZNE** (factors 1,3,5): +94.4% gain, R²=0.998

No changes needed — already optimal per IBM recommendations.

## References

- IBM: [Set transpiler optimization level](https://quantum.cloud.ibm.com/docs/guides/set-optimization)
- IBM: [Compilation methods for Hamiltonian simulation circuits](https://quantum.cloud.ibm.com/docs/tutorials/compilation-methods-for-hamiltonian-simulation-circuits)
- IBM: [Error mitigation and suppression techniques](https://quantum.cloud.ibm.com/docs/guides/configure-error-suppression)
- Qiskit 2.4.1: `generate_preset_pass_manager`, `HLSConfig`, Rustiq plugin
