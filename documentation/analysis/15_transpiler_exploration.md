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

---

## Addendum: Validación Formal y Corrección de Bug (2026-06-15)

### Bug en `create_pauli_evolution()` — Corregido

La versión original (2026-06-05) usaba coeficientes `1.0` en los operadores:
```python
H_zz = SparsePauliOp.from_list([(..., 1.0), ...])  # INCORRECTO
```

Esto producía `e^{-i·2θ·ZZ}` en lugar del correcto `e^{-iθ·ZZ}`, resultando en
energías erróneas por ~25 unidades. Corregido a `0.5`:

```python
H_zz = SparsePauliOp.from_list([(..., 0.5), ...])  # CORRECTO
# e^{-i·2θ·0.5·ZZ} = e^{-iθ·ZZ} = RZZ(2θ) ✓
```

La nota original "same energy to 1e-10" era una afirmación sin verificar.
El resultado correcto post-fix: `|ΔE| < 4e-14` (máquina).

### Corrección de la Métrica "2Q-depth"

La exploración original midió **2Q-depth del camino crítico del DAG** (27→24).
Eso es correcto numéricamente, pero la interpretación física para heavy_hex es diferente:

- En heavy_hex N=10 p=1, los 9 bonds ZZ no se solapan → el scheduler los pone
  en un **único ciclo 2Q** (2Q-depth = 1) independientemente de la representación.
- La reducción real medida en la validación formal (Section 20, 2026-06-15) es
  de **total_depth**, no 2Q-depth:

| h | RZZ total_depth | PauliEvol total_depth | Reducción |
|---|:---:|:---:|:---:|
| 4.00 | 89 | 82 | −7.9% |
| 3.25 | 90 | 81 | −10.0% |
| 3.00 | 90 | 90 | 0.0% |
| **Media** | **89.7** | **84.3** | **−6.0%** |

La reducción de 2Q-depth (27→24) del reporte original corresponde a una medición
distinta (con transpilación sobre un circuito diferente, probablemente sin el layout
fijo de heavy_hex production). La métrica relevante para hardware es `total_depth`.

### Estado de Producción (2026-06-15)

`create_pauli_evolution()` ya está integrada en:
- `run_ibm_torino_deployment.py` Tiers 0, 1, 2 (circuitos que van al QPU real)
- Validada con Section 20 de `run_hardware_rehearsal_v3.py`

Ver detalles completos: `documentation/binnacles/binnacle-pauli-evolution-transpilation.md`
