# Binnacle — PauliEvolutionGate Transpilation Validation

> **Fecha**: 2026-06-15
> **Sección relevante**: Section 20 de `run_hardware_rehearsal_v3.py`
> **Script de validación**: `run_hardware_rehearsal_v3.py --section 20`
> **Archivos modificados**: `src/qmbp_simulation/circuits/hva.py`,
> `scripts/experiment_runners/hardware/run_ibm_torino_deployment.py`
> **Estado**: ✅ VALIDADO — en producción (Tiers 0-2 del deployment script)

---

## Resumen Ejecutivo

`HVACircuitBuilder.create_pauli_evolution()` usa `PauliEvolutionGate` en lugar de
gates `RZZ`/`RX` explícitos. Expone la estructura conmutativa de cada capa HVA al
transpilador de Qiskit, permitiendo mejor scheduling paralelo y reduciendo el
`total_depth` del circuito entre 6% y 10% en heavy_hex N=10 p=1.

**Resultado clave**: Funcionalmente idéntico (`|ΔE| < 1e-13` en todos los casos),
pero con un circuito más corto que reduce la exposición a decoherencia temporal en
hardware real. Los Tiers 0, 1, y 2 de IBM Torino deployment ya usan esta representación.

---

## Contexto y Motivación

La exploración de transpiladores del 2026-06-05 (`15_transpiler_exploration.md`)
reportó una reducción de "2Q-depth de 27→24 (−11%)". Sin embargo, ese reporte medía
el **camino crítico del DAG** (2Q-depth), no el `total_depth`. En la validación formal
de 2026-06-15 (Section 20) se descubrió:

- En **heavy_hex**, todos los bonds ZZ son **no-solapantes** → el scheduler ya los
  paraleliza en un único ciclo 2Q, independientemente de si se usa `create()` o
  `create_pauli_evolution()`.
- Por eso la 2Q-depth del DAG es **1 para ambas representaciones** en heavy_hex.
- La métrica correcta para hardware es **total_depth**, que incluye los gates de un
  qubit entre los ciclos 2Q. Esa sí difiere: PauliEvol genera un scheduling más
  compacto de los gates RX (campo transverso).

---

## Bug Corregido: Coeficiente en `create_pauli_evolution()`

### Antes (incorrecto — 2026-06-05 hasta 2026-06-15)

```python
# Coefficients = 1.0 → wrong unitary
H_zz = SparsePauliOp.from_list([("ZZ...", 1.0), ...])
H_x  = SparsePauliOp.from_list([("X...", 1.0), ...])
# PauliEvolutionGate(H_zz, time=2θ) = e^{-i·2θ·1.0·ZZ} ≠ RZZ(2θ) = e^{-iθ·ZZ}
```

**Impacto**: Energía errónea en 25 unidades (|ΔE| = 25 vs. |ΔE| < 1e-14 correcto).
El método `create_pauli_evolution()` no se usaba en producción (solo `create()`) así
que este bug nunca afectó ningún resultado publicado.

### Después (correcto — 2026-06-15)

```python
# Coefficients = 0.5 → correct unitary
H_zz = SparsePauliOp.from_list([("ZZ...", 0.5), ...])
H_x  = SparsePauliOp.from_list([("X...", 0.5), ...])
# PauliEvolutionGate(H_zz, time=2θ) = e^{-i·2θ·0.5·ZZ} = e^{-iθ·ZZ} = RZZ(2θ) ✓
```

**Convención**: `PauliEvolutionGate(H, time=t)` implementa `e^{-itH}`.
Para igualar `RZZ(2θ) = e^{-iθ·ZZ}`: necesitamos `H = 0.5·ZZ` con `time = 2θ`.
Análogamente `RX(2θ) = e^{-iθ·X}`: necesitamos `H = 0.5·X` con `time = 2θ`.

---

## Resultados de Section 20 (validación formal)

**Config**: topology=heavy_hex, N=10, p=1, 3 h-test points [4.0, 3.25, 3.0]
**Layout compartido**: qubits [24, 25, 23, 26, 35, 16, 22, 27, 44, 4], CES=0.1251

### Tabla de métricas transpiladas (optimization_level=2, FakeTorino)

| h | RZZ total_depth | PauliEvol total_depth | Reducción | n_2Q | \|ΔE\| |
|---|:---:|:---:|:---:|:---:|:---:|
| 4.00 | 89 | 82 | **+7.9%** | 34 (igual) | 3.6e-14 |
| 3.25 | 90 | 81 | **+10.0%** | 34 (igual) | 1.4e-14 |
| 3.00 | 90 | 90 | 0.0% | 34 (igual) | 2.1e-14 |
| **Media** | **89.7** | **84.3** | **+6.0%** | **34** | **3.5e-14** |

### Nota sobre 2Q-depth = 1 en ambos casos

En heavy_hex N=10 p=1, los 9 bonds ZZ conectan qubits no-adyacentes entre sí
(grafo planar). El scheduler de Qiskit los coloca automáticamente en paralelo
independientemente de la representación. Por eso:
- `2Q-depth` del camino crítico = **1** para RZZ/RX y PauliEvol.
- El beneficio es de scheduling de los gates de 1 qubit entre ciclos, no de
  paralelización adicional de los CX.

---

## Veredicto Section 20 (FakeTorino)

```
✅ PASS — Section 20 (27.6s)

    energy_identity:    max|ΔE| = 3.55e-14  (✓ < 1e-8)
    total_depth:        mean = +6.0%          (✓ any reduction)
    n_2Q unchanged:     True                  (✓ same gate count)
    noisy ΔE/gap diff:  0.67%                 (informational)

    Recommendation: USE PauliEvolutionGate for IBM Torino deployment
```

**Por qué `noisy_de_gap_diff = 0.67%` es esperado**:
FakeTorino modela ruido **por gate** (no por tiempo). Si n_2Q = 34 igual en
ambos, el ruido acumulado es el mismo. La diferencia del 0.67% es ruido
estadístico de shot noise (shots=16384). En hardware **real** (IBM Torino), el
ruido depende del tiempo de ejecución → la reducción de total_depth SÍ importa.

---

## Cambios en Producción

### 1. `src/qmbp_simulation/circuits/hva.py`

- **Fix**: Coeficientes ZZ y X corregidos de `1.0` → `0.5`
- **Docstring**: Actualizado con datos reales de Section 20, convención del
  coeficiente, nota sobre 2Q-depth=1 en heavy_hex

### 2. `scripts/experiment_runners/hardware/run_ibm_torino_deployment.py`

Aplicado en todos los Tiers que van al QPU real:

```python
# Tier 0: run_tier_0()
circuit, _ = circuit_builder.create_pauli_evolution(N_QUBITS, P_LAYERS, lattice)

# Tier 1: run_tier_1()
circuit, _ = circuit_builder.create_pauli_evolution(N_QUBITS, P_LAYERS, lattice)

# Tier 2: run_tier_2()
circuit, _ = circuit_builder.create_pauli_evolution(N_QUBITS, P_LAYERS, lattice)
```

**No cambiado** (correcto usar `create()` para noiseless):
- `prepare_mpnn_predictions()` — VQE de training, StatevectorEstimator
- `compute_kappa_per_h()` — curvatura noiseless, StatevectorEstimator
- `run_tier_3()` — usa `tfim_longitudinal` que no tiene versión PauliEvol

### 3. `scripts/experiment_runners/run_hardware_rehearsal_v3.py`

- Section 20 añadida: comparación completa RZZ/RX vs PauliEvolutionGate
- CLI `--skip-pauli-evolution` para saltarla
- Criterio de PASS: `energy_identity AND any_depth_reduction`

---

## Por Qué los Scripts de Simulación No Cambian

Los scripts de VQE noiseless usan `StatevectorEstimator` directamente (sin
transpilación a backend físico). Para ese caso, `create()` y `create_pauli_evolution()`
son equivalentes pero `create()` es más directo (sin PauliEvolutionGate synthesis).

El beneficio de scheduling de PauliEvolutionGate **solo existe durante la
transpilación al backend físico** (real o FakeTorino con optimization_level≥2).
Para StatevectorEstimator no hay transpilación → ninguna diferencia.

Esto incluye todos los scripts en:
- `scripts/experiment_runners/scaling/`
- `scripts/experiment_runners/cross_topology/`
- `scripts/experiment_runners/bond_resolved/`
- `src/qmbp_simulation/models/model_registry.py` → `_create_tfim()`

---

## Recomendaciones para Trabajo Futuro

1. **`tfim_longitudinal`** no tiene versión PauliEvol — trivial de añadir si se
   necesita para hardware (mismo patrón: añadir `create_pauli_evolution_longitudinal()`
   con coeficientes 0.5 para ZZ, X, Z).

2. **La diferencia entre h=3.0 (0%) y h=4.0 (7.9%)** sugiere que la ganancia
   depende de los valores de theta. A theta cercanos a 0 (h grande, parámetros
   triviales), el transpilador ya cancela gates y ambas representaciones colapsan.
   A theta no triviales (h más cercano a h_c), la ganancia es mayor.

3. **En hardware real (IBM Torino)**, se espera un impacto más visible que en
   FakeTorino porque el modelo de ruido es temporal (T1/T2), no solo por gate.

---

## Referencias

| Documento | Relevancia |
|-----------|-----------|
| `documentation/analysis/15_transpiler_exploration.md` | Exploración inicial 2026-06-05 (datos pre-bug-fix) |
| `src/qmbp_simulation/circuits/hva.py` | Implementación con bug fix |
| `scripts/experiment_runners/run_hardware_rehearsal_v3.py` | Section 20 |
| IBM tutorial: Compilation methods for Hamiltonian simulation circuits | Base del enfoque |
