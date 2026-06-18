# Mitiq Integration Plan — Módulo de Mitigación Complementario

**Fecha**: 2026-06-17
**Objetivo**: Integrar Mitiq como módulo escalable complementario a nuestro stack PEA/GF/adaptive ZNE,
habilitando CDR, DDD+ZNE stacking, y simulaciones noisy enriquecidas para validación pre-hardware.

---

## 1. Análisis: ¿Es útil Mitiq para simulaciones noisy vs. nuestros métodos actuales?

### 1.1 Comparación de enfoque noisy actual vs. Mitiq

| Aspecto | Nuestro stack actual | Mitiq |
|---------|---------------------|-------|
| **Noise model** | FakeTorino (calibration-aware, 133q) | Depolarizing/custom (programmatic) |
| **ZNE folding** | Determinístico (fold all 2Q gates) | Random, global, from_left, from_right |
| **ZNE extrapolation** | Linear o exponencial | Linear, Richardson, Poly, Exp con asíntota |
| **PEA** | ✅ Learned noise amplification (IBM) | ❌ No tiene |
| **CDR** | ❌ No tenemos | ✅ Learning-based con near-Clifford circuits |
| **DDD** | Via transpiler pass (hardware) | Software-level, compositivo con ZNE |
| **PEC** | ❌ No tenemos | ✅ Exacto pero exponencial |
| **Composabilidad** | Sequential (affine post-ZNE) | Nativa (DDD+ZNE, CDR+ZNE stacking) |
| **Noise model fidelity** | Alta (FakeTorino reproduce IBM Heron) | Configurable (desde simple depol hasta custom) |

### 1.2 Valor de Mitiq para noisy simulations

**Sí es útil**, pero con matices:

1. **CDR es la contribución más valiosa** (arXiv:2011.01157):
   - vnCDR mejora el error absoluto de energía en un factor 33× sobre resultados sin mitigar
   - Factor 20× sobre ZNE y 1.8× sobre CDR básico para Ising 8-qubit
   - No requiere noise model → robusto a cambios de calibración
   - Usa circuitos near-Clifford como training data (simulables clásicamente)

2. **Random folding aporta robustez** sobre nuestro fold determinístico:
   - Distribución estadística de resultados ZNE (múltiples folds)
   - Reduce sesgo de folding parcial en circuitos cortos (nuestro N=10 p=1: 17 CZ)

3. **DDD+ZNE composition** agrega una capa sin costo adicional de gates:
   - DDD inserta secuencias de identidad en idle periods
   - La composición DDD→ZNE mitiga coherent errors + incoherent errors

4. **PEC es solo para benchmarking**:
   - Overhead exponencial → no viable en producción
   - Útil para establecer "ground truth mitigada" contra la cual medir CDR/ZNE

### 1.3 Cuándo Mitiq NO reemplaza nuestros métodos

- **PEA-ZNE (+94.4% gain)** sigue siendo primario para hardware real: usa learned noise model
  (Sparse Pauli-Lindblad), amplifica solo el ruido real, no el coherente
- **Affine correction** (zero-cost, physics-bounds): Mitiq no tiene equivalente
- **FakeTorino** es mejor modelo de ruido que depolarizing uniforme para validación pre-hardware
- **Nuestro adaptive ZNE** (PEA→GF fallback) es más sofisticado que cualquier factory sola

### 1.4 Estrategia recomendada

```
Hardware real:     PEA-ZNE (primary) → Affine → Mitiq CDR (verification)
Noisy simulation:  Mitiq ZNE/CDR comparison + Nuestro PEA local (validation crosscheck)
Benchmark offline: Mitiq PEC (exponential reference) vs all methods
```

---

## 2. Fases de Implementación

### Fase 1: Foundation — Módulo base y executor adapter
**Prioridad**: Alta | **Esfuerzo**: ~2h | **Bloquea**: Todas las demás

**Tareas**:
- [ ] Crear `src/qmbp_simulation/execution/mitiq_utils.py`
- [ ] Implementar `MitiqExecutorFactory`: adapta `(backend, observable, config)` → `executor(circuit) → float`
- [ ] Implementar `_check_mitiq_available()`: lazy import guard
- [ ] Agregar `mitiq = ["mitiq>=0.38"]` a `pyproject.toml` optional-dependencies
- [ ] Agregar exports a `src/qmbp_simulation/execution/__init__.py`
- [ ] Implementar `MitiqNoisyExecutor`: envuelve FakeTorino/AerSimulator con nuestro NoisyEstimatorConfig
- [ ] Implementar `MitiqNoiselessExecutor`: StatevectorEstimator (para CDR training)

**Diseño del executor adapter**:
```python
def make_mitiq_executor(
    observable: SparsePauliOp,
    backend,
    config: NoisyEstimatorConfig,
    transpile: bool = True,
) -> Callable[[QuantumCircuit], float]:
    """Adapta nuestro backend+config al patrón executor(circuit) → float de Mitiq.

    Mitiq opera sobre circuitos sin transpile — el executor se encarga
    de transpile + estimación + retornar escalar.
    """
```

**Constraint**: El executor DEBE manejar que Mitiq pasa circuitos con gates foldeados
(potencialmente más profundos). Si `transpile=True`, re-transpila cada circuito foldeado.

---

### Fase 2: Mitiq ZNE — Random folding + factories ricas
**Prioridad**: Alta | **Esfuerzo**: ~2h | **Bloquea**: Fase 4

**Tareas**:
- [ ] Implementar `MitiqZNEConfig` dataclass (factory, folding_method, scale_factors, asymptote)
- [ ] Implementar `MitiqZNEResult` dataclass (compatible con nuestro patrón: extrapolated_value, r_squared, etc.)
- [ ] Implementar `run_mitiq_zne()` — wrapper sobre `mitiq.zne.execute_with_zne()`
- [ ] Soportar folding methods: `"random"`, `"global"`, `"from_left"`, `"from_right"`
- [ ] Soportar factories: `"linear"`, `"richardson"`, `"exponential"`, `"poly"`
- [ ] Soportar scale_factors fraccionarios: (1.0, 1.5, 2.0, 2.5, 3.0) — ventaja sobre nuestro GF (solo enteros impares)
- [ ] Implementar `run_mitiq_zne_batch()` — ejecución sobre múltiples h-points
- [ ] Validar: comparar resultado Mitiq-ZNE vs nuestro `run_gate_folding_zne()` en N=6 chain_1d

**Análisis valor vs. nuestro GF-ZNE**:
- Mitiq ZNE con random folding → ensemble de extrapolaciones → error bars estadísticos
- Factories ricas (Richardson, Poly) → mejor extrapolación para circuitos con non-linear E(λ)
- Scale factors fraccionarios → 5 puntos en rango [1,3] vs nuestros 3 puntos {1,3,5}
- **Desventaja**: No usa learned noise model (PEA es mejor para hardware real)

---

### Fase 3: Mitiq CDR — La contribución principal
**Prioridad**: Alta | **Esfuerzo**: ~3h | **Bloquea**: Fase 5, Fase 7

**Tareas**:
- [ ] Implementar `MitiqCDRConfig` dataclass (n_training_circuits, fit_method, seed)
- [ ] Implementar `MitiqCDRResult` dataclass (mitigated_value, improvement_pct, model_coefficients, training_circuits_used)
- [ ] Implementar `run_mitiq_cdr()` — wrapper sobre `mitiq.cdr.execute_with_cdr()`
- [ ] Construir `noiseless_executor` para CDR (usar StatevectorEstimator para training data)
- [ ] Implementar selección de near-Clifford circuits para HVA (respetar estructura del ansatz)
- [ ] Implementar `run_mitiq_cdr_batch()` para sweep de h-points
- [ ] Validar: CDR sobre N=6 p=2 FakeTorino vs ZNE vs raw

**Nota sobre CDR para nuestro caso**:
- CDR requiere ejecutar ~10 near-Clifford circuits en noisy Y noiseless
- Para N=10 p=1 (17 CZ): ~10 circuits × 2 (noisy+noiseless) = 20 ejecuciones extra
- Overhead: ~150% sobre raw (vs PEA ~50%) pero sin noise model → robusto a drift
- CDR aprende la relación `E_noisy → E_ideal` via regresión lineal en Clifford data
- Funciona especialmente bien cuando la relación E_noisy/E_ideal es cuasi-lineal
  (lo cual es cierto para TFIM en régimen paramagnético h>h_c)

---

### Fase 4: DDD+ZNE Composition — Stack de mitigación
**Prioridad**: Media | **Esfuerzo**: ~2h | **Bloquea**: Fase 7

**Tareas**:
- [ ] Implementar `MitiqDDDConfig` dataclass (rule: "xx"|"yy"|"xyxy", num_trials)
- [ ] Implementar `MitiqDDDZNEResult` dataclass
- [ ] Implementar `run_mitiq_ddd_zne()` — DDD como inner + ZNE como outer
- [ ] Soportar reglas DDD: XX, YY, XYXY (estándar para qubits superconductores)
- [ ] Validar en FakeTorino N=10 p=1: ¿DDD+ZNE > ZNE solo?
- [ ] Documentar cuándo la composición ayuda vs. cuándo introduce overhead innecesario

**Análisis DDD para nuestro caso**:
- DDD mitiga T2 decoherence en idle periods
- Nuestros circuitos HVA p=1 N=10 son cortos (~17 CZ, depth ~10)
- Idle periods son mínimos → beneficio DDD probablemente marginal
- Pero en heavy_hex con routing, idle periods aparecen → DDD puede ayudar
- **Veredicto**: Implementar, medir, activar solo si mejora > 3%

---

### Fase 5: PEC Benchmark — Reference offline
**Prioridad**: Baja | **Esfuerzo**: ~2h | **No bloquea nada**

**Tareas**:
- [ ] Implementar `MitiqPECConfig` dataclass (n_samples, noise_model_type)
- [ ] Implementar `MitiqPECResult` dataclass (mitigated_value, overhead_factor, variance)
- [ ] Implementar `run_mitiq_pec()` — wrapper sobre `mitiq.pec.execute_with_pec()`
- [ ] Construir noise model representación compatible con Mitiq PEC format
- [ ] Validar: PEC como gold standard vs CDR vs ZNE en N=6 (solo offline, nunca hardware)
- [ ] Documentar overhead: n_samples necesarios, tiempo de ejecución

**Nota**: PEC tiene overhead O(exp(n_2q * ε)) donde ε es error rate por gate.
Para N=10 (17 CZ, ε≈0.003 en Heron R2): overhead ≈ exp(17×0.003) ≈ 1.05× → **sorpresivamente viable para nuestros circuitos cortos**.
Esto cambia el cálculo — PEC podría ser production-viable para N=10 p=1.

---

### Fase 6: Noisy Simulation Pipeline via Mitiq
**Prioridad**: Alta | **Esfuerzo**: ~3h | **Bloquea**: Fase 7

**Tareas**:
- [ ] Implementar `MitiqNoisySimulator`: clase que encapsula FakeTorino + Mitiq mitigation
- [ ] Soportar modes: `"raw"`, `"zne"`, `"cdr"`, `"ddd_zne"`, `"pec"`, `"cdr_zne"`
- [ ] Implementar `run_mitiq_noisy_comparison()`: ejecuta raw + todos los métodos y produce tabla
- [ ] Integrar con `NoisyEstimatorConfig` para reproducibilidad (seeds, shots)
- [ ] Soportar `initialized_depolarizing_noise()` de Mitiq como modelo simple
- [ ] Soportar FakeTorino noise model como modelo realista
- [ ] Producir artefactos: `results/mitiq/comparison_*.json`
- [ ] Implementar `MitiqComparisonReport`: tabla de methods × h-points con ΔE/gap, R², ranking

**Diseño de la comparación noisy**:
```python
@dataclass
class MitiqNoisyComparisonResult:
    """Resultado de comparación multi-método para un h-point."""
    h_value: float
    e_exact: float
    gap: float
    raw_energy: float
    methods: dict[str, MitigatedResult]  # method_name → result
    best_method: str
    best_delta_e_gap: float
    rankings: list[str]  # sorted best→worst
```

---

### Fase 7: Strategy Comparator & Integration con Hardware Deployment
**Prioridad**: Alta | **Esfuerzo**: ~3h | **Bloquea**: Rehearsal + Deployment

**Tareas**:
- [ ] Implementar `compare_mitigation_strategies()` — ejecuta todos los métodos disponibles
- [ ] Agregar `MitiqMitigationOptions` a `MitigationOptions` (nueva sección)
- [ ] Integrar CDR como verification layer en `HardwareBackend.run_deployment()`
- [ ] Agregar flag `--mitiq-cdr` al deployment script
- [ ] Agregar flag `--mitiq-benchmark` para comparison offline
- [ ] Integrar en hardware rehearsal V3 como sección nueva (Section 20+)
- [ ] Persistir resultados Mitiq en `HardwareRunResult` (campos nuevos)
- [ ] Documentar decisión tree: cuándo activar cada método

**Decision tree para hardware**:
```
IF qiskit-aer available AND PEA R²>0.90:
    PRIMARY: PEA-ZNE (nuestro, +94.4% gain)
    VERIFY:  Mitiq CDR (cross-check independiente)
ELIF PEA unavailable:
    PRIMARY: Mitiq CDR (learning-based, no noise model)
    FALLBACK: Mitiq ZNE (random folding + Richardson)

ALWAYS:
    POST-PROCESS: Affine correction (nuestro, zero cost)

IF benchmark_mode:
    RUN ALL: PEA, GF, Mitiq-ZNE, Mitiq-CDR, Mitiq-DDD+ZNE, Mitiq-PEC
    PRODUCE: comparison table for thesis
```

---

### Fase 8: Tests y Validación
**Prioridad**: Alta | **Esfuerzo**: ~2h | **Bloquea**: CI green

**Tareas**:
- [ ] Crear `tests/test_mitiq_integration.py`
- [ ] Test: `test_executor_factory_returns_float` — executor produce float para circuito simple
- [ ] Test: `test_mitiq_zne_improves_over_raw` — ZNE mitigado < raw en N=4 depolarizing
- [ ] Test: `test_mitiq_cdr_basic` — CDR funciona con near-Clifford training (N=4)
- [ ] Test: `test_mitiq_ddd_zne_composition` — DDD+ZNE no crashea, produce float
- [ ] Test: `test_comparison_report_structure` — comparison produce JSON serializable
- [ ] Test: `test_mitiq_unavailable_graceful` — sin mitiq instalado → ImportError claro
- [ ] Agregar `mitiq` a CI test matrix (optional — skip if not installed)
- [ ] Marcar tests con `@pytest.mark.mitiq` para filtrado

---

### Fase 9: Documentación y Binnacle
**Prioridad**: Media | **Esfuerzo**: ~1h

**Tareas**:
- [ ] Crear `documentation/binnacles/binnacle-mitiq-integration.md`
- [ ] Registrar resultados de comparación: CDR vs PEA vs GF para N=6, N=10
- [ ] Actualizar `documentation/analysis/13_hardware_zne_improvements.md` con Mitiq options
- [ ] Actualizar `project-status.md` (steering) con Mitiq module status
- [ ] Agregar entrada a code-style.md para imports de mitiq_utils
- [ ] Agregar referencia en README o deployment spec

---

## 3. Diseño Técnico Detallado

### 3.1 Módulo DAG (sin circular imports)

```
execution/noisy_utils.py      ← NO importa mitiq (producción sin mitiq)
execution/mitiq_utils.py      ← Lazy import de mitiq
                               ← Importa de noisy_utils (NoisyEstimatorConfig)
                               ← Importa de backends (ExecutionBackend, MitigationOptions)
execution/hardware/backend.py ← Puede importar mitiq_utils (lazy, para CDR verification)
```

### 3.2 Dataclasses (output pattern)

Todas siguen el patrón existente: typed dataclass con campos tipados, JSON-serializable.

```python
@dataclass
class MitiqZNEResult:
    extrapolated_value: float
    r_squared: float
    factory_name: str           # "linear"|"richardson"|"poly"|"exp"
    folding_method: str         # "random"|"global"|"from_left"
    scale_factors: list[float]
    measured_values: list[float]
    execution_time_s: float

@dataclass
class MitiqCDRResult:
    mitigated_value: float
    raw_value: float
    improvement_pct: float
    n_training_circuits: int
    model_coefficients: list[float]
    training_energies_noisy: list[float]
    training_energies_ideal: list[float]
    execution_time_s: float

@dataclass
class MitiqDDDZNEResult:
    extrapolated_value: float
    r_squared: float
    ddd_rule: str               # "xx"|"yy"|"xyxy"
    zne_factory: str
    scale_factors: list[float]
    measured_values: list[float]
    execution_time_s: float

@dataclass
class MitiqPECResult:
    mitigated_value: float
    raw_value: float
    improvement_pct: float
    n_samples: int
    overhead_factor: float
    variance: float
    execution_time_s: float
```

### 3.3 pyproject.toml changes

```toml
[project.optional-dependencies]
hardware = ["qiskit-ibm-runtime>=0.20", "qiskit-aer>=0.14"]
mitiq = ["mitiq>=0.38"]
all = ["qiskit-ibm-runtime>=0.20", "qiskit-aer>=0.14", "mitiq>=0.38"]
```

### 3.4 Integration con `__init__.py`

```python
# En execution/__init__.py — exports condicionales
try:
    from qmbp_simulation.execution.mitiq_utils import (
        MitiqCDRResult,
        MitiqDDDZNEResult,
        MitiqPECResult,
        MitiqZNEResult,
        compare_mitigation_strategies,
        make_mitiq_executor,
        run_mitiq_cdr,
        run_mitiq_ddd_zne,
        run_mitiq_pec,
        run_mitiq_zne,
    )
    _MITIQ_AVAILABLE = True
except ImportError:
    _MITIQ_AVAILABLE = False
```

---

## 4. Análisis de Valor por Método para Nuestro Caso Específico

### 4.1 N=10 p=1 heavy_hex (hardware deployment target)

| Método | QPU Overhead | Expected ΔE/gap Improvement | Viable? |
|--------|:---:|:---:|:---:|
| PEA-ZNE (nuestro) | +50% | -94% (baseline) | ✅ Primary |
| GF-ZNE (nuestro) | +200% (3 factors) | -20% | ✅ Fallback |
| Mitiq ZNE random | +200-400% | -15-25% (estimate) | ✅ Alternative to GF |
| Mitiq CDR | +150% (10 Clifford circuits) | -30-50% (from literature) | ✅ **High value** |
| Mitiq DDD+ZNE | +200% | -20-25% (marginal DDD on short circuit) | 🟡 Conditional |
| Mitiq PEC | +5% (low error rate!) | -70-90% (near-exact for low ε) | 🟢 **Surprising!** |
| Affine (nuestro) | 0% | ~0% (safety net) | ✅ Always |

**Finding inesperado**: Para N=10 p=1 en Heron R2 (ε≈0.003/CZ, 17 CZ gates):
- PEC overhead = exp(Σε) = exp(17×0.003) ≈ 1.052 → **solo 5% overhead**
- Esto significa que PEC es potencialmente production-viable para nuestros circuitos cortos
- Requiere validación experimental pero cambia completamente el cálculo de prioridades

### 4.2 Simulación noisy (pre-hardware validation)

| Escenario | Método recomendado | Motivo |
|-----------|-------------------|--------|
| Validar pipeline (quick) | Mitiq ZNE + depolarizing simple | Rápido, reproduce patrón general |
| Validar pre-hardware (realistic) | PEA local + FakeTorino | Más fiel a hardware real |
| Benchmark all methods | `compare_mitigation_strategies()` | Thesis table material |
| CDR feasibility | Mitiq CDR + FakeTorino | Validar que CDR funciona para TFIM HVA |
| PEC feasibility | Mitiq PEC + depolarizing | Verificar overhead real para N=10 |

---

## 5. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|:---:|:---:|---|
| Mitiq incompatible con Qiskit 1.4+ | Baja | Alto | Pin mitiq version, test en CI |
| CDR no mejora para TFIM HVA | Media | Medio | Validar en Fase 3 antes de integrar |
| PEC overhead mayor a esperado | Media | Bajo | Solo benchmark, no producción (salvo validación) |
| Mitiq modifica circuito de forma incompatible | Baja | Alto | Transpile post-fold en executor |
| Circular import con noisy_utils | Baja | Medio | Lazy imports estrictos |
| Degradación de CI time | Media | Bajo | `@pytest.mark.mitiq` + skip condicional |

---

## 6. Criterios de Éxito

1. **Mitiq ZNE vs nuestro GF-ZNE**: Si Mitiq random folding produce R² ≥ 0.95 Y ΔE/gap
   comparable (±5% relativo) → validado como alternativa.

2. **CDR aporta valor**: Si CDR mejora ΔE/gap en ≥10% sobre ZNE raw en ≥50% de h-points
   → integrar como verification layer en deployment.

3. **DDD+ZNE worth it**: Si composición mejora ≥3% absoluto sobre ZNE solo en heavy_hex
   → activar por defecto en hardware. Si no → dejar como optional flag.

4. **PEC viable para N=10**: Si overhead medido < 2× Y ΔE/gap < 1% → considerar
   como primary strategy sobre PEA para circuitos cortos (thesis finding).

5. **No regression**: El módulo mitiq_utils NO debe afectar la ejecución del pipeline
   cuando mitiq no está instalado (lazy imports, optional dependency).

---

## 7. Timeline Estimado

| Fase | Esfuerzo | Dependencia | Status |
|:---:|:---:|:---:|:---:|
| 1. Foundation | 2h | — | ✅ Done |
| 2. Mitiq ZNE | 2h | Fase 1 | ✅ Done |
| 3. Mitiq CDR | 3h | Fase 1 | ✅ Done |
| 4. DDD+ZNE | 2h | Fase 2 | ✅ Done |
| 5. PEC Benchmark | 2h | Fase 1 | ✅ Done |
| 6. Noisy Pipeline | 3h | Fases 2,3,4 | ✅ Done (via compare_mitigation_strategies) |
| 7. Strategy Comparator | 3h | Fases 2,3,4,6 | ✅ Done |
| 8. Tests | 2h | Fases 1-5 | ✅ Done (13/13 pass) |
| 9. Documentación | 1h | Fases 1-7 | ✅ Done |
| **Total** | **~20h** | | |

**Ruta crítica**: Fase 1 → (Fase 2 ∥ Fase 3 ∥ Fase 5) → Fase 6 → Fase 7

---

## 8. Referencias

- Mitiq documentation: https://mitiq.readthedocs.io/en/stable/
- Mitiq GitHub: https://github.com/unitaryfoundation/mitiq
- QMBS + ZNE example: https://mitiq.readthedocs.io/en/stable/examples/quantum_simulation_scars_ibmq.html
- CDR original paper: Czarnik et al., Quantum 5, 592 (2021)
- vnCDR (unified CDR+ZNE): Lowe et al., arXiv:2011.01157
- eCDR (enhanced): arXiv:2409.14632
- PEA (IBM): Kim et al., Nature 618 (2023)
- GNN-QEM (Wang et al.): arXiv:2604.16815
- Mitiq + Qiskit noise models: arXiv:2401.06535
- Robustness evaluation (ZNE/DDD/LRE/PEC): arXiv:2604.17515
