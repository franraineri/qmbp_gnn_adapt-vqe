# Guía de Métodos de Simulación — MPS, Statevector y Backends

## 1. El Problema

Para ejecutar VQE necesitamos evaluar ⟨ψ(θ)|H|ψ(θ)⟩ miles de veces. Esto requiere simular el circuito cuántico clásicamente. Cada método de simulación tiene tradeoffs fundamentalmente distintos en memoria, tiempo, precisión y escalabilidad.

---

## 2. Métodos Disponibles

### 2.1 Exact Diagonalization (ExactDiag)

**Representación**: Diagonalización completa de la matriz H (2^N × 2^N).

**Implementación**: `ClassicalSolver.solve(H, lattice)` → `numpy.linalg.eigh`

**Propósito**: Obtener el ground state exacto E₀, gap, eigenstate |ψ₀⟩.

| Propiedad | Valor |
|-----------|-------|
| Complejidad memoria | O(2^N) = 16 bytes × 2^N |
| Complejidad tiempo | O(2^(3N)) para eigh completo |
| Precisión | Machine epsilon (~10⁻¹⁵) |
| N máximo práctico | ~15 (1 GB RAM) |
| Rol en el pipeline | Phase 1 ground truth (N≤15) |

---

### 2.2 DMRG (Density Matrix Renormalization Group)

**Representación**: El ground state como un MPS variacional optimizado iterativamente.

**Implementación**: `ClassicalSolver.solve(H, lattice, method="dmrg")`

**Propósito**: Ground truth para N>15. Converge al eigenvalue más bajo explorando el espacio de estados restringido a MPS con bond dimension χ.

| Propiedad | Valor |
|-----------|-------|
| Complejidad memoria | O(N·χ²) — lineal en N |
| Complejidad tiempo | O(N·χ³) por sweep |
| Precisión (1D) | ~10⁻¹⁰ (truncation error despreciable) |
| Precisión (2D) | Variable — depende de χ y ancho del sistema |
| N máximo práctico | ~200+ en 1D |
| Rol en el pipeline | Phase 1 ground truth (N>15) |

**Fuentes de error**:
- Truncation: al limitar χ, se descartan componentes del estado (cuantificable como ε_trunc = Σᵢ σᵢ²)
- Convergence: DMRG puede estancarse en mínimos locales (raro en 1D gapped)

**Validación empírica**: DMRG vs ExactDiag a N≤15 da |ΔE| < 10⁻¹⁰. χ_actual observado en TFIM 1D: 9-15 para todo N=40-200.

---

### 2.3 StatevectorEstimator (NoiselessBackend)

**Representación**: Vector completo de 2^N amplitudes complejas.

```
|ψ⟩ = [α₀, α₁, ..., α_{2^N - 1}]
```

**Implementación**: `NoiselessBackend().evaluate(circuit, H, params)`

**Propósito**: Simular el circuito HVA con parámetros θ y calcular ⟨H⟩ exactamente. Cada gate se aplica como multiplicación matrix×vector sobre el estado completo.

| Propiedad | Valor |
|-----------|-------|
| Complejidad memoria | O(2^N) = 16 bytes × 2^N |
| Complejidad tiempo/gate | O(2^N) |
| Precisión | Machine epsilon (~10⁻¹⁵) |
| N máximo práctico | ~22 (64 MB, ~2s/eval) |
| Rol en el pipeline | VQE evaluation para N≤10 (loops) o N≤15 (single) |

**Performance en este proyecto**:

| N | Memoria | Tiempo/eval |
|---|---------|-------------|
| 6 | 1 KB | ~1 ms |
| 10 | 16 KB | ~5 ms |
| 20 | 16 MB | ~500 ms |
| 22 | 64 MB | ~2 s |
| 30+ | 16+ GB | Imposible |

---

### 2.4 MPS Backend — Modo Determinístico (Default)

**Representación**: Cadena de tensores de rango 3 con bond dimension máxima χ.

```
|ψ⟩ ≈ A₁[i₁] · A₂[i₂] · A₃[i₃] · ... · Aₙ[iₙ]
Cada Aₖ: tensor (χ × 2 × χ)
Total: 2Nχ² parámetros
```

**Implementación**: `MPSBackend(strategy="aer_mps", chi_max=64, deterministic=True)`

**Cómo funciona**:
1. Inicia con |+⟩^N como MPS trivial
2. Aplica gates RZZ/RX del HVA como operaciones locales sobre el MPS
3. Usa `save_expectation_value` de Qiskit Aer para calcular ⟨H⟩ directamente desde la representación MPS interna
4. Si χ ≥ χ_necesario, el resultado es exacto (no aproximado)

| Propiedad | Valor |
|-----------|-------|
| Complejidad memoria | O(N·χ²) — lineal en N |
| Complejidad tiempo/eval | O(N·χ³) |
| Precisión (χ suficiente) | Machine epsilon (~10⁻¹⁴) — idéntico a Statevector |
| Precisión (χ insuficiente) | O(ε_trunc) — puede ser grande |
| N máximo práctico | ~500+ |
| Rol en el pipeline | VQE evaluation para N>10 |

**Performance en este proyecto (χ=64)**:

| N | Memoria | Tiempo/eval | vs Statevector |
|---|---------|-------------|----------------|
| 20 | 164 KB | ~12 ms | 40× más rápido |
| 40 | 328 KB | ~12 ms | SV imposible |
| 100 | 820 KB | ~20 ms | SV imposible |
| 200 | 1.6 MB | ~35 ms | SV imposible |

---

### 2.5 MPS Backend — Modo Estocástico (Legacy)

**Mismo motor MPS** pero con sampling estadístico en vez de evaluación exacta.

**Implementación**: `MPSBackend(strategy="aer_mps", deterministic=False, precision=0.005)`

| Aspecto | Determinístico | Estocástico |
|---------|:-:|:-:|
| Precisión | 10⁻¹⁴ (exacto) | ~precision (~0.005) |
| Tiempo/eval N=40 | 12 ms | 6 s |
| Speedup | — | 375-3000× más lento |
| Shot noise | No | Sí (σ ≈ precision) |
| Optimizador ideal | L-BFGS-B | COBYLA |
| Status | **Default (recomendado)** | Legacy (backward compat) |

**Datos empíricos** (mps_mode_comparison.json):

| N | ΔE/gap deterministic | ΔE/gap stochastic | Speedup |
|---|---|---|---|
| 40 | 0.20% | 4.87% | 779× |
| 50 | 0.12% | 1.12% | 3025× |

El modo estocástico es estrictamente inferior: más lento y menos preciso.

---

### 2.6 NoisyBackend (FakeTorino)

**Propósito**: Simular el comportamiento de un QPU real con errores de gate, readout y decoherencia.

**Implementación**: `NoisyBackend(n_layouts=3, seed=42)` + ZNE post-processing

| Propiedad | Valor |
|-----------|-------|
| Fuentes de error | Gate error (~0.5-1%/CX), readout (~1-3%/qubit), T1/T2 |
| ΔE/gap raw típico | 5-50% |
| ΔE/gap post-PEA-ZNE | 0.6-5% |
| N máximo | ~133 (FakeTorino = 133 qubits) |
| Rol en el pipeline | Phase 4b — validar ZNE antes de QPU real |

---

### 2.7 HardwareBackend (IBM QPU)

**Propósito**: Ejecución real en procesador cuántico superconductor.

| Propiedad | Valor |
|-----------|-------|
| Fuentes de error adicionales | Calibration drift, TLS, crosstalk, queue effects |
| ΔE/gap post-PEA-ZNE esperado | 1-10% |
| Criterio de éxito | ΔE/gap < 5% AND phase label correcto |
| N máximo | ~133 (IBM Heron) |
| Rol en el pipeline | Phase 4 — deployment final |

---

## 3. ¿Por Qué χ=64 es Exacto para Este Proyecto?

El entrelazamiento generado por HVA p≤2 en 1D TFIM es bajo. La entanglement entropy máxima a través de cualquier bipartición está acotada:

- Para p=1 con 2 parámetros: ~2-3 bits de entrelazamiento
- χ=64 → log₂(64) = 6 bits de entrelazamiento máximo representable
- Margen: 3× por encima de lo necesario

**Validación empírica (V7 exp 3A/3B)**:
```
N=6:   |E_MPS - E_Statevector| = 1.4 × 10⁻¹⁴  (machine epsilon)
N=10:  |E_MPS - E_Statevector| = 2.1 × 10⁻¹⁴  (machine epsilon)
```

**χ_actual observado por DMRG**: Para TODOS los N testeados (40-200), el bond dimension efectivo es solo 9-15. χ_max=64 nunca se satura.

**Limitación en 2D**: Para topologías como `square` o `triangular`, el entanglement area-law en 2D hace que χ_necesario ∝ e^(ancho). A N=16 en square (4×4), χ puede necesitar ser ~10³. El χ=64 default NO es suficiente para 2D ancho.

---

## 4. Tabla Resumen de Tasas de Error

| Método | Error típico | Tipo de error | ¿Controlable? | ¿Measurable? |
|--------|:---:|---|:---:|:---:|
| ExactDiag | 10⁻¹⁵ | Floating point | No (intrínseco) | Sí (analítico) |
| DMRG (1D) | 10⁻¹⁰ | Truncation | Sí (aumentar χ) | Sí (vs ExactDiag) |
| Statevector | 10⁻¹⁵ | Floating point | No | Sí (vs ExactDiag) |
| MPS det (χ=64, 1D) | 10⁻¹⁴ | Ninguno (exacto) | N/A | Sí (vs Statevector) |
| MPS det (χ insuficiente) | 10⁻² a 10⁻¹ | Truncation | Sí (aumentar χ) | Sí (χ vs 2χ) |
| MPS estocástico | ~precision (0.005) | Shot noise | Sí (bajar precision) | Sí (variance) |
| FakeTorino raw | 0.05 - 0.50 | Gate+readout noise | No directamente | Sí (vs noiseless) |
| FakeTorino + PEA-ZNE | 0.006 - 0.05 | Residual + shot | Parcialmente | Sí (vs noiseless) |
| IBM QPU + PEA-ZNE | 0.01 - 0.10 | Todo + drift + TLS | Parcialmente | Sí (vs DMRG) |

---

## 5. Jerarquía de Validación Cruzada

Cada nivel del pipeline se valida contra el nivel superior de precisión:

```
Nivel 0: Solución analítica (TFIM 1D, N→∞, Jordan-Wigner)
  ↓ verificar
Nivel 1: ExactDiag (numpy.linalg.eigh, N≤15, exacto)
  ↓ verificar (DMRG vs ED: |ΔE| < 10⁻¹⁰)
Nivel 2: DMRG (N≤200+, cuasi-exacto en 1D)
  ↓ verificar (MPS vs SV: |ΔE| = 10⁻¹⁴)
Nivel 3: MPS Determinístico / Statevector (simula circuito exactamente)
  ↓ verificar (3-mode comparison)
Nivel 4: NoisyBackend (FakeTorino + ZNE)
  ↓ verificar (ZNE ΔE/gap vs noiseless)
Nivel 5: Hardware QPU (IBM Torino + PEA-ZNE)
```

---

## 6. Matriz de Selección de Método

### Árbol de decisión

```
¿Objetivo = ground truth (E₀, gap)?
├── N ≤ 15 → ExactDiag
└── N > 15 → DMRG

¿Objetivo = evaluar circuito E(θ)?
├── N ≤ 10 (VQE loop) → NoiselessBackend (StatevectorEstimator)
├── N ∈ [11, 22] topología 1D → MPSBackend(χ=64)
├── N ∈ [11, 22] topología 2D → NoiselessBackend (lento pero exacto)
├── N > 22 topología 1D → MPSBackend(χ=64) — obligatorio
└── N > 22 topología 2D → MPSBackend(χ grande) + verificar convergencia χ

¿Objetivo = validar error mitigation?
└── NoisyBackend + 3-mode comparison (noiseless / raw / ZNE)

¿Objetivo = hardware deployment?
└── HardwareBackend + referencia DMRG + PEA-ZNE
```

### Tabla por escenario

| Escenario | Backend | Ground Truth | Métricas clave |
|---|---|---|---|
| Desarrollo/debug (N=4-6) | NoiselessBackend | ExactDiag | fidelity, de_gap |
| Thesis noiseless (N=6-20) | select_backend(auto) | ExactDiag/DMRG | de_gap, fidelity, entanglement, θ_smooth |
| Scaling (N=40-200) | MPSBackend(χ=64) | DMRG | de_gap, chi_actual, θ_smooth, timing |
| Cross-N transfer | MPSBackend | DMRG per-N | de_gap per-N, MPNN MSE |
| Noisy validation | NoisyBackend | Noiseless eval | de_gap_raw, de_gap_zne, R² |
| Hardware rehearsal | FakeKingston | Noiseless eval | de_gap, SNR, layout_std |
| QPU deployment | HardwareBackend | DMRG | de_gap, phase_label, SNR |

---

## 7. Selección Automática en el Código

La función `select_backend()` en `src/qmbp_simulation/execution/backends.py` implementa la lógica de selección automática:

```python
from qmbp_simulation.execution import select_backend

# Para VQE loops (threshold bajo: N>10 → MPS)
backend = select_backend(n_qubits=20, for_vqe_loop=True)  # → MPSBackend

# Para evaluaciones puntuales (threshold alto: N>15 → MPS)
backend = select_backend(n_qubits=12)  # → NoiselessBackend

# Manual override (scaling runs)
from qmbp_simulation.execution import MPSBackend
backend = MPSBackend(strategy="aer_mps", chi_max=64, seed=42)
```

En `ValidationRunner`, se accede via:
```python
self._backend = self.select_backend(N, for_vqe_loop=True)
```

---

## 8. Descomposición del Error (Error Budget)

El ΔE/gap total observado en cualquier run se descompone en componentes independientes:

```
ΔE_total = ΔE_ansatz + ΔE_optimizer + ΔE_simulator + ΔE_noise
```

| Componente | Qué mide | Magnitud típica | Cómo aislarlo |
|---|---|---|---|
| ΔE_ansatz | HVA p≤2 no alcanza el ground state | 0.1-5% (depende de h) | VQE con muchos restarts vs E₀ |
| ΔE_optimizer | VQE no convergió al mínimo global | 0-2% | Comparar seeds, aumentar restarts |
| ΔE_simulator | Error numérico del backend | ~0 (MPS/SV) | Comparar MPS vs SV (mismo θ) |
| ΔE_noise | Ruido de hardware/simulación noisy | 5-50% raw, 1-5% post-ZNE | 3-mode comparison |

**Para runs noiseless**: ΔE_simulator ≈ 0, entonces ΔE_total ≈ ΔE_ansatz + ΔE_optimizer.

**Para runs noisy**: ΔE_noise domina. ZNE reduce este componente ~90% (PEA) o ~20% (gate-folding).

**Implicación**: El ΔE/gap < 5% criterion de la tesis mide la suma de todos los componentes. En el régimen válido (h > h_min), ΔE_ansatz < 1% y ΔE_optimizer < 1%, dando margen para ΔE_noise ≤ 3% en hardware.

---

## 9. Detección de Problemas

### χ insuficiente (MPS)

**Síntoma**: Energías que varían al cambiar χ.

**Test**: Correr el mismo punto con χ y 2χ. Si |E(χ) - E(2χ)| > 10⁻⁸, χ es insuficiente.

**Cuándo ocurre**: Topologías 2D con N grande, o circuitos muy profundos (p>2, no permitido en este proyecto).

### Violación del principio variacional

**Síntoma**: E_vqe < E_exact (energía VQE por debajo del ground state).

**Causa**: Error numérico acumulado, o E_exact incorrecto.

**Check automático**: Cada runner verifica `E_vqe >= E_exact - 1e-8` y emite WARNING si viola.

### Convergencia insuficiente del optimizador

**Síntoma**: Alto θ_change_linf entre puntos consecutivos, o n_iterations = maxiter.

**Mitigación**: Aumentar n_restarts, usar warm-start más cercano, o bidirectional sweep.

### Shot noise excesivo (modo estocástico/hardware)

**Síntoma**: Alta varianza entre ejecuciones con mismos parámetros.

**Cuantificación**: σ_shot ≈ 1/√shots. Con 16k shots → σ ≈ 0.008.

**Regla**: Si |⟨O⟩| < σ_shot → la medición no tiene signal (SNR < 1). Usar ≥8192 shots.

---

## 10. Resumen Operacional

```
N=4-10, cualquier topología:
  → StatevectorEstimator + ExactDiag reference
  → Resultado: EXACTO (fidelity computable, entanglement computable)

N=10-22, 1D/quasi-1D:
  → MPSBackend(χ=64, deterministic) + DMRG reference
  → Resultado: EXACTO (idéntico a Statevector, validado 10⁻¹⁴)

N=10-22, 2D:
  → StatevectorEstimator (lento pero seguro)
  → O MPSBackend con chi-convergence test
  → Resultado: EXACTO si se verifica χ

N=40-200, 1D:
  → MPSBackend(χ=64) + DMRG
  → Resultado: EXACTO (χ_actual=9-15, muy por debajo de χ_max=64)
  → ΔE/gap < 0.1% típico para N≥80

Hardware (N=10, heavy_hex):
  → QPU + PEA-ZNE + referencia DMRG
  → Resultado: ~1-5% ΔE/gap con mitigation
  → Criterio: ΔE/gap < 5% AND phase label correcto
```

---

*Documento generado 2026-07-13. Basado en evidencia experimental (V7 3A/3B, mps_mode_comparison.json, binnacle-mps-scaling.md) y análisis del código fuente del proyecto.*
