# Próximos Pasos — Análisis de Resultados

**Estado actual**: 7 estudios completados, thresholds corregidos, tablas de tesis generadas.
**Fecha**: 2026-05-27

---

## Prioridad 1 — Fortalecer Claims Débiles (antes de escribir tesis)

### 1A. Validar p=1 ZNE a N=10 (confianza actual: MEDIA, n=2)

**Por qué**: Es el hallazgo más novedoso — si p=1 recupera ZNE a N=10, es una contribución
original significativa. Pero solo tenemos 2 data points.

**Qué ejecutar**:
```bash
# Necesita: 3 seeds × 3 topologías × p=1 noisy
# chain_1d, ladder, triangular — cada uno con seeds 42, 43, 44
# Si 7/9 tienen gain > +30%, el claim es sólido
```

**Criterio de éxito**: gain > +30% en ≥7/9 runs.
**Tiempo estimado**: ~15 min (9 runs × ~100s cada uno).
**Impacto en tesis**: Alto — convierte un "preliminary finding" en un resultado definitivo.

---

### 1B. Controlled restarts comparison (confianza actual: MEDIA, confounding)

**Por qué**: El claim "restarts=5 es óptimo" tiene confounding (diferentes h-grids).
Necesitamos una comparación limpia.

**Qué ejecutar**:
```bash
# Misma config exacta, solo varía restarts: 1, 3, 5, 7
# Topology: ladder N=10 (la más estudiada)
# h_values: [4.0, 3.5, 3.0, 2.5, 2.0], h_test: [2.5]
# Seed: 42
# hidden=128, patience=500, epochs=6000
```

**Criterio de éxito**: Curva clara de ΔE/gap vs restarts con plateau a 5.
**Tiempo estimado**: ~5 min (4 runs × ~35s).
**Impacto en tesis**: Medio — confirma una recomendación práctica.

---

## Prioridad 2 — Análisis Adicionales (enriquecen la narrativa)

### 2A. Análisis de Error Decomposition por topología

**Por qué**: Sabemos que error = error_circuit + error_mpnn. ¿Cambia la proporción por topología?
Si en triangular el error_mpnn domina, el GNN necesita más capacidad. Si error_circuit domina,
es un límite del HVA.

**Qué hacer**:
```bash
# Extraer energy_decomposition de diagnostics.json por topología
# Comparar: ¿qué fracción del error es MPNN vs circuit?
python -m scripts.digest --kind noiseless --verbose --topology ladder -o /tmp/decomp_ladder.txt
python -m scripts.digest --kind noiseless --verbose --topology triangular -o /tmp/decomp_tri.txt
# Buscar en los JSON: diagnostics.phase4.energy_decomposition
```

**Impacto en tesis**: Alto — explica POR QUÉ triangular es más difícil.

---

### 2B. Análisis temporal — ¿Mejoraron los resultados con el tiempo?

**Por qué**: Tenemos múltiples runs por variante (timestamps diferentes). ¿Las correcciones
de config a lo largo del proyecto mejoraron los resultados?

**Qué hacer**:
```bash
# Comparar primer run vs último run por variante
# Si el último es siempre mejor → la metodología iterativa funciona
# Si no → los resultados son estables (también bueno)
```

**Impacto en tesis**: Medio — valida la metodología de desarrollo iterativo.

---

### 2C. Correlación θ-smoothness vs ΔE/gap

**Por qué**: θ-smoothness aparece como diagnóstico en todos los runs. ¿Es un buen predictor
de calidad? Si sí, se puede usar como early-warning sin ejecutar Phase 4.

**Qué hacer**:
```bash
# Extraer (theta_smoothness, delta_e_over_gap) de todos los noiseless runs
# Calcular correlación de Pearson
# Si r > 0.5 → θ-smoothness es un predictor útil
python -m scripts.digest --json /tmp/all_noiseless.json --kind noiseless
# Luego analizar con Python: correlación entre campos
```

**Impacto en tesis**: Medio-alto — propone un "quality indicator" sin deployment.

---

## Prioridad 3 — Documentación Final (para escritura de tesis)

### 3A. Generar todas las tablas en markdown definitivo

**Qué hacer**:
```bash
python -m scripts.digest --kind noiseless --group-by topology --n-qubits 10 --markdown -o documentation/analysis/table_topology_n10.md
python -m scripts.digest --kind noisy --group-by n_qubits --markdown -o documentation/analysis/table_zne_boundary.md
python -m scripts.digest --kind experiment --sort verdict --verbose --markdown -o documentation/analysis/table_experiments.md
```

---

### 3B. Generar JSON completo para figuras

**Qué hacer**:
```bash
python -m scripts.digest --json documentation/analysis/raw_data/all_results.json
# Usar para generar plots con matplotlib/seaborn en notebooks
```

---

### 3C. Escribir "thesis implications" consolidadas

Compilar los párrafos de "Implicación para la Tesis" de cada estudio (01-07) en un
solo documento que sirva como borrador del Chapter 5 (Results & Discussion).

---

## Orden de Ejecución Recomendado

| # | Paso | Tiempo | Impacto | Requiere ejecución? |
|---|------|--------|---------|---------------------|
| 1 | ~~1A: p=1 ZNE validation~~ | 15 min | ★★★ | ✅ COMPLETADO (9 runs, gain=+49%) |
| 2 | ~~2A: Error decomposition~~ | 5 min | ★★★ | ✅ COMPLETADO |
| 3 | ~~1B: Controlled restarts~~ | 5 min | ★★ | ✅ COMPLETADO |
| 4 | ~~2C: θ-smoothness correlation~~ | 5 min | ★★ | ✅ COMPLETADO |
| 5 | ~~3A: Tablas markdown~~ | 2 min | ★★ | ✅ COMPLETADO |
| 6 | ~~3B: JSON para figuras~~ | 1 min | ★ | ✅ COMPLETADO |
| 7 | ~~2B: Análisis temporal~~ | 10 min | ★ | SKIPPED (low impact) |
| 8 | 3C: Thesis implications | 30 min | ★★★ | PENDING (writing task) |

**Total para pasos que requieren ejecución**: ~20 min (13 runs nuevos).
**Total para análisis de datos existentes**: ~20 min.
**Total para documentación**: ~30 min.

---

## Criterio de "Done"

El análisis está completo cuando:
1. ✅ Todos los claims tienen confianza ALTA (n≥5, sin confounding)
2. ✅ Todas las tablas de tesis están generadas en markdown
3. ✅ Los hallazgos negativos están documentados como contribuciones
4. ✅ Los outliers están explicados y documentados
5. ✅ El JSON completo está disponible para generar figuras
6. ✅ El borrador de Chapter 5 tiene todos los "thesis statements"
