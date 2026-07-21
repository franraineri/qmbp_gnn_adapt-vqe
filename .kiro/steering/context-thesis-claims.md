---
inclusion: fileMatch
fileMatchPattern: "**/thesis*,documentation/analysis/09*,documentation/analysis/08*,**/*.tex"
---

# Thesis Claims Context — Noiseless Pipeline (2026-07-16)

> Pre-digested context for thesis writing. ALL claims are noiseless simulation only.

## General Objective (CANONICAL)

"Demostrar que la integración de predicción GNN con HVA en un pipeline unificado permite reducir el costo de la clasificación de fases entre 29× y 500×, manteniendo ΔE/gap < 5% dentro del régimen operativo válido, y documentar formalmente los límites de dicho régimen."

## Core Contributions (Chapter 5)

### 1. Pipeline GNN-HVA clasifica fases correctamente
- ΔE/gap < 5% en régimen válido (h ≥ h_min)
- Pass rate 95-100% en configuraciones óptimas (cadena p=3, heavy-hex p=3)
- 400+ ejecuciones, 5 topologías, p=1-4, N=4-20
- ΔE absoluto estable: 0.003-0.006 (N=8-20)
- Fidelidad ≥ 0.995 (media) en runs exitosos

### 2. Aceleración 29-500× respecto a VQE random
- Speedup varía con configuración: 29× (N=10 p=3) a 500× (N=20 p=4)
- MPNN predice θ en una pasada forward clásica (0 iteraciones VQE)
- Warm-start produce datos de alta calidad para entrenamiento

### 3. Cross-topology validado (5 topologías, hiperparámetros unificados)
- Jerarquía: cadena > heavy-hex >> escalera ≈ cuadrada >> triangular
- Topología domina sobre profundidad (+19-20% vs +4-10% al subir p)
- Transfer cross-topology FALLA (cada topología requiere entrenamiento propio)
- ΔE discrimina: cadena/heavy-hex ΔE~0.003-0.01 vs 2D ΔE~0.08 (fallo de expresividad)

### 4. Extensibilidad documentada
- TFIM longitudinal: deploy 90%+ sin compuertas 2Q adicionales
- Heisenberg: 0% en 46 runs exhaustivos (p=1-4, 5 topos) — límite fundamental
- Regla: funciona si nuevos términos → compuertas 1Q

### 5. Cross-N generalization (zero-shot)
- Train N=40+80 → predict N=50-100: 30/30 PASS (ΔE 0.011-0.033)
- BatchNorm perjudicial en grafos regulares (descubrimiento + fix: 142× mejora)
- Fidelidad no disponible para N>22 (MPS backend)

### 6. Frontera de expresividad (h_min) documentada formalmente
- Para p=1-2: h_min crece linealmente (p=1: 2.36+0.0073·N, p=2: 1.57+0.005·N)
- Para p≥3: h_min es cuasi-constante (~1.6 para p=3, ~1.4 para p=4), independiente de N
- Consistente con area law: entrelazamiento a h/h_c fijo no depende de N en 1D
- Heisenberg: h_min≈3.5 constante ∀p (mismatch circuito-Hamiltoniano, Wiersema2020)
- Topología: z_max domina (chain≈heavy_hex=1.09-1.12, triangular=2.20)
- Confirmado por Tripathi2026, Sumeet2025
- Con p=5 (N=8): h_min → h_c ≈ 0.97 (desaparece)

## Key Finding: ΔE es Estable

| N | ΔE medio | Gap medio | ΔE/gap |
|---|---|---|---|
| 10 | 0.003-0.010 | 1.5 | 0.2-0.7% |
| 16 | 0.006 | 1.9 | 0.3% |
| 20 | 0.005 | 2.0 | 0.2% |
| 50 (cross-N) | 0.016 | 8.7 | 0.18% |
| 100 (cross-N) | 0.033 | 18.7 | 0.18% |

ΔE/gap "mejora" con N porque el gap crece, NO porque el predictor mejore.

## Lo que NO es contribución nuestra (ATRIBUIR)

- Warm-start: Mele2022, Puig2025
- GNN prediction: Miao2024, Zhang2025
- GINConv: Xu2019
- HVA design: Wiersema2020
- DMRG: Hauschild2018

## Lo que SÍ es nuestro

1. Integración en pipeline unificado
2. Validación sistemática multi-topología + multi-p + multi-modelo
3. Hallazgo BatchNorm en cross-N
4. Documentación formal de límites (Heisenberg, h_min, topología)
5. Script de análisis/diagnóstico automatizado
