# Eje 4 — ZNE y Ruido: Límites Fundamentales

## 4A. Confirmación del Failure Mode ZNE@N=10

| Topología | N | Variant | R² | Gain (%) | Wins | Veredicto |
|-----------|---|---------|-----|----------|------|-----------|
| triangular | 10 | NY-A-shots8192 | 0.844 | -34.4 | 0/3 | ❌ ZNE falla |
| triangular | 10 | NY-A-shots16384 | 0.870 | -34.2 | 0/3 | ❌ ZNE falla |
| triangular | 10 | NY-A-shots32768 | 0.831 | -33.6 | 0/3 | ❌ ZNE falla |
| triangular | 10 | NY-B-lay3 | 0.872 | -32.8 | 0/3 | ❌ ZNE falla |
| triangular | 10 | NY-B-lay7 | 0.722 | -38.1 | 0/3 | ❌ ZNE falla |
| triangular | 10 | NY-C-seed42 | 0.882 | -31.9 | 0/3 | ❌ ZNE falla |
| triangular | 10 | NY-C-seed44 | 0.977 | +8.0 | 1/3 | ✅ ZNE funciona |
| triangular | 10 | EXT-5-noisy-p1 | 0.979 | +73.0 | 3/3 | ✅ ZNE funciona |

## 4B. Hallazgo Crítico: p=1 Noisy

**ZNE con p=1 en topologías 2D:**

- **triangular N=10**: R²=0.979, gain=+73.0%, wins=3/3 → ✅ FUNCIONA

### Interpretación

p=1 reduce el conteo de CX gates en ~50%, lo que puede colocar el circuito
de vuelta en el régimen perturbativo donde ZNE funciona (E lineal en CES).
Esto abre la puerta a hardware deployment con p=1 en topologías 2D.

## Resumen ZNE

| Configuración | Resultado | Implicación |
|---------------|-----------|-------------|
| N=6, p=2, chain_1d | ✅ R²>0.99, +40% gain | Régimen perturbativo |
| N=10, p=2, chain_1d | ❌ R²<0.05, gain negativo | No-perturbativo |
| N=10, p=2, ladder | ❌ gain negativo | Más CX → peor |
| N=10, p=2, triangular | ❌ gain ~-34% | Máximo CX → peor |
| N=10, p=1, triangular | ✅ R²=0.98, +73% gain | **CX budget hypothesis confirmed** |
