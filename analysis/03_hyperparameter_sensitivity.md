# Eje 3 — Sensibilidad de Hiperparámetros

## 3A. MPNN Hidden Dimension

| Topología | N | h=64 | h=128 | h=256 | Mejor | Diferencia |
|-----------|---|------|-------|-------|-------|------------|
| chain_1d | 6 | 0.0672 | 0.0278 | N/A | h=128 | 0.0395 |
| ladder | 6 | 0.1157 | 0.0040 | 0.0869 | h=128 | 0.1117 |
| triangular | 6 | 0.1708 | 0.0026 | 0.1838 | h=128 | 0.1812 |
| ladder | 10 | 0.0168 | 0.0170 | 0.0345 | h=64 | 0.0178 |
| triangular | 10 | 0.0384 | 0.0376 | 0.0389 | h=128 | 0.0012 |

## 3B. Densidad del h-Grid

| Topología | N | Sparse (5pts) | Standard (7pts) | Dense (9pts) | Mínimo viable |
|-----------|---|---------------|-----------------|--------------|----------------|
| chain_1d | 6 | 0.4290 | 0.0276 | 0.0273 | standard |
| ladder | 6 | 1.8892 | 0.0397 | 0.0040 | standard |
| triangular | 6 | 0.1934 | 0.0024 | 0.0026 | standard |
| ladder | 10 | 0.0666 | 0.0341 | 0.0164 | standard |
| triangular | 10 | 0.0372 | 0.0371 | 0.0373 | sparse |

## 3C. VQE Restarts

| Topología | N | 1 rst | 3 rst | 5 rst | 7 rst | Mínimo para PASS |
|-----------|---|-------|-------|-------|-------|------------------|
| chain_1d | 6 | 0.0293 | 0.1230 | 0.0280 | 0.0286 | 1 |
| ladder | 6 | 0.1394 | 0.0965 | 0.0913 | 0.1186 | N/A |
| triangular | 6 | 0.0030 | 0.0026 | 0.1917 | 0.0756 | 1 |
| ladder | 10 | 0.0165 | 0.0750 | 0.0170 | 0.0395 | 1 |
| triangular | 10 | 0.0626 | 0.0502 | 0.0371 | 0.9705 | 5 |
