# Plan de Pruebas — Scaling Runners (migrados)

## Prerequisitos
```bash
cd /Users/franco.raineri/devTools/QuantumDev/qmbp_gnn_adapt-vqe
source .venv/bin/activate
```

---

## 1. `run_scaling_validation.py` — Smoke Tests

### Test A: Dry-run (verifica preflight + CLI)
```bash
.venv/bin/python scripts/experiment_runners/scaling/run_scaling_validation.py --dry-run
```
Esperado: lista 2 secciones, exit 0.

### Test B: Run mínimo (N=20, 3 h-points, ~2 min)
```bash
.venv/bin/python scripts/experiment_runners/scaling/run_scaling_validation.py \
    --n-qubits 20 --h-points 3 --h-min 2.5 --h-max 3.5
```
Esperado:
- Section 1 (DMRG): 3 puntos calculados
- Section 2 (VQE): ΔE/gap < 5% en todos
- JSON guardado en `results/experiments/exp_scaling/tfim/chain_1d/run_*.json`
- Auto-index actualizado

### Test C: Run con topología diferente (N=20, heavy_hex)
```bash
.venv/bin/python scripts/experiment_runners/scaling/run_scaling_validation.py \
    --n-qubits 20 --topology heavy_hex --h-points 3 --h-min 3.0 --h-max 4.5
```
Esperado: heavy_hex funciona con MPS idéntico a chain_1d.

### Test D: Section individual (solo DMRG)
```bash
.venv/bin/python scripts/experiment_runners/scaling/run_scaling_validation.py \
    --n-qubits 40 --h-points 3 --section 1
```
Esperado: solo Phase 1 ejecutada, datos DMRG guardados.

### Test E: N=40 full (validación real, ~5 min)
```bash
.venv/bin/python scripts/experiment_runners/scaling/run_scaling_validation.py \
    --n-qubits 40 --h-points 5
```
Esperado: ΔE/gap < 5% (la ley de scaling predice h_min ≈ 3.58 para N=40).

---

## 2. `run_scaling_phase3_mpnn.py` — Smoke Tests

### Test F: Dry-run con archivo existente
```bash
.venv/bin/python scripts/experiment_runners/scaling/run_scaling_phase3_mpnn.py \
    --result-file results/scaling/scaling_N40_aer_mps_20260608_001053.json --dry-run
```
Esperado: lista 2 secciones, exit 0, carga datos correctamente.

### Test G: Run completo con datos existentes (~3 min)
```bash
.venv/bin/python scripts/experiment_runners/scaling/run_scaling_phase3_mpnn.py \
    --result-file results/scaling/scaling_N40_aer_mps_20260608_001053.json \
    --n-epochs 1000 --hidden-dim 64
```
Esperado:
- Section 1: MSE < 1e-3 (con 1000 epochs puede ser suficiente para N=40 p=1)
- Section 2: ΔE/gap < 5% en la mayoría de test points
- JSON en `results/experiments/exp_scaling/tfim/chain_1d/run_*.json`

### Test H: Con --use-all-seeds (más datos de entrenamiento)
```bash
.venv/bin/python scripts/experiment_runners/scaling/run_scaling_phase3_mpnn.py \
    --result-file results/scaling/scaling_N40_aer_mps_20260608_001053.json \
    --use-all-seeds --n-epochs 2000
```
Esperado: más puntos de entrenamiento → menor MSE → mejor deployment.

### Test I: Pipeline completo (validation → phase3)
```bash
# Paso 1: Generar datos frescos
.venv/bin/python scripts/experiment_runners/scaling/run_scaling_validation.py \
    --n-qubits 30 --h-points 8

# Paso 2: Entrenar MPNN sobre esos datos (usar el JSON más reciente)
LATEST=$(ls -t results/experiments/exp_scaling/tfim/chain_1d/run_*.json | head -1)
.venv/bin/python scripts/experiment_runners/scaling/run_scaling_phase3_mpnn.py \
    --result-file "$LATEST" --n-epochs 3000
```
Esperado: pipeline end-to-end funciona con el nuevo formato v2.

---

## Verificación Post-Run

```bash
# Ver resultados en el index
python -m project_health --diagnose --model tfim

# Verificar que los runs aparecen
python project_health/cli/query_index.py --stats
```

---

## Criterios de Éxito

| Runner | Criterio | Valor |
|--------|----------|-------|
| scaling_validation | ΔE/gap en valid regime | < 5% todos los puntos |
| scaling_validation | Tiempo N=40, 5 h-points | < 10 min |
| scaling_phase3 | Training MSE | < 1e-3 |
| scaling_phase3 | Deploy pass rate | ≥ 80% |
| Ambos | JSON envelope correcto | schema_version 2.0 |
| Ambos | Auto-index update | resultado visible en `--stats` |
