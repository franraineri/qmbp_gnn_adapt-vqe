.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology chain_1d --p-layers 2 --train-n 6 --target-n 6 10 --multi-n-train --n-anchors 10 --n-restarts 6 --h-min 1.5 --h-max 3.5 --h-points 16
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology chain_1d --p-layers 2 --train-n 8 --target-n 6 10 --multi-n-train --n-anchors 10 --n-restarts 6
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology chain_1d --p-layers 2 --train-n 10 --target-n 10 12 14 16 --multi-n-train --n-anchors 12 --n-restarts 6
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology ladder --p-layers 2 --train-n 6 --target-n 6 10 --multi-n-train --n-anchors 10 --n-restarts 6
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology ladder --p-layers 2 --train-n 8 --target-n 8 10 --multi-n-train --n-anchors 10 --n-restarts 6
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology ladder --p-layers 2 --train-n 10 --target-n 10 12 14 16 --multi-n-train --n-anchors 12 --n-restarts 6
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology square --p-layers 2 --train-n 4 --target-n 4 6 8 --multi-n-train --n-anchors 10 --n-restarts 6
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology square --p-layers 2 --train-n 6 --target-n 6 8 10 --multi-n-train --n-anchors 10 --n-restarts 6
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --p-layers 2 --train-n 6 --target-n 6 8 10 --multi-n-train --n-anchors 10 --n-restarts 6
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --p-layers 2 --train-n 10 --target-n 10 12 14 16 --multi-n-train --n-anchors 12 --n-restarts 6
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology chain_1d --p-layers 2 --train-n 10 --target-n 10 14 16 20 --multi-n-train --force-retrain --iterative-improve --max-iterations 2
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology ladder --p-layers 2 --train-n 10 --target-n 10 12 14 16 20 --multi-n-train --force-retrain --iterative-improve --max-iterations 2


Paso 1: Re-entrenar MT universal (con curriculum)
Runner:
run_multi_topology_training.py
 Datos: 2474 graphs (después de filtro de calidad) de 5 topologías Arquitectura: residual + FiLM (la misma que dio los mejores resultados la vez anterior)

.venv/bin/python scripts/experiment_runners/cross_topology/run_multi_topology_training.py \
    --use-residual --film --curriculum \
    --epochs 5000 --patience 300 \
    --max-n 40 --max-de-gap 0.10 \
    --hidden-dim 256 --n-layers 3 \
    -v
Por qué estos parámetros:

--curriculum: entrena primero en chain_1d + heavy_hex (mayor calidad), luego fine-tune en todas. Esto funcionó antes.
--epochs 5000 --patience 300: el modelo anterior hizo 2000 epochs con diverged status. Más epochs + más patience para que converja realmente.
--max-n 40: incluir los datos de N=30-40 que ahora tenemos para heavy_hex y ladder (antes no estaban)
--max-de-gap 0.10: filtro estricto, solo puntos con ΔE/gap < 10%
Output: unified_tfim_br_MT_residual+film_p1_v2.pt (gracias al nuevo versionado, el actual _p1.pt no se pierde)

Paso 2: Fine-tune para cada topología
Runner: run_finetune_from_mt.py

Una vez que el MT está entrenado, especializamos para cada topología:

# Heavy_hex (la que mejor respondió al fine-tune antes)
.venv/bin/python scripts/experiment_runners/cross_topology/run_finetune_from_mt.py \
    --topology heavy_hex --epochs 1000 --lr 3e-4 --max-n 40 -v

# Ladder
.venv/bin/python scripts/experiment_runners/cross_topology/run_finetune_from_mt.py \
    --topology ladder --epochs 1000 --lr 3e-4 --max-n 30 -v

# Square
.venv/bin/python scripts/experiment_runners/cross_topology/run_finetune_from_mt.py \
    --topology square --epochs 1000 --lr 3e-4 --max-n 16 -v
Output: unified_tfim_br_heavy_hex_fromMT_..._p1.pt, etc. (nuevos, sin pisar nada)

Paso 3: Validación (model_comparison)
# Evaluar MT vs ST vs fromMT en cada topología
for topo in heavy_hex ladder square; do
    .venv/bin/python scripts/experiment_runners/cross_topology/run_model_comparison.py \
        --topology $topo --target-n 10 16 20 --auto-detect --promote-best -v
done
Esto genera los eval reports que alimentan el scoreboard automáticamente.
