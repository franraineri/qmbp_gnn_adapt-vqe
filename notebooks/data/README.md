# Notebook Data

Pre-computed data for the demo notebooks. These files enable running
the notebooks without executing expensive training or hardware runs.

## Files

| File | Size | Description |
|------|:---:|-------------|
| `pretrained_mpnn_tfim_chain.pt` | ~2 MB | MPNN trained on TFIM chain_1d N=6 p=2 |
| `pretrained_gnn_qem.pt` | ~1 MB | GNN-QEM trained on chain+ladder N=10 p=1 |
| `sample_results.json` | ~5 KB | Pre-computed noisy samples for demo |

## Regenerating

To regenerate these files from scratch:

```bash
# MPNN checkpoint (Notebook 01)
python scripts/experiment_runners/noiseless/run_noiseless_pipeline.py \
    --n-qubits 6 --p-layers 2 --topology chain_1d --model tfim_longitudinal \
    --save-checkpoint notebooks/data/pretrained_mpnn_tfim_chain.pt

# GNN-QEM checkpoint (Notebook 03)
python scripts/experiment_runners/gnn_experiments/run_gnn_qem_training.py \
    --topology chain_1d ladder --n-qubits 10 --p-layers 1 \
    --save-checkpoint notebooks/data/pretrained_gnn_qem.pt

# Sample results (Notebook 03)
python -c "
from notebooks.data.generate_samples import generate_demo_samples
generate_demo_samples('notebooks/data/sample_results.json')
"
```

## Note

The `.pt` files are PyTorch model checkpoints. They require the same
package version to load correctly. If you get errors, regenerate them
using the commands above.
