# Config Presets

YAML-based experiment configurations for the qmbp pipeline.

## Usage

```bash
# Run a preset
python scripts/experiment_runners/noiseless/run_noiseless_pipeline.py \
    --preset noiseless/tfim_heavy_hex_n20_p4

# Override specific values from preset
python scripts/experiment_runners/noiseless/run_noiseless_pipeline.py \
    --preset noiseless/tfim_heavy_hex_n20_p4 --h-points 20

# List available presets
python -c "from qmbp_simulation.framework.presets import list_presets; print(list_presets())"
```

## Structure

```
configs/presets/
├── noiseless/          ← StatevectorEstimator pipelines
│   ├── defaults.yaml   ← Shared defaults for noiseless category
│   └── *.yaml          ← Individual preset configs
├── noisy/              ← FakeTorino + ZNE validation
│   └── defaults.yaml
└── hardware/           ← Real QPU deployment
    └── defaults.yaml
```

## Preset Format

```yaml
runner: noiseless_pipeline    # Runner ID (informational)
model: tfim                   # Hamiltonian model
topology: heavy_hex           # Lattice topology
n_qubits: 20                  # System size
p_layers: 4                   # HVA circuit depth
h_min: 1.25                   # Transverse field range
h_max: 3.0
h_points: 40                  # Sweep density
maxiter: 1000                 # VQE max iterations
n_restarts: 7                 # VQE restarts per h-point
seeds: [42, 43, 44]           # Random seeds
description: "..."            # Human-readable description
```

## Precedence

CLI args > preset values > category defaults.yaml > runner defaults
