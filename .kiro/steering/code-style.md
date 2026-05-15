---
inclusion: fileMatch
fileMatchPattern: "src/poc/v6/**/*.py"
---

# Code Style — V6 Modules

## Package Structure

```
src/poc/v6/
├── __init__.py              ← Public API (core modules only)
├── config.py                ← Shared dataclasses (stable)
├── config_v61.py            ← V6.1 constants + dataclasses
├── hamiltonian_builder.py   ← Phase 1: Hamiltonian + lattice (stable)
├── classical_solver.py      ← Phase 1: Exact diag + DMRG (stable)
├── hva_builder.py           ← Phase 2: HVA circuit (stable)
├── vqe_optimizer.py         ← Phase 2: Multi-start VQE (stable)
├── mpnn_predictor.py        ← Phase 3: MPNN model + training
├── pipeline_core.py         ← Shared 4-phase execution logic (NEW)
├── pipeline_utils.py        ← Cross-phase: dataset save/load (stable)
├── hardware_deployer.py     ← Phase 4: V6.0 legacy deployer
├── hardware_deployer_v61.py ← Phase 4: V6.1 full hardware path
├── qrc_pipeline.py          ← Phase 4: QRC fallback route
├── analysis_utils.py        ← Post-training: gradient analysis + metrics
├── diagnostics.py           ← Pipeline observability
└── experimental/            ← Deprecated/rejected approaches
    ├── __init__.py
    ├── gat_predictor.py     ← GATConv (rejected: instability for 1D)
    └── augmentation.py      ← θ interpolation (rejected: hurts accuracy)
```

## Lattice Construction
- Always use `make_lattice()` to create LatticeConfig instances.
- Never use `copy.copy()` or `copy.deepcopy()` on dataclasses to change `h`.
- To vary h across a sweep, construct a new LatticeConfig per point via `make_lattice()`.

## Imports

### Core (from package `__init__.py`)
```python
from src.poc.v6 import (
    HamiltonianBuilder, make_lattice, ClassicalSolver,
    HVACircuitBuilder, VQEOptimizer,
    LatticeConfig, VQEConfig, GroundTruthResult, VQEResult, DeployResult,
    save_phase12_dataset, load_phase12_dataset,
)
```

### MPNN / QRC (not in `__all__` to avoid heavy imports)
```python
from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn
from src.poc.v6.mpnn_predictor import save_mpnn_checkpoint, load_mpnn_checkpoint
from src.poc.v6.qrc_pipeline import QRCPipeline
```

### Pipeline Core (shared execution logic)
```python
from src.poc.v6.pipeline_core import (
    PipelineCoreConfig, PipelineResult,
    generate_h_grid, run_phase1, run_phase2, run_phase3, run_phase4,
    run_full_pipeline,
)
```

### V6.1 extensions (hardware + analysis)
```python
from src.poc.v6.hardware_deployer_v61 import HardwareDeployerV61
from src.poc.v6.analysis_utils import WeightGradientAnalyzer
from src.poc.v6.config_v61 import DeployResultV61, LayoutResult, GradientAnalysisResult
from src.poc.v6.config_v61 import NoisySweepResult, SweepSummary
```

### Observability
```python
from src.poc.v6.diagnostics import DiagnosticCollector, configure_pipeline_logging
from src.poc.v6.analysis_utils import (
    compute_snr, compute_theta_smoothness,
    compute_classification_confidence, compute_energy_decomposition,
)
```

### Experimental (deprecated — only for benchmark reproducibility)
```python
from src.poc.v6.experimental import GATPredictor, augment_graph_dataset
# WARNING: These are REJECTED approaches. Do NOT use in new code.
```

### V6.0 legacy deployer (use V6.1 instead)
```python
from src.poc.v6.hardware_deployer import HardwareDeployer  # LEGACY — prefer HardwareDeployerV61
```

## Qiskit Patterns
- SparsePauliOp via `from_sparse_list()` only.
- Energy evaluation: `StatevectorEstimator().run([(bound_qc, H)]).result()[0].data.evs`
- Never use `H.to_matrix()` for validation on large systems — use Pauli-level checks.

## Naming
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE` (in config.py / config_v61.py)
- Parameters: match the physics notation where possible (h, J, theta, psi)

## Error Handling
- Raise `ValueError` for constraint violations (p>2, invalid topology, phase coupling mismatch).
- Use `logging.warning()` for recoverable issues (DMRG fallback, gap computation failure).
- Use `assert` only for invariant checks (QRC no-training invariant).

## Pipeline Core Conventions
- `pipeline_core.py` is the single-source-of-truth for the Phase 1→4 execution pattern.
- Scripts should delegate to `run_full_pipeline()` or individual `run_phaseN()` functions.
- Do NOT duplicate pipeline logic in scripts — if a script needs custom behavior, extend `PipelineCoreConfig` or wrap the phase functions.
- `PipelineCoreConfig` contains all tunable parameters; scripts map their own CLI args to it.
