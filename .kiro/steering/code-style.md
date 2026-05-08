---
inclusion: fileMatch
fileMatchPattern: "src/poc/v6/**/*.py"
---

# Code Style — V6 Modules

## Lattice Construction
- Always use `make_lattice()` to create LatticeConfig instances.
- Never use `copy.copy()` or `copy.deepcopy()` on dataclasses to change `h`.
- To vary h across a sweep, construct a new LatticeConfig per point via `make_lattice()` or the `LatticeConfig(...)` constructor directly.

## Imports
- Prefer package-level imports: `from src.poc.v6 import HamiltonianBuilder, make_lattice`
- For MPNN/QRC/HardwareDeployer (not in `__all__` to avoid heavy imports at package level):
  `from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn`
  `from src.poc.v6.mpnn_predictor import save_mpnn_checkpoint, load_mpnn_checkpoint`
  `from src.poc.v6.qrc_pipeline import QRCPipeline`
  `from src.poc.v6.hardware_deployer import HardwareDeployer`
- V6.1 extensions (hardware + analysis):
  `from src.poc.v6.hardware_deployer_v61 import HardwareDeployerV61`
  `from src.poc.v6.analysis_utils import WeightGradientAnalyzer`
  `from src.poc.v6.config_v61 import DeployResultV61, LayoutResult, GradientAnalysisResult`

## Qiskit Patterns
- SparsePauliOp via `from_sparse_list()` only.
- Energy evaluation: `StatevectorEstimator().run([(bound_qc, H)]).result()[0].data.evs`
- Never use `H.to_matrix()` for validation on large systems — use Pauli-level checks.

## Naming
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE` (in config.py)
- Parameters: match the physics notation where possible (h, J, theta, psi)

## Error Handling
- Raise `ValueError` for constraint violations (p>2, invalid topology, phase coupling mismatch).
- Use `logging.warning()` for recoverable issues (DMRG fallback, gap computation failure).
- Use `assert` only for invariant checks (QRC no-training invariant).
