# Diagrams

| File | Topic |
|------|-------|
| `diagram_pipeline_overview.txt` | GNN-HVA pipeline: Phase 1 (ED/DMRG) → 2 (VQE) → 3 (MPNN training) → 4 (zero-shot θ) |
| `diagram_ibm_comparison.txt` | GNN-HVA vs IBM/Qedma (arXiv:2607.24937): non-trivial state prep + quench dynamics |

## Cross-refs

### `diagram_pipeline_overview.txt`
- Hamiltonian: `src/qmbp_simulation/models/hamiltonian.py` (`HamiltonianBuilder`, `make_lattice`)
- ED/DMRG solver: `src/qmbp_simulation/solvers/classical.py` (`ClassicalSolver`)
- VQE: `src/qmbp_simulation/optimizers/vqe.py` (`VQEOptimizer`)
- HVA circuit: `src/qmbp_simulation/circuits/hva.py` (`HVACircuitBuilder`)
- MPNN: `src/qmbp_simulation/predictors/unified_mpnn.py` (`UnifiedMPNN`), `src/qmbp_simulation/predictors/mpnn.py`
- Graph build: `src/qmbp_simulation/predictors/unified_graph.py`
- Model zoo: `src/qmbp_simulation/predictors/model_zoo.py`
- Training NPZ: `data/multi_n_training/{topo}_N{n}_p{p}.npz`
- Quality metric (ΔE/gap): `src/qmbp_simulation/analysis/metrics.py` (`compute_deploy_summary`)
- Eval reports: `results/extrapolation_evals/{topo}_p{p}/`, `results/model_evaluation_report.md`
- Status: `.kiro/steering/project-status.md`, `results/best_results_scoreboard_p1.md`

### `diagram_ibm_comparison.txt`
- MPS χ study (Panel A/B viability): `scripts/analysis/run_representability_mps.py`, `scripts/analysis/analyze_representability.py`, `scripts/analysis/representability_io.py`
- MPS χ results: `results/classical_representability/` (`SUMMARY.md`, `study_*.json`, `figures/`)
- Trotter evolution: `src/qmbp_simulation/circuits/trotter.py`
- MPS backend: `src/qmbp_simulation/execution/mps_backend.py` (`MPSBackend`)
- Hardware (QPU) backend: `src/qmbp_simulation/execution/hardware/backend.py` (`HardwareBackend`)
- Quench study: `scripts/experiment_runners/scaling/run_quench_dynamics_study.py`
- Hardware results: `results/ibm_pittsburgh/`, `results/ibm_boston_mitigated/`, `results/haiqu_recovered/`
- Literature review (entanglement barrier, IBM classical-sim caveat): `internal/documentation/analysis/26_bond_dimension_classical_simulability_review.md`
