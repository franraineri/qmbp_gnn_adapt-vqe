# Project Status — GNN-HVA Framework

## Current Phase
V6.0 modular architecture is complete, validated, and exhaustively benchmarked. 40+ pipeline executions across 14 configurations (including GATConv, data augmentation, grid density, restart sigma) have confirmed the optimal hyperparameters and established that the h=1.25 checklist ceiling (2–3/6) is a physics limit of HVA p=2, not a pipeline deficiency. The pipeline achieves 5/6 at h=1.5 and 4–5/6 at h=1.4.

## Current Priority
1. Test MPNN on **ladder topology** (N=6 ladder) to validate lattice-agnostic generalization.
2. Scale to **N=10** — use `--mpnn-hidden 128` (augmentation hurts at N=10, confirmed by binnacle).
3. Prepare thesis results chapter with the benchmark data.
4. Hardware deployment on IBM Torino (Phase 4 with EstimatorV2 + ZNE).
5. Implement **inhomogeneous ZNE** (Uvarov et al. 2024) — multiple qubit mappings for natural noise scaling.
6. Add **learned DD sequences** (Pokharel et al. 2025) via Qiskit DD pass manager.

## Literature-Informed Insights (see `.kiro/knowledge/literature-synthesis.md`)
- h=1.25 ceiling confirmed as physics limit by Tripathi et al. 2026 (independent validation)
- Hardware noise will broaden critical crossover (Sharma 2026) — expected, not a failure
- GNN > CNN by 36% for circuit property prediction (Meng 2025) — validates GINConv
- Shot noise (~1.6e-2 at 4096 shots) will dominate over gate errors on hardware
- MPNN weights may encode phase transition info (Hernandes 2025) — analyze post-training
- Consider NN-enhanced ZNE (Sun 2025) for better noise extrapolation on hardware

## Where to Start
Read `.kiro/knowledge/project-guide.md` first — it maps the entire repository and explains where to find everything.

## Stable Code (do NOT modify unless explicitly asked)
- `src/poc/v6/config.py` — shared dataclasses
- `src/poc/v6/hamiltonian_builder.py` — lattice generators and Hamiltonian construction
- `src/poc/v6/classical_solver.py` — exact diag + DMRG paths
- `src/poc/v6/hva_builder.py` — HVA circuit construction
- `src/poc/v6/pipeline_utils.py` — dataset save/load and integrity checks
- `scripts/benchmark_v6.py` — benchmark runner
- `scripts/smoke_test.py` — smoke test
- `Makefile` — unified entry point

## Active Development Areas
- `src/poc/v6/mpnn_predictor.py` — MPNN architecture (potential per-parameter heads)
- `src/poc/v6/hardware_deployer.py` — real hardware deployment path
- `src/poc/v6/qrc_pipeline.py` — QRC reservoir design improvements
- Notebooks in `src/poc/v6/` — orchestration and visualization

## Key Constraints (always enforce)
- HVA only, never HEA. p ≤ 2 layers. |+⟩^N initial state.
- Descending sweep h=2→0. No angle wrapping.
- Pure energy cost in Phase 2. Never hybrid/observable cost (V5.x lesson).
- SparsePauliOp only. Primitives V2 only. Local observables on hardware.
- Fidelity filter ≥ 0.93 in Phase 3 training data.

## Optimal Configuration (from 40+ benchmark runs)
VQE: 5 restarts, σ=0.1, maxiter=1000 | MPNN: GINConv, h=64, L=3, 6000 epochs, lr=1e-3 | fid≥0.93
- GATConv tested and rejected (adds instability for 1D chains)
- Data augmentation: tested and rejected for N=10 (hurts — linear interpolation inaccurate in complex θ landscape)
- Denser h-grid: not needed (27 points sufficient)

## Validation Targets
| h_test | Expected checklist |
|--------|-------------------|
| 1.25 | 2–3/6 (near critical region, HVA ceiling limits fidelity) |
| 1.4 | 4–5/6 (fidelity crosses 99.5% threshold) |
| 1.5 | 5/6 (best achievable — only ΔE<1e-2 fails, physics limit) |
