# Project Status — GNN-HVA Framework

## Current Phase
V6.1 modular architecture is complete and validated. V6.0 core (Phases 1–3) is exhaustively benchmarked with 40+ pipeline executions across 14 configurations. V6.1 adds production-ready hardware deployment (inhomogeneous ZNE, DD, twirling, TREX, NN extrapolation), MPNN enhancements (per-parameter heads, edge features, checkpoint save/load), and weight gradient analysis (Hernandes et al. 2025). The h=1.25 checklist ceiling (2–3/6) is confirmed as a physics limit of HVA p=2, not a pipeline deficiency.

## Current Priority
1. Test MPNN on **ladder topology** (N=6 ladder) to validate lattice-agnostic generalization.
2. Scale to **N=10** — use `--mpnn-hidden 128` (augmentation hurts at N=10, confirmed by binnacle).
3. Prepare thesis results chapter with the benchmark data.
4. Hardware deployment on IBM Torino (Phase 4 with EstimatorV2 + ZNE via `HardwareDeployerV61`).
5. Validate **inhomogeneous ZNE** on real hardware (Uvarov et al. 2024) — `LayoutSelector` ready.
6. Add **learned DD sequences** (Pokharel et al. 2025) via Qiskit DD pass manager.

## Literature-Informed Insights (see `.kiro/knowledge/literature-synthesis.md`)
- h=1.25 ceiling confirmed as physics limit by Tripathi et al. 2026 (independent validation)
- Hardware noise will broaden critical crossover (Sharma 2026) — expected, not a failure
- GNN > CNN by 36% for circuit property prediction (Meng 2025) — validates GINConv
- Shot noise (~1.6e-2 at 4096 shots) will dominate over gate errors on hardware
- MPNN weights encode phase transition info (Hernandes 2025) — **validated in V6.1 smoke test** (peak at h≈1.20)
- NN-enhanced ZNE (Sun 2025) implemented in `NNExtrapolator` — ready for hardware

## Where to Start
Read `.kiro/knowledge/project-guide.md` first — it maps the entire repository and explains where to find everything.

## Stable Code (do NOT modify unless explicitly asked)
- `src/poc/v6/config.py` — shared dataclasses
- `src/poc/v6/hamiltonian_builder.py` — lattice generators and Hamiltonian construction
- `src/poc/v6/classical_solver.py` — exact diag + DMRG paths
- `src/poc/v6/hva_builder.py` — HVA circuit construction
- `src/poc/v6/pipeline_utils.py` — dataset save/load and integrity checks
- `src/poc/v6/vqe_optimizer.py` — multi-start VQE with callbacks
- `scripts/benchmark_v6.py` — benchmark runner
- `scripts/smoke_test.py` — V6.0 smoke test
- `Makefile` — unified entry point

## Active Development Areas
- `src/poc/v6/mpnn_predictor.py` — MPNN architecture (per-parameter heads, edge features)
- `src/poc/v6/hardware_deployer_v61.py` — real hardware deployment path (ready for IBM Torino)
- `src/poc/v6/analysis_utils.py` — weight gradient analysis (validated, may add more analyzers)
- `src/poc/v6/qrc_pipeline.py` — QRC reservoir design improvements
- `scripts/run_notebooks.py` — notebook executor with auto-registry
- `scripts/smoke_test_v61.py` — V6.1 integration smoke test
- Notebooks in `src/poc/v6/` — orchestration and visualization

## Key Constraints (always enforce)
- HVA only, never HEA. p ≤ 2 layers. |+⟩^N initial state.
- Descending sweep h=2→0. No angle wrapping.
- Pure energy cost in Phase 2. Never hybrid/observable cost (V5.x lesson).
- SparsePauliOp only. Primitives V2 only. Local observables on hardware.
- Fidelity filter ≥ 0.93 in Phase 3 training data.
- Hardware success criterion: ΔE/gap < 5% AND correct phase label (not fidelity).

## Optimal Configuration (from 40+ benchmark runs)
VQE: 5 restarts, σ=0.1, maxiter=1000 | MPNN: GINConv, h=64, L=3, 6000 epochs, lr=1e-3 | fid≥0.93
- GATConv tested and rejected (adds instability for 1D chains)
- Data augmentation: tested and rejected for N=10 (hurts — linear interpolation inaccurate in complex θ landscape)
- Denser h-grid: not needed (27 points sufficient)
- Per-parameter heads: available but optional (marginal improvement for 1D TFIM)

## Validation Targets
| h_test | Expected checklist |
|--------|-------------------|
| 1.25 | 2–3/6 (near critical region, HVA ceiling limits fidelity) |
| 1.4 | 4–5/6 (fidelity crosses 99.5% threshold) |
| 1.5 | 5/6 (best achievable — only ΔE<1e-2 fails, physics limit) |

## Scripts Quick Reference

| Script | Purpose | Time |
|--------|---------|------|
| `scripts/smoke_test.py` | V6.0 end-to-end (3 h-points, basic) | ~7s |
| `scripts/smoke_test_v61.py` | V6.1 full pipeline (12 h-points, deployer, gradient analysis) | ~16s |
| `scripts/benchmark_v6.py` | Configurable multi-run benchmark | ~50s (N=6) |
| `scripts/run_notebooks.py` | Notebook executor with auto-registry + binnacle | ~15min |

### Notebook Executor Features (V6.x)
- Wall-clock timeout guard (prevents runaway VQE)
- Auto-extracts structured metrics (fidelity, MSE, ΔE/gap, checklist, phase, gradients, ZNE)
- Peak memory tracking (critical for N=10 scaling)
- Environment capture (git state, package versions, platform)
- Binnacle-ready markdown output (auto-observations, run-to-run comparison)
- Results pruning (`--keep-last N`)
- Structured exit codes: 0=pass, 1=execution fail, 2=validation fail, 3=pre-flight fail
