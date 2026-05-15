# Project Status — GNN-HVA Framework

## Experiment Discipline (ALWAYS ENFORCE)

Before running ANY experiment or notebook execution:
1. State the hypothesis being tested. No hypothesis → no execution.
2. Check binnacles and poc-results.md — if the result is already established, do NOT re-run.
3. Every run must produce new learning. "Confirming what we know" is not learning after 3 seeds.
4. Do not duplicate information across binnacles — reference existing entries, don't copy them.
5. Known physics limits cannot be tuned past. See `experiment-protocol.md` for the full list.

## Current Phase
V6.1 complete and thesis-ready. All features validated at N=6 and N=10 (15 definitive runs).
Pipeline observability (DiagnosticCollector) now always-on — every run captures full metrics.

## Active Priority
1. **Hardware deployment on IBM Torino** — the only way to validate ZNE at N=10 (local simulation exhausted).
2. Start with **N=6, h=1.5** (safest — ZNE works in simulation, expect it works on hardware).
3. Then **N=10, h=1.5** with full mitigation stack (DD + twirling + TREX + ZNE via EstimatorV2 options).

## Critical Finding (2026-05-14/15)
Inhomogeneous ZNE (3 layouts) works at N=6 (R²>0.99, +40% gain) but **completely fails at N=10** (R²<0.05, negative gain). This is predicted by Tsubouchi et al. (2023): mitigation cost grows exp(depth × qubits).
- Experiment A: 7 layouts → R²=0.08 (still fails). Failure is fundamental, not statistical.
- Experiment B: DD cannot be tested locally (YGate not in FakeTorino basis). Must test on real hardware.
- **Conclusion: Local noisy simulation cannot validate ZNE at N=10. Go to real hardware where DD+twirling+TREX are native.**

## Key Constraints (always enforce)
- HVA only, never HEA. p ≤ 2 layers. |+⟩^N initial state.
- Descending sweep h=2→0. No angle wrapping.
- Pure energy cost in Phase 2. Never hybrid/observable cost.
- SparsePauliOp only. Primitives V2 only. Local observables on hardware.
- Fidelity filter ≥ 0.93 in Phase 3 training data.
- Hardware success criterion: ΔE/gap < 5% AND correct phase label (not fidelity).

## Stable Code (do NOT modify unless explicitly asked)
- `src/poc/v6/config.py` — shared dataclasses
- `src/poc/v6/hamiltonian_builder.py` — lattice generators and Hamiltonian construction
- `src/poc/v6/classical_solver.py` — exact diag + DMRG paths
- `src/poc/v6/hva_builder.py` — HVA circuit construction
- `src/poc/v6/pipeline_utils.py` — dataset save/load and integrity checks
- `src/poc/v6/vqe_optimizer.py` — multi-start VQE with callbacks
- `scripts/benchmark_v6.py` — benchmark runner
- `scripts/smoke_test.py` — V6.0 legacy smoke test (superseded by smoke_test_v61.py)
- `Makefile` — unified entry point

## Active Development Areas
- `src/poc/v6/pipeline_core.py` — shared 4-phase execution logic (PipelineCoreConfig, run_full_pipeline)
- `src/poc/v6/hardware_deployer_v61.py` — hardware + noisy_simulation modes (**next: DD pass, n_layouts scaling**)
- `src/poc/v6/mpnn_predictor.py` — MPNN architecture (per-parameter heads, edge features)
- `src/poc/v6/analysis_utils.py` — weight gradient analysis + diagnostic metrics
- `src/poc/v6/diagnostics.py` — pipeline observability (DiagnosticCollector, always-on)
- `src/poc/v6/experimental/` — isolated deprecated approaches (GATPredictor, augmentation)
- `scripts/run_v61_parametric.py` — parametric pipeline runner (now with N=12 configs, always-on diagnostics)
- `scripts/run_thesis_results.py` — thesis results consolidation
- `scripts/run_v61_noisy.py` — noisy simulation sweep (now with always-on diagnostics)
- `scripts/smoke_test_v61.py` — V6.1 integration smoke test

## Optimal Config (quick reference)
- **N=6**: GINConv h=64, L=3, 6000ep, lr=1e-3, 5 VQE restarts, fid≥0.93
- **N=10**: GINConv **h=128**, L=3, 6000ep, lr=1e-3, **patience=500**, **seed=43**
- **N=12**: Too slow for iterative experimentation on local hardware (~30+ min per run)

## ZNE Scaling Rule (from experiments + literature)
- N=6: 3 layouts sufficient (R²>0.99, linear regime)
- N=10: 3 layouts fails (R²<0.05, non-perturbative regime). Need O(n) layouts or DD pre-mitigation.
- General: n_layouts should scale with system size. CLP-ZNE (Rabinovich et al. 2025) uses O(n) cyclic permutations.
- N=12 take very long time and resources, do not execute this experiments

## Where to Start
Read `.kiro/knowledge/project-guide.md` first.
