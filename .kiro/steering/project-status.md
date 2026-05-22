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
**V7 experiments complete** — all simulation-testable questions answered (12/22 experiments run, 10 skipped with justification).
**V8 experiments complete** — landscape analysis, scaling laws, methodological validation (13/19 executed, 6 skipped with justification).

## V8 Key Results (2026-05-22)
- **Scaling law (A3)**: h_min = 1.0 + 0.020·N^1.31 (R²=1.0000). Predicts N=20→2.00 (exact match). Exponent ≠ ν=1.
- **p=1 vs p=2**: β(p=1)=0.60 < β(p=2)=1.33 → p=1 scales better at large N.
- **No barren plateaus (F3)**: Landscape fluctuation >1.0 everywhere (confirms Mele et al. 2026).
- **p=1 landscape simpler (F3@p=1)**: Lower fluctuation (1.38 vs 1.99), higher fraction_near_gs.
- **fraction_near_gs**: Novel training-free boundary predictor (0% at h<1.0, 5%+ at h>1.5).
- **Hessian (B4)**: ALL VQE minima are genuine (0 saddle points). 73% eval savings with 1 restart. ✅
- **Hessian N=10 (B4@N=10)**: Saddle-free confirmed at N=10. Condition numbers N-independent. ✅
- **Analytical init (B1)**: 97% fewer iterations but converges to wrong basin. Warm-start wins.
- **DyPP (F1)**: Only 8-13% iteration savings (hypothesis was 30-50%). Rejected. ❌
- **Weight-space phase detection (D1)**: MPNN-A peak at h≈0.7 (near h_c=1.0). Novel zero-QPU method. ✅
- **D1 regularized**: Dropout=0.1 makes detection robust (std=0.13 vs 0.90). 5 seeds all consistent. ✅
- **Parameter freezing (B2)**: 2/4 params frozen at h≥1.5, 0% accuracy loss. ✅
- **Longitudinal field (E4)**: HVA p=2 fails at g>0 (fidelity drops to 0.89 at g=0.1). HVA is model-specific. ❌
- Binnacles: `binnacle-v8-experiments.md`, `binnacle-v8-experiments-round1.md`

## V8 Validated Decisions (2026-05-22)
- **VQE restarts:** 1 restart sufficient at N=6 AND N=10 (B4: no saddle points in HVA landscape at either size)
- **Landscape is N-independent:** Condition numbers at N=10 match N=6 within 10% (B4@N=10)
- **VQE at N=20 p=1:** 3 restarts + 100 maxiter + MPS (chi=64) → ΔE/gap=1.58% (2/3 seeds). 5 restarts needed for full reliability (C3)
- **Sign canonicalization:** NOT needed — descending warm-start breaks Z₂ naturally (C3, 3 runs confirm 0% effect)
- **N=20 p=1 local minimum:** ΔE/gap=0.437 basin exists — requires ≥5 restarts to reliably escape (C3)
- **Parameter freezing:** Freeze θ_zz2, θ_x2 at h≥1.5 (B2: 0% accuracy loss)
- **Optimal VQE at h≥1.5:** 1 restart + 2 active params = **75% cost reduction**
- **DyPP:** Rejected — standard warm-start is near-optimal for 4-param HVA (F1)
- **Longitudinal field:** HVA p=2 is TFIM-specific, not model-agnostic (E4)
- **Phase detection:** Weight gradient peaks detect h_c when MPNN loss≈0.002 (D1, D1-dense)
- **Phase detection robustness:** Dropout=0.1 → std=0.13 (vs 0.90 without). Reliable across 5 seeds (D1-reg)
- **Physics-informed loss (C1):** +3.9% mean improvement (max +17.5% at h=1.75). Safe, no regression. Modest at N=6.
- **Physics-informed loss (C1@N=10):** -12.3% (HURTS). Only helps with full h-range training, not valid-regime-only.
- **Phase detection caveat:** Overfitting (loss=0) shifts peak to h≈0.69; needs regularization (D1-dense)
- **p=1 landscape (F3@p=1):** Simpler (fluctuation 1.38 vs 1.99), higher random GS accessibility at h≥1.5

## Active Priority
1. **Hardware deployment on IBM Torino** — the only way to validate ZNE at N=10 (local simulation exhausted).
2. Start with **N=6, h=1.5** (safest — ZNE works in simulation, expect it works on hardware).
3. Then **N=10, h=1.5** with full mitigation stack (DD + twirling + TREX + ZNE via EstimatorV2 options).
4. Use **SPSA (a=0.1, c=0.05, A=10)** for hardware VQE refinement — validated as 3× better than COBYLA under noise (V7 experiment 4C).
5. **Random baseline comparison now default** — every Phase 4 run compares warm-start vs cold-start (gain metric). Use `--no-baseline` to skip.
6. **Heisenberg model extension** — validate framework is model-agnostic (in progress).

## Critical Findings (2026-05-14/15/18)
Inhomogeneous ZNE (3 layouts) works at N=6 (R²>0.99, +40% gain) but **completely fails at N=10** (R²<0.05, negative gain). This is predicted by Tsubouchi et al. (2023): mitigation cost grows exp(depth × qubits).
- Experiment A: 7 layouts → R²=0.08 (still fails). Failure is fundamental, not statistical.
- Experiment B: DD cannot be tested locally (YGate not in FakeTorino basis). Must test on real hardware.
- **Conclusion: Local noisy simulation cannot validate ZNE at N=10. Go to real hardware where DD+twirling+TREX are native.**

### V7 Key Results (2026-05-18)
- **L-BFGS-B definitively optimal** for noiseless VQE (1A: wins by 31-95% over all Nevergrad methods)
- **SPSA optimal config: a=0.1, c=0.05, A=10** (4A: grid search over 36 configs × 10 seeds)
- **SPSA refinement HURTS warm-start** (4B: -146% at h=2.0) — don't refine good MPNN predictions
- **QRC = MPNN at N=10** (2B: <1% difference, both ceiling-limited) — predictor is NOT the bottleneck
- **MPS exact for 1D HVA** (3A/3B: |MPS-SV|=1e-14, chi=64 sufficient) — enables N=20 scaling
- **MPS VQE at N=20 passes at h=2.0** (3C: ΔE/gap≈1%) — valid regime shifts with N
- **Noise-aware training fails** under shot noise (5B: 6× worse) — only coherent errors could help
- **Iterative refinement modest** (5E: 9% gain, saturates in 2 rounds)

## Key Constraints (always enforce)
> Full constraint list with rationale in `.kiro/skills/quantum/SKILL.md`.
> Summary for quick reference:
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
- `src/poc/v6/hardware_deployer_v61.py` — hardware + noisy_simulation modes + **deploy_with_baseline()** (**next: DD pass, n_layouts scaling**)
- `src/poc/v6/mpnn_predictor.py` — MPNN architecture (per-parameter heads, edge features)
- `src/poc/v6/analysis_utils.py` — weight gradient analysis + diagnostic metrics
- `src/poc/v6/diagnostics.py` — pipeline observability (DiagnosticCollector, always-on, **+record_baseline()**)
- `scripts/run_v61_parametric.py` — parametric pipeline runner (now with N=12 configs, always-on diagnostics, **+baseline comparison**)
- `scripts/run_thesis_results.py` — thesis results consolidation
- `scripts/run_v61_noisy.py` — noisy simulation sweep (now with always-on diagnostics)
- `scripts/smoke_test_v61.py` — V6.1 integration smoke test
- `scripts/experiments_hamed_v7/` — V7 full experiment suite (22 sub-experiments, master runner)
- `scripts/experiments_hamed_v7/experiment_p1_scaling.py` — p=1 depth-scaling study (6A-6D)

## Dead Code (removed 2026-05-22)
- `src/poc/v6/pipeline_core.py` — documented but zero imports anywhere (DELETED)
- `src/poc/v6/experimental/` — GATPredictor + augmentation (both rejected, never existed on disk)
- `src/poc/v6/hardware_deployer.py` — V6.0 legacy (DELETED, superseded by V6.1)

## Optimal Config (quick reference)
- **N=6**: GINConv h=64, L=3, 6000ep, lr=1e-3, 5 VQE restarts, fid≥0.93
- **N=10**: GINConv **h=128**, L=3, 6000ep, lr=1e-3, **patience=500**, **seed=43**
- **N=20 (MPS)**: chi=64 sufficient, L-BFGS-B + 3-5 restarts, descending warm-start, valid at h≥2.0
- **N=20 (full pipeline)**: h∈[1.5,2.0] only (11 pts), 7 restarts σ=0.3, NO filter, MPNN h=128, **ΔE/gap=1.75% ✅**
- **N=12**: Too slow for iterative experimentation on local hardware (~30+ min per run)
- **Hardware SPSA**: a=0.1, c=0.05, A=10, n_iterations=200 (from V7 4A grid search)
- **N=20 (p=1)**: 2 params, h∈[2.25,4.0], 5 restarts, StatevectorEstimator, MPNN h=128 (trivial mapping)

## V7 Validated Decisions (2026-05-18)
- **Optimizer (noiseless):** L-BFGS-B with 5 restarts. Nevergrad 31-95% worse (1A).
- **Optimizer (hardware):** SPSA (a=0.1, c=0.05). 3× better than COBYLA under noise (4C).
- **Warm-start refinement:** Do NOT apply SPSA after MPNN prediction in simulation (4B: hurts).
- **Predictor:** MPNN = QRC at N=10 (2B). Predictor is NOT the bottleneck. Keep MPNN for scalability.
- **MPS simulator:** Exact for 1D HVA (3A/3B). chi=64 sufficient. Enables N=20 VQE.
- **Noise-aware training:** Fails under shot noise (5B: 6× worse). Only coherent errors could help.
- **Iterative refinement:** Modest 9% gain, saturates in 2 rounds (5E). Not worth the complexity.
- **Valid regime scales with N:** N=6 h≥1.25, N=10 h≥1.5, N=20 h≥2.0 (HVA expressibility limit).

## p=1 Scaling Results (2026-05-21)
- **p=1 valid regime**: N=6 h≥1.6, N=10 h≥1.9, N=20 h≥2.25 (shift of +0.25 to +0.40 vs p=2)
- **Shift decreases with N**: +0.35 at N=6, +0.40 at N=10, +0.25 at N=20
- **Seed-independent at N≤10**: All 3 seeds give identical θ_opt (single global minimum)
- **N=20 has Z₂ symmetry issue**: Seeds find equivalent minima with different sign conventions
- **θ_x constant**: ±1.178 (= ±3π/8) for all h; only θ_zz varies → effectively 1D mapping
- **CX reduction**: Exactly 50% at all N (p=1 N=20 = 38 CX ≈ p=2 N=10 = 36 CX)
- **MPNN deployment at N=20**: Only h=3.0 passes (6 training points too few; sign canonicalization needed)
- **Hardware candidate**: p=1 N=20 on IBM Torino (VQE validated, same CX budget as p=2 N=10)
- **TODO**: Fix init at N=20 (analytical guess), canonicalize signs, increase training density
- Binnacle: `documentation/binnacles/binnacle-p1-scaling.md`
- Script: `scripts/experiments_hamed_v7/experiment_p1_scaling.py`

## ZNE Scaling Rule (from experiments + literature)
- N=6: 3 layouts sufficient (R²>0.99, linear regime)
- N=10: 3 layouts fails (R²<0.05, non-perturbative regime). Need O(n) layouts or DD pre-mitigation.
- General: n_layouts should scale with system size. CLP-ZNE (Rabinovich et al. 2025) uses O(n) cyclic permutations.
- N=12 take very long time and resources, do not execute this experiments

## Where to Start
Read `.kiro/knowledge/project-guide.md` first.
