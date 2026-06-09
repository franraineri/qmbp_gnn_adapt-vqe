# Project Status — GNN-HVA Framework

**Last updated**: 2026-06-08

## Experiment Discipline (ALWAYS ENFORCE)

1. State the hypothesis being tested. No hypothesis → no execution.
2. Check binnacles and poc-results.md — if the result is already established, do NOT re-run.
3. Every run must produce new learning. "Confirming what we know" is not learning after 3 seeds.
4. Do not duplicate information across binnacles — reference existing entries, don't copy them.
5. Known physics limits cannot be tuned past. See `experiment-protocol.md` for the full list.

## Current Phase

**All simulation work complete.** Next: IBM Torino hardware deployment + thesis writing.
- V7 (12/22 experiments), V8 (18/19), V9 Heisenberg (30 runs), S-series (6 experiments) — all done.
- Tier 1 extensions (T1a/T1b/T1c) — executed 2026-06-03, 3 confirmed.
- Hardware rehearsal — critical finding: CES-ZNE fails on heavy_hex, need gate-folding ZNE.
- ZNE cross-topology validation (ZNE_CROSS_TOPO) — PEA wins 18/18, t=46.32, p<10⁻¹⁹. All 6 ZNE experiments confirmed.
- **PEA_TRIANGULAR** (2026-06-05) — PEA +96.8% on triangular, t=111.22, 9/9 wins. All 4 topologies now have PEA validation.
- **GNN-QEM cross-topology** (2026-06-05) — 100% improvement on unseen heavy_hex (zero-shot: +72.3% error reduction). Thesis contribution validated.
- **Affine overshoot audit** (2026-06-05) — 0% overshoot in 102 ZNE records. Safety net confirmed (zero-cost insurance).
- Confirmed: 22 experiments. Rejected: 8 (valid negative results). Failed: 2.
- Useful-outcome rate: 93% (28/30 formal experiments produce actionable knowledge).
- 210+ pipeline runs executed across 5 topologies (chain_1d, ladder, triangular, kagome, heavy-hex).
- **GNN-QEM cross-topology validated** (2026-06-05): Zero-shot transfer from chain_1d+ladder → heavy_hex gives 100% improvement rate (+72.3% error reduction). Model: GINConv(3L, h=64), 30K params. Ref: `results/gnn_qem/cross_topology_results.json`.
- **GNN-QEM NOT composable with PEA** (2026-06-05): Post-ZNE pipeline shows GNN regresses 15/15 points. Model trained on large errors (10-25 units) over-corrects post-PEA residuals (0.01 units). Use as ALTERNATIVE to ZNE, not after it. Ref: `results/gnn_qem/post_zne_validation.json`.
- **GNN-QEM ablation** (2026-06-06): Graph IS essential without E_noisy (GNN 100% vs MLP 67% vs Linear 0%). With E_noisy, correction is 99.96% linear — graph adds +11% precision only. Claim reframed: graph captures noise propagation in predictive mode; regularization in correction mode. Ref: `results/gnn_qem/ablation_no_enoisy_results.json`.
- **GNN-QEM circuit selection validated** (2026-06-06): Predictive mode (no E_noisy) with VQE-realistic errors achieves Spearman ρ=0.945 for ranking circuits by expected error (100% binary accuracy). Cannot CORRECT but perfectly RANKS difficulty. Use for pre-execution layout/config selection. Ref: `results/gnn_qem/vqe_realistic_results.json`.
- **PEA-ZNE now validated on ALL 4 topologies**: chain_1d (+97%), ladder (+91%), heavy_hex (+98%), triangular (+97%). Universal superiority confirmed.

## Active Priority

1. **Hardware deployment on IBM Torino** — local simulation exhausted, hardware is the only remaining validation.
   - Hardware module fixes applied (2026-06-06): credential passing, TLS drift monitoring, QPU metrics, structured EstimatorV2 options.
   - Persistence enhanced: summary.json now saves 13 additional fields (mitigation_strategy, per_site_x, affine, GNN-QEM, etc.).
   - Deployment script: `scripts/experiment_runners/hardware/run_ibm_torino_deployment.py` (Tier 0→3 auto-advancing).
   - Rehearsal V2 passed 3/3 after fixes (2026-06-06): Section 1 ΔE/gap 0.2-0.8%, Section 2 PEA-ZNE 2-4%, Section 3 verdict=PASS.
   - **Status: READY FOR QPU** — only IBM credentials needed.
2. **Thesis writing** — Chapter 5 compilation from `documentation/analysis/09_thesis_tables.md`.
3. **MPS Scaling to N>30** (2026-06-07) — Pipeline validated beyond statevector limits.
   - **N=40 PASS ✅**: 5/5 h-points, mean ΔE/gap=0.49%, max=0.60%, 26 min total.
   - **N=50 PASS ✅**: 5/5 h-points, mean ΔE/gap=0.36%, max=0.49%, 30 min total.
   - **N=80 PASS ✅**: 5/5 h-points, mean ΔE/gap=0.08%, max=0.10%, 109s total.
   - **Scaling law confirmed** at N=40, 50, 80: consistent +0.50 offset from prediction.
   - Infrastructure: `MPSBackend` (aer_mps + direct path for N>63), COBYLA dispatch, dynamic χ.
   - Multi-seed run (42,43,44) at N=40 in progress for Phase 3 MPNN training data.
   - Ref: `documentation/binnacles/binnacle-mps-scaling.md`.
4. **θ_pred Validation Module** (2026-06-07) — 7-level modular quality assurance for MPNN outputs.
   - Auto-integrated in `PipelineRunner.run_phase4()` (L1-L4 default, configurable up to L7).
   - Levels: L1=bounds, L2=NaN/Inf, L3=interpolation, L4=fidelity, L5=gradient, L6=MC-Dropout, L7=sensitivity.
   - Outputs `diagnostics.theta_validation[]` in result JSON. Accessible via `ValidationRunner.validate_theta_prediction()`.
   - Ref: `src/qmbp_simulation/analysis/theta_validator.py`, `tests/test_theta_validator.py`.
5. **Cross-N Zero-Shot GNN** (2026-06-08) — GNN generalizes to unseen system sizes.
   - **Discovery**: BatchNorm is harmful for cross-N on topologies with nodal symmetry (chain_1d). Fix: `norm_type="none"`.
   - Package fix: `MPNNPredictor(norm_type="none"|"layer"|"batch")` — backward compatible.
   - Validated: Train N=40+80 → predict N=50,60,70,100: **25/25 PASS**, mean ΔE/gap=0.16%.
   - Multi-seed confirmed: seeds 42/43/44 all 5/5 PASS (std=0.074%).
   - Extrapolation: N=100 (beyond training) achieves 0.18%, GNN beats scipy 2.6×.
   - Pending: Bond-resolved (79D necessity proof).
   - Scripts: `run_zero_shot_cross_n_v3.py`, `run_cross_n_ablation_suite.py`, `run_bond_resolved_cross_n.py`.
   - Ref: `documentation/binnacles/binnacle-cross-n-zero-shot.md`, `documentation/analysis/19_cross_n_validation_plan.md`.

## Key Constraints (always enforce)

> Full list with rationale: `.kiro/skills/quantum/SKILL.md`

- HVA only, never HEA. p ≤ 2 layers. |+⟩^N initial state (TFIM). Néel state (Heisenberg).
- Descending sweep h_max→h_min. No angle wrapping. Pure energy cost in Phase 2.
- SparsePauliOp only. Primitives V2 only. Local observables on hardware.
- Fidelity filter ≥ 0.93 (TFIM), ≥ 0.60 (Heisenberg) in Phase 3 training data.
- Hardware success: ΔE/gap < 5% AND correct phase label (not fidelity).
- **VQE validation runs automatically** after Phase 2: variational principle, energy bounds, θ bounds, convergence rate, sweep quality. Results in `diagnostics.vqe_validation`. CLI: `--no-validate-vqe` to disable, `--strict-validation` to abort on CRITICAL.
- **Heisenberg HVA p≤2 CANNOT work** — do not attempt (V9: 30 runs + N=10/16 scaling confirm).
- **Kitaev chain NOT viable** — 20 CZ@N=6 (exceeds ZNE), fid=16% max. Do not implement.
- **TFIM+longitudinal WORKS** — fid≥0.98 at g=0.5, 0 extra CX gates (E4b validated).
- **TFIM frustrated (J1-J2) WORKS in simulation** — fid≥0.99 at J₂=0.5, but 27 CZ@N=6 (no ZNE for N≥6).
- **ZNE threshold**: ~18 CX gates. p=2 N=10 (36 CX) fails. Use p=1 for N≥10 hardware.
- **CES-ZNE fails on heavy_hex**: All good layouts have CES≈0.15 (no spread). Use IBM gate-folding ZNE instead. Ref: `documentation/analysis/11_hardware_rehearsal_findings.md`.
- **Gate-folding ZNE validated**: +12% mean gain, R²>0.99, wins 9/12 h-points vs CES-ZNE (t=3.28, p<0.01). Robust across chain_1d/heavy_hex/ladder. Ref: `documentation/binnacles/binnacle-gate-folding-zne.md`.
- **PEA-ZNE validated**: +95% mean gain (8.4× GF-ZNE), R²=0.998, std=2.9% (3 seeds × 4 h-points). Requires qiskit-aer. Recommended primary strategy for hardware. Ref: `documentation/binnacles/binnacle-gate-folding-zne.md`.
- **Adaptive ZNE default changed to pea_primary** (2026-06-05): `run_adaptive_zne()` now uses PEA as primary (not GF). GF R²>0.99 does NOT guarantee accuracy (89.8% ΔE/gap observed). Use `strategy="gf_primary"` only for legacy compat. Ref: Kim et al. Nature 618 (2023), QESEM arXiv:2508.10997.
- **Block-level ZNE available for p≥2**: `run_block_zne()` folds only 1 HVA layer → shallower folded depth, better linearity. Ref: arXiv:2507.23314.
- **Affine correction**: `affine_correct_energy()` clips ZNE energies to [E_ground, E_upper]. Zero cost. Ref: Wang et al. arXiv:2604.16815.
- **TLS drift monitoring**: `take_calibration_snapshot()` + `check_calibration_drift()` for hardware runs. Abort if T1 drift > 20%. Ref: Nature Comms 2025 (arXiv:2407.02467).
- **GNN-QEM validated** (2026-06-06): +99.4% error reduction in-distribution, 100% zero-shot transfer to heavy_hex N=10 (t=13.28, p<10⁻⁶). BUT: does NOT help after PEA-ZNE (0% improvement on post-ZNE residuals). Use only when PEA unavailable. Ref: Wang et al. arXiv:2604.16815.
- **GNN-QEM + PEA are alternatives, not complements**: Both remove structured noise. After one removes structure, residual is unstructured shot noise. Deploy: PEA (primary) → affine (always). GNN-QEM only if PEA unavailable.
- **PEA available as fallback**: If gate-folding ZNE gives R²<0.90 or ΔE/gap>5%, switch `zne_amplifier="pea"`. Learns actual noise model via Pauli-Lindblad fitting, then amplifies probabilistically. ~50% extra QPU overhead. Use `--zne-amplifier pea` in CLI. Ref: IBM Nature 618 (2023).
- **D1 generalizes to frustrated TFIM**: Weight gradient peaks track crossover for all J₂ tested (T1c: 100% agreement).
- **PauliEvolutionGate gives 11% less 2Q-depth**: Use `create_pauli_evolution()` for hardware. Same n_2Q (34), same energy, better scheduling. Level 3 / Rustiq provide no benefit for HVA. Ref: `documentation/analysis/15_transpiler_exploration.md`.

## Unsupervised Phase Detection (Task 2+3 findings)

- **PCA of θ_opt(h) detects h_c for chain_1d**: Peak at h=1.25 (Δh=0.25), PC1 explains 99.96% variance.
- **Detection requires h-grid covering h_c**: Ladder data (h∈[2,4]) cannot detect h_c — data limitation, not method failure.
- **|∂θ/∂h| corroborates D1**: Agreement Δh=0.18 with D1 valid-regime peak (h=1.07). Ref: Fontana et al. (2024, arXiv:2402.18953).
- **K-means NOT recommended**: Boundary at h≈1.58 (too far from h_c). Use PCA or derivative instead.
- **Zero-cost analysis**: All results from existing VQE data — no additional QPU overhead.

## Optimal Config (quick reference)

| System | MPNN | VQE Restarts | Valid Regime (p=2) | Valid Regime (p=1) |
|--------|------|:------------:|:------------------:|:------------------:|
| N=6 | h=64, L=3, 6000ep, lr=1e-3 | 5 (p=2) / 1 (p=1) | h≥1.25 (chain) | h≥1.6 (chain), h≥4.0 (tri) |
| N=10 | **h=128**, L=3, 6000ep, patience=500 | 5 (p=2) / 1 (p=1) | h≥1.5 (chain) | h≥1.9 (chain), h≥3.25 (ladder) |
| N=20 | h=128, MPS chi=64 | 7 (p=2) / 5 (p=1) | h≥2.0 | h≥2.0 (chain) |
| N=40 | h=128, MPS chi=64, COBYLA | 3 (p=1) | — | h≥4.0 (chain, aer_mps) |
| N=50 | h=128, MPS chi=64, COBYLA | 3 (p=1) | — | h≥4.9 (chain, validated) |
| N=80 | h=128, MPS chi=64, COBYLA | 3 (p=1) | — | h≥7.7 (chain, validated) |

- **MPS Scaling**: Use `MPSBackend(strategy="aer_mps")` for N>22. COBYLA optimizer (L-BFGS-B fails with shots).
- **N>63 direct path**: AerSimulator Target has 63 qubits max. For N>63, `save_expectation_value` bypass is used automatically.
- **Scaling law (corrected)**: h_min_safe = 1.5 + 0.020·N^1.31 (original formula +0.50 offset validated at N=40/50/80).
- **Seeds**: All pass at N=40 (27/27). Seed 44 is noisier (max 2.36%) but never fails 5% threshold.
- **Phase 3 MPNN at N=40**: 0.46% mean ΔE/gap with 27 training points. Interpolation only — do not extrapolate to h < h_min.
- **Zero-shot cross-N GNN WORKS with norm_type="none"** (2026-06-08): Train N=40+N=80 (14 pts) → predict N=60: ΔE/gap=0.13% (5/5 PASS). BatchNorm causes 25-40% θ_x underprediction on chain_1d (zero intra-graph variance). Fix: `MPNNPredictor(norm_type="none")`. Interpolation (scipy) also achieves 0.11% but cannot scale to bond-resolved (79D). Ref: `results/scaling/zero_shot/zero_shot_v3_N40_80_to_N60_20260608_110212.json`, `documentation/analysis/19_cross_n_validation_plan.md`.
- **BatchNorm HARMFUL for cross-N on chain_1d**: All nodes identical post-GINConv → BN running_stats capture graph-size artifact, not feature variation. With BN: 18.5% error. Without: 0.13%. Use `norm_type="none"` for cross-N, `norm_type="batch"` for fixed-N (unchanged default).
- **Cross-N VQE warm-start is useless** at 2 params: COBYLA always converges to global min regardless of init (trivial landscape). Warm-start only matters for bond-resolved (79+ params).
- **Noise-aware MPNN training FAILS**: V7 5B showed 6× worse than noiseless. Shot noise corrupts θ_opt targets.
- **χ=64 is validated exact** for HVA p≤2 on 1D TFIM at ANY N (V7 3A/3B: diff=1e-14, actual χ used by DMRG: 9-15).
- **Timing**: T(N) ≈ 0.08·N^2.56 for VQE at boundary h-values. N=80 at h>>h_c is anomalously fast (trivial landscape).
- **Hardware viability** (transpilation audit): N=40: 78 CX (✅ PEA viable), N=50: 98 CX (✅), N=80: 158 CX (⚠️ marginal).

- **Seeds**: Use median of 3 seeds (42/43/44). Seed 43 problematic for ladder, seed 44 for triangular.
- **Hardware deployment (p=1 heavy-hex N=10)**: 1 restart, 3 layouts, 16k shots, h_test≥3.25, SPSA (a=0.1, c=0.05, A=10). Seed-independent (std=0.0003).
- **N=12**: Too slow for iterative experimentation (~30+ min/run). Do not execute.

## ZNE Scaling Rule

- CX budget rule: p=1 N=10 ≈ p=2 N=6 ≈ 18 CX → ZNE works. p=2 N=10 ≈ 36 CX → ZNE fails.
- N=6 p=2: 3 layouts, R²>0.99, gain=+48.5%.
- N=10 p=1: 3 layouts, R²>0.99, gain=+49% (9 runs cross-topology). Heavy-hex: +62.7%.
- **p=1 + ZNE is the recommended strategy for hardware deployment at N≥10.**
- **Amplifier strategy**: PEA (primary) → gate_folding (fallback). CLI: `--zne-amplifier pea`.
- **PEA-ZNE definitive**: +94.4% gain, R²=0.998, 18/18 wins vs GF (t=46.32, p<10⁻¹⁹). Cross-validated on ladder/heavy_hex/chain_1d. Ref: `ZNE_CROSS_TOPO`.
- **GF-ZNE fallback**: +20.6% gain, R²=0.997, always positive. Use if PEA unavailable or `qiskit-aer` not installed.
- **PEA overhead**: ~50% extra QPU time (noise learning phase). Justified by 4.6× better gain vs GF.

## Scaling Law

`h_min = 1.0 + 0.020·N^1.31` (R²=1.0000). Predicts N=20→2.00 (exact match).
- p=1 scales better: β(p=1)=0.60 < β(p=2)=1.33.
- Exponent ≠ ν=1 (expressibility limit, not critical exponent).

## Code Map

### Stable (do NOT modify unless explicitly asked)
- `src/qmbp_simulation/models/` — data models, Hamiltonians, lattices, constants
- `src/qmbp_simulation/solvers/` — exact diag + DMRG
- `src/qmbp_simulation/circuits/` — HVA construction (p≤2 enforced)
- `src/qmbp_simulation/execution/` — backend ABC + noiseless/noisy/hardware + PEA simulation
- `src/qmbp_simulation/optimizers/` — multi-start VQE + SPSA
- `src/qmbp_simulation/pipeline/` — dataset save/load, orchestration
- `src/qmbp_simulation/utils/` — seed, JSON, timing
- `tests/smoke_test.py` — package smoke test (N=4, p=1, <30s)
- `Makefile` — unified entry point

### Active Development
- `src/qmbp_simulation/predictors/mpnn.py` — MPNN architecture
- `src/qmbp_simulation/analysis/` — gradient, diagnostics, metrics, theta_validator, vqe_validator
- `src/qmbp_simulation/framework/` — experiment engine, CLI, benchmarking, logging, preflight, validation args
- `src/qmbp_simulation/pipeline/runner.py` — PipelineRunner
- `experiments/` — categorized experiment scripts
- `scripts/experiment_runners/` — variant runners, pipeline CLIs
- `scripts/experiment_runners/t1_experiments/` — Tier 1 experiments (T1a, T1a_dense)
- `project_health/` — health reports, figures, digest, analysis tools
- `tests/test_project_health_coverage.py` — 72 tests: state, coverage, verify, sanity, scaling, reporter, models
- `scripts/run_hardware_rehearsal.py` — Hardware deployment rehearsal (5 sections)
- `.github/workflows/ci.yml` — CI gate (lint + mypy strict + test + smoke)
- `analysis/` — coverage scanner, diagnostics, verification

### Do NOT Overwrite
- `results/thesis/` — committed definitive results

## References (detailed information lives here)

| Topic | Location |
|-------|----------|
| How To (create experiments, run pipeline, preflight, etc.) | `.kiro/knowledge/project-guide.md` |
| Validated decisions (V7/V8/V9) | `.kiro/knowledge/validated-decisions.md` |
| V8 experiment results | `documentation/binnacles/binnacle-v8-experiments-*.md` |
| V9 Heisenberg results | `documentation/binnacles/binnacle-heisenberg-extension.md` |
| S-series results | `documentation/binnacles/binnacle-s-series-results.md` |
| p=1 scaling results | `documentation/binnacles/binnacle-p1-scaling.md` |
| Thesis tables (5.1–5.21) | `documentation/analysis/09_thesis_tables.md` |
| Key findings (corrected) | `analysis/10_key_findings_corrected.md` |
| Hamiltonian comparison | `documentation/binnacles/binnacle-hamiltonian-comparison.md` |
| Hamiltonian candidates | `documentation/binnacles/binnacle-hamiltonian-candidates.md` |
| Analysis summary | `documentation/analysis/08_summary.md` |
| Experiment framework guide | `.kiro/steering/v8-experiments.md` (conditional: experiments/**) |
| Hardware deployment strategy | `.kiro/steering/hardware-deployment.md` |
| Hardware run checklist | `.kiro/steering/hardware-checklist.md` (manual: #hardware-checklist) |
| Hardware deployment script | `scripts/experiment_runners/hardware/run_ibm_torino_deployment.py` |
| Hardware rehearsal V2 | `scripts/experiment_runners/run_hardware_rehearsal_v2.py` |
| Physics constraints (full) | `.kiro/skills/quantum/SKILL.md` |
| Code style | `.kiro/steering/code-style.md` |
| Error patterns | `.kiro/knowledge/error-patterns.md` |
| Tier 1 session results (2026-06-03) | `documentation/analysis/12_tier1_session_results.md` |
| Hardware rehearsal findings | `documentation/analysis/11_hardware_rehearsal_findings.md` |
| Hardware deployment spec | `HARDWARE_DEPLOYMENT_SPEC.md` |
| Gate-folding ZNE validation | `documentation/binnacles/binnacle-gate-folding-zne.md` |
| ZNE cross-topology (PEA definitive) | `results/experiments/exp_zne_cross_topo/run_20260604_155548.json` |
| Hardware ZNE improvements plan | `documentation/analysis/13_hardware_zne_improvements.md` |
| Thesis figures (21 PDF, vector) | `documentation/thesis_figures/` |
| Figure generation | `make figures` (PNG) or `make figures-thesis` (PDF 300dpi) |
| θ_opt PCA unsupervised detection (Tasks 2+3) | `documentation/binnacles/binnacle-theta-pca-unsupervised-detection.md` |
| Transpiler exploration findings | `documentation/analysis/15_transpiler_exploration.md` |
| Advanced mitigation techniques (2025-2026 lit.) | `documentation/analysis/15_advanced_mitigation_techniques.md` |
| GNN-QEM validation (error correction GNN) | `documentation/binnacles/binnacle-gnn-qem-validation.md` |
| GNN-QEM cross-topology results | `results/gnn_qem/cross_topology_results.json` |
| PEA triangular validation | `results/experiments/exp_pea_triangular/run_20260605_212333.json` |
| Noise suppression gap analysis | `documentation/analysis/16_noise_suppression_analysis.md` |
| MPS scaling plan + results | `documentation/analysis/17_scaling_N30_research_plan.md` |
| MPS scaling binnacle (N=40/50) | `documentation/binnacles/binnacle-mps-scaling.md` |
| MPS scaling analyzer | `python -m project_health.analysis.scaling_analyzer` |
| MPS scaling results | `results/scaling/scaling_N*_*.json` |
| D1 weight-space phase detection | `documentation/binnacles/binnacle-d1-weight-space-phase-detection.md` |
| IBM hardware generations (Eagle→Heron→Nighthawk) | `documentation/analysis/18_ibm_hardware_generations.md` |
| Analysis sanity check | `python -m project_health.analysis.sanity_check` |
| VQE result validation (variational principle, bounds, sweep) | `src/qmbp_simulation/analysis/vqe_validator.py` |
| θ_pred validation module | `src/qmbp_simulation/analysis/theta_validator.py` |
| θ_pred validation tests | `tests/test_theta_validator.py` |
| NLCE module (cluster expansion) | `src/qmbp_simulation/analysis/nlce.py` |
| Scaling extensions runner (E5) | `scripts/experiment_runners/bond_resolved/run_scaling_extensions.py` |
| Scaling extensions analyzer | `python -m project_health.analysis.scaling_extensions_analyzer` |
| Scaling extensions plan | `documentation/analysis/20_scaling_extensions_plan.md` |

## CI & Quality Gates

- **CI workflow**: `.github/workflows/ci.yml` — lint + mypy strict modules + fast tests + smoke test.
- **mypy strict modules**: `framework/criteria.py`, `framework/result_io.py` (0 errors).
- **Make targets**: `make typecheck`, `make coverage`, `make health`, `make figures`, `make figures-thesis`, `make sanity`, `make scaling`, `make extensions`, `make cross-topology`.
- **Preflight**: Mandatory before any variant runner execution (`make preflight SCRIPT=<path>`).

## Pending Execution (Optional)

- `T1a_dense` (J₂-Grid Density Study): Script ready at `scripts/experiment_runners/t1_experiments/run_t1a_dense_j2.py`. ~30 min execution. Run only if thesis needs denser-grid evidence.

## Early-Stopping Rules (from 174 runs diagnosed)

```
PRE-RUN:  Verify h_test ≥ valid_regime_boundary + 0.5
Phase 2:  IF θ_smoothness > 1.0 → WARN (chain break, 45% of failures)
Phase 3:  IF gen_gap > 0.01 → ABORT (MPNN overfit, 25% of failures)
Combined: 69% of failures preventable without losing any passing run.
```

## Failure Mode Summary

| Root Cause | % | Detectable at |
|-----------|---|---------------|
| CHAIN_BREAK (θ>1.0) | 45% | Phase 2 |
| MPNN_OVERFIT (gen_gap>0.01) | 25% | Phase 3 |
| BOUNDARY_EFFECT | 14% | Pre-run (config) |
| OUTSIDE_REGIME | 9% | Pre-run (config) |
| VQE_DIVERGENCE | 7% | Phase 2 |
