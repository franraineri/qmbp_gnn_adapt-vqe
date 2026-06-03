# Project Status — GNN-HVA Framework

**Last updated**: 2026-06-03

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
- Total useful-outcome rate: 90% (19/21 formal experiments → 24/29 with Tier 1).
- 210+ pipeline runs executed across 5 topologies (chain_1d, ladder, triangular, kagome, heavy-hex).

## Active Priority

1. **Hardware deployment on IBM Torino** — local simulation exhausted, hardware is the only remaining validation.
2. **Thesis writing** — Chapter 5 compilation from `documentation/analysis/09_thesis_tables.md`.

## Key Constraints (always enforce)

> Full list with rationale: `.kiro/skills/quantum/SKILL.md`

- HVA only, never HEA. p ≤ 2 layers. |+⟩^N initial state (TFIM). Néel state (Heisenberg).
- Descending sweep h_max→h_min. No angle wrapping. Pure energy cost in Phase 2.
- SparsePauliOp only. Primitives V2 only. Local observables on hardware.
- Fidelity filter ≥ 0.93 (TFIM), ≥ 0.60 (Heisenberg) in Phase 3 training data.
- Hardware success: ΔE/gap < 5% AND correct phase label (not fidelity).
- **Heisenberg HVA p≤2 CANNOT work** — do not attempt (V9: 30 runs + N=10/16 scaling confirm).
- **Kitaev chain NOT viable** — 20 CZ@N=6 (exceeds ZNE), fid=16% max. Do not implement.
- **TFIM+longitudinal WORKS** — fid≥0.98 at g=0.5, 0 extra CX gates (E4b validated).
- **TFIM frustrated (J1-J2) WORKS in simulation** — fid≥0.99 at J₂=0.5, but 27 CZ@N=6 (no ZNE for N≥6).
- **ZNE threshold**: ~18 CX gates. p=2 N=10 (36 CX) fails. Use p=1 for N≥10 hardware.
- **CES-ZNE fails on heavy_hex**: All good layouts have CES≈0.15 (no spread). Use IBM gate-folding ZNE instead. Ref: `documentation/analysis/11_hardware_rehearsal_findings.md`.
- **D1 generalizes to frustrated TFIM**: Weight gradient peaks track crossover for all J₂ tested (T1c: 100% agreement).

## Optimal Config (quick reference)

| System | MPNN | VQE Restarts | Valid Regime (p=2) | Valid Regime (p=1) |
|--------|------|:------------:|:------------------:|:------------------:|
| N=6 | h=64, L=3, 6000ep, lr=1e-3 | 5 (p=2) / 1 (p=1) | h≥1.25 (chain) | h≥1.6 (chain), h≥4.0 (tri) |
| N=10 | **h=128**, L=3, 6000ep, patience=500 | 5 (p=2) / 1 (p=1) | h≥1.5 (chain) | h≥1.9 (chain), h≥3.25 (ladder) |
| N=20 | h=128, MPS chi=64 | 7 (p=2) / 5 (p=1) | h≥2.0 | h≥2.25 (chain) |

- **Seeds**: Use median of 3 seeds (42/43/44). Seed 43 problematic for ladder, seed 44 for triangular.
- **Hardware deployment (p=1 heavy-hex N=10)**: 1 restart, 3 layouts, 16k shots, h_test≥3.25, SPSA (a=0.1, c=0.05, A=10). Seed-independent (std=0.0003).
- **N=12**: Too slow for iterative experimentation (~30+ min/run). Do not execute.

## ZNE Scaling Rule

- CX budget rule: p=1 N=10 ≈ p=2 N=6 ≈ 18 CX → ZNE works. p=2 N=10 ≈ 36 CX → ZNE fails.
- N=6 p=2: 3 layouts, R²>0.99, gain=+48.5%.
- N=10 p=1: 3 layouts, R²>0.99, gain=+49% (9 runs cross-topology). Heavy-hex: +62.7%.
- **p=1 + ZNE is the recommended strategy for hardware deployment at N≥10.**

## Scaling Law

`h_min = 1.0 + 0.020·N^1.31` (R²=1.0000). Predicts N=20→2.00 (exact match).
- p=1 scales better: β(p=1)=0.60 < β(p=2)=1.33.
- Exponent ≠ ν=1 (expressibility limit, not critical exponent).

## Code Map

### Stable (do NOT modify unless explicitly asked)
- `src/qmbp_simulation/models/` — data models, Hamiltonians, lattices, constants
- `src/qmbp_simulation/solvers/` — exact diag + DMRG
- `src/qmbp_simulation/circuits/` — HVA construction (p≤2 enforced)
- `src/qmbp_simulation/execution/` — backend ABC + noiseless/noisy/hardware
- `src/qmbp_simulation/optimizers/` — multi-start VQE + SPSA
- `src/qmbp_simulation/pipeline/` — dataset save/load, orchestration
- `src/qmbp_simulation/utils/` — seed, JSON, timing
- `scripts/smoke_test.py` — package smoke test (N=4, p=1, <30s)
- `Makefile` — unified entry point

### Active Development
- `src/qmbp_simulation/predictors/mpnn.py` — MPNN architecture
- `src/qmbp_simulation/analysis/` — gradient, diagnostics, metrics
- `src/qmbp_simulation/framework/` — experiment engine, CLI, benchmarking, logging, preflight
- `src/qmbp_simulation/pipeline/runner.py` — PipelineRunner
- `experiments/` — categorized experiment scripts
- `scripts/experiment_runners/` — variant runners, pipeline CLIs
- `scripts/digest/` — result digest tool
- `scripts/run_t1a_mpnn_2d_predictor.py` — Tier 1A: 2D MPNN predictor (h × J₂)
- `scripts/run_t1b_longitudinal_zne.py` — Tier 1B: ZNE for TFIM+longitudinal
- `scripts/run_t1c_d1_frustrated.py` — Tier 1C: D1 weight-space for frustrated TFIM
- `scripts/run_hardware_rehearsal.py` — Hardware deployment rehearsal (5 sections)
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
| Physics constraints (full) | `.kiro/skills/quantum/SKILL.md` |
| Code style | `.kiro/steering/code-style.md` |
| Error patterns | `.kiro/knowledge/error-patterns.md` |
| Tier 1 session results (2026-06-03) | `documentation/analysis/12_tier1_session_results.md` |
| Hardware rehearsal findings | `documentation/analysis/11_hardware_rehearsal_findings.md` |
| Hardware deployment spec | `HARDWARE_DEPLOYMENT_SPEC.md` |

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
