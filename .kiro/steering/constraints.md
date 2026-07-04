# Project Constraints (ALWAYS ENFORCE)

## Physics & Circuit Constraints

- HVA only, never HEA. p ≤ 2 layers. |+⟩^N initial state (TFIM). Néel state (Heisenberg).
- Descending sweep h_max→h_min. No angle wrapping. Pure energy cost in Phase 2.
- SparsePauliOp only. Primitives V2 only. Local observables on hardware.
- Fidelity filter ≥ 0.93 (TFIM), ≥ 0.60 (Heisenberg) in Phase 3 training data.
- Hardware success: ΔE/gap < 5% AND correct phase label (not fidelity).
- **Heisenberg HVA p≤2 CANNOT work** — do not attempt.
- **Kitaev chain NOT viable** — do not implement.

## ZNE & Mitigation

- ZNE CX threshold: ~18 CX for gate-folding. PEA handles up to ~50 CX.
- p=1 for N≥10 hardware (18 CX within ZNE regime). p=2 N=10 = 36 CX → fails.
- PEA-ZNE is primary strategy (+94.4%). Gate-folding is fallback (+20.6%).
- CES-ZNE DEPRECATED on heavy_hex (uniform CES, R²≈0.04).
- GNN-QEM + PEA are alternatives, NOT complements (both remove structured noise).
- Affine correction: always apply after ZNE (zero cost). Simple clip to [E_ground, E_upper]. NEVER use soft interpolation formulas — they amplify errors (bug fixed 2026-06-22, see validated-decisions.md).
- **CES spread guard**: min_ces_spread=0.02 in HardwareConfig. Escalates 3→5 layouts automatically if spread insufficient.
- **WLS ZNE**: Weighted least squares with σ_i ∝ √(nf_i). Active by default in PEA and GF-ZNE.
- **Multi-layout observables**: hardware mode submits on ALL layouts (not just first), averages for √n reduction.
- **Outlier detection requires n_layouts≥4** (Grubbs test). With n_layouts=3 (default), only log outliers — do NOT remove them from averaging.
- **Adaptive shot budget is second-order**: inter-layout variance (0.25) >> intra-layout variance (0.16). Do NOT prioritize shot redistribution over layout/mitigation selection.
- **Stale calibration comparison**: diagnostic only (never abort). Useful for runs >1h; overkill for 30 min.

## Scaling

- Scaling law: `h_min_safe = 1.5 + 0.020·N^1.31` (validated N=40-200).
- χ=64 sufficient for HVA p≤2 on 1D TFIM at ANY N.
- MPS deterministic mode default (exact, 12ms/eval). Stochastic only for noise testing.

## Experiment Discipline

1. State the hypothesis being tested. No hypothesis → no execution.
2. Check binnacles and poc-results.md — if the result is already established, do NOT re-run.
3. Every run must produce new learning. "Confirming what we know" is not learning after 3 seeds.
4. Do not duplicate information across binnacles — reference existing entries.

## Code Stability

### Do NOT modify without explicit ask:
- `src/qmbp_simulation/models/` — data models, Hamiltonians, lattices
- `src/qmbp_simulation/solvers/` — exact diag + DMRG
- `src/qmbp_simulation/circuits/` — HVA construction
- `src/qmbp_simulation/execution/` — backend ABC + implementations
- `src/qmbp_simulation/optimizers/` — multi-start VQE + SPSA
- `src/qmbp_simulation/pipeline/` — dataset save/load, orchestration
- `src/qmbp_simulation/utils/` — seed, JSON, timing
- `results/thesis/` — committed definitive results

**Always** Import and reuse the already written code. Don't copy paste duplicate or repeat code. I want you import and reuse the already written code.

## Seeds & Reproducibility

- Default seeds: 42, 43, 44. Seed 43 problematic for ladder, seed 44 for triangular.
- Hardware deployment (p=1 heavy-hex N=10): 1 restart, 3 layouts, 16k shots, h_test≥3.25.
- **SWAP-free layout on ibm_kingston**: [22,23,24,25,26,27,28,16,37,17] — verified 9 CZ only. Use as VF2 fallback.
- **NEVER use BFS for heavy_hex N=10 on Kingston** — produces 34-44 CZ (catastrophic routing). VF2 or fallback_layout mandatory.
