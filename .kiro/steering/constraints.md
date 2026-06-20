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
- Affine correction: always apply after ZNE (zero cost, 0% overshoot in 102 records).

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

## Seeds & Reproducibility

- Default seeds: 42, 43, 44. Seed 43 problematic for ladder, seed 44 for triangular.
- Hardware deployment (p=1 heavy-hex N=10): 1 restart, 3 layouts, 16k shots, h_test≥3.25.
