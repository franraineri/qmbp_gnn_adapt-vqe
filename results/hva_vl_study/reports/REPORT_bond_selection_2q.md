# Bond-selection & circuit compression — fewer 2q gates on the frustrated HVA

How far the 2-qubit-gate count of a bond-resolved frustrated HVA ansatz can be
reduced while keeping the ground-state fidelity, on the **frustrated
square-lattice TFIM** (`tfim_frustrated`, `J2 = 0.5`, `J = 1.0`). Two axes:
the ansatz **structure** and the **bond selection** on top of it.

- Model: `tfim_frustrated`, `J2 = 0.5`, `J = 1.0`; lattice `square`.
- Reference regime: `h = 0.5` (near h_c, gap ≈ 0.006 — the hard regime).
- Optimizer: multi-restart L-BFGS-B best-of-by-energy, noiseless statevector
  backend; warm-start via the mandatory `prepare_warmstart` cascade (calibrated
  + regime + transferred donors + micro-descent). Reported fidelities are the
  ansatz expressivity ceiling.
- Fidelity: `F = |⟨ψ_exact | ψ_HVA(θ*)⟩|²`. Gate counts transpiled to
  `rz, sx, x, cx`.
- Source artifacts: `results/hva_vl_study/bond_ablation/*.json`.

## The winning structure: `p1_half_nn_rx`

The best fidelity-per-2q structure is **not** raw depth. `p1_half_nn_rx` (one
frustrated layer + a half-layer of nn-only RZZ + a trailing RX block) recovers
p=2 expressivity from a p=1-cost base, because the extra nn half-layer adds
balanced entanglement and the final RX adds free rotations at **0 CX**.

- Registry: `src/qmbp_simulation/circuits/hva_variants.py` → `VARIANTS["p1_half_nn_rx"]`
- `blocks = ["nn","nnn","x","nn","x"]`, `rx_final = True`

## Compression frontier — N = 10, h = 0.5

Each row is the most relevant point of its selection technique (ranked from the
converged full θ). All share the same structure and warm-start; only the kept
bonds differ.

| Variant                       | nn / nnn | 2q (CX) | Fidelity | fid/CX  | note                     |
|-------------------------------|:--------:|:-------:|:--------:|:-------:|--------------------------|
| p1_half_nn_rx (full)          | 13 / 15  |   82    |  0.9892  | 1.21e-2 | structure ceiling        |
| T1 prune (tol 0.1)            | 10 / 4   |   48    |  0.9843  | 2.05e-2 | **sweet spot** (−41% 2q) |
| T1 prune (tol 0.15)           |  8 / 4   |   40    |  0.9792  | 2.45e-2 | min useful (F ≥ 0.98)    |
| T1 prune (tol 0.3)            |  7 / 4   |   36    |  0.9598  | 2.67e-2 | aggressive floor         |

For comparison, the standard `p2_base` full reference is `0.9890 @ 112 2q`
(fid/CX 8.8e-3): `p1_half_nn_rx` + prune reaches the same fidelity class at
**less than half the 2q cost**.

### Collapse walls (hard limits)

Two independent, abrupt failure points bound the pruning:

- **nnn ≥ 3 bonds.** Dropping to 2 nnn collapses fidelity (0.977 → 0.686).
- **nn ≥ 7 backbone bonds.** nn 10 → 0.984, nn 8 → 0.979, nn 7 → 0.960,
  nn 5 → 0.555 (total collapse). The nn backbone is the dominant structure;
  below 7 the state is destroyed.

The practical floor without collapse is **40 2q @ 0.979** (prune 0.15). Below
36 2q both walls trigger.

## Circuit compression with our own tools — what works

Findings on reducing 2q with the in-repo tooling (no external optimizers):

1. **Bond selection is the only lever that reduces 2q here.** Every CX comes
   from an RZZ, so dropping bonds is the direct knob. `top_k_by_weight` (T2) and
   `prune_by_theta` (T1) both work; **T1 prune is preferred** — one tolerance
   selects which bonds survive by `|θ|`, automatically respecting both collapse
   walls (it keeps ~4 nnn and thins nn only as far as safe), whereas a fixed
   top-k count can be set below a wall by mistake.

2. **Transpilation does NOT compress.** `transpile(optimization_level = 2 or 3)`
   leaves the 2q count unchanged (82→82, 48→48). The bond-resolved HVA is
   already in minimal-CX form; the transpiler finds no cancellations. Do not
   expect opt-level to buy gates.

3. **AQC-Tensor compression is dominated by direct pruning here.** Re-fitting a
   converged circuit to a shallower p=1 ansatz via MPS
   (`circuits/aqc_compression.py`, best-of-3) always *lost* fidelity vs the
   pruned original at the same 2q:

   | input (pruned)        | pruned F @ 2q | AQC F @ 2q   | verdict               |
   |-----------------------|:-------------:|:------------:|-----------------------|
   | prune 0.1 (nn10/nnn4) | 0.9843 @ 48   | 0.9642 @ 39  | AQC −0.020 fid        |
   | prune 0.15 (nn8/nnn4) | 0.9792 @ 40   | 0.9596 @ 39  | AQC −0.020 fid        |
   | prune 0.3 (nn7/nnn4)  | 0.9598 @ 36   | 0.9557 @ 39  | AQC **enlarges** +3 2q|

   AQC converges to a fixed ~39-2q p=1 ansatz regardless of input, so for an
   already-small circuit (36 2q) it *adds* gates. Its high internal `aqc_fid`
   (0.986–0.997, compressed-vs-target) is misleading: the pruned target is no
   longer the exact ground state, and AQC cannot recover what pruning removed.
   **Prefer direct T1 pruning; keep AQC only for compressing a dense, unpruned
   high-depth circuit toward a ZNE budget.**

4. **More optimizer budget does not raise fidelity.** Re-running the pruned
   ansätze at maxiter 2000 (vs 1000), restarts 3, micro-descent 120 changed
   fidelity by ≤ 0.0006 — noise. These circuits are already converged; the
   ceiling is ansatz expressivity, not the optimization budget.

5. **Warm-start is mandatory and homogeneous.** Every row — reference and every
   masked variant — is converged through `prepare_warmstart`, so comparisons are
   on equal footing (`seed_kind` recorded per row). The full reference seeds
   every masked transfer, so a well-converged full matters.

## Scaling to N = 18, h = 0.5 — the nnn-layer curve and the Pareto frontier

At N = 18 (the extrapolation regime, gap ≈ 0.006) the picture holds but sharpens.
Two questions were settled here: (a) how many nnn layers a p3 ansatz actually
needs, and (b) whether encoding fewer nnn *natively* (an "nn-dominant" structure)
beats the full `half_nn_rx` + top-k prune recipe.

### The nnn-layer curve (p3, nn backbone fixed at 4 layers)

Holding the nn backbone at 4 layers and varying only the number of nnn layers
maps the marginal value of nnn directly:

| Variant        | nnn layers | 2q (CX) | Fidelity | Δfid vs prev |
|----------------|:----------:|:-------:|:--------:|:------------:|
| p3_nn_dom_rx   |     1      |   294   |  0.8385  |      —       |
| p3_nn_dom2_rx  |     2      |   372   |  0.9252  |   **+0.087** |
| p3_half_nn_rx  |     3      |   450   |  0.9516  |    +0.026    |

The return on nnn layers is sharply decreasing, with a **critical 1 → 2 jump**:

- **1 nnn layer is a structural ceiling** at 0.84 (both restarts identical — not
  an optimizer artefact). The p2 analogue `p2_nn_dom_rx` (nnn = 1 at p2) tops out
  even lower at 0.7785, consistent with the "nn-only plateau 0.73–0.77" seen at
  small N.
- **2 nnn layers capture 97%** of the full-ansatz fidelity (0.9252 vs 0.9516)
  with 78 fewer gates.
- A 3rd nnn layer adds only +0.026 for +78 gates.

**nnn is structurally necessary near h_c**, and the minimum useful count is ~2
layers — "one nnn layer suffices" is refuted.

### The nn-dominant design is Pareto-dominated

The nn-dominant variants were built to encode *fewer* nnn natively. They lose to
simply pruning the full `half_nn_rx`: `p3_nn_dom2_rx` (0.9252 @ 372) is beaten by
`p3_half_nn_rx_topk20` (0.9359 @ 336) — **higher fidelity AND fewer gates**.
Reducing nnn in the ansatz design is strictly worse than keeping the full nn +
nnn structure and letting top-k drop the low-`|θ|` nnn bonds. This is the same
"full nn + prune nnn > structurally reduce" pattern as at N = 10.

### Consolidated Pareto frontier (N = 18, h = 0.5)

Every point is the best fidelity of its variant; a point is on the frontier when
no other point has both ≥ fidelity and ≤ 2q. **All frontier points are
`half_nn_rx`** — no `nn_dom` or `base` variant survives.

| 2q (CX) | Fidelity | Variant                   | note                     |
|:-------:|:--------:|---------------------------|--------------------------|
|   128   |  0.8202  | p2_half_nn_rx_topk8       | cheap, low-fidelity      |
|   148   |  0.8483  | p2_half_nn_rx_topk13      |                          |
|   276   |  0.8903  | p3_half_nn_rx_topk10      |                          |
|   318   |  0.9084  | p2_half_nn_rx_full        | **cheapest ≥ 0.90**      |
|   336   |  0.9359  | p3_half_nn_rx_topk20      | **best efficiency** (−25% vs full) |
|   390   |  0.9515  | p3_half_nn_rx_topk29      | near-full at −60 2q      |
|   450   |  0.9516  | p3_half_nn_rx_full        | fidelity ceiling         |

Cheapest point above 0.90 is `p2_half_nn_rx_full` at **318 2q**; below that the
0.90 line is not crossed. The best high-fidelity efficiency is
`p3_half_nn_rx_topk20` at **336 2q → 0.9359** (114 fewer gates than the full for
−0.016 fidelity).

Source artifacts: `results/hva_vl_study/bond_ablation/variant_topk_*_N18_h0.50.json`.
Variants in `src/qmbp_simulation/circuits/hva_variants.py`
(`p3_nn_dom_rx`, `p3_nn_dom2_rx`, `p2_nn_dom_rx`).

## Bond importance is collective, not transferable

A natural idea for scaling the selection is to *predict* which bonds matter at
large N from a cheap property, instead of converging a full θ first. We tested
this directly on the converged full specs (pure analysis, no extra circuits) and
it does not hold — the per-bond importance (`|θ|`) is a collective,
gauge-dependent quantity, not a stable local one.

1. **Geometry does not predict importance.** Rank-correlation (Spearman) of
   each nnn bond's `|θ|` against local geometric features — frustration-triangle
   count, Euclidean length, midpoint-to-center distance, diagonal orientation,
   common-neighbour and min-degree — averages `|ρ| ≤ 0.34` with inconsistent
   sign across specs. A multivariate fit looks strong only at N=8 (R² ~0.9) but
   that is overfitting: with 11 nnn bonds and 5 features the model is
   underdetermined; at N=10/14 the R² collapses to 0.16–0.66. No geometric
   combination generalizes.

2. **Importance is not stable across depth.** The nnn `|θ|` ranking of the same
   lattice at p=1/2/3 jumps erratically (Spearman from +0.92 to −0.48 between
   adjacent depths; top-half Jaccard ≈ 0.33, i.e. chance). Adding a layer
   reshuffles which bonds carry weight.

3. **Importance is not conserved across N.** Restricting to the nnn edges shared
   between two sizes, the ranking anti-correlates (N8↔N10 ρ = −0.80, N10↔N14
   ρ = −0.60). A bond that is important at a small N tends to be *less* important
   at a larger N.

**Consequence for warm-starting.** The cross-N donor does **not** work by
preserving per-bond importance (it is not preserved). It works because it
transfers a *globally coherent* θ configuration that lands in a good
optimization basin, which the micro-descent then refines — the value is the
whole vector as a starting point, not any single bond. This is exactly why the
donor is injected as a *competing* candidate (it does not force angles) and why a
structural seed of the form "put large angles on geometrically special bonds"
would not help. Selection still has to come from a converged θ, which is what the
runner's `prepare_warmstart` cascade provides.

Tools (pure analysis over `ansatz_specs/`): `analyze_bond_importance.py`
(geometry baseline), `analyze_bond_importance_deep.py` (richer geometry +
depth / cross-N persistence).

## Reproduction

```bash
# Full frontier (structure + T1 prune sweep), mandatory warm-start:
.venv/bin/python scripts/analysis/vl_vs_hva/run_variant_topk.py \
    --n 10 --h 0.5 --variant p1_half_nn_rx \
    --prune-tols 0.1 0.15 0.3 --restarts 3 --maxiter 1000 --micro-descent 60

# N=18 nnn-layer curve (p3 nn-dominant 1 vs 2 nnn layers vs full half_nn):
.venv/bin/python scripts/analysis/vl_vs_hva/run_variant_topk.py \
    --n 18 --h 0.5 --variant p3_nn_dom2_rx \
    --keep-fracs 0.5 --restarts 2 --maxiter 800 --micro-descent 60 --no-sync

# N=18 top-k frontier (the Pareto-winning recipe):
.venv/bin/python scripts/analysis/vl_vs_hva/run_variant_topk.py \
    --n 18 --h 0.5 --variant p3_half_nn_rx \
    --keep-fracs 0.75 0.5 --restarts 2 --maxiter 800 --micro-descent 60 --no-sync

# Bond-importance study (pure analysis — reads converged AnsatzSpecs):
.venv/bin/python scripts/analysis/vl_vs_hva/analyze_bond_importance_deep.py \
    --topology square --h 0.5

# Compression shootout (baseline / opt2 / opt3 / AQC best-of-3) on a pruned circuit:
.venv/bin/python scripts/analysis/vl_vs_hva/run_compression_shootout.py \
    --n 10 --h 0.5 --variant p1_half_nn_rx --prune-tol 0.15 \
    --restarts 3 --aqc-restarts 3 --aqc-iters 300
```

## Recommendations

**N = 10 (small):**

- **Max fidelity, minimal loss:** `p1_half_nn_rx` full — 0.9892 @ 82 2q.
- **Sweet spot (general):** T1 prune tol 0.1 — **0.9843 @ 48 2q** (−57% vs p2_base).
- **Aggressive floor:** T1 prune tol 0.15 — 0.9792 @ 40 2q (last point ≥ 0.98).

**N = 18 (scaling regime):**

- **Max fidelity:** `p3_half_nn_rx` full — 0.9516 @ 450 2q.
- **Best efficiency:** `p3_half_nn_rx_topk20` — **0.9359 @ 336 2q** (−25% vs full).
- **Cheapest ≥ 0.90:** `p2_half_nn_rx_full` — 0.9084 @ 318 2q.
- **nnn layers:** 2 is the structural minimum near h_c (97% of full fidelity);
  1 collapses to ~0.84.

**General:**

- **Do not** use transpile opt-levels or AQC to chase fewer 2q here; direct T1
  pruning is strictly better.
- **Do not** try to pick bonds from geometry or a small-N importance map — bond
  importance is collective and does not transfer; select from a converged θ.
- **Do not** encode fewer nnn natively (nn-dominant designs); keep the full
  `half_nn_rx` structure and prune nnn by top-k — it is Pareto-superior.
