# Classical simulability of 2D square frustrated TFIM dynamics

Status: **temporarily closed, 2026-11-16.** Supersedes the 2026-10-02 version, whose
"[WEAK]" items were resolved or re-scoped by later measurements (§3, §8).
Session log: `../binnacles/binnacle-session-20261116-square-dynamics-limits.md`.
Hardware budget: `../hardware/nighthawk_r2_gate_limits_and_noise.md`.
Earlier literature review: `26_bond_dimension_classical_simulability_review.md`.

Evidence tags: **[MEASURED]** = computed by our code on exact states, reproducible
with the commands in §10. **[EXTRAPOLATED]** = fit beyond the largest measured N.
**[LITERATURE]** = external source, not re-verified by us.

All raw JSON lives under `results/experiments/` which is **git-ignored**, so the
numbers below are the durable record. Regenerate with §10 if a number is in doubt.

---

## 1. Summary

- **Ground state is trivial for MPS** [MEASURED]: half-chain entropy 0.7–1.3 nats at
  N ≤ 18, so χ of tens suffices. Area law holds.
- **Dynamics across the transition is where classical cost grows** [MEASURED]: the
  step t* at which a χ-MPS loses the exact state shrinks with N. At N=18 even χ=128
  fails after 13 steps; at N=10 χ=32 never fails.
- **No practical classical limit exists at N ≤ 20** [MEASURED]: the exact statevector is
  ≤ 17 MB, so an exact reference is always available.
- **N=30** [EXTRAPOLATED]: exact statevector 17 GB (cluster, not laptop); MPS needs
  χ ≈ 3×10³ at saturation. Still classically comfortable.
- **N≈50** [EXTRAPOLATED]: exact statevector ≈ 18 PB (impossible); MPS needs χ ≈ 10⁶,
  the practical ceiling. The frontier is N ≈ 50–53.
- **Quantum side** [MEASURED + EXTRAPOLATED]: outlasting a χ=10⁴ MPS at N=18 needs
  ≈ 13.7k transpiled 2q gates, versus ≈ 333 (median CZ, no mitigation) or ≈ 5,000
  (architectural) on Nighthawk r2. Gap ×41 over the noise limit, ×2.7 over the
  architectural limit.
- **No hardware quantum advantage is claimed.** This document bounds the classical side.

---

## 2. Setup and definitions

- Model: `H = −J₁ Σ_nn Z_iZ_j + J₂ Σ_nnn Z_iZ_j − h Σ X_i`, J₁ = 1, J₂ = 0.5.
- Lattice: `ceil(√N)` columns, `ceil(N/cols)` rows (`generate_square`), open boundaries.
- **nnn definition caveat.** `HamiltonianBuilder._generate_nnn_edges` returns *all pairs
  at graph distance 2*, i.e. diagonals **and** straight two-step pairs (N=16: 18
  diagonal + 16 straight). The standard J₁–J₂ model couples diagonals only, so
  comparison with J₁–J₂ literature is approximate. Both kinds are non-native on a
  square coupler graph.
- Quench: prepare the ground state at `h_prep`, evolve under `H(h₂)`.
- **Microstate limit t\*(χ):** first Trotter step (dt = 0.05) where the χ-truncated
  copy of the *exact* evolved state has infidelity > 0.01 against it. The evolved
  state is exact (sparse `expm_multiply`), so this measures the MPS limit with no
  ansatz error. `None`/"never" means no crossing in the simulated window.
- **Macrostate:** averaged observables (M_z, S, DQPT times). They stay classically
  cheap. IBM's 2023 kicked-Ising utility claim was refuted classically on exactly
  this kind of quantity, which is why every claim here targets the microstate.

---

## 3. Measured results

### 3.1 Critical point [MEASURED, N=10]

Minimum gap at h ≈ 0.3, maximum |d²E/dh²| at h ≈ 0.6. The J₁–J₂ literature reports
Γc ≈ 0.28 ([arXiv:1901.00278][arxiv-1901]), consistent but for a slightly different
model (see §2). `h = 0.5` sits on the critical region.

Consequence: the original default quench `h_prep = 0.5 → h₂ high` *leaves* the critical
region. To study a real crossing, prepare at `h = 3.0` (paramagnet) and quench to
`h ≈ 0.3`.

### 3.2 Crossing quench, N = 10–20 (h_prep 3.0 → h₂ 0.3) [MEASURED]

| N  | F_prep | S_sat | t\*(χ=8) | t\*(χ=16) | t\*(χ=32) | t\*(χ=64) | t\*(χ=128) |
|----|:------:|:-----:|:--------:|:---------:|:---------:|:---------:|:----------:|
| 10 | 0.996  | 2.75  |    5     |     9     |   never   |   never   |   never    |
| 12 | 0.995  | 3.14  |    4     |     7     |    21     |   never   |   never    |
| 16 | 0.994  | 4.22  |    3     |     5     |     8     |    13     |     21     |
| 18 | 0.991  | 4.83  |    2     |     4     |     6     |     9     |     13     |
| 20 |   –    | 5.25  |    –     |     –     |     –     |     –     |     –      |

- F_prep is the fidelity of the pruned (prune0.3, `--derive-prep`) HVA ansatz to the
  exact ground state at `h = 3.0`. The crossover itself evolves the exact state.
- N=10 row: from an ad-hoc run (120 steps, χ ≤ 64), not persisted as a JSON. The
  N=12/16/18 rows are in `prune03_dynamics_*.json` (§10). N=20 S_sat only.
- Reading: the classical limit exists and tightens monotonically with N. At N=10 it
  is absent for χ ≥ 32; at N=18 it is present even at χ=128.

### 3.3 Saturation entropy is linear in N [MEASURED]

Half-chain S_sat (max over the quench, nats), five points N=10,12,16,18,20:

- 4-point fit (N ≤ 18): `S_sat = 0.2619·N + 0.068`.
- Held-out check: predicts N=20 as 5.31, measured 5.25 (**1.1 % error**).
- 5-point fit: `S_sat = 0.2580·N + 0.117`, max residual 0.073.
- Crossing slope (0.26) is ~1.8× the slope when *leaving* the critical region (0.14),
  consistent with maximal entanglement growth at the transition
  ([arXiv:1305.6880][arxiv-1305]).
- N=22 and N=24 were started and did not finish (SVD of 2¹¹×2¹¹ per step is too slow
  here). **Not measured.**

### 3.4 Quench from the exact ground state, leaving the critical region [MEASURED]

N=10, h_prep 0.5 → h₂. S_max of the *prepared-state* evolution: 0.88 (h₂=0.5), 1.47
(1.0), 1.70 (1.5), 1.83 (2.0), 1.88 (2.5), 1.94 (3.0); the exact-state evolution gives
slightly lower values (0.77 to 1.91). At h₂=3.0, t\*(χ=4,8,16,32) = 4, 6, 27, never.
N=14, h₂=3.0: prepared-state S_max 2.47, t\*(χ=4,8,16,32) = 0, 0, 7, 15.
Prepared-state fidelity: N=10 F=0.959 (18 2q), N=14 F=0.913 (60 2q). The ansatz has a
fidelity ceiling near h_c (≈ 0.96 at p=2), which is why crossing quenches prepare at
h=3.0 instead.

### 3.5 2D cut versus 1D half-chain [MEASURED, negative result]

Concern: an MPS serpentine over a 2D grid may hide the 2D area law. Measured S_max
along the crossing quench with balanced cuts:

| N  | Grid | S half-chain 1D | S column cut | S row cut | max 2D / 1D |
|----|------|:---------------:|:------------:|:---------:|:-----------:|
| 16 | 4×4  |      4.18       |     4.18     |   4.18    |    1.00     |
| 18 | 4×5  |      4.69       |     4.25     |   4.43    |    0.95     |
| 20 | 4×5  |      5.25       |     4.46     |   5.25    |    1.00     |

The 2D cut never exceeds the 1D cut at these widths (4–5). The half-chain metric used
for every t\* above is therefore not underestimating the cost for N ≤ 20. A width
effect is expected only at much larger widths (iMPS→iPEPS crossover ≈ 11 for
Heisenberg, [arXiv:1705.03222][arxiv-1705]). A first cut at `cols//2` was unbalanced
(8 vs 10 qubits) and gave a spurious ratio < 1; the table uses balanced cuts.

### 3.6 Ground-state entropy [MEASURED, N ≤ 18]

| N  | h=0.3 | h=0.5 | h=2.5 |
|----|:-----:|:-----:|:-----:|
| 10 | 0.70  | 0.77  | 0.22  |
| 12 | 0.70  | 0.79  | 0.24  |
| 14 | 1.13  | 1.12  | 0.25  |
| 16 | 0.70  | 0.73  | 0.24  |
| 18 | 1.07  | 1.34  | 0.31  |

Entropy is 0.2–1.3 nats and shows an odd/even-like jump at N=14 and N=18 (grid-shape
effect). N=20 timed out. Contrast with dynamics: 5–8 nats at saturation.

---

## 4. Projections beyond measured N [EXTRAPOLATED]

Using `S_sat(N) ≈ 0.26·N + 0.1` (§3.3), validated to N=20 only.

| N  | S_sat | e^S      | Exact SV | χ for F≈0.99 (exp-decay model) | Verdict                |
|----|:-----:|:--------:|:--------:|:------------------------------:|------------------------|
| 20 | 5.25  |   190    |  17 MB   | measured regime                | trivial                |
| 30 | 7.9   |  2,600   |  17 GB   | ≈ 10⁴ (F ≥ 0.9999)             | comfortable            |
| 50 | 13.1  | 450,000  |  18 PB   | ≈ 10⁶ (F ≈ 0.99)               | at the practical limit |

- Fidelity for a given χ is the sum of the largest Schmidt weights, which depends on
  the full spectrum, not on S alone. We bracket it with a flat spectrum (pessimistic)
  and an exponential-decay spectrum (typical). **`χ ≈ e^S` is only a loose lower
  heuristic**: measured χ at crossover exceeds it (χ=32 ≫ e^2.75 ≈ 16 at N=10).
- S_sat reaches ln(10⁶) = 13.8 nats at N ≈ 53 (5-point fit). Past that, no practical
  MPS holds the saturated state and the exact state is unavailable.
- The half-chain cap `(N/2)·ln 2` is not binding (S_sat/cap ≈ 0.77 at N=18–30).
- Fitting `t*(χ)` coefficient-by-coefficient from two N values breaks down at N ≥ 24
  (negative t\*). That extrapolation was discarded; only the entropy-based argument is used.

---

## 5. Ground state versus dynamics

| Quantity (N=50 estimate)  | Ground state            | Dynamics, crossing quench |
|---------------------------|-------------------------|---------------------------|
| Entropy scaling           | area law, ∝ grid width  | volume law, grows with t  |
| S at N=50                 | ≈ 2–3 nats [WEAK]       | ≈ 13 nats [EXTRAPOLATED]  |
| χ needed                  | ~10²                    | ~10⁶                      |
| Classical difficulty      | easy to N ≈ 100+        | frontier N ≈ 50–53        |

The cost of a 1D MPS on a 2D ground state depends on the **grid width**, not on N:
N=50 as 7×8 has width 7 and is comfortable; DMRG on 2D is viable to widths ~10–12
([arXiv:1105.1374][arxiv-1105], [arXiv:2206.12764][arxiv-2206]). The N=50 ground-state
figure is a crude fit (≈ 0.55 nats per unit width, from N ≤ 18 data with grid-shape
noise) and is **not validated**. Consequence for the project: any quantum advantage
here would come from dynamics, not from ground-state preparation.

---

## 6. Quantum-side budget for outlasting the classical limit

### 6.1 Routing overhead on the real square coupling map [MEASURED]

`run_native_routing_budget.py`, FakeNighthawk coupling map (120 qubits, 218 coupler
pairs, degrees 4/6/8), native basis `rz, sx, x, cz`, no noise model:

| N  | nn layer (logical→transpiled) | nn + nnn layer (logical→transpiled) | routing excess |
|----|:-----------------------------:|:-----------------------------------:|:--------------:|
| 10 |         13→26 (2.00×)         |            28→87 (3.11×)            |      1.11      |
| 12 |         17→34 (2.00×)         |           39→128 (3.28×)            |      1.28      |
| 14 |         20→40 (2.00×)         |           47→156 (3.32×)            |      1.32      |
| 16 |         24→48 (2.00×)         |           58→205 (3.53×)            |      1.53      |
| 18 |         27→54 (2.00×)         |           66→234 (3.55×)            |      1.55      |

- **nn bonds cost exactly 2.00×** at every N: that is RZZ → 2·CZ, with zero SWAP. nn
  bonds are physical couplers.
- The excess over 2.0 is the SWAP cost of the nnn pairs (diagonal and straight), and
  it grows with N.
- **Correction of an earlier statement.** The previously quoted ×6 (N=21: 160→973)
  was measured on `FakeTorino`, Heron's **heavy-hex** graph (`qd1` Section 4), i.e.
  mapping a square lattice onto the wrong topology. It was not a "generic
  transpilation artefact". Native square gives ×3.5.
- `FakeNighthawk` connectivity is correct. Only its *noise model* is unreliable
  (couplers with error 1.0), which is why noise uses the analytic model instead.

### 6.2 Gate budget at N = 16 and 18 [MEASURED + EXTRAPOLATED]

Steps to outlast a χ=10⁴ MPS use `t*(χ) = a·log₂χ + b` fitted on χ ∈ {8,…,128} and
extrapolated two decades in χ. Per step: 2 half-layers of the nn+nnn layer.

| N  | Steps to outlast χ=10⁴ | 2q logical/step | Overhead | 2q transpiled/step | Total 2q |
|----|:----------------------:|:---------------:|:--------:|:------------------:|:--------:|
| 16 |          ≈ 46          |       116       |   3.53   |        409         | ≈ 19,000 |
| 18 |          ≈ 29          |       132       |   3.55   |        469         | ≈ 13,700 |

Against Nighthawk r2 (`../hardware/nighthawk_r2_gate_limits_and_noise.md`):

- Noise limit, median CZ ε = 3×10⁻³, no mitigation: ≈ 333 2q to F = 1/e → **gap ×41**.
- Architectural limit ≈ 5,000 2q → **gap ×2.7**.
- Required effective error: ε ≈ 7×10⁻⁵, i.e. noise suppression of ×41.
- Larger N shortens the classical horizon (fewer steps) but raises the per-step cost.

Levers: (1) noise suppression (ZNE/QESEM/PEC), dominant; (2) fewer nnn bonds in the
evolution, bounded below by the irreducible 2.0×; (3) N ≥ 16 and a crossing quench.

---

## 7. Hypothesis (conservative)

> In the 2D square frustrated TFIM, the exact microstate of a quench across h_c is
> followed by a 1D MPS of fixed χ only up to a step t\* that shrinks with N (measured
> to N=18). Entanglement growth is linear in time and S_sat is linear in N (measured
> to N=20), so the classical frontier for the saturated state sits near N ≈ 50–53.

Not claimed: hardware advantage; that observables are hard (they are not); that the
GNN-prepared state is the hard case (see §8.1); any N > 20 number as measured.

---

## 8. Open caveats and unfinished work

1. **Contradiction with the project's GNN objective.** `26_bond_dimension_…` frames the
   goal as "a GNN-prepared initial state needs a *larger* χ than a trivial quench from
   |0⟩ᴺ". In the square runs (`qd1` Section 1, h₁ 0.5 → h₂ 2.5) S_max is
   2.72 / 4.00 / 0.88 (N=10) and 4.24 / 7.98 / 1.46 (N=18) for GNN / |0⟩ᴺ / |+⟩ᴺ. The
   GNN state exceeds |+⟩ᴺ but is **below** |0⟩ᴺ. The comparison is confounded by energy
   density (|0⟩ᴺ is a high-energy state). The claim is not supported as stated.
2. **The crossing quench starts from a near-product state.** The ground state at h=3.0
   is close to |+⟩ᴺ, so F_prep ≈ 0.99 is easy and says little about the GNN's value.
   It supports the *quantum-hardware* side (cheap preparation), not the GNN thesis.
3. **S_sat(N) beyond N=20 is extrapolated.** N=22/24 were not completed.
4. **t\*(χ=10⁴) is a log-linear extrapolation** from χ ≤ 128 using two values of N.
5. **Ground-state N=50 entropy is unvalidated.** Needs real DMRG at N=24–36 over
   several widths.
6. **nnn is not standard J₁–J₂** (§2): literature numbers are indicative only.
7. **Trotter versus exact.** The classical side uses exact evolution; the quantum side
   uses Trotter at dt=0.05. Trotter error was not budgeted.
8. **Spectrum shape** for the fidelity-versus-χ table is a model, not measured.

---

## 9. Suggested next experiments

1. DMRG ground state at N=24–36 over several grid widths (closes §8.5).
2. S_sat at N=22/24 on a machine with more time or memory (closes §8.3).
3. Measure t\*(χ) at N=20 with χ up to 256 to shorten the χ extrapolation.
4. Prune nnn in the evolution and measure the physics lost versus 2q saved.
5. Add the analytic noise model to `run_prep_evolution_noise.py` (currently fake only).
6. Repeat the GNN-versus-trivial comparison at matched energy density (closes §8.1).

---

## 10. Reproduction (flags verified with `--help` on 2026-11-16)

Crossing-quench series (this produced §3.2; one run per N):

```bash
.venv/bin/python scripts/analysis/vl_vs_hva/run_prune03_dynamics.py --n 16 --derive-prep --h-prep 3.0 --h2-list 0.3 0.5 1.0 2.5 --dt 0.05 --n-trotter 120 --restarts 6 --maxiter 1500 --chi-values 8 16 32 64 128
```

Native routing budget (§6.1, a minute or less):

```bash
.venv/bin/python scripts/analysis/vl_vs_hva/run_native_routing_budget.py --n-values 10 12 14 16 18 --h2 2.5 --j2 0.5
```

Multi-N initial-state study (`qd1`, §8.1 data):

```bash
.venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py --section 1 4 --topology square --j2 0.5 --dqpt-n-values 10 12 14 16 18 --h1 0.5 --h2 2.5 --dt 0.1 --n-trotter 40 --chi-values 8 16 32 64
```

Notes:

- `--derive-prep` is required for N ≠ 10 (the stored artefact is N=10 specific and the
  runner raises a clear error otherwise). It optimizes two variants, so N=16–18 takes
  hours.
- Evolution uses sparse `expm_multiply`; dense `expm` ran out of memory at N=18.
- Outputs: `results/experiments/exp_frustrated/prune03_dynamics/tfim_frustrated/square_N{N}_hprep{h}/`,
  `.../qd1/tfim/square_p1/`, `.../native_routing_budget/` (all git-ignored).
- Helpers: `qmbp_simulation.analysis.dynamics`, `...analysis.observables`
  (`bipartition_entropy`, `square_vertical_cut_qubits`), `...circuits.trotter`.
- Tests: `tests/unit/test_dynamics.py` (34 tests).

---

## 11. Literature

- Entanglement barrier and exponential χ in time: [arXiv:0801.2078][arxiv-0801],
  [arXiv:1911.09108][arxiv-1911].
- Entanglement growth peaks at the transition: [arXiv:1305.6880][arxiv-1305].
- Microstate versus macrostate tractability: [arXiv:2310.08567][arxiv-2310].
- IBM kicked-Ising reproduced classically: [arXiv:2306.14887][arxiv-2306],
  [arXiv:2309.15642][arxiv-2309].
- 2D MPS cost and area law: [arXiv:1105.1374][arxiv-1105],
  [arXiv:1705.03222][arxiv-1705], [arXiv:2601.08132][arxiv-2601].

<!-- Links -->
[arxiv-0801]: https://arxiv.org/html/0801.2078v2
[arxiv-1105]: https://arxiv.org/html/1105.1374v2
[arxiv-1305]: https://arxiv.org/pdf/1305.6880v1.pdf
[arxiv-1705]: https://arxiv.org/html/1705.03222v1
[arxiv-1901]: https://arxiv.org/html/1901.00278v1
[arxiv-1911]: https://arxiv.org/html/1911.09108v2
[arxiv-2206]: https://arxiv.org/html/2206.12764v2
[arxiv-2306]: https://arxiv.org/html/2306.14887v1
[arxiv-2309]: https://arxiv.org/html/2309.15642v3
[arxiv-2310]: https://arxiv.org/html/2310.08567v3
[arxiv-2601]: https://arxiv.org/html/2601.08132v1
