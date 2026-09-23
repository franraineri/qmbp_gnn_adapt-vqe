# Bond Dimension as the Classical Simulability Frontier — Literature Review (2026-09-01)

> External-literature review supporting the study objective:
> **"The GNN-prepared initial state shifts the entanglement barrier, requiring a
> larger χ than a trivial quench from |0⟩^N."**
>
> Answers: does published work establish χ-growth as the classical-simulation
> limit, and how does it constrain our thesis claim? Content rephrased for
> compliance with source licensing restrictions.

---

## Thesis objective this review supports

The claim under test is NOT "MPS cannot vs QPU can" (that framing is refuted by
the literature below). It is the sharper, defensible statement:

> A quench from a **non-trivial, correlated initial state** |ψ₀(h₁)⟩ (prepared by
> the GNN-HVA pipeline) drives entanglement growth faster than a quench from a
> product state |0⟩^N or |+⟩^N, so the bond dimension χ needed to keep the MPS
> accurate grows faster — pushing the classical-simulation frontier to earlier
> times / smaller sizes.

---

## Two regimes (established physics)

| Regime | Entanglement | χ scaling | Classically viable? |
|--------|--------------|-----------|---------------------|
| Ground state (area-law, 1D gapped) | S ∝ log(ξ), bounded | χ ~ const in N | Yes — our own N=40–200 results confirm χ=64 exact |
| Post-quench dynamics (volume-law) | S(t) grows linearly, then saturates | χ ~ e^{S(t)} → exponential in t | No — hard time-horizon limit |

Because χ ~ e^S, a linear-in-time entropy implies an exponential-in-time bond
dimension. This is the "entanglement barrier".

---

## Key references

### The entanglement barrier (core mechanism)

- **Overcoming the entanglement barrier with sampled tensor networks** — arXiv:2505.09714.
  States the exact bottleneck: rapid entanglement growth under unitary evolution
  is the primary limit of MPS for time-dependent expectation values, and this
  barrier both restricts classical simulation and underpins the anticipated
  quantum advantage. <https://arxiv.org/abs/2505.09714>
- **MPS out-of-equilibrium decomposition** — arXiv:1911.09108. Out-of-equilibrium
  states require a bond dimension that grows exponentially, a hard limit on
  simulable timescales. <https://arxiv.org/pdf/1911.09108>

### Linear entanglement growth after a quench (why χ explodes)

- **Calabrese & Cardy, Quantum quenches in 1+1D CFT** — arXiv:1603.02889.
  Canonical framework: S(t) grows linearly then saturates after a global quench.
  <https://arxiv.org/abs/1603.02889>
- **Entanglement growth with variable-range interactions** — arXiv:1305.6880.
  In the transverse Ising chain, the maximum entanglement-growth rate occurs when
  Hamiltonian parameters match the quantum phase transition. **Directly justifies
  crossing h_c in our quench.** <https://arxiv.org/abs/1305.6880>
- **XY chain analytic results** — arXiv:0804.3559. Explicit linear-growth-then-
  saturation; usable as an exact validation reference. <https://arxiv.org/abs/0804.3559>

### IBM kicked-Ising was simulated classically (reframes our IBM diagram)

The IBM quantum-advantage experiment (Nature 618, 2023) was matched or beaten by
tensor-network methods. A trivial |0⟩^N + Floquet protocol is simulable at
moderate χ — which is exactly why our non-trivial-preparation angle matters.

- **Tindall et al., Efficient TN simulation of IBM's Eagle kicked Ising** — arXiv:2306.14887
  (Flatiron): 127-qubit heavy-hex simulated more accurately than the QPU via belief
  propagation. <https://arxiv.org/abs/2306.14887>
- **gPEPS simulation of IBM's largest processors** — arXiv:2309.15642. <https://arxiv.org/abs/2309.15642>
- **PEPO (Heisenberg picture) simulation** — arXiv:2308.03082. <https://arxiv.org/abs/2308.03082>
- **Quantum Advantage: a Tensor Network Perspective** — arXiv:2603.18825. 2026 review
  of IBM/D-Wave/Google cases where classical TN reached or exceeded hardware.
  <https://arxiv.org/abs/2603.18825>
- **Fast classical simulation of Fermi-Hubbard hardware run** — arXiv:2608.13805.
  χ=32 already reproduces a recent hardware experiment and extends it to longer
  times. <https://arxiv.org/abs/2608.13805>

### χ-vs-precision frontier (concrete numbers)

- **Practical Scalability of TN Quantum Emulators** — arXiv:2504.11399. At fixed χ,
  scaling is favorable, but the χ needed to hit chemical accuracy grows rapidly
  with active-space size — the analogue of our "χ needed for ΔE ≤ tol" metric.
  <https://arxiv.org/abs/2504.11399>
- **MPS/TTNS cost scaling under area-law** — arXiv:2601.08132. Confirms Panel A:
  under area-law, χ scales exponentially in subsystem surface (1D → constant).
  <https://arxiv.org/abs/2601.08132>

### GNN/NN state preparation (competing methods — positioning)

- **Improving Generalization of VQE via Graph Neural Encoding** — arXiv:2602.19752.
  <https://arxiv.org/abs/2602.19752>
- **GNN for Fast Operator Selection in Adaptive VQE** — arXiv:2606.08794.
  <https://arxiv.org/abs/2606.08794>
- **NN-encoded VQE** — arXiv:2308.01068. <https://arxiv.org/abs/2308.01068>

---

## Verdict

| Question | Answer |
|----------|--------|
| Literature on χ as the classical limit? | Yes — established (entanglement barrier; Calabrese–Cardy) |
| Is χ-growth worst near h_c? | Yes (arXiv:1305.6880) → justifies crossing h_c |
| Is IBM kicked-Ising a good advantage example? | No — classically simulated (arXiv:2306.14887 et al.) |
| Is our differentiator (non-trivial GNN initial state) supported? | Yes — a correlated initial state raises entanglement growth vs |0⟩^N |

**Framing for the data study:** compare χ_needed(t) for |ψ₀(h₁)⟩ (GNN) vs |0⟩^N
vs |+⟩^N. Expected: the GNN-prepared state requires larger χ sooner. This is the
Panel B counterpart to the already-saturated Panel A (ground-state, area-law).

---

## Cross-refs (repo)

Existing engine (reuse, do not duplicate):
- Quench + initial-state dependence: `scripts/experiment_runners/scaling/run_quench_dynamics_study.py`
  (`section_initial_state_dependence`, `section_mps_crossover`, `_mps_energy_evolution`, `_build_trotter_step_circuit`)
- MPS backend: `src/qmbp_simulation/execution/mps_backend.py` (`MPSBackend`, O(N·χ³))
- Entropy growth validation: `scripts/analysis/validate_dqpt_results.py` (`check_entropy_growth`)
- Entanglement analyzer: `src/qmbp_simulation/analysis/entanglement.py` (`EntanglementAnalyzer`)

Panel B (quench, done — supports this review's claim):
- Data + SUMMARY + figure: `results/classical_representability/panel_b_quench/`
  (S_max per initial state vs N; implied χ ~ 2^S_max)
- Consolidation analyzer: `project_health/analysis/scaling/entanglement_barrier_analyzer.py`
  (run: `python -m project_health.analysis.scaling.entanglement_barrier_analyzer --h1 0.5 --h2 1.5`)
- Envelopes: `results/experiments/exp_qd1/tfim/chain_1d_p1/` (section 1, exact evolution)
- Finding (N=6,10,14): S_max(GNN) − S_max(|0⟩^N) grows +(-0.02 → 0.20 → 0.50) bits,
  i.e. the GNN-prepared state needs a larger χ and the gap widens with N.
- Units note: half-chain entropy is reported in bits (log2). A prior nats/bits
  units bug (factor ln2) in the quench runner was fixed 2026-09-21; DQPT NPZs
  migrated via `scripts/general_project_maintenance/fix_dqpt_entropy_units.py`.

Preparation cost (HVA vs VL — supports the χ-vs-depth trade-off):
- Study: `scripts/analysis/run_hva_vs_vl_resources.py` + `analyze_hva_vs_vl_resources.py`
  (deps: `mps-to-circuit` 0.1.2 — Schön 2006 exact + Ran 2019 brick-wall synthesis)
- Data + SUMMARY + figures: `results/hva_vs_vl_resources/`
- Finding (chain_1d N=10): HVA prepares |ψ₀(h)⟩ at constant 18 CX (fidelity ceiling
  ~0.78 near h_c, ~0.99 paramagnetic); VL reaches fidelity→1 but 2q-cost grows with
  χ (VL exact χ=8: 515 CX @ h=1.0) and with the state's entanglement — 1–2 orders of
  magnitude above HVA. Preparation-side counterpart of the entanglement barrier.

Ground-state (Panel A, done):
- Study runner: `scripts/analysis/run_representability_mps.py`
- Analyzer + schema: `scripts/analysis/analyze_representability.py`, `scripts/analysis/representability_io.py`
- Results: `results/classical_representability/` (`SUMMARY.md`, `study_*.json`, `figures/`)
- Prior χ scaling: `internal/documentation/binnacles/binnacle-mps-scaling.md` (χ=64 exact to N=200)
- Memory limits by method: `internal/documentation/analysis/23_noisy_simulation_scalability_limits.md`

Diagrams:
- `docs/diagrams/diagram_ibm_comparison.txt` (Panel A vs Panel B argument)
- `docs/diagrams/diagram_pipeline_overview.txt`
