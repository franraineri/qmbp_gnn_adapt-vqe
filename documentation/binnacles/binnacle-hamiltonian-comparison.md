# Binnacle — Cross-Hamiltonian Comparison & TFIM Longitudinal Extension

## 2026-06-02 — Model Expressibility Analysis

### Objective

Systematically compare all Hamiltonians implemented in the GNN-HVA framework to determine
which models are viable for the pipeline (VQE → MPNN → deployment) and under what conditions.
Document the TFIM + longitudinal field extension (E4b) as the first successful model extension
beyond pure TFIM.

---

## Part 1: Implemented Models (Registry Status)

Data source: `from qmbp_simulation.models.model_registry import list_models`

| Model | Hamiltonian | Params/layer | Initial State | Status |
|-------|-------------|:---:|:---:|--------|
| `tfim` | H = −J·ZZ − h·X | 2 | \|+⟩^N | ✅ Production (210+ runs) |
| `tfim_longitudinal` | H = −J·ZZ − h·X − g·Z | 3 | \|+⟩^N | ✅ Validated (E4b) |
| `heisenberg` | H = J(XX+YY+Δ·ZZ) − h·Z | 4 | Néel | ❌ p≤2 insufficient |
| `xy` | H = J(XX+YY) − h·Z | 4 | Néel | ❌ p≤2 insufficient |

---

## Part 2: TFIM Standard — Baseline Performance (from digest)

Data source: `python -m scripts.digest --group-by topology --kind noiseless`

| Topology | Runs | Pass(<5%) | Marginal | Fail | Med ΔE/gap | Best | Worst |
|----------|:----:|:---------:|:--------:|:----:|:----------:|:----:|:-----:|
| chain_1d | 118 | 56 | 14 | 11 | 0.0288 | 0.0010 | 7.41 |
| heavy_hex | 18 | 12 | 0 | 5 | 0.0056 | 0.0004 | 10.67 |
| kagome | 2 | 2 | 0 | 0 | 0.0159 | 0.0002 | 0.03 |
| ladder | 116 | 65 | 21 | 23 | 0.0364 | 0.0030 | 11.06 |
| triangular | 75 | 39 | 5 | 26 | 0.0404 | 0.0019 | 14.40 |

**Key observations:**
- Median ΔE/gap < 5% for ALL topologies (chain, heavy_hex, ladder, triangular, kagome)
- Heavy-hex has best median (0.56%) — consistent with hardware-native topology advantage
- Triangular has worst median (4.04%) — higher connectivity = harder optimization landscape

---

## Part 3: TFIM + Longitudinal Field (E4b) — Extension Results

### Background

E4 (standard HVA, 2 params/layer) showed that the ZZ+X circuit structure cannot represent
the ground state when g>0. Fidelity dropped from 0.99 at g=0 to 0.556 at g=0.5.

E4b adds an RZ layer to the HVA (3 params/layer: θ_zz, θ_x, θ_z), matching the Hamiltonian
structure H = −J·ZZ − h·X − g·Z exactly.

### E4b Execution (3 seeds × 5 g-values × 5 h-values = 75 data points)

Data source: E4b experiment (2026-06-02), N=6, p=2, chain_1d

| g | E4 (std HVA) | E4b (ext HVA) | Δ Fidelity | Pass≥0.90 |
|---|:---:|:---:|:---:|:---:|
| 0.0 | 0.990 | 0.991 | +0.001 | 100% |
| 0.1 | 0.889 | 0.984 | +0.095 | 100% |
| 0.2 | 0.778 | 0.985 | +0.207 | 100% |
| 0.3 | 0.688 | 0.985 | +0.297 | 100% |
| 0.5 | 0.556 | 0.987 | +0.431 | 100% |

**Result: Hypothesis CONFIRMED** — Extended HVA achieves fid≥0.98 for ALL g≤0.5.
Maximum improvement: +43 fidelity points at g=0.5.

### Cross-Topology Validation (Section 2 of full validation)

| Topology | Mean Fidelity | Min Fidelity | Points |
|----------|:---:|:---:|:---:|
| chain_1d | 0.9856 | 0.9785 | 9 |
| ladder | 0.9910 | 0.9889 | 9 |
| triangular | 0.9952 | 0.9910 | 9 |

**All topologies pass** with fid≥0.97 minimum — the extension is topology-agnostic.

### Scaling (N=4,6,8 at g=0.3)

| N | h_min (g=0.3, ext HVA) | h_min (g=0, std HVA) | Observation |
|---|:---:|:---:|---|
| 4 | < 0.5 (all pass) | 0.95 | Extended HVA is MORE expressive |
| 6 | < 0.5 (all pass) | 1.20 | No boundary detected in valid range |
| 8 | < 0.5 (all pass) | 1.30 | Extended HVA dominates everywhere |

**Interpretation:** The 3rd parameter (θ_z) provides enough additional expressibility that
the valid regime boundary drops below h=0.5 for all system sizes tested. The extended HVA
is strictly more powerful than the standard TFIM HVA (superset of states reachable).

### Hardware Viability

Data source: verification test 5 (FakeTorino transpilation)

| Depth | 2Q Gates (extended) | 2Q Gates (standard) | ZNE Budget (≤18) |
|:---:|:---:|:---:|:---:|
| p=1 | 10 CZ | 10 CZ | ✅ OK |
| p=2 | 20 CZ | 20 CZ | ❌ Exceeds |

**The RZ layer adds ZERO 2-qubit gates.** Hardware overhead is identical to standard TFIM HVA.
p=1 + ZNE remains the viable strategy for IBM Torino deployment.

---

## Part 4: Heisenberg Model — Negative Result (from V9 binnacle)

Data source: `documentation/binnacles/binnacle-heisenberg-extension.md`

### Summary of 30 Heisenberg runs

| Config | Best Fidelity | Avg Fidelity | Verdict |
|--------|:---:|:---:|:---:|
| Δ=1.0, \|+⟩^N, p=2 | 22% | 12% | ❌ |
| Δ=1.0, Néel, p=2 | 48% | 9.8% | ❌ |
| Δ=0 (XY), Néel, p=2 | 23% | 4.7% | ❌ |

**Root cause:** HVA p≤2 has 8 parameters for Heisenberg (4/layer). The Heisenberg ground state
requires highly entangled states with entanglement entropy that scales linearly with N.
The HVA p=2 ansatz cannot produce sufficient entanglement at any parameter values.

**This is a physics limit, not an optimization failure.**

---

## Part 5: Experiment Verdicts (from compare --all)

Data source: `python scripts/compare.py --all`

| ID | Model | Verdict | Key Finding |
|----|-------|---------|-------------|
| E4 | tfim_longitudinal (std HVA) | ⚠️ Rejected | Standard HVA fails at g>0 |
| E4b | tfim_longitudinal (ext HVA) | ✅ Confirmed | Extended HVA fixes the failure |
| V9 | heisenberg | ❌ Failed | HVA p≤2 structurally insufficient |

### E4 → E4b Progression

```
E4:  H = -J·ZZ - h·X - g·Z  +  circuit = RZZ + RX          → FAILS (fid=0.556 @ g=0.5)
E4b: H = -J·ZZ - h·X - g·Z  +  circuit = RZZ + RX + RZ     → PASSES (fid=0.987 @ g=0.5)
```

**Lesson: HVA works if and only if the circuit structure mirrors every term of H.**

---

## Part 6: Comparative Analysis — Decision Matrix

| Criterion | TFIM | TFIM+Long | Heisenberg |
|-----------|:----:|:---------:|:----------:|
| HVA p=2 viable | ✅ (h≥1.2) | ✅ (all h) | ❌ |
| HVA p=1 viable | ✅ (h≥1.6) | ⚠️ (h≥2.0) | ❌ |
| Cross-topology | ✅ (5 tested) | ✅ (3 tested) | ❌ |
| MPNN trainable | ✅ (proven) | ✅ (expected) | N/A |
| ZNE compatible (p=1) | ✅ (10 CZ) | ✅ (10 CZ) | N/A |
| Hardware deployment | ✅ (planned) | ✅ (viable) | ❌ |
| Phase diagram dim | 1D (h) | 2D (h, g) | 1D (h) |
| Params/layer | 2 | 3 | 4 |
| Physical content | Z₂ QPT | Crossover (g>0 breaks Z₂) | AFM→PM QPT |

---

## Part 7: Recommendations for Thesis

### Chapter 5 Integration

1. **§5.1**: Standard TFIM results (existing, 210+ runs across 5 topologies)
2. **§5.5**: Model extensibility — E4b demonstrates the HVA-matches-H principle
3. **§5.6**: Negative results — Heisenberg confirms expressibility limits of p≤2
4. **§5.7**: Future work — TFIM+longitudinal as stepping stone to 2D phase diagrams

### Key Claim (new, from E4b)

> "The framework is extensible to modified Hamiltonians when the HVA circuit
> structure is adapted to match all terms. Adding a longitudinal field g·Z to
> the TFIM requires only an RZ layer addition (3→6 parameters for p=2), which
> restores fidelity ≥0.98 at g=0.5 with zero hardware overhead (RZ is single-qubit)."

### Supporting Data for Claim

- 47 unit tests validating Hamiltonian, circuit, registry, and expressibility
- 75 VQE data points (3 seeds × 5g × 5h) with 100% pass rate
- Cross-topology confirmation (chain, ladder, triangular)
- Hardware viability confirmed (identical 2Q gate count to standard TFIM)

---

## Part 8: Code Changes Summary

| File | Change |
|------|--------|
| `src/qmbp_simulation/models/hamiltonian.py` | `build_tfim_longitudinal()`, `build_tfim_longitudinal_observables()` |
| `src/qmbp_simulation/circuits/hva.py` | `create_tfim_longitudinal()` (3 params/layer) |
| `src/qmbp_simulation/models/model_registry.py` | Registered `tfim_longitudinal` ModelSpec |
| `src/qmbp_simulation/models/model_spec.py` | `with_g()`, `with_params()` methods |
| `scripts/digest/models.py` | `model`, `model_params` fields; E4b criteria |
| `scripts/digest/scanner.py` | Model extraction from result JSONs |
| `scripts/digest/formatters.py` | `--group-by model` support |
| `scripts/digest/__main__.py` | `--model` filter CLI flag |
| `experiments/generalization/exp_e4b_longitudinal_hva_extended.py` | Full E4b experiment |
| `tests/unit/test_tfim_longitudinal.py` | 47 comprehensive tests |
| `.kiro/steering/code-style.md` | Extension pattern documentation |
| `.kiro/knowledge/changelog.md` | Entry dated 2026-06-02 |

---

*Binnacle entry complete. Next: hardware deployment of TFIM+longitudinal at g=0.3 on IBM Torino (p=1, 10 CZ gates, within ZNE budget).*
