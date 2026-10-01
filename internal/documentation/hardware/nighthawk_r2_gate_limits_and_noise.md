# IBM Quantum Nighthawk r2 — gate limits, error rates, and the t_noise model

Reference note for the square-lattice frustrated-TFIM study. Collects the
hardware specs, the gate-count limits, and the real two-qubit error rates we use
to build a realistic (fake-backend-free) `t_noise` model.

Last updated: 2026-10-02. All figures from public IBM sources + arXiv papers
(cited inline). Content was rephrased for compliance with licensing restrictions.

---

## 1. Hardware specifications (Nighthawk r2)

Available on IBM Quantum Platform since January 2026.

| Property | Value |
|----------|-------|
| Programmable qubits | 120 |
| Total physical elements | 458 (120 qubits + 218 couplers + 120 reset elements) |
| Connectivity | Square lattice, each qubit coupled to 4 neighbours |
| Coupler pairs | 218 (~20% more connectivity than prior heavy-hex generations) |
| Native 2q gate | **CZ** (not CX) |
| Qubit reset | Dissipative high-speed reset — effective T1 from ~200 µs to ~25 ns on demand |
| Throughput | 100,000+ circuits/second (~25× IBM Heron) |
| Gate fidelity | "Heron-class" (preserved from Heron while improving speed) |

Sources: [IBM Nighthawk r2 blog](https://www.ibm.com/quantum/blog/nighthawk-r2),
[IBM newsroom 2025-11-12](https://newsroom.ibm.com/2025-11-12-IBM-Delivers-New-Quantum-Processors,-Software,-and-Algorithm-Breakthroughs-on-Path-to-Advantage-and-Fault-Tolerance),
[Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/ibm-unveils-new-120-qubit-processor-and-software-stack).

---

## 2. Gate-count limits — two distinct numbers

These are frequently conflated. They are not the same thing.

| Metric | Value | What it means |
|--------|-------|---------------|
| **2q (entangling) gates supported today** | **~5,000** | Raw architectural capacity of the chip |
| **Total gates with error mitigation (2026 milestone)** | **7,500+** | Demonstrated accurate observable estimation, *with* error-reduction techniques on top |

- The ~5,000 two-qubit gate figure is the architectural operating point
  ([The Quantum Insider, 2026-01-13](https://thequantuminsider.com/2026/01/13/ibm-announces-nighthawk-and-latest-heron-are-now-available/)).
- The 7,500+ figure is the roadmap milestone already demonstrated, but the IBM
  roadmap page is explicit that it is reached "paired with techniques that reduce
  errors" — it is **not** a raw, unmitigated gate count
  ([IBM Nighthawk r2 blog](https://www.ibm.com/quantum/blog/nighthawk-r2),
  [IBM Quantum Roadmap 2026](https://www.ibm.com/roadmaps/quantum/2026/)).

### Roadmap trajectory

| Year | Gates | Qubits |
|------|-------|--------|
| 2025 (r1) | 5,000 | 120 |
| 2026 (r2) | 7,500 | up to 3× 120-qubit modules (360) |
| 2027 | 10,000 | up to 1,080 |
| 2028 | 15,000 | — |

Source: [IBM Quantum Roadmap](https://www.ibm.com/roadmaps/quantum/).

---

## 3. Real two-qubit error rates

### Primary anchor — median CZ error ≈ 3×10⁻³

The hardware-optimization benchmark ([arXiv:2607.11637](https://arxiv.org/abs/2607.11637))
places the onset of noise-dominated execution at **~770 two-qubit gates for an
estimated fidelity F ≈ 0.1**, at the median Heron-r2 CZ error rate. Inverting the
exponential-decay model `F = exp(−N₂q · ε₂q)`:

```
ε₂q = −ln(0.1) / 770 ≈ 2.99×10⁻³
```

Because the Nighthawk r2 blog states it **preserves "Heron-class gate fidelity"**,
this median CZ error applies to Nighthawk r2 as well. We adopt:

> **ε₂q (median CZ, Heron-class) = 3×10⁻³** — the default for the realistic model.

### Cross-validation — random-circuit sampling on Nighthawk r2

The RCS quantum-advantage paper ([arXiv:2609.28657](https://arxiv.org/abs/2609.28657))
ran directly on a Nighthawk r2 device (`ibm_phoenix`), 61 qubits, native CZ,
reporting `F_XEB(36 cycles) = 2.3×10⁻³`. With ~30 two-qubit gates per cycle on 61
square-lattice qubits (~1,080 2q gates), the effective per-gate error is ≈5.6×10⁻³.
This is an **upper bound** on the per-gate error — XEB fidelity also absorbs SPAM,
readout, and 1q-gate error. It brackets the real value together with the median:

> **ε₂q ∈ [3×10⁻³, 5.6×10⁻³]** (median CZ → XEB-effective with SPAM).

---

## 4. The realistic t_noise model

Model: `F(N₂q) = exp(−N₂q · ε₂q)`. Define `t_noise` as the number of 2q gates at
which fidelity crosses the `1/e` floor (`N* = 1/ε₂q`).

| ε₂q | Interpretation | Gates to F = 1/e |
|-----|----------------|------------------|
| 1×10⁻³ | optimistic / future | 1,000 |
| **3×10⁻³** | **median CZ (default)** | **333** |
| 5.6×10⁻³ | XEB-effective (with SPAM) | 180 |

Implemented in `qmbp_simulation.analysis.dynamics`:

- `NIGHTHAWK_CZ_ERROR = 3e-3` — the default median CZ error.
- `fidelity_from_2q(n_2q, eps_2q=NIGHTHAWK_CZ_ERROR)` — the decay model.
- `tnoise_from_2q_curve(cumulative_2q, eps_2q, floor)` — first step crossing 1/e.

This analytic model is preferred over the Qiskit **fake backends**
(`FakeNighthawk`, `FakeTorino`) for the realistic budget because:

1. Fake-backend calibration is pessimistic and poisoned by dead couplers
   (error = 1.0) that inflate the chip-wide mean — we previously had to pass the
   transpiled layout to work around this.
2. The analytic model is transparent, reproducible, and has a single tunable knob
   (`ε₂q`) — lowering it is exactly how a future **noise-suppression model**
   (ZNE/QESEM/circuit compression) will enter the budget.

---

## 5. Contrast with our circuits

| Quantity | Value |
|----------|-------|
| Chip architectural limit | ~5,000 2q gates |
| Noise limit (median CZ, no mitigation) | ~333 2q gates to F = 1/e |
| Our prune0.3 ansatz @ N=14 | ~60 logical 2q (158 transpiled) |
| Our prune0.3 ansatz @ N=10 | ~18 logical 2q (34–44 transpiled) |

Our preparation circuits sit **well below** the realistic noise limit at median CZ
error (60–158 ≪ 333). The `t_noise ≈ 1` we saw earlier was an artefact of the
pessimistic fake-backend calibration, **not** a true hardware limit for these
circuits. The real bottleneck for square-lattice frustrated TFIM at larger N is
the **SWAP-routing overhead** (×6 at N=21: 160 logical → 973 transpiled 2q), which
is a mapping cost, not a per-gate-error cost.

---

## 6. Working priority

- **Now:** study the physics in **ideal (noiseless) simulation** — dynamics, DQPT,
  classical crossover t*(χ). None of this depends on a hardware noise model.
- **t_noise budget:** use the analytic model with `ε₂q = 3×10⁻³` (median CZ) as a
  transparent, reproducible alternative to the fake backend.
- **Later:** a dedicated noise-suppression model will enter by lowering `ε₂q`
  (and/or reducing the transpiled 2q count via compression).

<!-- Links -->
[arxiv-opt]: https://arxiv.org/abs/2607.11637
[arxiv-rcs]: https://arxiv.org/abs/2609.28657
[ibm-blog]: https://www.ibm.com/quantum/blog/nighthawk-r2
[ibm-roadmap]: https://www.ibm.com/roadmaps/quantum/
