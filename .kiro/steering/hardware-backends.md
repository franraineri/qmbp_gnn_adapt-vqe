---
inclusion: fileMatch
fileMatchPattern: "**/hardware/**,**/hardware_deployer*,scripts/run_hardware*,**/ibm_*deployment*"
---

# IBM QPU Backend Reference (auto-activated for hardware files)

## Current Target: ibm_kingston (Heron r2, 156 qubits)

**Default backend since 2026-06-14.** Previous default was ibm_torino (Heron r1, 133 qubits).

### Real-World Performance Observed (2026-06-14 evening)

| Metric | Spec (IBM published) | Observed (runtime) | Note |
|--------|---------------------|-------------------|------|
| Mean 2Q error | 1.95×10⁻³ (median) | **3.36%** (chip-wide mean) | Includes degraded qubits |
| Layout 2Q error | — | ~1-2% (selected subgraph) | BFS+CES avoids bad qubits |
| Min T1 | — | 6.5 μs | Isolated TLS defect |
| P5 T1 | — | 125.9 μs | Most qubits healthy |
| Readout error | 8.3×10⁻³ (median) | 1.96% (mean) | TREX mitigates |
| QPU time (1 job, PEA 32×128) | — | 284s | Includes noise learning |
| Layout CES | — | 0.050 | Very good (low noise subgraph) |

**Key insight**: IBM published specs are MEDIAN per-gate values. The chip-wide MEAN is
significantly higher (3.36% vs 0.2%) because large processors have degraded outlier qubits.
Our pipeline must tolerate 2-4% chip-wide mean while relying on layout selection to find
the ~1% subgraphs.

### ibm_kingston Specifications

| Property | Value | Impact for HVA |
|----------|-------|---------------|
| Processor | Heron r2 | Current generation, best available |
| Qubits | 156 | More layout options than Heron (133) |
| Topology | Heavy-hex | Same as our `heavy_hex` lattice topology |
| Couplers | 176 tunable | Low crosstalk (~10⁻⁵) |
| Native 2Q gate | **CZ** | Our HVA uses RZZ → decomposes to CZ natively |
| Native gates | cz, id, rx, rz, **rzz**, sx, x | **RZZ is native** — HVA executes directly |
| 2Q error (median) | 1.95×10⁻³ | 2× better than Heron (~4×10⁻³) |
| 2Q error (best) | 8.28×10⁻⁴ | Excellent for selected layouts |
| Readout error (median) | 8.30×10⁻³ | TREX mitigates this |
| CZ error (median) | 1.947×10⁻³ | Key metric for ZNE budget |
| SX error (median) | 2.637×10⁻⁴ | Negligible vs 2Q error |
| T1 (median) | 258.88 μs | Better than Heron (~200 μs) |
| T2 (median) | 131.6 μs | Better than Heron (~100 μs) |
| CLOPS | 340K | ~1.5× faster than Heron |
| Region | Washington DC (us-east) | Low latency from Americas |
| Status | Online | Available on Open Plan |
| TLS mitigation | Built-in (Heron r2 feature) | Improved coherence stability |

### Key Advantage: RZZ is Native

Our HVA circuit uses `qc.rzz(2*θ_zz, i, j)` for ZZ interaction terms. On Kingston,
**RZZ is a native basis gate** — the transpiler keeps it as-is without decomposition.

- **On Heron (Eagle)**: RZZ → 2 CX + RZ rotations (2× gate overhead)
- **On Kingston (Heron r2)**: RZZ → RZZ (no decomposition, 1 native pulse)

This means our N=10 p=1 circuit with 9 ZZ bonds uses **9 native RZZ gates** instead
of 18 CX gates. Effective 2Q depth is halved. ZNE budget improves proportionally.

### Preflight Behavior on Kingston

Large processors (156 qubits) typically have 1-5 qubits with anomalously low T1
due to TLS defects. The preflight system uses **5th-percentile T1** (not minimum)
to determine abort threshold:
- **Abort**: p5_T1 < 30 μs (widespread decoherence)
- **Warning**: min_T1 < 50 μs but p5_T1 healthy (isolated defect, layout avoids it)
- **Pass**: min_T1 ≥ 50 μs (all qubits healthy)

---

## ibm_boston (Heron r3, 156 qubits) — Premium/Flex/PAYG Only

| Property | Value | vs Kingston |
|----------|-------|-------------|
| Processor | Heron r3 | Latest revision |
| Qubits | 156 | Same |
| EPLG (100q) | 2.15×10⁻³ | Better (Kingston: ~3.4×10⁻³) |
| 2Q gates below 10⁻³ | 57/176 | More low-error gates |
| Topology | Heavy-hex | Same |
| T1 median | ~300 μs | Slightly better |
| Access | Premium/Flex/PAYG plans | NOT available on Open Plan |

### Advantages over Kingston

1. **Lower EPLG**: 2.15×10⁻³ vs ~3.4×10⁻³ → PEA noise model fits better
2. **More low-error gates**: 57/176 below 10⁻³ → better layout options
3. **Fewer TLS events**: r3 revision has improved coherence stability
4. **PEA default budget should WORK**: With lower base error, IBM default (32×128) may suffice

### Disadvantages

1. **Paid plan required**: Not available on Open Plan (our current access level)
2. **Same topology**: Still heavy-hex with uniform CES issue
3. **Queue may be longer**: Premium backends shared among paying users

### When to use ibm_boston

- If our IBM instance gets Premium/Flex access
- If Kingston continues to show >3% mean 2Q error
- For final thesis data (lower error = more defensible results)

Switch with `--backend ibm_boston`.

---

## ibm_torino (Heron r1, 133 qubits) — Legacy

| Property | Value | vs Kingston |
|----------|-------|-------------|
| Processor | Heron r1 | Previous generation |
| Qubits | 133 | Fewer (less layout space) |
| Native 2Q | ECR (via fractional rzz) | Same effective behavior |
| 2Q error | ~4×10⁻³ | 2× worse than Kingston |
| T1 median | ~200 μs | Lower coherence |
| CLOPS | ~2500 (effective) | Slower |

**Status**: Still available but superseded by Kingston. Use `--backend ibm_torino`
only if Kingston is unavailable.

---

## Nighthawk (ibm_miami, ibm_berlin) — Next Generation (2026+)

| Property | Value | vs Kingston |
|----------|-------|-------------|
| Processor | Nighthawk | Next-gen architecture |
| Qubits | 120 | Fewer, but better connected |
| Topology | **Square lattice** | 4-connected (vs ~2.5 heavy-hex) |
| Couplers | 218 | +20% vs Heron |
| EPLG (100q) | 2.15×10⁻³ | Same as best Heron r3 |
| T1 median | 350 μs | Best coherence in fleet |
| 2Q gate time | 68 ns | 2× faster than Heron (138 ns) |
| Gates per coherence | ~5000 | 2× more than Heron |

**Impact**: Square lattice means chain_1d maps with ~0 SWAP overhead (vs ~0.3N on heavy-hex).
N=80 becomes viable at ΔE/gap < 5%. Currently available only on Premium plans.

---

## Backend Selection Decision Tree

```
Is ibm_boston available (paid plan)?
  YES → Use ibm_boston (best error rates)
  NO  → Is ibm_kingston available?
    YES → Use ibm_kingston (default, Open Plan)
    NO  → Check ibm_torino (legacy fallback)
```

CLI: `python scripts/experiment_runners/hardware/run_ibm_deployment.py --backend <name>`

---

## Transpilation Considerations by Backend

| Backend | Native 2Q | HVA RZZ handling | Expected 2Q gates (N=10 p=1) |
|---------|-----------|------------------|:---:|
| Kingston/Boston | CZ + RZZ | RZZ native, no decomposition | 9 |
| Heron | ECR (+ rzz fractional) | RZZ via fractional gate | 9-18 |
| Nighthawk | CZ + RZZ | RZZ native + better routing | 9 |
| FakeTorino (local) | ECR | RZZ → 2 CX + rotations | ~18 |

**Important**: FakeTorino (used in rehearsal) uses ECR-based decomposition and shows
~18 2Q gates. On real Kingston/Boston the transpiled circuit will have ~9 RZZ gates.
This means rehearsal results are CONSERVATIVE (more noise than real hardware).

---

## PEA Learning Budget vs Backend Quality (2026-06-14 finding)

The PEA noise amplification quality depends directly on calibration quality:

| Backend class | Typical mean 2Q error | IBM default PEA (32×128) | Balanced (48×192) | Recommendation |
|---------------|:----:|:----:|:----:|---|
| Best (ibm_boston, r3) | <1% | ✅ Should work | Overkill | Use default |
| Good (Kingston fresh cal) | 1-2% | ⚠️ Borderline | ✅ Safe | Use balanced |
| Degraded (Kingston evening/weekend) | 2-4% | ❌ Fails (32.5% ΔE/gap) | Pending test | Use balanced or aggressive |
| Bad (any, post-TLS event) | >4% | ❌ | ❌ | Wait for recalibration |

**Rule of thumb**: If preflight reports chip-wide mean 2Q > 2%, use at least `--pea-config balanced`.
If chip-wide mean > 4%, defer execution.

---

## Configuration Defaults (updated 2026-06-14)

```python
# In src/qmbp_simulation/execution/hardware/config.py
HardwareConfig(
    backend_name="ibm_kingston",  # Changed from "ibm_torino"
    ...
)

# In scripts/experiment_runners/hardware/run_ibm_deployment.py
BACKEND_NAME = "ibm_kingston"

# QPU throughput profile
QPUThroughputProfile.ibm_kingston()  # CLOPS=3750 effective
```

---

## Scalability Notes

| N | 2Q gates (native RZZ) | Est. fidelity (Kingston) | ΔE/gap post-PEA | Viable? |
|:--:|:--:|:--:|:--:|:--:|
| 10 | 9 | 98.3% | <1% | ✅ |
| 20 | 19 | 96.4% | ~1% | ✅ |
| 40 | 39 | 92.7% | ~2% | ✅ |
| 50 | 49 | 90.9% | ~3% | ✅ |
| 80 | 79 | 85.7% | ~4.4% | ⚠️ borderline |
| 100 | 99 | 82.4% | ~5.2% | ❌ exceeds 5% |

For N>50, use Nighthawk (square lattice, fewer SWAPs, better T1).
