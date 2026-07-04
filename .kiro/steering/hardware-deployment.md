---
inclusion: manual
---

# Hardware Deployment — Phase 4 Guidelines

## IBM Kingston Target (156 qubits, Heron r2)

### Connection
- Use `QiskitRuntimeService` with `channel="ibm_quantum_platform"`
- Token from `os.environ["IBM_KEY"]`, instance from `os.environ["IBM_INSTANCE_CRN"]`
- Backend: `"ibm_kingston"` (Heron r2, 156 qubits). Override with `--backend <name>`.
- Alternative: `"ibm_boston"` (Heron r3, if on paid plan — better error rates, EPLG=2.15×10⁻³).
- **API (qiskit-ibm-runtime 0.47.0)**:
  - `EstimatorV2(mode=backend)` — NOT `backend=` (changed ~v0.40)
  - Multi-job submissions MUST use `Batch(backend=backend)` context (prevents CANCELLED jobs)
  - `job.metrics()["timestamps"]["running"]` is ISO string, NOT numeric seconds

### Circuit Preparation
- `generate_preset_pass_manager(backend=backend, optimization_level=2)`
- Apply layout to observables: `obs.apply_layout(isa_qc.layout)`
- Add dynamical decoupling: `PadDynamicalDecoupling` pass after transpilation
- **AQC-Tensor compression** (optional, `--aqc-compress`): compresses p=2 circuits to
  p=1-equivalent 2Q-gate count while retaining expressibility. Ref: `.kiro/steering/context-aqc-compression.md`

### Error Mitigation Stack (in order of application)
1. **Dynamical Decoupling** — free, always apply. Use optimized sequences if available.
2. **Pauli Twirling** — 32 randomizations × 128 shots. Converts coherent → stochastic noise.
3. **TREX** — twirled readout error extinction. Enable via EstimatorV2 options.
4. **ZNE** (configurable amplifier — select via `MitigationOptions.zne_amplifier`):
   - **PEA** (`"pea"`, default): Probabilistic Error Amplification. Learns noise model
     via Pauli-Lindblad fitting, then amplifies probabilistically. ~50% QPU overhead from
     noise learning phase. Validated +94.4% gain across all topologies (t=46.32, p<10⁻¹⁹).
     IBM Runtime handles it automatically via `options.resilience.zne.amplifier = "pea"`.
   - **Gate-folding** (`"gate_folding"`, fallback): Digital noise amplification U→U·U†·U at
     factors [1,3,5]. Simple, zero overhead, validated locally (R²>0.99 on chain_1d).
     May give low R² on heavy_hex p=1 shallow circuits.
   - **Adaptive** (`"adaptive"`): Tries gate-folding first; if R²<threshold (default 0.90),
     falls back to PEA. Best for unattended deployment where topology properties are uncertain.
     Configure threshold via `MitigationOptions.zne_r2_fallback_threshold` or `--zne-r2-threshold`.
   - **Inhomogeneous CES-ZNE** (deprecated for heavy_hex): Different layouts → different CES.
     Fails on heavy_hex due to uniform CES≈0.15. Only used as legacy path when `zne_enabled=False`.
5. **NN-enhanced extrapolation** — optional improvement:
   - After collecting ZNE data, fit 2-layer MLP instead of linear regression
   - `MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000)`

### Mitigation Benchmark V2 Findings (2026-06-18, FakeTorino, θ_opt corrected)

Systematic evaluation of 21 configs × 15 h-values (V2 fixes critical θ=zeros bug from V1).
Ref: `documentation/binnacles/binnacle-mitigation-benchmark-v2.md`.

**Circuit metrics (post-transpilation):**
- Standard (opt_level=2): n_2Q=18, depth_2q=14, total_depth=59-62
- AQC (opt_level=2): n_2Q=27, depth_2q=21, total_depth=103
- Mitiq (opt_level=0): n_2Q=45, depth_2q=32, total_depth=136 ← destructive

**Per-regime results (h≥3.0 = production target):**
- **All PEA variants (C4-C8, C10, C15): 0.37% ΔE/gap** — hardware-viable ✅
- C16_aqc_pea: 2.1% — global champion, best in critical regime (p=2 expressibility)
- C3_full_gf: 27-30% — GF-ZNE fallback (31% reduction vs raw)
- C0_raw: 40-44% — baseline
- C11_mitiq_zne: 81% — **destructive** (opt_level=0 routing → 45 CZ)

**Key findings relevant to hardware deployment:**
- **PEA budget does NOT differentiate in simulation** (all converge to 0.37% — depolarizing perfectly learned). On real HW with noise fluctuations, budget WILL matter. Test C4 vs C5 vs C6.
- **DD/Twirling: zero effect in sim** (depolarizing only). Enable on hardware anyway (free, helps coherent errors).
- **GNN-QEM after PEA: 0% improvement** (post-PEA residual is unstructured shot noise).
- **AQC+PEA wins in critical regime (h<2.0)**: 70% vs 71% for standard PEA. The p=2 compressed target provides more expressibility.
- **AQC without PEA is WORSE than raw**: 27 CZ > 18 CZ → more noise without mitigation.
- **Mitiq ZNE is contraproducente at N≥10**: opt_level=0 forces routing → 45 CZ (2.5× more gates). Do NOT use.
- **Phase classification: 100% correct** regardless of ΔE/gap (H19 CONFIRMED).

**Hardware execution order (7 configs × 4 h × 16K shots):**
```bash
python scripts/experiment_runners/hardware/run_mitigation_benchmark.py \
    --mode hardware --configs C0,C1,C3,C4,C5,C6,C16 \
    --h-values 3.25,3.5,3.75,4.0 --shots 16384
```

Runner: `python scripts/experiment_runners/hardware/run_mitigation_benchmark.py`
Analyzer: `python -m project_health.analysis.mitigation_benchmark_analyzer --thesis-table`

### Shot Budget
- Minimum: **8192 shots** (σ ≈ 1.1e-2, comparable to ⟨X⟩ signal)
- Recommended for N=10: **16384 shots** (σ ≈ 7.8e-3, below ⟨X⟩ signal of 8.4e-3)
- Use `EstimatorV2` `precision` parameter to control shot allocation

### Observable Grouping
- ⟨X_i⟩ observables: all commute (single measurement basis)
- ⟨Z_iZ_{i+1}⟩ observables: all commute (single measurement basis)
- Total: 2 circuit executions per noise level (not N+N-1 separate runs)

### Expected Hardware Behavior (from literature)
- Ground-state energies: reliably captured across full parameter space
- Magnetic order parameters: noise broadening near critical crossover
- Phase classification: correct away from h_c, "smeared" near transition
- Success criterion: ΔE/gap < 5% AND correct phase label — NOT fidelity ≥ 99.5%

### AdaptVQE on Hardware
- max_iterations = 2 (Mele et al. constraint)
- gradient_threshold = 1e-3
- If AlgorithmError at iteration 0 → ideal outcome (warm-start was optimal)
- Pauli pool: Hamiltonian terms only (ZZ bonds + X sites)
- Use COBYLA or SPSA optimizer (gradient-free, noise-robust)

---

## CRITICAL PITFALLS (learned from V6.1 implementation)

### EstimatorV2 Observable Return Types — NEVER FORGET
- `(circuit, single_SparsePauliOp)` → returns **SCALAR** (the weighted sum)
- `(circuit, [list_of_SparsePauliOps])` → returns **ARRAY** (one value per op)
- For per-site ⟨X_i⟩ or per-bond ⟨Z_iZ_j⟩: ALWAYS submit as a LIST of individual single-term operators
- For total energy: submit the full Hamiltonian as a single SparsePauliOp (scalar is what you want)
- This applies to BOTH StatevectorEstimator AND IBM Runtime EstimatorV2

### Energy Computation — Don't Reconstruct Manually
- WRONG: `energy = -J * np.sum(zz_vals) - h * np.sum(x_vals)` (error-prone, ignores per-bond J)
- RIGHT: Submit the full Hamiltonian as a PUB → get energy directly from Estimator
- The Estimator handles the coefficient weighting correctly for any Hamiltonian structure

### Inhomogeneous ZNE — Two Types of CES
- **Topology CES** (`_compute_subset_ces`): sum of edge errors in the qubit subset's connectivity. Fast heuristic for RANKING candidate layouts. Does NOT account for routing overhead.
- **Circuit CES** (`compute_ces(transpiled)`): sum of 2Q gate errors in the actual transpiled circuit. This is the TRUE noise axis for ZNE extrapolation.
- Use topology CES for selection, circuit CES for extrapolation. Never mix them.

### NNConv — Use Sum Aggregation
- `aggr="add"` not `"mean"` — mean loses node degree information (Xu et al. 2019)
- This matters for lattices where sites have different coordination numbers

### Calibration Timestamp — May Be None
- Modern IBM backends (Target API) don't always expose `backend.properties().last_update_date`
- Default to assuming FRESH calibration when timestamp unavailable
- The error rates themselves ARE accessible via `backend.target[op_name].get(qargs).error`

### Phase Classification — Use Magnitudes
- ⟨X⟩ ≥ 0 always for TFIM with |+⟩^N initial state
- ⟨ZZ⟩ ≤ 0 for our convention (H = -J*ZZ - h*X)
- Compare `|⟨X⟩|` vs `|⟨ZZ⟩|` for crossover criterion
- Return "indeterminate" when difference < σ = 1/√shots

### Layout Selection — Seed for Reproducibility
- BFS-based subset search uses random starting nodes
- Always use a seeded `random.Random(seed)` instance, not module-level `random.sample()`
- This ensures reproducible layout selection across runs
- **Mapomatic VF2** (2026-06-17): VF2 subgraph isomorphism finds SWAP-free layouts
  with ~6× lower CES than BFS. Enabled by default via `HardwareConfig(use_mapomatic=True)`.
  See: `.kiro/steering/context-layout-optimizer.md`

### No Libraries Exist For
- Inhomogeneous ZNE (Uvarov 2024) — must implement ourselves
- ~~Layout selection on heavy-hex topology~~ — **SOLVED**: mapomatic VF2 (`layout_optimizer.py`, 2026-06-17)
- Weight gradient analysis (Hernandes 2025) — must implement ourselves
- Mitiq does gate-folding ZNE only (different paradigm, not applicable)
- PEA local simulation — implemented in `noisy_utils.py` (IBM Runtime handles it on hardware)
- DD/twirling/TREX are native to Qiskit Runtime (just set options, no custom code)

---

## Do NOT
- Measure global fidelity on hardware (requires exponential tomography)
- Use more than p=2 total HVA layers (including ADAPT additions)
- Use Primitives V1 or `backend.run()`
- Hardcode h_c = 1.0 for phase classification (use data-driven crossover)
- Submit multi-term SparsePauliOp when you need per-term values
- Use `random.sample()` without a seed in layout selection
- Use `aggr="mean"` in NNConv (use `"add"`)
- Manually reconstruct energy from observables (submit Hamiltonian PUB instead)

---

## Noisy Simulation Mode (FakeKingston)

### Overview
`mode="fake_backend"` exercises the full ZNE pipeline locally using `FakeKingston` +
`BackendEstimatorV2`. No IBM credentials required. Validates that PEA-ZNE reduces
errors before real QPU deployment.

### Backend & Estimator
- Backend: `FakeKingston` from `qiskit_ibm_runtime.fake_provider` (156 qubits, Heron R2, heavy-hex)
- Alternative: `FakeBoston` (Heron R3, best calibration snapshot — 350/352 good edges)
- Estimator: `BackendEstimatorV2` from `qiskit.primitives` (local simulation)
- Uses `default_precision = 1/sqrt(shots)` parameter
- **Note**: FakeKingston calibration is a snapshot — CZ error median 0.18% but some edges at 100%.
  Real ibm_kingston has ~0.18% median on all functional edges.

### ZNE Strategy (2026-06-06 — post-refactoring)
- **Primary**: PEA-ZNE via `run_pea_zne()` — validated +94.4% gain, R²=0.998
- **Fallback**: Gate-folding ZNE via `run_gate_folding_zne()` — +20.6% gain
- **Adaptive**: Auto-selects GF first, falls back to PEA if R² < threshold
- **CES-ZNE**: DEPRECATED on heavy_hex (uniform CES≈0.15 → R²≈0.04)

### Post-ZNE Correction Stack
Applied automatically by `run_deployment()`:
1. **GNN-QEM** (optional) — Only if model loaded AND amplifier ≠ PEA. Confidence-gated.
2. **Affine correction** (always) — Clips to [E_ground, E_upper]. Zero cost.

### What's Included (same as hardware)
- Layout selection via VF2 mapomatic (primary) or BFS (fallback) on heavy-hex topology
- Transpilation with `generate_preset_pass_manager(optimization_level=2)`
- PEA noise amplification (learned from FakeTorino calibration data)
- Observable grouping (commuting Paulis — 2 circuit executions total)
- TLS calibration drift monitoring (between h-points in sweeps)

### What's Excluded (isolates ZNE contribution)
- No DD/twirling/TREX (no effect on local simulation)
- No IBM server-side ZNE (only available on real hardware)

### Usage Pattern
```python
from qmbp_simulation.execution.hardware import HardwareBackend, HardwareConfig

# Fake backend (local PEA-ZNE simulation)
config = HardwareConfig(mode="fake_backend", n_qubits=10, shots=16384)
backend = HardwareBackend(config=config)
energy = backend.evaluate(circuit, H, params)  # PEA-ZNE mitigated energy
result = backend.run_deployment(circuit, H, params, h, e_exact, gap)  # Full pipeline
```

### Rehearsal (mandatory before real QPU)
```bash
# Full rehearsal V2 (9 sections: ZNE, noise, circuit audit — ~60s)
python scripts/experiment_runners/run_hardware_rehearsal_v2.py
make hw-rehearsal

# MPNN Evaluation Suite V3 (sections 10-19 — MPNN-only, no FakeTorino)
# Validates warm-start speedup, LOO-CV, landscape quality, interp/extrap, scaling
python scripts/experiment_runners/run_hardware_rehearsal_v3.py \
  --skip-hardware-sections \
  --n-qubits 10 --topology heavy_hex --p-layers 1 \
  --h-train 4.5 4.25 4.0 3.75 3.5 3.25 3.0 --h-test 4.0 3.25 \
  --mpnn-epochs 3000 --vqe-restarts 1

# V3 individual sections (useful for targeted re-validation)
python scripts/experiment_runners/run_hardware_rehearsal_v3.py --section 10  # Warm-start speedup
python scripts/experiment_runners/run_hardware_rehearsal_v3.py --section 11  # LOO-CV
python scripts/experiment_runners/run_hardware_rehearsal_v3.py --section 19 \  # κ risk proxy
  --h-kappa-grid 4.5 4.0 3.5 3.25 3.0 2.75   # Extended grid recommended for N=10

# Full V2 rehearsal (sections 1-9) — required before any real QPU run
python scripts/experiment_runners/run_hardware_rehearsal_v2.py
make hw-rehearsal

# With optional Section 0 (backend preflight)
python scripts/experiment_runners/run_hardware_rehearsal_v2.py --run-preflight

# Quick check (cost + circuit audit, ~2s)
python scripts/experiment_runners/run_hardware_rehearsal_v2.py --section 8 9
make hw-rehearsal-quick

# Analyze results → GO/NO-GO verdict
python -m project_health.analysis.mpnn_eval_analyzer --thesis-table
```

### Deployment Script (real QPU — calibration-first)
```bash
# Step 0: Cost estimate (no QPU)
make hw-deploy-dry

# Step 1: Calibration run — with PEA preset selection
python scripts/experiment_runners/hardware/run_ibm_deployment.py \
    --no-spsa --tier 0 --pea-config balanced --backend ibm_kingston
# → Output: "Full sweep will take X min based on measured T_one_job"

# PEA preset options (from calibration study):
#   --pea-config default        → 32×128=4K learning (fast, may fail on degraded cal)
#   --pea-config balanced       → 48×192=9K learning (recommended sweet spot)
#   --pea-config aggressive     → 64×256=16K learning + 3 layouts (slow, max accuracy)
#   --pea-config default_3layout → 32×128 + 3 layouts (tests variance reduction)

# Step 2: Full deployment (recommended with --no-spsa safety)
make hw-deploy
# Equivalent to:
python scripts/experiment_runners/hardware/run_ibm_deployment.py --no-spsa

# Manual options:
python scripts/experiment_runners/hardware/run_ibm_deployment.py --tier 0      # Calibration
python scripts/experiment_runners/hardware/run_ibm_deployment.py --tier 1 2 3  # Execution
python scripts/experiment_runners/hardware/run_ibm_deployment.py --no-spsa     # No SPSA (safe)
```

### Real QPU Findings (2026-06-14, ibm_kingston)

| Config | ΔE/gap | Wall-clock | Verdict | Note |
|--------|--------|-----------|---------|------|
| default (32×128, 1 layout) | 32.5% | 12.4 min | FAIL | PEA model too coarse for 3.4% error |
| aggressive (64×256, 3 layouts) | — | >17 min (cancelled) | — | Too expensive for iterative testing |

**Lesson**: IBM default PEA budget is insufficient when chip-wide 2Q error > 2%.
The `balanced` preset (48×192, [1,1.5,3]) targets the sweet spot between accuracy and QPU cost.

### Preflight Thresholds (updated 2026-06-14)

| Check | Abort threshold | Warning threshold | Rationale |
|-------|:-:|:-:|---|
| Mean 2Q error (chip-wide) | >5% | >3% | Large chips (150+q) have degraded outliers; layout selection avoids them |
| P5 T1 | <30 μs | — | Widespread decoherence |
| Min T1 | — | <50 μs | Isolated TLS defect (layout avoids) |
| Queue depth | — | >50 jobs | Execute during off-peak |

### SPSA Budget Warning
On real hardware with T_one_job ≈ 60s:
- SPSA trigger on ONE h-point = 200 iters × 2 evals × 60s = **400 min**
- Always use `--no-spsa` unless you specifically need refinement
- If MPNN predictions pass rehearsal (ΔE/gap < 5%), SPSA is unnecessary

---

## Hardware Module Integration with Runner Framework

### Runner Pattern for Hardware Scripts

All `scripts/run_hardware*.py` MUST use `HardwareValidationRunner`:

```python
from qmbp_simulation.framework.runner_base import HardwareValidationRunner, Section

class MyHWRunner(HardwareValidationRunner):
    runner_id = "hw_deploy_n10"
    experiment_id = "HW_DEPLOY"
    ...
```

### Automatic Validations (enforced by framework)

1. **Structural preflight** — runner_id, hypothesis, sections (from ValidationRunner).
2. **QPU preflight** — backend status, calibration, topology (from HardwareBackend).
3. **Cost ceiling check** — `shots × n_layouts ≤ max_total_shots` (in preflight.py).
4. **Circuit ZNE check** — 2Q gate count ≤ 18 (in run_deployment, before submission).
5. **Input validation** — params shape, gap>0, finite values (in run_deployment).
6. **Timeout handling** — job.result() respects job_timeout_s (in submission.py).

### CLI Arguments (HardwareValidationRunner)

```bash
--mode hardware|fake_backend    # Execution mode
--shots 16384                   # Shots per circuit
--n-layouts 3                   # Number of low-CES layouts
--n-qubits 10                   # System size
--topology heavy_hex            # Lattice topology
--zne-amplifier gate_folding|pea|adaptive  # ZNE noise amplification strategy
--zne-noise-factors 1 3 5      # Noise amplification factors
--zne-r2-threshold 0.90        # R² threshold for adaptive fallback
--no-mapomatic                  # Disable VF2 layout optimization (use BFS fallback)
--layout-strategy lowest_cost|ces_spread|hybrid  # Layout selection strategy
--run-preflight                 # Include Section 0 (HardwareBackend preflight)
--no-spsa                       # Disable SPSA refinement (deployment script)
--p-layers 1                    # HVA layers (rehearsal)
--h-test 4.0 3.25 3.0          # Override test h-values (rehearsal)
--h-train 4.5 4.25 ...         # Override training h-values (rehearsal)
--vqe-restarts 1               # VQE restart count (rehearsal)
--mpnn-epochs 6000             # MPNN training epochs (rehearsal)
--mpnn-hidden-dim 128          # MPNN hidden dimension (rehearsal)
```

**V3 MPNN Evaluation Suite — additional flags** (sections 10-19):

```bash
--skip-hardware-sections        # Run only MPNN sections (10-19), skip ZNE sections (1-9)
--skip-noisy-mpnn               # Skip section 14 (FakeTorino noisy eval)
--skip-extended-sections        # Skip sections 15-19 (scaling, LOO, topology transfer, κ)
--n-vqe-bench-restarts 5       # Random-init VQE runs to avg over (section 10)
--maxiter-refine 200            # Max VQE iters in benchmark (section 10)
--loo-min-train-size 5         # Min fold size for LOO-CV (section 11)
--noisy-shots 8192              # Shots for FakeTorino eval (section 14)
--noisy-n-layouts 3             # Layouts for FakeTorino eval (section 14)
--scaling-sizes 4 6 10          # System sizes for scaling benchmark (section 15)
--scaling-p-layers 2 2 1        # p_layers per size — CRITICAL: use p=1 for N≥10 (section 15)
--source-topology chain_1d      # Source topology for transfer experiment (section 17)
--loo-n-seeds 3                 # Seeds for multi-seed LOO robustness (section 18)
--noise-sigmas 0.01 0.05 0.1 0.2  # σ values for κ-noise correlation (section 19)
--h-kappa-grid 4.5 4.0 3.5 3.25 3.0 2.75  # Dedicated κ grid (section 19)
                                # Extend below training range toward h_c for best results
```

### κ-Based Hardware Risk Assessment (validated 2026-06-15)

The deployment script automatically computes `κ(h)` from the MPNN predictions
(noiseless, zero QPU cost) and logs a per-h risk profile before each tier.

```python
# In run_ibm_deployment.py — executed before every QPU submission:
kappa_per_h = compute_kappa_per_h(params_per_h, lattice)
recommendations = kappa_go_no_go(kappa_per_h, topology=TOPOLOGY, sigma_flow_per_h=sigma_flow_per_h)
# → per-h: risk_level, n_layouts, shots, spsa_recommended, sigma_flow_boost
```

### σ_flow Adaptive Resource Allocation (2026-06-17)

When `--sigma-flow-results <path>` is passed to the deployment script, the
`FlowWarmstartManager`'s uncertainty estimate σ_flow is combined with κ for
finer-grained shot/layout allocation:

```bash
# Full pipeline: rehearsal with flow → deployment with σ_flow
make hw-flow-full

# Or step by step:
make hw-flow-rehearsal           # produces sigma_flow_per_h in run_*.json
make hw-flow-deploy-dry          # loads σ_flow from latest rehearsal, dry-run
make hw-flow-deploy              # real QPU with σ_flow safety net
```

**Decision logic (per h-point):**

| Signal | Condition | Action |
|--------|-----------|--------|
| κ only | κ < P25 | HIGH: shots×2, 3 layouts |
| κ only | κ ∈ [P25, P75) | MEDIUM: 3 layouts |
| κ only | κ ≥ P75 | LOW: 1 layout |
| σ_flow boost | σ_flow > 0.5 (configurable) | Additionally: shots×2, layouts≥3 |

The σ_flow boost STACKS on top of κ-based allocation. A high-risk h-point with
high σ_flow gets shots×4 (κ doubles + σ doubles again).

### P2-C: Stale Calibration Comparison (2026-06-22)

Post-sweep diagnostic that compares pre-execution vs post-execution calibration
to tag results affected by drift during long runs (>1h). Implemented in:
- `HardwareBackend.run_h_sweep()` — logs `stale_calibration_comparison` event
- `run_mitigation_benchmark.py` — prints comparison after h-loop (only if elapsed > 30 min)

**Behavior:**
- Takes snapshot before first h-point and after last h-point
- Computes T1 drift %, gate error drift %, max single-qubit drift
- Logs warning if `is_stable=False`, but does NOT abort (diagnostic only)
- Saved in sweep summary JSON for post-hoc result filtering
- `HardwareRunResult` now has `stale_calibration_t1_drift_pct` and `stale_calibration_stable`

**When useful:** Runs >1h where TLS events can shift calibration mid-execution.
For typical 30-min runs, calibration doesn't change significantly.

**Consumed by:** `mitigation_benchmark_analyzer.py` can filter/flag results from
runs where `stale_calibration_stable=False`.

### P3: Adaptive Shot Budget in Mitigation Benchmark (2026-06-22)

κ-based shot adjustment in `run_mitigation_benchmark.py` when `--adaptive` is enabled.
Second-order optimization: inter-layout variance (0.25) >> intra-layout variance (0.16),
so redistributing shots between h-points has modest impact. Implemented conservatively:

**Decision logic:**
- HIGH κ risk → shots × 2 (near h_c, more noise sensitivity)
- MEDIUM κ risk → shots × 1 (unchanged)
- LOW κ risk → shots × 1 (unchanged — NOT reduced, to avoid introducing risk)

**Important constraints:**
- Only active with `--adaptive` CLI flag (NOT default)
- Envelope records `adaptive_shot_budget` metadata for post-hoc analysis
- Total budget is approximately neutral (HIGH points get 2× but are minority)
- Does NOT touch outlier detection (requires n_layouts≥4 for Grubbs, we use n_layouts=3)

**New fields in ResultEnvelope:**
- `adaptive_shot_budget.base_shots` — base shot count from CLI
- `adaptive_shot_budget.effective_shots` — actual shots used
- `adaptive_shot_budget.multiplier` — scaling factor applied
- `adaptive_shot_budget.risk_level` — κ classification that triggered boost
- `adaptive_shot_budget.reason` — "kappa_adaptive"

**New fields in `HardwareRunResult`:**
- `effective_shots: int | None` — actual shots used (None = base)
- `adaptive_shot_reason: str` — "kappa_high", "sigma_flow", or ""

**Result**: `sigma_flow_per_h` is saved in the tier_1 JSON, and every per-h
recommendation includes `"sigma_flow_boost": true|false` for auditability.

**Multi-seed training** (3 seeds): The flow model is trained 3 times with
seeds [42, 43, 44] and the best (lowest NLL) is kept. Checkpoint is saved
automatically at `results/flow_checkpoints/flow_{topology}_N{n}_p{p}.pt`.

**Findings (2026-06-17)**:
- heavy_hex N=10 p=1: σ_flow≈0.47 (below threshold → no boost by default)
- chain_1d N=6 p=2: σ_flow≈0.26 (well below → high confidence)
- The flow does NOT improve θ_init quality vs MPNN direct prediction
- Its value is purely as an uncertainty signal for resource allocation

**Key finding (Section 19, 2026-06-15):**

| Config | κ range | Anti-correlation |r| | κ reliable? |
|--------|---------|----------------------|------------|
| chain_1d N=6 p=2 | [41, 53] | **0.84** | ✅ Valid proxy |
| heavy_hex N=10 p=1 | [111, 174] | **0.52** | ❌ Weak (use V2 go/no-go) |

For `chain_1d`, low κ → h near h_c → HIGH noise sensitivity → more resources.
For `heavy_hex N=10 p=1`, κ thresholds are **auto-calibrated** via percentiles
(P25 = high-risk threshold, P75 = medium-risk threshold) because the absolute κ
scale differs from chain_1d. The `kappa_go_no_go()` function handles this
automatically when `topology` parameter is passed.

**Per-h results JSON now includes:** `kappa`, `hardware_risk` (high/medium/low),
`spsa_recommended` — saved for every tier for post-run analysis.

### Dual Persistence

Results are saved in TWO locations:
- `results/experiments/exp_{id}/run_{ts}.json` — digest/compare.py compatible
- `results/hardware/{runner_id}/run_{ts}/` — full provenance + input_params.json

### Key Rule: NEVER Skip Preflight for Real Hardware

- `--skip-preflight` is available for FakeTorino debugging only.
- For `--mode hardware`, preflight is MANDATORY — it prevents wasting IBM credits on misconfigured runs.
- If preflight aborts, fix the underlying issue. Do not bypass.


---

## Circuit Resource Estimation & Quality Validation (2026-06-17)

### ResourceEstimation Integration

Qiskit's `ResourceEstimation` pass is integrated into `_capture_transpiled_stats()`
in `HardwareBackend`. Every hardware execution now records per-layout:

```json
{
  "depth": 59, "depth_2q": 14,
  "n_2q_gates": 18, "n_1q_gates": 129, "total_gates": 147,
  "count_ops": {"cz": 18, "rz": 64, "sx": 57, "x": 8},
  "num_tensor_factors": 124, "width": 156, "active_qubits": 10
}
```

- **depth_2q** = critical path through 2Q gates only. Strongest predictor of hardware error.
- **count_ops** = per-gate-type breakdown for error budget calculation.
- **active_qubits** = `width - num_tensor_factors + 1`. Sanity check against routing expansion.

### Post-Transpilation Quality Check (NEW)

Call `validate_transpiled_circuit_quality()` AFTER layout selection, BEFORE QPU submission:

```python
from qmbp_simulation.execution.hardware.preflight import validate_transpiled_circuit_quality

quality = validate_transpiled_circuit_quality(
    transpiled_circuit, backend, layout=layout_qubits, logger=logger,
)
if quality["abort"]:
    raise RuntimeError(quality["abort_reason"])
```

Or from a `HardwareValidationRunner` section:
```python
quality = self.validate_transpiled_quality(transpiled, layout=layout_qubits)
```

### Error Budget Prediction (analysis module)

```python
from qmbp_simulation.analysis import compute_error_budget, build_error_prediction

# Pre-QPU fidelity estimate:
budget = compute_error_budget(transpiled, backend=backend, layout=layout)
# → {"error_budget": 0.095, "fidelity_estimate": 0.91, "source": "calibration"}

# Combined prediction (κ + error budget):
pred = build_error_prediction(transpiled, h_value=4.0, backend=backend, kappa=155.0)
# → {"predicted_risk": "low", "explanation": "..."}
```

### Layout Ranking by depth_2q

```python
from qmbp_simulation.analysis import select_best_layout_for_zne

best_idx, info = select_best_layout_for_zne(layout_selection.transpiled_circuits)
# → best_idx=0, info={"depth_2q": 14, ...}
```

Use the layout with lowest depth_2q as ZNE primary (less decoherence accumulation).

### Full Preflight Stack (chronological)

| Stage | Module | Checks | When |
|-------|--------|--------|------|
| Experiment preflight | `framework/preflight.py` | p≤2, regime, seeds, hardware_circuit_budget | Before any execution |
| QPU preflight | `execution/hardware/preflight.py` | topology, calibration, T1/T2, native gates | Before session |
| Circuit ZNE check | `execution/hardware/preflight.py` | n_2q vs ZNE threshold | After circuit build |
| **Transpiled quality** | `execution/hardware/preflight.py` | **depth_2q, error_budget, defective edges, active_qubits** | **After transpilation** |
| Post-execution | `analysis/circuit_visualizer.py` | prediction vs actual correlation | After QPU result |

### Hardware Backend Comparison (fake providers, snapshot data)

| Backend | Gen | Qubits | CZ error median | CZ gate time | T1 median | T2 median |
|---------|-----|:------:|:---------------:|:------------:|:---------:|:---------:|
| ibm_boston | Heron R3 | 156 | **0.13%** | 68ns | 292µs | **353µs** |
| ibm_aachen | Heron R2 | 156 | 0.15% | 68ns | 232µs | 255µs |
| ibm_pittsburgh | Heron R2 | 156 | 0.15% | 88ns | 299µs | 317µs |
| ibm_kingston | Heron R2 | 156 | 0.18% | 68ns | 283µs | 144µs |
| ibm_marrakesh | Heron R2 | 156 | 0.33% | 68ns | 197µs | 119µs |
| ibm_torino | Heron R1 | 133 | 0.41% | 68ns | 185µs | 141µs |
| Nighthawk | NH | 120 | ~0.1% | 68ns | 350µs | — |

**Current default**: `ibm_kingston` (Open Plan, free). For thesis: `ibm_boston` (paid, best quality).
