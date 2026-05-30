# Project Status — GNN-HVA Framework

## Experiment Discipline (ALWAYS ENFORCE)

Before running ANY experiment or notebook execution:
1. State the hypothesis being tested. No hypothesis → no execution.
2. Check binnacles and poc-results.md — if the result is already established, do NOT re-run.
3. Every run must produce new learning. "Confirming what we know" is not learning after 3 seeds.
4. Do not duplicate information across binnacles — reference existing entries, don't copy them.
5. Known physics limits cannot be tuned past. See `experiment-protocol.md` for the full list.

## Current Phase
V6.1 complete and thesis-ready. All features validated at N=6 and N=10 (15 definitive runs).
Pipeline observability (DiagnosticCollector) now always-on — every run captures full metrics.
**V7 experiments complete** — all simulation-testable questions answered (12/22 experiments run, 10 skipped with justification).
**V8 experiments complete** — landscape analysis, scaling laws, methodological validation (18/19 executed, 1 skipped with justification).

## V8 Key Results (2026-05-22/25)
- **Scaling law (A3)**: h_min = 1.0 + 0.020·N^1.31 (R²=1.0000). Predicts N=20→2.00 (exact match). Exponent ≠ ν=1.
- **p=1 vs p=2**: β(p=1)=0.60 < β(p=2)=1.33 → p=1 scales better at large N.
- **No barren plateaus (F3)**: Landscape fluctuation >1.0 everywhere (confirms Mele et al. 2026).
- **p=1 landscape simpler (F3@p=1)**: Lower fluctuation (1.38 vs 1.99), higher fraction_near_gs.
- **fraction_near_gs**: Novel training-free boundary predictor (0% at h<1.0, 5%+ at h>1.5).
- **Hessian (B4)**: ALL VQE minima are genuine (0 saddle points). 73% eval savings with 1 restart. ✅
- **Hessian N=10 (B4@N=10)**: Saddle-free confirmed at N=10. Condition numbers N-independent. ✅
- **Analytical init (B1)**: 97% fewer iterations but converges to wrong basin. Warm-start wins.
- **DyPP (F1)**: Only 8-13% iteration savings (hypothesis was 30-50%). Rejected. ❌
- **Weight-space phase detection (D1)**: MPNN-A peak at h≈0.7 (near h_c=1.0). Novel zero-QPU method. ✅
- **D1 regularized**: Dropout=0.1 makes detection robust (std=0.13 vs 0.90). 5 seeds all consistent. ✅
- **Parameter freezing (B2)**: 2/4 params frozen at h≥1.5, 0% accuracy loss. ✅
- **Longitudinal field (E4)**: HVA p=2 fails at g>0 (fidelity drops to 0.89 at g=0.1). HVA is model-specific. ❌
- **Data efficiency (G1)**: 9 points sufficient (47% reduction from 17). Seeds 43/44 pass with 5 points. ✅
- **Cross-seed (G5)**: Pipeline is seed-independent (std=0.004, all seeds pass). ✅
- **Condition number (G4)**: κ does NOT predict restart needs (r=-0.29). h-value is the real predictor. ❌
- **Ensemble UQ (G2)**: Naive ensemble variance not calibrated (r=0.195). Needs bootstrap/MC-Dropout. ❌
- **N=20 optimized (G3)**: 1 restart + freeze FAILS at N=20 (ΔE/gap=1.26). N=6 findings don't transfer. ❌
- **p=1 ZNE at N=10 (analysis 1A)**: CONFIRMED. Mean gain=+49% (9 runs, 3 topos × 3 seeds). CX-budget hypothesis validated. ✅
- Binnacles: `binnacle-v8-experiments.md`, `binnacle-v8-experiments-round1.md`, `binnacle-v8-round2-pipeline-characterization.md`

## V8 Validated Decisions (2026-05-22)
- **VQE restarts:** 1 restart sufficient at N=6 AND N=10 (B4: no saddle points in HVA landscape at either size)
- **Landscape is N-independent:** Condition numbers at N=10 match N=6 within 10% (B4@N=10)
- **VQE at N=20 p=1:** 3 restarts + 100 maxiter + MPS (chi=64) → ΔE/gap=1.58% (2/3 seeds). 5 restarts needed for full reliability (C3)
- **Sign canonicalization:** NOT needed — descending warm-start breaks Z₂ naturally (C3, 3 runs confirm 0% effect)
- **N=20 p=1 local minimum:** ΔE/gap=0.437 basin exists — requires ≥5 restarts to reliably escape (C3)
- **Parameter freezing:** Freeze θ_zz2, θ_x2 at h≥1.5 (B2: 0% accuracy loss)
- **Optimal VQE at h≥1.5:** 1 restart + 2 active params = **75% cost reduction**
- **DyPP:** Rejected — standard warm-start is near-optimal for 4-param HVA (F1)
- **Longitudinal field:** HVA p=2 is TFIM-specific, not model-agnostic (E4)
- **Phase detection:** Weight gradient peaks detect h_c when MPNN loss≈0.002 (D1, D1-dense)
- **Phase detection robustness:** Dropout=0.1 → std=0.13 (vs 0.90 without). Reliable across 5 seeds (D1-reg)
- **Physics-informed loss (C1):** +3.9% mean improvement (max +17.5% at h=1.75). Safe, no regression. Modest at N=6.
- **Physics-informed loss (C1@N=10):** -12.3% (HURTS). Only helps with full h-range training, not valid-regime-only.
- **Phase detection caveat:** Overfitting (loss=0) shifts peak to h≈0.69; needs regularization (D1-dense)
- **p=1 landscape (F3@p=1):** Simpler (fluctuation 1.38 vs 1.99), higher random GS accessibility at h≥1.5
- **Data efficiency (G1):** 9 points sufficient for ΔE/gap < 5% (47% reduction from 17-point baseline)
- **Cross-seed (G5):** Pipeline is seed-independent — all seeds produce ΔE/gap < 2.1% (std=0.004)
- **N=20 p=2 (G3):** 1 restart + freeze FAILS. N=6 landscape findings do NOT transfer to N=20. Use 7 restarts, no freeze.
- **Condition number (G4):** κ does NOT predict restart needs. Use h-value as difficulty proxy instead.
- **Ensemble UQ (G2):** Naive ensemble (same data, different init) not calibrated. Need bootstrap for real UQ.

## Active Priority
1. **Hardware deployment on IBM Torino** — the only way to validate ZNE at N=10 (local simulation exhausted).
2. Start with **N=6, h=1.5** (safest — ZNE works in simulation, expect it works on hardware).
3. Then **N=10, h=1.5** with full mitigation stack (DD + twirling + TREX + ZNE via EstimatorV2 options).
4. Use **SPSA (a=0.1, c=0.05, A=10)** for hardware VQE refinement — validated as 3× better than COBYLA under noise (V7 experiment 4C).
5. **Random baseline comparison now default** — every Phase 4 run compares warm-start vs cold-start (gain metric). Use `--no-baseline` to skip.
6. **Heisenberg model extension** — validate framework is model-agnostic (in progress).

## Critical Findings (2026-05-14/15/18)
Inhomogeneous ZNE (3 layouts) works at N=6 (R²>0.99, +40% gain) but **completely fails at N=10** (R²<0.05, negative gain). This is predicted by Tsubouchi et al. (2023): mitigation cost grows exp(depth × qubits).
- Experiment A: 7 layouts → R²=0.08 (still fails). Failure is fundamental, not statistical.
- Experiment B: DD cannot be tested locally (YGate not in FakeTorino basis). Must test on real hardware.
- **Conclusion: Local noisy simulation cannot validate ZNE at N=10. Go to real hardware where DD+twirling+TREX are native.**

### V7 Key Results (2026-05-18)
- **L-BFGS-B definitively optimal** for noiseless VQE (1A: wins by 31-95% over all Nevergrad methods)
- **SPSA optimal config: a=0.1, c=0.05, A=10** (4A: grid search over 36 configs × 10 seeds)
- **SPSA refinement HURTS warm-start** (4B: -146% at h=2.0) — don't refine good MPNN predictions
- **QRC = MPNN at N=10** (2B: <1% difference, both ceiling-limited) — predictor is NOT the bottleneck
- **MPS exact for 1D HVA** (3A/3B: |MPS-SV|=1e-14, chi=64 sufficient) — enables N=20 scaling
- **MPS VQE at N=20 passes at h=2.0** (3C: ΔE/gap≈1%) — valid regime shifts with N
- **Noise-aware training fails** under shot noise (5B: 6× worse) — only coherent errors could help
- **Iterative refinement modest** (5E: 9% gain, saturates in 2 rounds)

## Key Constraints (always enforce)
> Full constraint list with rationale in `.kiro/skills/quantum/SKILL.md`.
> Summary for quick reference:
- HVA only, never HEA. p ≤ 2 layers. |+⟩^N initial state.
- Descending sweep h=2→0. No angle wrapping.
- Pure energy cost in Phase 2. Never hybrid/observable cost.
- SparsePauliOp only. Primitives V2 only. Local observables on hardware.
- Fidelity filter ≥ 0.93 in Phase 3 training data.
- Hardware success criterion: ΔE/gap < 5% AND correct phase label (not fidelity).

## Stable Code (do NOT modify unless explicitly asked)
- `src/qmbp_simulation/models/` — data models, Hamiltonian builder, lattice construction, constants
- `src/qmbp_simulation/solvers/` — exact diag + DMRG paths
- `src/qmbp_simulation/circuits/` — HVA circuit construction (p≤2 enforced)
- `src/qmbp_simulation/execution/` — backend ABC + noiseless/noisy/hardware implementations
- `src/qmbp_simulation/optimizers/` — multi-start VQE + SPSA with warm-start
- `src/qmbp_simulation/pipeline/` — dataset save/load, pipeline orchestration
- `src/qmbp_simulation/utils/` — seed, JSON, timing utilities
- `scripts/smoke_test.py` — package smoke test (N=4, p=1, <30s)
- `Makefile` — unified entry point

## Active Development Areas
- `src/qmbp_simulation/predictors/mpnn.py` — MPNN architecture (per-parameter heads, edge features)
- `src/qmbp_simulation/analysis/gradient.py` — weight gradient analysis + phase detection
- `src/qmbp_simulation/analysis/diagnostics.py` — pipeline observability (DiagnosticCollector, always-on)
- `src/qmbp_simulation/analysis/metrics.py` — SNR, smoothness, energy decomposition
- `src/qmbp_simulation/framework/base.py` — BaseExperiment lifecycle (setup → run → analyze → report)
- `src/qmbp_simulation/framework/cli.py` — shared CLI argument groups and validation (includes `--seed`)
- `src/qmbp_simulation/framework/result_io.py` — standardized result saving/loading
- `src/qmbp_simulation/framework/benchmarking.py` — BenchmarkSuite for performance regression
- `src/qmbp_simulation/framework/logging.py` — StructuredLogger + ProgressReporter
- `src/qmbp_simulation/framework/result_store.py` — result querying, comparison, CATEGORY_MAP
- `src/qmbp_simulation/framework/variant_runner.py` — shared variant runner (PipelineVariant, RunResult, VariantRunner, run_variant_script)
- `src/qmbp_simulation/pipeline/runner.py` — PipelineRunner + run_exact_diag_sweep helper
- `experiments/` — categorized experiment scripts (optimization, scaling, landscape, predictor, hardware, generalization)
- `scripts/experiment_runners/run_experiment.py` — unified CLI for running experiments by ID (in experiment_run_helpers_CHECK/)
- `scripts/experiment_runners/experiment_run_helpers_CHECK/run_pipeline.py` — full 4-phase pipeline CLI
- `scripts/experiment_runners/run_thesis_variants-*.py` — topology-specific variant runners (chain_1d, ladder, triangular)
- `scripts/experiment_runners/run_p1_pipeline_variants.py` — p=1 multi-topology variants (R1)
- `scripts/experiment_runners/run_p1_pipeline_variants_r2.py` — p=1 corrected + complementary (R2)
- `scripts/compare.py` — cross-experiment result comparison (uses ResultStore)
- `scripts/benchmark.py` — performance benchmarking (uses BenchmarkSuite)
- `analysis/scan_coverage.py` — coverage scanner (inventario + gap analysis)
- `analysis/diagnose.py` — automated failure root cause analysis

## Dead Code (removed 2026-05-22, deleted 2026-05-25/26/27)
- `pipeline_core.py` — documented but zero imports anywhere (DELETED)
- `experimental/` — GATPredictor + augmentation (both rejected, never existed on disk)
- `hardware_deployer.py` — V6.0 legacy (DELETED, superseded by V6.1)
- `archive/` — All _BAK directories removed in v8_clean branch
- `scripts/run_v1_p1_noisy.py` — exploratory ZNE script (DELETED 2026-05-26, results in binnacles)
- `scripts/run_noisy_v2_batch.py` — exploratory ZNE batch (DELETED 2026-05-26, code in noisy_utils.py)
- Inline `RunResult`/`run_variant`/`main` in variant scripts — REMOVED 2026-05-27, replaced by `framework/variant_runner.py` (~650 lines eliminated)
- `scripts/run_v2_extended.py` — exploratory analysis (DELETED 2026-05-26, results in binnacles)
- `scripts/run_v2_nonlinear.py` — exploratory non-linear ZNE (DELETED 2026-05-26, results in binnacles)
- `scripts/run_v3_per_obs_zne.py` — exploratory per-obs ZNE (DELETED 2026-05-26, results in binnacles)
- `scripts/run_zne_robustness.py` — ZNE robustness validation (DELETED 2026-05-26, results in binnacles)

## Optimal Config (quick reference)
- **N=6**: GINConv h=64, L=3, 6000ep, lr=1e-3, 5 VQE restarts, fid≥0.93
- **N=10**: GINConv **h=128**, L=3, 6000ep, lr=1e-3, **patience=500**, **seed=43**
- **N=20 (MPS)**: chi=64 sufficient, L-BFGS-B + 3-5 restarts, descending warm-start, valid at h≥2.0
- **N=20 (full pipeline)**: h∈[1.5,2.0] only (11 pts), 7 restarts σ=0.3, NO filter, MPNN h=128, **ΔE/gap=1.75% ✅**
- **N=12**: Too slow for iterative experimentation on local hardware (~30+ min per run)
- **Hardware SPSA**: a=0.1, c=0.05, A=10, n_iterations=200 (from V7 4A grid search)
- **N=20 (p=1)**: 2 params, h∈[2.25,4.0], 5 restarts, StatevectorEstimator, MPNN h=128 (trivial mapping)

## V7 Validated Decisions (2026-05-18)
- **Optimizer (noiseless):** L-BFGS-B with 5 restarts. Nevergrad 31-95% worse (1A).
- **Optimizer (hardware):** SPSA (a=0.1, c=0.05). 3× better than COBYLA under noise (4C).
- **Warm-start refinement:** Do NOT apply SPSA after MPNN prediction in simulation (4B: hurts).
- **Predictor:** MPNN = QRC at N=10 (2B). Predictor is NOT the bottleneck. Keep MPNN for scalability.
- **MPS simulator:** Exact for 1D HVA (3A/3B). chi=64 sufficient. Enables N=20 VQE.
- **Noise-aware training:** Fails under shot noise (5B: 6× worse). Only coherent errors could help.
- **Iterative refinement:** Modest 9% gain, saturates in 2 rounds (5E). Not worth the complexity.
- **Valid regime scales with N:** N=6 h≥1.25, N=10 h≥1.5, N=20 h≥2.0 (HVA expressibility limit).

## p=1 Scaling Results (2026-05-21)
- **p=1 valid regime**: N=6 h≥1.6, N=10 h≥1.9, N=20 h≥2.25 (shift of +0.25 to +0.40 vs p=2)
- **Shift decreases with N**: +0.35 at N=6, +0.40 at N=10, +0.25 at N=20
- **Seed-independent at N≤10**: All 3 seeds give identical θ_opt (single global minimum)
- **N=20 has Z₂ symmetry issue**: Seeds find equivalent minima with different sign conventions
- **θ_x constant**: ±1.178 (= ±3π/8) for all h; only θ_zz varies → effectively 1D mapping
- **CX reduction**: Exactly 50% at all N (p=1 N=20 = 38 CX ≈ p=2 N=10 = 36 CX)
- **MPNN deployment at N=20**: Only h=3.0 passes (6 training points too few; sign canonicalization needed)
- **Hardware candidate**: p=1 N=20 on IBM Torino (VQE validated, same CX budget as p=2 N=10)
- **p=1 ZNE CONFIRMED (2026-05-28)**: 9 runs (3 topologies × 3 seeds), mean gain=+49%, topology-independent
- **p=1 pipeline CONFIRMED (2026-05-30)**: 9/9 PASS at N=10 (chain_1d 3/3, ladder 3/3, triangular 3/3) with correct h_test
- **p=1 vs p=2 direct comparison (COMP-4)**: p=1 more consistent (std=0.002 vs 0.47 for p=2)
- **N=16 p=1 scaling limit**: Phase 3 does not complete (fidelity filter rejects data). Needs MPS.
- **Failure diagnosis (2026-05-30)**: 45% CHAIN_BREAK, 25% MPNN_OVERFIT, 14% BOUNDARY_EFFECT. 70% detectable pre-Phase 4.
- **TODO**: Fix init at N=20 (analytical guess), canonicalize signs, increase training density
- Binnacle: `documentation/binnacles/binnacle-p1-scaling.md`
- Script: historical (removed in v8_clean)

## ZNE Scaling Rule (from experiments + literature)
- N=6: 3 layouts sufficient (R²>0.99, linear regime)
- N=10 p=2: 3 layouts fails (R²>0.95 but gain=-14.4%, non-perturbative regime).
- **N=10 p=1: 3 layouts WORKS (R²>0.99, gain=+49%, 9 runs confirmed cross-topology)**
- General: ZNE effectiveness governed by CX gate count (~18 threshold), not N alone.
- CX budget rule: p=1 N=10 ≈ p=2 N=6 ≈ 18 CX → ZNE works. p=2 N=10 ≈ 36 CX → ZNE fails.
- N=12 take very long time and resources, do not execute this experiments
- **p=1 + ZNE is the recommended strategy for hardware deployment at N≥10.**

## Where to Start
Read `.kiro/knowledge/project-guide.md` first.
The installable package is at `src/qmbp_simulation/`. Use `from qmbp_simulation import ...` for all imports.
See `README.md` for quick-start instructions and the full directory layout.

---

## How To: Create a New Experiment

1. **Choose a category**: `experiments/optimization/`, `experiments/scaling/`, `experiments/landscape/`, `experiments/predictor/`, `experiments/hardware/`, or `experiments/generalization/`
2. **Create the file**: `experiments/<category>/exp_<id>_<name>.py`
3. **Inherit from BaseExperiment**:
   ```python
   from qmbp_simulation.framework import BaseExperiment, ExperimentConfig, ExperimentMetrics

   class ExperimentX1(BaseExperiment):
       @classmethod
       def default_config(cls) -> ExperimentConfig:
           return ExperimentConfig(
               experiment_id="X1",
               name="My New Experiment",
               n_qubits=6, p_layers=2, j_coupling=1.0,
               h_values=[1.25, 1.5, 1.75, 2.0],
               seeds=[42, 43, 44],
           )

       def run_single(self, seed: int) -> list[ExperimentMetrics]:
           # Your experiment logic here
           ...
   ```
4. **Register** in `experiments/<category>/__init__.py`
5. **Run**: `python scripts/run_experiment.py --exp X1 --verbose`

## How To: Run the Pipeline

```bash
# Full 4-phase pipeline (N=6, p=2)
python scripts/experiment_runners/experiment_run_helpers_CHECK/run_pipeline.py --n-qubits 6 --p 2

# With custom MPNN config
python scripts/experiment_runners/experiment_run_helpers_CHECK/run_pipeline.py --n-qubits 10 --hidden-dim 128 --n-epochs 6000 --patience 500

# Skip phases (resume from checkpoint)
python scripts/experiment_runners/experiment_run_helpers_CHECK/run_pipeline.py --n-qubits 6 --skip-phase3 --skip-phase4

# p=1 multi-topology variants (9 runs, ~14 min)
python scripts/experiment_runners/run_p1_pipeline_variants.py

# p=1 corrected + complementary (13 runs, ~24 min)
python scripts/experiment_runners/run_p1_pipeline_variants_r2.py

# Smoke test (N=4, p=1, <30s)
python scripts/smoke_test.py

# Run tests
make test          # Fast tests (~12s)
make test-full     # All tests including slow (~60s)
```

## How To: Create a New Script

Use the framework's CLI utilities to avoid boilerplate:

```python
#!/usr/bin/env python3
"""My new script."""
from qmbp_simulation.framework import (
    create_base_parser, add_system_args, add_sweep_args, add_output_args,
    validate_descending_sweep, configure_logging, resolve_output_dir,
    save_experiment_result, build_result_envelope, ProgressReporter,
)

def main():
    parser = create_base_parser("My Script", epilog="Examples: ...")
    add_system_args(parser)
    add_sweep_args(parser)
    add_output_args(parser)
    args = parser.parse_args()

    configure_logging(verbose=args.verbose, debug=args.debug)
    h_values = validate_descending_sweep(args.h_values)
    output_dir = resolve_output_dir(args.output_dir)

    reporter = ProgressReporter("My Script")
    with reporter.phase(1, "Computing") as p:
        # ... do work ...
        p.detail("Done")
    reporter.summary({"key_metric": 0.014})

if __name__ == "__main__":
    main()
```

## How To: Compare Results

```bash
# Compare all experiments
python scripts/compare.py --all

# Compare by category
python scripts/compare.py --category optimization

# Analyze noisy/ZNE results
python scripts/compare.py --noisy --group-by n_layouts

# Programmatic access
from qmbp_simulation.framework import ResultStore
store = ResultStore()
comparisons = store.compare_experiments(store.list_experiments())
```

## How To: Benchmark Performance

```bash
# Run all benchmarks
python scripts/benchmark.py

# Specific components
python scripts/benchmark.py --components solver vqe --n-qubits 4 6 8 10

# Programmatic access
from qmbp_simulation.framework import BenchmarkSuite
suite = BenchmarkSuite(n_qubits=[4, 6, 8], n_repeats=5)
results = suite.run(components=["solver", "vqe"])
```

## How To: Add a New Technique to `experiments/helpers/`

1. **Create** `experiments/helpers/<technique_name>.py`
2. **Import from the package** (never from experiments or scripts):
   ```python
   from qmbp_simulation import HamiltonianBuilder, make_lattice, ClassicalSolver
   from qmbp_simulation.optimizers import VQEOptimizer
   from qmbp_simulation.execution import NoiselessBackend
   ```
3. **Implement** as a standalone function or class
4. **Export** from `experiments/helpers/__init__.py`
5. **Use** in experiments: `from experiments.helpers import <technique>`

## What NOT to Touch

- **Stable modules** (listed in "Stable Code" section above) — Only modify if explicitly asked.
- **`results/thesis/`** — Committed definitive results. Do not overwrite.
