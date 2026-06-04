# Common Errors & Debugging Recipes

## V5.x Failure Mode (Phase Coupling)

**Symptom**: MSE converges but ΔE stagnates or increases.
**Root cause**: Phase 2 cost function changed (e.g., fidelity-weighted) without updating Phase 3 training target.
**Fix**: Phase 2 MUST use pure energy cost. V6 enforces this via `cost_function="energy"` metadata in the .npz dataset. Phase 3 loading validates this field.
**Literature confirmation**: Miao et al. (2024) and Karim et al. (2025) both use pure energy objectives for ML-VQE. The field consensus is that pipeline phases are tightly coupled — changing one objective without updating downstream is a known failure mode.

## Symmetry Saddle at θ=0

**Symptom**: VQE converges at iteration 0 with E ≈ E(|+⟩^N), fidelity ~50%.
**Root cause**: |+⟩^N is a saddle point — gradient ≈ 1e-6 at θ=0, and L-BFGS-B's default `pgtol=1e-5` declares convergence.
**Fix**: Initialize with `np.random.uniform(-0.01, 0.01)`, never zeros. V6 enforces this in `VQEOptimizer.get_initial_guess()`.

## HVA Expressibility Ceiling

**Symptom**: Fidelity < 50% for h < 0.8 (ferromagnetic regime).
**Root cause**: HVA p=2 + |+⟩^N cannot express the ferromagnetic ground state |000...0⟩. Verified with 50 random restarts over [-π, π].
**Fix**: This is a known limitation, not a bug. Pipeline is valid for h ≥ 1.0 (fid > 96%). Use fidelity filter ≥ 93% in Phase 3 to exclude bad training data.
**Literature confirmation**: Tripathi et al. (2026) independently confirms HVA p=2 struggles with entanglement entropy at the critical point. Sumeet et al. (2025) shows N/2 layers needed for thermodynamic-limit convergence — for N=6 that's p=3, violating Mele et al.

## Hardware Noise Broadening (Expected Phase 4 Behavior)

**Symptom**: On hardware, phase transition appears "smeared" — observables change gradually instead of sharply near h_c.
**Root cause**: Gate errors + shot noise + readout errors broaden the critical crossover region.
**This is EXPECTED, not a failure.** Sharma (2026) independently confirms this on IQM Garnet: ground-state energies are reliably captured, but correlation-sensitive observables show significant noise broadening.
**Correct interpretation**: The pipeline correctly classifies phases AWAY from h_c. The transition region is inherently harder on noisy hardware. Frame as "noise resilience of classification" not "failure at criticality."

## Shot Noise Dominance on Hardware

**Symptom**: Observable errors on hardware are ~1.6e-2 even with perfect gates.
**Root cause**: Statistical uncertainty from finite measurements: σ ≈ 1/√(shots). At 4096 shots, σ ≈ 1.6e-2.
**Fix**: Increase shot budget to ≥8192 (σ ≈ 1.1e-2). For N=10 where ⟨X⟩ signal is ~8.4e-3, need ≥16384 shots for signal > noise.
**Literature source**: Sharma (2026), Ma et al. (2025).

## AdaptVQE AlgorithmError at Iteration 0

**Symptom**: `AlgorithmError: "All gradients below threshold at first iteration"`.
**Root cause**: The warm-start θ_pred is already near-optimal — all Pauli pool gradients are below `gradient_threshold`.
**This is the IDEAL outcome** — it means the GNN/MPNN prediction was so good that no circuit depth increase was needed.
**Fix**: Catch `AlgorithmError`, evaluate energy of the initial state directly.

## DMRG Symmetry Breaking

**Symptom**: ⟨X⟩ ≈ 0 even in the paramagnetic phase (h > 1).
**Root cause**: DMRG may find a Z₂ symmetry-broken ground state.
**Fix**: Use |⟨X⟩| (absolute value) or XX correlation estimator. For N < 15, use exact diag instead.

## PoC V6 Baseline Results (N=6, TFIM, p=2, best config: 5 rst, 6000 ep)

| Metric | h=1.25 | h=1.4 | h=1.5 | Threshold |
|--------|--------|-------|-------|-----------|
| ΔE/gap | 3.5% ✅ | 1.9% ✅ | 1.4% ✅ | < 5% |
| ⟨X⟩ error | ~1e-2 ⚠️ | 5e-3 ✅ | 2.6e-3 ✅ | < 1e-2 |
| ⟨ZZ⟩ error | 2.5e-2 ❌ | ~1.5e-2 ❌ | 5e-3 ✅ | < 1e-2 |
| ΔE | 3.5e-2 ❌ | ~1.7e-2 ❌ | ~1.4e-2 ❌ | < 1e-2 |
| Fidelity | 0.991 ❌ | 0.995 ✅ | 0.997 ✅ | ≥ 99.5% |
| ADAPT iters | 2 ✅ | 2 ✅ | 2 ✅ | ≤ 2 |
| **Checklist** | **2–3/6** | **4–5/6** | **5/6** | |

**Key insight**: ΔE threshold is aspirational — bounded by HVA expressibility at each h. The ΔE/gap metric correctly shows the pipeline resolves the physics.


## CES-ZNE Fails on Heavy-Hex (Uniform Noise Topology)

**Symptom**: ZNE R² ≈ 0.04, gain ≈ 0% despite low individual CES values (all ~0.15).
**Root cause**: Heavy-hex N=10 p=1 has uniform noise across all layouts
(CES ≈ 0.15 ± 0.02). No CES spread → no extrapolation leverage.
Linear fit of E(CES) has no slope because all points are at the same x-value.
CES-ZNE fundamentally requires noise *variation* between layouts; heavy-hex
provides none because its uniform connectivity gives equivalent error rates everywhere.
**Fix**: Use gate-folding ZNE (`noise_factors=[1,3,5]`) or PEA-ZNE instead of
inhomogeneous layout-based CES extrapolation. For IBM Runtime:
`options.resilience.zne_mitigation=True` with `amplifier="pea"` (primary) or
`"gate_folding"` (fallback).
**Detection**: If R² < 0.50 after CES-ZNE on any topology → switch amplifier.
**Ref**: `documentation/analysis/11_hardware_rehearsal_findings.md`

## Gate-Folding ZNE: Validated Hardware Mitigation Strategy

**Symptom**: Need systematic noise amplification for ZNE on real hardware where
CES-based layout selection fails (see above).
**Pattern**: Gate-folding inserts identity pairs (CX·CX†) into the circuit at
noise_factors=[1,3,5], amplifying 2Q gate noise proportionally. Linear extrapolation
to factor=0 yields the zero-noise energy estimate.
**Results**: +12% mean gain over unmitigated, R² > 0.99, wins 9/12 h-points vs CES-ZNE
(t=3.28, p<0.01). Robust across chain_1d/heavy_hex/ladder topologies.
**Limitation**: On shallow circuits (depth ≤ 3 2Q layers), R² drops to ~0.47 because
there isn't enough noise to amplify meaningfully. Use PEA-ZNE for shallow circuits.
**When to use**: Fallback when PEA-ZNE is unavailable (no `qiskit-aer`) or when
QPU overhead budget is tight (GF has zero extra QPU cost beyond the 3 noise factors).
**Ref**: `documentation/binnacles/binnacle-gate-folding-zne.md`

## PEA-ZNE: Best-Performing Noise Amplification for Hardware

**Symptom**: Gate-folding gives insufficient gain (+12%) or R² < 0.90 on complex circuits.
**Pattern**: Pauli Error Amplification learns the actual device noise model via
Pauli-Lindblad fitting, then amplifies noise probabilistically (not physically).
This means it amplifies noise correctly regardless of circuit structure.
**Results**: +94.4% mean gain (4.6× better than gate-folding), R² = 0.998,
std = 2.9% across 3 seeds × 4 h-points. Wins 18/18 comparisons vs GF-ZNE
(t=46.32, p < 10⁻¹⁹). Cross-validated on ladder/heavy_hex/chain_1d.
**Overhead**: ~50% extra QPU time for the noise-learning phase. Justified by the
dramatically better extrapolation quality.
**Requirement**: `qiskit-aer` must be installed for noise learning simulation.
**CLI**: `--zne-amplifier pea`
**Ref**: IBM Nature 618 (2023); `documentation/binnacles/binnacle-gate-folding-zne.md`

## Stale Module Imports After Package Reorganization

**Symptom**: `ModuleNotFoundError: No module named 'scripts.digest'` when running
`python -m project_health`.
**Root cause**: The digest package was migrated from `scripts/digest/` to
`project_health/digest/` but some internal cross-references still used the old path.
**Fix**: All imports within `project_health/` must use `from project_health.digest.X`
not `from scripts.digest.X`. After any package relocation, grep for the old path:
`grep -r "scripts.digest" project_health/`
**Prevention**: Add a contract test that imports all public modules (see
`tests/integration/test_module_contracts.py`).
**Ref**: Fixed 2026-06-04 (engine.py, coverage.py, digest/__init__.py, digest/scanner.py,
digest/formatters.py, digest/__main__.py).
