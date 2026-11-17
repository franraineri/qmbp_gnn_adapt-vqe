# Common Errors & Debugging Recipes

## DMRG TFIChain E_exact Mismatch for Non-1D Topologies (N>15)

**Symptom**: VQE energy E_vqe < E_exact by 1e-3 to 1e-2 on heavy_hex or ladder at N>15. Triggers variational violation warnings. Pattern: violations appear at lower h values and increase monotonically as h decreases.
**Root cause**: `ClassicalSolver._solve_dmrg_1d()` uses TeNPy's `TFIChain` model which assumes sequential nearest-neighbor coupling (site i → site i+1). For heavy_hex/ladder, the actual Hamiltonian has non-sequential bonds. DMRG computes E₀ of the WRONG Hamiltonian (1D chain approximation), giving a result ~5-10 mEh higher than the true ground state.
**Verification**: Compute `eigsh(H_actual.to_matrix(sparse=True), k=2)` for the REAL Hamiltonian and compare with the stored E_exact. The VQE energy should be ABOVE the true E_exact (variational principle satisfied).
**Impact**: 57 runs affected (heavy_hex N=16-40). All these runs have artificially GOOD ΔE/gap metrics (because E_exact is too high). The actual VQE performance is slightly WORSE than reported.
**NOT affected**: chain_1d (TFIChain is exact), N≤15 any topology (uses eigsh on actual H), square/triangular (use dedicated 2D DMRG).
**Workaround** (current): Auto-abort threshold requires violation > 1e-2 to abort. `verify_thesis_runs.py` uses loose threshold (1e-2) for heavy_hex/ladder N>15. `e_exact_method` field in per-point data documents which method was used.
**Proper fix** (TODO): Use eigsh on actual SparsePauliOp for N=16 (64K×64K sparse, tractable). For N=20+ where eigsh is too slow, implement topology-aware DMRG that maps the actual lattice graph to the TeNPy model (not just TFIChain).
**Ref**: Discovered 2026-07-13 during simulation_diagnostics implementation.

## DMRG Drops NNN/J2 Couplings for Frustrated Models (N>16)

**Symptom**: `ClassicalSolver.solve()` returns a ground energy that is far too LOW for `tfim_frustrated` at N>16 (e.g. square N=18, h=1.0, J2=0.5: -30.2355 instead of -23.0280). No error or warning.
**Root cause**: The lattice-based DMRG paths (`_solve_dmrg_2d`, `_solve_dmrg_graph`, `_solve_dmrg_1d`) rebuild the TeNPy model from `lattice.edges` + scalar J/h and never read the `SparsePauliOp`. NNN bonds are generated inside `HamiltonianBuilder.build_frustrated_tfim` and are not in `lattice.edges`, so DMRG solves the UNFRUSTRATED TFIM.
**Verification**: Run operator-based DMRG with `J2=0.0` and `J2=0.5` on the same lattice. The solver's value equals the `J2=0.0` result. At N≤16 compare against `solve(..., method="exact")`.
**Impact**: N≤16 was never wrong (exact diag, plus the DMRG-vs-eigsh guard falls back to exact). At N>16 only frustrated/NNN models were affected. Audit at closure: 0 frustrated entries with N>16 in the GT cache, 0 in the training NPZs, and every VL `|ΔE|` used exact diag (N≤10) or the operator DMRG (N=18). No stored result was contaminated.
**Fix** (applied, commit `6cf7465`): `solve(..., dmrg_from_operator=None)`. `None` auto-selects the operator-based MPO (`_solve_dmrg_from_operator`, built from the `SparsePauliOp` via `MPOGraph.from_term_list`) when the operator has more ZZ terms than lattice edges; `True` forces it; `False` keeps the legacy path and logs a warning if couplings would be dropped. Regression tests: `tests/unit/test_guards_and_validations.py::TestDMRGCrossValidation`.
**Open risk**: for N>18 the solver gap falls back to `_estimate_tfim_gap_analytical`, which ignores frustration. Not validated.
**Ref**: Found 2026-09-24 while validating `mps_loading` at N=18. See `internal/documentation/binnacles/binnacle-vl-haiqu-square-frustrated.md`.

## Haiqu Job Stuck at RUNNING (Client Poll Budget = Compute Budget)

**Symptom**: `run_vl_mps_job` / `run_vl_job` returns `fidelity=None`, status `RUNNING`, `circuit_id=None` for long jobs (e.g. `fine_tuning_iterations=200`), although the job completes in the cloud.
**Root cause**: the client poll timeout (900 s) equalled Haiqu's compute `max_time` (900 s), so polling ended before the cloud reported DONE.
**Fix**: pass `max_time` explicitly to the job and poll for `max_time + 300 s` (default in `run_vl_mps_job`).
**Ref**: 2026-09-25, `scripts/analysis/vl_vs_hva/hva_vl_study_common.py`.

## Haiqu HTTP 402 Payment Required (Credits Exhausted)

**Symptom**: every Haiqu call fails with `HTTPError: 402 Client Error: Payment Required`; result rows contain only `error`.
**Root cause**: the account ran out of credits after dozens of VL jobs (quality sweep, re-runs, N=18 runs). Not a code bug.
**Handling**: the batch scripts catch per-point errors and persist them, so nothing is lost. Mark the error-only result file `deprecated` (see `vl_mps_n18/vl_mps_square_N18_L8_F200_tc1e-08_standard.json`) and re-run after a top-up. Prefer the result cache (`vl_quality_sweep_frustrated.py`) so completed configs never re-spend credits.
**Ref**: 2026-09-25.

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

## Symmetry-Group Order Capped at 17 on PBC Clusters
**Symptom**: `detect_symmetry_group` reports `order == 17` for every tilted cluster (N=8, 10, 16, 18, 20), and the tying reduction looks too small.
**Root cause**: the default `max_syms=16` stops the enumeration after 16 non-identity permutations. The true groups are translations x point group (128 at N=16, 144 at N=18, 72 at N=9). A prime order is the tell: a permutation group of this kind cannot have order 17.
**Fix**: pass `max_syms=100000`. Orbit counts were not affected (one orbit per block at N=9/16/18); only the reported order and any code that reads it.
**Prevention**: `tests/unit/test_tilted_square.py::TestSymmetry` pins both the full orders and the capped value 17.
**Ref**: 2026-11-16, `docs/research/symmetry_frustrated_vqe_findings.md` section 10.1.

## Manifold Cache Key Collision Between OBC and Tilted PBC
**Symptom**: none visible. A tilted run silently measures fidelity against the wrong ground state.
**Root cause**: the cache key is `(topology, N, model, h, J2)`. The OBC square at N=16 is degenerate (md=2, e0=-16.25) while the tilted N=16 cluster is not (md=1, e0=-17.13). With `topology="square"` both map to the same file.
**Fix**: tilted runners use `topology="square_tilted"` (files `data/manifold_cache/square_tilted_*`).
**Prevention**: any new lattice variant needs its own topology key before it touches `cached_degenerate_ground_manifold`.
**Ref**: 2026-11-16.

## Wrong Angles Loaded by JSON Name Collision
**Symptom**: a drop-nnn baseline that should start at fidelity 0.795 starts at 0.100 and ends at 0.5565; every variant is then compared against a wrong reference.
**Root cause**: `_load_src_tied_theta(N, h, p)` reads `tilted_pareto_2q_N{N}`. At N=16 that file holds the failed calibrated sweep, whose p=4 row also has 13 parameters, so the length check passes. The intended angles are in `tilted_transfer_N9to16`.
**Fix**: `run_tilted_drop_nnn.py` reads the transfer JSON and aborts if the p4_full seed starts below 0.75.
**Prevention**: validate a loaded seed by its measured fidelity, not only by its length.
**Ref**: 2026-11-16, `docs/research/symmetry_frustrated_vqe_findings.md` section 10.6.

## Energy Objective Degrades a Good Seed at Large N (Tilted Tied Ansatz)
**Symptom**: starting from a seed with fidelity 0.62 at N=16, `converge_circuit` lowers the energy and ends at 0.37; the energy gap to the ground state shrinks while fidelity falls.
**Root cause**: not isolated. Observed: with a non-degenerate ground state (gap 0.038) the lowest-energy point reachable by the 13-parameter tied ansatz is not the highest-fidelity point. Treat any mechanism as a hypothesis.
**Fix**: optimize `1 - fidelity` directly (`optimize_fidelity` in `run_tilted_transfer.py`): 0.62 to 0.7951.
**Caveat**: this needs the exact ground state, so it is a test of expressivity, not a hardware method.
**Ref**: 2026-11-16, findings note section 10.5.

## JSON Study Outputs Are Overwritten, Not Appended
**Symptom**: a re-run loses rows from earlier runs (for example `variant_topk_*.json`, `tilted_transfer_N9to16_h0.50.json` keeps only the last invocation).
**Root cause**: `StudyPersister.persist` writes the full `rows` list built in the current process via `save_json`.
**Prevention**: pass every tolerance, depth or variant in a single invocation, or copy the log of the earlier run; the p=3 N=16 fidelity (0.6707) survives only in `results/hva_vl_study/poolA_logs/OVERNIGHT_tilted_N16_fidelity.log`.
**Ref**: 2026-11-16.
