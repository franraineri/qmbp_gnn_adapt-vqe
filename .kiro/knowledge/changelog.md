# Knowledge Base Changelog

> Track when knowledge files were last updated to detect staleness.

| File | Last Updated | Trigger |
|------|-------------|---------|
| changelog.md | 2026-11-16 | Temporary closure of the lattice-symmetry / tilted-PBC ansatz workstream. Binnacle: `internal/documentation/binnacles/binnacle-session-20261116-symmetry-tilted.md`. Handoff: `docs/research/HANDOFF_symmetry_tilted.md`. Findings: `docs/research/symmetry_frustrated_vqe_findings.md` (sections 9-11). |
| validated-decisions.md | 2026-11-16 | NEW section "Symmetry-tied ansatz and tilted PBC clusters" (exact group/degeneracy facts, N=9 and N=16 tilted results, N=20 OBC closure). Single-run results are tagged [PROJECTED]. |
| error-patterns.md | 2026-11-16 | NEW patterns (5): symmetry-group order capped at 17 on PBC clusters; manifold cache key collision OBC vs tilted; wrong angles loaded by JSON name collision; energy objective degrades a good seed at large N; JSON study outputs are overwritten, not appended. |
| next_steps.md | 2026-11-16 | Added A0-sym: open items of the symmetry / tilted-PBC front (p=5/6 at N=16, drop-nnn confirmation, N=20 cluster repeat, move `optimize_fidelity` to `src/`). |
| changelog.md | 2026-11-16 | Temporary closure of the Haiqu VL study on the frustrated square TFIM (work 2026-09-22 → 09-25, commits `6cf7465`, `6e41376`). Handoff doc: `internal/documentation/binnacles/binnacle-vl-haiqu-square-frustrated.md`. Report: `results/hva_vl_study/reports/REPORT_VL_square_frustrated.md`. |
| error-patterns.md | 2026-11-16 | NEW patterns: ClassicalSolver graph/2D DMRG drops NNN/J2 at N>16 (fixed, `dmrg_from_operator`); Haiqu poll timeout == compute budget leaves jobs `RUNNING`; Haiqu HTTP 402 credit exhaustion. |
| validated-decisions.md | 2026-11-16 | "Haiqu Data Loading" section: added MEASURED results (VL fidelity is gap-limited; L8/F50 recovers N=9; N=18 via `mps_loading` F=0.79–0.86; χ=64 sufficient). Replaced the "key unset" plan with status + pending runs. |
| haiqu-integration.md | 2026-11-16 | Added "Data loading (vector_loading / mps_loading)" section: tooling, cost/credit and timeout rules. |
| next_steps.md | 2026-11-16 | Added VL study open items (F200/Vidal blocked by credits, triangular, repeats, tech debt). |
| solvers/classical.py (code) | 2026-09-26 | `solve(..., dmrg_from_operator=None)`: operator-based DMRG MPO with auto-detection of extra couplings. 4 regression tests in `test_guards_and_validations.py`. |
| changelog.md | 2026-06-03 | E4c full pipeline PASS (2/2 test points, ΔE/gap=0.007). MPNN with extra_node_features=[J₂] works. 15 training points required (8 insufficient). Digest shows 13 confirmed, 5 rejected, 5 failed across 23 experiments. |
| changelog.md | 2026-06-03 | E4b+E4c standard execution: E4c ✅ 96% pass ΔE/gap=0.009 (J₂≤0.7), E4b ✅ fid≥0.98 (g≤0.5). Both in digest. E4b analyze() fixed to include `analysis.summary`. Next: MPNN with J₂ feature for full pipeline. |
| changelog.md | 2026-06-02 | Scalability refactoring: (1) auto-preflight in BaseExperiment.execute(), (2) centralized CLI args (add_result_filter_args, add_format_args, add_variant_runner_args), (3) run_vqe_sweep helper in BaseExperiment, (4) EXPERIMENT_CRITERIA unified in criteria.py, (5) json_serialize unified — all _json_default implementations delegate to utils/helpers.py |
| changelog.md | 2026-06-02 | Frustrated TFIM (J1-J2) implemented: `build_frustrated_tfim()`, `create_frustrated_tfim()`, registered as `tfim_frustrated`. CX budget 27@N=6 (noiseless only). See `binnacle-hamiltonian-candidates.md` |
| changelog.md | 2026-06-02 | Kitaev chain verified NOT viable (20 CZ N=6, fid=16%). TFIM+longitudinal confirmed as only viable extension. See `binnacle-hamiltonian-candidates.md` |
| changelog.md | 2026-06-02 | TFIM+longitudinal extension: HVA extended (ZZ+X+Z), ModelSpec.with_params(), digest --model filter, E4b experiment |
| changelog.md | 2026-06-01 | Heisenberg V9 experiments: 30 variants executed, definitive negative result documented in binnacle + project-status |
| changelog.md | 2026-05-31 | Heisenberg XXZ extension: ModelSpec, ModelRegistry, EntanglementAnalyzer, PipelineRunner model-aware dispatch |
| changelog.md | 2026-05-30 | Verification R1 results: p=1 valid regime corrected, new tools documented |
| poc-results.md | 2026-05-30 | Added p=1 valid regime per topology (corrected ladder N=10 → h≥3.0, triangular N=6 → h≥4.0) |
| project-guide.md | 2026-05-14 | Restructuring: added pipeline_core.py, experimental/, updated repo map |
| optimization-hardware.md | 2026-05-18 | Added V7 SPSA optimal config, warm-start rule, optimizer evidence table |
| validation-targets.md | 2026-05-14 | NEW: split from project-status.md (historical data + tables) |
| literature-synthesis.md | 2026-05-08 | Added Slavin 2025, updated Section 9 (V6.1 lessons) |
| error-patterns.md | 2026-05-04 | Added shot noise dominance pattern |
| gnn-architecture.md | 2026-05-06 | Added NNConv, capacity scaling rule, transferability limits |
| physics-reference.md | 2026-05-06 | Added Kagome 103-site reference, quantum advantage boundary |
| workflow-recipes.md | 2026-05-14 | Added pipeline_core patterns, experimental imports |

## Confidence Level Convention

Knowledge claims use these tags:
- **[VERIFIED]** — experimentally confirmed with ≥3 seeds or ≥3 independent experiments
- **[PROJECTED]** — extrapolated from trends or single-seed results
- **[LITERATURE]** — cited from papers but not reproduced in our pipeline

## Update Protocol

When modifying source code that changes behavior documented in knowledge:
1. Update the relevant knowledge file
2. Update this changelog with date and trigger
3. If the change affects numerical baselines, re-run `make test` first
4. If the change affects the repository map, update `project-guide.md`
5. Propagate binnacle findings to knowledge files within 48h of experiment completion
