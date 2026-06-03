# Knowledge Base Changelog

> Track when knowledge files were last updated to detect staleness.

| File | Last Updated | Trigger |
|------|-------------|---------|
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
