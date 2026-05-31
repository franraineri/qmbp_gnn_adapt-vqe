# Knowledge Base Changelog

> Track when knowledge files were last updated to detect staleness.

| File | Last Updated | Trigger |
|------|-------------|---------|
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
