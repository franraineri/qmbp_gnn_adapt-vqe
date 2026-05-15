# Knowledge Base Changelog

> Track when knowledge files were last updated to detect staleness.

| File | Last Updated | Trigger |
|------|-------------|---------|
| project-guide.md | 2026-05-14 | Restructuring: added pipeline_core.py, experimental/, updated repo map |
| validation-targets.md | 2026-05-14 | NEW: split from project-status.md (historical data + tables) |
| literature-synthesis.md | 2026-05-08 | Added Slavin 2025, updated Section 9 (V6.1 lessons) |
| error-patterns.md | 2026-05-04 | Added shot noise dominance pattern |
| gnn-architecture.md | 2026-05-06 | Added NNConv, capacity scaling rule, transferability limits |
| optimization-hardware.md | 2026-05-08 | Added Heron r2 comparison, QESEM, shot noise table |
| physics-reference.md | 2026-05-06 | Added Kagome 103-site reference, quantum advantage boundary |
| poc-results.md | 2026-05-05 | Added N=10 results section |
| workflow-recipes.md | 2026-05-14 | Added pipeline_core patterns, experimental imports |
| changelog.md | 2026-05-14 | NEW: created for maintenance hygiene |

## Update Protocol

When modifying source code that changes behavior documented in knowledge:
1. Update the relevant knowledge file
2. Update this changelog with date and trigger
3. If the change affects numerical baselines, re-run `make test` first
4. If the change affects the repository map, update `project-guide.md`
