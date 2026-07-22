---
inclusion: always
---

# Reuse-First Workflow (MANDATORY)

Every time new Python code is about to be created (script, helper, module, plotter, analyzer):

## Step A — Check Index
1. Read `.kiro/steering/module-index.md`
2. Search for existing modules with similar functionality by:
   - Matching class/function names
   - Matching docstring descriptions
   - Matching category (LIB, SCRIPT, HEALTH, EXP, etc.)
3. If a match exists → prefer extending it (add function, subclass, new flag)

## Step B — Implement with Reuse
- Import from existing modules rather than copy-pasting code
- If extending: add to the same file or create a minimal subclass
- If truly novel: create new file following existing patterns in same category

## Step C — Update Index
- After implementation, run: `python scripts/maintenance/generate_module_index.py`
- This is automated via the `refresh-module-index` hook on fileCreated events

## Step D — Verify Integration
- Run relevant tests or at minimum `python -c "import <new_module>"`
- Check no circular imports introduced

## Anti-patterns (NEVER do these)
- Creating a new script when an existing one accepts `--flags` to do the same
- Duplicating utility functions that exist in `qmbp_simulation.utils`
- Creating analysis helpers that replicate `project_health/analysis/` functionality
- Writing JSON serialization logic (use `json_serialize` from utils)
- Duplicating experiment criteria (use `framework/criteria.py`)
