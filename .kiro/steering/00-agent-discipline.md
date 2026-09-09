---
inclusion: always
---

# Agent Discipline (MANDATORY — top priority)

Three non-negotiable rules for every task in this repo. Violating any of them
has caused real bugs and wasted user time in past sessions.

## 1. Test commands before assuming their flags or arguments

NEVER hand the user a command whose flags/arguments you have not verified.
Assumed CLI flags have been wrong repeatedly (e.g. `--train-n` is a single int,
not a list; `--dqpt-*` flags that never existed; extra `--section` values).

Before giving ANY command:

- Run it with `--dry-run` if the runner supports it, or `--help`, or execute it
  for real when cheap.
- If a flag is rejected (`unrecognized arguments`, `error: ...`), read the
  runner's `parse_args` / `_add_custom_args` via graphify and correct it.
- Only present the command once it has passed a dry-run or you have read the
  exact argument definition. Prefer single-line commands to avoid shell
  line-continuation errors (trailing spaces after `\`).

## 2. Always prioritize graphify for context

graphify is the primary tool for understanding this codebase. grep/find/rg/
file_search remain BANNED (see `reuse-workflow`).

- Any question about code, a symbol, a call path, or "who uses X" → start with
  `graphify_explore` (Read-equivalent: its output IS the file read).
- Need a file skeleton → `graphify_inspect`; a symbol body → `graphify_source`;
  blast radius before editing a shared helper → `graphify_impact`.
- Treat graphify output as already-read: do not re-open those files.
- Only use `read_file` for JSON/config graphify does not index, exact known
  paths, or files graphify reports as not-found.

## 3. Always hunt for bugs after any new improvement or implementation

A change is not "done" when it compiles. After EVERY new implementation or
improvement:

- Run `py_compile` and check diagnostics on every edited file.
- Run the relevant tests; if failures appear, verify whether they are
  PRE-EXISTING (via `git stash`) before claiming a regression — and fix real
  regressions.
- Actively look for the failure modes the change introduces: dimension/shape
  mismatches, silent no-ops (a term/feature that never activates), missing
  persistence, unbounded caches, stores left inconsistent, and callers of the
  changed symbol that were not updated (`graphify_callers` / `graphify_impact`).
- Add a regression test for the specific bug class when the change touches a
  path that has broken before (e.g. graph/model feature-dim adaptivity).
- Regenerate the module index when new modules/symbols are added.

These three rules override convenience. When unsure, verify — do not assume.
