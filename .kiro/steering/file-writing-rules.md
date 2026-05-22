---
inclusion: always
---

# File Writing Rules

## Large File Creation (ALWAYS ENFORCE)

When creating or writing files with content longer than 38 lines:

1. **First call**: Use `fsWrite` with only the first ~38 lines of content.
2. **Subsequent calls**: Use `fsAppend` to add the remaining content in chunks of ~40 lines each.
3. **Never** attempt to write more than 50 lines in a single `fsWrite` call.

## Editing Files

- Before using `strReplace`, confirm the file exists and contains the `oldStr` you expect.
- If a file creation failed or is incomplete, **delete it and start fresh** rather than trying to edit partial content.
- Never retry the same failing edit more than once. If it fails, diagnose why before trying again.

## Retry Discipline

- If a file write or edit fails, do NOT immediately retry with the same approach.
- Instead: check the file state (does it exist? what's in it?), then choose the correct tool.
- Maximum 2 retries on any file operation. After that, explain the issue to the user.

## Command Execution Rules

### CRITICAL: Never use `python -c` for multi-line code
- The `python -c "..."` pattern BREAKS when code contains quotes, f-strings, or newlines.
- Bash cannot parse multi-line Python inside quotes — it shows `dquote>` and hangs forever.
- **Instead**: Write the code to a temporary file, then execute it:
  1. `fsWrite` to `/tmp/kiro_snippet.py` (or similar)
  2. `executeBash` with `.venv/bin/python /tmp/kiro_snippet.py`
- This applies to ANY inline code longer than a single simple expression.
- One-liners are OK: `python -c "print(2+2)"` — but anything with imports or multiple statements → use a file.

### Long-running commands
- Any experiment, benchmark, or script that may run longer than 30 seconds: use `controlBashProcess` with action "start" instead of `executeBash`.
- NEVER re-run a command just because the output was truncated. Truncated output ≠ failure.
- If output is truncated, use `getProcessOutput` to read the tail, or check for output files.

### Large output commands
- For commands expected to produce large output, redirect to a file: `python script.py > output.log 2>&1`
- Then read the relevant parts of the log file with `readFile`.
- Do NOT re-run a command to "see the full output" — read the log instead.

### Retry limits for commands
- Maximum 2 retries on any shell command. After that, explain the error to the user.
- If the same error appears twice, diagnose the root cause — do not make incremental tweaks.
- NEVER retry a command that timed out with the exact same parameters.
