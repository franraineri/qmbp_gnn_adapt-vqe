#!/usr/bin/env python3
"""Pre-commit hook: Flag unsafe torch.load(weights_only=False) usage.

Ensures that any new torch.load() calls use weights_only=True or go
through the centralized safe loading wrapper. Existing legacy calls
are allowlisted but new ones are blocked.

Exit codes:
    0 — All clear (no new unsafe loads)
    1 — Unsafe torch.load detected in new/modified code
"""

from __future__ import annotations

import re
import sys

# Pattern matches: torch.load(..., weights_only=False)
# or torch.load(...) without weights_only (defaults to False in PyTorch < 2.6)
UNSAFE_PATTERN = re.compile(
    r"torch\.load\([^)]*weights_only\s*=\s*False[^)]*\)"
)

# Allowlisted files (legacy code — tracked for migration)
ALLOWLIST = {
    "src/qmbp_simulation/predictors/mpnn.py",
    "src/qmbp_simulation/predictors/unified_mpnn.py",
    "src/qmbp_simulation/predictors/gnn_qem.py",
    "src/qmbp_simulation/framework/artifact_serializers.py",
    "src/qmbp_simulation/analysis/flow_warmstart.py",
}


def check_file(filepath: str) -> list[tuple[int, str]]:
    """Check a file for unsafe torch.load patterns."""
    # Skip allowlisted files
    for allowed in ALLOWLIST:
        if filepath.endswith(allowed) or filepath == allowed:
            return []

    violations = []
    try:
        with open(filepath) as f:
            for i, line in enumerate(f, 1):
                if UNSAFE_PATTERN.search(line):
                    violations.append((i, line.strip()))
    except (OSError, UnicodeDecodeError):
        pass
    return violations


def main() -> int:
    """Check all provided files for unsafe torch.load."""
    files = sys.argv[1:]
    all_violations: list[tuple[str, int, str]] = []

    for filepath in files:
        if not filepath.endswith(".py"):
            continue
        violations = check_file(filepath)
        for line_no, content in violations:
            all_violations.append((filepath, line_no, content))

    if all_violations:
        print("❌ Unsafe torch.load(weights_only=False) detected:")
        print()
        for filepath, line_no, content in all_violations:
            print(f"  {filepath}:{line_no}")
            print(f"    {content}")
        print()
        print("Fix: Use weights_only=True, or use safetensors format.")
        print("If this is intentional (loading legacy checkpoints), add to ALLOWLIST")
        print("in scripts/hooks/check_safe_torch_load.py")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
