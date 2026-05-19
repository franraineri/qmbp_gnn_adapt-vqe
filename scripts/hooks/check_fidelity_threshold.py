#!/usr/bin/env python3
"""Pre-commit hook: reject fidelity thresholds below 0.93.

The GNN-HVA pipeline requires fidelity_threshold ≥ 0.93 for Phase 3
training data filtering. Lower thresholds contaminate the MPNN training
set with poor VQE solutions, degrading prediction quality.
"""

import re
import sys

MIN_FIDELITY = 0.93

# Patterns that set fidelity thresholds
PATTERNS = [
    re.compile(r"fidelity_threshold\s*[=:]\s*([\d.]+)"),
    re.compile(r"fid_threshold\s*[=:]\s*([\d.]+)"),
    re.compile(r"min_fidelity\s*[=:]\s*([\d.]+)"),
]


def check_file(path: str) -> list[str]:
    """Check a single file for fidelity threshold violations."""
    # Skip test files and hook scripts themselves
    if "/tests/" in path or "/test_" in path or path.startswith("tests/"):
        return []
    if "/hooks/" in path or path.startswith("scripts/hooks/"):
        return []
    # Skip experiment scripts — they intentionally use varied thresholds for comparison
    if "/experiments_hamed_v7/" in path or "scripts/experiments" in path:
        return []
    violations = []
    try:
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                stripped = line.lstrip()
                if stripped.startswith("#") or "# noqa" in line:
                    continue
                for pat in PATTERNS:
                    for m in pat.finditer(line):
                        try:
                            val = float(m.group(1))
                        except ValueError:
                            continue
                        if val < MIN_FIDELITY:
                            violations.append(
                                f"{path}:{lineno}: fidelity threshold={val} < {MIN_FIDELITY} "
                                f"(minimum required for Phase 3 data quality)"
                            )
    except (OSError, UnicodeDecodeError):
        pass
    return violations


def main() -> int:
    errors = []
    for path in sys.argv[1:]:
        errors.extend(check_file(path))
    if errors:
        print("❌ Fidelity threshold violations (minimum ≥ 0.93):")
        for e in errors:
            print(f"  {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
