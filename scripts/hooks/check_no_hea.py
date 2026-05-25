#!/usr/bin/env python3
"""Pre-commit hook: reject HEA circuit usage in Python source files.

This project uses HVA (Hamiltonian Variational Ansatz) exclusively.
HEA circuits (EfficientSU2, TwoLocal, RealAmplitudes, NLocal) are
forbidden — they lack the physics-informed structure required for
topological phase characterization.
"""

import re
import sys

# HEA circuit classes from Qiskit that must never appear
HEA_PATTERNS = [
    (re.compile(r"\bEfficientSU2\b"), "EfficientSU2"),
    (re.compile(r"\bTwoLocal\b"), "TwoLocal"),
    (re.compile(r"\bRealAmplitudes\b"), "RealAmplitudes"),
    (re.compile(r"\bNLocal\b"), "NLocal"),
    (re.compile(r"\bExcitationPreserving\b"), "ExcitationPreserving"),
    (re.compile(r"\bPauliTwoDesign\b"), "PauliTwoDesign"),
]


def check_file(path: str) -> list[str]:
    """Check a single file for HEA usage."""
    # Skip test files and hook scripts themselves
    if "/tests/" in path or "/test_" in path or path.startswith("tests/"):
        return []
    if "/hooks/" in path or path.startswith("scripts/hooks/"):
        return []
    # Skip archived code — historical, not subject to current constraints
    if "/archive/" in path or path.startswith("archive/"):
        return []
    violations = []
    try:
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                stripped = line.lstrip()
                # Skip comments and noqa lines
                if stripped.startswith("#") or "# noqa" in line or "# hea-ok" in line:
                    continue
                for pat, name in HEA_PATTERNS:
                    if pat.search(line):
                        violations.append(
                            f"{path}:{lineno}: uses {name} (HEA forbidden — use HVA only)"
                        )
    except (OSError, UnicodeDecodeError):
        pass
    return violations


def main() -> int:
    errors = []
    for path in sys.argv[1:]:
        errors.extend(check_file(path))
    if errors:
        print("❌ HEA guard violations (HVA only, never HEA):")
        for e in errors:
            print(f"  {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
