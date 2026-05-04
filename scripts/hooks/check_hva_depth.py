#!/usr/bin/env python3
"""Pre-commit hook: reject p_layers > 2 in Python source files.

Mele et al. (Nature Physics, 2026) show non-unital noise truncates
circuits to O(log n). HVA circuits MUST have p ≤ 2.
"""

import re
import sys

MAX_P = 2
# Patterns that indicate someone is setting p_layers to a literal > 2
PATTERNS = [
    re.compile(r"\bp_layers\s*[=:]\s*(\d+)"),
    re.compile(r"\bhva\.create\([^,]+,\s*(\d+)\s*,"),
    re.compile(r"HVACircuitBuilder\(\)\.create\([^,]+,\s*(\d+)\s*,"),
]


def check_file(path: str) -> list[str]:
    # Skip test files — they intentionally test constraint violations
    if "/test" in path or path.startswith("test"):
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
                        val = int(m.group(1))
                        if val > MAX_P:
                            violations.append(
                                f"{path}:{lineno}: p={val} exceeds MAX_P={MAX_P} "
                                f"(Mele et al. depth constraint)"
                            )
    except (OSError, UnicodeDecodeError):
        pass
    return violations


def main() -> int:
    errors = []
    for path in sys.argv[1:]:
        errors.extend(check_file(path))
    if errors:
        print("❌ HVA depth guard violations:")
        for e in errors:
            print(f"  {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
