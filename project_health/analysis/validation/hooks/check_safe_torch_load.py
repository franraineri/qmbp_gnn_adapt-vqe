#!/usr/bin/env python3
"""Pre-commit hook: flag unsafe torch.load(weights_only=False).

Allows the pattern when annotated with `# nosec: trusted checkpoint`
on the same line (for our own checkpoint format that requires pickle).

Exit 0 = pass, Exit 1 = violations found.
"""
import sys
import re

UNSAFE_PATTERN = re.compile(r"torch\.load\(.*weights_only\s*=\s*False")
SUPPRESS_COMMENT = "# nosec"


def check_file(path: str) -> list[str]:
    violations = []
    try:
        with open(path) as f:
            for i, line in enumerate(f, 1):
                if UNSAFE_PATTERN.search(line) and SUPPRESS_COMMENT not in line:
                    violations.append(f"  {path}:{i}: {line.strip()}")
    except (OSError, UnicodeDecodeError):
        pass
    return violations


def main():
    files = sys.argv[1:]
    all_violations = []
    for f in files:
        if f.endswith(".py"):
            all_violations.extend(check_file(f))

    if all_violations:
        print("❌ Unsafe torch.load(weights_only=False) found:")
        for v in all_violations:
            print(v)
        print()
        print("Fix: use weights_only=True, or add '# nosec: trusted checkpoint' to suppress.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
