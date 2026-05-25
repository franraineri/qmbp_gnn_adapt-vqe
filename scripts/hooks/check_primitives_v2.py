#!/usr/bin/env python3
"""Pre-commit hook: reject deprecated Qiskit Primitives V1 usage.

The GNN-HVA framework requires Primitives V2 exclusively:
- StatevectorEstimator (simulation) or EstimatorV2 (hardware)
- StatevectorSampler (simulation) or SamplerV2 (hardware)

Deprecated V1 patterns: Estimator (bare), qiskit.primitives.Estimator,
BackendEstimator, etc.
"""

import re
import sys

# Deprecated V1 imports and usages
DEPRECATED_PATTERNS = [
    # V1 Estimator imports (not StatevectorEstimator or EstimatorV2)
    (
        re.compile(
            r"from\s+qiskit\.primitives\s+import\s+.*\bEstimator\b"
            r"(?!V2|.*StatevectorEstimator)"
        ),
        "qiskit.primitives.Estimator (V1) — use StatevectorEstimator or EstimatorV2",
    ),
    (
        re.compile(r"\bBackendEstimator\b"),
        "BackendEstimator (V1) — use EstimatorV2 with backend",
    ),
    (
        re.compile(r"\bBackendSampler\b"),
        "BackendSampler (V1) — use SamplerV2 with backend",
    ),
    # qiskit.execute (fully deprecated)
    (
        re.compile(r"from\s+qiskit\s+import\s+.*\bexecute\b"),
        "qiskit.execute (deprecated) — use Primitives V2",
    ),
    (
        re.compile(r"\bqiskit\.execute\b"),
        "qiskit.execute (deprecated) — use Primitives V2",
    ),
    # QuantumInstance (deprecated)
    (
        re.compile(r"\bQuantumInstance\b"),
        "QuantumInstance (deprecated) — use Primitives V2",
    ),
    # qiskit.opflow (deprecated)
    (
        re.compile(r"from\s+qiskit\.opflow\b"),
        "qiskit.opflow (deprecated) — use SparsePauliOp from qiskit.quantum_info",
    ),
]


def check_file(path: str) -> list[str]:
    """Check a single file for deprecated primitives usage."""
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
                if stripped.startswith("#") or "# noqa" in line:
                    continue
                for pat, msg in DEPRECATED_PATTERNS:
                    if pat.search(line):
                        violations.append(f"{path}:{lineno}: {msg}")
    except (OSError, UnicodeDecodeError):
        pass
    return violations


def main() -> int:
    errors = []
    for path in sys.argv[1:]:
        errors.extend(check_file(path))
    if errors:
        print("❌ Primitives V2 guard violations:")
        for e in errors:
            print(f"  {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
