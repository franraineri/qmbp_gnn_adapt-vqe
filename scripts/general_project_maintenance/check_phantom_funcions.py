# TODO move mainteinance tools to a new project repository in order to reuse them. Also a refactor and modular parameters will be needed
"""Check for phantom imports from any module across all Python files.

Validates that every symbol imported via `from X import Y` actually exists
in module X. Catches:
- Typos in import names
- Symbols removed during refactoring but still imported elsewhere
- Stale re-exports from __init__.py that reference deleted functions

Usage:
    # Default: check analysis.metrics (original behavior)
    python check_phantom_functions.py

    # Check any module:
    python check_phantom_functions.py qmbp_simulation.predictors.model_zoo
    python check_phantom_functions.py qmbp_simulation.framework.result_io
    python check_phantom_functions.py qmbp_simulation.analysis.cross_n_validator

    # Check multiple modules:
    python check_phantom_functions.py qmbp_simulation.analysis.metrics qmbp_simulation.predictors.model_zoo

    # Check ALL qmbp_simulation submodules:
    python check_phantom_functions.py --all

    # Self-test (verify the checker itself works):
    python check_phantom_functions.py --self-test
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

# Directories to always skip
_SKIP_DIRS = {"__pycache__", ".venv", "_deprecated", ".git", "node_modules", ".hypothesis"}


def check_phantom_imports(module_path: str, *, verbose: bool = False) -> list[tuple[str, str]]:
    """Find all imports from `module_path` that don't actually exist in that module.

    Parameters
    ----------
    module_path : str
        Dotted module path (e.g. "qmbp_simulation.analysis.metrics").
    verbose : bool
        If True, print each file being scanned.

    Returns
    -------
    list[tuple[str, str]]
        List of (file_path, symbol_name) for phantom imports.
    """
    # Import the target module to check hasattr
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        print(f"  ERROR: Cannot import {module_path}: {e}")
        return []

    # Build regex patterns for this module
    module_re = re.escape(module_path)
    single_line_pattern = re.compile(rf"from\s+{module_re}\s+import\s+([^\n]+)")
    multi_line_pattern = re.compile(
        rf"from\s+{module_re}\s+import\s+\(\s*\n((?:.*?\n)*?)\s*\)",
        re.MULTILINE,
    )

    problems: list[tuple[str, str]] = []
    src_root = Path(".")
    n_scanned = 0

    for py_file in src_root.rglob("*.py"):
        # Skip excluded directories
        if any(skip in py_file.parts for skip in _SKIP_DIRS):
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        # Fast path: skip files that don't reference the module at all
        if module_path not in source:
            continue

        n_scanned += 1
        path_str = str(py_file)
        if verbose:
            print(f"    Scanning: {path_str}")

        # Extract imported names from single-line imports
        for match in single_line_pattern.finditer(source):
            names = _extract_names_from_import(match.group(1))
            for name in names:
                if not hasattr(module, name):
                    problems.append((path_str, name))

        # Extract imported names from multi-line imports
        for match in multi_line_pattern.finditer(source):
            names = _extract_names_from_import(match.group(1))
            for name in names:
                if not hasattr(module, name):
                    problems.append((path_str, name))

    # Deduplicate preserving order
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for item in problems:
        if item not in seen:
            seen.add(item)
            unique.append(item)

    if verbose:
        print(f"    Scanned {n_scanned} file(s) referencing {module_path}")

    return unique


def _extract_names_from_import(import_text: str) -> list[str]:
    """Extract clean symbol names from an import statement text.

    Handles:
    - Comma-separated names: "A, B, C"
    - Trailing commas: "A, B,"
    - Aliases: "A as alias" → extracts "A"
    - Parenthesized: "(A, B, C)"
    - Multiline blocks with comments: "A,  # comment\\nB,"
    - Noqa annotations: "A  # noqa: E402" → extracts "A"
    - Empty/whitespace-only entries

    Returns
    -------
    list[str]
        Clean symbol names (without aliases, comments, or whitespace).
    """
    # Remove parentheses and normalize
    text = import_text.strip().strip("()")

    # Split by comma and newline
    raw_names = re.split(r"[,\n]", text)

    names: list[str] = []
    for raw in raw_names:
        # Strip inline comments (# noqa, # type: ignore, etc.)
        name = raw.split("#")[0].strip()
        if not name or name == ")":
            continue
        # Handle "Name as alias" → keep "Name"
        if " as " in name:
            name = name.split(" as ")[0].strip()
        # Handle trailing whitespace or stray characters after the name
        # A valid import name is just an identifier, no spaces inside
        name = name.split()[0] if " " in name else name
        # Final validation: must be a valid Python identifier and not a keyword
        if name.isidentifier() and name not in ("self", "cls", "True", "False", "None"):
            names.append(name)

    return names


def _discover_all_submodules() -> list[str]:
    """Discover all importable qmbp_simulation submodules."""
    src_dir = Path("src/qmbp_simulation")
    if not src_dir.exists():
        print("  ERROR: src/qmbp_simulation not found. Run from project root.")
        return []

    modules: list[str] = []
    for py_file in sorted(src_dir.rglob("*.py")):
        if "__pycache__" in str(py_file) or py_file.name.startswith("_"):
            continue
        # Convert path to module: src/qmbp_simulation/analysis/metrics.py → qmbp_simulation.analysis.metrics
        rel = py_file.relative_to(Path("src"))
        module_path = str(rel.with_suffix("")).replace("/", ".")
        modules.append(module_path)

    return modules


def _run_self_test() -> bool:
    """Verify the checker works correctly with known cases."""
    print("\n" + "=" * 60)
    print("SELF-TEST: Verifying checker correctness")
    print("=" * 60)

    all_ok = True

    # Test 1: _extract_names_from_import handles basic cases
    assert _extract_names_from_import("A, B, C") == ["A", "B", "C"], "Basic comma-sep failed"
    assert _extract_names_from_import("A, B,") == ["A", "B"], "Trailing comma failed"
    assert _extract_names_from_import("A as x, B") == ["A", "B"], "Alias failed"
    assert _extract_names_from_import("A,  # comment\nB,") == ["A", "B"], "Comment failed"
    assert _extract_names_from_import("(A, B)") == ["A", "B"], "Parens failed"
    assert _extract_names_from_import("") == [], "Empty failed"
    assert _extract_names_from_import("   ") == [], "Whitespace failed"
    assert _extract_names_from_import("123bad") == [], "Invalid identifier filtered"
    assert _extract_names_from_import("A  # noqa: E402") == ["A"], "Noqa comment failed"
    assert _extract_names_from_import("self") == [], "self should be filtered"
    assert _extract_names_from_import("cls") == [], "cls should be filtered"
    print("  ✅ _extract_names_from_import: all cases pass")

    # Test 2: check_phantom_imports finds real phantoms
    # We know 'NONEXISTENT_SYMBOL' doesn't exist in metrics
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=".", delete=False) as f:
        f.write("from qmbp_simulation.analysis.metrics import NONEXISTENT_SYMBOL_XYZ\n")
        tmp_name = f.name

    try:
        phantoms = check_phantom_imports("qmbp_simulation.analysis.metrics")
        found = any(name == "NONEXISTENT_SYMBOL_XYZ" for _, name in phantoms)
        if found:
            print("  ✅ Detects phantom import correctly")
        else:
            print("  ❌ Failed to detect phantom import!")
            all_ok = False
    finally:
        os.unlink(tmp_name)

    # Test 3: Real module with no phantoms should return empty (or only pre-existing issues)
    phantoms_real = check_phantom_imports("qmbp_simulation.utils.helpers")
    # This should have 0 phantoms (utils.helpers is stable)
    if not phantoms_real:
        print("  ✅ No false positives for stable module")
    else:
        print(f"  ⚠️  {len(phantoms_real)} phantom(s) in utils.helpers (may be pre-existing)")

    if all_ok:
        print("\n  ✅ Self-test PASSED")
    else:
        print("\n  ❌ Self-test FAILED")

    return all_ok


def main() -> int:
    args = sys.argv[1:]

    # Handle flags
    verbose = "--verbose" in args or "-v" in args
    args = [a for a in args if a not in ("--verbose", "-v")]

    if "--self-test" in args:
        return 0 if _run_self_test() else 1

    if "--all" in args:
        modules = _discover_all_submodules()
        if not modules:
            return 1
        print(f"Discovered {len(modules)} submodules to check")
    elif args:
        modules = args
    else:
        # Default: check the most critical modules
        modules = ["qmbp_simulation.analysis.metrics"]

    total_phantoms = 0
    all_phantoms: list[tuple[str, str, str]] = []  # (module, file, symbol)
    for module_path in modules:
        print(f"\n{' '}")
        print(f"Checking: {module_path}")
        phantoms = check_phantom_imports(module_path, verbose=verbose)
        total_phantoms += len(phantoms)

        if phantoms:
            print(f"  PHANTOM IMPORTS ({len(phantoms)}):")
            for path, name in phantoms:
                print(f"    {path}: {name}")
                all_phantoms.append((module_path, path, name))
        else:
            print("  ✅ All imports resolve correctly")

    print()
    if total_phantoms == 0:
        print(f"✅ No phantom imports found across {len(modules)} module(s)")
        return 0
    else:
        print(f"{'=' * 60}")
        print(f"❌ SUMMARY: {total_phantoms} phantom import(s) found")
        print(f"{'=' * 60}")
        # Group by file for actionable output
        by_file: dict[str, list[tuple[str, str]]] = {}
        for mod, path, name in all_phantoms:
            by_file.setdefault(path, []).append((mod, name))
        for path, items in sorted(by_file.items()):
            print(f"\n  {path}:")
            for mod, name in items:
                short_mod = mod.split(".")[-1]
                print(f"    ✗ {name}  (from {short_mod})")
        print()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
