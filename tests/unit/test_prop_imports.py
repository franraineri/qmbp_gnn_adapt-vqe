"""Property-based tests for import dependency order in qmbp_simulation.

# Feature: framework-restructure, Property 19: Import dependency order
# No downstream or circular imports in the package dependency graph.
"""

from __future__ import annotations

import ast
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Dependency DAG definition
# Each module lists its ALLOWED upstream dependencies (transitive closure).
# A module may only import from modules in its allowed set.
# ---------------------------------------------------------------------------

DEPENDENCY_DAG: dict[str, set[str]] = {
    "utils": set(),
    "models": {"utils"},
    "solvers": {"models", "utils"},
    "circuits": {"models", "utils"},
    "execution": {"circuits", "models", "utils"},
    "optimizers": {"execution", "circuits", "models", "utils"},
    "predictors": {"models", "utils"},
    "analysis": {"predictors", "models", "utils"},
    "pipeline": {
        "analysis",
        "solvers",
        "optimizers",
        "predictors",
        "execution",
        "circuits",
        "models",
        "utils",
    },
    "framework": {
        "pipeline",
        "analysis",
        "optimizers",
        "predictors",
        "solvers",
        "circuits",
        "execution",
        "models",
        "utils",
    },
}

ALL_MODULES = list(DEPENDENCY_DAG.keys())
PACKAGE_ROOT = Path("src/qmbp_simulation")


# ---------------------------------------------------------------------------
# Helper: extract internal imports from a module's source files
# ---------------------------------------------------------------------------


def _get_internal_imports(module_name: str) -> set[str]:
    """Parse all .py files in a module and return set of imported submodules.

    Returns the set of qmbp_simulation submodule names that this module
    imports from (e.g., {"models", "utils"}).
    """
    module_dir = PACKAGE_ROOT / module_name
    if not module_dir.is_dir():
        return set()

    imported_modules: set[str] = set()

    for py_file in module_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if len(parts) >= 2 and parts[0] == "qmbp_simulation":
                        imported_modules.add(parts[1])
            elif isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                if len(parts) >= 2 and parts[0] == "qmbp_simulation":
                    imported_modules.add(parts[1])

    # Remove self-imports (a module importing from itself is fine)
    imported_modules.discard(module_name)
    return imported_modules


def _get_forbidden_modules(module_name: str) -> set[str]:
    """Return the set of modules that this module must NOT import from."""
    allowed = DEPENDENCY_DAG[module_name]
    all_submodules = set(ALL_MODULES)
    return all_submodules - allowed - {module_name}


# ---------------------------------------------------------------------------
# Property 19: Import dependency order — no downstream or circular imports
#
# For any module in the package, all of its internal imports must come from
# allowed upstream modules only. No module should import from downstream
# modules or from experiments/scripts.
#
# **Validates: Requirements 12.1, 12.3, 13.3**
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(module_name=st.sampled_from(ALL_MODULES))
def test_module_only_imports_from_upstream(module_name: str) -> None:
    """Each module only imports from its allowed upstream dependencies.

    For any module drawn from the dependency DAG, all internal imports
    (from qmbp_simulation.X) must be in the module's allowed set.
    No downstream imports are permitted.

    **Validates: Requirements 12.1, 12.3, 13.3**
    """
    module_dir = PACKAGE_ROOT / module_name
    if not module_dir.is_dir():
        # Module not yet created — skip (structural test)
        return

    actual_imports = _get_internal_imports(module_name)
    forbidden = _get_forbidden_modules(module_name)

    violations = actual_imports & forbidden
    assert violations == set(), (
        f"Module '{module_name}' has forbidden downstream imports: {violations}. "
        f"Allowed: {DEPENDENCY_DAG[module_name]}"
    )


@settings(max_examples=50, deadline=None)
@given(module_name=st.sampled_from(ALL_MODULES))
def test_module_does_not_import_from_experiments_or_scripts(
    module_name: str,
) -> None:
    """No package module imports from experiments/ or scripts/.

    **Validates: Requirements 12.3, 13.3**
    """
    module_dir = PACKAGE_ROOT / module_name
    if not module_dir.is_dir():
        return

    violations: list[str] = []

    for py_file in module_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if "experiments" in node.module or "scripts" in node.module:
                    violations.append(f"{py_file.name}: from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "experiments" in alias.name or "scripts" in alias.name:
                        violations.append(f"{py_file.name}: import {alias.name}")

    assert violations == [], (
        f"Module '{module_name}' imports from experiments/ or scripts/:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


@settings(max_examples=50, deadline=None)
@given(
    source=st.sampled_from(ALL_MODULES),
    target=st.sampled_from(ALL_MODULES),
)
def test_no_downstream_import_pair(source: str, target: str) -> None:
    """For any (source, target) pair where target is downstream of source,
    source does NOT import from target.

    This tests the contrapositive: if target is NOT in source's allowed set
    (and target != source), then source must not import target.

    **Validates: Requirements 12.1, 12.3, 13.3**
    """
    if source == target:
        return  # Self-import is fine

    # Check if target is forbidden for source
    allowed = DEPENDENCY_DAG[source]
    if target in allowed:
        return  # target is an allowed upstream dep — nothing to check

    # target is downstream or unrelated — source must NOT import it
    module_dir = PACKAGE_ROOT / source
    if not module_dir.is_dir():
        return

    actual_imports = _get_internal_imports(source)
    assert target not in actual_imports, (
        f"Module '{source}' imports from downstream module '{target}'. "
        f"Allowed imports for '{source}': {allowed}"
    )
