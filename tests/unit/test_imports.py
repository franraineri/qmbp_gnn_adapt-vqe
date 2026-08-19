"""Unit tests for import dependency order in qmbp_simulation."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Module dependency DAG (upstream → downstream)
# Each module may only import from modules listed as its dependencies.
DEPENDENCY_ORDER = {
    "utils": [],
    "models": ["utils"],
    "solvers": ["models", "utils"],
    "circuits": ["models", "utils"],
    "execution": ["circuits", "models", "utils", "framework", "predictors"],
    "optimizers": ["execution", "circuits", "models", "utils"],
    "predictors": ["models", "utils", "execution", "solvers", "circuits", "analysis", "pipeline", "framework"],
    "analysis": ["predictors", "models", "utils", "solvers", "execution", "circuits", "framework"],
    "pipeline": [
        "analysis",
        "solvers",
        "optimizers",
        "predictors",
        "execution",
        "circuits",
        "models",
        "utils",
        "framework",
    ],
    "framework": [
        "pipeline",
        "analysis",
        "optimizers",
        "predictors",
        "solvers",
        "circuits",
        "execution",
        "models",
        "utils",
    ],
}

PACKAGE_ROOT = Path("src/qmbp_simulation")


class TestDependencyOrder:
    """Verify dependency order: each module only imports from upstream."""

    @pytest.mark.parametrize("module_name", list(DEPENDENCY_ORDER.keys()))
    def test_module_respects_dependency_order(self, module_name):
        module_dir = PACKAGE_ROOT / module_name
        if not module_dir.is_dir():
            pytest.skip(f"{module_dir} not found")

        allowed = set(DEPENDENCY_ORDER[module_name])
        all_submodules = set(DEPENDENCY_ORDER.keys())
        forbidden = all_submodules - allowed - {module_name}

        violations = []
        for py_file in module_dir.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for fb in forbidden:
                            if f"qmbp_simulation.{fb}" in alias.name:
                                violations.append(f"{py_file.name}: imports {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for fb in forbidden:
                        if f"qmbp_simulation.{fb}" in node.module:
                            violations.append(f"{py_file.name}: from {node.module}")

        assert violations == [], f"Module '{module_name}' has forbidden imports:\n" + "\n".join(
            f"  - {v}" for v in violations
        )


class TestNoExperimentOrScriptImports:
    """Verify no imports from experiments/ or scripts/.

    Exception: function-scoped lazy imports are allowed for optional
    integrations (e.g., active_learning helper used only when called).
    """

    # Known lazy imports inside function bodies (not structural dependencies)
    _ALLOWED_LAZY = {
        "runner_base.py": {"experiments.helpers.active_learning"},
    }

    def test_no_experiment_imports(self):
        violations = []
        for py_file in PACKAGE_ROOT.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                continue

            allowed_for_file = self._ALLOWED_LAZY.get(py_file.name, set())

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if "experiments" in node.module or "scripts" in node.module:
                        if node.module in allowed_for_file:
                            continue
                        violations.append(
                            f"{py_file.relative_to(PACKAGE_ROOT)}: from {node.module}"
                        )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if "experiments" in alias.name or "scripts" in alias.name:
                            if alias.name in allowed_for_file:
                                continue
                            violations.append(
                                f"{py_file.relative_to(PACKAGE_ROOT)}: import {alias.name}"
                            )
        assert violations == [], "Package imports from experiments/ or scripts/:\n" + "\n".join(
            f"  - {v}" for v in violations
        )
