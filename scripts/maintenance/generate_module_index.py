#!/usr/bin/env python3
"""Generate a compact module index for .kiro/steering/module-index.md.

Introspects src/, scripts/, project_health/, experiments/ and extracts:
- Public classes and functions from each .py module
- One-line docstring summary
- Key exports per sub-package

Output: ultra-compact markdown optimized for LLM context (min tokens, max info).

Usage:
    python scripts/maintenance/generate_module_index.py
    python scripts/maintenance/generate_module_index.py --output .kiro/steering/module-index.md
    python scripts/maintenance/generate_module_index.py --dry-run
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / ".kiro" / "steering" / "module-index.md"

SCAN_DIRS = [
    ("src/qmbp_simulation", "LIB"),
    ("scripts/analysis", "SCRIPT"),
    ("scripts/benchmarks", "SCRIPT"),
    ("scripts/experiment_runners", "RUNNER"),
    ("scripts/hardware", "SCRIPT"),
    ("scripts/maintenance", "MAINT"),
    ("scripts/validation", "SCRIPT"),
    ("project_health", "HEALTH"),
    ("experiments", "EXP"),
    ("notebooks", "NB"),
]

# Skip these file patterns
SKIP_FILES = {"__pycache__", ".DS_Store", "__init__.py"}
SKIP_DIRS = {"__pycache__", ".hypothesis", ".mypy_cache", ".ruff_cache", "data"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ModuleEntry:
    """One .py module's extracted metadata."""

    rel_path: str  # relative to project root
    category: str  # LIB, SCRIPT, RUNNER, EXP, HEALTH, MAINT, NB
    docstring: str  # first line of module docstring
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)


@dataclass
class PackageEntry:
    """A sub-package (directory) summary."""

    rel_path: str
    category: str
    docstring: str  # from __init__.py
    exports: list[str] = field(default_factory=list)  # __all__ from __init__
    modules: list[ModuleEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AST extraction
# ---------------------------------------------------------------------------


def _first_line_docstring(node: ast.Module | ast.ClassDef | ast.FunctionDef) -> str:
    """Extract first line of a node's docstring, or empty string."""
    doc = ast.get_docstring(node)
    if not doc:
        return ""
    return doc.split("\n")[0].strip()[:120]


def _is_public(name: str) -> bool:
    """Check if a name is public (no leading underscore)."""
    return not name.startswith("_")


def _extract_all_list(tree: ast.Module) -> list[str]:
    """Extract __all__ = [...] from module AST."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        return [
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]
    return []


def _extract_constants(tree: ast.Module) -> list[str]:
    """Extract UPPER_CASE module-level constants."""
    constants = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper() and _is_public(target.id):
                    constants.append(target.id)
    return constants[:10]  # cap to avoid noise


def extract_module(filepath: Path, category: str) -> ModuleEntry:
    """Parse a .py file and extract public API metadata."""
    rel = filepath.relative_to(PROJECT_ROOT).as_posix()
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        return ModuleEntry(rel_path=rel, category=category, docstring="[parse error]")

    docstring = _first_line_docstring(tree)
    classes = []
    functions = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and _is_public(node.name):
            classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(node.name):
            functions.append(node.name)

    constants = _extract_constants(tree)

    return ModuleEntry(
        rel_path=rel,
        category=category,
        docstring=docstring,
        classes=classes,
        functions=functions,
        constants=constants,
    )


def extract_package(dirpath: Path, category: str) -> PackageEntry:
    """Extract summary from a package's __init__.py."""
    rel = dirpath.relative_to(PROJECT_ROOT).as_posix()
    init_file = dirpath / "__init__.py"
    docstring = ""
    exports: list[str] = []

    if init_file.exists():
        try:
            source = init_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
            docstring = _first_line_docstring(tree)
            exports = _extract_all_list(tree)
        except (SyntaxError, OSError):
            pass

    return PackageEntry(
        rel_path=rel,
        category=category,
        docstring=docstring,
        exports=exports,
    )


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


def scan_directory(rel_dir: str, category: str) -> tuple[list[PackageEntry], list[ModuleEntry]]:
    """Scan a directory tree and extract packages + standalone modules."""
    base = PROJECT_ROOT / rel_dir
    if not base.exists():
        return [], []

    packages: list[PackageEntry] = []
    standalone_modules: list[ModuleEntry] = []

    # Check if base itself is a package
    if (base / "__init__.py").exists():
        pkg = extract_package(base, category)
        # Scan direct .py files (not __init__)
        for f in sorted(base.glob("*.py")):
            if f.name in SKIP_FILES:
                continue
            pkg.modules.append(extract_module(f, category))
        packages.append(pkg)

    # Scan sub-directories
    for child in sorted(base.iterdir()):
        if not child.is_dir() or child.name in SKIP_DIRS:
            continue
        if (child / "__init__.py").exists():
            sub_pkg = extract_package(child, category)
            for f in sorted(child.glob("*.py")):
                if f.name in SKIP_FILES:
                    continue
                sub_pkg.modules.append(extract_module(f, category))
            # One level deeper for nested packages (e.g. execution/hardware)
            for grandchild in sorted(child.iterdir()):
                if not grandchild.is_dir() or grandchild.name in SKIP_DIRS:
                    continue
                if (grandchild / "__init__.py").exists():
                    deep_pkg = extract_package(grandchild, category)
                    for f in sorted(grandchild.glob("*.py")):
                        if f.name in SKIP_FILES:
                            continue
                        deep_pkg.modules.append(extract_module(f, category))
                    packages.append(deep_pkg)
            packages.append(sub_pkg)
        else:
            # Plain directory with scripts (no __init__)
            for f in sorted(child.glob("*.py")):
                if f.name in SKIP_FILES:
                    continue
                standalone_modules.append(extract_module(f, category))
            # Check deeper dirs too
            for grandchild in sorted(child.iterdir()):
                if not grandchild.is_dir() or grandchild.name in SKIP_DIRS:
                    continue
                for f in sorted(grandchild.glob("*.py")):
                    if f.name in SKIP_FILES:
                        continue
                    standalone_modules.append(extract_module(f, category))

    # Top-level .py files not in a package
    for f in sorted(base.glob("*.py")):
        if f.name in SKIP_FILES:
            continue
        if not (base / "__init__.py").exists():
            standalone_modules.append(extract_module(f, category))

    return packages, standalone_modules


# ---------------------------------------------------------------------------
# Formatter — ultra-compact markdown
# ---------------------------------------------------------------------------


def _short_path(path: str) -> str:
    """Abbreviate common prefixes for compactness."""
    return (
        path.replace("src/qmbp_simulation/", "qsim/")
        .replace("project_health/", "ph/")
        .replace("scripts/", "s/")
        .replace("experiments/", "exp/")
        .replace("notebooks/", "nb/")
    )


def _exports_compact(exports: list[str], max_show: int = 12) -> str:
    """Format exports as compact comma-separated list."""
    if not exports:
        return ""
    shown = exports[:max_show]
    suffix = f" +{len(exports) - max_show}" if len(exports) > max_show else ""
    return ", ".join(shown) + suffix


def _module_line(m: ModuleEntry) -> str:
    """One compact line per module: path | doc | C:classes F:funcs."""
    parts = [f"`{_short_path(m.rel_path)}`"]
    if m.docstring:
        parts.append(m.docstring[:80])
    symbols = []
    if m.classes:
        symbols.append(f"C:{','.join(m.classes[:5])}")
        if len(m.classes) > 5:
            symbols[-1] += f"+{len(m.classes)-5}"
    if m.functions:
        symbols.append(f"F:{','.join(m.functions[:6])}")
        if len(m.functions) > 6:
            symbols[-1] += f"+{len(m.functions)-6}"
    if m.constants:
        symbols.append(f"K:{','.join(m.constants[:4])}")
    if symbols:
        parts.append(" | ".join(symbols))
    return " — ".join(parts)


def format_index(
    all_packages: list[PackageEntry],
    all_modules: list[ModuleEntry],
) -> str:
    """Generate the ultra-compact module-index.md content."""
    lines: list[str] = []
    lines.append("# Module Index (auto-generated)")
    lines.append("")
    lines.append("Compact catalog of all code modules. Use to find reusable functionality.")
    lines.append("Run `python scripts/maintenance/generate_module_index.py` to refresh.")
    lines.append("")

    # Group packages by category
    cat_order = ["LIB", "HEALTH", "EXP", "RUNNER", "SCRIPT", "MAINT", "NB"]
    cat_labels = {
        "LIB": "Library (src/qmbp_simulation)",
        "HEALTH": "Project Health (project_health/)",
        "EXP": "Experiments (experiments/)",
        "RUNNER": "Runners (scripts/experiment_runners/)",
        "SCRIPT": "Scripts (scripts/)",
        "MAINT": "Maintenance (scripts/maintenance/)",
        "NB": "Notebooks",
    }

    for cat in cat_order:
        pkgs = [p for p in all_packages if p.category == cat]
        mods = [m for m in all_modules if m.category == cat]
        if not pkgs and not mods:
            continue

        lines.append(f"## {cat_labels.get(cat, cat)}")
        lines.append("")

        for pkg in pkgs:
            sp = _short_path(pkg.rel_path)
            doc_part = f" — {pkg.docstring}" if pkg.docstring else ""
            lines.append(f"### `{sp}/`{doc_part}")
            if pkg.exports:
                lines.append(f"Exports: {_exports_compact(pkg.exports)}")
            lines.append("")
            for m in pkg.modules:
                lines.append(f"- {_module_line(m)}")
            lines.append("")

        if mods:
            lines.append("### Standalone scripts")
            lines.append("")
            for m in mods:
                lines.append(f"- {_module_line(m)}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate compact module index for .kiro/steering/module-index.md",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output file (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print to stdout instead of writing file",
    )
    args = parser.parse_args()

    all_packages: list[PackageEntry] = []
    all_modules: list[ModuleEntry] = []

    for rel_dir, category in SCAN_DIRS:
        pkgs, mods = scan_directory(rel_dir, category)
        all_packages.extend(pkgs)
        all_modules.extend(mods)

    result = format_index(all_packages, all_modules)

    if args.dry_run:
        print(result)
        print(f"\n--- {len(all_packages)} packages, {len(all_modules)} standalone modules ---",
              file=sys.stderr)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
        total = len(all_packages) + len(all_modules)
        print(f"✓ Module index written to {args.output.relative_to(PROJECT_ROOT)}")
        print(f"  {len(all_packages)} packages, {len(all_modules)} standalone, {total} total entries")


if __name__ == "__main__":
    main()
