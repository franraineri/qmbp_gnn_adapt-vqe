#!/usr/bin/env python3
#TODO move mainteinance tools to a new project repository in order to reuse them. Also a refactor and modular parameters will be needed

"""Detecta funciones Python cuyo docstring tiene más líneas que su implementación.

Uso:
    .venv/bin/python scripts/maintenance/detect_overdocumented_functions.py [ruta_proyecto]

Si no se pasa ruta, escanea el directorio actual.

Criterio: una función se reporta si:
    líneas_docstring > líneas_cuerpo (excluyendo el docstring y líneas vacías del cuerpo)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def count_docstring_lines(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Cuenta las líneas del docstring de una función (0 si no tiene)."""
    docstring_node = ast.get_docstring(node, clean=False)
    if docstring_node is None:
        return 0
    # El docstring ocupa las líneas del nodo Expr que lo contiene
    expr_node = node.body[0]
    return expr_node.end_lineno - expr_node.lineno + 1


def count_body_lines(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Cuenta líneas de implementación (cuerpo sin el docstring, sin líneas vacías)."""
    has_docstring = (
        isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, (ast.Constant, ast.Str))
    )

    if has_docstring:
        body_nodes = node.body[1:]
    else:
        body_nodes = node.body

    if not body_nodes:
        return 0

    # Rango de líneas del cuerpo (sin docstring)
    first_line = body_nodes[0].lineno
    last_line = max(n.end_lineno for n in body_nodes)
    return last_line - first_line + 1


def scan_file(filepath: Path) -> list[dict]:
    """Escanea un archivo Python y devuelve funciones over-documented."""
    try:
        source = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return []

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    results = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        doc_lines = count_docstring_lines(node)
        if doc_lines == 0:
            continue

        body_lines = count_body_lines(node)

        if doc_lines > body_lines:
            results.append({
                "file": str(filepath),
                "line": node.lineno,
                "function": node.name,
                "docstring_lines": doc_lines,
                "body_lines": body_lines,
                "ratio": round(doc_lines / max(body_lines, 1), 1),
            })

    return results


def scan_project(root: Path) -> list[dict]:
    """Escanea recursivamente un proyecto buscando archivos .py."""
    all_results = []

    # Directorios a ignorar
    skip_dirs = {
        ".venv", "venv", "env", ".env",
        "node_modules", "__pycache__", ".git",
        ".tox", ".mypy_cache", ".pytest_cache",
        "dist", "build", "egg-info",
    }

    for py_file in root.rglob("*.py"):
        # Saltear directorios ignorados
        parts = set(py_file.relative_to(root).parts)
        if parts & skip_dirs:
            continue
        if any(p.endswith(".egg-info") for p in py_file.parts):
            continue

        all_results.extend(scan_file(py_file))

    # Ordenar por ratio descendente
    all_results.sort(key=lambda r: r["ratio"], reverse=True)
    return all_results


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    root = root.resolve()

    if not root.is_dir():
        print(f"Error: {root} no es un directorio válido", file=sys.stderr)
        sys.exit(1)

    print(f"Escaneando: {root}\n")

    results = scan_project(root)

    if not results:
        print("✓ No se encontraron funciones over-documented.")
        return

    print(f"Encontradas {len(results)} funciones con docstring > implementación:\n")
    print(f"{'Archivo':<60} {'Línea':>6} {'Función':<30} {'Doc':>4} {'Body':>5} {'Ratio':>6}")
    print("-" * 115)

    for r in results:
        # Mostrar path relativo si es posible
        try:
            display_path = str(Path(r["file"]).relative_to(root))
        except ValueError:
            display_path = r["file"]

        # Truncar si es muy largo
        if len(display_path) > 58:
            display_path = "..." + display_path[-55:]

        print(
            f"{display_path:<60} {r['line']:>6} {r['function']:<30} "
            f"{r['docstring_lines']:>4} {r['body_lines']:>5} {r['ratio']:>5}x"
        )

    print(f"\nTotal: {len(results)} funciones")
    print(f"Peor ratio: {results[0]['function']} en {results[0]['file']}:{results[0]['line']} ({results[0]['ratio']}x)")


if __name__ == "__main__":
    main()
