#!/usr/bin/env python3
"""Compila las figuras TikZ de la tesis (``fig_*.tex``) a PDF individuales.

Cada fragmento TikZ en ``internal/tesis-figures/fig_*.tex`` es un cuerpo de
``tikzpicture`` sin preámbulo (se incluye en la tesis con ``\\input``). Este
script envuelve cada fragmento en un documento ``standalone`` mínimo —con los
mismos paquetes y la paleta que usa la tesis— y lo compila a ``<nombre>.pdf``
con ``pdflatex``. Así se obtiene una vista previa de cada figura y se detecta
cualquier error de LaTeX antes de compilar la memoria completa.

Uso:
    python scripts/general_project_maintenance/build_thesis_figures.py
    python scripts/general_project_maintenance/build_thesis_figures.py --figure fig_pipeline_tikz
    python scripts/general_project_maintenance/build_thesis_figures.py --keep-aux

Salida: un PDF por figura junto al ``.tex`` fuente, y un resumen por consola.
No requiere el estilo de la tesis (``estilo_unir``): usa ``standalone``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parents[2] / "internal" / "tesis" / "tesis-figures"
PALETTE = "palette"

# Documento envoltorio. Se prefiere 'standalone' (recorta al bounding box) si
# está instalado; si no, se usa 'article' apaisado A3 con márgenes mínimos y
# página vacía (mismo enfoque que _preview.tex). Se cargan los mismos paquetes
# y bibliotecas TikZ que necesita la tesis.
WRAPPER_STANDALONE = r"""\documentclass[border=6pt]{standalone}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, shapes.geometric, fit, calc}
\input{%(palette)s}
\begin{document}
\input{%(figure)s}
\end{document}
"""

WRAPPER_ARTICLE = r"""\documentclass{article}
\usepackage[a3paper,landscape,margin=1cm]{geometry}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, shapes.geometric, fit, calc}
\input{%(palette)s}
\pagestyle{empty}
\begin{document}
\centering
\null\vfill
\input{%(figure)s}
\vfill
\end{document}
"""


def _has_pdflatex() -> bool:
    return shutil.which("pdflatex") is not None


def _has_standalone() -> bool:
    """True si la clase standalone.cls está instalada (kpsewhich la localiza)."""
    if shutil.which("kpsewhich") is None:
        return False
    try:
        r = subprocess.run(["kpsewhich", "standalone.cls"], capture_output=True, text=True, timeout=15)
        return r.returncode == 0 and r.stdout.strip() != ""
    except subprocess.TimeoutExpired:
        return False


def build_figure(fig_stem: str, keep_aux: bool = False) -> tuple[bool, str]:
    """Compila una figura ``fig_stem`` (sin extensión) a ``fig_stem.pdf``.

    Returns:
        (ok, mensaje): ok=True si el PDF se generó; mensaje con el detalle.
    """
    fig_src = FIGURES_DIR / f"{fig_stem}.tex"
    if not fig_src.exists():
        return False, f"no existe {fig_src.name}"

    # Documento envoltorio temporal DENTRO de FIGURES_DIR para que los \input
    # de la figura y de la paleta resuelvan por ruta relativa.
    wrapper_name = f"_build_{fig_stem}"
    wrapper_tex = FIGURES_DIR / f"{wrapper_name}.tex"
    wrapper = WRAPPER_STANDALONE if _has_standalone() else WRAPPER_ARTICLE
    wrapper_tex.write_text(wrapper % {"palette": PALETTE, "figure": fig_stem}, encoding="utf-8")

    try:
        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"-output-directory={td}",
                    wrapper_tex.name,
                ],
                cwd=FIGURES_DIR,
                capture_output=True,
                text=True,
                timeout=120,
            )
            out_pdf = Path(td) / f"{wrapper_name}.pdf"
            if proc.returncode != 0 or not out_pdf.exists():
                # Extraer la primera línea de error de LaTeX (empieza con '!').
                err = next(
                    (ln for ln in proc.stdout.splitlines() if ln.startswith("!")),
                    "error de compilación (ver log completo con --keep-aux)",
                )
                if keep_aux:
                    (FIGURES_DIR / f"{wrapper_name}.log").write_text(proc.stdout, encoding="utf-8")
                return False, err
            # Copiar el PDF resultante junto al fuente, con el nombre de la figura.
            final_pdf = FIGURES_DIR / f"{fig_stem}.pdf"
            shutil.copyfile(out_pdf, final_pdf)
            return True, f"→ {final_pdf.name}"
    except subprocess.TimeoutExpired:
        return False, "timeout (>120 s)"
    finally:
        if not keep_aux:
            wrapper_tex.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        default=None,
        help="Compila solo esta figura (nombre sin extensión, p. ej. fig_pipeline_tikz).",
    )
    parser.add_argument(
        "--keep-aux",
        action="store_true",
        help="Conserva el .tex envoltorio y guarda el log en caso de error.",
    )
    args = parser.parse_args()

    if not _has_pdflatex():
        print("❌ pdflatex no está disponible en el PATH; no se pueden compilar figuras.")
        return 1

    if args.figure:
        stems = [args.figure.replace(".tex", "")]
    else:
        # Todas las figuras fig_*.tex (excluye palette, _preview y envoltorios _build_).
        stems = sorted(p.stem for p in FIGURES_DIR.glob("fig_*.tex") if not p.stem.startswith("_build_"))

    if not stems:
        print(f"No se encontraron figuras fig_*.tex en {FIGURES_DIR}")
        return 0

    print(f"🖼  Compilando {len(stems)} figura(s) TikZ en {FIGURES_DIR.name}/ …")
    n_ok = 0
    for stem in stems:
        ok, msg = build_figure(stem, keep_aux=args.keep_aux)
        mark = "✅" if ok else "❌"
        print(f"  {mark} {stem}  {msg}")
        n_ok += ok

    print(f"\n{n_ok}/{len(stems)} figuras compiladas correctamente.")
    return 0 if n_ok == len(stems) else 1


if __name__ == "__main__":
    sys.exit(main())
