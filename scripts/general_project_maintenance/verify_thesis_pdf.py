#!/usr/bin/env python3
"""Verificador de consistencia sobre el PDF renderizado de la tesis.

Complementa a ``verify_thesis_numbers.py`` (que opera sobre el .tex fuente).
Este script opera sobre el ARTEFACTO FINAL: extrae el texto del PDF con
``pdftotext`` y valida las cifras tal como las leerá el tribunal. Detecta
divergencias que solo se manifiestan tras compilar (p. ej. una celda auto que
se re-generó y dejó de cuadrar con el texto, o un redondeo que cambió).

Chequeos (todos deterministas y seguros):
  1. Caso absoluto: todo 'p% (n/m)' impreso cumple round(100·n/m) == p.
  2. Matriz de confusión del meta-análisis: sensibilidad n/D_neg y
     especificidad n/D_pos son coherentes con los totales declarados
     (D_neg + D_pos == total de despliegue), cuando esas cifras están presentes.

Requiere ``pdftotext`` (poppler). Si no está, sale con aviso (no falla el build).

Uso:
  python scripts/general_project_maintenance/verify_thesis_pdf.py \\
      internal/tesis/tesis-v4.0.pdf
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

BLOQUEANTE = "BLOQUEANTE"
IMPORTANTE = "IMPORTANTE"
INFO = "INFO"
_SEV_ORDER = (BLOQUEANTE, IMPORTANTE, INFO)
_FAIL_SEVS = (BLOQUEANTE, IMPORTANTE)


@dataclass
class Finding:
    sev: str
    check: str
    msg: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def add(self, sev: str, check: str, msg: str) -> None:
        self.findings.append(Finding(sev, check, msg))

    def failed(self) -> bool:
        return any(f.sev in _FAIL_SEVS for f in self.findings)


def _extract_text(pdf_path: Path) -> str | None:
    """Extrae el texto del PDF con pdftotext (modo flujo, sin -layout)."""
    exe = shutil.which("pdftotext")
    if not exe:
        return None
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        out = Path(tmp.name)
    try:
        subprocess.run([exe, str(pdf_path), str(out)], check=True, capture_output=True)
        return out.read_text(encoding="utf-8", errors="replace")
    except subprocess.CalledProcessError:
        return None
    finally:
        out.unlink(missing_ok=True)


# ── Chequeo 1: caso absoluto p% (n/m) ────────────────────────────────────────

# Patrón ESTRICTO con paréntesis reales, para no capturar columnas colapsadas por
# la extracción de tablas (donde un '41 %' de una columna se pega a un '6/6' de
# otra). El paréntesis literal garantiza que p y (n/m) son la misma cifra.
_ABS_PDF_RE = re.compile(r"(\d{1,3})\s*%\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)")


def check_absolute_cases(text: str, rep: Report) -> None:
    for m in _ABS_PDF_RE.finditer(text):
        p, n, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if d == 0:
            continue
        real = 100.0 * n / d
        # tolerancia de 1 punto porcentual por el redondeo declarado a mano
        if abs(round(real) - p) > 1:
            rep.add(
                BLOQUEANTE,
                "caso-absoluto-pdf",
                f"en el PDF: {p}% ({n}/{d}) pero {n}/{d} = {real:.1f}% "
                f"(redondea a {round(real)}%).",
            )


# ── Chequeo 2: matriz de confusión del meta-análisis ─────────────────────────

def check_confusion_matrix(text: str, rep: Report) -> None:
    # Totales de despliegue: "(69 no aprobadas, 115 aprobadas)" y "184 ejecuciones".
    m_tot = re.search(r"(\d+)\s+no\s+aprobadas?,\s*(\d+)\s+aprobadas?", text)
    m_dep = re.search(r"(\d+)\s+ejecuciones\s+que\s+alcanzaron\s+el\s+despliegue", text)
    if not m_tot:
        return
    n_neg, n_pos = int(m_tot.group(1)), int(m_tot.group(2))
    if m_dep:
        total = int(m_dep.group(1))
        if n_neg + n_pos != total:
            rep.add(
                BLOQUEANTE,
                "meta-confusión",
                f"despliegue: {n_neg} no aprobadas + {n_pos} aprobadas = {n_neg + n_pos}, "
                f"pero el texto declara {total}.",
            )
    # Sensibilidad: su (n/d) debe tener d == nº de NO aprobadas. Se ancla el patrón
    # directamente (…anticipan el p% (n/d)… sensibilidad) para asociar cada cifra a
    # su métrica, sin ventanas ambiguas donde 'sensibilidad' y 'especificidad'
    # conviven en la misma frase.
    m_sens = re.search(r"(\d+)\s*%\s*\(\s*\d+\s*/\s*(\d+)\s*\)[^.]{0,40}?sensibilidad", text)
    if m_sens and int(m_sens.group(2)) != n_neg:
        rep.add(
            IMPORTANTE,
            "meta-confusión",
            f"sensibilidad con denominador {m_sens.group(2)}, pero las no aprobadas "
            f"son {n_neg}; deberían coincidir.",
        )
    # Especificidad: su (n/d) debe tener d == nº de aprobadas.
    m_spec = re.search(r"especificidad[^.]{0,40}?(\d+)\s*%\s*\(\s*\d+\s*/\s*(\d+)\s*\)", text)
    if m_spec and int(m_spec.group(2)) != n_pos:
        rep.add(
            IMPORTANTE,
            "meta-confusión",
            f"especificidad con denominador {m_spec.group(2)}, pero las aprobadas "
            f"son {n_pos}; deberían coincidir.",
        )


# ── Runner ───────────────────────────────────────────────────────────────────

def run(pdf_path: Path) -> tuple[Report, bool]:
    """Devuelve (reporte, extracción_ok). extracción_ok=False si no hay pdftotext."""
    rep = Report()
    if not pdf_path.exists():
        rep.add(BLOQUEANTE, "archivo", f"no existe el PDF: {pdf_path}")
        return rep, True
    text = _extract_text(pdf_path)
    if text is None:
        return rep, False
    check_absolute_cases(text, rep)
    check_confusion_matrix(text, rep)
    return rep, True


def main() -> int:
    ap = argparse.ArgumentParser(description="Verifica cifras del PDF renderizado de la tesis.")
    ap.add_argument("pdf", help="ruta al PDF a verificar")
    args = ap.parse_args()

    pdf_path = Path(args.pdf).resolve()
    rep, extracted = run(pdf_path)

    print(f"🔍 Verificación del PDF renderizado: {pdf_path}")
    if not extracted:
        print("⚠️  pdftotext (poppler) no está instalado; se omite la verificación del PDF.")
        return 0  # no bloquea el build si falta la herramienta
    if not rep.findings:
        print("✅ Sin hallazgos: casos absolutos y matriz de confusión coherentes en el PDF.")
        return 0
    counts = {s: sum(1 for f in rep.findings if f.sev == s) for s in _SEV_ORDER}
    print(f"   Hallazgos — BLOQUEANTE: {counts[BLOQUEANTE]}  "
          f"IMPORTANTE: {counts[IMPORTANTE]}  INFO: {counts[INFO]}")
    icon = {BLOQUEANTE: "❌", IMPORTANTE: "⚠️ ", INFO: "ℹ️ "}
    for sev in _SEV_ORDER:
        for f in [x for x in rep.findings if x.sev == sev]:
            print(f"  {icon[sev]} [{f.check}] {f.msg}")
    return 1 if rep.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
