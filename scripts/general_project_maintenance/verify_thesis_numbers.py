#!/usr/bin/env python3
"""Verificador de credibilidad numérica y de fuentes para la tesis LaTeX.

Complementa ``generate_thesis_tables.py --check-tex`` (que cubre estilo y
consistencia interna del .tex). Este script se centra en lo que un tribunal
comprueba con una calculadora y en lo que se desincroniza al regenerar tablas:

  1. Cocientes por fila en tablas de resultados embebidas a mano
     (|ΔE| / gap vs la columna ΔE/gap impresa).
  2. Coherencia texto ↔ tabla auto-generada (p. ej. el total de campaña que el
     texto declara vs el Total de auto_campaign.tex).
  3. Suma de columnas: en tablas de conteo, Total == Σ filas.
  4. Semántica A vs R: que el factor 4414× no se etiquete como "aceleración A".
  5. Colisiones de símbolos ya resueltas que no deben reaparecer
     (Δ como anisotropía, k como puntos de entrenamiento, d como profundidad,
     S como aceleración).
  6. Recursos referenciados que faltan en disco (\\includegraphics, \\input).

Todo es autocontenido (stdlib). No modifica el .tex. Devuelve exit code:
  0  sin hallazgos BLOQUEANTE/IMPORTANTE
  1  hay al menos un hallazgo BLOQUEANTE o IMPORTANTE

Uso:
  python scripts/general_project_maintenance/verify_thesis_numbers.py \\
      internal/tesis/tesis-v4.0.tex
  python .../verify_thesis_numbers.py <tex> --tables-dir internal/tesis/tables
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── Severidades ──────────────────────────────────────────────────────────────
BLOQUEANTE = "BLOQUEANTE"  # dato incorrecto / contradicción que un tribunal ve
IMPORTANTE = "IMPORTANTE"  # incoherencia que el lector notará
INFO = "INFO"  # aviso; no bloquea

_SEV_ORDER = (BLOQUEANTE, IMPORTANTE, INFO)
_FAIL_SEVS = (BLOQUEANTE, IMPORTANTE)


@dataclass
class Finding:
    sev: str
    check: str
    msg: str
    line: int | None = None


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def add(self, sev: str, check: str, msg: str, line: int | None = None) -> None:
        self.findings.append(Finding(sev, check, msg, line))

    def failed(self) -> bool:
        return any(f.sev in _FAIL_SEVS for f in self.findings)


# ── Utilidades de parseo ─────────────────────────────────────────────────────

def _strip_comments(text: str) -> str:
    """Quita comentarios LaTeX (% no escapado) conservando saltos de línea."""
    out = []
    for line in text.split("\n"):
        m = re.search(r"(?<!\\)%", line)
        out.append(line[: m.start()] if m else line)
    return "\n".join(out)


def _num_es(s: str) -> float | None:
    """Convierte un número en formato español ('0,075', '15,5') a float."""
    s = s.strip().replace("\\%", "").replace("%", "").strip()
    s = s.replace(".", "").replace(",", ".") if s.count(",") == 1 and "." in s else s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _iter_tabular_blocks(text: str):
    """Genera (start_line, header_line, rows) por cada entorno tabular.

    rows: lista de (line_number, [celdas]) para filas de datos (con &).
    """
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if re.search(r"\\begin\{tabular\}", lines[i]):
            start = i
            block_rows = []
            j = i + 1
            while j < len(lines) and not re.search(r"\\end\{tabular\}", lines[j]):
                raw = lines[j]
                if "&" in raw and not raw.lstrip().startswith("%"):
                    body = re.sub(r"\\\\.*$", "", raw)  # quita el fin de fila y lo que sigue
                    cells = [c.strip() for c in body.split("&")]
                    block_rows.append((j + 1, cells))
                j += 1
            yield (start + 1, block_rows)
            i = j
        i += 1


# ── Chequeo 1: cocientes |ΔE| / gap por fila ─────────────────────────────────

def check_row_ratios(text: str, rep: Report) -> None:
    """En tablas con columnas |ΔE|, gap y ΔE/gap, verifica que el cociente
    impreso coincida con |ΔE|/gap fila a fila (tolerancia por redondeo).

    Solo aplica cuando la tabla NO declara en su pie que ΔE/gap es 'media de
    cocientes por punto' (ese caso legítimamente no cuadra columna a columna).
    """
    lines = text.split("\n")
    for start_ln, rows in _iter_tabular_blocks(text):
        if not rows:
            continue
        # Buscar el pie/leyenda de la tabla (hasta 12 líneas tras \end{tabular})
        end_idx = start_ln
        for k in range(start_ln, min(start_ln + 200, len(lines))):
            if re.search(r"\\end\{tabular\}", lines[k]):
                end_idx = k
                break
        caption_region = "\n".join(lines[max(0, start_ln - 15): end_idx + 14])
        is_mean_of_ratios = bool(
            re.search(r"media de (?:los )?cocientes", caption_region, re.IGNORECASE)
        )
        if is_mean_of_ratios:
            continue  # legítimo que no cuadre columna a columna

        # Detectar el índice de columnas |ΔE|, gap, ΔE/gap por la fila de encabezado.
        # El encabezado suele ser la primera fila con '&' que contiene 'gap' o Delta.
        header = None
        for ln, cells in rows:
            joined = " ".join(cells).lower()
            if "gap" in joined and ("delta" in joined or "\\delta" in joined or "de/" in joined):
                header = cells
                break
        if not header:
            continue
        col_de = col_gap = col_ratio = None
        for idx, c in enumerate(header):
            cl = c.lower()
            if col_de is None and re.search(r"\\?\|?\\?delta e\|?", cl) and "gap" not in cl:
                col_de = idx
            if col_gap is None and re.search(r"\bgap\b|\\text\{gap\}", cl):
                col_gap = idx
            if col_ratio is None and "gap" in cl and ("delta" in cl or "/" in cl) and idx != col_gap:
                col_ratio = idx
        if col_de is None or col_gap is None or col_ratio is None:
            continue

        for ln, cells in rows:
            if cells is header:
                continue
            if max(col_de, col_gap, col_ratio) >= len(cells):
                continue
            de = _num_es(cells[col_de])
            gap = _num_es(cells[col_gap])
            printed = _num_es(cells[col_ratio])
            if de is None or gap is None or printed is None or gap == 0:
                continue
            computed_pct = 100.0 * de / gap
            # printed puede venir en % (con \%) o como fracción; normalizar a %
            printed_pct = printed if "%" in cells[col_ratio] or "\\%" in cells[col_ratio] else printed * 100
            # tolerancia: 0,3 puntos porcentuales o 8% relativo (redondeo de entradas)
            tol = max(0.3, 0.08 * computed_pct)
            if abs(printed_pct - computed_pct) > tol:
                rep.add(
                    BLOQUEANTE,
                    "cociente-fila",
                    f"|ΔE|/gap = {de:g}/{gap:g} = {computed_pct:.2f}%% pero la tabla "
                    f"imprime {printed_pct:.2f}%% (col {col_ratio + 1}).",
                    ln,
                )


# ── Chequeo 2: coherencia texto ↔ Total de auto_campaign ─────────────────────

def check_campaign_total(text: str, tables_dir: Path, rep: Report) -> None:
    camp = tables_dir / "auto_campaign.tex"
    if not camp.exists():
        rep.add(INFO, "campaña-total", f"no existe {camp} (no se puede contrastar el total).")
        return
    ct = camp.read_text(encoding="utf-8")
    m = re.search(r"\\textbf\{Total\}.*?\\textbf\{(\d+)\}", ct, re.DOTALL)
    if not m:
        rep.add(INFO, "campaña-total", "no se pudo leer el Total de auto_campaign.tex.")
        return
    total_tabla = int(m.group(1))
    # Totales declarados en el texto: "campaña total comprende N", "de las N ejecuciones"
    body = _strip_comments(text)
    declared = set()
    for pat in (
        r"campaña total comprende (\d+) ejecuciones",
        r"de las (\d+) ejecuciones de la campaña",
        r"fracción de las (\d+) ejecuciones",
    ):
        for mm in re.finditer(pat, body):
            declared.add(int(mm.group(1)))
    for d in sorted(declared):
        if d != total_tabla:
            rep.add(
                BLOQUEANTE,
                "campaña-total",
                f"el texto declara {d} ejecuciones pero auto_campaign.tex suma "
                f"{total_tabla}. Regenerar el número del texto tras 'make tables'.",
            )
    if not declared:
        rep.add(INFO, "campaña-total", f"auto_campaign Total={total_tabla}; el texto no lo cita explícitamente.")


# ── Chequeo 3: suma de columna Total == Σ filas (tablas de conteo) ───────────

def check_column_sums(tables_dir: Path, rep: Report) -> None:
    camp = tables_dir / "auto_campaign.tex"
    if not camp.exists():
        return
    ct = _strip_comments(camp.read_text(encoding="utf-8"))
    # Filas de datos: "Modelo & <ejec> & ..."; Total en \textbf{Total} & \textbf{N}
    per_row = []
    for m in re.finditer(r"^\s*([^&%\\][^&]*?)\s*&\s*(\d+)\s*&", ct, re.MULTILINE):
        per_row.append(int(m.group(2)))
    mt = re.search(r"\\textbf\{Total\}\s*&\s*\\textbf\{(\d+)\}", ct)
    if per_row and mt:
        suma = sum(per_row)
        total = int(mt.group(1))
        if suma != total:
            rep.add(
                BLOQUEANTE,
                "suma-columna",
                f"auto_campaign: Σ filas = {suma} pero Total impreso = {total}.",
            )


# ── Chequeo 4: semántica A vs R (el 4414× no es aceleración) ─────────────────

def check_speedup_semantics(text: str, rep: Report) -> None:
    body = _strip_comments(text)
    lines = body.split("\n")
    for i, line in enumerate(lines, 1):
        if "4414" not in line:
            continue
        # ¿se llama aceleración / factor A / speedup en la MISMA frase?
        window = line.lower()
        calls_it_A = re.search(
            r"(aceleraci[oó]n|factor de aceleraci[oó]n|\bA\b\s*[=≈]|speedup)\s*[^.]{0,40}4414",
            window,
        ) or re.search(r"4414[^.]{0,40}(aceleraci[oó]n|factor\s+de\s+aceleraci[oó]n|speedup)", window)
        mentions_R = re.search(r"raz[oó]n de error|\bR\b\s*[≈=]|\$R\b", line)
        if calls_it_A and not mentions_R:
            rep.add(
                IMPORTANTE,
                "A-vs-R",
                "4414× descrito como aceleración/factor A sin aclarar que es razón "
                "de error R (cociente de ΔE/gap, no de evaluaciones).",
                i,
            )


# ── Chequeo 5: colisiones de símbolos que no deben reaparecer ────────────────

def check_symbol_collisions(text: str, rep: Report) -> None:
    body = _strip_comments(text)
    lines = body.split("\n")
    for i, line in enumerate(lines, 1):
        # Δ como anisotropía XXZ: "XX + YY + \Delta ... ZZ" (debe ser \lambda)
        if re.search(r"XX\s*\+\s*YY\s*\+\s*\\Delta\b", line):
            rep.add(
                IMPORTANTE,
                "colisión-Δ",
                "Δ usado como anisotropía XXZ (colisiona con el gap espectral Δ=E1-E0); "
                "usar λ.",
                i,
            )
        # S como factor de aceleración (debe ser A; S es entropía)
        if re.search(r"(factor de aceleraci[oó]n|aceleraci[oó]n)\s*\$?S\$?", line) or re.search(
            r"\$S\$\s*=\s*r\s*\\cdot", line
        ):
            rep.add(
                IMPORTANTE,
                "colisión-S",
                "S usado como factor de aceleración (S es la entropía de entrelazamiento); usar A.",
                i,
            )
        # 'Ecuación'/'Ecuaciones' largo con \ref (estilo debe ser Ec./Ecs.)
        if re.search(r"Ecuaci[oó]n(?:es)?~\\ref", line):
            rep.add(
                INFO,
                "estilo-ref",
                "referencia 'Ecuación(es)~\\ref' (el estilo dominante es 'Ec.~'/'Ecs.~').",
                i,
            )
        # símbolo § con \ref (debe ser 'Sección')
        if re.search(r"\\S\\ref\{", line):
            rep.add(INFO, "estilo-ref", "referencia con '\\S\\ref' (usar 'Sección~\\ref').", i)


# ── Chequeo 6: recursos referenciados que faltan en disco ────────────────────

def check_missing_resources(text: str, tex_path: Path, rep: Report) -> None:
    base = tex_path.parent
    body = _strip_comments(text)
    lines = body.split("\n")
    for i, line in enumerate(lines, 1):
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", line):
            target = m.group(1)
            cands = [base / target] + [base / f"{target}{ext}" for ext in (".pdf", ".png", ".jpg", ".jpeg", ".eps")]
            if not any(c.exists() for c in cands):
                rep.add(
                    BLOQUEANTE,
                    "recurso-faltante",
                    f"\\includegraphics{{{target}}} no existe en disco (rompe la compilación).",
                    i,
                )
        for m in re.finditer(r"\\input\{([^}]+)\}", line):
            target = m.group(1)
            cands = [base / target, base / f"{target}.tex"]
            if not any(c.exists() for c in cands):
                rep.add(
                    BLOQUEANTE,
                    "recurso-faltante",
                    f"\\input{{{target}}} no existe en disco (rompe la compilación).",
                    i,
                )


# ── Chequeo 7: rango que mezcla aceleración A con razón de error R ───────────

# Un rango de "aceleración/mejora" cuyos extremos son un factor pequeño (~2–30×)
# y uno muy grande (cientos–miles×) mezcla dos magnitudes distintas: el factor de
# aceleración A (evaluaciones) y la razón de error R(N) (precisión). El steering
# lo prohíbe explícitamente (errores #14 y #16): p.ej. "2,5×–4400×" o "29×–500×".
_FACTOR = r"(\d+(?:[.,]\d+)?)\s*\$?\s*(?:\\times|×|x\b)\s*\$?"
_RANGE_MIX_RE = re.compile(
    _FACTOR + r"\s*(?:--|–|-|\ba\b|\bhasta\b|\by\b)\s*" + _FACTOR,
)


def check_speedup_range_mix(text: str, rep: Report) -> None:
    body = _strip_comments(text)
    lines = body.split("\n")
    for i, line in enumerate(lines, 1):
        low = line.lower()
        # Solo interesa cuando el rango se presenta como aceleración/mejora/speedup.
        if not re.search(r"acelera|mejora|speedup|\bA\b|factor", low):
            continue
        for m in _RANGE_MIX_RE.finditer(line):
            lo = _num_es(m.group(1))
            hi = _num_es(m.group(2))
            if lo is None or hi is None or lo == 0:
                continue
            # Rango sospechoso: extremos que difieren en >~1 orden de magnitud.
            # Umbral 10× separa el rango-mezcla A/R prohibido (p.ej. 29×–500× ≈ 17×,
            # 2,5×–4400× ≈ 1760×) de un rango legítimo de reducción de |ΔE| con p
            # (15×–55× ≈ 3,7×), que sí comparte magnitud.
            if hi / lo >= 10:
                rep.add(
                    BLOQUEANTE,
                    "rango-A/R",
                    f"rango de aceleración '{m.group(0).strip()}' mezcla dos magnitudes "
                    f"(factor {hi / lo:.0f}×): separar aceleración A (evaluaciones) de "
                    f"razón de error R(N) (precisión), steering §16.",
                    i,
                )


# ── Chequeo 8: letra 'x' como símbolo de factor (debe ser × / \times) ────────

# En prosa/tablas, un factor se escribe con × (\times), no con la letra x pegada a
# un número: '4400x', '62x'. Se excluyen usos math legítimos (x_i, eje x, 2x2 en
# nombres de rejilla no aplica aquí porque exigimos que sea 'mejora/aceleración').
_LETTER_X_FACTOR_RE = re.compile(r"(?<![\w\\])(\d+(?:[.,]\d+)?)x(?![\w])")


def check_letter_x_factor(text: str, rep: Report) -> None:
    body = _strip_comments(text)
    lines = body.split("\n")
    for i, line in enumerate(lines, 1):
        # neutralizar modo matemático inline para no marcar '2^x' u 'x_i'
        prose = re.sub(r"(?<!\\)\$[^$]*\$", "", line)
        low = prose.lower()
        if not re.search(r"acelera|mejora|speedup|factor|reduc|\brazón\b", low):
            continue
        for m in _LETTER_X_FACTOR_RE.finditer(prose):
            rep.add(
                IMPORTANTE,
                "x-factor",
                f"'{m.group(0)}' usa la letra 'x' como factor; usar '×' ('{m.group(1)}$\\times$'), "
                "steering §11.",
                i,
            )


# ── Chequeo 9: 'N máx' como número suelto (deben ser tres escalas) ───────────

# §16: la escala máxima del pipeline no es un número único, son tres regímenes
# (22 statevector / 40 DMRG / 250 MPS). Un 'N máx = <n>' suelto en la comparación
# con literatura es engañoso. Se acepta cuando la línea ya cita las tres escalas.
_NMAX_RE = re.compile(r"N\s*m[aá]x\w*\.?\s*[:=]?\s*(\d{1,3})", re.IGNORECASE)


def check_nmax_single(text: str, rep: Report) -> None:
    body = _strip_comments(text)
    lines = body.split("\n")
    for i, line in enumerate(lines, 1):
        # Si la línea ya declara las tres escalas (22/40/250 en cualquier orden), OK.
        if re.search(r"22\s*/\s*40\s*/\s*250|250.*40.*22", line):
            continue
        for m in _NMAX_RE.finditer(line):
            val = int(m.group(1))
            # Solo marcamos si el número suelto es una de las tres escalas canónicas
            # (o cercano): sugiere que se está dando UNA como si fuera 'la' máxima.
            if val in (22, 40, 250):
                rep.add(
                    IMPORTANTE,
                    "N-máx-suelto",
                    f"'N máx = {val}' dado como número único; el pipeline tiene tres "
                    "escalas (22 statevector / 40 DMRG / 250 MPS), steering §16.",
                    i,
                )


# ── Chequeo 10: aritmética de la frontera h_min(N) = a + b·N ─────────────────

# El texto da la ley de frontera como 'h_min ≈ a + b·N' y luego cita un valor
# evaluado a un N concreto (p.ej. 'a N=20 da 2,5'). Verifica a + b·N ≈ valor.
_HMIN_LAW_RE = re.compile(
    r"h_?\{?\\?min\}?\s*(?:\([^)]*\))?\s*(?:≈|\\approx|=)\s*"
    r"(\d+(?:[.,]\d+)?)\s*\+\s*(\d+(?:[.,]\d+)?)\s*\\?\s*N",
)


def check_frontier_arithmetic(text: str, rep: Report) -> None:
    body = _strip_comments(text)
    lines = body.split("\n")
    for i, line in enumerate(lines, 1):
        for m in _HMIN_LAW_RE.finditer(line):
            a = _num_es(m.group(1))
            b = _num_es(m.group(2))
            if a is None or b is None:
                continue
            # Buscar en la misma línea o la siguiente un 'a N=<n> ... <valor>'.
            ctx = line + " " + (lines[i] if i < len(lines) else "")
            for ev in re.finditer(
                r"N\s*=\s*(\d{1,3})[^0-9]{0,40}?(\d+(?:[.,]\d+)?)", ctx
            ):
                n = int(ev.group(1))
                cited = _num_es(ev.group(2))
                if cited is None:
                    continue
                predicted = a + b * n
                # tolerancia por redondeo del texto (1 decimal): 0,1
                if abs(predicted - cited) > 0.15 and abs(predicted - cited) / max(cited, 0.1) > 0.1:
                    rep.add(
                        BLOQUEANTE,
                        "frontera-aritmética",
                        f"h_min = {a:g} + {b:g}·N a N={n} da {predicted:.2f}, pero el texto "
                        f"cita {cited:g}.",
                        i,
                    )
                break  # solo el primer 'N=' relevante por ley


# ── Chequeo 11: convenio de compuertas de dos qubits CZ = 2·E·p ──────────────

# §4.5: cada término R_ZZ se compila como CX·Rz·CX = 2 compuertas de dos qubits.
# Un circuito con E enlaces activos por capa y profundidad p usa 2·E·p. Cuando el
# .tex declara un número de CZ junto con 'E enlaces × 2' (o × 2 explícito), se
# verifica la aritmética. Instancias reales: '10 CZ a N=6, p=1' (cadena, E=5),
# '18 CZ a N=6, p=1: 9 enlaces × 2'.
_CZ_EDGES_RE = re.compile(
    r"(\d+)\s*(?:CZ|CX)\b[^.]{0,60}?(\d+)\s*enlaces?\s*\$?\\?times\$?\s*2",
    re.IGNORECASE,
)


def check_cz_convention(text: str, rep: Report) -> None:
    body = _strip_comments(text)
    lines = body.split("\n")
    for i, line in enumerate(lines, 1):
        for m in _CZ_EDGES_RE.finditer(line):
            n_cz = int(m.group(1))
            n_edges = int(m.group(2))
            # p en la misma cláusula (por defecto 1 si dice explícitamente p=1).
            pm = re.search(r"p\s*=\s*(\d+)", line)
            p = int(pm.group(1)) if pm else 1
            expected = 2 * n_edges * p
            if n_cz != expected:
                rep.add(
                    BLOQUEANTE,
                    "CZ-2Ep",
                    f"{n_cz} CZ declaradas pero {n_edges} enlaces × 2 × p={p} = "
                    f"{expected} (convenio 2·E·p, §4.5).",
                    i,
                )


# ── Chequeo 12: escala de calificación A–E monótona y sin solape ─────────────

# El scoreboard clasifica por |ΔE|: A (<0,05), B (<0,10), C (<0,30), D (<1,00),
# E (≥1,00). Los umbrales deben ser estrictamente crecientes. Un solapamiento o
# desorden (p.ej. B <0,04) rompe la clasificación.
_GRADE_RE = re.compile(r"([A-E])\s*\(\s*\$?\s*<\s*(\d+(?:[.,]\d+)?)\s*\$?\s*\)")


def check_grade_scale(text: str, tables_dir: Path, rep: Report) -> None:
    sources = [text]
    sb = tables_dir / "auto_scoreboard.tex"
    if sb.exists():
        sources.append(sb.read_text(encoding="utf-8"))
    for src in sources:
        for chunk_ln, chunk in _grade_chunks(src):
            pairs = [(g, _num_es(v)) for g, v in _GRADE_RE.findall(chunk)]
            pairs = [(g, v) for g, v in pairs if v is not None]
            if len(pairs) < 2:
                continue
            for (g1, v1), (g2, v2) in zip(pairs, pairs[1:]):
                if v2 <= v1:
                    rep.add(
                        BLOQUEANTE,
                        "escala-calificación",
                        f"umbral {g2} (<{v2:g}) no es mayor que {g1} (<{v1:g}): "
                        "la escala A–E debe ser estrictamente creciente.",
                        chunk_ln,
                    )
                    break


def _grade_chunks(src: str):
    """Genera (línea, fragmento) donde aparecen 2+ umbrales A/B/.. juntos."""
    lines = src.split("\n")
    for i, line in enumerate(lines, 1):
        if len(_GRADE_RE.findall(line)) >= 2:
            yield i, line


# ── Chequeo 13: sumas aritméticas escritas explícitamente en prosa ───────────

# El texto escribe sumas de conteo como "35 XXZ + 40 transversal = 75 ejecuciones"
# o "35 + 40 = 75". Son verificables sin mapear semántica: X + Y (+ Z...) == Total.
# Determinista y seguro: solo se activa cuando hay un '=' con operandos numéricos.
# Forma "operandos = total": '35 XXZ + 40 transversal = 75'. El '=' explícito con
# operandos unidos por '+' garantiza que es una suma (determinista, seguro). La
# forma en lenguaje natural ('X ... y Y ...') se omite a propósito: distinguir una
# enumeración sumable de dos números cualesquiera requiere criterio humano.
_SUM_EQ_RE = re.compile(r"(\d+)(?:[^=\d\n]*?\+\s*(?:\d+))+[^=\d\n]*?=\s*(\d+)")
# Operadores no aditivos que invalidan un span como suma pura.
_NON_ADDITIVE_RE = re.compile(r"\\?times|×|[*/^]|(?<![,\d])-(?!-)")


def check_written_sums(text: str, rep: Report) -> None:
    body = _strip_comments(text)
    lines = body.split("\n")
    for i, line in enumerate(lines, 1):
        for m in _SUM_EQ_RE.finditer(line):
            span = m.group(0)
            if _NON_ADDITIVE_RE.search(span):
                continue
            nums = [int(x) for x in re.findall(r"\d+", span)]
            if len(nums) < 3:
                continue
            *operands, total = nums
            if sum(operands) != total:
                rep.add(
                    BLOQUEANTE,
                    "suma-escrita",
                    f"suma escrita '{span.strip()}': {' + '.join(map(str, operands))} = "
                    f"{sum(operands)}, no {total}.",
                    i,
                )


# ── Chequeo 14: integridad de la lista canónica de topologías en tablas ──────

# §16: la lista canónica de topologías evaluadas es {cadena 1D, escalera,
# triangular, cuadrada, heavy-hex}. kagomé es trabajo futuro (error #7) y XY no
# forma parte de la narrativa (error #17): ninguna debe aparecer como FILA de una
# tabla de resultados. El chequeo se restringe a filas de tabular (celda inicial),
# para no marcar menciones legítimas en prosa, motivación o trabajo futuro.
# Lista canónica de referencia: cadena 1D, escalera, triangular, cuadrada, heavy-hex.
_FORBIDDEN_TOPO_RE = re.compile(r"(kagom[eé]|\bXY\b)", re.IGNORECASE)


def check_topology_canon(text: str, rep: Report) -> None:
    for start_ln, rows in _iter_tabular_blocks(text):
        for ln, cells in rows:
            if not cells:
                continue
            first = cells[0]
            m = _FORBIDDEN_TOPO_RE.search(first)
            if m:
                rep.add(
                    IMPORTANTE,
                    "topología-canónica",
                    f"'{m.group(0)}' aparece como fila de una tabla de resultados; "
                    "kagomé es trabajo futuro (§16 #7) y XY no es narrativa canónica "
                    "(§16 #17): no deben figurar como topología evaluada.",
                    ln,
                )


# ── Runner ───────────────────────────────────────────────────────────────────

def run(tex_path: Path, tables_dir: Path) -> Report:
    rep = Report()
    if not tex_path.exists():
        rep.add(BLOQUEANTE, "archivo", f"no existe el .tex: {tex_path}")
        return rep
    raw = tex_path.read_text(encoding="utf-8")
    text = _strip_comments(raw)

    check_row_ratios(text, rep)
    check_campaign_total(raw, tables_dir, rep)
    check_column_sums(tables_dir, rep)
    check_speedup_semantics(raw, rep)
    check_symbol_collisions(raw, rep)
    check_missing_resources(raw, tex_path, rep)
    check_speedup_range_mix(raw, rep)
    check_letter_x_factor(raw, rep)
    check_nmax_single(raw, rep)
    check_frontier_arithmetic(raw, rep)
    check_cz_convention(raw, rep)
    check_grade_scale(raw, tables_dir, rep)
    check_written_sums(raw, rep)
    check_topology_canon(text, rep)
    return rep


def _print_report(rep: Report, tex_path: Path) -> None:
    print(f"🔍 Verificación numérica y de fuentes: {tex_path}")
    if not rep.findings:
        print("✅ Sin hallazgos: cocientes, totales, símbolos y recursos coherentes.")
        return
    counts = {s: sum(1 for f in rep.findings if f.sev == s) for s in _SEV_ORDER}
    print(f"   Hallazgos — BLOQUEANTE: {counts[BLOQUEANTE]}  "
          f"IMPORTANTE: {counts[IMPORTANTE]}  INFO: {counts[INFO]}")
    icon = {BLOQUEANTE: "❌", IMPORTANTE: "⚠️ ", INFO: "ℹ️ "}
    for sev in _SEV_ORDER:
        for f in [x for x in rep.findings if x.sev == sev]:
            loc = f"L{f.line}" if f.line else "—"
            print(f"  {icon[sev]} [{f.check}] {loc}: {f.msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Verifica credibilidad numérica de la tesis LaTeX.")
    ap.add_argument("tex", help="ruta al .tex a verificar")
    ap.add_argument(
        "--tables-dir",
        default=None,
        help="carpeta de tablas auto (default: <dir del tex>/tables)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="devolver exit 1 también si hay hallazgos INFO",
    )
    args = ap.parse_args()

    tex_path = Path(args.tex).resolve()
    tables_dir = Path(args.tables_dir).resolve() if args.tables_dir else tex_path.parent / "tables"

    rep = run(tex_path, tables_dir)
    _print_report(rep, tex_path)

    if args.strict and rep.findings:
        return 1
    return 1 if rep.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
