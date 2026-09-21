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
  7. Coherencia cross-N: la tabla de transferencia entre tamaños no debe publicar
     tasas de éxito concretas mientras el texto declara ese régimen como límite
     abierto (el "100%/86%" previo venía del pass_rate de secciones-del-runner,
     no de la aprobación de puntos físicos).
  8. Integridad LaTeX estática (sin compilar): \\ref a un label inexistente,
     \\cite sin \\bibitem, \\label duplicado. Es la red rápida del hook
     PostFileSave; 'make verify' hace la validación pesada auditando el .log tras
     compilar. Además: celda de tasa/fidelidad > 100 % (imposible por definición).
  9. Centinelas de regresión de correcciones ya aplicadas (informe de corrección):
     - kbar-valor-único (§1.8): A = r·k̄ con el k̄ de cada configuración, no un
       valor exclusivo (37--85 nombrado junto a 4--7).
     - hmin-reconciliación (§1.7): el promedio h_min(p=3)≈1,6 declarado como
       promedio de las cinco topologías, no atribuido a cadena 1D.
     - frontera-independiente-N: la frontera h_min NO se declara 'independiente de
       N' a p>=3 (pendiente positiva pequeña, no nula).
     - hmin-topo-profundidad: los h_min por topología se etiquetan con la p real
       (los óptimos 1,09/1,12/2,20 son de p=7--8, no de p=3--4; a p=4 son
       1,18/1,31/1,84/1,88/2,72, fuente results/H_FRONTIER_TOPOLOGIES.md).
     - hmin-topo-fuente: ancla cada h_min por topología del .tex a la matriz de
       verdad results/H_FRONTIER_TOPOLOGIES.md a la p declarada; si un valor no
       coincide, nombra la p real a la que corresponde (caza números atribuidos a
       una (topología, p) que no les toca; sin la fuente → INFO 'omitido').
     - frontera-pendiente: los ajustes h_min = a + b·N deben tener b > 0 (la
       frontera crece con N; base de H2). Un b <= 0 sería incoherente.

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
        caption_region = "\n".join(lines[max(0, start_ln - 15) : end_idx + 14])
        is_mean_of_ratios = bool(re.search(r"media de (?:los )?cocientes", caption_region, re.IGNORECASE))
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
                "Δ usado como anisotropía XXZ (colisiona con el gap espectral Δ=E1-E0); usar λ.",
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
        # Saltar definiciones de macro: un \includegraphics/\input dentro de un
        # \newcommand/\def usa parámetros (#1, #2), no una ruta real. Marcarlo daría
        # un falso positivo (p. ej. la macro \figinput que conmuta PDF vs TikZ).
        if re.search(r"\\(?:re)?newcommand|\\def\b|\\providecommand", line) or "#" in line:
            continue
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
                f"'{m.group(0)}' usa la letra 'x' como factor; usar '×' ('{m.group(1)}$\\times$'), steering §11.",
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
    r"(\d+(?:[.,]\d+)?)\s*\+\s*(\d+(?:[.,]\d+)?)\s*"
    # separador antes de N: '\cdot', '\times', '\,', '·', '*', espacio, o nada
    # (tolera el comando LaTeX completo, no solo un backslash suelto — el .tex
    # usa '\cdot N'). Sin esto el regex nunca capturaba las Ecs. reales.
    r"(?:\\cdot|\\times|\\,|·|\*)?\s*N",
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
            # Neutralizar los RANGOS de validez 'N = X--Y' (X, Y son cotas del
            # ajuste, no un valor evaluado): sin esto se leería 'N=20--250' como
            # 'a N=20 da 250' (falso positivo). También 'R^2 = ...' no es un valor
            # de h_min. Se eliminan esos tramos antes de buscar evaluaciones.
            ctx = re.sub(r"N\s*=\s*\d{1,3}\s*(?:--|[–-])\s*\d{1,3}", " ", ctx)
            ctx = re.sub(r"R\^?2?\s*=\s*[\d.,]+", " ", ctx)
            # Solo aceptar formas EXPLÍCITAS de evaluación: 'a/para N=<n> ... da/es/:
            # <valor>' con un h_min plausible (0..6). Evita capturar cualquier par.
            for ev in re.finditer(
                r"(?:a|para|en)\s+N\s*=\s*(\d{1,3})[^0-9]{0,25}?"
                r"(?:da|es|vale|:|=|\\approx|≈)?\s*(\d(?:[.,]\d+)?)\b",
                ctx,
            ):
                n = int(ev.group(1))
                cited = _num_es(ev.group(2))
                if cited is None or cited > 6:
                    continue
                predicted = a + b * n
                # tolerancia por redondeo del texto (1 decimal): 0,1
                if abs(predicted - cited) > 0.15 and abs(predicted - cited) / max(cited, 0.1) > 0.1:
                    rep.add(
                        BLOQUEANTE,
                        "frontera-aritmética",
                        f"h_min = {a:g} + {b:g}·N a N={n} da {predicted:.2f}, pero el texto cita {cited:g}.",
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
                    f"{n_cz} CZ declaradas pero {n_edges} enlaces × 2 × p={p} = {expected} (convenio 2·E·p, §4.5).",
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
            for (g1, v1), (g2, v2) in zip(pairs, pairs[1:], strict=False):
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
                    f"suma escrita '{span.strip()}': {' + '.join(map(str, operands))} = {sum(operands)}, no {total}.",
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


# ── Chequeo 15: coherencia de la afirmación de generalización cross-N ────────
# Contexto: la tabla cross_n_transfer llegó a publicar tasas 86–100% que provenían
# del pass_rate de secciones-del-runner (n_ok/n_secciones), NO de la aprobación de
# puntos físicos, contradiciendo el texto que dice que la generalización a N grande
# "falla por insuficiencia de datos". Este chequeo detecta esa contradicción exacta:
# si la tabla de transferencia entre tamaños publica tasas de éxito concretas y a la
# vez el documento declara ese régimen como límite abierto, hay contradicción interna.

_CROSS_N_TABLE_LABEL = "tab:cross_n_transfer"
_RATE_CELL_RE = re.compile(r"\b\d{1,3}\s*\\?%")
_OPEN_LIMIT_RE = re.compile(
    r"(falla por insuficiencia|queda como línea (?:de trabajo )?abierta|"
    r"queda abiert[oa]|no está establecida|límite abierto|frontera de investigación)"
)


def _tabular_after_label(text: str, label: str):
    """Devuelve (start_line, rows) del primer tabular que sigue a un \\label dado."""
    lines = text.split("\n")
    lbl_ln = next((i for i, ln in enumerate(lines) if f"\\label{{{label}}}" in ln), None)
    if lbl_ln is None:
        return None
    tail = "\n".join(lines[lbl_ln:])
    for start_off, rows in _iter_tabular_blocks(tail):
        abs_rows = [(ln + lbl_ln, cells) for ln, cells in rows]
        return (lbl_ln + start_off, abs_rows)
    return None


def check_cross_n_claim_consistency(text: str, rep: Report) -> None:
    found = _tabular_after_label(text, _CROSS_N_TABLE_LABEL)
    if found is None:
        return
    _start, rows = found
    # Última celda de cada fila de datos = columna "Tasa aprob."
    numeric_rate_lines = [ln for ln, cells in rows if cells and _RATE_CELL_RE.search(cells[-1])]
    if not numeric_rate_lines:
        return  # tabla con --- (no reproducible): estado correcto, sin hallazgo
    if _OPEN_LIMIT_RE.search(text):
        rep.add(
            BLOQUEANTE,
            "cross-n-coherencia",
            f"la tabla {_CROSS_N_TABLE_LABEL} publica tasas de aprobación concretas "
            f"(líneas {numeric_rate_lines}) mientras el documento declara la "
            "generalización entre tamaños como límite abierto / que falla a N grande. "
            "Verificar que las tasas no provengan del pass_rate de secciones-del-runner "
            "(n_ok/n_secciones) en lugar de la aprobación de puntos físicos "
            "(model_registry pass_rate_by_n / n_pass/n_total del eval report). "
            "Si no son reproducibles desde los stores, marcar la columna con --- "
            "(steering §1, fidelidad al dato).",
            numeric_rate_lines[0],
        )


# ── Chequeo: comando/fragmento shell pegado en la prosa (informe 1.1) ────────

# Detecta órdenes de shell o LaTeX de edición que se colaron en el cuerpo: el
# caso real fue 'make -C internal/tesis all' pegado dentro de "Se denota por p".
# También marca \vspace/\bigskip sueltos en el cuerpo (residuos de edición).
# Solo marcadores INEQUÍVOCOS de shell (evita falsos positivos con math LaTeX
# como $(i,j)$ o comillas ``...''): make -C, rutas de venv/scripts, banderas
# largas de comando pegadas a texto.
_SHELL_IN_PROSE_RE = re.compile(
    r"make\s+-C\b|\.venv/bin/|\bpython\s+scripts/\S|\bpip\s+install\b|"
    r"\bcd\s+/\w|--out-dir\b|--check-tex\b"
)
_STRAY_SPACING_RE = re.compile(r"(?<!%)\\(?:vspace|bigskip|medskip|smallskip)\b")


def check_command_in_prose(text: str, rep: Report) -> None:
    body = _strip_comments(text)
    lines = body.split("\n")
    in_verbatim = False
    for i, line in enumerate(lines, 1):
        if re.search(r"\\begin\{(verbatim|lstlisting|minted)\}", line):
            in_verbatim = True
        if re.search(r"\\end\{(verbatim|lstlisting|minted)\}", line):
            in_verbatim = False
            continue
        if in_verbatim:
            continue
        # neutralizar \texttt{...}, \url{...}, \path{...} (comandos donde un
        # fragmento shell/comando es legítimo, p. ej. citar un make en \texttt).
        prose = re.sub(r"\\(?:texttt|url|path|href|lstinline|verb)\b\s*\{[^}]*\}", "", line)
        prose = re.sub(r"\\verb\|[^|]*\|", "", prose)
        m = _SHELL_IN_PROSE_RE.search(prose)
        if m:
            rep.add(
                BLOQUEANTE,
                "comando-en-prosa",
                f"fragmento de shell/comando '{m.group(0).strip()}' en el cuerpo del "
                "texto (¿pegado por error del portapapeles?). Debe ir en \\texttt{} o "
                "en un entorno de código, no en la prosa (informe §1.1).",
                i,
            )


def check_stray_spacing(text: str, rep: Report) -> None:
    """Marca comandos de espaciado CONSECUTIVOS en el cuerpo (residuos de edición).

    Un único \\vspace es un patrón legítimo (p. ej. antes de 'Palabras clave').
    La firma del residuo que señaló el informe (p. 38, 'tres seguidos') es DOS o
    más comandos de espaciado en líneas contiguas sin texto entre ellos."""
    lines = _strip_comments(text).split("\n")
    spacing_re = re.compile(r"\\(?:vspace|bigskip|medskip|smallskip)\b(\{[^}]*\})?\*?")

    def is_spacing_only(s: str) -> bool:
        return bool(s) and bool(re.fullmatch(spacing_re, s))

    run_start = None
    run_len = 0
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if is_spacing_only(stripped):
            if run_len == 0:
                run_start = i
            run_len += 1
        elif stripped == "":
            continue  # las líneas en blanco no rompen la racha de espaciado
        else:
            if run_len >= 2:
                ctx = " ".join(lines[max(0, run_start - 3) : run_start + run_len + 1])
                if not re.search(r"\\(begin|end)\{(figure|table)", ctx):
                    rep.add(
                        INFO,
                        "espaciado-suelto",
                        f"{run_len} comandos de espaciado consecutivos desde la línea "
                        f"{run_start}: probable residuo de edición (informe §1.1, "
                        "p. 27/p. 38).",
                        run_start,
                    )
            run_len = 0
            run_start = None


# ── Chequeo: aritmética de semillas (informe 1.5, 1.6) ───────────────────────

# "18 configuraciones ... 3-4 semillas ... 79 ejecuciones": verifica C*a <= T <= C*b.
# También "N topologias x P profundidades ... T ejecuciones" con >= min semillas.
_SEEDS_ARITH_RE = re.compile(
    r"(\d+)\s+configuraciones[^.]{0,80}?(\d+)\s*(?:--|–|-|\ba\b)\s*(\d+)\s+semillas"
    r"[^.]{0,80}?(\d+)\s+ejecuciones",
    re.IGNORECASE,
)
# "5 topologias x 3 profundidades ... (21 ejecuciones)" con regla de >=3 semillas.
_TOPO_DEPTH_RUNS_RE = re.compile(
    r"(\d+)\s+topolog[ií]as?\s*(?:×|x|\\times)\s*(\d+)\s+profundidades?"
    r"[^.]{0,90}?\(?(\d+)\s+ejecuciones\)?",
    re.IGNORECASE,
)
# "15 configuraciones, 84 ejecuciones" SIN rango de semillas explícito: aplica la
# regla de >=3 semillas por configuración (min = N_cfg * 3). Complementa a
# _SEEDS_ARITH_RE (que exige el rango de semillas en la misma frase).
_CFG_RUNS_RE = re.compile(
    r"(\d+)\s+configuraciones\s*,\s*(\d+)\s+ejecuciones",
    re.IGNORECASE,
)
_MIN_SEEDS = 3  # regla del documento (§4.6): cada configuracion >= 3 semillas


def check_seeds_arithmetic(text: str, rep: Report) -> None:
    body = _strip_comments(text)
    for m in _SEEDS_ARITH_RE.finditer(body):
        n_cfg, s_lo, s_hi, total = (int(m.group(k)) for k in (1, 2, 3, 4))
        if s_lo > s_hi:
            continue
        lo, hi = n_cfg * s_lo, n_cfg * s_hi
        if not (lo <= total <= hi):
            ln = body.count("\n", 0, m.start()) + 1
            rep.add(
                BLOQUEANTE,
                "aritmetica-semillas",
                f"{n_cfg} configuraciones con {s_lo}--{s_hi} semillas dan entre {lo} y "
                f"{hi} ejecuciones, no {total} (informe §1.5). Corregir el rango de "
                "semillas, el nº de configuraciones o el total.",
                ln,
            )
    for m in _TOPO_DEPTH_RUNS_RE.finditer(body):
        n_topo, n_depth, total = (int(m.group(k)) for k in (1, 2, 3))
        n_cfg = n_topo * n_depth
        min_runs = n_cfg * _MIN_SEEDS
        if total < min_runs:
            ln = body.count("\n", 0, m.start()) + 1
            rep.add(
                BLOQUEANTE,
                "aritmetica-semillas",
                f"{n_topo}×{n_depth} = {n_cfg} configuraciones con ≥{_MIN_SEEDS} semillas "
                f"exigen ≥{min_runs} ejecuciones, pero se declaran {total} (informe §1.6). "
                "Corregir la cobertura o aclarar el subconjunto.",
                ln,
            )
    for m in _CFG_RUNS_RE.finditer(body):
        n_cfg, total = int(m.group(1)), int(m.group(2))
        min_runs = n_cfg * _MIN_SEEDS
        if total < min_runs:
            ln = body.count("\n", 0, m.start()) + 1
            rep.add(
                BLOQUEANTE,
                "aritmetica-semillas",
                f"{n_cfg} configuraciones con ≥{_MIN_SEEDS} semillas exigen ≥{min_runs} "
                f"ejecuciones, pero se declaran {total} (informe §1.6). Corregir la "
                "cobertura o aclarar el subconjunto.",
                ln,
            )


# ── Chequeo: subconjunto de ejecuciones vs total de campaña (informe §1.7) ───

# "84 ejecuciones ... subconjunto de las 189" — el subconjunto no puede superar el
# total, y ese total debe coincidir con una fila de auto_campaign.tex. Caza el error
# del corrector (comparar un parcial con un total sin declarar de qué es subconjunto,
# o citar un total que no existe en la tabla de conteo).
_SUBSET_RE = re.compile(
    r"(\d+)\s+ejecuciones[^.]{0,80}?subconjunto de las\s+(\d+)",
    re.IGNORECASE,
)


def _campaign_row_totals(tables_dir: Path) -> set[int]:
    """Conjunto de totales por-modelo de auto_campaign.tex (columna de ejecuciones).

    Reutiliza el patrón de fila de ``check_column_sums`` (Modelo & <n> & ...), sin
    duplicar la lógica de parseo; devuelve {} si la tabla no existe.
    """
    camp = tables_dir / "auto_campaign.tex"
    if not camp.exists():
        return set()
    ct = _strip_comments(camp.read_text(encoding="utf-8"))
    totals = {int(m.group(2)) for m in re.finditer(r"^\s*([^&%\\][^&]*?)\s*&\s*(\d+)\s*&", ct, re.MULTILINE)}
    mt = re.search(r"\\textbf\{Total\}\s*&\s*\\textbf\{(\d+)\}", ct)
    if mt:
        totals.add(int(mt.group(1)))  # el Total global también es un ancla válida
    return totals


def check_subset_of_campaign(text: str, tables_dir: Path, rep: Report) -> None:
    body = _strip_comments(text)
    row_totals = _campaign_row_totals(tables_dir)
    for m in _SUBSET_RE.finditer(body):
        subset, declared_total = int(m.group(1)), int(m.group(2))
        ln = body.count("\n", 0, m.start()) + 1
        if subset > declared_total:
            rep.add(
                BLOQUEANTE,
                "subconjunto-campana",
                f"se declara un subconjunto de {subset} ejecuciones sobre un total de "
                f"{declared_total}, pero el subconjunto no puede superar al total "
                "(informe §1.7). Corregir una de las dos cifras.",
                ln,
            )
        if row_totals and declared_total not in row_totals:
            rep.add(
                BLOQUEANTE,
                "subconjunto-campana",
                f"el total de {declared_total} ejecuciones citado como referencia del "
                "subconjunto no coincide con ninguna fila (ni el Total) de "
                f"auto_campaign.tex ({sorted(row_totals)}). Regenerar el número tras "
                "'make tables' o citar el total correcto.",
                ln,
            )


# ── Chequeo: recuento de referencias por año (informe 2.5) ───────────────────

# "40+ referencias 2022-2026" / "más de 40 referencias de 2022 a 2026" vs conteo real.
# El nº que cuenta es el ADYACENTE al rango de años: en "50 referencias, 30 de
# ellas de 2022--2026" el 30 (no el 50) es el que afirma el rango. Por eso se
# captura el ÚLTIMO número antes del rango (grupo 2 si existe, si no el grupo 1).
_REFCOUNT_CLAIM_RE = re.compile(
    r"(\d+)\s*\+?\s*referencias?[^.\n]{0,30}?(?:(\d+)\s*(?:\+|de\s+ellas)?[^.\n]{0,10}?)?"
    r"(\d{4})\s*(?:--|–|-|a|y)\s*(\d{4})",
    re.IGNORECASE,
)


def check_reference_count(text: str, rep: Report) -> None:
    # Delimitar el entorno thebibliography y contar un año por \bibitem tomando
    # el PRIMER (YYYY) que sigue a cada \bibitem (el año de publicación de esa
    # entrada), robusto a que las entradas estén en una o varias líneas.
    m_bib = re.search(r"\\begin\{thebibliography\}(.*?)\\end\{thebibliography\}", text, re.DOTALL)
    bib_block = m_bib.group(1) if m_bib else text
    parts = re.split(r"\\bibitem(?:\[[^\]]*\])?\{[^}]*\}", bib_block)[1:]
    years: list[int] = []
    for entry in parts:
        ym = re.search(r"\((\d{4})[a-z]?\)", entry)
        if ym:
            years.append(int(ym.group(1)))
    if not years:
        return
    body = _strip_comments(text)
    for m in _REFCOUNT_CLAIM_RE.finditer(body):
        # El nº que afirma el rango es el adyacente a los años (grupo 2 si existe;
        # p. ej. '30 de ellas de 2022--2026'), no el total (grupo 1).
        claimed = int(m.group(2)) if m.group(2) else int(m.group(1))
        y0, y1 = int(m.group(3)), int(m.group(4))
        actual = sum(1 for y in years if y0 <= y <= y1)
        # "40+" afirma "al menos 40": solo es falso si el real es MENOR que lo dicho.
        if actual < claimed:
            ln = body.count("\n", 0, m.start()) + 1
            rep.add(
                IMPORTANTE,
                "recuento-referencias",
                f"el texto afirma {claimed}+ referencias de {y0}--{y1}, pero el listado "
                f"tiene {actual} en ese rango (de {len(years)} con año) (informe §2.5). "
                f"Usar la cifra real (p. ej. '{len(years)} referencias, {actual} de "
                f"{y0}--{y1}').",
                ln,
            )


# ── Chequeo: tabla con columna de resultado vacía citada como evidencia (1.2) ─

# Palabras de "afirmación de respaldo" que, si citan por \ref una tabla cuya
# columna de resultado está toda en '---', delatan el patrón del informe §1.2:
# una tabla en blanco usada como evidencia (H4/OE4).
_CLAIM_WORDS_RE = re.compile(
    r"respaldad|se\s+establece|se\s+sostiene|demuestra|confirma|"
    r"satisface[n]?\s+\$?\\?Delta|evidencia|verificad",
    re.IGNORECASE,
)
_DASH_CELL_RE = re.compile(r"^\s*(?:---|--|—|-)\s*$")


def check_empty_cited_table(text: str, rep: Report) -> None:
    """Marca una tabla cuya única columna de resultado está toda en '---' cuando
    el documento la cita (\\ref) desde una frase de respaldo. Es el patrón §1.2."""
    for start_ln, rows in _iter_tabular_blocks(text):
        # Excluir la fila de encabezado (celdas en \textbf o sin dato numérico ni
        # guion): solo interesan las filas de datos.
        data_rows = [(ln, cells) for ln, cells in rows if len(cells) >= 2 and "\\textbf" not in cells[-1]]
        if len(data_rows) < 2:
            continue
        # Localizar el \label de esta tabla (mirando ~8 líneas antes del tabular).
        lines = text.split("\n")
        header_zone = "\n".join(lines[max(0, start_ln - 9) : start_ln])
        lm = re.search(r"\\label\{(tab:[^}]+)\}", header_zone)
        if not lm:
            continue
        label = lm.group(1)
        # ¿La última columna (resultado) está toda en '---' en las filas de datos?
        last_cells = [cells[-1] for _, cells in data_rows]
        if not last_cells or not all(_DASH_CELL_RE.match(c) for c in last_cells):
            continue
        # ¿El documento cita esta tabla desde una frase de respaldo?
        cited_supported = False
        for m in re.finditer(r"\\ref\{" + re.escape(label) + r"\}", text):
            window = text[max(0, m.start() - 220) : m.end() + 60]
            if _CLAIM_WORDS_RE.search(window):
                cited_supported = True
                break
        if cited_supported:
            rep.add(
                BLOQUEANTE,
                "tabla-vacia-citada",
                f"la tabla {label} tiene la columna de resultado toda en '---' pero se "
                "cita como evidencia de una afirmación de respaldo (informe §1.2). "
                "Rellenar la columna desde los datos o retirar la tabla y acotar la "
                "afirmación a lo documentado.",
                start_ln,
            )


# ── Chequeo: valor publicado para una config declarada "no evaluada" (1.4) ────

# Cruza celdas '---'/"no evaluada" de una tabla contra celdas con valor numérico
# para la misma (topología, p) en otra tabla. Firma del informe §1.4 (cadena 1D
# p=4 marcada no-ejecutada en Tabla 5.4 pero con |ΔE| en Tabla B.2).
_TOPO_ROW_RE = re.compile(r"(cadena\s*1d|heavy[- ]?hex|escalera|cuadrada|triangular)", re.IGNORECASE)


def _topo_key(cell: str) -> str | None:
    m = _TOPO_ROW_RE.search(cell)
    if not m:
        return None
    return re.sub(r"[\s-]", "", m.group(1).lower())


def check_declared_vs_published(text: str, rep: Report) -> None:
    """Detecta que una celda (topología, columna-p) marcada '---' en una tabla
    tenga un valor numérico en la MISMA posición (misma topología, misma columna)
    en otra tabla del documento. Conservador: solo compara tablas cuyos
    encabezados de columna sean profundidades p=1..4."""
    # Recolectar, por tabla, {(topo, col_idx): "dash"|"num"} para tablas con
    # cabecera de profundidades.
    per_table: list[tuple[int, dict]] = []
    for start_ln, rows in _iter_tabular_blocks(text):
        # ¿el encabezado tiene columnas p=1..4? (buscar 'p = 1'..'p=4' o '$p=..$')
        has_depth_header = False
        for _ln, cells in rows:
            joined = " ".join(cells)
            if re.search(r"p\s*=?\s*1", joined) and re.search(r"p\s*=?\s*[34]", joined):
                has_depth_header = True
                break
        if not has_depth_header:
            continue
        cellmap: dict[tuple[str, int], str] = {}
        for _ln, cells in rows:
            topo = _topo_key(cells[0]) if cells else None
            if topo is None:
                continue
            for idx, c in enumerate(cells[1:], 1):
                if _DASH_CELL_RE.match(c):
                    cellmap[(topo, idx)] = "dash"
                elif _num_es(c) is not None:
                    cellmap[(topo, idx)] = "num"
        if cellmap:
            per_table.append((start_ln, cellmap))
    # Cruce: misma (topo, idx) marcada 'dash' en una tabla y 'num' en otra.
    for i, (ln_a, map_a) in enumerate(per_table):
        for key, va in map_a.items():
            if va != "dash":
                continue
            for j, (ln_b, map_b) in enumerate(per_table):
                if i == j:
                    continue
                if map_b.get(key) == "num":
                    topo, idx = key
                    rep.add(
                        BLOQUEANTE,
                        "declarada-vs-publicada",
                        f"la configuración ({topo}, columna {idx}) está marcada '---' "
                        f"(no evaluada) en la tabla de la línea {ln_a} pero tiene valor "
                        f"numérico en la tabla de la línea {ln_b} (informe §1.4). "
                        "Decidir si se corrió: quitar el guion o quitar el valor.",
                        ln_a,
                    )
                    break


# ── Chequeo: la prosa cita un h que la tabla referenciada no contiene (2.9) ───

_H_VALUE_RE = re.compile(r"h\s*=\s*(\d+(?:[.,]\d+)?)")


def check_h_in_table(text: str, rep: Report) -> None:
    """Cuando la prosa dice 'la Tabla~\\ref{L} ... h = X', verifica que X esté
    entre los valores de h que esa tabla contiene (informe §2.9). Conservador:
    solo actúa si la tabla tiene una columna de h reconocible."""
    for start_ln, rows in _iter_tabular_blocks(text):
        lines = text.split("\n")
        header_zone = "\n".join(lines[max(0, start_ln - 9) : start_ln])
        lm = re.search(r"\\label\{(tab:[^}]+)\}", header_zone)
        if not lm:
            continue
        label = lm.group(1)
        # ¿hay columna de h? localizar su índice por el encabezado.
        h_idx = None
        for _ln, cells in rows:
            for idx, c in enumerate(cells):
                if re.search(r"\bh\b|h_\{|h_\\text", c) and _num_es(re.sub(r".*", "", c)) is None:
                    # heurística: encabezado que menciona 'h'
                    if re.search(r"\$?h\$?|h_\{\\text\{test", c) and not _num_es(c):
                        h_idx = idx
                        break
            if h_idx is not None:
                break
        if h_idx is None:
            continue
        table_hs = set()
        for _ln, cells in rows:
            if h_idx < len(cells):
                v = _num_es(cells[h_idx])
                if v is not None:
                    table_hs.add(round(v, 2))
        if not table_hs:
            continue
        # Buscar en toda la prosa frases que citen esta tabla con un 'h = X'.
        for m in re.finditer(r"(?:Tabla|Cuadro)~?\\ref\{" + re.escape(label) + r"\}", text):
            window = text[m.start() : m.end() + 220]
            for hm in _H_VALUE_RE.finditer(window):
                hv = _num_es(hm.group(1))
                if hv is None:
                    continue
                hv = round(hv, 2)
                if not any(abs(hv - th) < 0.005 for th in table_hs):
                    ln_no = text.count("\n", 0, m.start()) + 1
                    rep.add(
                        IMPORTANTE,
                        "h-fuera-de-tabla",
                        f"la prosa cita h = {hm.group(1)} para la tabla {label}, pero "
                        f"esa tabla solo contiene h ∈ {sorted(table_hs)} (informe §2.9). "
                        "Citar un h que exista o ajustar el rango.",
                        ln_no,
                    )
                    break


# ── Chequeo: cabeceras de tabla pegadas / desbordadas (informe 2.8) ──────────

# Palabras de cabecera que, pegadas sin separador, delatan un desbordamiento:
# 'ArquitecturaSistema', 'MétodoResultado', etc. Se detecta CamelCase con dos
# mayúsculas internas en una celda de cabecera (\textbf).
_GLUED_HEADER_RE = re.compile(r"\b([A-ZÁÉÍÓÚ][a-záéíóú]{3,})([A-ZÁÉÍÓÚ][a-záéíóú]{3,})\b")


def check_table_overflow(text: str, rep: Report) -> None:
    """Detecta cabeceras de tabla con dos palabras pegadas sin espacio (síntoma
    de desbordamiento horizontal, informe §2.8) y preámbulos p{} cuyo ancho total
    supera \\textwidth."""
    lines = text.split("\n")
    for start_ln, rows in _iter_tabular_blocks(text):
        # (a) cabecera: primera fila con \textbf; buscar palabras pegadas
        for ln, cells in rows[:2]:
            for c in cells:
                if "\\textbf" not in c and "textbf" not in c:
                    continue
                inner = re.sub(r"\\textbf\{|\}|\$", "", c)
                gm = _GLUED_HEADER_RE.search(inner)
                if gm:
                    rep.add(
                        IMPORTANTE,
                        "tabla-desbordada",
                        f"cabecera '{gm.group(0)}' con dos palabras pegadas sin espacio "
                        f"(línea {ln}): la tabla se desborda horizontalmente (informe "
                        "§2.8). Reducir anchos de columna (p{...}) o mover citas a "
                        "notas al pie.",
                        ln,
                    )
        # (b) preámbulo p{Xcm}...: suma de anchos > ~15cm (ancho de texto típico)
        preamble_line = lines[start_ln - 1] if start_ln - 1 < len(lines) else ""
        widths = [float(w.replace(",", ".")) for w in re.findall(r"p\{([\d.,]+)cm\}", preamble_line)]
        if widths and sum(widths) > 15.0:
            rep.add(
                IMPORTANTE,
                "tabla-desbordada",
                f"la suma de anchos p{{}} del tabular (línea {start_ln}) es "
                f"{sum(widths):.1f} cm, por encima del ancho de texto (~15 cm): "
                "probable desbordamiento (informe §2.8).",
                start_ln,
            )


# ── Chequeo: h_min(p) con valores divergentes para la misma config (1.7) ─────

# Recolecta menciones 'h_{\min}(p=X) ≈ VALUE' (o 'h_min(p=X) = VALUE') y agrupa
# por p; si para un mismo p hay valores que difieren más de una tolerancia y NO
# se distinguen por topología en la misma cláusula, es la contradicción §1.7.
_HMIN_MENTION_RE = re.compile(
    r"h_?\{?\\?min\}?\s*\(?p\s*(?:=|\{=\}|\{)?\s*(\d)\}?\)?\s*(?:≈|\\approx|=)\s*"
    r"(\d+(?:[.,]\d+)?)"
)


# Valor h_min por topología: 'cadena 1D (...): h_{\min} = 1,09' (sin (p=X) explícito).
_HMIN_TOPO_RE = re.compile(
    r"(cadena\s*1d|heavy[- ]?hex|escalera|cuadrada|triangular)[^.\n]{0,60}?"
    r"h_?\{?\\?min\}?\s*(?:=|≈|\\approx)\s*(\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)


def check_hmin_consistency(text: str, rep: Report) -> None:
    """Marca la contradicción del informe §1.7: la 'frontera cuasi-constante' de
    §5.3.1 (h_min(p=3) ≈ 1,6) atribuida a UNA topología, y un valor por-topología
    distinto (cadena 1D h_min = 1,09) para la misma topología a esa p. Detecta:
      (a) un valor por-topología que difiere en >0,3 del valor h_min(p=3/4)
          declarado, cuando la reconciliación afirma que este último 'agrega las
          cinco topologías' pero el párrafo de origen dice 'para cadena 1D'."""
    body = _strip_comments(text)
    # valores h_min(p=X) globales (leyes/constantes)
    global_pvals: dict[int, list[tuple[float, int]]] = {}
    for m in _HMIN_MENTION_RE.finditer(body):
        p = int(m.group(1))
        val = _num_es(m.group(2))
        if val is not None:
            global_pvals.setdefault(p, []).append((round(val, 2), body.count("\n", 0, m.start()) + 1))
    # valores h_min por topología
    topo_vals: dict[str, list[tuple[float, int]]] = {}
    for m in _HMIN_TOPO_RE.finditer(body):
        topo = re.sub(r"[\s-]", "", m.group(1).lower())
        val = _num_es(m.group(2))
        if val is not None:
            topo_vals.setdefault(topo, []).append((round(val, 2), body.count("\n", 0, m.start()) + 1))
    # El defecto §1.7 es una reconciliación FALSA: el valor global (1,6) atribuido
    # a cadena 1D en un sitio y a "las cinco topologías" en otro. NO es defecto si
    # el texto explica correctamente la dispersión: declara el global como PROMEDIO
    # sobre las cinco topologías y sitúa el valor por-topología por debajo/encima de
    # ese promedio en la misma cláusula. Se suprime ese caso bien explicado.
    well_explained = bool(
        re.search(
            r"promediad|promedio\s+sobre\s+las\s+cinco|dispersi[oó]n\s+entre\s+topolog",
            body,
            re.IGNORECASE,
        )
        and re.search(r"por\s+debajo|por\s+encima|quedan?\s+bastante", body, re.IGNORECASE)
    )
    if well_explained:
        return
    # Señal §1.7: para p=3 (o p=4) hay un valor global (~1,6/~1,4) y un valor
    # por-topología muy distinto (>0,3) para cadena 1D o heavy-hex, y el documento
    # afirma que el valor global 'agrega' topologías o es 'para cadena 1D'.
    reconc = re.search(r"agregan\s+las\s+cinco\s+topolog|para\s+cadena\s*1d", body, re.IGNORECASE)
    if not reconc:
        return
    for p, gvals in global_pvals.items():
        if p not in (3, 4):
            continue
        g = gvals[0][0]
        g_ln = gvals[0][1]
        for topo in ("cadena1d", "heavyhex"):
            for tv, tln in topo_vals.get(topo, []):
                if abs(tv - g) > 0.3:
                    rep.add(
                        IMPORTANTE,
                        "hmin-inconsistente",
                        f"h_min(p={p}) se declara ≈{g:g} (línea {g_ln}) mientras "
                        f"{topo} aparece con h_min = {tv:g} (línea {tln}); si el ≈{g:g} "
                        "'agrega las cinco topologías' pero §5.3.1 lo atribuye a cadena "
                        "1D, la reconciliación es falsa (informe §1.7). Unificar o "
                        "nombrar la magnitud distinta.",
                        tln,
                    )
                    return


# ── Chequeo: denominadores que implican h_min decreciente en N (1.3/1.10) ────

# Malla canónica §4.2: 52 puntos en [0,5;5,0], 39 con h>=1,3. Para una tabla que
# declara "h >= h_min" con denominadores por fila, el h_min implícito por cada
# denominador debe CRECER (o mantenerse) con N segun la ley de frontera; si
# DECRECE, la leyenda contradice el dato (informe §1.3/§1.10).
# Grilla reconstruida: Δh=0,1 en [0,5;0,8), Δh=0,05 en [0,80;1,40], Δh=0,1 en (1,4;5,0].


def _build_h_grid() -> list[float]:
    grid: list[float] = []
    h = 0.5
    while h < 0.80 - 1e-9:
        grid.append(round(h, 2))
        h += 0.1
    h = 0.80
    while h < 1.40 + 1e-9:
        grid.append(round(h, 2))
        h += 0.05
    h = 1.5
    while h < 5.0 + 1e-9:
        grid.append(round(h, 2))
        h += 0.1
    return sorted(set(grid))


_H_GRID = _build_h_grid()


def _implied_hmin(n_points: int) -> float | None:
    """Dado un nº de puntos en el régimen (subconjunto superior de la malla),
    devuelve el h_min que produce exactamente ese conteo (el h del punto en
    posición -n_points)."""
    if not (1 <= n_points <= len(_H_GRID)):
        return None
    return _H_GRID[len(_H_GRID) - n_points]


def check_h_grid_denominator(text: str, rep: Report) -> None:
    """En una tabla cuya leyenda dice 'h >= h_min' y da denominadores por fila
    crecientes en N, verifica que el h_min implícito NO decrezca con N. Muy
    conservador: solo actúa si la leyenda menciona explícitamente 'h_{\\min}' y
    'malla'/'régimen válido', y si hay >=3 denominadores asociados a N crecientes."""
    # Localizar tablas con leyenda que atribuye los puntos a h>=h_min.
    for start_ln, rows in _iter_tabular_blocks(text):
        lines = text.split("\n")
        zone = "\n".join(lines[max(0, start_ln - 12) : start_ln])
        legend_below = "\n".join(lines[start_ln : start_ln + 30])
        if not (re.search(r"h_?\{?\\?min", zone) and re.search(r"malla|r[eé]gimen", zone)):
            continue
        # Excepción: si la leyenda reconoce que el conteo variable también se debe a
        # la cobertura de datos (no solo al recorte por h_min), la no-monotonía es
        # esperada y no es contradicción (alineado con _check_legend_vs_manifest #4
        # de generate_thesis_tables). Aplica a tablas auto y manuales por igual.
        if re.search(r"cobertura de datos|puntos.*disponibles|datos disponibles", zone + legend_below):
            continue
        # extraer (N, n_points) de las filas: primera col numérica = N, y una
        # celda con un entero pequeño (20..52) plausible como nº de puntos, o un
        # caso absoluto (n/m) cuyo m sea el nº de puntos.
        seq: list[tuple[int, int]] = []
        for _ln, cells in rows:
            if not cells:
                continue
            n_val = _num_es(cells[0])
            if n_val is None or n_val < 4 or n_val > 300 or abs(n_val - round(n_val)) > 1e-6:
                continue
            N = int(round(n_val))
            pts = None
            for c in cells[1:]:
                mm = re.search(r"(\d+)\s*/\s*(\d+)", c)
                if mm:
                    pts = int(mm.group(2))
                    break
                iv = _num_es(c)
                if iv is not None and 20 <= iv <= 52 and abs(iv - round(iv)) < 1e-6:
                    pts = int(round(iv))
                    break
            if pts is not None:
                seq.append((N, pts))
        if len(seq) < 3:
            continue
        seq.sort()
        hmins = [(N, pts, _implied_hmin(pts)) for N, pts in seq]
        hmins = [(N, pts, hm) for N, pts, hm in hmins if hm is not None]
        if len(hmins) < 3:
            continue
        # La ley de frontera hace CRECER h_min con N, lo que implica MENOS puntos
        # (recorte mayor) a mayor N. La contradicción del informe §1.3 es que los
        # puntos NO decrecen monótonamente con N: si un N mayor tiene MÁS puntos
        # que uno menor, el h_min implícito decreció, contra la ley. Señal robusta:
        # existe un par (N_i < N_j) con pts_i < pts_j (más puntos al crecer N).
        violations = []
        for a in range(len(hmins)):
            for b in range(a + 1, len(hmins)):
                Na, pa, ha = hmins[a]
                Nb, pb, hb = hmins[b]
                if Nb > Na and pb > pa:  # N mayor con MÁS puntos → h_min bajó
                    violations.append((Na, ha, Nb, hb))
        if violations:
            Na, ha, Nb, hb = violations[0]
            desc = ", ".join(f"N={N}→h_min≈{hm:g}" for N, _p, hm in hmins)
            rep.add(
                BLOQUEANTE,
                "hmin-denominador",
                f"la tabla de la línea {start_ln} declara 'h ≥ h_min' pero N={Nb} tiene "
                f"más puntos que N={Na} (h_min implícito {hb:g} < {ha:g}): el h_min "
                f"decrece con N, contra la ley de frontera ({desc}) (informe §1.3/§1.10). "
                "Probablemente son puntos de test, no el recorte por h_min: corregir la "
                "leyenda (retirar la atribución a h_min).",
                start_ln,
            )


# ── Chequeo: afirmación de cumplimiento vs celdas de la tabla citada (2.3) ───


def check_claim_vs_cells(text: str, rep: Report) -> None:
    """(a) 'íntegramente' / 'se cumplen todos' co-ocurriendo con 'línea abierta'
    o 'queda abierta' en el mismo párrafo (contradicción de cierre, §2.3).
    (b) 'fidelidad media > 0,95' cuando el documento contiene celdas de fidelidad
    por debajo de 0,95 en tablas de fidelidad (señal, conservadora)."""
    body = _strip_comments(text)
    # (a) "objetivos se cumplen íntegramente" cuando el documento declara en
    # cualquier parte una 'línea abierta' de un objetivo/hipótesis. El "se valida
    # íntegramente mediante simulación ideal" (metodología) NO es una afirmación de
    # cumplimiento de objetivos: se exige la co-ocurrencia con 'objetivos'/'OE'.
    doc_has_open = bool(re.search(r"l[ií]nea\s+(?:de\s+trabajo\s+)?abierta|queda[n]?\s+abiert", body, re.IGNORECASE))
    for m in re.finditer(
        r"objetivos?\s+(?:OE)?[^.]{0,40}?se\s+cumplen\s+[ií]ntegramente|"
        r"se\s+cumplen\s+[ií]ntegramente[^.]{0,40}?objetiv",
        body,
        re.IGNORECASE,
    ):
        if doc_has_open:
            ln = body.count("\n", 0, m.start()) + 1
            rep.add(
                IMPORTANTE,
                "cumplimiento-vs-abierto",
                f"'{m.group(0).strip()[:50]}...' (línea {ln}): se declaran los objetivos "
                "cumplidos íntegramente, pero el documento deja una línea abierta "
                "(informe §2.3). 'Íntegramente' no procede si un objetivo queda abierto; "
                "matizar el alcance.",
                ln,
            )
    # (b) 'fidelidad media > 0,95' / '>= 0,95' cuando hay celdas de fidelidad < 0,95
    fid_claim = re.search(r"fidelidad\s+media\s*(?:>|≥|\\geq|mayor)\s*(?:que\s*)?0[,.]95", body, re.IGNORECASE)
    if fid_claim:
        # recolectar celdas de fidelidad (0,7xx-0,9xx) en tablas con columna F
        low_fids = []
        for start_ln, rows in _iter_tabular_blocks(text):
            zone_lines = text.split("\n")
            zone = "\n".join(zone_lines[max(0, start_ln - 10) : start_ln])
            if not re.search(
                r"fidelidad|\\bar\{F\}|F_\{VQE|F\\text|\bF\b", zone + " ".join(c for _, cs in rows for c in cs)
            ):
                continue
            for ln, cells in rows:
                for c in cells:
                    v = _num_es(c)
                    if v is not None and 0.70 <= v < 0.95:
                        low_fids.append((round(v, 3), ln))
        if low_fids:
            worst = min(low_fids)
            ln_claim = body.count("\n", 0, fid_claim.start()) + 1
            rep.add(
                IMPORTANTE,
                "fidelidad-vs-celdas",
                f"se afirma 'fidelidad media > 0,95' (línea {ln_claim}) pero hay celdas "
                f"de fidelidad por debajo de 0,95 (p. ej. {worst[0]:g} en línea "
                f"{worst[1]}) (informe §2.3). Declarar el criterio por topología.",
                ln_claim,
            )


# ── Chequeo: cobertura de campaña (configs × mín. semillas vs total) (§1.6) ──


def check_coverage_arithmetic(text: str, rep: Report) -> None:
    """Un total de ejecuciones que declara 'T topologías ... P profundidades ...
    (K ejecuciones)' debe cumplir K >= (T·P)·min_semillas cuando el documento fija
    un mínimo de semillas por configuración. Atrapa el '21 ejecuciones para 15
    configuraciones con >=3 semillas' del informe §1.6 (mínimo 45). Determinista.
    """
    body = _strip_comments(text)
    # Mínimo de semillas declarado en el documento ("mínimo 3 semillas",
    # ">= 3 semillas", "3 semillas aleatorias").
    seed_min = 3
    ms = re.search(r"(?:m[ií]nimo|≥|>=|al menos)\s*(\d+)\s+semillas", body, re.IGNORECASE)
    if ms:
        seed_min = int(ms.group(1))
    # Frases del tipo "... 5 topologías ... p = 2-4 ... (21 ejecuciones)".
    for m in re.finditer(
        r"(\d+)\s+topolog[ií]as[^.]{0,90}?p\s*=?\s*\$?\s*(\d+)\s*\$?\s*"
        r"(?:--|[–-])\s*\$?(\d+)\$?[^.]{0,50}?\((\d+)\s+ejecuciones\)",
        body,
        re.IGNORECASE,
    ):
        n_topo = int(m.group(1))
        p_lo, p_hi = int(m.group(2)), int(m.group(3))
        total = int(m.group(4))
        n_cfg = n_topo * (p_hi - p_lo + 1)
        minimum = n_cfg * seed_min
        if total < minimum:
            ln = body.count("\n", 0, m.start()) + 1
            rep.add(
                BLOQUEANTE,
                "cobertura-campaña",
                f"se declaran {total} ejecuciones para {n_topo} topologías × "
                f"{p_hi - p_lo + 1} profundidades = {n_cfg} configuraciones, pero con "
                f"el mínimo de {seed_min} semillas serían >= {minimum} (informe §1.6). "
                "Revisar la cifra de cobertura o el alcance declarado.",
                ln,
            )


# ── Chequeo: coherencia de las tres escalas de N (§2.1) ──────────────────────


def check_scale_consistency(text: str, rep: Report) -> None:
    """Si el documento define las tres escalas canónicas (22/40/250) pero en la
    formulación del ALCANCE/OBJETIVO usa 'N = 4-20' como si fuera el límite del
    trabajo, el tribunal ve dos respuestas distintas (informe §2.1). Determinista y
    conservador: solo marca 'N = 4-20/4 a 20' en cláusulas de validación/alcance
    y solo si el documento define en otra parte la escala de 250."""
    body = _strip_comments(text)
    defines_three_scales = bool(re.search(r"\b250\b", body) and re.search(r"\b(22|40)\b.*\b(40|250)\b", body))
    if not defines_three_scales:
        return
    scope_ctx = re.compile(
        r"(validaci[oó]n|valida|abarca|escalas?|tama[nñ]os?|objetivo)[^.]{0,70}?"
        r"N\s*=?\s*\$?\s*4\s*\$?\s*(?:--|[–-]|\ba\b)\s*\$?\s*20",
        re.IGNORECASE,
    )
    hits = [(body.count("\n", 0, m.start()) + 1) for m in scope_ctx.finditer(body)]
    if hits:
        rep.add(
            IMPORTANTE,
            "escala-N-incoherente",
            f"el alcance se formula como 'N = 4-20' en {len(hits)} sitio(s) "
            f"(líneas {hits[:6]}) mientras el documento define tres escalas (22/40/250). "
            "Un tribunal ve dos respuestas al 'N máximo'. Unificar con la distinción "
            "de las tres escalas (informe §2.1).",
            hits[0],
        )


# ── Chequeo: leyenda 'media de cocientes por punto' exige columna de puntos ──


def check_ratio_needs_points_column(text: str, rep: Report) -> None:
    """Si la leyenda de una tabla declara que ΔE/gap es la 'media de los cocientes
    por punto', esa tabla DEBE tener una columna de nº de puntos para que el
    convenio sea auditable (informe §2.4b). Determinista: leyenda con la frase y
    tabla sin encabezado de puntos."""
    lines = text.split("\n")
    for start_ln, rows in _iter_tabular_blocks(text):
        legend = "\n".join(lines[start_ln : start_ln + 30])
        if "media de los cocientes" not in legend and "cocientes por punto" not in legend:
            continue
        header = rows[0][1] if rows else []
        header_txt = " ".join(header).lower()
        has_points_col = (
            any(re.search(r"\bpts?\b|puntos|n_?\{?\\?text\{?pts|n\s*pts", h.lower()) for h in header)
            or "pts" in header_txt
            or "puntos" in header_txt
        )
        if not has_points_col:
            rep.add(
                IMPORTANTE,
                "convenio-sin-puntos",
                f"la tabla de la línea {start_ln} declara ΔE/gap como 'media de "
                "cocientes por punto' pero no tiene columna de nº de puntos: el "
                "convenio no es auditable desde la tabla (informe §2.4b). Añadir "
                "columna de puntos.",
                start_ln,
            )


# ── Centinela: 'agregación global diseñada para escalar' contradice §5.4 (§2.6) ─


def check_global_aggregation_claim(text: str, rep: Report) -> None:
    """Centinela acotado del informe §2.6: el trabajo futuro no debe presentar la
    'agregación global' como apta para escalar entre tamaños, porque §5.4 la mide
    fallando (ΔE/gap = 688 %) y la UnifiedMPNN por enlace se introduce justamente
    para superarla. NO es un detector genérico de contradicción (marcaría los usos
    legítimos del término en §2.6/§4.4/§5.4): solo dispara ante la combinación
    'agregación global' + 'diseñad/prepar/apta/permite escalar/escalación/entre
    tamaños' en la MISMA frase, que solo aparece en el error. Determinista."""
    body = _strip_comments(text)
    # 'agregación global' seguida (<=120 chars) de una afirmación de aptitud para escalar.
    pat = re.compile(
        r"agregaci[oó]n\s+global[^.]{0,120}?"
        r"(dise[nñ]ad[ao]s?|preparad[ao]s?|apta|adecuad[ao]s?|permite|habilita|escala)"
        r"[^.]{0,60}?(escal|entre\s+tama[nñ]os|tama[nñ]os\s+no\s+vistos|generaliz)",
        re.IGNORECASE,
    )
    for m in pat.finditer(body):
        ln = body.count("\n", 0, m.start()) + 1
        rep.add(
            IMPORTANTE,
            "agregación-global-vs-5.4",
            f"'{m.group(0).strip()[:70]}...' (línea {ln}): se presenta la agregación "
            "global como apta para escalar entre tamaños, pero §5.4 la mide fallando "
            "(ΔE/gap = 688 %) y la UnifiedMPNN por enlace se introduce para superarla "
            "(informe §2.6). La escalación se apoya en la agregación local por enlace, "
            "no en la global.",
            ln,
        )


# ── Chequeo: referencias colgantes \ref a un label inexistente ───────────────


def _labels_including_inputs(tex_path: Path, text: str) -> set[str]:
    """Labels definidos en el .tex y en cada \\input{} resuelto (las tablas auto
    llevan su \\label dentro del fichero importado, no en el cuerpo)."""
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    base = tex_path.parent
    for inp in re.findall(r"\\input\{([^}]+)\}", text):
        for cand in (base / inp, base / f"{inp}.tex"):
            if cand.exists():
                sub = _strip_comments(cand.read_text(encoding="utf-8"))
                labels |= set(re.findall(r"\\label\{([^}]+)\}", sub))
                break
    return labels


def check_kbar_single_value(text: str, rep: Report) -> None:
    """Centinela de regresión del informe §1.8: A = r·k̄ se calcula con el k̄ de
    CADA configuración (≈4--7 a p=1, 37--85 en heavy-hex p=3), no con un valor
    único. Dispara si el documento presenta 37--85 como 'el valor empleado para
    calcular A' SIN mencionar también el régimen de 4--7 en el mismo contexto
    (que es lo que hace que el ~5× de la Tabla 6.1 sea auditable). Hoy da 0 (el
    texto ya nombra ambos); es red de regresión. Determinista."""
    body = _strip_comments(text)
    claim = re.compile(
        r"37\$?\s*(?:--|[–-])\s*\$?85[^.]{0,90}?"
        r"(?:empleado|se\s+emplea)[^.]{0,30}?calcular[^.]{0,15}?\$?A\$?",
        re.IGNORECASE,
    )
    for m in claim.finditer(body):
        window = body[max(0, m.start() - 500) : m.end() + 500]
        # Correcto si el contexto nombra el régimen bajo (4--7) O aclara que el k̄
        # es por-configuración / uno de varios valores. Cualquiera de las dos
        # señales basta: ambas son la corrección A.6 del informe.
        mentions_low_regime = bool(re.search(r"4\s*(?:--|[–-])\s*7", window)) or bool(
            re.search(
                r"cada\s+configuraci|en\s+ambos\s+casos|correspondiente\s+a\s+cada|"
                r"uno\s+de\s+los\s+valores",
                window,
                re.IGNORECASE,
            )
        )
        if not mentions_low_regime:
            ln = body.count("\n", 0, m.start()) + 1
            rep.add(
                IMPORTANTE,
                "kbar-valor-único",
                f"'{m.group(0).strip()[:60]}...' (línea {ln}): se presenta k̄ = 37--85 "
                "como el valor único para calcular A, pero A se calcula con el k̄ de "
                "cada configuración (≈4--7 a p=1) y así lo exige la Tabla 6.1 (~5×). "
                "Nombrar ambos regímenes o decir 'uno de los valores' (informe §1.8).",
                ln,
            )


def check_hmin_reconciliation(text: str, rep: Report) -> None:
    """Centinela de regresión del informe §1.7: h_min(p=3)≈1,6 (§5.3.1) y
    h_min=1,09 (cadena 1D, §interpretación) conviven SOLO si §5.3.1 declara el 1,6
    como promedio sobre las cinco topologías. Dispara si el documento introduce
    las leyes de frontera 'para cadena 1D' Y luego un párrafo dice que ≈1,6/≈1,4
    'agregan las cinco topologías', SIN que §5.3.1 declare el ≈1,6 como promedio
    (reconciliación falsa). Hoy da 0 (§5.3.1 ya dice 'Promediada sobre las cinco').
    Complementa a check_hmin_consistency; determinista."""
    body = _strip_comments(text)
    reconc = re.search(r"agregan\s+las\s+cinco\s+topolog", body, re.IGNORECASE)
    if not reconc:
        return
    declares_average = bool(
        re.search(
            r"promediad[ao]\s+sobre\s+las\s+cinco|promedio\s+sobre\s+las\s+cinco\s+topolog",
            body,
            re.IGNORECASE,
        )
    )
    intro_chain = bool(re.search(r"ajustes\s+lineales\s+para\s+cadena\s*1d", body, re.IGNORECASE))
    if intro_chain and not declares_average:
        ln = body.count("\n", 0, reconc.start()) + 1
        rep.add(
            IMPORTANTE,
            "hmin-reconciliación-falsa",
            f"(línea {ln}): el párrafo dice que h_min(p=3)≈1,6 y ≈1,4 'agregan las "
            "cinco topologías', pero §5.3.1 introduce las leyes 'para cadena 1D' sin "
            "declarar el ≈1,6 como promedio: la reconciliación es falsa (informe §1.7). "
            "Declarar en §5.3.1 el ≈1,6 como promedio sobre las cinco topologías, o "
            "nombrar la magnitud distinta del 1,09.",
            ln,
        )


def check_frontier_independent_of_n(text: str, rep: Report) -> None:
    """Centinela de regresión: la frontera h_min NO debe declararse 'independiente
    de N' ni 'no se desplaza al escalar N' para p>=3. Los ajustes empíricos tienen
    pendiente positiva (h_min = a + b·N, b>0) en todo el rango de p medido; a p>=3
    la pendiente se atenúa pero no se anula. Afirmar pendiente cero es más fuerte
    de lo que sostienen los datos y contradice §interpretación (informe: objeción
    sobre la inmovilidad de la frontera).

    Determinista y seguro: solo dispara cuando 'independiente de N' / 'no se
    desplaza' / 'no depende del tamaño' aparece en la MISMA frase que
    'frontera'/'h_min' Y una marca de p>=3 (p>=3, p=3, p=4, cuasi-constante). NO
    marca los usos legítimos de 'independiente de N' (dimensión de parámetros
    d_theta=2p; ley de área de MPS S<=cte), que no hablan de la frontera.
    """
    body = _strip_comments(text)
    # Frases (acotadas a ~200 chars) que ligan frontera/h_min con inmovilidad en N.
    frontier_ctx = re.compile(
        r"(frontera|h_?\{?\\?min\}?)[^.]{0,200}?"
        r"(independiente[s]?\s+de\s+\$?N\$?|no\s+se\s+desplaza|no\s+depende\s+del\s+tama[nñ]o)",
        re.IGNORECASE,
    )
    for m in frontier_ctx.finditer(body):
        span = m.group(0)
        # Debe referirse a p>=3 (donde estaba la afirmación de pendiente cero).
        if not re.search(r"p\s*\\?geq\s*3|p\s*≥\s*3|p\s*=\s*[34]|cuasi-?constante", span, re.IGNORECASE):
            continue
        ln = body.count("\n", 0, m.start()) + 1
        rep.add(
            IMPORTANTE,
            "frontera-independiente-N",
            f"(línea {ln}): se declara la frontera h_min 'independiente de N' / que "
            "'no se desplaza' a p>=3. Los ajustes tienen pendiente positiva (b>0) en "
            "todo p medido; a p>=3 se atenúa pero no se anula. Enunciar 'pendiente "
            "pequeña, sin anularse', no independencia de N.",
            ln,
        )
        return


def check_hmin_topo_depth_label(text: str, rep: Report) -> None:
    """Centinela: los valores h_min por topología (cadena 1D, heavy-hex, ...) deben
    etiquetarse con la profundidad REAL a la que se miden. El defecto detectado: se
    atribuían a 'su mejor profundidad (p=3--4)' cuando los valores citados eran los
    óptimos sobre p=2--8 (cadena 1,09 es p=7; triangular 2,20 es p=8), fuera del
    alcance de la campaña sistemática. Los valores canónicos a p=4 son
    cadena 1,18 / heavy-hex 1,31 / escalera 1,84 / cuadrada 1,88 / triangular 2,72
    (fuente results/H_FRONTIER_TOPOLOGIES.md).

    Determinista y seguro: dispara solo si el mismo bloque de texto (i) enumera
    h_min por topología para >=4 topologías, (ii) los etiqueta como 'p=3--4' o
    'mejor/óptima profundidad (p=3--4)', y (iii) incluye alguno de los valores
    óptimos de p alto (1,09 cadena / 1,12 heavy-hex / 2,20 triangular), que a p=4
    NO se alcanzan. Esa combinación es exactamente el error; los valores a p=4 no
    la disparan.
    """
    body = _strip_comments(text)
    # Localizar un fragmento que enumere h_min por topología (>=4 menciones topo).
    topo_hits = list(_HMIN_TOPO_RE.finditer(body))
    if len(topo_hits) < 4:
        return
    lo = min(h.start() for h in topo_hits)
    hi = max(h.end() for h in topo_hits)
    block = body[max(0, lo - 200) : hi + 200]
    # Tolerar '$' de math intercalado en el rango LaTeX 'p = 3$--$4' y el en-dash
    # como '--', '–' o '\$--\$' (mismo criterio que el resto de chequeos del script).
    _p34 = r"p\s*=\s*3\s*\$?\s*(?:--|[–-])\s*\$?\s*4"
    labels_p34 = re.search(
        r"(mejor|[oó]ptim[ao])\s+profundidad\s*\(?\s*" + _p34 + r"|" + _p34,
        block,
        re.IGNORECASE,
    )
    if not labels_p34:
        return
    # Valores óptimos de p alto que NO se alcanzan a p=4.
    high_p_optima = re.search(r"1,09|1,12|2,20", block)
    if high_p_optima:
        ln = body.count("\n", 0, lo) + 1
        rep.add(
            IMPORTANTE,
            "hmin-topo-profundidad",
            f"(línea {ln}): valores h_min por topología etiquetados como 'p=3--4' "
            "pero incluyen óptimos de p alto (1,09 cadena es p=7; 2,20 triangular es "
            "p=8). A p=4 los valores canónicos son cadena 1,18 / heavy-hex 1,31 / "
            "escalera 1,84 / cuadrada 1,88 / triangular 2,72 "
            "(results/H_FRONTIER_TOPOLOGIES.md). Etiquetar con la p real.",
            ln,
        )


def _load_frontier_matrix(tex_path: Path) -> dict[str, dict[int, float]]:
    """Parsea results/H_FRONTIER_TOPOLOGIES.md → {topo: {p: h_frontier}}.

    Fuente de verdad de la frontera h_min por (topología, p) a N=10 (TFIM,
    StatevectorEstimator). Devuelve {} si no se encuentra o no se puede parsear:
    en ese caso el chequeo que la usa se degrada a un aviso INFO 'omitido', nunca
    a un falso BLOQUEANTE.
    """
    # results/ está bajo la raíz del repo; el .tex vive en internal/tesis/.
    root = tex_path.resolve().parent.parent.parent
    src = root / "results" / "H_FRONTIER_TOPOLOGIES.md"
    if not src.exists():
        return {}
    try:
        t = src.read_text(encoding="utf-8")
    except OSError:
        return {}
    hdr = re.search(r"Topology\s+((?:p=\d+\s+)+)edges", t)
    if not hdr:
        return {}
    p_cols = [int(x.split("=")[1]) for x in hdr.group(1).split()]
    out: dict[str, dict[int, float]] = {}
    for topo in ("chain_1d", "heavy_hex", "ladder", "square", "triangular", "kagome"):
        m = re.search(rf"^{topo}\s+([\d.\sFAIL]+?)\s+\d+\s*$", t, re.MULTILINE)
        if not m:
            continue
        vals = m.group(1).split()
        row: dict[int, float] = {}
        for p, v in zip(p_cols, vals, strict=False):
            if v != "FAIL":
                try:
                    row[p] = float(v)
                except ValueError:
                    pass
        if row:
            out[topo] = row
    return out


# Nombre en el .tex (español) → clave en la matriz de frontera.
_TOPO_ES_TO_KEY = {
    "cadena1d": "chain_1d",
    "heavyhex": "heavy_hex",
    "escalera": "ladder",
    "cuadrada": "square",
    "triangular": "triangular",
}


def check_hmin_topo_value_source(text: str, tex_path: Path, rep: Report) -> None:
    """Ancla los valores h_min por topología del .tex a la fuente de verdad
    results/H_FRONTIER_TOPOLOGIES.md. Es el guardián de la clase de error raíz de
    varias sesiones: un número citado atribuido a una (topología, p) que no le
    corresponde (p.ej. 1,09 de cadena 1D presentado 'a p=4' cuando 1,09 es de p=7;
    a p=4 la fuente da 1,18).

    Para cada bloque que enumera h_min por topología Y declara una profundidad
    'p = X' (o 'p = X--Y'), compara cada valor citado con la matriz fuente a esa p.
    Con rango p=X--Y, acepta el valor si coincide con CUALQUIER p del rango. Si no
    coincide con ninguna p declarada pero sí con otra p de la fuente, nombra la p
    real (diagnóstico accionable). Tolerancia 0,03 (dos decimales). Seguro: sin
    fuente → INFO 'omitido'; sin p declarada en el bloque → no actúa.
    """
    matrix = _load_frontier_matrix(tex_path)
    if not matrix:
        rep.add(
            INFO,
            "hmin-topo-fuente",
            "no se pudo cargar results/H_FRONTIER_TOPOLOGIES.md; no se validaron "
            "los h_min por topología contra la fuente.",
        )
        return
    body = _strip_comments(text)
    topo_hits = list(_HMIN_TOPO_RE.finditer(body))
    if len(topo_hits) < 4:
        return
    lo = min(h.start() for h in topo_hits)
    hi = max(h.end() for h in topo_hits)
    block_start = max(0, lo - 250)
    block = body[block_start : hi + 100]
    # Profundidad(es) declarada(s) en el bloque: 'p = 4' o 'p = 3--4'.
    p_decl: set[int] = set()
    for pm in re.finditer(r"p\s*=\s*(\d)\s*(?:\$?\s*(?:--|[–-])\s*\$?\s*(\d))?", block):
        p_decl.add(int(pm.group(1)))
        if pm.group(2):
            p_decl.add(int(pm.group(2)))
    if not p_decl:
        return  # sin p declarada no se puede anclar; otro chequeo cubre la etiqueta
    for m in topo_hits:
        if not (block_start <= m.start() <= hi + 100):
            continue
        topo_es = re.sub(r"[\s-]", "", m.group(1).lower())
        key = _TOPO_ES_TO_KEY.get(topo_es)
        val = _num_es(m.group(2))
        if key is None or val is None or key not in matrix:
            continue
        row = matrix[key]
        # ¿coincide con alguna p declarada?
        if any(abs(val - row[p]) <= 0.03 for p in p_decl if p in row):
            continue
        # No coincide: buscar a qué p de la fuente sí corresponde (diagnóstico).
        real_p = next((p for p, v in sorted(row.items()) if abs(val - v) <= 0.03), None)
        expected = (
            ", ".join(f"p={p}:{row[p]:.2f}" for p in sorted(p_decl) if p in row) or "(p declarada no está en la fuente)"
        )
        ln = body.count("\n", 0, m.start()) + 1
        rep.add(
            BLOQUEANTE,
            "hmin-topo-fuente",
            f"(línea {ln}): {m.group(1)} h_min = {val:g} atribuido a p={sorted(p_decl)} "
            f"pero la fuente (H_FRONTIER_TOPOLOGIES.md) da {expected}"
            + (f"; {val:g} corresponde a p={real_p}." if real_p else "; valor no hallado en la fuente.")
            + " Etiquetar con la p real o corregir el valor.",
            ln,
        )


def check_frontier_slope_positive(text: str, rep: Report) -> None:
    """Refuerzo de coherencia: los ajustes de frontera h_min(p) = a + b·N deben
    tener pendiente b > 0 (la frontera CRECE con N; es la base de H2 y de toda la
    narrativa de escalado). Una recta con b <= 0 contradiría el resto del texto.
    Determinista: solo actúa sobre 'h_min(p=..) = a + b·N' con b numérico."""
    body = _strip_comments(text)
    for m in _HMIN_LAW_RE.finditer(body):
        b = _num_es(m.group(2))
        if b is None:
            continue
        if b <= 0:
            ln = body.count("\n", 0, m.start()) + 1
            rep.add(
                BLOQUEANTE,
                "frontera-pendiente",
                f"(línea {ln}): ajuste de frontera con pendiente b = {b:g} <= 0; "
                "la frontera h_min crece con N (b > 0) en todo el rango medido. "
                "Revisar el signo/valor del coeficiente.",
                ln,
            )


def check_dangling_refs(text: str, tex_path: Path, rep: Report) -> None:
    """\\ref/\\autoref/\\eqref/\\Cref a un label que no existe: LaTeX lo compila
    como '??' y el tribunal lo ve. Resuelve \\input para no marcar los labels de
    las tablas auto, y opera sin comentarios (una ref comentada no cuenta)."""
    labels = _labels_including_inputs(tex_path, text)
    for m in re.finditer(r"\\(?:ref|autoref|eqref|[Cc]ref)\{([^}]+)\}", text):
        key = m.group(1)
        if key not in labels:
            ln = text.count("\n", 0, m.start()) + 1
            rep.add(
                BLOQUEANTE,
                "ref-colgante",
                f"\\ref{{{key}}} no tiene \\label correspondiente: LaTeX lo compila "
                "como '??'. Definir el label o corregir la referencia.",
                ln,
            )


# ── Chequeo: integridad de citas \citep/\citet vs \bibitem ───────────────────


def check_citation_integrity(text: str, rep: Report) -> None:
    """Cita a una clave sin \\bibitem (sale como '[?]', BLOQUEANTE) y \\bibitem
    nunca citado (INFO). Multi-clave: \\citep{a,b} se separa por comas."""
    cited: set[str] = set()
    cite_line: dict[str, int] = {}
    for m in re.finditer(r"\\cite[a-zA-Z]*\{([^}]+)\}", text):
        ln = text.count("\n", 0, m.start()) + 1
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                cited.add(k)
                cite_line.setdefault(k, ln)
    bibitems = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", text))
    if not bibitems:
        return  # sin bibliografía embebida: nada que contrastar
    for key in sorted(cited - bibitems):
        rep.add(
            BLOQUEANTE,
            "cita-sin-bibitem",
            f"\\cite{{{key}}} no tiene \\bibitem: la cita sale como '[?]' en el PDF.",
            cite_line.get(key),
        )
    for key in sorted(bibitems - cited):
        rep.add(INFO, "bibitem-no-citado", f"\\bibitem{{{key}}} nunca se cita en el texto.")


# ── Chequeo: labels duplicados (LaTeX referencia uno solo, en silencio) ──────


def check_duplicate_labels(text: str, rep: Report) -> None:
    """El mismo \\label definido dos veces: LaTeX resuelve las \\ref a uno solo
    sin avisar, y una de las dos tablas/figuras queda mal referenciada."""
    seen: dict[str, int] = {}
    dupes: dict[str, list[int]] = {}
    for m in re.finditer(r"\\label\{([^}]+)\}", text):
        key = m.group(1)
        ln = text.count("\n", 0, m.start()) + 1
        if key in seen:
            dupes.setdefault(key, [seen[key]]).append(ln)
        else:
            seen[key] = ln
    for key, lns in dupes.items():
        rep.add(
            BLOQUEANTE,
            "label-duplicado",
            f"\\label{{{key}}} definido {len(lns)} veces (líneas {lns}): las \\ref "
            "resuelven a uno solo en silencio. Renombrar los duplicados.",
            lns[1],
        )


# ── Chequeo: celda de tasa/fidelidad con porcentaje > 100 % ──────────────────

# Acotado a columnas de tasa/aprobación/fidelidad (0–100 % por definición). NO
# toca ΔE/gap ni razón de error, que legítimamente superan 100 % (p. ej. 688 %).
_RATE_HEADER_RE = re.compile(r"tasa|aprob|fidelidad|precisi[oó]n|acierto|cobertura", re.IGNORECASE)


def check_percentage_ceiling(text: str, rep: Report) -> None:
    """En una tabla con columna de tasa/aprobación/fidelidad, una celda > 100 %
    es imposible por definición. Localiza la columna por su encabezado para no
    marcar razones de error (ΔE/gap) que sí exceden 100 %."""
    for _start_ln, rows in _iter_tabular_blocks(text):
        if not rows:
            continue
        rate_cols: set[int] = set()
        for _ln, cells in rows:
            for idx, c in enumerate(cells):
                if _RATE_HEADER_RE.search(c) and _num_es(c) is None:
                    rate_cols.add(idx)
            if rate_cols:
                break
        if not rate_cols:
            continue
        for ln, cells in rows:
            for idx in rate_cols:
                if idx >= len(cells):
                    continue
                cell = cells[idx]
                if "%" not in cell and "\\%" not in cell:
                    continue
                v = _num_es(cell)
                if v is not None and v > 100.0:
                    rep.add(
                        IMPORTANTE,
                        "porcentaje-imposible",
                        f"celda de tasa/fidelidad = {v:g}% (línea {ln}, columna "
                        f"{idx + 1}) supera 100 %, imposible para esa magnitud. "
                        "Revisar el dato o la etiqueta de la columna.",
                        ln,
                    )


# "alcanza/es del NN% de tasa de aprobación": un hallazgo TITULADO con una tasa
# puntual. Solo valores de 1--2 dígitos (excluye 100 y umbrales/rangos, que usan
# >=, --, +). El verbo de logro ('alcanza/es del') marca que es un resultado
# afirmado, no una definición de criterio.
_PASSRATE_CLAIM_RE = re.compile(
    r"(?:alcanza|es del|del)\s+(\d{1,2})\\%\s+(?:de\s+)?tasa de aprobaci[oó]n",
    re.IGNORECASE,
)
# Ancla exigida por §5: la frase (o la contigua) debe acompañar la tasa con la
# métrica primaria |ΔE| o la fidelidad, o el caso absoluto (n/m).
_PASSRATE_ANCHOR_RE = re.compile(
    r"\\Delta E|\|\\Delta E\||fidelidad|\\bar\{F\}|\bF\s*(?:=|\\geq|\\approx)|\(\s*\d+\s*/\s*\d+\s*\)",
    re.IGNORECASE,
)


def check_passrate_needs_abs_error(text: str, rep: Report) -> None:
    """(§5, jerarquía de métricas) La tasa de aprobación es una métrica derivada:
    nunca se reporta como hallazgo sin el |ΔE| (o la fidelidad) que la acompaña. Este
    centinela marca las frases que TITULAN un resultado con una tasa puntual
    ('alcanza 74\\% de tasa de aprobación') cuando ni esa frase ni la contigua citan
    |ΔE|, fidelidad ni el caso absoluto (n/m). Acotado a valores de 1--2 dígitos con
    verbo de logro (excluye umbrales '≥90\\%', rangos '90--95\\%', '0\\%' y '100\\%').
    Hoy da 0; es red de regresión determinista."""
    body = _strip_comments(text)
    for m in _PASSRATE_CLAIM_RE.finditer(body):
        lo, hi = max(0, m.start() - 220), min(len(body), m.end() + 120)
        if not _PASSRATE_ANCHOR_RE.search(body[lo:hi]):
            ln = body.count("\n", 0, m.start()) + 1
            ctx = re.sub(r"\s+", " ", body[max(0, m.start() - 20) : m.end() + 20]).strip()
            rep.add(
                IMPORTANTE,
                "passrate-sin-abs",
                f"(línea {ln}): tasa de aprobación '{m.group(1)}%' presentada como "
                f"hallazgo sin el |ΔE| o la fidelidad que la acompaña: '...{ctx}...'. "
                "La tasa es métrica derivada; leerla junto al error absoluto (steering §5).",
                ln,
            )


# Caso absoluto INICIADO pero incompleto: 'NN% (n/' sin denominador, 'NN% (n)' sin
# fracción. Un residuo de edición inequívoco (no un umbral ni una aproximación).
_ABS_CASE_INCOMPLETE_RE = re.compile(
    r"\d{1,3}\s*\\?%\s*\(\s*\d+\s*(?:/\s*)?\)",
)


def check_abs_case_incomplete(text: str, rep: Report) -> None:
    """(§11, caso absoluto) Toda tasa lleva su caso absoluto 'NN\\% (n/m)'. Este
    centinela marca los casos absolutos EMPEZADOS pero incompletos ('(37/)' sin
    denominador, o '(37)' sin fracción) tras un porcentaje: son residuos de edición
    que dejan el caso a medias. Determinista y sin falso positivo (no toca umbrales
    ni aproximaciones). Hoy da 0."""
    body = _strip_comments(text)
    for m in _ABS_CASE_INCOMPLETE_RE.finditer(body):
        ln = body.count("\n", 0, m.start()) + 1
        frag = re.sub(r"\s+", " ", m.group(0)).strip()
        rep.add(
            IMPORTANTE,
            "caso-absoluto-incompleto",
            f"(línea {ln}): caso absoluto incompleto '{frag}' (falta el denominador o "
            "la fracción n/m). Completar 'NN% (n/m)' (steering §11).",
            ln,
        )


# ── Runner ───────────────────────────────────────────────────────────────────

# Registro (id → callable) para permitir --only y el conteo por chequeo. Cada
# entrada se adapta a la firma real en _dispatch.
_CHECK_IDS = (
    "row_ratios",
    "campaign_total",
    "column_sums",
    "speedup_semantics",
    "symbol_collisions",
    "missing_resources",
    "speedup_range_mix",
    "letter_x_factor",
    "nmax_single",
    "frontier_arithmetic",
    "cz_convention",
    "grade_scale",
    "written_sums",
    "topology_canon",
    "cross_n_claim_consistency",
    "command_in_prose",
    "stray_spacing",
    "seeds_arithmetic",
    "subset_of_campaign",
    "reference_count",
    "empty_cited_table",
    "declared_vs_published",
    "h_in_table",
    "table_overflow",
    "hmin_consistency",
    "h_grid_denominator",
    "claim_vs_cells",
    "coverage_arithmetic",
    "scale_consistency",
    "ratio_needs_points_column",
    "global_aggregation_claim",
    "dangling_refs",
    "citation_integrity",
    "duplicate_labels",
    "percentage_ceiling",
    "kbar_single_value",
    "hmin_reconciliation",
    "frontier_independent_of_n",
    "hmin_topo_depth_label",
    "hmin_topo_value_source",
    "frontier_slope_positive",
    "passrate_needs_abs_error",
    "abs_case_incomplete",
)


def run(tex_path: Path, tables_dir: Path, only: str | None = None) -> Report:
    rep = Report()
    if not tex_path.exists():
        rep.add(BLOQUEANTE, "archivo", f"no existe el .tex: {tex_path}")
        return rep
    raw = tex_path.read_text(encoding="utf-8")
    text = _strip_comments(raw)

    # id → thunk. Los checks usan 'raw' (con % ya neutralizado dentro) o 'text'
    # (sin comentarios) según su diseño original; se respeta esa elección.
    dispatch = {
        "row_ratios": lambda: check_row_ratios(text, rep),
        "campaign_total": lambda: check_campaign_total(raw, tables_dir, rep),
        "column_sums": lambda: check_column_sums(tables_dir, rep),
        "speedup_semantics": lambda: check_speedup_semantics(raw, rep),
        "symbol_collisions": lambda: check_symbol_collisions(raw, rep),
        "missing_resources": lambda: check_missing_resources(raw, tex_path, rep),
        "speedup_range_mix": lambda: check_speedup_range_mix(raw, rep),
        "letter_x_factor": lambda: check_letter_x_factor(raw, rep),
        "nmax_single": lambda: check_nmax_single(raw, rep),
        "frontier_arithmetic": lambda: check_frontier_arithmetic(raw, rep),
        "cz_convention": lambda: check_cz_convention(raw, rep),
        "grade_scale": lambda: check_grade_scale(raw, tables_dir, rep),
        "written_sums": lambda: check_written_sums(raw, rep),
        "topology_canon": lambda: check_topology_canon(text, rep),
        "cross_n_claim_consistency": lambda: check_cross_n_claim_consistency(text, rep),
        "command_in_prose": lambda: check_command_in_prose(raw, rep),
        "stray_spacing": lambda: check_stray_spacing(raw, rep),
        "seeds_arithmetic": lambda: check_seeds_arithmetic(raw, rep),
        "subset_of_campaign": lambda: check_subset_of_campaign(raw, tables_dir, rep),
        "passrate_needs_abs_error": lambda: check_passrate_needs_abs_error(raw, rep),
        "abs_case_incomplete": lambda: check_abs_case_incomplete(raw, rep),
        "reference_count": lambda: check_reference_count(raw, rep),
        "empty_cited_table": lambda: check_empty_cited_table(text, rep),
        "declared_vs_published": lambda: check_declared_vs_published(text, rep),
        "h_in_table": lambda: check_h_in_table(text, rep),
        "table_overflow": lambda: check_table_overflow(text, rep),
        "hmin_consistency": lambda: check_hmin_consistency(raw, rep),
        "h_grid_denominator": lambda: check_h_grid_denominator(text, rep),
        "claim_vs_cells": lambda: check_claim_vs_cells(raw, rep),
        "coverage_arithmetic": lambda: check_coverage_arithmetic(raw, rep),
        "scale_consistency": lambda: check_scale_consistency(raw, rep),
        "ratio_needs_points_column": lambda: check_ratio_needs_points_column(text, rep),
        "global_aggregation_claim": lambda: check_global_aggregation_claim(raw, rep),
        "dangling_refs": lambda: check_dangling_refs(text, tex_path, rep),
        "citation_integrity": lambda: check_citation_integrity(text, rep),
        "duplicate_labels": lambda: check_duplicate_labels(text, rep),
        "percentage_ceiling": lambda: check_percentage_ceiling(text, rep),
        "kbar_single_value": lambda: check_kbar_single_value(raw, rep),
        "hmin_reconciliation": lambda: check_hmin_reconciliation(raw, rep),
        "frontier_independent_of_n": lambda: check_frontier_independent_of_n(raw, rep),
        "hmin_topo_depth_label": lambda: check_hmin_topo_depth_label(raw, rep),
        "hmin_topo_value_source": lambda: check_hmin_topo_value_source(raw, tex_path, rep),
        "frontier_slope_positive": lambda: check_frontier_slope_positive(raw, rep),
    }
    for cid in _CHECK_IDS:
        if only and cid != only:
            continue
        dispatch[cid]()
    return rep


def _print_report(rep: Report, tex_path: Path, n_checks: int) -> None:
    print(f"🔍 Verificación numérica y de fuentes: {tex_path}")
    checks_with_findings = len({f.check for f in rep.findings})
    print(f"   {n_checks} chequeos ejecutados; {checks_with_findings} con hallazgos.")
    if not rep.findings:
        print("✅ Sin hallazgos: cocientes, totales, símbolos y recursos coherentes.")
        return
    counts = {s: sum(1 for f in rep.findings if f.sev == s) for s in _SEV_ORDER}
    print(f"   Hallazgos — BLOQUEANTE: {counts[BLOQUEANTE]}  IMPORTANTE: {counts[IMPORTANTE]}  INFO: {counts[INFO]}")
    icon = {BLOQUEANTE: "❌", IMPORTANTE: "⚠️ ", INFO: "ℹ️ "}
    for sev in _SEV_ORDER:
        for f in [x for x in rep.findings if x.sev == sev]:
            loc = f"L{f.line}" if f.line else "—"
            print(f"  {icon[sev]} [{f.check}] {loc}: {f.msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Verifica credibilidad numérica de la tesis LaTeX.")
    ap.add_argument("tex", nargs="?", help="ruta al .tex a verificar")
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
    ap.add_argument(
        "--only",
        default=None,
        metavar="CHECK",
        help=f"correr un solo chequeo por id. Opciones: {', '.join(_CHECK_IDS)}",
    )
    ap.add_argument(
        "--list-checks",
        action="store_true",
        help="listar los ids de chequeo disponibles y salir",
    )
    args = ap.parse_args()

    if args.list_checks:
        for cid in _CHECK_IDS:
            print(cid)
        return 0
    if not args.tex:
        ap.error("se requiere la ruta al .tex (o usa --list-checks)")
    if args.only and args.only not in _CHECK_IDS:
        print(f"chequeo desconocido: {args.only!r}. Usa --list-checks.", file=sys.stderr)
        return 2

    tex_path = Path(args.tex).resolve()
    tables_dir = Path(args.tables_dir).resolve() if args.tables_dir else tex_path.parent / "tables"

    rep = run(tex_path, tables_dir, only=args.only)
    n_checks = 1 if args.only else len(_CHECK_IDS)
    _print_report(rep, tex_path, n_checks)

    if args.strict and rep.findings:
        return 1
    return 1 if rep.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
