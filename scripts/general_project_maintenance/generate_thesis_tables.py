#!/usr/bin/env python3
"""Auto-genera tablas LaTeX de la tesis desde las fuentes de verdad del proyecto.

Reutiliza la infraestructura existente (NO re-parsea datos crudos):
  - ``scripts/analysis/generate_best_results_scoreboard.py``
        scan_all_reports / parse_eval_report / compute_best_per_topology_n
        + generate_scoreboard (regenera best_results_scoreboard.json)
  - ``qmbp_simulation.framework.result_index.ResultIndex``  (conteos de campaña)
  - ``qmbp_simulation.analysis.metrics.compute_deploy_summary``  (agregados por-N)

Produce tablas centradas en información general y de perspectiva, en el mismo
espíritu que ``results/best_results_scoreboard.md``:

  auto_scoreboard         Mejor resultado por (topología × N)  [todas las topologías]
  auto_coverage           Matriz de cobertura: grade por topología × N
  auto_campaign           Veredictos/conteos de campaña (incl. taxonomía 46 = 26 + 20)
  auto_heavy_hex_intra_n  Tabla por-h intra-N de heavy_hex (interpolación)
  auto_heavy_hex_large_n  Tabla por-h large-N de heavy_hex (extrapolación zero-shot)

Reglas (steering thesis-style-and-process):
  - Fidelidad al dato: nunca inventa números; donde falta un dato deja un
    marcador ``%TODO-<TOPICO>`` en el lugar exacto del .tex.
  - Todo TODO se vuelca a ``tesis_todos.txt`` agrupado por tópico, con el número
    de línea donde aparece. Las inconsistencias detectadas se anexan al final.

Uso:
    .venv/bin/python scripts/general_project_maintenance/generate_thesis_tables.py
    .venv/bin/python scripts/general_project_maintenance/generate_thesis_tables.py --out-dir internal/tables
    .venv/bin/python scripts/general_project_maintenance/generate_thesis_tables.py --no-refresh
    .venv/bin/python scripts/general_project_maintenance/generate_thesis_tables.py --only auto_scoreboard,auto_coverage
    .venv/bin/python scripts/general_project_maintenance/generate_thesis_tables.py --check-tex internal/tesis-v4.0.tex

Salida:
    <out-dir>/auto_*.tex        una tabla LaTeX por archivo (booktabs)
    <out-dir>/tesis_todos.txt   TODOs por tópico + inconsistencias detectadas
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

DEFAULT_OUT_DIR = ROOT / "internal" / "tables"
SCOREBOARD_JSON = ROOT / "results" / "best_results_scoreboard_p1.json"

# Topología focal del capítulo de resultados (steering thesis-writing)
FOCUS_TOPOLOGY = "heavy_hex"

# Nombres canónicos en español para topologías (steering: unificar grafías)
TOPO_ES = {
    "chain_1d": "cadena 1D",
    "heavy_hex": "heavy-hex",
    "ladder": "escalera",
    "square": "red cuadrada",
    "triangular": "triangular",
    "kagome": "kagome",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Recolector de TODOs e inconsistencias
# ═══════════════════════════════════════════════════════════════════════════════


TODO_MARKER_RE = re.compile(r"^\s*%TODO-([A-Z]+):\s*(.*)$")

# Tono comercial a moderar (steering §6). Palabra completa, sin distinguir mayúsc.
TONE_WORDS_RE = re.compile(
    r"\b(demuestra\w*|garantiza\w*|exhaustiv\w+|notable\w*|excelente\w*|"
    r"potente\w*|radicalment\w*|robust\w+|óptim\w+|optim[oa]s?|"
    r"ampliamente validad\w+|madurez|coste cuántico cero|días a minutos)\b",
    re.IGNORECASE,
)
# Métricas: speedup (eliminado) y PassRate como sujeto principal (steering §5).
SPEEDUP_RE = re.compile(r"\b(speedup|aceleraci[oó]n de\s*\d|factor de mejora)\b", re.IGNORECASE)
PASSRATE_RE = re.compile(r"\b(passrate|tasa de aprobaci[oó]n)\b", re.IGNORECASE)
ABS_ERR_RE = re.compile(r"\\Delta E|\|\\Delta E\||error(?:\s+energético)?\s+absoluto")
# Reproducibilidad: "exacto/a" junto a DMRG/MPS, y N grande + statevector.
EXACT_TN_RE = re.compile(
    r"\b(exact[oa]s?)\b[^.]*\b(DMRG|MPS)\b|\b(DMRG|MPS)\b[^.]*\b(exact[oa]s?)\b", re.IGNORECASE
)
SV_BIGN_RE = re.compile(r"statevector", re.IGNORECASE)
BIGN_RE = re.compile(r"N\s*[=>]\s*(\d+)")


@dataclass
class TodoCollector:
    """Acumula marcadores TODO por tópico e inconsistencias detectadas.

    Los generadores insertan comentarios ``%TODO-<TOPICO>: ...`` como líneas
    normales de la tabla mediante :meth:`marker`. Tras escribir cada archivo,
    :meth:`scan_file` re-lee el ``.tex`` y registra cada TODO con su número de
    línea REAL en el archivo final (no una posición relativa estimada).
    """

    # topico -> lista de (archivo, linea_real, descripcion)
    todos: dict[str, list[tuple[str, int, str]]] = field(default_factory=lambda: defaultdict(list))
    inconsistencies: list[str] = field(default_factory=list)

    @staticmethod
    def marker(topic: str, description: str) -> str:
        """Devuelve el comentario LaTeX a insertar en la tabla."""
        return f"%TODO-{topic}: {description}"

    def scan_file(self, path: Path, rel_name: str) -> None:
        """Escanea un .tex ya escrito y registra TODOs con su línea real."""
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            m = TODO_MARKER_RE.match(line)
            if m:
                self.todos[m.group(1)].append((rel_name, i, m.group(2).strip()))

    def add_inconsistency(self, description: str) -> None:
        self.inconsistencies.append(description)

    def n_todos(self) -> int:
        return sum(len(v) for v in self.todos.values())

    def render_txt(self) -> str:
        """Genera el contenido de tesis_todos.txt."""
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            "TODOs de la tesis — generado automáticamente",
            f"Generado: {now}",
            "Fuente: scripts/general_project_maintenance/generate_thesis_tables.py",
            f"Total TODOs: {self.n_todos()} | Inconsistencias: {len(self.inconsistencies)}",
            "=" * 72,
            "",
        ]

        if not self.todos:
            lines.append("(sin TODOs pendientes)")
        else:
            for topic in sorted(self.todos):
                entries = self.todos[topic]
                lines.append(f"## TODO-{topic}  ({len(entries)})")
                for filename, line_no, desc in entries:
                    lines.append(f"  [{filename}:{line_no}] {desc}")
                lines.append("")

        lines.append("=" * 72)
        lines.append(f"## INCONSISTENCIAS / ERRORES DETECTADOS  ({len(self.inconsistencies)})")
        lines.append("")
        if not self.inconsistencies:
            lines.append("(ninguna detectada en esta ejecución)")
        else:
            for i, desc in enumerate(self.inconsistencies, 1):
                lines.append(f"  {i}. {desc}")
        lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Utilidades de formato LaTeX
# ═══════════════════════════════════════════════════════════════════════════════


def _esc(text: str) -> str:
    """Escapa caracteres especiales de LaTeX en texto plano."""
    if text is None:
        return ""
    out = str(text)
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        out = out.replace(a, b)
    return out


def _num(value: float, decimals: int = 4) -> str:
    """Formatea un número con coma decimal (convención española, steering §6)."""
    if value is None:
        return "---"
    return f"{value:.{decimals}f}".replace(".", ",")


def _topo_es(topo: str) -> str:
    return TOPO_ES.get(topo, topo.replace("_", " "))


def _wrap_table(
    caption: str,
    label: str,
    col_spec: str,
    header: list[str],
    rows: list[list[str]],
    notes: str = "",
    pre_lines: list[str] | None = None,
) -> list[str]:
    """Envuelve una tabla en el entorno LaTeX estándar con booktabs."""
    lines: list[str] = []
    lines.append(
        "% ==== AUTO-GENERADA — no editar a mano (regenerar con generate_thesis_tables.py) ===="
    )
    if pre_lines:
        lines.extend(pre_lines)
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if notes:
        lines.append(f"\\\\[2pt]\n\\footnotesize {notes}")
    lines.append("\\end{table}")
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# Carga de datos (reutiliza generadores existentes)
# ═══════════════════════════════════════════════════════════════════════════════


def load_scoreboard(refresh: bool, collector: TodoCollector) -> dict:
    """Carga best_results_scoreboard.json, regenerándolo si se solicita."""
    if refresh:
        try:
            from generate_best_results_scoreboard import generate_scoreboard

            generate_scoreboard(output_json=True)
        except Exception as e:  # noqa: BLE001
            collector.add_inconsistency(
                f"No se pudo regenerar el scoreboard ({e}); se usa el JSON existente si lo hay."
            )
    if not SCOREBOARD_JSON.exists():
        collector.add_inconsistency(
            f"No existe {SCOREBOARD_JSON.relative_to(ROOT)}: correr "
            "generate_best_results_scoreboard.py --json primero."
        )
        return {}
    return json.loads(SCOREBOARD_JSON.read_text(encoding="utf-8"))


def load_heavy_hex_per_h(collector: TodoCollector) -> dict[int, list]:
    """Devuelve {N: [PerHResult...]} para heavy_hex agrupando por N (mejor por h).

    Reutiliza ``parse_eval_report`` del generador del scoreboard (misma lógica de
    parsing, dedup y validación física) pero con una tolerancia de h muy amplia
    para aceptar TODOS los puntos h (las tablas por-h necesitan el barrido
    completo, no solo h≈2.5).
    """
    try:
        import generate_best_results_scoreboard as sb
    except Exception as e:  # noqa: BLE001
        collector.add_inconsistency(f"No se pudo importar el generador del scoreboard: {e}")
        return {}

    eval_dir = ROOT / "results" / "extrapolation_evals"
    # Aceptar todo el rango de h reutilizando el parser existente.
    saved_tol = sb.H_TOLERANCE
    sb.H_TOLERANCE = 1e9
    per_n: dict[int, dict[float, sb.PerHResult]] = defaultdict(dict)
    try:
        for topo_dir in sorted(eval_dir.glob(f"{FOCUS_TOPOLOGY}_p*")):
            for report in sorted(topo_dir.glob("eval_*.md")):
                try:
                    entries = sb.parse_eval_report(report, target_h=0.0)
                except Exception as exc:  # noqa: BLE001
                    collector.add_inconsistency(f"Error parseando {report.name}: {exc}")
                    continue
                for entry in entries:
                    r = entry.result
                    h_key = round(r.h, 2)
                    prev = per_n[entry.n_qubits].get(h_key)
                    # Quedarse con el mejor |ΔE| por (N, h)
                    if prev is None or r.abs_error < prev.abs_error:
                        per_n[entry.n_qubits][h_key] = r
    finally:
        sb.H_TOLERANCE = saved_tol

    # Colapsar {h: PerHResult} -> lista ordenada por h
    return {n: [hmap[h] for h in sorted(hmap)] for n, hmap in per_n.items()}


def load_campaign_index(collector: TodoCollector) -> list[dict]:
    """Carga las entradas del ResultIndex (conteos de campaña)."""
    try:
        from qmbp_simulation.framework.result_index import ResultIndex

        idx = ResultIndex()
        return list(idx.entries)
    except Exception as e:  # noqa: BLE001
        collector.add_inconsistency(f"No se pudo cargar ResultIndex: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Generadores de tablas
# ═══════════════════════════════════════════════════════════════════════════════


def gen_scoreboard(scoreboard: dict, collector: TodoCollector) -> list[str]:
    """Tabla resumen: mejor grade por topología (perspectiva general)."""
    by_topo = scoreboard.get("best_by_topology", {})
    if not by_topo:
        return [collector.marker("DATOS", "best_by_topology vacío en el scoreboard JSON.")]

    header = ["Topología", "$N$ máx.", "Mejor grade", "Mejor $|\\Delta E|$", "Tipo", "Fuente"]
    rows: list[list[str]] = []
    for topo in sorted(by_topo):
        n_results = by_topo[topo]
        if not n_results:
            continue
        ns = sorted(int(k) for k in n_results)
        best = min(n_results.values(), key=lambda r: r["best_abs_error"])
        date_str = best.get("date", "")[:10]
        rows.append(
            [
                _esc(_topo_es(topo)),
                str(max(ns)),
                _esc(best["grade"]),
                _num(best["best_abs_error"]),
                _esc(best["model_type"]),
                _esc(date_str) if date_str else "---",
            ]
        )

    corte = scoreboard.get("generated_at", "")[:10]
    n_reports = scoreboard.get("n_reports_scanned", "?")
    caption = (
        "Mejor resultado por topología en el punto de operación más exigente "
        f"($h \\approx {_num(scoreboard.get('target_h', 2.5), 2)}$). "
        "Grade por $|\\Delta E|$: A ($<0{,}05$), B ($<0{,}10$), C ($<0{,}30$), "
        "D ($<1{,}00$), F ($\\geq 1{,}00$)."
    )
    notes = (
        f"Fuente: \\texttt{{best\\_results\\_scoreboard.json}} "
        f"(corte {corte}, {n_reports} reportes). "
        "Tipo: ST = modelo por topología, MT = multi-topología."
    )
    return _wrap_table(caption, "tab:auto_scoreboard", "lccccc", header, rows, notes=notes)


def gen_coverage(
    scoreboard: dict, collector: TodoCollector, coverage_ns: set[int] | None = None
) -> list[str]:
    """Matriz de cobertura: grade alcanzado por topología × N.

    Si ``coverage_ns`` se provee, restringe las columnas a esos tamaños (evita
    una tabla demasiado ancha para la página).
    """
    by_topo = scoreboard.get("best_by_topology", {})
    if not by_topo:
        return [collector.marker("DATOS", "best_by_topology vacío en el scoreboard JSON.")]

    # Conjunto de N presentes (columnas). Si se pasó coverage_ns, filtrar a esos.
    all_ns = sorted({int(n) for tr in by_topo.values() for n in tr})
    if coverage_ns:
        all_ns = [n for n in all_ns if n in coverage_ns]
    header = ["Topología"] + [f"$N{{=}}{n}$" for n in all_ns]
    rows: list[list[str]] = []
    for topo in sorted(by_topo):
        n_results = by_topo[topo]
        row = [_esc(_topo_es(topo))]
        for n in all_ns:
            entry = n_results.get(str(n))
            row.append(_esc(entry["grade"]) if entry else "---")
        rows.append(row)

    col_spec = "l" + "c" * len(all_ns)
    caption = (
        "Matriz de cobertura: grade alcanzado por topología y tamaño del sistema "
        f"$N$ en $h \\approx {_num(scoreboard.get('target_h', 2.5), 2)}$."
    )
    notes = "Un guion (---) indica que no hay resultado evaluado para esa combinación."
    return _wrap_table(caption, "tab:auto_coverage", col_spec, header, rows, notes=notes)


def gen_campaign(index: list[dict], collector: TodoCollector) -> list[str]:
    """Conteos de campaña por modelo, con la taxonomía de conteo del steering."""
    if not index:
        return [
            collector.marker("CAMPANA", "ResultIndex vacío: no se pudieron contar ejecuciones.")
        ]

    # Filtrar entradas con modelo válido
    valid = [e for e in index if e.get("model")]
    by_model: dict[str, list[dict]] = defaultdict(list)
    for e in valid:
        by_model[e["model"]].append(e)

    header = ["Modelo", "Ejecuciones", "Aprobadas", "Tasa aprob."]
    rows: list[list[str]] = []
    for model in sorted(by_model):
        runs = by_model[model]
        n = len(runs)
        n_pass = sum(1 for r in runs if r.get("passed"))
        rate = n_pass / n if n else 0.0
        rows.append(
            [
                _esc(model),
                str(n),
                str(n_pass),
                f"{n_pass}/{n} ({_num(rate * 100, 1)}\\%)",
            ]
        )

    # Taxonomía Heisenberg 46 = 26 + 20 (verificar contra el índice)
    n_heis = len(by_model.get("heisenberg", []))
    n_heis_t = len(by_model.get("heisenberg_transverse", []))
    pre_lines: list[str] = []
    total_heis = n_heis + n_heis_t
    if total_heis:
        expected = " (46 = 26 XXZ + 20 transversal)" if total_heis == 46 else ""
        note = (
            f"Heisenberg: {n_heis} XXZ + {n_heis_t} transversal = {total_heis} ejecuciones"
            f"{expected}."
        )
        if total_heis != 46:
            collector.add_inconsistency(
                f"Conteo Heisenberg en ResultIndex = {total_heis} "
                f"({n_heis} XXZ + {n_heis_t} transversal), no 46 como en la tesis. "
                "Verificar campaña o texto."
            )
            # marcador visible en el .tex
            pre_lines.append(
                collector.marker(
                    "CAMPANA",
                    f"ResultIndex reporta {total_heis} runs Heisenberg "
                    f"({n_heis} XXZ + {n_heis_t} transv.), la tesis dice 46. Revisar.",
                )
            )

    caption = (
        "Conteo de ejecuciones del pipeline por modelo en la campaña experimental "
        "(fuente: ResultIndex). Una \\emph{ejecución} es una corrida completa de "
        "las Fases 1--3 para una (configuración, semilla)."
    )
    notes = note if total_heis else ""
    return _wrap_table(
        caption, "tab:auto_campaign", "lccc", header, rows, notes=notes, pre_lines=pre_lines or None
    )


def _per_h_table(
    per_n: dict[int, list],
    n_values: list[int],
    label: str,
    caption: str,
    collector: TodoCollector,
    topic_kind: str,
) -> list[str]:
    """Construye una tabla por-N (una fila por N) con métricas agregadas por-h.

    Columnas: N, rango de h, nº puntos, ΔE/gap (media), |ΔE| (media), gap (media).
    La fidelidad por-h NO está en los eval reports -> se marca %TODO-DATOS.
    """
    from qmbp_simulation.analysis.metrics import compute_deploy_summary

    # Métrica primaria |ΔE| primero (steering §5); ΔE/gap relegado a apoyo.
    header = [
        "$N$",
        "Rango de $h$",
        "Ptos.",
        "$|\\Delta E|$",
        "$|\\Delta E|/N$",
        "gap",
        "$\\Delta E/\\mathrm{gap}$",
    ]
    rows: list[list[str]] = []
    pre_lines: list[str] = []

    missing_ns = [n for n in n_values if n not in per_n or not per_n[n]]
    if missing_ns:
        pre_lines.append(
            collector.marker(
                topic_kind,
                f"Faltan datos por-h de heavy_hex para N={missing_ns}; "
                "no hay eval report per-h para esos tamaños.",
            )
        )

    for n in n_values:
        results = per_n.get(n)
        if not results:
            continue
        hs = [r.h for r in results]
        per_h_dicts = [{"de_gap": r.de_gap, "abs_error": r.abs_error} for r in results]
        summary = compute_deploy_summary(per_h_dicts)
        mean_gap = sum(r.gap for r in results) / len(results)
        mean_abs = summary.get("mean_abs_error", 0.0)
        rows.append(
            [
                str(n),
                f"[{_num(min(hs), 2)}, {_num(max(hs), 2)}]",
                str(summary["n_points"]),
                _num(mean_abs),
                _num(mean_abs / n, 5),
                _num(mean_gap),
                _num(summary["mean_de_gap"]),
            ]
        )

    if not rows:
        return [collector.marker(topic_kind, f"Sin datos por-h de heavy_hex para {label}.")]

    notes = (
        "Métricas promediadas sobre los puntos $h$ del reporte de evaluación, "
        "ordenadas por la métrica primaria $|\\Delta E|$ (error energético absoluto "
        "respecto al estado de referencia; energías en unidades de $J = 1$). "
        "$|\\Delta E|/N$: error por sitio (comparable entre tamaños). "
        "$\\Delta E/\\mathrm{gap}$ se incluye como referencia normalizada, no como "
        "criterio de calidad. La dispersión entre puntos $h$ no se agrega aquí "
        "(véase el barrido completo por $h$)."
    )
    return _wrap_table(caption, label, "rlrrrrr", header, rows, notes=notes, pre_lines=pre_lines)


def gen_heavy_hex_intra_n(per_n: dict[int, list], collector: TodoCollector) -> list[str]:
    """Tabla intra-N (interpolación) de heavy_hex: N de entrenamiento."""
    intra_ns = [n for n in sorted(per_n) if n <= 20]
    caption = (
        "Aprendizaje intra-$N$ (interpolación) en heavy-hex: la GNN reproduce "
        "$\\theta^*(h)$ en valores de $h$ no vistos, para cada $N$ de entrenamiento."
    )
    return _per_h_table(per_n, intra_ns, "tab:auto_heavy_hex_intra_n", caption, collector, "DATOS")


def gen_heavy_hex_large_n(per_n: dict[int, list], collector: TodoCollector) -> list[str]:
    """Tabla large-N (extrapolación zero-shot) de heavy_hex."""
    large_ns = [n for n in sorted(per_n) if n > 20]
    caption = (
        "Extrapolación cross-$N$ (sobre casos no observados) en heavy-hex: la "
        "UnifiedMPNN entrenada con $N$ chicos predice ángulos para $N$ grandes."
    )
    return _per_h_table(per_n, large_ns, "tab:auto_heavy_hex_large_n", caption, collector, "DATOS")


# ═══════════════════════════════════════════════════════════════════════════════
# Detección de inconsistencias (a partir del scoreboard JSON)
# ═══════════════════════════════════════════════════════════════════════════════


def detect_inconsistencies(scoreboard: dict, collector: TodoCollector) -> None:
    """Propaga inconsistencias del scoreboard (grades vs gaps, campañas mezcladas)."""
    by_topo = scoreboard.get("best_by_topology", {})
    for topo, n_results in by_topo.items():
        # Campañas mezcladas: fechas muy dispersas dentro de una misma topología
        dates = sorted({e.get("date", "")[:10] for e in n_results.values() if e.get("date")})
        if len(dates) > 1:
            span = f"{dates[0]}..{dates[-1]}"
            # Solo informativo; no bloquea
            if dates[0][:7] != dates[-1][:7]:
                collector.add_inconsistency(
                    f"{topo}: resultados de campañas de distintos meses ({span}); "
                    "verificar que no se mezclen criterios/rangos de h en una misma tabla."
                )
        # Grade A/B con gap muy pequeño (posible gap_masked cerca de frontera)
        for n, e in n_results.items():
            if e.get("grade") in ("A", "B") and e.get("gap_at_best", 1.0) < 0.5:
                collector.add_inconsistency(
                    f"{topo} N={n}: grade {e['grade']} con gap={_num(e['gap_at_best'], 3)} "
                    "pequeño; posible enmascaramiento por gap (revisar |ΔE| absoluto)."
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Chequeos de consistencia sobre el documento LaTeX
# ═══════════════════════════════════════════════════════════════════════════════


def check_tex(tex_path: Path, out_dir: Path, collector: TodoCollector) -> None:
    """Chequeos ligeros de consistencia sobre el .tex, alimentando el TODO log.

    Registra hallazgos como inconsistencias (con [archivo:línea]) para:
      - refs/labels: \\ref sin \\label (rompe -> ??), y label de tabla sin \\ref.
      - decimales con punto en lugar de coma (convención española del steering).
      - tablas auto_*.tex no conectadas al documento (sin \\input o label sin \\ref).
    """
    if not tex_path.exists():
        collector.add_inconsistency(f"No existe el .tex a chequear: {tex_path}")
        return

    lines = tex_path.read_text(encoding="utf-8").splitlines()
    rel = tex_path.name

    labels: dict[str, int] = {}  # label -> primera linea
    refs: dict[str, int] = {}  # ref -> primera linea
    inputs: set[str] = set()  # basenames incluidos via \input
    in_verbatim = False

    for i, line in enumerate(lines, start=1):
        # Ignorar comentarios completos y bloques verbatim/lstlisting
        if re.search(r"\\begin\{(verbatim|lstlisting)\}", line):
            in_verbatim = True
        if re.search(r"\\end\{(verbatim|lstlisting)\}", line):
            in_verbatim = False
            continue
        if in_verbatim:
            continue
        # Quitar comentario de línea respetando el porcentaje escapado (\%)
        code = re.sub(r"(?<!\\)%.*$", "", line)
        if not code.strip():
            continue

        for m in re.finditer(r"\\label\{([^}]+)\}", code):
            labels.setdefault(m.group(1), i)
        for m in re.finditer(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", code):
            refs.setdefault(m.group(1), i)
        for m in re.finditer(r"\\input\{([^}]+)\}", code):
            inputs.add(Path(m.group(1)).name.replace(".tex", ""))

        # (2) Decimales con punto: número.número (no versiones tipo v2.x ni URLs)
        for m in re.finditer(r"(?<![\w.])\d+\.\d+(?![\w.])", code):
            collector.add_inconsistency(
                f"[{rel}:{i}] decimal con punto '{m.group(0)}' "
                "(usar coma decimal en tablas/texto, steering §7)."
            )

        # (5) Tono comercial (steering §6): moderar, añadir matiz cuantitativo.
        for m in TONE_WORDS_RE.finditer(code):
            collector.add_inconsistency(
                f"[{rel}:{i}] %TODO-TONO tono a moderar '{m.group(0)}' "
                "(usar muestra/sugiere; añadir comparación cuantitativa o "
                "'dentro de las configuraciones evaluadas')."
            )

        # (6) Métricas (steering §5): speedup eliminado; PassRate sin |ΔE|.
        for m in SPEEDUP_RE.finditer(code):
            collector.add_inconsistency(
                f"[{rel}:{i}] %TODO-METRICA speedup/aceleración '{m.group(0)}' "
                "(eliminar hasta tener definición reproducible de coste)."
            )
        if PASSRATE_RE.search(code) and not ABS_ERR_RE.search(code):
            collector.add_inconsistency(
                f"[{rel}:{i}] %TODO-METRICA PassRate sin |ΔE| acompañante "
                "(reportar la métrica primaria |ΔE|, steering §5)."
            )

        # (7) Reproducibilidad (steering §8): 'exacto' + DMRG/MPS; statevector a N grande.
        if EXACT_TN_RE.search(code):
            collector.add_inconsistency(
                f"[{rel}:{i}] %TODO-REPRO 'exacto' aplicado a DMRG/MPS "
                "(usar 'convergido dentro de la tolerancia'; mostrar convergencia en χ)."
            )
        if SV_BIGN_RE.search(code):
            for bm in BIGN_RE.finditer(code):
                if int(bm.group(1)) > 22:
                    collector.add_inconsistency(
                        f"[{rel}:{i}] %TODO-REPRO statevector con N={bm.group(1)}>22 "
                        "(inviable en memoria; ¿era backend MPS? indicar χ y tolerancia)."
                    )

    # (1) refs -> label inexistente (renderiza como ??)
    for ref, ln in sorted(refs.items(), key=lambda kv: kv[1]):
        if ref not in labels:
            collector.add_inconsistency(
                f"[{rel}:{ln}] \\ref{{{ref}}} sin \\label correspondiente (saldrá como ??)."
            )
    # (1b) label de tabla nunca referenciado
    for lab, ln in sorted(labels.items(), key=lambda kv: kv[1]):
        if lab.startswith("tab:") and lab not in refs:
            collector.add_inconsistency(
                f"[{rel}:{ln}] tabla \\label{{{lab}}} nunca referenciada con \\ref."
            )

    # (4) tablas auto_*.tex no conectadas al documento
    for auto_tex in sorted(out_dir.glob("auto_*.tex")):
        stem = auto_tex.stem
        auto_label = f"tab:{stem}"
        if stem not in inputs:
            collector.add_inconsistency(
                f"tabla generada '{auto_tex.name}' no está incluida en {rel} "
                f"(falta \\input{{tables/{stem}}})."
            )
        elif auto_label not in refs:
            collector.add_inconsistency(
                f"tabla generada '{auto_tex.name}' incluida pero su \\label{{{auto_label}}} "
                "no se referencia con \\ref."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Orquestación
# ═══════════════════════════════════════════════════════════════════════════════

GENERATORS = {
    "auto_scoreboard": ("scoreboard", gen_scoreboard),
    "auto_coverage": ("scoreboard", gen_coverage),
    "auto_campaign": ("index", gen_campaign),
    "auto_heavy_hex_intra_n": ("per_n", gen_heavy_hex_intra_n),
    "auto_heavy_hex_large_n": ("per_n", gen_heavy_hex_large_n),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directorio de salida (default: {DEFAULT_OUT_DIR.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Lista separada por comas de tablas a generar (default: todas)",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="No regenerar el scoreboard JSON; usar el existente",
    )
    parser.add_argument(
        "--coverage-ns",
        type=str,
        default="",
        help="Lista de N (coma) para las columnas de auto_coverage "
        "(default: todos los N presentes)",
    )
    parser.add_argument(
        "--check-tex",
        type=Path,
        default=None,
        help="Chequear consistencia de un .tex (refs/labels, decimales, "
        "tablas auto no conectadas) y volcar hallazgos a tesis_todos.txt",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    collector = TodoCollector()
    only = {s.strip() for s in args.only.split(",") if s.strip()} or set(GENERATORS)
    coverage_ns: set[int] | None = None
    if args.coverage_ns:
        try:
            coverage_ns = {int(x) for x in args.coverage_ns.split(",") if x.strip()}
        except ValueError:
            print("  ⚠️ --coverage-ns inválido; se ignora", file=sys.stderr)

    # Cargar fuentes (perezoso según lo que se pida)
    scoreboard = load_scoreboard(refresh=not args.no_refresh, collector=collector)
    index = load_campaign_index(collector) if "auto_campaign" in only else []
    per_n = (
        load_heavy_hex_per_h(collector)
        if only & {"auto_heavy_hex_intra_n", "auto_heavy_hex_large_n"}
        else {}
    )

    if scoreboard:
        detect_inconsistencies(scoreboard, collector)

    written: list[str] = []
    for table_id, (src, gen) in GENERATORS.items():
        if table_id not in only:
            continue
        if table_id == "auto_coverage":
            lines = gen(scoreboard, collector, coverage_ns)
        elif src == "scoreboard":
            lines = gen(scoreboard, collector)
        elif src == "index":
            lines = gen(index, collector)
        else:
            lines = gen(per_n, collector)
        out_path = out_dir / f"{table_id}.tex"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        # Registrar TODOs con su número de línea REAL en el archivo final
        collector.scan_file(out_path, f"{table_id}.tex")
        written.append(str(out_path.relative_to(ROOT)))
        print(f"  ✅ {out_path.relative_to(ROOT)}")

    # Chequeo del documento LaTeX (opcional)
    if args.check_tex is not None:
        check_tex(args.check_tex, out_dir, collector)
        print(f"  🔍 Chequeo LaTeX: {args.check_tex}")

    # Volcar tesis_todos.txt
    todos_path = out_dir / "tesis_todos.txt"
    todos_path.write_text(collector.render_txt(), encoding="utf-8")
    print(
        f"  📝 {todos_path.relative_to(ROOT)} "
        f"({collector.n_todos()} TODOs, {len(collector.inconsistencies)} inconsistencias)"
    )

    print(f"\n  {len(written)} tablas generadas en {out_dir.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
