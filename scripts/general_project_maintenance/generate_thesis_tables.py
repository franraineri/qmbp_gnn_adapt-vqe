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
  auto_coverage           Matriz de cobertura: calificación por topología × N
  auto_campaign           Veredictos/conteos de campaña (conteo derivado del ResultIndex)
  auto_heavy_hex_intra_n  Tabla por-h intra-N de heavy_hex (interpolación)
  auto_heavy_hex_large_n  Tabla por-h large-N de heavy_hex (extrapolación zero-shot)

────────────────────────────────────────────────────────────────────────────────
División de responsabilidades con los otros dos verificadores (evitar duplicación)
────────────────────────────────────────────────────────────────────────────────
Existen tres verificadores complementarios; cada uno es la ÚNICA fuente de verdad
de su dimensión. No re-implementar una regla que ya vive en otro:

  1. ESTE script (``--check-tex``)  →  ESTILO + CONSISTENCIA INTERNA del .tex y las
     tablas auto: LaTeX (math/entornos/llaves/refs), aritmética de las tablas AUTO
     y su prosa (cociente, suma Total, escala de calificación), plantilla (§4:
     keywords, índice de acrónimos, fuente en captions), bibliografía completa (§9),
     anglicismos/tono/guiones, y los cruces contra las fuentes vivas del proyecto
     (noiseless_v2, scoreboard JSON, dashboard). Es el linter rico del documento.

  2. ``verify_thesis_numbers.py``  →  CREDIBILIDAD NUMÉRICA de valores EMBEBIDOS a
     mano en el .tex que este generador no produce (cocientes fila-a-fila de tablas
     manuales como critical_sweep, sumas escritas 'a+b=c', semántica A/R del 4414×,
     colisiones de símbolo, recursos \\includegraphics/\\input, CZ=2·E·p, escala
     A–E). Autocontenido (stdlib). Se solapa a propósito en unas pocas reglas
     (CZ, escala, A/R) porque opera sin importar el proyecto; si se unifican, la
     copia canónica de esas reglas es la de ESTE script (tiene el contexto de datos).

  3. ``verify_thesis_pdf.py``  →  el ARTEFACTO RENDERIZADO (pdftotext): casos
     absolutos p%(n/m) y matriz de confusión tal como los lee el tribunal.

Regla práctica: una cifra que este generador PRODUCE se valida aquí; una cifra
EMBEBIDA a mano se valida en verify_thesis_numbers; una cifra IMPRESA en el PDF se
valida en verify_thesis_pdf.

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
DASHBOARD_JSON = ROOT / "data" / "model_quality_dashboard.json"

# Topología focal del capítulo de resultados (steering thesis-writing)
FOCUS_TOPOLOGY = "heavy_hex"

# ─────────────────────────────────────────────────────────────────────────────
# Grillas de h canónicas por régimen (consistencia intra-sección, steering §1/§11).
# Cada régimen usa UNA sola grilla; las tablas de un régimen comparten denominador.
#   - Extrapolación/comparación (h >= 2.5): grilla común verificada presente en las
#     5 topologías. Se usa para comparar topologías y tamaños entre sí.
#   - Crítico/frontera (h en [1.3, 2.4]): grilla densa para caracterizar dónde falla
#     el HVA cerca de h_c; la densidad es intrínseca al mensaje (no se uniformiza a 0.5).
# Tolerancia para casar un h medido con un nodo de la grilla.
H_GRID_EXTRAPOLATION = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
H_GRID_TOL = 0.02

# Régimen de h por tabla (para el chequeo de coherencia de rangos). Cada régimen
# tiene un intervalo canónico esperado; una tabla que declare un h fuera de él
# (o con otra convención) se señala. "critico" = frontera densa [1.3, 5.0];
# "extrapolacion" = grilla [2.5, 5.0]. Los rangos son los límites admisibles.
H_REGIME_BY_LABEL = {
    "tab:cross_topo": ("critico", 1.3, 5.0),
    "tab:cross_topo_depth": ("critico", 1.3, 5.0),
    "tab:scaling": ("critico", 1.3, 5.0),
    "tab:cross_topo_de": ("critico", 1.3, 5.0),
    "tab:delta_e_vs_p": ("critico", 1.3, 5.0),
    "tab:tfim_long_deploy": ("critico", 1.3, 5.0),
    "tab:large_n_chain": ("extrapolacion", 2.5, 5.0),
    "tab:auto_heavy_hex_intra_n": ("extrapolacion", 2.5, 5.0),
    "tab:auto_heavy_hex_large_n": ("extrapolacion", 2.5, 5.0),
}


def snap_to_grid(h: float, grid: list[float], tol: float = H_GRID_TOL) -> float | None:
    """Devuelve el nodo de la grilla más cercano a h si está dentro de tol, si no None."""
    best = min(grid, key=lambda g: abs(g - h))
    return best if abs(best - h) <= tol else None


def load_h_frontier(topology: str, p_layers: int = 1) -> dict[int, float]:
    """Devuelve {N: h_frontier} para una topología desde el dashboard de calidad.

    h_frontier es el h por debajo del cual el pipeline no aprueba (régimen no
    válido). Se usa para restringir las tablas por-h al régimen operativo válido
    (h >= h_frontier), coherente con la narrativa de interpolación exitosa.
    """
    if not DASHBOARD_JSON.exists():
        return {}
    try:
        d = json.loads(DASHBOARD_JSON.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    out: dict[int, float] = {}
    for c in d.get("configs", []):
        if c.get("topology") != topology or c.get("p_layers", 1) != p_layers:
            continue
        hf = c.get("h_frontier")
        if hf is not None:
            out[int(c["n_qubits"])] = float(hf)
    return out


# Nombres canónicos en español para topologías (steering: unificar grafías)
TOPO_ES = {
    "chain_1d": "cadena 1D",
    "heavy_hex": "heavy-hex",
    "ladder": "escalera",
    "square": "cuadrada",
    "triangular": "triangular",
    "kagome": "kagome",
}

# Nombres de modelo en prosa (no claves snake_case) para tablas de la tesis.
MODEL_ES = {
    "tfim": "TFIM",
    "tfim_longitudinal": "TFIM longitudinal",
    "tfim_frustrated": "TFIM frustrado ($J_1$-$J_2$)",
    "tfim_bond_resolved": "TFIM por enlace",
    "heisenberg": "Heisenberg XXZ",
    "heisenberg_transverse": "Heisenberg transversal",
    "kitaev": "Cadena de Kitaev",
    "xy": "XY",
}

# Modelos excluidos de las tablas de conteo de campaña (decisión de la tesis):
# XY tiene una base muestral muy fina (7 ejecuciones) y no forma parte de la
# narrativa (objetivos, hipótesis, viabilidad ni conclusiones). Ver steering §16.
CAMPAIGN_EXCLUDED_MODELS = {"xy"}

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
# Error por sitio |ΔE|/N: PROHIBIDO (steering §5, se retiró por no aportar sobre
# |ΔE|). Detecta las formas LaTeX: |\Delta E|/N, \Delta E/N, \frac{\Delta E}{N},
# y la prosa "error por sitio" / "por sitio".
PER_SITE_RE = re.compile(
    r"\\Delta\s*E\s*(?:\\rvert|\|)?\s*/\s*N\b"
    r"|\\frac\{[^{}]*\\Delta\s*E[^{}]*\}\{\s*N\s*\}"
    r"|error\s+por\s+sitio|\bpor\s+sitio\b|per[-\s]site"
)
PASSRATE_RE = re.compile(r"\b(passrate|tasa de aprobaci[oó]n)\b", re.IGNORECASE)
ABS_ERR_RE = re.compile(r"\\Delta E|\|\\Delta E\||error(?:\s+energético)?\s+absoluto")
# "óptimo" TÉCNICO (no laudatorio): parámetros/ángulos/theta óptimos, profundidad
# óptima, configuración óptima, inicialización óptima, valor óptimo. Es el óptimo
# de una optimización (mínimo de la función objetivo), no un adjetivo comercial.
TONE_OPTIMO_TECNICO_RE = re.compile(
    r"(par[aá]metros?|[aá]ngulos?|theta|\\theta|profundidad|configuraci[oó]n|"
    r"inicializaci[oó]n|valor(es)?|punto)\s+(\w+\s+)?(casi[-\s])?[oó]ptim",
    re.IGNORECASE,
)
# "robusto/a/ez" TÉCNICO (no laudatorio): propiedad medible de un optimizador
# frente al ruido, de un estadístico (mediana) frente a valores atípicos, o de un
# resultado frente a las semillas. Es una propiedad, no un adjetivo comercial.
TONE_ROBUSTO_TECNICO_RE = re.compile(
    r"robust\w+\s+(a|al|de|frente\s+a|ante|entre|respecto)\b"
    r"[^.]{0,40}?(ruido|muestreo|atípic\w+|semilla|inicializaci|perturbaci)",
    re.IGNORECASE,
)
# "demuestra/demostraron" que REPORTA un resultado de la literatura citada (los
# autores de X demostraron ...) no es una afirmación fuerte propia: es reporte fiel
# de un teorema publicado. Se detecta por una cita cercana o un sujeto autoral.
TONE_DEMUESTRA_CITA_RE = re.compile(
    r"(\\cite[pt]?\{|los\s+autores|autores\s+de|\bet\s+al)",
    re.IGNORECASE,
)
# Casos negativos (Heisenberg, Kitaev): el indicador de calidad es la fidelidad F,
# NO |ΔE| (un error energético carece de sentido cuando F≈0, steering §3.4). Una
# línea que menciona estos modelos con su 0% no necesita |ΔE| acompañante.
NEGATIVE_MODEL_RE = re.compile(r"Heisenberg|Kitaev", re.IGNORECASE)
# Anglicismos de métrica prohibidos en el cuerpo (usar equivalentes en español).
ENGLISH_METRIC_RE = re.compile(r"\b(Grade|Pass|Rate|PassRate)\b")
# Énfasis en profundidad requerida (a de-enfatizar, steering §5): p ∝ N, p ≈ N/2, p = N-1.
DEPTH_EMPHASIS_RE = re.compile(r"p\s*\\propto\s*N|p\s*\\approx\s*N|p\s*=\s*N\s*-\s*1|p\s*=\s*N/2|N/2\s*capas")

# (Editorial 1) Siglas que deben definirse en el primer uso (sigla -> expansión esperada).
ACRONYMS: dict[str, str] = {
    "VQE": "Variational Quantum Eigensolver",
    "HVA": "Hamiltonian Variational Ansatz",
    "GNN": "Graph Neural Network",
    "MPNN": "Message-Passing Neural Network",
    "GIN": "Graph Isomorphism Network",
    "TFIM": "Transverse Field Ising Model",
    "MPS": "Matrix Product State",
    "DMRG": "Density Matrix Renormalization Group",
    "NISQ": "Noisy Intermediate-Scale Quantum",
    "PCA": "Principal Component Analysis",
}

# Expansiones alternativas en español aceptadas como definición válida de la sigla.
# El documento está en español: una sigla puede definirse con su traducción
# (p. ej. "estados de producto matricial (MPS)") y sigue siendo una definición
# legítima en el primer uso. Se comprueba una palabra clave discriminante por sigla.
ACRONYM_ES_KEYWORDS: dict[str, tuple[str, ...]] = {
    "VQE": ("resolvedor", "propio variacional"),
    "HVA": ("ansatz variacional hamiltoniano", "variacional hamiltoniano"),
    "GNN": ("red neuronal de grafo", "redes neuronales de grafo"),
    "MPNN": ("paso de mensajes",),
    "GIN": ("isomorfismo de grafo",),
    "TFIM": ("campo transversal", "modelo de ising"),
    "MPS": ("producto matricial",),
    "DMRG": ("matriz densidad", "renormalización de la matriz"),
    "NISQ": ("ruidosa de escala intermedia", "escala intermedia"),
    "PCA": ("componentes principales",),
}

# (Editorial 2) Grupos de variantes que deben unificarse (steering §6/§11).
TERM_VARIANTS: list[tuple[str, list[str]]] = [
    ("heavy-hex", [r"heavy-hex", r"Heavy-Hex", r"heavy hex", r"Heavy Hex"]),
    ("coste", [r"\bcoste\b", r"\bcosto\b"]),
    ("gap espectral", [r"gap espectral", r"brecha espectral"]),
    # "cadena unidimensional" NO es variante a unificar: es descripción física
    # legítima del modelo (p. ej. "cadena unidimensional de N espines"), distinta
    # del nombre de la topología "cadena 1D". Solo se marca "chain_1d" (grafía de código).
    ("cadena 1D", [r"cadena 1D", r"chain_1d"]),
    ("warm-start", [r"warm-start", r"warm start"]),
]

# (Editorial 4) Rango numérico con guion simple (debería usar -- en LaTeX).
BAD_RANGE_RE = re.compile(r"(?<![-\d])\d+(?:[.,]\d+)?-\d+(?:[.,]\d+)?(?![-\d])")
# (Editorial 3) Porcentaje sin caso absoluto (n/m) cercano.
PCT_RE = re.compile(r"\d{1,3}\\%")
ABS_CASE_RE = re.compile(r"\(\s*\d+\s*/\s*\d+\s*\)")
# Reproducibilidad: "exacto/a" junto a DMRG/MPS, y N grande + statevector.
EXACT_TN_RE = re.compile(r"\b(exact[oa]s?)\b[^.]*\b(DMRG|MPS)\b|\b(DMRG|MPS)\b[^.]*\b(exact[oa]s?)\b", re.IGNORECASE)
# Usos legítimos de "exacto" que NO deben marcarse: "diagonalización exacta" y su
# abreviatura "diag. exacta" (método clásico estándar, exact diagonalization),
# "simulación exacta" (statevector sin truncamiento), "MPS exacto con chi=1"
# (estado producto representado sin aproximación), y "estado fundamental exacto"
# (el objetivo verdadero, no un resultado aproximado).
EXACT_OK_RE = re.compile(
    r"diagonalizaci[oó]n\s+exact|diag\.?\\?\s*exact|simulaci[oó]n\s+exact"
    r"|MPS\s+exact[oa]\s+con\s+\$?\\?chi\s*=\s*1"
    r"|estado\s+fundamental\s+\\?e?m?p?h?\{?\s*exact"  # admite \emph{ intercalado
    r"|evaluaci[oó]n\s+exact"  # "evaluación exacta del gradiente" (no es DMRG/MPS)
    # "vector de estado exacto": el statevector a N<=22 SÍ es exacto (no DMRG/MPS).
    r"|vector\s+de\s+estado\s+exact"
    # "datos de referencia exactos": referencia contra la que validar; una frase que
    # dice "sin datos de referencia exactos" o "el DMRG deja de ser referencia" NO
    # llama exacto a DMRG/MPS, solo describe la disponibilidad de la referencia.
    r"|datos\s+de\s+referencia\s+exact|referencia\s+exact",
    re.IGNORECASE,
)
SV_BIGN_RE = re.compile(r"statevector", re.IGNORECASE)
BIGN_RE = re.compile(r"N\s*[=>]\s*(\d+)")
# (C) Referencia "exacta" (diagonalización / vector de estado exacto) aplicada a un
# N>22: físicamente inviable (2^22 ≈ 4M amplitudes ya es el límite práctico). Para
# N>22 solo hay DMRG/MPS, nunca diagonalización exacta ni vector de estado. Detecta
# la coincidencia de un marcador de exactitud con un N>22 en la misma cláusula.
EXACT_METHOD_RE = re.compile(
    r"diagonalizaci[oó]n\s+exact|vector\s+de\s+estado\s+exact|"
    r"diag\.?\s*exact|estado\s+exacto\s+por\s+vector",
    re.IGNORECASE,
)

# Sustituciones de tono DETERMINISTAS (frase fija -> reemplazo seguro), steering §6.
# Solo entran aquí las que no dependen del contexto (no rompen el sentido).
TONE_FIXES: list[tuple[str, str]] = [
    ("el pipeline funciona", "el pipeline satisface los criterios establecidos"),
    ("coste cuántico cero", "coste de optimización nulo en inferencia"),
    ("ampliamente validado", "validado en el régimen operativo evaluado"),
    ("ampliamente validada", "validada en el régimen operativo evaluado"),
    ("límite fundamental", "límite observado en las configuraciones evaluadas"),
    ("resultado exhaustivo", "resultado sistemático en las configuraciones evaluadas"),
    ("madurez del procedimiento", "consistencia del procedimiento"),
]

# Anglicismos (steering §3). Un ÚNICO término español por concepto.
# ANGLICISM_FIXES: reemplazo unívoco en prosa (auto-corregible con --fix-anglicisms).
# La clave es un patrón regex (con \b para límites de palabra); el valor es el
# reemplazo español canónico. Se aplican SOLO en prosa: se excluyen \texttt{},
# math $...$, comandos protegidos, comentarios, el entorno abstract y la
# bibliografía (títulos en inglés legítimos). Orden: los más específicos primero
# (p.ej. "ground truth" antes que "ground").
# SEGUROS: reemplazo que no cambia la concordancia con un artículo/adjetivo
# precedente (van sin artículo pegado, o el reemplazo mantiene género/número).
# Opción A del usuario: el auto-fix solo toca lo determinista y sin riesgo gramatical.
ANGLICISM_FIXES: list[tuple[str, str]] = [
    (r"\bnoiseless\b", "simulación ideal"),
    (r"zero-shot", "sin reentrenamiento"),
    (r"cross-topology", "entre topologías"),
    (r"cross-N", "entre tamaños"),
    (r"cross-seed", "entre semillas"),
    (r"multi-seed", "con múltiples semillas"),
    (r"multi-semilla", "con múltiples semillas"),
    (r"random initialization", "inicialización aleatoria"),
    (r"end-to-end", "de extremo a extremo"),
    (r"machine learning", "aprendizaje automático"),
    (r"\bMachine Learning\b", "Aprendizaje automático"),
    (r"message passing", "paso de mensajes"),
    (r"forward pass", "inferencia"),
    (r"pasada forward", "inferencia"),
    (r"early stopping", "parada temprana"),
    (r"early-stopping", "parada temprana"),
    (r"post-hoc", "a posteriori"),
    (r"\bepochs\b", "épocas"),
    (r"\brestarts\b", "reinicios"),
    (r"\bnegligible\b", "despreciable"),
    (r"sweet spot", "mejor compromiso"),
    (r"machine epsilon", "precisión de máquina"),
    (r"smoke test", "prueba de humo"),
]

# ANGLICISM_DETECT: términos que se DETECTAN pero NO se auto-corrigen porque:
#  - el reemplazo cambia género/número y rompería la concordancia con un artículo
#    o adjetivo vecino (ground truth, dataset, framework, area law, ...), o
#  - el reemplazo depende del contexto (deploy, benchmark, run, input/output), o
#  - el término se conserva por convención (bond-resolved), definido 1 vez en cursiva.
# Se marcan como %TODO-ANGLICISMO para corrección manual con criterio.
# (patrón, nota).
ANGLICISM_DETECT: list[tuple[str, str]] = [
    (
        r"ground truth",
        "ground truth -> datos de referencia (ajustar artículo: 'los datos', no 'el')",
    ),
    (r"ground state", "ground state -> estado fundamental"),
    (r"\bdatasets\b", "datasets -> conjuntos de datos"),
    (r"\bdataset\b", "dataset -> conjunto de datos (ajustar artículo: 'el conjunto')"),
    (r"\bframework\b", "framework -> marco (ajustar artículo/adjetivo al género masculino)"),
    (r"generalization gap", "generalization gap -> brecha de generalización (fem.: 'la brecha')"),
    (r"area law", "area law -> ley de área (fem.: 'la ley'); definir en cursiva 1 vez"),
    (r"hidden dimension", "hidden dimension -> dimensión oculta"),
    (r"learning rate", "learning rate -> tasa de aprendizaje"),
    (r"edge feature", "edge feature -> atributo de arista"),
    (r"node feature", "node feature -> atributo de nodo"),
    (r"global mean pool", "global mean pool -> agregación global por media"),
    (r"\bpooling\b", "pooling -> agregación"),
    (r"\boverfit(ting)?\b", "overfit/overfitting -> sobreajuste"),
    (
        r"conocimiento accionable",
        "conocimiento accionable -> información útil para delimitar el dominio",
    ),
    (r"\bactionable\b", "actionable -> útil (parafrasear)"),
    (r"\bdeploy\b", "deploy -> despliegue / evaluación en inferencia (según contexto)"),
    (r"\bbenchmark\b", "benchmark -> prueba comparativa / banco de pruebas (según contexto)"),
    (r"\brun(s)?\b", "run/runs -> ejecución/ejecuciones (si es prosa, no nombre de comando)"),
    (r"\blandscape\b", "landscape -> paisaje de optimización"),
    (r"\binput\b", "input -> entrada"),
    (r"\boutput\b", "output -> salida"),
    (r"\bgrid\b", "grid -> malla / rejilla"),
    (
        r"bond-resolved",
        "conservable como 'parametrización por enlace (bond-resolved)'; definir 1 vez",
    ),
]

# Términos que se conservan (asentados) — no marcar. Definir en cursiva 1ª vez.
ANGLICISM_KEEP = ("pipeline", "warm-start", "ansatz", "qubit", "gap")

# Guiones tipográficos: el usuario pide no usar incisos con guiones ('-- --').
# Detecta em-dash de LaTeX '---' usado como inciso (separador visual). Es criterio
# humano decidir reemplazar por comas/paréntesis, así que solo se DETECTA.
LATEX_EMDASH_RE = re.compile(r"---")

# Errores de gramática/registro (steering §4). (patrón, nota, auto_fix|None).
# Cuando auto_fix no es None, el reemplazo es determinista y seguro.
GRAMMAR_FIXES: list[tuple[str, str]] = [
    (r"\ben base a\b", "a partir de"),
    (r"\ben función a\b", "en función de"),
]
# Detección de "a N=..." donde debería ser "para N=..." (steering §4). Se marca
# (no auto-fix: "a $N$" puede ser legítimo en otros contextos matemáticos).
A_N_RE = re.compile(r"\ba\s+\$?N\$?\s*=")
# Muletillas encadenadas a moderar (steering §4): "Se observa que", "Esto confirma".
FILLER_RE = re.compile(r"Se observa que|Esto confirma que|El resultado confirma que")
# "Se puede observar como" sin tilde (debería ser "cómo") — steering §4.
COMO_RE = re.compile(r"observar\s+como\b|ver\s+como\b")


# ═══════════════════════════════════════════════════════════════════════════════
# Clasificación de hallazgos (severidad / categoría / accionabilidad)
# ═══════════════════════════════════════════════════════════════════════════════
#
# La detección produce cadenas planas. Para que humanos y agentes prioricen sin
# releer las 90+ líneas, cada hallazgo se clasifica por su TEXTO (patrones ya
# estables en los mensajes que emite _check_tex/_check_editorial) en:
#
#   severidad   BLOQUEANTE  rompe la compilación o la credibilidad del dato
#               IMPORTANTE  incoherencia de contenido que el lector notará
#               EDITORIAL   estilo/idioma (anglicismo, guion, tono)
#               INFO        aviso no accionable directamente
#   categoria   etiqueta corta para agrupar dentro de la severidad
#   accion      AUTO-FIX    hay bandera del script que lo corrige
#               MANUAL      requiere criterio humano
#               VERIFICAR   probable falso positivo ya revisado; confirmar y suprimir
#
# Cada regla es (patrón_regex_sobre_el_mensaje, severidad, categoría, acción, pista).
# El orden importa: la primera que casa gana. Añadir reglas nuevas ARRIBA de la
# genérica final.

FindingClass = tuple[str, str, str, str]  # (severidad, categoria, accion, pista)

_CLASSIFY_RULES: list[tuple[re.Pattern[str], FindingClass]] = [
    # ---- BLOQUEANTE: fallo interno del propio validador ----
    (
        re.compile(r"chequeo '.*' falló con"),
        (
            "BLOQUEANTE",
            "validador-bug",
            "MANUAL",
            "Un chequeo lanzó una excepción; revisar el script (no es un problema del .tex).",
        ),
    ),
    # ---- BLOQUEANTE: sintaxis LaTeX / integridad estructural ----
    (
        re.compile(r"delimitador \$|math mode"),
        ("BLOQUEANTE", "latex-math", "MANUAL", "Balancear los $...$ de la línea."),
    ),
    (
        re.compile(r"\\end\{.*sin \\begin|entorno huérfano|entornos cruzados|\\begin.*sin \\end"),
        ("BLOQUEANTE", "latex-entorno", "MANUAL", "Revisar apertura/cierre de entornos."),
    ),
    (re.compile(r"llave|\{.*sin cerrar|desbalance"), ("BLOQUEANTE", "latex-llaves", "MANUAL", "Balancear llaves { }.")),
    # \ref a tablas auto_* cuyo \label vive dentro de \input{tables/auto_*}: el
    # validador no lee dentro del \input, así que es falso positivo conocido.
    (
        re.compile(r"\\ref\{tab:auto_.*sin \\label"),
        (
            "INFO",
            "ref-auto-input",
            "VERIFICAR",
            "El \\label vive dentro de \\input{tables/auto_*}; falso positivo conocido.",
        ),
    ),
    (
        re.compile(r"\\ref sin \\label|referencia rota|\?\?"),
        ("BLOQUEANTE", "ref-rota", "MANUAL", "Referencia sin destino: saldrá como ?? en el PDF."),
    ),
    # ---- IMPORTANTE: credibilidad del dato ----
    (
        re.compile(r"statevector.*N|N.*statevector"),
        ("IMPORTANTE", "backend", "MANUAL", "N>22 debe usar MPS, no statevector (steering §8)."),
    ),
    (
        re.compile(r"exacto.*DMRG|DMRG.*exacto|convergido|referencia exacta.*N=\d+"),
        ("IMPORTANTE", "repro", "MANUAL", "No usar 'exacto' para DMRG/MPS sin convergencia en χ, ni exacto a N>22."),
    ),
    (
        re.compile(r"caso absoluto inconsistente|= .*\\%"),
        ("IMPORTANTE", "cifra-aritmetica", "MANUAL", "El porcentaje no cuadra con n/m; corregir uno."),
    ),
    (
        re.compile(r"porcentajes distintos|aparece con porcentajes"),
        (
            "IMPORTANTE",
            "cifra-coherencia",
            "VERIFICAR",
            "Cada % puede ser una config distinta; confirmar contra fuente de verdad.",
        ),
    ),
    (
        re.compile(r"campañas de distintos meses|no se mezclen"),
        ("IMPORTANTE", "cifra-epoca", "MANUAL", "Verificar que no se mezclen épocas de campaña."),
    ),
    (
        re.compile(r"enmascaramiento por gap"),
        ("IMPORTANTE", "metrica-gap", "MANUAL", "Revisar |ΔE| absoluto, no solo ΔE/gap."),
    ),
    (
        re.compile(r"tasa de aprobación.*sin caso absoluto|%TODO-CIFRA tasa"),
        ("IMPORTANTE", "cifra-passrate", "MANUAL", "Añadir caso absoluto: '95\\% (37/39)'."),
    ),
    (
        re.compile(r"confiable/robusto|r² <"),
        ("IMPORTANTE", "metrica-correlacion", "MANUAL", "Matizar correlación con r² bajo."),
    ),
    (
        re.compile(r"sigla '.*' usada sin definir"),
        ("IMPORTANTE", "sigla", "MANUAL", "Definir la sigla en su primer uso (ES o EN)."),
    ),
    (
        re.compile(r"No existe|No se pudo|Error parseando|No se pudo cargar"),
        ("IMPORTANTE", "datos-fuente", "MANUAL", "Falta un artefacto de datos; regenerarlo."),
    ),
    # ---- EDITORIAL: estilo / idioma ----
    # Frases fijas de tono comercial que --fix-tone SÍ corrige de forma determinista.
    (
        re.compile(
            r"el pipeline funciona|coste cuántico cero|ampliamente validado|"
            r"límite fundamental|resultado exhaustivo|madurez del procedimiento"
        ),
        ("EDITORIAL", "tono-fijo", "AUTO-FIX", "Corregible con --fix-tone (frase fija determinista)."),
    ),
    # Tono que requiere criterio (adjetivos, 'demuestra'/'garantiza'): NO lo toca --fix-tone.
    (
        re.compile(r"%TODO-TONO|tono a moderar|\bdemuestra|\bgarantiza"),
        (
            "EDITORIAL",
            "tono",
            "MANUAL",
            "Requiere criterio: a veces el término está justificado; --fix-tone no lo toca.",
        ),
    ),
    (
        re.compile(r"%TODO-ANGLICISMO|anglicismo|palabra inglesa"),
        (
            "EDITORIAL",
            "anglicismo",
            "MANUAL",
            "Reemplazo unívoco con --fix-anglicisms; los ambiguos (género/artículo), a mano.",
        ),
    ),
    (
        re.compile(r"guion simple|em-dash|em/en-dash|---"),
        ("EDITORIAL", "guion", "MANUAL", "Decidir coma/paréntesis/-- según el caso."),
    ),
    (
        re.compile(r"coma decimal en modo matemático"),
        ("EDITORIAL", "icomma", "AUTO-FIX", "Ya resuelto con \\usepackage{icomma}."),
    ),
    (re.compile(r"decimal con punto"), ("EDITORIAL", "decimal-punto", "MANUAL", "Usar coma decimal (steering §7).")),
    (
        re.compile(r"\\caption sin título corto|%TODO-INDICE"),
        ("EDITORIAL", "caption-indice", "MANUAL", "Usar \\caption[breve]{largo}."),
    ),
    (
        re.compile(r"%TODO-PROFUNDIDAD|capas requeridas"),
        ("EDITORIAL", "profundidad", "MANUAL", "Quitar énfasis en p∝N (steering §5)."),
    ),
    (
        re.compile(r"Grade->|Pass->|Rate->|Grade/Pass/Rate"),
        ("EDITORIAL", "anglicismo-metrica", "MANUAL", "Traducir Grade/Pass/Rate."),
    ),
    # ---- IMPORTANTE (métrica) que caía en 'otros' ----
    (
        re.compile(r"%TODO-METRICA|PassRate sin"),
        (
            "IMPORTANTE",
            "metrica-passrate-de",
            "MANUAL",
            "Acompañar todo PassRate con la métrica primaria |ΔE| (steering §5).",
        ),
    ),
    (
        re.compile(r"rangos de .*distintos entre|puntos de entrenamiento.*distintos"),
        (
            "IMPORTANTE",
            "cifra-coherencia",
            "MANUAL",
            "Unificar el rango a su valor canónico entre capítulos (steering §1).",
        ),
    ),
    (
        re.compile(
            r"símbolo 'S' usado para la aceleración|"
            r"mezcla la razón de error|aceleración antiguo"
        ),
        (
            "IMPORTANTE",
            "aceleracion-vs-error",
            "MANUAL",
            "Distinguir A (evaluaciones) de R(N) (precisión); símbolo A, no S (steering §5/§11).",
        ),
    ),
    # ---- EDITORIAL que caía en 'otros' ----
    (
        re.compile(r"aparece \(L\d+\) antes de su primera mención"),
        ("EDITORIAL", "orden-label", "MANUAL", "Mover la primera \\ref antes del objeto, o reordenar (cosmético)."),
    ),
    (
        re.compile(r"%TODO-ESTILO|muletilla"),
        (
            "EDITORIAL",
            "estilo-muletilla",
            "MANUAL",
            "Variar la redacción; reservar 'confirma' para evidencia concluyente (§4).",
        ),
    ),
    # ---- PLANTILLA (§4): palabras clave, índice de acrónimos, fuente en captions ----
    (
        re.compile(r"palabra clave|palabras clave en"),
        (
            "IMPORTANTE",
            "plantilla-keywords",
            "MANUAL",
            "Ajustar las palabras clave (4–6 términos, sin siglas, minúscula, §4.2).",
        ),
    ),
    (
        re.compile(r"Índice de acrónimos|índice de acrónimos|índice inflado"),
        (
            "IMPORTANTE",
            "plantilla-acronimos",
            "MANUAL",
            "Sincronizar el índice de acrónimos con el uso real en el cuerpo (§4.3).",
        ),
    ),
    (
        re.compile(r"caption sin 'Fuente:'|fuente '.*' cita un objeto"),
        (
            "IMPORTANTE",
            "plantilla-fuente",
            "MANUAL",
            "Añadir/normalizar 'Fuente:' (elaboración propia o cita, no clase de código, §4.1).",
        ),
    ),
    (
        re.compile(r"%TODO-NOTACION|símbolo Δ se usa|conviven 'θ"),
        ("IMPORTANTE", "notacion-colision", "MANUAL", "Desambiguar el símbolo con doble significado (informe §3)."),
    ),
    (
        re.compile(r"%TODO-BIBLIO|identificador '.*arXiv.*' citado|citar por autor-año"),
        ("IMPORTANTE", "biblio", "MANUAL", "Citar por autor-año con \\citep/\\citet, no por id de arXiv (informe §6)."),
    ),
    (
        re.compile(r"100\\% de aprobación para N=30|afirmación.*100.*aparece"),
        (
            "IMPORTANTE",
            "claim-sin-tabla",
            "MANUAL",
            "Anclar la afirmación del 100% a una tabla con datos (informe §1.1).",
        ),
    ),
    (
        re.compile(r"constantes aditivas sin simplificar|se describe como 'lineal'"),
        (
            "IMPORTANTE",
            "ecuacion",
            "MANUAL",
            "Simplificar constantes / corregir 'lineal' vs superlineal (informe §1.2).",
        ),
    ),
    (
        re.compile(r"producto inline no cuadra"),
        (
            "IMPORTANTE",
            "cifra-aritmetica",
            "MANUAL",
            "El producto declarado no coincide con los factores; dar el desglose real (§5).",
        ),
    ),
    # ---- BIBLIOGRAFÍA (§9 / informe §6) ----
    (
        re.compile(r"BIBLIO"),
        ("IMPORTANTE", "biblio", "MANUAL", "Corregir la entrada bibliográfica (localización, ancla, preprint, §9)."),
    ),
    # ---- INFO real ----
    (
        re.compile(r"^INFO:|hipótesis .*se mencionan"),
        ("INFO", "revision-manual", "MANUAL", "Aviso informativo; revisar a mano."),
    ),
    # ---- genérica final ----
    (re.compile(r".*"), ("INFO", "otros", "MANUAL", "Revisar manualmente.")),
]

# Orden de presentación de las severidades (de más urgente a menos).
_SEVERITY_ORDER = ("BLOQUEANTE", "IMPORTANTE", "EDITORIAL", "INFO")


def classify_finding(message: str) -> FindingClass:
    """Clasifica un mensaje de inconsistencia por su texto.

    Devuelve (severidad, categoría, acción, pista) según la primera regla de
    ``_CLASSIFY_RULES`` que casa. La última regla siempre casa (genérica).
    """
    for pattern, cls in _CLASSIFY_RULES:
        if pattern.search(message):
            return cls
    return ("INFO", "otros", "MANUAL", "Revisar manualmente.")


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
            lines.append("")
            return "\n".join(lines)

        # Clasificar cada hallazgo y agrupar por severidad -> categoría.
        classified = [(msg, classify_finding(msg)) for msg in self.inconsistencies]

        # Resumen priorizado (arriba del todo, para decidir por dónde empezar).
        by_sev: dict[str, int] = defaultdict(int)
        by_action: dict[str, int] = defaultdict(int)
        for _msg, (sev, _cat, action, _hint) in classified:
            by_sev[sev] += 1
            by_action[action] += 1

        lines.append("RESUMEN POR PRIORIDAD")
        for sev in _SEVERITY_ORDER:
            if by_sev.get(sev):
                lines.append(f"  {sev:<11} {by_sev[sev]}")
        lines.append("")
        lines.append("RESUMEN POR ACCIÓN")
        for action in ("AUTO-FIX", "MANUAL", "VERIFICAR"):
            if by_action.get(action):
                hint = {
                    "AUTO-FIX": "corregibles con banderas del script (--fix-tone / --fix-anglicisms)",
                    "MANUAL": "requieren criterio humano",
                    "VERIFICAR": "probables falsos positivos ya revisados; confirmar y suprimir",
                }[action]
                lines.append(f"  {action:<10} {by_action[action]:>3}  — {hint}")
        lines.append("")
        lines.append("=" * 72)
        lines.append("")

        # Detalle agrupado: severidad -> categoría -> hallazgos.
        # Índice global estable para poder referirse a un hallazgo por número.
        idx = 0
        for sev in _SEVERITY_ORDER:
            sev_items = [(m, c) for (m, c) in classified if c[0] == sev]
            if not sev_items:
                continue
            lines.append(f"### {sev}  ({len(sev_items)})")
            # agrupar por categoría dentro de la severidad
            cats: dict[str, list[tuple[str, FindingClass]]] = defaultdict(list)
            for m, c in sev_items:
                cats[c[1]].append((m, c))
            for cat in sorted(cats):
                cat_items = cats[cat]
                # la pista y la acción son homogéneas por categoría: tomarlas del 1º
                _, (_, _, action, hint) = cat_items[0]
                lines.append(f"  [{cat}]  ({len(cat_items)})  ·  {action}  ·  {hint}")
                for m, _c in cat_items:
                    idx += 1
                    lines.append(f"    {idx:>3}. {m}")
                lines.append("")
            lines.append("")

        return "\n".join(lines)

    def render_checklist(self) -> str:
        """Genera una checklist Markdown accionable de chequeos pendientes.

        A diferencia de ``render_txt`` (informe completo de detección), esta salida
        es una lista de tareas marcable (``- [ ]``), ordenada por acción para que
        humanos y agentes ataquen primero lo auto-arreglable, luego lo manual y por
        último confirmen los probables falsos positivos. Se agrupa por acción y,
        dentro, por categoría, con el comando sugerido cuando lo hay.
        """
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        classified = [(msg, classify_finding(msg)) for msg in self.inconsistencies]

        # Comando sugerido por categoría (si una bandera del script lo aborda,
        # total o parcialmente). Se muestra en la checklist junto a la categoría.
        cmd_by_cat = {
            "tono-fijo": "--fix-tone internal/tesis-v4.0.tex",
            "anglicismo": "--fix-anglicisms internal/tesis-v4.0.tex  "
            "# solo reemplazos unívocos; los de género/artículo, a mano",
        }

        # Buckets por acción, preservando severidad para ordenar dentro.
        order = {"AUTO-FIX": 0, "MANUAL": 1, "VERIFICAR": 2}
        buckets: dict[str, list[tuple[str, FindingClass]]] = defaultdict(list)
        for m, c in classified:
            buckets[c[2]].append((m, c))

        n_total = len(classified)
        n_auto = len(buckets.get("AUTO-FIX", []))
        n_manual = len(buckets.get("MANUAL", []))
        n_verify = len(buckets.get("VERIFICAR", []))

        out: list[str] = [
            "# Chequeos pendientes de la tesis",
            "",
            f"Generado: {now}  ",
            "Fuente: `generate_thesis_tables.py --check-tex`  ",
            f"Total pendientes: **{n_total}** "
            f"(auto-arreglables {n_auto} · manuales {n_manual} · a verificar {n_verify})",
            "",
            "Orden recomendado: primero **AUTO-FIX** (banderas del script), luego "
            "**MANUAL** (criterio humano), por último **VERIFICAR** (confirmar que "
            "son falsos positivos y suprimirlos).",
            "",
        ]

        titles = {
            "AUTO-FIX": "Auto-arreglables (aplicar bandera del script)",
            "MANUAL": "Requieren criterio humano",
            "VERIFICAR": "Probables falsos positivos (confirmar y suprimir)",
        }

        for action in sorted(buckets, key=lambda a: order.get(a, 9)):
            items = buckets[action]
            out.append(f"## {titles.get(action, action)}  ({len(items)})")
            out.append("")
            # agrupar por categoría, ordenando categorías por severidad
            cats: dict[str, list[tuple[str, FindingClass]]] = defaultdict(list)
            for m, c in items:
                cats[c[1]].append((m, c))

            def _cat_sev(cat: str) -> int:
                sev = cats[cat][0][1][0]
                return _SEVERITY_ORDER.index(sev) if sev in _SEVERITY_ORDER else 9

            for cat in sorted(cats, key=_cat_sev):
                cat_items = cats[cat]
                sev = cat_items[0][1][0]
                hint = cat_items[0][1][3]
                out.append(f"### `{cat}` · {sev} · {len(cat_items)}")
                if cat in cmd_by_cat:
                    out.append("")
                    out.append(
                        "```bash\n.venv/bin/python "
                        "scripts/general_project_maintenance/generate_thesis_tables.py "
                        f"{cmd_by_cat[cat]}\n```"
                    )
                out.append(f"_{hint}_")
                out.append("")
                for m, _c in cat_items:
                    # recortar el prefijo "%TODO-XXX " ruidoso para la checklist
                    text = re.sub(r"^%TODO-\w+\s*", "", m.strip())
                    out.append(f"- [ ] {text}")
                out.append("")

        return "\n".join(out)


# ═══════════════════════════════════════════════════════════════════════════════
# Utilidades de formato LaTeX
# ═══════════════════════════════════════════════════════════════════════════════


_PROTECT_CMD_RE = re.compile(
    r"\\(?:texttt|url|href|ref|eqref|autoref|cite[tp]?|label|input|includegraphics)"
    r"\{[^}]*\}"
)
_MATH_INLINE_RE = re.compile(r"(?<!\\)\$[^$]*\$")


def _strip_protected(code: str) -> str:
    """Reemplaza por espacios las zonas que no son prosa (código, math, comandos).

    Se usa para buscar anglicismos y guiones solo en prosa, evitando falsos
    positivos dentro de \\texttt{...}, $...$, \\ref{...}, etc. Conserva longitud
    aproximada reemplazando por espacios (no altera índices de forma crítica).
    """
    out = _PROTECT_CMD_RE.sub(lambda m: " " * len(m.group(0)), code)
    out = _MATH_INLINE_RE.sub(lambda m: " " * len(m.group(0)), out)
    return out


# Comentario LaTeX de línea (respeta el porcentaje escapado \%). Compilado una vez.
_COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)


def _strip_comments(text: str) -> str:
    """Quita los comentarios LaTeX de línea (todo lo que sigue a un % no escapado).

    Preserva la posición de las líneas (usa MULTILINE, no borra los saltos), de modo
    que ``_line_number_at`` sigue devolviendo el número de línea correcto. Reemplaza
    las ~18 copias sueltas de ``re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)``.
    """
    return _COMMENT_RE.sub("", text)


def _prose_text(text: str) -> str:
    """Texto listo para buscar términos en PROSA: sin comentarios y con las zonas de
    código/matemáticas/comandos neutralizadas (\\texttt, $...$, \\ref, etc.).

    Combina ``_strip_comments`` + ``_strip_protected`` en un solo paso, que es el
    preprocesado que repetían muchos chequeos de estilo/anglicismos. Conserva la
    longitud (índices/líneas estables).
    """
    return _strip_protected(_strip_comments(text))


def _line_number_at(text: str, pos: int) -> int:
    """Número de línea (1-indexado) de la posición ``pos`` en ``text``.

    Reemplaza el patrón repetido ``text.count("\\n", 0, m.start()) + 1``.
    """
    return text.count("\n", 0, pos) + 1


def _rel(path: Path) -> str:
    """Ruta relativa a ROOT para imprimir, robusta a rutas fuera de ROOT o
    relativas. Antes ``path.relative_to(ROOT)`` lanzaba ValueError cuando el
    ``--out-dir`` se pasaba como ruta relativa (p. ej. ``internal/tesis/tables``)
    o apuntaba fuera del repo. Ahora se resuelve a absoluta primero y, si aún así
    no cuelga de ROOT, se devuelve la ruta tal cual.
    """
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except (ValueError, OSError):
        return str(path)


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
    """Nombre de topología en prosa (minúscula, como se usa a media frase)."""
    return TOPO_ES.get(topo, topo.replace("_", " "))


def _topo_es_cell(topo: str) -> str:
    """Nombre de topología para CELDA de tabla: primera letra en mayúscula, para
    unificar con las tablas manuales del cuerpo ('Cadena 1D', 'Escalera',
    'Cuadrada', 'Triangular', 'Heavy-hex'). La prosa usa la forma en minúscula
    (_topo_es); las celdas de tabla, esta. Evita la doble grafía que señaló el
    corrector (§5)."""
    name = _topo_es(topo)
    return name[:1].upper() + name[1:] if name else name


def _wrap_table(
    caption: str,
    label: str,
    col_spec: str,
    header: list[str],
    rows: list[list[str]],
    notes: str = "",
    pre_lines: list[str] | None = None,
    short_caption: str = "",
) -> list[str]:
    """Envuelve una tabla en el entorno LaTeX estándar con booktabs.

    ``short_caption`` (opcional) produce \\caption[corto]{largo} para que el
    índice de tablas muestre solo el título breve (steering §7 / pedido usuario).
    """
    lines: list[str] = []
    lines.append("% ==== AUTO-GENERADA — no editar a mano (regenerar con generate_thesis_tables.py) ====")
    if pre_lines:
        lines.extend(pre_lines)
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    if short_caption:
        lines.append(f"\\caption[{short_caption}]{{{caption}}}")
    else:
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
            f"No existe {_rel(SCOREBOARD_JSON)}: correr generate_best_results_scoreboard.py --json primero."
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
    # Restringir al régimen operativo válido (h >= h_frontier(N)): las tablas de
    # interpolación/predicción reflejan el rendimiento donde el HVA es expresivo,
    # coherente con la narrativa. Los puntos h < h_frontier (región crítica, donde
    # el ansatz no representa el estado fundamental) se excluyen por construcción.
    frontier = load_h_frontier(FOCUS_TOPOLOGY, p_layers=1)
    if not frontier:
        collector.add_inconsistency(
            f"No se pudo leer h_frontier de {FOCUS_TOPOLOGY} desde el dashboard; las "
            "tablas por-h incluirán todo el rango de h (incluida la región crítica)."
        )
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
                    # (1) Régimen válido: descartar h por debajo de la frontera.
                    h_min = frontier.get(entry.n_qubits)
                    if h_min is not None and r.h < h_min - 0.005:
                        continue
                    # (2) Grilla canónica de extrapolación: quedarse solo con los h
                    # que caen en la grilla común (consistencia intra-régimen).
                    node = snap_to_grid(r.h, H_GRID_EXTRAPOLATION)
                    if node is None:
                        continue
                    prev = per_n[entry.n_qubits].get(node)
                    # Quedarse con el mejor |ΔE| por (N, nodo de grilla)
                    if prev is None or r.abs_error < prev.abs_error:
                        per_n[entry.n_qubits][node] = r
    finally:
        sb.H_TOLERANCE = saved_tol

    # Chequeo de cobertura: reportar N que no cubren toda la grilla (dentro del
    # régimen válido). No se fixea; es informativo para trazabilidad.
    for n, hmap in sorted(per_n.items()):
        h_min = frontier.get(n)
        expected = [g for g in H_GRID_EXTRAPOLATION if h_min is None or g >= h_min - 0.005]
        missing = [g for g in expected if g not in hmap]
        if missing:
            collector.add_inconsistency(
                f"{FOCUS_TOPOLOGY} N={n}: la grilla canónica de extrapolación "
                f"{H_GRID_EXTRAPOLATION} no cubre {missing} en el régimen válido "
                f"(h_frontier={_num(h_min, 2) if h_min else 'N/A'}); tabla con menos "
                f"puntos para ese N (INFO, no error)."
            )

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


def _grade_from_abs_error(abs_error: float) -> str:
    """Calificación por |ΔE| (misma escala que generate_best_results_scoreboard).

    A: <0,05 | B: <0,10 | C: <0,30 | D: <1,00 | F: ≥1,00. Fuente única de umbrales
    para que la calificación de la media sea coherente con la del mejor punto.
    Devuelve la letra del pipeline (A/B/C/D/F); traducir a la escala de la tesis
    con _grade_es antes de renderizar en una tabla.
    """
    if abs_error < 0.05:
        return "A"
    if abs_error < 0.10:
        return "B"
    if abs_error < 0.30:
        return "C"
    if abs_error < 1.00:
        return "D"
    return "F"


def _grade_es(letter: str) -> str:
    """Traduce la letra del pipeline (A/B/C/D/F) a la escala de la tesis (A--E).

    La peor nota se escribe E en la tesis para no colisionar con el símbolo de
    fidelidad F. El pipeline de datos conserva F (JSON, zoo, eval reports); la
    traducción vive solo en la capa de presentación de la memoria.
    """
    return "E" if letter.strip() == "F" else letter


def gen_scoreboard(scoreboard: dict, collector: TodoCollector) -> list[str]:
    """Tabla resumen por topología: calificación del mejor punto y de la media.

    Distingue explícitamente dos agregaciones que antes convivían sin rótulo (y que
    parecían contradecirse): la calificación del *mejor punto* (|ΔE| mínimo) y el
    |ΔE| *medio* con su propia calificación. Añade un chequeo que verifica que la
    calificación de la media concuerda con su |ΔE| según la escala declarada.
    """
    by_topo = scoreboard.get("best_by_topology", {})
    if not by_topo:
        return [collector.marker("DATOS", "best_by_topology vacío en el scoreboard JSON.")]

    header = [
        "Topología",
        "$N$ máx.",
        "Calif. mejor punto",
        "$|\\Delta E|$ medio",
        "Calif. media",
    ]
    rows: list[list[str]] = []
    for topo in sorted(by_topo):
        n_results = by_topo[topo]
        if not n_results:
            continue
        ns = sorted(int(k) for k in n_results)
        best = min(n_results.values(), key=lambda r: r["best_abs_error"])
        mean_abs = best.get("mean_abs_error", best["best_abs_error"])
        grade_mean = _grade_from_abs_error(mean_abs)
        # Chequeo (no fix): la calificación del mejor punto debe ser >= que la de la
        # media (mejor punto nunca peor que la media). Si el JSON trae una calif del
        # mejor punto que ni siquiera concuerda con best_abs_error, reportarlo.
        grade_best_expected = _grade_from_abs_error(best["best_abs_error"])
        if best.get("grade") and best["grade"] != grade_best_expected:
            collector.add_inconsistency(
                f"scoreboard {topo}: calif del mejor punto '{best['grade']}' no "
                f"concuerda con best_abs_error={best['best_abs_error']:.4f} "
                f"(esperada '{grade_best_expected}'). Revisar generador de scoreboard."
            )
        rows.append(
            [
                _esc(_topo_es_cell(topo)),
                str(max(ns)),
                _esc(_grade_es(best["grade"])),
                _num(mean_abs),
                _esc(_grade_es(grade_mean)),
            ]
        )

    caption = (
        "Resultado por topología en el punto de operación más exigente "
        f"($h \\approx {_num(scoreboard.get('target_h', 2.5), 2)}$). "
        "La columna \\emph{Calif.\\ mejor punto} califica el punto de menor "
        "$|\\Delta E|$ de la configuración; \\emph{$|\\Delta E|$ medio} promedia "
        "sobre todos los puntos evaluados y \\emph{Calif.\\ media} lo califica. Las "
        "dos calificaciones difieren porque agregan de forma distinta. La columna "
        "\\emph{$N$ máx.} es el mayor tamaño evaluado para esa topología en la "
        "campaña (no el límite del método, que sigue las tres escalas de la "
        "Tabla~\\ref{tab:comparison_literature}). Escala de calificación A--E por "
        "$|\\Delta E|$: A ($<0{,}05$), B ($<0{,}10$), C ($<0{,}30$), D ($<1{,}00$), "
        "E ($\\geq 1{,}00$); se usa E como peor nota para no confundirla con el "
        "símbolo de fidelidad $F$. Fuente: elaboración propia."
    )
    return _wrap_table(
        caption,
        "tab:auto_scoreboard",
        "lcccc",
        header,
        rows,
        short_caption="Resultado por topología",
    )


def gen_coverage(scoreboard: dict, collector: TodoCollector, coverage_ns: set[int] | None = None) -> list[str]:
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
        row = [_esc(_topo_es_cell(topo))]
        for n in all_ns:
            entry = n_results.get(str(n))
            row.append(_esc(_grade_es(entry["grade"])) if entry else "---")
        rows.append(row)

    col_spec = "l" + "c" * len(all_ns)
    caption = (
        "Matriz de cobertura: calificación alcanzada por topología y tamaño del sistema "
        f"$N$ en $h \\approx {_num(scoreboard.get('target_h', 2.5), 2)}$."
    )
    notes = "Un guion (---) indica que no hay resultado evaluado para esa combinación."
    return _wrap_table(
        caption,
        "tab:auto_coverage",
        col_spec,
        header,
        rows,
        notes=notes,
        short_caption="Matriz de cobertura por topología y tamaño",
    )


def gen_campaign(index: list[dict], collector: TodoCollector) -> list[str]:
    """Conteos de campaña por modelo, con la taxonomía de conteo del steering."""
    if not index:
        return [collector.marker("CAMPANA", "ResultIndex vacío: no se pudieron contar ejecuciones.")]

    # Filtrar entradas con modelo válido
    valid = [e for e in index if e.get("model")]
    by_model: dict[str, list[dict]] = defaultdict(list)
    for e in valid:
        by_model[e["model"]].append(e)

    # Excluir modelos fuera de la narrativa (XY). La exclusión es una decisión
    # firme y documentada (steering §16): no se lista en la tabla ni se reporta
    # como pendiente. Se deja constancia en un log informativo para trazabilidad.
    for excluded in sorted(CAMPAIGN_EXCLUDED_MODELS):
        if excluded in by_model:
            n_ex = len(by_model.pop(excluded))
            print(
                f"  ℹ️  campaña: modelo '{excluded}' ({n_ex} ejecuciones) excluido "
                f"de la tabla de conteo por decisión firme de la tesis (steering §16)."
            )

    header = ["Modelo", "Ejecuciones", "Tasa aprob."]
    rows: list[list[str]] = []
    total_runs = 0
    total_pass = 0
    for model in sorted(by_model):
        runs = by_model[model]
        n = len(runs)
        n_pass = sum(1 for r in runs if r.get("passed"))
        rate = n_pass / n if n else 0.0
        total_runs += n
        total_pass += n_pass
        # MODEL_ES es contenido LaTeX controlado (puede incluir math como
        # $J_1$-$J_2$): no pasa por _esc. El fallback (clave desconocida) sí se
        # escapa por seguridad.
        model_label = MODEL_ES.get(model)
        if model_label is None:
            model_label = _esc(model.replace("_", " "))
        rows.append(
            [
                model_label,
                str(n),
                f"{n_pass}/{n} ({_num(rate * 100, 1)}\\%)",
            ]
        )
    # Fila de total (cifra canónica de campaña, para reconciliar con el texto).
    rows.append(
        [
            "\\textbf{Total}",
            f"\\textbf{{{total_runs}}}",
            f"\\textbf{{{total_pass}/{total_runs} ({_num(total_pass / total_runs * 100, 1) if total_runs else 0}\\%)}}",
        ]
    )

    # Taxonomía Heisenberg: el conteo canónico se deriva del ResultIndex
    # (fuente de verdad), no de un número fijo. La tesis debe reflejar este total.
    n_heis = len(by_model.get("heisenberg", []))
    n_heis_t = len(by_model.get("heisenberg_transverse", []))
    total_heis = n_heis + n_heis_t
    heis_note = f"Heisenberg: {n_heis} XXZ + {n_heis_t} transversal = {total_heis} ejecuciones. " if total_heis else ""

    caption = (
        "Conteo de ejecuciones del pipeline por modelo en la campaña experimental "
        "(fuente: ResultIndex, elaboración propia). Una \\emph{ejecución} es una "
        "corrida completa de las Fases 1--3 para una (configuración, semilla)."
    )
    notes = (
        heis_note + "La columna \\emph{Tasa aprob.} agrega \\emph{todas} las ejecuciones del "
        "modelo, incluidas las de fuera del régimen operativo válido, las cinco "
        "topologías y las cuatro profundidades; por tanto \\emph{no} es comparable "
        "con las tasas por configuración de las Tablas~\\ref{tab:cross_topo}, "
        "\\ref{tab:cross_topo_depth} y \\ref{tab:scaling} (restringidas al régimen "
        "válido y a la mejor profundidad). Es un indicador de cobertura de la "
        "campaña, no del rendimiento del pipeline en su régimen de operación."
    )
    return _wrap_table(
        caption,
        "tab:auto_campaign",
        "lcc",
        header,
        rows,
        notes=notes,
        short_caption="Conteo de ejecuciones por modelo",
    )


def _per_h_table(
    per_n: dict[int, list],
    n_values: list[int],
    label: str,
    caption: str,
    collector: TodoCollector,
    topic_kind: str,
    short_caption: str = "",
) -> list[str]:
    """Construye una tabla por-N (una fila por N) con métricas agregadas por-h.

    Columnas: N, nº puntos, |ΔE| (media), gap (media), ΔE/gap (media).
    El rango de h se unifica al más abarcativo y se declara en el pie (no como
    columna, para evitar rangos heterogéneos entre filas).
    """
    from qmbp_simulation.analysis.metrics import compute_deploy_summary

    # Métrica primaria |ΔE| primero (steering §5); ΔE/gap relegado a apoyo.
    # La columna ΔE/gap se reporta en % (media de los cocientes por punto), con la
    # unidad explícita para que no se confunda con el cociente de las medias de las
    # dos columnas anteriores (que da un valor distinto; ver pie).
    header = [
        "$N$",
        "$|\\Delta E|$",
        "gap",
        "$\\Delta E/\\mathrm{gap}$ (\\%)",
    ]
    rows: list[list[str]] = []
    pre_lines: list[str] = []
    h_min_global = float("inf")
    h_max_global = float("-inf")

    missing_ns = [n for n in n_values if n not in per_n or not per_n[n]]
    if missing_ns:
        pre_lines.append(
            collector.marker(
                topic_kind,
                f"Faltan datos por-h de heavy_hex para N={missing_ns}; no hay eval report per-h para esos tamaños.",
            )
        )

    for n in n_values:
        results = per_n.get(n)
        if not results:
            continue
        hs = [r.h for r in results]
        h_min_global = min(h_min_global, min(hs))
        h_max_global = max(h_max_global, max(hs))
        per_h_dicts = [{"de_gap": r.de_gap, "abs_error": r.abs_error} for r in results]
        summary = compute_deploy_summary(per_h_dicts)
        mean_gap = sum(r.gap for r in results) / len(results)
        mean_abs = summary.get("mean_abs_error", 0.0)
        mean_de_gap = summary["mean_de_gap"]  # media de los cocientes por punto (fracción)
        # Chequeo (no fix): si la media de cocientes difiere mucho del cociente de
        # medias, es porque algún punto tiene el gap estrecho e infla ΔE/gap. Es
        # legítimo, pero conviene que el lector lo sepa (lo explica el pie).
        ratio_of_means = mean_abs / mean_gap if mean_gap else 0.0
        if ratio_of_means and mean_de_gap > 3 * ratio_of_means:
            collector.add_inconsistency(
                f"heavy_hex N={n} ({label}): ΔE/gap medio={mean_de_gap * 100:.1f}\\% "
                f"es media de cocientes por punto, muy por encima del cociente de "
                f"medias ({ratio_of_means * 100:.1f}\\%); dominado por puntos de gap "
                f"estrecho. Verificar que el pie de tabla lo aclare (INFO, no error)."
            )
        rows.append(
            [
                str(n),
                _num(mean_abs),
                _num(mean_gap),
                _num(mean_de_gap * 100.0, 2),
            ]
        )

    if not rows:
        return [collector.marker(topic_kind, f"Sin datos por-h de heavy_hex para {label}.")]

    # Rango de h unificado (el más abarcativo cubierto por el conjunto de filas).
    # El pie describe QUÉ contiene la tabla (no explica las métricas — eso va en
    # el texto / la sección de métricas del capítulo).
    grilla_str = "\\{" + "; ".join(_num(g, 1) for g in H_GRID_EXTRAPOLATION) + "\\}"
    notes = (
        f"Evaluado sobre la grilla de $h$ canónica del régimen de extrapolación, "
        f"$h \\in {grilla_str}$, restringida al régimen válido ($h \\geq h_{{\\min}}$) "
        "de cada $N$; por eso los $N$ mayores contribuyen con menos puntos. "
        "$|\\Delta E|$ y gap son medias sobre esos puntos, en unidades de $J = 1$. "
        "La columna $\\Delta E/\\mathrm{gap}$ es la media de los cocientes \\emph{por "
        "punto} (no el cociente de las dos columnas anteriores): cerca de la frontera "
        "algún punto tiene el gap estrecho y su cociente individual es grande, de modo "
        "que la media de cocientes puede superar al cociente de las medias. La métrica "
        "primaria es $|\\Delta E|$; $\\Delta E/\\mathrm{gap}$ es el criterio de "
        "clasificación normalizado. Fuente: elaboración propia."
    )
    return _wrap_table(
        caption,
        label,
        "rrrr",
        header,
        rows,
        notes=notes,
        pre_lines=pre_lines,
        short_caption=short_caption,
    )


def gen_heavy_hex_intra_n(per_n: dict[int, list], collector: TodoCollector) -> list[str]:
    """Tabla de interpolación a un mismo tamaño de heavy_hex."""
    intra_ns = [n for n in sorted(per_n) if n <= 20]
    caption = (
        "Interpolación a un mismo tamaño en heavy-hex: la GNN reproduce "
        "$\\theta^*(h)$ en valores de $h$ no vistos, para cada $N$ de entrenamiento."
    )
    return _per_h_table(
        per_n,
        intra_ns,
        "tab:auto_heavy_hex_intra_n",
        caption,
        collector,
        "DATOS",
        short_caption="Interpolación a un mismo tamaño (heavy-hex)",
    )


def gen_heavy_hex_large_n(per_n: dict[int, list], collector: TodoCollector) -> list[str]:
    """Tabla de predicción a N grande de heavy_hex (entre tamaños)."""
    large_ns = [n for n in sorted(per_n) if n > 20]
    caption = (
        "Predicción entre tamaños (sobre casos no observados) en heavy-hex: la "
        "UnifiedMPNN entrenada con $N$ pequeños predice ángulos para $N$ grandes."
    )
    return _per_h_table(
        per_n,
        large_ns,
        "tab:auto_heavy_hex_large_n",
        caption,
        collector,
        "DATOS",
        short_caption="Predicción a $N$ grande (heavy-hex)",
    )


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


def _run_check_guarded(check, payload, rel: str, collector: TodoCollector) -> None:
    """Ejecuta un chequeo aislando sus fallos.

    Si el chequeo lanza una excepción, se registra como inconsistencia (visible, no
    silenciosa) y se continúa con los demás chequeos. Sin este guard, un único
    chequeo que fallara abortaba toda la fase de validación y el conteo de hallazgos
    caía sin aviso. `payload` es `full_text` (str) o `lines` (list[str]) según el
    chequeo; la firma común es check(payload, rel, collector).
    """
    try:
        check(payload, rel, collector)
    except Exception as exc:  # noqa: BLE001 - queremos aislar cualquier fallo
        collector.add_inconsistency(
            f"[{rel}] chequeo '{getattr(check, '__name__', str(check))}' falló con "
            f"{type(exc).__name__}: {exc}. Los demás chequeos continuaron."
        )


def check_tex(tex_path: Path, out_dir: Path, collector: TodoCollector) -> None:
    """Chequeos ligeros de consistencia sobre el .tex, alimentando el TODO log.

    Registra hallazgos como inconsistencias (con [archivo:línea]) para:
      - refs/labels: \\ref sin \\label (rompe -> ??), y label de tabla sin \\ref.
      - decimales con punto en lugar de coma (convención española del steering).
      - tablas auto_*.tex no conectadas al documento (sin \\input o label sin \\ref).
      - integridad de compilación: delimitador $ de math mode impar por línea,
        entornos \\begin/\\end desbalanceados o cruzados, llaves { } desbalanceadas.
      - aritmética de tablas (informe de corrección §1.3-§1.5, §1.9): columna que
        pretende ser un cociente y no lo es, fila 'Total' cuya suma no cuadra,
        escala de calificación (letra) incoherente con su columna numérica, y caso
        absoluto (n/m) cuyo porcentaje no coincide.
      - coma decimal en modo matemático sin \\usepackage{icomma} (§1.9a).
      - correlación r baja (r<0,7) presentada como 'predictor confiable' (§2.4).
    """
    if not tex_path.exists():
        collector.add_inconsistency(f"No existe el .tex a chequear: {tex_path}")
        return

    lines = tex_path.read_text(encoding="utf-8").splitlines()
    rel = tex_path.name
    full_text = "\n".join(lines)

    # El speedup S deja de ser un TODO cuando el documento provee una definición
    # formal reproducible (steering §5: "eliminado hasta tener definición
    # reproducible"). Detectamos \label{eq:speedup} como esa definición.
    has_speedup_def = bool(re.search(r"\\label\{eq:speedup\}", full_text))
    # (F) ¿carga el preámbulo el paquete icomma? Sin él, la coma decimal en modo
    # matemático ($0,93$) se renderiza con espacio y parece una lista de números.
    has_icomma = bool(re.search(r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\bicomma\b[^}]*\}", full_text))
    icomma_warned = False  # advertir una sola vez

    labels: dict[str, int] = {}  # label -> primera linea
    refs: dict[str, int] = {}  # ref -> primera linea
    inputs: set[str] = set()  # basenames incluidos via \input
    float_appear: dict[str, int] = {}  # label fig/tab -> linea del \begin{float} que lo contiene
    in_verbatim = False
    cur_float_line = 0  # linea del \begin{figure/table} abierto
    in_english = False  # dentro del Abstract (inglés) o bibliografía: no marcar anglicismos
    in_acronyms = False  # dentro del índice de acrónimos: las expansiones EN son legítimas
    # (12) Balance de entornos \begin{X}...\end{X}: pila de (nombre, línea de apertura)
    env_stack: list[tuple[str, int]] = []
    # (13) Balance de llaves { }: contador acumulado (multilínea) + línea de arranque del desbalance
    brace_balance = 0
    brace_open_line = 0

    for i, line in enumerate(lines, start=1):
        # Ignorar comentarios completos y bloques verbatim/lstlisting
        if re.search(r"\\begin\{(verbatim|lstlisting)\}", line):
            in_verbatim = True
        if re.search(r"\\end\{(verbatim|lstlisting)\}", line):
            in_verbatim = False
            continue
        if in_verbatim:
            continue
        # Zonas en inglés legítimo: capítulo Abstract y bibliografía.
        if re.search(r"\\chapter\{Abstract\}|\\begin\{thebibliography\}", line):
            in_english = True
        if re.search(r"\\mainmatter|\\end\{thebibliography\}", line):
            in_english = False
        # Índice de acrónimos: las expansiones inglesas (\emph{Machine Learning}...)
        # son definiciones de sigla, no anglicismos de prosa.
        if re.search(r"Índice de acrónimos", line):
            in_acronyms = True
        if in_acronyms and re.search(r"\\end\{description\}", line):
            in_acronyms = False
        # Quitar comentario de línea respetando el porcentaje escapado (\%)
        code = re.sub(r"(?<!\\)%.*$", "", line)
        if not code.strip():
            continue

        # (2b) Balance de $ (math mode) por línea. Un $ impar rompe el math mode
        # y corrompe el resto del párrafo. Se ignoran los \$ escapados. El TFIM
        # no usa $$...$$ display, así que el conteo por línea es fiable.
        dollar_count = code.replace(r"\$", "").count("$")
        if dollar_count % 2 != 0:
            collector.add_inconsistency(
                f"[{rel}:{i}] delimitador $ de modo matemático impar "
                f"({dollar_count} sin escapar): rompe el math mode. Revisar la línea."
            )

        # (12) Balance de entornos \begin{X}...\end{X}: pila LIFO. Un \begin sin
        # \end (o cierre cruzado) impide la compilación. Ignora \begin{document}
        # y entornos verbatim (ya filtrados arriba).
        for bm in re.finditer(r"\\(begin|end)\{([A-Za-z*]+)\}", code):
            kind, env = bm.group(1), bm.group(2)
            if env in ("document",):
                continue
            if kind == "begin":
                env_stack.append((env, i))
            else:  # end
                if not env_stack:
                    collector.add_inconsistency(f"[{rel}:{i}] \\end{{{env}}} sin \\begin previo (entorno huérfano).")
                elif env_stack[-1][0] != env:
                    open_env, open_ln = env_stack[-1]
                    collector.add_inconsistency(
                        f"[{rel}:{i}] \\end{{{env}}} no coincide con el último abierto "
                        f"\\begin{{{open_env}}} (L{open_ln}): entornos cruzados."
                    )
                    env_stack.pop()
                else:
                    env_stack.pop()

        # (13) Balance de llaves { }: contador acumulado multilínea. Ignora las
        # llaves escapadas \{ y \}. Un desbalance persistente señala un \cmd{
        # sin cerrar. Solo se reporta al final (una llave puede abrirse en una
        # línea y cerrarse en otra legítimamente).
        code_no_escaped = re.sub(r"\\[{}]", "", code)
        delta = code_no_escaped.count("{") - code_no_escaped.count("}")
        if brace_balance == 0 and delta != 0:
            brace_open_line = i
        brace_balance += delta

        # (D) Caso absoluto (n/m) cuyo porcentaje no coincide con n/m.
        # Se aceptan dos órdenes: "95\% (37/39)" y "(37/39) ... 95\%", con hasta
        # ~15 caracteres de separación (permite "95\,\% (37/39)" o texto breve).
        # La coincidencia se juzga contra el redondeo real round(100*n/m): un
        # porcentaje entero p es válido si p == round(100*n/m) (tolerancia de 1
        # punto solo para absorber redondeos hacia arriba/abajo declarados a mano).
        _dm = r"(?:(\d{1,3})\s*\\?%[^()\n]{0,15}?\((\d+)\s*/\s*(\d+)\))"
        _md = r"(?:\((\d+)\s*/\s*(\d+)\)[^%\n]{0,15}?(\d{1,3})\s*\\?%)"
        for pm in re.finditer(rf"{_dm}|{_md}", code):
            if pm.group(1) is not None:  # orden "% (n/m)"
                pct, n, m = int(pm.group(1)), int(pm.group(2)), int(pm.group(3))
            else:  # orden "(n/m) ... %"
                n, m, pct = int(pm.group(4)), int(pm.group(5)), int(pm.group(6))
            if m > 0:
                real = 100.0 * n / m
                if abs(real - pct) > 1.0 and round(real) != pct:
                    collector.add_inconsistency(
                        f"[{rel}:{i}] caso absoluto inconsistente: {pct}\\% ({n}/{m}) "
                        f"pero {n}/{m} = {real:.1f}\\% (redondea a {round(real)}\\%). "
                        "Corregir el porcentaje o el caso."
                    )

        # (F) Coma decimal en modo matemático sin icomma: "$0,93$", "$h \geq 1,3$".
        # Detecta \d,\d dentro de $...$; si no hay icomma, LaTeX lo renderiza mal.
        if not has_icomma and not icomma_warned:
            for mm in re.finditer(r"\$[^$]*\$", code):
                if re.search(r"\d,\d", mm.group(0)):
                    collector.add_inconsistency(
                        f"[{rel}:{i}] coma decimal en modo matemático sin \\usepackage{{icomma}}: "
                        f"'{mm.group(0)[:40]}' se renderiza con espacio (parece lista de números). "
                        f"Cargar icomma en el preámbulo o usar '.' en math. (se avisa una vez)"
                    )
                    icomma_warned = True
                    break

        # (G) Correlación r baja presentada como "predictor confiable".
        rm = re.search(r"\br\s*=\s*0[.,](\d+)", code)
        if rm:
            r_val = float("0." + rm.group(1))
            if r_val < 0.7 and re.search(
                r"confiable|fiable|predictor\s+(confiable|fiable|robusto)|robusto",
                code,
                re.IGNORECASE,
            ):
                collector.add_inconsistency(
                    f"[{rel}:{i}] r = 0,{rm.group(1)} (r²={r_val**2:.2f}) descrito como "
                    f"'confiable/robusto': r² < 0,5 explica menos de la mitad de la varianza. "
                    f"Matizar (relación monótona pero ruidosa) o dar sensibilidad/especificidad."
                )

        # Prosa: quita \texttt{...}, \url{...}, math $...$ y comandos con label/ref
        # para no marcar anglicismos dentro de código o identificadores.
        prose = _strip_protected(code)

        if re.search(r"\\begin\{(figure|table)\*?\}", code):
            cur_float_line = i
        # (11) Caption sin título corto para el índice (pedido del usuario): las
        # figuras/tablas deben usar \caption[corto]{largo} para que el índice no
        # muestre la descripción completa. Detecta \caption{ (sin corchete).
        if re.search(r"\\caption\{", code) and not re.search(r"\\caption\[", code):
            collector.add_inconsistency(
                f"[{rel}:{i}] %TODO-INDICE \\caption sin título corto "
                "(usar \\caption[título breve]{descripción} para el índice de "
                "figuras/tablas; requiere criterio para el título breve)."
            )
        for m in re.finditer(r"\\label\{([^}]+)\}", code):
            labels.setdefault(m.group(1), i)
            # Aparición visual del float = línea de su \begin (o la del label si va suelto)
            if m.group(1).startswith(("fig:", "tab:")):
                float_appear.setdefault(m.group(1), cur_float_line or i)
        if re.search(r"\\end\{(figure|table)\*?\}", code):
            cur_float_line = 0
        for m in re.finditer(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", code):
            refs.setdefault(m.group(1), i)
        for m in re.finditer(r"\\input\{([^}]+)\}", code):
            inputs.add(Path(m.group(1)).name.replace(".tex", ""))

        # (2) Decimales con punto: número.número (no versiones tipo v2.x ni URLs,
        # ni longitudes LaTeX tipo 0.90\textwidth / 0.5cm que no son decimales de prosa,
        # ni versiones de software tipo "Python 3.11").
        for m in re.finditer(r"(?<![\w.])\d+\.\d+(?![\w.])", code):
            after = code[m.end() : m.end() + 14]
            before = code[max(0, m.start() - 16) : m.start()]
            if re.match(r"\s*(\\(?:text|line|column)width|\\height|cm|mm|pt|em|ex|in|\+)\b", after):
                continue
            if _VERSION_RE.search(before):
                continue
            # DOI: '10.1038/...' — el prefijo '10.NNNN' seguido de '/' es un DOI, no
            # un decimal de prosa. Se salta (típico en \bibitem y \url de bibliografía).
            if m.group(0).startswith("10.") and after.lstrip().startswith("/"):
                continue
            if "doi" in before.lower() or re.search(r"10\.\d{4,9}/", code):
                continue
            collector.add_inconsistency(
                f"[{rel}:{i}] decimal con punto '{m.group(0)}' (usar coma decimal en tablas/texto, steering §7)."
            )

        # (2c) Anglicismos de métrica (Grade/Pass/Rate): usar equivalentes en español.
        for m in ENGLISH_METRIC_RE.finditer(code):
            collector.add_inconsistency(
                f"[{rel}:{i}] palabra inglesa '{m.group(0)}' "
                "(usar español: Grade->calificación/nota, Pass->aprobado, "
                "Rate->tasa/proporción)."
            )

        # (2d) Énfasis en profundidad requerida (de-enfatizar, steering §5).
        for m in DEPTH_EMPHASIS_RE.finditer(code):
            collector.add_inconsistency(
                f"[{rel}:{i}] %TODO-PROFUNDIDAD fórmula de capas requeridas "
                f"'{m.group(0)}' (quitar énfasis; la profundidad es un parámetro "
                "fijo, no el eje del trabajo)."
            )

        # (Editorial 3) Tasa de aprobación sin caso absoluto (n/m), steering §5.
        # Solo aplica a porcentajes de aprobación (no a tasas de error, fidelidades,
        # umbrales o mejoras porcentuales, que son magnitudes físicas sin (n/m)).
        _is_passrate = re.search(r"(aprobaci|aprobad|tasa de aprob)", code)
        _pct_threshold = re.search(r"[<>]\s*\d{1,3}\\%|\\leq|\\geq|umbral|criterio", code)
        # Falsos positivos: el % es un ΔE/gap, una proporción de fallos, o un
        # ahorro de iteraciones; o la línea ya declara un conteo (N ejecuciones).
        _pct_other = re.search(
            r"gap|de los fallos|de fallos|prevenible|ahorro|iteraciones|"
            r"\d+\s+ejecuciones|de acuerdo",
            code,
        )
        # El % candidato debe estar cerca de la palabra 'aprobaci' (misma cláusula).
        _pr_pct = re.search(r"aprobaci[oó]n[^.]{0,20}?(\d{1,3}\\%)|(\d{1,3}\\%)[^.]{0,20}?aprob", code)
        if _is_passrate and _pr_pct and not ABS_CASE_RE.search(code) and not _pct_threshold and not _pct_other:
            pct = _pr_pct.group(1) or _pr_pct.group(2)
            collector.add_inconsistency(
                f"[{rel}:{i}] %TODO-CIFRA tasa de aprobación '{pct}' sin caso "
                "absoluto entre paréntesis (usar formato '95\\% (37/39)', steering §5)."
            )

        # (Editorial 4) Rango numérico con guion simple (usar -- en LaTeX).
        for m in BAD_RANGE_RE.finditer(code):
            # Excluir dentro de comandos protegidos ya filtrados por 'code' sin comentario.
            collector.add_inconsistency(
                f"[{rel}:{i}] rango '{m.group(0)}' con guion simple (usar '--' en LaTeX para rangos numéricos)."
            )

        # (2b) Guiones Unicode em/en-dash: prohibidos; usar '---' (em) o '--' (rango) de LaTeX.
        for ch, name, repl_hint in (("\u2014", "em-dash", "---"), ("\u2013", "en-dash", "--")):
            col = line.find(ch)
            if col != -1:
                n_occ = line.count(ch)
                collector.add_inconsistency(
                    f"[{rel}:{i}] símbolo Unicode '{ch}' ({name}) x{n_occ} en col {col + 1} "
                    f"(prohibido; reemplazar por '{repl_hint}' de LaTeX)."
                )

        # (8) Anglicismos (steering §3). Solo en prosa (no Abstract/biblio/índice/código).
        if not in_english and not in_acronyms:
            for pat, repl in ANGLICISM_FIXES:
                for m in re.finditer(pat, prose):
                    collector.add_inconsistency(
                        f"[{rel}:{i}] %TODO-ANGLICISMO '{m.group(0)}' -> '{repl}' "
                        "(auto-corregible con --fix-anglicisms, steering §3)."
                    )
            # Contexto de LISTA DE MÓDULOS de código: 'framework' (y otros) es el
            # nombre literal del subpaquete cuando aparece rodeado de otros nombres
            # de módulo en inglés (models, solvers, analysis, pipeline, utils...).
            # No es el anglicismo 'framework' como sustantivo -> no marcarlo ahí.
            _module_list_ctx = bool(
                re.search(
                    r"\b(models|solvers|circuits|execution|optimizers|predictors|"
                    r"pipeline|analysis|utils)\b.*\bframework\b|"
                    r"\bframework\b.*\b(models|solvers|utils|analysis|pipeline)\b",
                    prose,
                    re.IGNORECASE,
                )
            )
            for pat, note in ANGLICISM_DETECT:
                for m in re.finditer(pat, prose):
                    if m.group(0).lower() == "framework" and _module_list_ctx:
                        continue  # nombre de módulo, no anglicismo
                    # Glosa legítima "definir 1 vez en cursiva": el término inglés
                    # aparece dentro de (\emph{...}) o (\textit{...}), que es la
                    # definición única que pide el steering §3, no un uso sin
                    # traducir. En el CÓDIGO original (no la prosa neutralizada) se
                    # comprueba que el match esté envuelto así.
                    term = re.escape(m.group(0))
                    if re.search(r"\((?:\\emph|\\textit)\{" + term + r"\}\)", code, re.IGNORECASE):
                        continue  # ya glosado en cursiva una vez -> legítimo
                    collector.add_inconsistency(
                        f"[{rel}:{i}] %TODO-ANGLICISMO '{m.group(0)}': {note} (requiere criterio; no auto-corregible)."
                    )
            # (9) Inciso con em-dash '---' (el usuario pide no usar guiones como
            # separador). Criterio humano para reemplazar por comas/paréntesis.
            # Exclusiones (NO son incisos separadores):
            #  - celda de tabla: la línea tiene '&' (un '---' solo marca "no medido");
            #  - leyenda que documenta el guion ('un guion (---) indica ...');
            #  - '---' pegado a '&' o '\\' (marcador de dato ausente en tabular).
            _emdash_is_cell = (
                "&" in code
                or re.search(r"guion\s*\(-*\)|guion\s+\(---\)", prose, re.IGNORECASE)
                or re.search(r"(?:&|\\\\)\s*-*---\s*(?:&|\\\\)", code)
            )
            if LATEX_EMDASH_RE.search(prose) and not _emdash_is_cell:
                n_dash = len(LATEX_EMDASH_RE.findall(prose))
                collector.add_inconsistency(
                    f"[{rel}:{i}] inciso con '---' (em-dash LaTeX) x{n_dash} "
                    "(el usuario pide no usar guiones como separador; usar comas o "
                    "paréntesis — requiere criterio, no auto-corregible)."
                )

            # (10) Gramática/registro (steering §4).
            for pat, repl in GRAMMAR_FIXES:
                for m in re.finditer(pat, prose):
                    collector.add_inconsistency(
                        f"[{rel}:{i}] %TODO-GRAMATICA '{m.group(0)}' -> '{repl}' "
                        "(auto-corregible con --fix-grammar, steering §4)."
                    )
            if A_N_RE.search(prose):
                collector.add_inconsistency(
                    f"[{rel}:{i}] %TODO-GRAMATICA 'a N=...' -> 'para N=...' "
                    "(steering §4; verificar que no sea un uso matemático legítimo)."
                )
            for m in COMO_RE.finditer(prose):
                collector.add_inconsistency(
                    f"[{rel}:{i}] %TODO-GRAMATICA '{m.group(0)}' -> usar 'cómo' con "
                    "tilde cuando explica el modo (steering §4)."
                )
            if FILLER_RE.search(prose):
                collector.add_inconsistency(
                    f"[{rel}:{i}] %TODO-ESTILO muletilla '{FILLER_RE.search(prose).group(0)}' "
                    "(variar la redacción; reservar 'confirma' para evidencia concluyente, §4)."
                )

        # (5) Tono comercial (steering §6): moderar, añadir matiz cuantitativo.
        for m in TONE_WORDS_RE.finditer(code):
            word = m.group(0).lower()
            # "óptimo" técnico (parámetros/profundidad/configuración óptimos) es el
            # mínimo de la optimización, no un adjetivo laudatorio: no marcar.
            if word.startswith(("óptim", "optim")):
                ventana = code[max(0, m.start() - 45) : m.start() + 12]
                if TONE_OPTIMO_TECNICO_RE.search(ventana):
                    continue
            # "robusto/a/ez" como propiedad técnica (frente a ruido, atípicos,
            # semillas, muestreo): es medible, no laudatorio.
            if word.startswith("robust"):
                ventana = code[m.start() : m.start() + 70]
                if TONE_ROBUSTO_TECNICO_RE.search(ventana):
                    continue
            # "demuestra/demostraron" que reporta un teorema citado (los autores de
            # X demostraron ...) es reporte fiel de literatura, no afirmación propia.
            if word.startswith("demuestra") or word.startswith("demostr"):
                ventana = code[max(0, m.start() - 60) : m.start() + 40]
                if TONE_DEMUESTRA_CITA_RE.search(ventana):
                    continue
            collector.add_inconsistency(
                f"[{rel}:{i}] %TODO-TONO tono a moderar '{m.group(0)}' "
                "(usar muestra/sugiere; añadir comparación cuantitativa o "
                "'dentro de las configuraciones evaluadas')."
            )

        # (6) Métricas (steering §5): speedup solo es TODO si NO hay definición
        # formal reproducible (\label{eq:speedup}); PassRate sin |ΔE|.
        if not has_speedup_def:
            for m in SPEEDUP_RE.finditer(code):
                collector.add_inconsistency(
                    f"[{rel}:{i}] %TODO-METRICA speedup/aceleración '{m.group(0)}' "
                    "(eliminar hasta tener definición reproducible de coste)."
                )
        if PASSRATE_RE.search(code) and not ABS_ERR_RE.search(code):
            # Falsos positivos que NO requieren |ΔE| acompañante:
            #  - casos negativos (Heisenberg/Kitaev): el indicador es F, no |ΔE| (§3.4);
            #  - leyendas de tabla (\caption): el |ΔE| va en la propia tabla, no en el pie;
            #  - la fidelidad F ya acompaña a la tasa en la misma línea;
            #  - la frase REFERENCIA una tabla (\ref{tab:...}): el |ΔE| vive en esa
            #    tabla, la prosa solo remite a ella (no es un reporte suelto);
            #  - la frase EXPLICA la variación de la tasa con la profundidad
            #    ('p=2 a p=3', 'crece con la profundidad'): es análisis de un
            #    fenómeno, no la presentación de un resultado de PassRate como éxito.
            #  - experimento de COSTE (DyPP / ahorro de iteraciones): la métrica
            #    primaria es el nº de iteraciones VQE, no |ΔE| (ambas rutas convergen
            #    al mismo mínimo variacional, mismo |ΔE|); la tasa mide que la calidad
            #    no se degrada. Reportar |ΔE| aquí sería redundante (§5): un ahorro de
            #    iteraciones se acompaña de su magnitud de coste (iteraciones), no de
            #    |ΔE|. Se exige co-ocurrencia de 'iteraciones' con un marcador de coste
            #    (ahorro/reducción/DyPP/warm-start) para no exceptuar de más.
            _cost_experiment = re.search(r"iteracion", code, re.IGNORECASE) and re.search(
                r"ahorro|reducci[oó]n|DyPP|warm[-\s]?start|predicci[oó]n\s+din[aá]mica",
                code,
                re.IGNORECASE,
            )
            _skip = (
                NEGATIVE_MODEL_RE.search(code)
                or "\\caption" in code
                or re.search(r"fidelidad|\bF\b|\\bar\{F\}|F_", code)
                or re.search(r"\\ref\{tab:", code)
                or _cost_experiment
                or re.search(
                    r"p\s*=\s*\d\s*a\s*p\s*=\s*\d|con\s+la\s+profundidad|"
                    r"crece\s+con\s+\$?p|salto\s+(cualitativo|de\s+la\s+tasa)",
                    code,
                    re.IGNORECASE,
                )
            )
            if not _skip:
                collector.add_inconsistency(
                    f"[{rel}:{i}] %TODO-METRICA PassRate sin |ΔE| acompañante "
                    "(reportar la métrica primaria |ΔE|, steering §5)."
                )
        if PER_SITE_RE.search(code):
            m = PER_SITE_RE.search(code)
            collector.add_inconsistency(
                f"[{rel}:{i}] %TODO-METRICA error por sitio '{m.group(0)}' PROHIBIDO "
                "(|ΔE|/N se retiró; no aporta sobre |ΔE|, steering §5). Eliminar."
            )

        # (7) Reproducibilidad (steering §8): 'exacto' + DMRG/MPS; statevector a N grande.
        # Se exceptúan los usos legítimos (diagonalización/simulación exacta, MPS
        # exacto con chi=1, estado fundamental exacto): esos no llaman "exacto" a
        # un resultado DMRG/MPS convergido aproximado, que es lo que la regla veda.
        if EXACT_TN_RE.search(code) and not EXACT_OK_RE.search(code):
            collector.add_inconsistency(
                f"[{rel}:{i}] %TODO-REPRO 'exacto' aplicado a DMRG/MPS "
                "(usar 'convergido dentro de la tolerancia'; mostrar convergencia en χ)."
            )
        # No marcar si la línea usa statevector en contexto negado o de contraste
        # con MPS (p. ej. "para N>22, donde NO es posible la comparación con
        # statevector, se verifica ... a N=40 con χ"): ahí el N grande es MPS, no SV.
        sv_negated = re.search(
            r"no\s+(es\s+posible|accesible|se\s+puede|alcanza)[^.]*statevector"
            r"|statevector[^.]*no\s+(es\s+posible|accesible|disponible)"
            r"|donde\s+no[^.]*statevector",
            code,
            re.IGNORECASE,
        )
        sv_mps_contrast = bool(re.search(r"\\?chi|\bMPS\b", code)) and sv_negated
        if SV_BIGN_RE.search(code) and not sv_mps_contrast:
            for bm in BIGN_RE.finditer(code):
                if int(bm.group(1)) > 22:
                    collector.add_inconsistency(
                        f"[{rel}:{i}] %TODO-REPRO statevector con N={bm.group(1)}>22 "
                        "(inviable en memoria; ¿era backend MPS? indicar χ y tolerancia)."
                    )

        # (7c) Referencia "exacta" (diagonalización / vector de estado) a N>22:
        # imposible; a esa escala solo hay DMRG/MPS.
        # Robustez (evita falsos positivos): NO marcar cuando el N>22 aparece en un
        # contexto de COMPARACIÓN DE COSTE ("más eficiente que la diag. exacta,
        # O(2^N)", "escala como") ni cuando la diag. exacta está anclada a otro N
        # pequeño en la misma frase (p. ej. "comparación con diag. exacta a N=14 ...
        # criterios para N<=40"). Se exige que el N>22 esté a <=25 chars del marcador
        # de exactitud y que la frase no sea una comparación de coste.
        if EXACT_METHOD_RE.search(code):
            cost_compare = re.search(
                r"m[aá]s\s+eficiente|O\(2\^|escala\s+como|coste\s+computacional|"
                r"varios\s+[oó]rdenes",
                code,
                re.IGNORECASE,
            )
            if not cost_compare:
                for em in EXACT_METHOD_RE.finditer(code):
                    # N cercano (a la derecha, misma cláusula corta) al marcador.
                    near = code[em.end() : em.end() + 25]
                    nm = re.search(r"N\s*[=>]\s*(\d+)", near)
                    if nm and int(nm.group(1)) > 22:
                        collector.add_inconsistency(
                            f"[{rel}:{i}] %TODO-REPRO referencia exacta (diagonalización "
                            f"/ vector de estado) a N={nm.group(1)}>22: inviable "
                            "(2^N amplitudes); a esa escala solo DMRG/MPS. Corregir el "
                            "método o el N (steering §8)."
                        )
                        break

    # (12b) Entornos abiertos y nunca cerrados (rompen la compilación)
    for env, ln in env_stack:
        collector.add_inconsistency(
            f"[{rel}:{ln}] \\begin{{{env}}} sin su \\end{{{env}}} correspondiente "
            "(entorno sin cerrar): impide la compilación."
        )

    # (13b) Desbalance global de llaves { } (un \cmd{ sin cerrar en alguna parte)
    if brace_balance != 0:
        signo = "más '{' que '}'" if brace_balance > 0 else "más '}' que '{'"
        collector.add_inconsistency(
            f"[{rel}:{brace_open_line}] llaves desbalanceadas en el documento "
            f"({signo}, saldo {brace_balance:+d}): revisar desde esa línea un \\cmd{{ "
            "sin cerrar."
        )

    # (1a) Resolver \label definidos DENTRO de los \input{...} incluidos.
    # Antes esto era un falso positivo conocido ([ref-auto-input]): los \label de
    # las tablas auto_*.tex viven en el archivo incluido, no en el .tex principal,
    # así que la comprobación ref->label los marcaba como rotos. Ahora leemos cada
    # archivo incluido (relativo al dir del .tex o al out_dir de tablas) y fundimos
    # sus labels, eliminando el falso positivo de raíz.
    tex_dir = tex_path.parent
    for stem in sorted(inputs):
        for cand in (
            tex_dir / f"{stem}.tex",
            tex_dir / stem,
            out_dir / f"{stem}.tex",
            out_dir / stem,
        ):
            if cand.exists() and cand.is_file():
                try:
                    inc_text = cand.read_text(encoding="utf-8")
                except OSError:
                    break
                for m in re.finditer(r"\\label\{([^}]+)\}", inc_text):
                    labels.setdefault(m.group(1), -1)  # -1: definido en \input
                break

    # (1) refs -> label inexistente (renderiza como ??)
    for ref, ln in sorted(refs.items(), key=lambda kv: kv[1]):
        if ref not in labels:
            collector.add_inconsistency(f"[{rel}:{ln}] \\ref{{{ref}}} sin \\label correspondiente (saldrá como ??).")
    # (1b) label de tabla nunca referenciado
    for lab, ln in sorted(labels.items(), key=lambda kv: kv[1]):
        if lab.startswith("tab:") and lab not in refs:
            collector.add_inconsistency(f"[{rel}:{ln}] tabla \\label{{{lab}}} nunca referenciada con \\ref.")

    # (4) tablas auto_*.tex no conectadas al documento
    for auto_tex in sorted(out_dir.glob("auto_*.tex")):
        stem = auto_tex.stem
        auto_label = f"tab:{stem}"
        if stem not in inputs:
            collector.add_inconsistency(
                f"tabla generada '{auto_tex.name}' no está incluida en {rel} (falta \\input{{tables/{stem}}})."
            )
        elif auto_label not in refs:
            collector.add_inconsistency(
                f"tabla generada '{auto_tex.name}' incluida pero su \\label{{{auto_label}}} no se referencia con \\ref."
            )

    # (21) Figuras/tablas que aparecen antes de mencionarse.
    for lab, appear_ln in sorted(float_appear.items(), key=lambda kv: kv[1]):
        ref_ln = refs.get(lab)
        if ref_ln is None:
            continue  # ya cubierto por 'nunca referenciada'
        if ref_ln > appear_ln:
            kind = "figura" if lab.startswith("fig:") else "tabla"
            collector.add_inconsistency(
                f"[{rel}:{appear_ln}] {kind} \\label{{{lab}}} aparece (L{appear_ln}) "
                f"antes de su primera mención \\ref (L{ref_ln})."
            )

    # (20), (22) y (15): cruces semánticos y bibliografía sobre el texto completo.
    full_text = tex_path.read_text(encoding="utf-8")

    # Despacho de chequeos con aislamiento de fallos: cada chequeo se ejecuta dentro
    # de un guard, de modo que una excepción en uno NO impide correr los demás (antes,
    # un fallo abortaba la mitad de los chequeos y el conteo caía en silencio). El
    # guard registra el fallo como inconsistencia para que sea visible, no silencioso.
    #
    # Los chequeos se agrupan por la firma de su entrada: (a) los que reciben el texto
    # completo `full_text`, y (b) los que reciben la lista de líneas `lines`.
    text_checks = (
        _check_cross_section_figures,  # 20: cruces de cifras entre secciones
        _check_hypotheses_coverage,  # 22: cobertura hipótesis -> conclusiones
        _check_bibliography,  # 15: bibliografía (arXiv dup, a/b, huérfanas)
        _check_editorial,  # 1/2: siglas sin definir, términos no unificados
        _check_regime_consistency,  # 23: tablas vs fuente canónica e inter-tabla
        _check_text_vs_table,  # 2: prosa <-> celda cross_topo_depth (por p)
        _check_cz_budget,  # 1: conteo CZ coherente (2 CZ/término)
        _check_n_scales,  # 3: N máximo 22/40/250
        _check_h_grid_counts,  # 4: malla de h (39/52)
        _check_speedup_error_ratio,  # F: A vs R(N), símbolo S
        _check_keywords,  # §4.2: palabras clave (3 sitios)
        _check_acronym_index,  # §4.3: cobertura del índice de acrónimos
        _check_caption_sources,  # §4.1: fuente en captions
        _check_unsupported_claim,  # §1.1: 100% sin tabla que lo respalde
        _check_equation_constants,  # §1.2: constantes sin simplificar / 'lineal'
        _check_code_model_names,  # §1.5.3: nombres de modelo en clave de código
        _check_source_is_code,  # §1.5.4/§4.1: fuente que cita clase de código
        _check_notation_collisions,  # §3: colisiones de notación (Δ, θ_opt)
        _check_decimal_comma_in_intervals,  # §5: intervalo '[1,00, 5,00]'
        _check_arxiv_in_text,  # §5/§6: id de arXiv citado en el cuerpo
        _check_math_comma_intervals,  # §1.9a: coma en modo matemático
        _check_energy_units,  # §4.5: 'energy units' en rótulos
        _check_xref_style,  # §5: estilo de referencia cruzada
        _check_entropy_coefficient,  # §1.10: coeficiente de la entropía crítica
        _check_duplicate_factor_values,  # §1.1/§4.5: factor duplicado entre figuras
        _check_inline_arithmetic,  # §5: producto inline 'A×B×C=D' que no cuadra
    )
    for check in text_checks:
        _run_check_guarded(check, full_text, rel, collector)

    # Chequeos POR BLOQUE de sección: se itera el documento una vez y cada chequeo
    # de bloque recibe su sección con la prosa y las \ref locales. Necesitan out_dir
    # (leen los auto_*.tex generados). Es más preciso que barrer con ventana fija:
    # el hallazgo queda anclado a la sección que contiene el \ref.
    block_checks = (
        _make_block_table_consistency_check(out_dir),  # tabla auto ↔ prosa de su sección
    )
    _run_block_checks(full_text, rel, collector, block_checks)

    # Chequeos que operan sobre la lista de líneas (posición exacta).
    line_checks = (
        _check_tabular_columns,  # desajuste nº de celdas vs preámbulo del tabular
        _check_table_arithmetic,  # A/B/C/D: cociente, suma total, calificación, n/m
        _check_topo_name_casing,  # §5/§11: grafía uniforme de topología en celdas
    )
    for check in line_checks:
        _run_check_guarded(check, lines, rel, collector)


def _count_tabular_cols(spec: str) -> int:
    """Cuenta columnas de un preámbulo de tabular (l/c/r/p{..}), ignorando | y @{}."""
    # Quitar @{...} y p{...}/m{...}/b{...} (cuentan como 1 columna cada uno).
    spec = re.sub(r"@\{[^}]*\}", "", spec)
    n_p = len(re.findall(r"[pmb]\{[^}]*\}", spec))
    spec_wo_p = re.sub(r"[pmb]\{[^}]*\}", "", spec)
    n_lcr = len(re.findall(r"[lcr]", spec_wo_p))
    return n_lcr + n_p


def _check_tabular_columns(lines: list[str], rel: str, collector: TodoCollector) -> None:
    """Detecta filas de un tabular cuyo nº de celdas no coincide con el preámbulo.

    Un desajuste ('lccc' con filas de 3 celdas) descuadra la tabla o rompe la
    compilación. Cuenta '&' no escapados (+1) por fila de datos, ignorando líneas
    de \\multicolumn (que agregan celdas) y reglas (\\toprule, \\midrule, etc.).
    """
    in_tab = False
    ncols = 0
    start_line = 0
    for i, line in enumerate(lines, start=1):
        code = re.sub(r"(?<!\\)%.*$", "", line)
        # Captura el preámbulo permitiendo un nivel de llaves anidadas (p{2cm}).
        m = re.search(r"\\begin\{tabular\}\s*(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{[^}]*\})*)\}", code)
        if m:
            in_tab = True
            ncols = _count_tabular_cols(m.group(1))
            start_line = i
            code = code[m.end() :]  # analizar el resto de la línea por si hay fila
        if not in_tab:
            continue
        if re.search(r"\\end\{tabular\}", code):
            in_tab = False
            continue
        # Solo filas de datos: deben terminar en '\\' y no ser reglas/comandos.
        if "\\\\" not in code:
            continue
        if re.search(r"\\(top|mid|bottom|cmid)rule|\\hline", code):
            continue
        if "\\multicolumn" in code or "\\multirow" in code:
            continue  # estos alteran el conteo simple de '&'; se omiten
        row = code.split("\\\\")[0]
        n_amp = len(re.findall(r"(?<!\\)&", row))
        n_cells = n_amp + 1
        if row.strip() and n_cells != ncols:
            collector.add_inconsistency(
                f"[{rel}:{i}] tabla con desajuste de columnas: preámbulo declara "
                f"{ncols} (\\begin{{tabular}} en L{start_line}) pero la fila tiene "
                f"{n_cells} celdas. Corregir el preámbulo o la fila."
            )


def _cell_to_float(cell: str) -> float | None:
    """Extrae el número de una celda LaTeX. Maneja coma decimal, \\%, $...$,
    \\textbf{}, notación científica (1,2e-3) y potencias 10^{-3}. Devuelve None
    si la celda no contiene un único número interpretable."""
    s = cell.strip()
    # quitar énfasis/formato y matemática de delimitadores
    s = re.sub(r"\\(textbf|textit|emph|mathbf|bm)\{([^}]*)\}", r"\2", s)
    s = s.replace("$", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\,", "").replace("~", "").strip()
    # notación 10^{-3} o \times 10^{-3}
    m_sci = re.search(
        r"([-+]?\d+(?:[.,]\d+)?)\s*(?:\\times|\\cdot|\bx\b)?\s*10\^\{?(-?\d+)\}?",
        s,
    )
    if m_sci:
        mant = float(m_sci.group(1).replace(",", "."))
        return mant * (10 ** int(m_sci.group(2)))
    # notación 1,2e-3
    m_e = re.fullmatch(r"([-+]?\d+(?:[.,]\d+)?)[eE]([-+]?\d+)", s)
    if m_e:
        return float(m_e.group(1).replace(",", ".")) * (10 ** int(m_e.group(2)))
    # número simple con coma o punto decimal (rechaza rangos "1--2" y listas)
    m = re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?", s)
    if m:
        return float(s.replace(",", "."))
    return None


def _parse_table_blocks(lines: list[str]) -> list[dict]:
    """Extrae bloques de tabla: encabezados, filas de datos numéricas, leyenda
    (\\caption) y línea inicial. Reutiliza el parseo de _check_tabular_columns."""
    blocks: list[dict] = []
    in_tab = False
    cur: dict = {}
    caption = ""
    # buscar el \caption más cercano por encima del \begin{tabular}
    for i, line in enumerate(lines, start=1):
        code = re.sub(r"(?<!\\)%.*$", "", line)
        cap_m = re.search(r"\\caption(?:\[[^\]]*\])?\{(.+)", code)
        if cap_m:
            caption = cap_m.group(1)
        m = re.search(r"\\begin\{tabular\}\s*(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{[^}]*\})*)\}", code)
        if m:
            in_tab = True
            cur = {"start": i, "caption": caption, "header": [], "rows": [], "raw_rows": []}
            code = code[m.end() :]
            caption = ""
        if not in_tab:
            continue
        if re.search(r"\\end\{tabular\}", code):
            in_tab = False
            blocks.append(cur)
            continue
        if "\\\\" not in code:
            continue
        if re.search(r"\\(top|mid|bottom|cmid)rule|\\hline", code):
            continue
        if "\\multicolumn" in code or "\\multirow" in code:
            continue
        row = code.split("\\\\")[0]
        cells = re.split(r"(?<!\\)&", row)
        cells = [c.strip() for c in cells]
        # ¿fila de datos (tiene ≥1 número) o encabezado (sin números)?
        nums = [_cell_to_float(c) for c in cells]
        if any(n is not None for n in nums):
            cur["rows"].append(nums)
            cur["raw_rows"].append((i, cells))
        elif not cur["header"]:
            cur["header"] = cells
    return blocks


# Forma canónica de la primera celda (topología) en las tablas del cuerpo: primera
# letra en mayúscula y grafía única (steering §11, informe §5). Mapea la forma en
# minúscula/variante a la forma canónica esperada en celda.
_TOPO_CELL_CANON = {
    "cadena 1d": "Cadena 1D",
    "heavy-hex": "Heavy-hex",
    "escalera": "Escalera",
    "cuadrada": "Cuadrada",
    "red cuadrada": "Cuadrada",
    "triangular": "Triangular",
}


def _check_topo_name_casing(lines: list[str], rel: str, collector: TodoCollector) -> None:
    """(§5/§11) Grafía uniforme del nombre de topología en las celdas de tabla.

    Recorre las tablas del documento y, cuando la primera celda de una fila de datos
    es un nombre de topología, verifica que use la forma canónica de celda
    ('Cadena 1D', 'Heavy-hex', 'Escalera', 'Cuadrada', 'Triangular'): primera letra
    en mayúscula y sin la variante 'red cuadrada'. Atrapa el 'heavy-hex' en minúscula
    que se coló en una tabla del apéndice y evita que reaparezca al editar a mano.

    Señal, no fix.
    """
    for blk in _parse_table_blocks(lines):
        for line_no, cells in blk.get("raw_rows", []):
            if not cells:
                continue
            first = cells[0].strip()
            key = first.lower()
            canon = _TOPO_CELL_CANON.get(key)
            if canon and first != canon:
                collector.add_inconsistency(
                    f"[{rel}:{line_no}] celda de topología '{first}' con grafía no "
                    f"canónica; usar '{canon}' (mayúscula inicial, grafía única) para "
                    f"unificar con el resto de las tablas (§5/§11)."
                )


# Escala de calificación tipo "A (< 0,05), B (< 0,10), C (< 0,30), D (< 1,00), F (>= 1,00)"
_GRADE_SCALE_RE = re.compile(r"([A-F])\s*\(\s*<\s*([\d.,]+)\s*\)", re.IGNORECASE)


def _check_table_arithmetic(lines: list[str], rel: str, collector: TodoCollector) -> None:
    """Chequeos aritméticos sobre tablas (informe de corrección §1.3, §1.4, §1.5, §1.9):
    A) columna que debería ser cociente col_a/col_b y no lo es;
    B) fila 'Total' cuya suma no coincide con las filas de arriba;
    C) columna de calificación (letra) incoherente con la escala de la leyenda;
    D) caso absoluto (n/m) cuyo porcentaje no coincide.
    """
    blocks = _parse_table_blocks(lines)
    for blk in blocks:
        rows = [r for r in blk["rows"] if r]
        if not rows:
            continue
        ncol = max(len(r) for r in rows)
        start = blk["start"]

        # --- A) detectar columna-cociente: para cada terna de columnas numéricas
        # (a, b, c) donde c ≈ a/b en la mayoría de filas, marcar las que no cuadran.
        # Solo si al menos 60% de las filas cuadran (indica que ESA es la relación).
        def col(vals, idx):
            return [r[idx] if idx < len(r) and r[idx] is not None else None for r in vals]

        # Columnas candidatas a numerador/denominador: excluye las que contengan
        # ceros o negativos (parámetros como g=0, deltas con signo) para evitar
        # cocientes espurios amplificados por división por ~0.
        def col_all_positive(idx):
            vals = [r[idx] for r in rows if idx < len(r) and r[idx] is not None]
            return len(vals) >= 4 and all(v > 1e-6 for v in vals)

        pos_cols = [i for i in range(ncol) if col_all_positive(i)]
        for a_idx in pos_cols:
            for b_idx in pos_cols:
                if a_idx == b_idx:
                    continue
                for c_idx in pos_cols:
                    if c_idx in (a_idx, b_idx):
                        continue
                    a, b, c = col(rows, a_idx), col(rows, b_idx), col(rows, c_idx)
                    triples = [
                        (ra, rb, rc)
                        for ra, rb, rc in zip(a, b, c, strict=False)
                        if ra is not None and rb is not None and rc is not None and abs(rb) > 1e-9
                    ]
                    if len(triples) < 4:
                        continue
                    # Error relativo de cada fila respecto al cociente col_a/col_b.
                    rel_err = [abs(rc - ra / rb) / max(abs(ra / rb), 1e-6) for ra, rb, rc in triples]
                    # Firma del bug (informe §1.5): la columna PRETENDE ser el
                    # cociente —la MAYORÍA de filas cuadran ajustadamente (<5%)— pero
                    # una minoría falla de forma clara (>30%, factores de 2-8). Esa
                    # coexistencia descarta el redondeo (error uniforme y pequeño) y
                    # que sea otra columna (error uniforme y grande): es
                    # media-de-cocientes frente a cociente-de-medias.
                    close = [e < 0.20 for e in rel_err]
                    broken = [e > 0.50 for e in rel_err]
                    n_close, n_broken = sum(close), sum(broken)
                    # Exigir que ~la mitad cuadre razonablemente (<20%) y que al menos
                    # una falle de forma catastrófica (>50%), con el error acotado a
                    # 20x (2000%) para descartar divisiones por valores diminutos.
                    # La exclusión previa de columnas con ceros (pos_cols) ya evita el
                    # grueso de los falsos positivos.
                    max_err = max(rel_err)
                    if n_close >= max(2, len(triples) // 2) and n_broken >= 1 and max_err < 20.0:
                        collector.add_inconsistency(
                            f"[{rel}:{start}] tabla (\\begin{{tabular}} en L{start}): la columna "
                            f"{c_idx + 1} parece el cociente col{a_idx + 1}/col{b_idx + 1} "
                            f"({n_close}/{len(triples)} filas cuadran <20\\%) pero {n_broken} se "
                            f"desvían >50\\% (máx {max_err * 100:.0f}\\%). La coexistencia descarta "
                            f"el redondeo: probable media de cocientes frente a cociente de medias, "
                            f"o unidad mal puesta. Recalcular con una única convención."
                        )

        # --- B) fila Total: suma de columnas numéricas vs total declarado.
        total_row = None
        for (ln, cells), r in zip(blk["raw_rows"], rows, strict=False):
            if cells and re.search(r"\btotal\b", cells[0], re.IGNORECASE):
                total_row = (ln, r)
        if total_row is not None:
            ln_tot, r_tot = total_row
            data_rows = [
                r
                for (_, cells), r in zip(blk["raw_rows"], rows, strict=False)
                if not (cells and re.search(r"\btotal\b", cells[0], re.IGNORECASE))
            ]
            for idx in range(ncol):
                colvals = [r[idx] for r in data_rows if idx < len(r) and r[idx] is not None]
                tot = r_tot[idx] if idx < len(r_tot) else None
                # solo columnas de enteros (conteos), ≥3 valores, total presente
                if tot is None or len(colvals) < 3:
                    continue
                if all(abs(v - round(v)) < 1e-6 for v in colvals) and abs(tot - round(tot)) < 1e-6:
                    suma = sum(colvals)
                    if abs(suma - tot) > 0.5:
                        collector.add_inconsistency(
                            f"[{rel}:{ln_tot}] fila 'Total' col {idx + 1}: la suma de las filas "
                            f"({suma:g}) no coincide con el total declarado ({tot:g})."
                        )

        # --- C) escala de calificación de la leyenda aplicada a su columna.
        scale = _GRADE_SCALE_RE.findall(blk["caption"] or "")
        if scale:
            # umbrales ordenados ascendente: [(letra, thr)]
            thr = sorted(((g.upper(), float(t.replace(",", "."))) for g, t in scale), key=lambda x: x[1])

            def grade_of(v: float) -> str:
                for g, t in thr:
                    if v < t:
                        return g
                return "F"  # por encima del último umbral

            # buscar una columna de letras (A-F) y una columna numérica adyacente
            letter_cols = []
            for idx in range(ncol):
                letters = 0
                for _, cells in blk["raw_rows"]:
                    if idx < len(cells):
                        c = re.sub(r"\\(textbf|emph|textit)\{([^}]*)\}", r"\2", cells[idx]).strip()
                        if re.fullmatch(r"[A-Fa-f]", c):
                            letters += 1
                if letters >= 3:
                    letter_cols.append(idx)
            for lc in letter_cols:
                for num_idx in range(ncol):
                    if num_idx == lc:
                        continue
                    mism = 0
                    checked = 0
                    for (_, cells), r in zip(blk["raw_rows"], rows, strict=False):
                        if lc >= len(cells) or num_idx >= len(r):
                            continue
                        letter = re.sub(r"\\(textbf|emph|textit)\{([^}]*)\}", r"\2", cells[lc]).strip().upper()
                        val = r[num_idx]
                        if not re.fullmatch(r"[A-F]", letter) or val is None:
                            continue
                        checked += 1
                        if grade_of(val) != letter:
                            mism += 1
                    # si TODAS las filas discrepan, la columna numérica no es la que
                    # rige la escala (probable "mejor punto" vs "media"): marcar.
                    if checked >= 3 and mism == checked:
                        collector.add_inconsistency(
                            f"[{rel}:{start}] tabla (L{start}): la columna de calificación "
                            f"(col {lc + 1}) no coincide con la escala de la leyenda aplicada "
                            f"a la col {num_idx + 1} en {mism}/{checked} filas. ¿Las dos columnas "
                            f"agregan cosas distintas (p. ej. mejor punto vs media)? Renombrar."
                        )


def _split_chapters(text: str) -> dict[str, str]:
    """Devuelve {titulo_capitulo: cuerpo} partiendo por \\chapter{...}."""
    parts = re.split(r"\\chapter\{([^}]+)\}", text)
    out: dict[str, str] = {}
    # parts = [pre, title1, body1, title2, body2, ...]
    for k in range(1, len(parts) - 1, 2):
        out[parts[k].strip()] = parts[k + 1]
    return out


def _load_canonical_tfim_matrix(collector: TodoCollector) -> dict[tuple[str, int], int]:
    """Carga la matriz canónica {(topo, p): pass_pct} del TFIM estándar desde la
    Sección 1 (Summary Table) de noiseless_v2_analysis.md, que es la fuente de
    verdad para las tasas cross-topología del cuerpo (régimen válido h>=1.3, /39).

    Devuelve {} si no puede leerse (el chequeo se salta silenciosamente).
    """
    src = ROOT / "internal" / "documentation" / "analysis" / "noiseless_v2_analysis.md"
    if not src.exists():
        return {}
    lines = src.read_text(encoding="utf-8").splitlines()
    # Delimitar la Sección 1: desde "## 1. TFIM Standard" hasta el próximo "## 2.".
    start = end = None
    for i, ln in enumerate(lines):
        if start is None and re.match(r"^##\s*1\.\s*TFIM", ln, re.IGNORECASE):
            start = i
        elif start is not None and re.match(r"^##\s*2\.", ln):
            end = i
            break
    if start is None:
        return {}
    block = lines[start : end or len(lines)]
    matrix: dict[tuple[str, int], int] = {}
    row_re = re.compile(
        r"^\|\s*(chain_1d|heavy_hex|ladder|square|triangular)\s*\|\s*([1-4])\s*\|"
        r".*?(\d+)%\s*\(\d+/39\)"
    )
    for ln in block:
        m = row_re.match(ln)
        if m:
            matrix[(m.group(1), int(m.group(2)))] = int(m.group(3))
    return matrix


# Nombre de topología en el .tex -> clave interna del store.
_TEX_TOPO_TO_KEY = {
    "cadena 1d": "chain_1d",
    "heavy-hex": "heavy_hex",
    "escalera": "ladder",
    "cuadrada": "square",
    "red cuadrada": "square",
    "triangular": "triangular",
}

# ── Convenios canónicos para los chequeos de cifras (steering §8, §16) ──────────
#
# Conteo de compuertas CZ del HVA: convenio de 2 CZ por término de interacción de
# dos cuerpos (cada R_ZZ se descompone en CZ–R_z–CZ → 2 CZ). Por capa p:
#   n_CZ(modelo, N, p) = p * 2 * (n_terminos_2cuerpos del modelo)
# donde el número de términos de dos cuerpos depende del modelo:
#   TFIM (cadena 1D, contorno abierto): N-1 enlaces          → 2(N-1) por capa
#   J1-J2 (frustrado 1D):               (N-1) + (N-2) enlaces → 2(2N-3) por capa
#   Kitaev (cadena):                    2(N-1) (XX + YY)      → 4(N-1) por capa
# A N=6, p=1 esto da 10 / 18 / 20 respectivamente (valores de referencia del plan).
CZ_TERMS_2BODY: dict[str, callable] = {
    "tfim": lambda N: (N - 1),
    "tfim_long": lambda N: (N - 1),  # el término longitudinal es de 1 cuerpo
    "j1-j2": lambda N: (2 * N - 3),
    "frustrado": lambda N: (2 * N - 3),
    "kitaev": lambda N: 2 * (N - 1),
}

# Las tres escalas de N canónicas (steering §16). Cualquier "N máximo" suelto que
# no sea una de estas y no declare a cuál escala pertenece es sospechoso.
CANONICAL_N_SCALES = (22, 40, 250)

# Malla de h de la Fase 1 (steering): 52 puntos en [0,5; 5,0]; de ellos 39 caen en
# el régimen válido [1,3; 5,0]. Denominadores de tasas /NN que no sean 39 (régimen
# válido) ni 52 (malla completa) en una leyenda de barrido son sospechosos.
H_GRID_TOTAL = 52
H_GRID_VALID = 39

# Acrónimos adicionales que el §4.3 del informe marcó sin desplegar en primer uso.
# Se fusionan con ACRONYMS para el chequeo editorial. Expansión canónica (EN) +
# palabras clave ES aceptadas (si las hubiera; aquí son siglas técnicas en inglés).
# 'DD' (Dynamical Decoupling) se omite a propósito: dos letras mayúsculas generan
# demasiados falsos positivos (aparece dentro de identificadores/otras siglas) y es
# contexto de hardware (trabajo futuro) mencionado una sola vez. Se despliega a mano.
ACRONYMS_EXTRA: dict[str, str] = {
    "DyPP": "Dynamic Parameter Prediction",
    "NLCE": "Numerical Linked-Cluster Expansion",
    "TREX": "Twirled Readout Error eXtinction",
    # 'PVLS' se omite: es el nombre propio del método de Yang et al. (2025), cuyo
    # artículo NO expande la sigla; no se le asigna una expansión inventada. En el
    # cuerpo se presenta como "el método de Yang et al. (denominado PVLS por sus
    # autores)", no como un acrónimo de glosario.
}

# ── Palabras clave (§4.2): reglas de las tres listas (pdfkeywords, Palabras clave,
#    Keywords). 4–6 términos, sin siglas, minúscula inicial salvo nombres propios.
KEYWORDS_MIN = 4
KEYWORDS_MAX = 6
# Nombres propios admitidos en mayúscula inicial dentro de una palabra clave.
KEYWORDS_PROPER_NOUNS = {"Ising"}

# ── Conteo total de la campaña (informe §1.4): la suma de la columna de ejecuciones
#    de la tabla de conteo por modelo debe coincidir con el total declarado en prosa.
#    El valor canónico se deriva de la tabla, no se hardcodea.
#    (chequeo _check_campaign_total)

# ── Nombres de modelo en clave de código que NO deben aparecer como texto de
#    tabla/prosa (informe §1.5.3). SOLO se listan identificadores INEQUÍVOCOS:
#    o llevan guion bajo (snake_case literal), o son combinaciones en INGLÉS que
#    nunca son la forma legible en español ('bond resolved', 'transverse',
#    'frustrated'). NO se incluyen palabras que también son nombres propios o forma
#    legible ('kitaev', 'heisenberg', 'tfim longitudinal') para evitar marcar
#    'Cadena de Kitaev' o 'TFIM longitudinal', que son correctos.
CODE_MODEL_NAMES: dict[str, str] = {
    # snake_case literal (inequívoco: solo aparece en volcados de código):
    "heisenberg_transverse": "Heisenberg transversal",
    "tfim_bond_resolved": "TFIM por enlace",
    "tfim_longitudinal": "TFIM longitudinal",
    "tfim_frustrated": "TFIM frustrado",
    # términos en inglés que nunca son la forma legible en español:
    "heisenberg transverse": "Heisenberg transversal",
    "tfim bond resolved": "TFIM por enlace",
    "tfim frustrated": "TFIM frustrado",
}

# ── Colisiones de notación (informe §3): un mismo símbolo con dos significados.
#    Se detecta el uso conflictivo por co-ocurrencia de dos patrones incompatibles.
#    (chequeo _check_notation_collisions)

# ── Índice de acrónimos (§4.3): cobertura. Una sigla usada al menos este nº de
#    veces en el cuerpo debe figurar en el índice; entradas del índice sin uso en
#    el cuerpo son "índice inflado".
ACRONYM_MIN_USES = 3
# Ruido a excluir del recuento de siglas del cuerpo: identificadores de objetivos/
# hipótesis (OE1, H3), numerales romanos, notación de puertas (CZ, RX, RZZ, XX, YY,
# ZZ), códigos internos, marcas/software (IBM, PyTorch, TeNPy) y nombres propios de
# métodos/arquitecturas que NO son acrónimos de glosario (se escriben desplegados o
# son nombres registrados): Flow-VQE, GNN-HVA, UnifiedMPNN, Qracle, PC1, etc.
ACRONYM_NOISE_RE = re.compile(
    r"^(OE\d+|H\d+|[IVXLC]+|CZ|RX|RY|RZ|RZZ|RXX|XX|YY|ZZ|XXZ|SU|"
    r"J1|J2|1D|2D|3D|MB|SPSA|BFGS|API|CPU|GPU|SO|PDF|URL|HTML|HTTP|ID|"
    r"UNIR|TFM|CFT|SVD|"
    # marcas y software (nombres propios, no acrónimos):
    r"IBM|PyTorch|TeNPy|Qiskit|NumPy|SciPy|Aer|StatevectorEstimator|COBYLA|GINConv|BatchNorm|"
    # nombres de métodos/arquitecturas propios (se citan desplegados):
    r"UnifiedMPNN|Qracle|PVLS|PC\d+|"
    # cualquier token con guion que combine dos siglas ya glosadas (Flow-VQE,
    # GNN-HVA): son nombres compuestos, no una entrada de glosario nueva:
    r".*-.*)$"
)

# ── Estilo de referencias cruzadas (§5, informe): la plantilla escribe "Sección xx"
#    y "Ecuación xx". Conviven a mano tres estilos ("Ec. 3.3", "§5.4.2"). El estilo
#    canónico usa \ref/\eqref (o "Sección"/"Ecuación" completos), nunca las formas
#    abreviadas literales en prosa. Estos patrones marcan las formas NO canónicas
#    escritas a mano (fuera de math y de comandos), para unificar.
_XREF_ABBREV_RES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "Ec. abreviada",
        re.compile(r"\bEc\.\s*~?\s*\d", re.IGNORECASE),
        "usar 'Ecuación~\\ref{...}' (o \\eqref); la plantilla escribe 'Ecuación xx'",
    ),
    (
        "seccion §",
        re.compile(r"§\s*\d"),
        "usar 'Sección~\\ref{...}'; la plantilla escribe 'Sección xx', no '§xx'",
    ),
    (
        "Ecuacion literal",
        re.compile(r"\bEcuaci[oó]n\s+\d"),
        "referenciar con 'Ecuación~\\ref{...}' en vez de un número literal",
    ),
    (
        "Seccion literal",
        re.compile(r"\bSecci[oó]n\s+\d"),
        "referenciar con 'Sección~\\ref{...}' en vez de un número literal",
    ),
)

# ── "energy units" / "unidades de energía" como unidad de un eje o columna (§4.5):
#    no es una unidad; la unidad es J = 1. Se marca el literal en prosa/figuras.
_ENERGY_UNITS_RE = re.compile(r"energy\s+units|unidades\s+de\s+energ[ií]a", re.IGNORECASE)

# ── Coeficiente de la entropía en el punto crítico (§1.10, informe): la CFT da
#    S ~ (c/6) log N (contorno abierto) o (c/3) (periódico), NUNCA "6 c log N".
#    Centinela: detecta el coeficiente invertido por si reaparece.
_ENTROPY_BADCOEF_RE = re.compile(r"6\s*c\s*\\?log|6c\s*\\?log")

# ── Intervalos con coma como separador en modo matemático (§1.9a): "[1,3, 3,0]"
#    se lee como una lista de cuatro números. La forma correcta usa punto y coma:
#    "[1{,}3;\, 3{,}0]". Este patrón detecta, dentro de $...$, un corchete con dos
#    pares decimal-coma separados por coma (no por ';'). Complementa el chequeo de
#    tablas (_check_h_range_coherence) extendiéndolo a TODA la prosa matemática.
_MATH_COMMA_INTERVAL_RE = re.compile(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]")


def _check_regime_consistency(text: str, rel: str, collector: TodoCollector) -> None:
    """(23) Consistencia de valores entre tablas y contra la fuente de verdad.

    Dos chequeos que evitan que las tablas del cuerpo se desincronicen de los
    datos (como ocurrió con la Tabla cross_topo_depth inflada):

    (a) Contra fuente: compara cada (topología, p) de la Tabla cross_topo_depth
        (régimen crítico TFIM estándar) con la matriz canónica de la Sección 1 de
        noiseless_v2_analysis.md. Señala desviaciones > 2 puntos porcentuales.
    (b) Inter-tabla: la mejor p de cross_topo debe coincidir con el máximo de esa
        fila en cross_topo_depth (misma topología, mismo régimen).

    Ambos son señales (no fixes): requieren criterio humano, pero atrapan el drift.
    """
    canonical = _load_canonical_tfim_matrix(collector)
    if not canonical:
        # No silenciar: dejar constancia de que el cruce tabla<->fuente NO se hizo,
        # para que un operador no confunda "sin hallazgos" con "chequeo omitido".
        collector.add_inconsistency(
            "INFO: chequeo tabla<->fuente canónica OMITIDO: no se pudo leer la matriz "
            "TFIM de noiseless_v2_analysis.md (Secc.1). Las tasas cross_topo/"
            "cross_topo_depth NO se validaron contra la fuente de verdad."
        )
        return

    # Helpers compartidos a nivel de módulo (parseo de tabla por label, % y topo).
    _rows_of = lambda label: _table_rows_by_label(text, label)  # noqa: E731
    _pct = _pct_of_cell
    _topo_key = _topo_key_of_cell

    # (a) cross_topo_depth vs fuente canónica.
    depth_rows = _rows_of("tab:cross_topo_depth")
    for row in depth_rows:
        if not row:
            continue
        topo = _topo_key(row[0])
        if topo is None or len(row) < 5:
            continue
        for p, cell in zip((1, 2, 3, 4), row[1:5], strict=False):
            pct = _pct(cell)
            if pct is None:  # '---' (no medido) es válido
                continue
            ref = canonical.get((topo, p))
            if ref is not None and abs(pct - ref) > 2:
                collector.add_inconsistency(
                    f"[{rel}] Tabla cross_topo_depth: {topo} p={p} = {pct}\\% "
                    f"discrepa de la fuente canónica (noiseless_v2 Secc.1: {ref}\\%). "
                    f"Corregir la tabla contra la fuente (steering §1)."
                )

    # (b) cross_topo (mejor p) vs máximo de cross_topo_depth.
    depth_by_topo: dict[str, dict[int, int]] = defaultdict(dict)
    for row in depth_rows:
        topo = _topo_key(row[0]) if row else None
        if topo is None or len(row) < 5:
            continue
        for p, cell in zip((1, 2, 3, 4), row[1:5], strict=False):
            pct = _pct(cell)
            if pct is not None:
                depth_by_topo[topo][p] = pct
    for row in _rows_of("tab:cross_topo"):
        topo = _topo_key(row[0]) if row else None
        if topo is None or len(row) < 3:
            continue
        # row[1] = mejor p (entero), row[2] = tasa
        try:
            best_p_val = int(re.sub(r"\D", "", row[1]))
        except ValueError:
            continue
        rate = _pct(row[2])
        depth = depth_by_topo.get(topo, {})
        if depth and rate is not None:
            max_depth = max(depth.values())
            if rate < max_depth - 2:
                collector.add_inconsistency(
                    f"[{rel}] Tabla cross_topo: {topo} 'mejor p={best_p_val}' da "
                    f"{rate}\\%, pero cross_topo_depth tiene una p con {max_depth}\\% "
                    f"para esa topología. Verificar que 'mejor' esté bien elegida "
                    f"(o justificar el criterio de selección)."
                )

    # (c) Coherencia de los rangos de h declarados por tabla con su régimen.
    _check_h_range_coherence(text, rel, collector)

    # (d) Régimen de extrapolación: fuentes vivas al día y tablas auto conectadas.
    _check_extrapolation_sources(text, rel, collector)

    # (e) Acuerdo texto <-> celda de tabla: el texto no debe afirmar un % para una
    # (topología, mejor p) que la tabla cross_topo contradice. Conservador.
    _check_text_table_agreement(text, rel, collector)


# Rango de h en captions: [a, b] o [a; b] con coma decimal española.
_H_RANGE_RE = re.compile(r"h\s*\\in\s*\[\s*(\d+(?:[,.]\d+)?)\s*([,;])\s*(\d+(?:[,.]\d+)?)\s*\]")


def _check_h_range_coherence(text: str, rel: str, collector: TodoCollector) -> None:
    """(23c) Cada tabla con régimen conocido debe declarar un rango de h dentro de
    los límites canónicos de ese régimen, y con la convención de separador correcta
    (punto y coma, no coma, para no confundir con lista — steering §1.9/§11).

    Señal, no fix: reporta rango fuera del régimen y coma como separador de intervalo.
    """
    for label, (regime, lo, hi) in H_REGIME_BY_LABEL.items():
        lab_pos = text.find("\\label{" + label + "}")
        if lab_pos < 0:
            continue
        # Buscar el \caption del entorno table que contiene el label.
        begin = text.rfind("\\begin{table}", 0, lab_pos)
        end = text.find("\\end{table}", lab_pos)
        if begin < 0 or end < 0:
            continue
        block = text[begin:end]
        m = _H_RANGE_RE.search(block)
        if not m:
            continue  # no declara rango explícito de h (no bloqueante)
        a = float(m.group(1).replace(",", "."))
        sep = m.group(2)
        b = float(m.group(3).replace(",", "."))
        if sep == ",":
            collector.add_inconsistency(
                f"[{rel}] Tabla {label}: intervalo de h con coma separadora "
                f"('$h \\in [{m.group(1)}, {m.group(3)}]$') se lee como lista de 4 "
                f"números; usar punto y coma ('[{m.group(1)}; {m.group(3)}]'), §1.9."
            )
        # Tolerancia de 0.01 para bordes; fuera del régimen => señal.
        if a < lo - 0.01 or b > hi + 0.01:
            collector.add_inconsistency(
                f"[{rel}] Tabla {label} (régimen {regime}): rango de h declarado "
                f"[{a}; {b}] excede los límites canónicos del régimen "
                f"[{lo}; {hi}]. Ajustar el rango o reclasificar la tabla (§ grilla h)."
            )


def _check_extrapolation_sources(text: str, rel: str, collector: TodoCollector) -> None:
    """(23d) Régimen de extrapolación: las tablas se generan desde fuentes vivas
    (scoreboard JSON, eval reports). Este chequeo verifica que esas fuentes estén
    al día y que las tablas auto estén efectivamente conectadas al documento, de
    modo que un scoreboard viejo o un \\input faltante no pasen inadvertidos.

    Señal, no fix.
    """
    import os

    # (d1) Staleness del scoreboard JSON respecto a los eval reports.
    if SCOREBOARD_JSON.exists():
        eval_dir = ROOT / "results" / "extrapolation_evals"
        evs = list(eval_dir.glob("*_p*/eval_*.md"))
        if evs:
            newest = max(os.path.getmtime(e) for e in evs)
            if os.path.getmtime(SCOREBOARD_JSON) < newest - 1:
                collector.add_inconsistency(
                    f"scoreboard JSON ({SCOREBOARD_JSON.name}) es más antiguo que el "
                    f"eval report más reciente: regenerar con "
                    f"generate_best_results_scoreboard.py --json antes de compilar "
                    f"las tablas de extrapolación (evita valores stale)."
                )

    # (d2) Tablas auto del régimen de extrapolación conectadas con \input.
    extrap_auto = [
        lab for lab, (reg, *_) in H_REGIME_BY_LABEL.items() if reg == "extrapolacion" and lab.startswith("tab:auto_")
    ]
    for lab in extrap_auto:
        stem = lab[len("tab:") :]
        if f"\\input{{tables/{stem}}}" not in text:
            collector.add_inconsistency(
                f"[{rel}] tabla de extrapolación auto '{stem}' no está incluida con "
                f"\\input{{tables/{stem}}}: la tesis mostraría datos desactualizados "
                f"o vacíos. Añadir el \\input o regenerar."
            )


# ── Iteración por bloques de sección (infraestructura reutilizable). ────────────
# Segmenta el .tex en bloques delimitados por \chapter/\section/\subsection y
# expone, por bloque, su título, rango de líneas, texto y las \ref que contiene.
# Permite escribir chequeos "por bloque" con contexto local (p. ej. validar una
# afirmación contra las tablas/figuras que se citan en la MISMA sección), en lugar
# de barrer todo el documento con una ventana de caracteres fija.


class DocBlock:
    """Un bloque de sección del documento: título, nivel, líneas y contenido."""

    __slots__ = ("title", "level", "start_line", "end_line", "text", "refs")

    def __init__(self, title: str, level: str, start_line: int, end_line: int, text: str, refs: list[str]) -> None:
        self.title = title
        self.level = level  # 'chapter' | 'section' | 'subsection'
        self.start_line = start_line
        self.end_line = end_line
        self.text = text
        self.refs = refs  # labels citados con \ref dentro del bloque


_SECTION_RE = re.compile(r"\\(chapter|section|subsection)\*?\{((?:[^{}]|\{[^{}]*\})*)\}")


def _iter_doc_blocks(text: str):
    """Genera DocBlock por cada \\chapter/\\section/\\subsection del cuerpo.

    Un bloque abarca desde su encabezado hasta el siguiente encabezado del mismo
    nivel o superior (o el fin del documento). El texto anterior al primer
    encabezado (preámbulo) se omite. Ignora comandos dentro de comentarios de línea.
    """
    lines = text.splitlines()
    # Localizar encabezados con su nº de línea (1-indexed), saltando comentarios.
    heads: list[tuple[int, str, str]] = []  # (line_no, level, title)
    for i, ln in enumerate(lines, start=1):
        code = re.sub(r"(?<!\\)%.*$", "", ln)
        m = _SECTION_RE.search(code)
        if m:
            heads.append((i, m.group(1), m.group(2).strip()))
    for idx, (ln_no, level, title) in enumerate(heads):
        end = heads[idx + 1][0] - 1 if idx + 1 < len(heads) else len(lines)
        block_text = "\n".join(lines[ln_no - 1 : end])
        refs = re.findall(r"\\ref\{([^}]+)\}", block_text)
        yield DocBlock(title, level, ln_no, end, block_text, refs)


def _run_block_checks(text: str, rel: str, collector: TodoCollector, block_checks: tuple) -> None:
    """Ejecuta una tanda de chequeos POR BLOQUE, aislando fallos por bloque+chequeo.

    Cada chequeo de bloque tiene la firma check(block: DocBlock, rel, collector).
    Un fallo en un bloque no impide procesar los demás (mismo criterio de guard que
    _run_check_guarded).
    """
    for block in _iter_doc_blocks(text):
        for check in block_checks:
            try:
                check(block, rel, collector)
            except Exception as exc:  # noqa: BLE001
                collector.add_inconsistency(
                    f"[{rel}] chequeo de bloque "
                    f"'{getattr(check, '__name__', str(check))}' falló en la sección "
                    f"'{block.title[:40]}' con {type(exc).__name__}: {exc}."
                )


# ── Consistencia tabla auto ↔ prosa (informe v3 §1.1: el texto afirma "100%" o
#    "|ΔE| constante" mientras la tabla contigua lo desmiente). ──────────────────

# Umbral de clasificación del criterio operativo (ΔE/gap < 5 %). En columnas de
# porcentaje o cociente, un valor por encima significa "no aprueba".
_PASS_THRESHOLD_PCT = 5.0
# Factor a partir del cual una columna |ΔE| NO puede calificarse de "constante".
_CONST_MAX_RATIO = 3.0


def _parse_auto_table(path: Path) -> dict | None:
    r"""Parsea un archivo auto_*.tex (booktabs) y devuelve label, encabezados y
    filas numéricas por columna.

    Devuelve None si el archivo no existe o no tiene una tabla reconocible.
    Estructura: {"label": str, "headers": [str], "cols": {header: [float|None]},
    "n_values": [int]}. Los valores no numéricos quedan como None.
    """
    if not path.exists():
        return None
    src = path.read_text(encoding="utf-8")
    m_lab = re.search(r"\\label\{([^}]+)\}", src)
    label = m_lab.group(1) if m_lab else path.stem
    # Cuerpo entre \midrule y \bottomrule (las filas de datos).
    m_body = re.search(r"\\midrule(.*?)\\bottomrule", src, re.DOTALL)
    if not m_body:
        return None
    # Encabezados: la fila entre \toprule y \midrule.
    m_head = re.search(r"\\toprule(.*?)\\midrule", src, re.DOTALL)
    headers: list[str] = []
    if m_head:
        head_line = m_head.group(1).strip().rstrip("\\").strip()
        headers = [h.strip() for h in head_line.split("&")]

    def _num_or_none(cell: str) -> float | None:
        # Extrae el primer número (coma o punto decimal) de la celda; None si no hay.
        c = cell.replace("\\textbf{", "").replace("}", "").strip()
        mm = re.search(r"-?\d+(?:[.,]\d+)?", c)
        if not mm:
            return None
        return float(mm.group(0).replace(",", "."))

    rows: list[list[str]] = []
    for raw in m_body.group(1).split("\\\\"):
        raw = raw.strip()
        if not raw or raw.startswith("%") or "\\midrule" in raw:
            continue
        cells = [c.strip() for c in raw.split("&")]
        if len(cells) >= 2:
            rows.append(cells)

    n_cols = len(headers) if headers else (max((len(r) for r in rows), default=0))
    cols: dict[str, list] = {}
    for j in range(n_cols):
        key = headers[j] if j < len(headers) else f"col{j}"
        cols[key] = [_num_or_none(r[j]) if j < len(r) else None for r in rows]

    # Detectar la columna de N (encabezado que contiene 'N' aislado).
    n_values: list[int] = []
    for key, vals in cols.items():
        if re.search(r"(^|[^A-Za-z])N([^A-Za-z]|$)", key) and "máx" not in key:
            n_values = [int(v) for v in vals if v is not None]
            break
    return {"label": label, "headers": headers, "cols": cols, "n_values": n_values}


def _prose_referencing(text: str, label: str, window: int = 900) -> str:
    """Devuelve la prosa alrededor de cada \\ref{label} (ventana de caracteres a
    ambos lados), concatenada. Sirve para inspeccionar lo que el texto afirma
    sobre una tabla concreta.
    """
    chunks: list[str] = []
    for m in re.finditer(r"\\ref\{" + re.escape(label) + r"\}", text):
        lo = max(0, m.start() - window)
        hi = min(len(text), m.end() + window)
        chunks.append(text[lo:hi])
    return "\n".join(chunks)


def _check_auto_table_text_consistency(text: str, out_dir: Path, rel: str, collector: TodoCollector) -> None:
    r"""(Nuevo) Coherencia entre las tablas AUTO-GENERADAS y la prosa que las cita.

    Es el chequeo que atrapa la clase de contradicción del informe v3 §1.1: el texto
    afirma "100 %" o "|ΔE| permanece constante" mientras la tabla contigua (auto_*.tex)
    muestra puntos que fallan o un |ΔE| que varía por un factor grande. Trabaja sobre
    los archivos generados en out_dir (números finales renderizados) y la prosa que
    los referencia con \\ref, de modo que detecta el desajuste con datos, no por reglas
    fijas.

    Señal, no fix: la reescritura del texto es criterio humano.

    Delega la detección en _flag_table_prose_contradictions, que también reutiliza
    la variante por bloque (_check_block_table_consistency) para dar localidad de
    sección al hallazgo.
    """
    for auto_path in sorted(out_dir.glob("auto_*.tex")):
        parsed = _parse_auto_table(auto_path)
        if not parsed:
            continue
        prose = _prose_referencing(text, parsed["label"])
        if not prose:
            continue  # la tabla no se cita: cubierto por otro chequeo
        _flag_table_prose_contradictions(prose, parsed, auto_path, rel, collector, scope="")


def _flag_table_prose_contradictions(
    prose: str,
    parsed: dict,
    auto_path: Path,
    rel: str,
    collector: TodoCollector,
    scope: str = "",
) -> None:
    r"""Núcleo reutilizable: contrasta la prosa dada contra las celdas de una tabla
    auto ya parseada, y registra las contradicciones (100 %/constante/rango-N).

    ``scope`` es un sufijo de localización opcional (p. ej. " (sección 5.3)") para
    que el hallazgo indique en qué bloque se detectó cuando se llama por sección.
    """
    label = parsed["label"]
    cols = parsed["cols"]
    de_col = next((v for k, v in cols.items() if "Delta E" in k and "gap" not in k), None)
    # Columna de cociente ΔE/gap: contiene 'gap' Y ('/' o '%'); distinta del gap solo.
    ratio_col = next((v for k, v in cols.items() if "gap" in k and ("/" in k or "%" in k or "mathrm{gap}" in k)), None)

    # (1) "100 %"/"todos los puntos" cerca de una tabla con fallos.
    claims_all_pass = bool(re.search(r"100\s*\\?%|todos los puntos", prose, re.IGNORECASE))
    table_has_failures = False
    if ratio_col is not None:
        table_has_failures = any(v is not None and v > _PASS_THRESHOLD_PCT for v in ratio_col)
    frac_fail = re.search(r"\b([0-9])\s*/\s*([0-9])\b", auto_path.read_text(encoding="utf-8"))
    if frac_fail and frac_fail.group(1) != frac_fail.group(2):
        table_has_failures = True
    if claims_all_pass and table_has_failures:
        worst = max((v for v in (ratio_col or []) if v is not None), default=None)
        collector.add_inconsistency(
            f"[{rel}] la prosa que cita \\ref{{{label}}}{scope} afirma '100 %'/'todos "
            f"los puntos' pero la tabla {auto_path.name} tiene puntos que superan el "
            f"umbral del {_PASS_THRESHOLD_PCT:g} %"
            + (f" (máx {worst:g} %)" if worst is not None else "")
            + ". Ajustar la afirmación a los conteos reales de la tabla (§1.1)."
        )

    # (2) "|ΔE| constante/estable/independiente de N" con |ΔE| que varía.
    claims_const = bool(
        re.search(r"\|?\\?Delta E\|?[^.]{0,80}?(constante|estable|independiente de \$?N)", prose, re.IGNORECASE)
        or re.search(r"(constante|estable|independiente de \$?N)[^.]{0,60}?\|?\\?Delta E", prose, re.IGNORECASE)
    )
    if claims_const and de_col is not None:
        vals = [v for v in de_col if v is not None and v > 0]
        if len(vals) >= 2 and max(vals) / min(vals) > _CONST_MAX_RATIO:
            ratio = max(vals) / min(vals)
            collector.add_inconsistency(
                f"[{rel}] la prosa que cita \\ref{{{label}}}{scope} describe |ΔE| como "
                f"'constante/estable' pero la columna |ΔE| de {auto_path.name} varía por "
                f"un factor {ratio:.1f} (de {min(vals):g} a {max(vals):g}). Matizar la "
                f"afirmación o acotarla al subconjunto donde sí es estable (§1.1c/§2.5)."
            )

    # (3) Rango 'N = a--b' afirmado en prosa que la tabla no cubre.
    if parsed["n_values"]:
        tset = set(parsed["n_values"])
        for mr in re.finditer(r"N\s*=\s*(\d{1,3})\s*(?:--|–|-|a)\s*(\d{1,3})", prose):
            a, b = int(mr.group(1)), int(mr.group(2))
            if not any(a <= n <= b for n in tset):
                collector.add_inconsistency(
                    f"[{rel}] la prosa que cita \\ref{{{label}}}{scope} habla de "
                    f"'N = {a}--{b}', pero la tabla {auto_path.name} solo evalúa "
                    f"N ∈ {sorted(tset)}: ninguna fila cae en ese rango. Alinear el "
                    f"rango del texto con los tamaños tabulados (§1.1)."
                )


def _make_block_table_consistency_check(out_dir: Path):
    r"""Fábrica de un chequeo POR BLOQUE que valida, para cada sección, las tablas
    auto que la sección cita con \\ref contra su prosa local.

    A diferencia de la variante global (ventana de caracteres fija), aquí la "prosa"
    es exactamente el texto de la sección que contiene el \\ref, de modo que el
    hallazgo queda anclado a la sección y no se contamina con prosa de secciones
    vecinas. Devuelve una función con la firma de chequeo de bloque.
    """
    # Cachear el parseo de las tablas auto (una vez, no por bloque).
    parsed_by_label: dict[str, tuple] = {}
    for p in sorted(out_dir.glob("auto_*.tex")):
        pr = _parse_auto_table(p)
        if pr:
            parsed_by_label[pr["label"]] = (pr, p)

    def _check(block: DocBlock, rel: str, collector: TodoCollector) -> None:
        # Para no atribuir a una tabla afirmaciones que pertenecen a OTRA tabla
        # citada en la misma sección (p. ej. un "100 %" de cross_n_transfer junto a
        # la tabla intra_n), se acota la prosa a la vecindad del \ref de CADA tabla:
        # el párrafo que contiene el \ref más el párrafo inmediatamente anterior
        # (donde suele estar la afirmación que la tabla ilustra).
        paras = re.split(r"\n\s*\n", block.text)
        for label in set(block.refs):
            if label not in parsed_by_label:
                continue
            parsed, path = parsed_by_label[label]
            ref_token = f"\\ref{{{label}}}"
            local_chunks: list[str] = []
            for i, para in enumerate(paras):
                if ref_token in para:
                    prev = paras[i - 1] if i > 0 else ""
                    local_chunks.append(prev + "\n" + para)
            if not local_chunks:
                continue
            local_prose = "\n".join(local_chunks)
            scope = f" (sección «{block.title[:40]}»)"
            _flag_table_prose_contradictions(local_prose, parsed, path, rel, collector, scope=scope)

    _check.__name__ = "_check_block_table_consistency"
    return _check


def _check_text_table_agreement(text: str, rel: str, collector: TodoCollector) -> None:
    """(23e) Acuerdo texto <-> celda de tabla (CONSERVADOR).

    Toma la Tabla cross_topo (tasa por topología en su mejor p, TFIM estándar) como
    referencia y busca en la prosa afirmaciones que asocien la MISMA topología con
    un porcentaje DISTINTO en un contexto inequívoco de TFIM estándar. Este es el
    chequeo que habría atrapado la Tabla cross_topo_depth inflada (texto decía
    "100% a p=4" mientras la tabla real daba 92%).

    Reglas conservadoras para no generar falsos positivos:
    - Solo se activa en frases que mencionan "TFIM estándar" o "TFIM estandar"
      (excluye longitudinal, cross-N, extensibilidad, que tienen otras tasas).
    - La topología y el % deben estar en la misma frase y a <= 60 caracteres.
    - Se ignora si la frase también menciona otra tabla por \\ref (remite a dato ajeno).
    - Umbral de discrepancia: > 2 puntos porcentuales.

    Dominio: frases simples e inequívocas (una topología, un %, TFIM estándar). Las
    frases que citan varias tablas (p. ej. la "aclaración sobre las tasas") se saltan
    a propósito; esos casos ya los cubre el chequeo (a) tabla-vs-fuente-canónica.
    """
    # Referencia: {topo_key: tasa%} de la mejor config (Tabla cross_topo).
    ref_rate: dict[str, int] = {}
    for row in _table_rows_by_label(text, "tab:cross_topo"):
        if not row or len(row) < 5:
            continue
        topo = _topo_key_of_cell(row[0])
        if topo is None:
            continue
        # La tasa está en la columna 'Tasa aprob.' (primera celda con formato NN%).
        for cell in row[1:]:
            pct = _pct_of_cell(cell)
            if pct is not None:
                ref_rate[topo] = pct
                break
    if not ref_rate:
        # Distinguir "tabla ausente" (legítimo, no se avisa) de "tabla presente
        # pero sin tasas parseables" (sospechoso: el parseo falló y el chequeo
        # texto<->tabla quedó inactivo sin que nadie lo note).
        if _table_rows_by_label(text, "tab:cross_topo"):
            collector.add_inconsistency(
                "INFO: chequeo texto<->tabla (cross_topo) OMITIDO: la tabla existe "
                "pero no se pudo extraer ninguna tasa por topología; el acuerdo "
                "prosa<->celda NO se validó. Revisar el formato de la tabla."
            )
        return

    # Nombres de topología tal como aparecen en prosa (para localizarlos).
    topo_names = {
        "chain_1d": r"cadena 1D",
        "heavy_hex": r"heavy-hex",
        "ladder": r"escalera",
        "square": r"cuadrada",
        "triangular": r"triangular",
    }
    # Recorrer frases (split por punto que no sea decimal ni de comando).
    frases = re.split(r"(?<=[a-z)])\.\s+", text)
    for frase in frases:
        low = frase.lower()
        if "tfim estándar" not in low and "tfim estandar" not in low:
            continue
        # Si la frase remite a otra tabla, es probable que cite un dato ajeno: saltar.
        # (excepto cross_topo/cross_topo_depth que son las de TFIM estándar).
        otras_refs = [
            r for r in re.findall(r"\\ref\{tab:([a-z_0-9]+)\}", frase) if r not in ("cross_topo", "cross_topo_depth")
        ]
        if otras_refs:
            continue
        for topo, name in topo_names.items():
            if topo not in ref_rate:
                continue
            for tm in re.finditer(re.escape(name), frase):
                # Buscar un % en una ventana de +-60 chars alrededor del nombre.
                win = frase[max(0, tm.start() - 60) : tm.end() + 60]
                for pm in re.finditer(r"(\d+)\s*\\?%", win):
                    pct = int(pm.group(1))
                    ref = ref_rate[topo]
                    if abs(pct - ref) > 2:
                        collector.add_inconsistency(
                            f"[{rel}] Texto afirma {pct}\\% para {name} (TFIM estándar) "
                            f"pero la Tabla cross_topo da {ref}\\% en su mejor "
                            f"configuración. Verificar coherencia texto<->tabla "
                            f"(steering §1); posible cifra de otra campaña."
                        )
                        break


def _table_rows_by_label(text: str, label: str) -> list[list[str]]:
    """Devuelve las filas (listas de celdas de texto) de la tabla con ese \\label.

    Aísla el entorno table mínimo que contiene el label (anclando al \\begin{table}
    más cercano hacia atrás) y separa filas por '\\\\' y celdas por '&'. Helper único
    de parseo de tablas por label, reutilizado por todos los chequeos tabla<->texto
    (antes había tres copias de esta lógica).
    """
    lab_pos = text.find("\\label{" + label + "}")
    if lab_pos < 0:
        return []
    begin = text.rfind("\\begin{table}", 0, lab_pos)
    end = text.find("\\end{table}", lab_pos)
    if begin < 0 or end < 0:
        return []
    block = text[begin : end + len("\\end{table}")]
    rows: list[list[str]] = []
    for ln in block.splitlines():
        if "&" in ln and "\\\\" in ln and "rule" not in ln and "hline" not in ln:
            cells = [c.strip() for c in ln.split("\\\\")[0].split("&")]
            rows.append(cells)
    return rows


def _pct_of_cell(cell: str) -> int | None:
    """Extrae el porcentaje entero de una celda ('92\\%' -> 92); None si no hay."""
    m = re.search(r"(\d+)\s*\\?%", cell)
    return int(m.group(1)) if m else None


def _topo_key_of_cell(cell: str) -> str | None:
    """Mapea una celda con nombre de topología a su clave interna; None si no casa."""
    clean = re.sub(r"\\textbf\{|\\emph\{|\}", "", cell).strip().lower()
    return _TEX_TOPO_TO_KEY.get(clean)


def _check_text_vs_table(text: str, rel: str, collector: TodoCollector) -> None:
    """(Chequeo 2) Coherencia prosa ↔ celda de tabla para tasas por topología.

    Es el hueco que dejó pasar una tabla inflada: el texto afirma un porcentaje para
    una (topología) que la tabla contradice. Para cada tabla con \\label conocido que
    reporte tasas por topología, se extrae {topo: {p: pct}} y se busca en el MISMO
    capítulo menciones "topología ... p=P ... NN\\%"; si el texto afirma un % que la
    tabla no respalda para esa (topo, p), se marca. Heurístico y conservador: solo
    dispara cuando texto y tabla hablan de la misma (topo, p) con cifras distintas.
    """
    # Tablas candidatas: las que tienen tasas por topología y profundidad.
    for label in ("tab:cross_topo_depth",):
        rows = _table_rows_by_label(text, label)
        if not rows:
            continue
        # Construir {topo: {p: pct}} desde la tabla (columnas p=1..4).
        table_vals: dict[str, dict[int, int]] = {}
        for row in rows:
            if len(row) < 5:
                continue
            topo = _topo_key_of_cell(row[0])
            if topo is None:
                continue
            for p, cell in zip((1, 2, 3, 4), row[1:5], strict=False):
                pct = _pct_of_cell(cell)
                if pct is not None:
                    table_vals.setdefault(topo, {})[p] = pct
        if not table_vals:
            continue

        # Buscar en la prosa "topología ... p=P ... NN%" (o "p=P ... topología ... NN%").
        # Localiza el capítulo que contiene la tabla para acotar el cruce.
        chapters = _split_chapters(text)
        lab_pos = text.find("\\label{" + label + "}")
        chap_body = text  # fallback: todo el texto
        acc = 0
        for _title, body in chapters.items():
            acc2 = text.find(body, acc)
            if acc2 <= lab_pos <= acc2 + len(body):
                chap_body = body
                break
            acc = acc2 + len(body) if acc2 >= 0 else acc

        prose = _strip_protected(re.sub(r"(?<!\\)%.*$", "", chap_body))
        # topo ... p=P ... NN%  dentro de una ventana corta (misma cláusula)
        pat = re.compile(
            r"(cadena 1D|heavy-hex|escalera|cuadrada|triangular)"
            r"(.{0,60}?)p\s*=\s*(\d)(.{0,40}?)(\d{2,3})\\%",
            re.IGNORECASE,
        )
        for m in pat.finditer(prose):
            topo = _TEX_TOPO_TO_KEY.get(m.group(1).lower())
            if topo is None:
                continue
            p = int(m.group(3))
            claimed = int(m.group(5))
            window = (m.group(2) + m.group(4)).lower()
            # Solo si el % huele a tasa de aprobación (no gap/error/fidelidad).
            if not re.search(r"aprob|pasan|tasa", window):
                continue
            if re.search(r"gap|error|fidelidad|umbral|[<>]", window):
                continue
            table_pct = table_vals.get(topo, {}).get(p)
            if table_pct is not None and abs(claimed - table_pct) > 2:
                collector.add_inconsistency(
                    f"[{rel}] %TODO-CIFRA texto afirma {topo} p={p} = {claimed}\\% "
                    f"pero la Tabla ({label}) dice {table_pct}\\%; "
                    "reconciliar prosa y tabla (steering §1)."
                )


def _check_unsupported_claim(text: str, rel: str, collector: TodoCollector) -> None:
    """(Informe §1.1) Afirmación de '100%' para N=30–60 sin tabla que la respalde.

    Detecta la coincidencia de '100\\%' con un rango o conjunto de N grandes
    ({30,40,50,60} o 30-60) en una cláusula de 'aprobación/interpolación', y señala
    que debe existir una tabla con esas filas. Es señal: no puede verificar la tabla
    exacta, pero marca la afirmación repetida para que se ancle a datos. Conservador:
    solo dispara cuando aparecen a la vez el 100%, el rango N grande y 'aprob/evalua'.
    """
    prose = _strip_comments(text)
    claim_re = re.compile(
        r"100\s*\\?%[^.]{0,120}?(?:N\s*=\s*30[^.]{0,10}?60|"
        r"30\s*(?:--|–|-|a)\s*60|\{\s*30\s*,\s*40\s*,\s*50\s*,\s*60\s*\})|"
        r"(?:N\s*=\s*30[^.]{0,10}?60|30\s*(?:--|–|-|a)\s*60|"
        r"\{\s*30\s*,\s*40\s*,\s*50\s*,\s*60\s*\})[^.]{0,120}?100\s*\\?%",
        re.IGNORECASE,
    )
    seen = 0
    for m in claim_re.finditer(prose):
        # Contexto ampliado (±60 chars) porque el match mínimo puede no incluir
        # la señal 'aprob/puntos' (el cuantificador lazy para al primer cierre).
        ctx = prose[max(0, m.start() - 60) : m.end() + 60].lower()
        if not re.search(r"aprob|evalua|interpolaci|puntos", ctx):
            continue
        seen += 1
    if seen >= 2:
        collector.add_inconsistency(
            f"[{rel}] %TODO-CIFRA la afirmación '100\\% de aprobación para N=30–60' "
            f"aparece {seen} veces en prosa; verificar que exista una tabla con filas "
            "N∈{30,40,50,60}, nº de puntos, |ΔE| y tasa con su caso absoluto que la "
            "respalde (informe §1.1). Sin esa tabla, usar la cifra real (p. ej. 19/22)."
        )


def _check_equation_constants(text: str, rel: str, collector: TodoCollector) -> None:
    """(Informe §1.2) Constantes sin simplificar en una ecuación/rótulo.

    Detecta patrones como '1.0 + ... + 0.50' (dos constantes aditivas sin sumar) en
    rótulos de figura o ecuaciones, que delatan una expresión copiada sin simplificar.
    También marca la incoherencia 'lineal' vs exponente >1 (N^{1.31}) en la misma
    cláusula: una ley con exponente distinto de 1 no es lineal.
    """
    prose = _strip_comments(text)
    # (a) dos constantes aditivas sin simplificar: "1.0 + <algo> + 0.50"
    for m in re.finditer(r"(\d+[.,]\d+)\s*\+[^=\n]{1,40}?\+\s*(\d+[.,]\d+)", prose):
        # solo si ambos extremos son constantes puras (sin variable pegada)
        collector.add_inconsistency(
            f"[{rel}] %TODO-CIFRA expresión con dos constantes aditivas sin "
            f"simplificar ('{m.group(0)[:45].strip()}'): sumar los términos "
            f"constantes ({m.group(1)} y {m.group(2)}) (informe §1.2)."
        )
    # (b) 'lineal' descrito para una ley con exponente != 1 en la misma cláusula.
    for m in re.finditer(
        r"lineal[^.]{0,80}?N\s*\^?\{?\s*(\d+[.,]\d+)|N\s*\^?\{?\s*(\d+[.,]\d+)[^.]{0,80}?lineal",
        prose,
        re.IGNORECASE,
    ):
        exp = m.group(1) or m.group(2)
        if exp and abs(float(exp.replace(",", ".")) - 1.0) > 0.05:
            collector.add_inconsistency(
                f"[{rel}] %TODO-CIFRA se describe como 'lineal' una ley con exponente "
                f"N^{exp} (≠1): no es lineal sino superlineal (informe §1.2). "
                "Corregir la descripción o el ajuste."
            )


def _check_code_model_names(text: str, rel: str, collector: TodoCollector) -> None:
    """(Informe §1.5.3) Nombres de modelo en snake_case (claves de código) en prosa/tablas.

    Los identificadores como 'heisenberg transverse' o 'tfim bond resolved' (guiones
    bajos convertidos en espacios) no deben aparecer como texto: usar la forma legible
    ('Heisenberg transversal', 'TFIM por enlace'). Se busca en el texto fuera de
    \\texttt / verbatim / comentarios.
    """
    prose = _prose_text(text)
    low = prose.lower()
    for code_name, legible in CODE_MODEL_NAMES.items():
        # buscar el nombre en minúscula como palabra (no dentro de otra palabra)
        if re.search(rf"(?<![\w-]){re.escape(code_name)}(?![\w-])", low):
            collector.add_inconsistency(
                f"[{rel}] %TODO-ANGLICISMO nombre de modelo en clave de código "
                f"'{code_name}' (identificador snake_case): usar la forma legible "
                f"'{legible}' (informe §1.5.3)."
            )


def _check_source_is_code(text: str, rel: str, collector: TodoCollector) -> None:
    """(Informe §1.5.4 / §4.1) Fuente que cita una clase de código en vez de la forma
    de plantilla. Detecta '(fuente: ResultIndex)' o 'Fuente: <ClaseCodigo>' en
    minúscula o citando un objeto del repositorio, en lugar de 'Fuente: elaboración
    propia' o una cita autor-año.
    """
    prose = _strip_comments(text)
    for m in re.finditer(r"\(?\s*fuente\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)?", prose):
        val = m.group(1)
        # 'elaboración' es el caso correcto; un identificador CamelCase/snake es código.
        if val.lower().startswith("elaborac"):
            continue
        if re.search(r"[A-Z][a-z]+[A-Z]|_", val) or val[0].islower():
            collector.add_inconsistency(
                f"[{rel}] %TODO-PLANTILLA fuente '{m.group(0).strip()}' cita un objeto "
                "de código o va en minúscula; usar 'Fuente: elaboración propia' o una "
                "cita autor-año (informe §1.5.4/§4.1)."
            )


def _check_notation_collisions(text: str, rel: str, collector: TodoCollector) -> None:
    """(Informe §3) Colisiones de notación: un símbolo con dos significados.

    Señala co-ocurrencias conflictivas conocidas:
      - Δ como gap espectral y como anisotropía XXZ ('XX + YY + Δ·ZZ').
      - θ_opt conviviendo con θ* (dos grafías del óptimo).
    Son señales conservadoras: solo marcan patrones inequívocos del informe.
    """
    prose = _strip_comments(text)
    # Δ como anisotropía en un término XXZ (además de su uso como gap en otras partes).
    if re.search(r"(?:XX\s*\+\s*YY\s*\+\s*)\\?[Dd]elta\s*\\?cdot", prose) or re.search(r"\\Delta\s*\\cdot\s*ZZ", prose):
        collector.add_inconsistency(
            f"[{rel}] %TODO-NOTACION el símbolo Δ se usa como anisotropía XXZ "
            "('Δ·ZZ') y también como gap espectral; renombrar la anisotropía a λ "
            "(informe §3)."
        )
    # θ_opt frente a θ* (dos grafías del mismo óptimo).
    if re.search(r"\\theta_\{?\\?(mathrm\{)?opt", prose) and re.search(r"\\theta\^\*|\\theta\^\{\\?\*\}", prose):
        collector.add_inconsistency(
            f"[{rel}] %TODO-NOTACION conviven 'θ_opt' y 'θ*' para el óptimo; unificar "
            "en una sola grafía (θ*, informe §3)."
        )


def _check_decimal_comma_in_intervals(text: str, rel: str, collector: TodoCollector) -> None:
    """(Informe §5, erratas) Intervalo con coma como separador de elementos:
    '[1,00, 5,00]' se lee como lista de 4 números. Debe usar punto y coma:
    '[1,00; 5,00]'. Detecta '[<num>, <num>]' con ambos extremos decimales por coma.
    """
    prose = _strip_comments(text)
    for m in re.finditer(r"\[\s*\d+,\d+\s*,\s*\d+,\d+\s*\]", prose):
        collector.add_inconsistency(
            f"[{rel}] %TODO-CIFRA intervalo '{m.group(0)}' usa coma como separador "
            "de elementos y como decimal (se lee como 4 números); usar punto y coma "
            "'[1,00; 5,00]' (informe §5, erratas §1.9)."
        )


def _check_arxiv_in_text(text: str, rel: str, collector: TodoCollector) -> None:
    """(Informe §5/§6) Identificador arXiv citado en el cuerpo en vez de autor-año.

    Detecta 'arXiv:NNNN.NNNNN' fuera del entorno de bibliografía (thebibliography):
    en el cuerpo debe citarse por autor-año con \\citep/\\citet, no por el id de arXiv.
    """
    body = re.split(r"\\begin\{thebibliography\}", text)[0]
    body = _strip_comments(body)
    for m in re.finditer(r"arXiv:\s*\d{4}\.\d{4,5}", body):
        collector.add_inconsistency(
            f"[{rel}] %TODO-BIBLIO identificador '{m.group(0)}' citado en el cuerpo; "
            "citar por autor-año con \\citep/\\citet, no por el id de arXiv "
            "(informe §5/§6)."
        )


def _split_keyword_terms(raw: str) -> list[str]:
    """Divide una lista de palabras clave por comas (respetando el punto final)."""
    raw = raw.strip().rstrip(".")
    return [t.strip() for t in raw.split(",") if t.strip()]


def _check_keywords(text: str, rel: str, collector: TodoCollector) -> None:
    """(§4.2) Valida los tres sitios de palabras clave.

    Sitios: pdfkeywords={...} (metadatos PDF), la línea 'Palabras clave:' del
    Resumen y 'Keywords:' del Abstract. Reglas por sitio:
      (1) 4 <= nº de términos <= 6;
      (2) ningún término contiene una sigla (secuencia de >=2 mayúsculas);
      (3) cada término empieza en minúscula, salvo nombres propios de la lista
          blanca (p. ej. 'Ising').
    El Abstract está en inglés: la regla (3) de minúscula inicial se aplica igual
    (los títulos en inglés en keywords van en minúscula en este estilo), pero los
    nombres propios de la lista blanca siguen permitidos.
    """
    sites: list[tuple[str, str]] = []
    m_pdf = re.search(r"pdfkeywords\s*=\s*\{([^}]*)\}", text)
    if m_pdf:
        sites.append(("pdfkeywords", m_pdf.group(1)))
    m_es = re.search(r"Palabras\s+clave:?\s*\}?\s*([^\n]*)", text)
    if m_es:
        sites.append(("Palabras clave", m_es.group(1)))
    m_en = re.search(r"Keywords:?\s*\}?\s*([^\n]*)", text)
    if m_en:
        sites.append(("Keywords", m_en.group(1)))

    sigla_re = re.compile(r"[A-Z]{2,}")
    for site, raw in sites:
        terms = _split_keyword_terms(raw)
        # (1) número de términos
        if not (KEYWORDS_MIN <= len(terms) <= KEYWORDS_MAX):
            collector.add_inconsistency(
                f"[{rel}] %TODO-PLANTILLA palabras clave en '{site}': {len(terms)} "
                f"términos (deben ser entre {KEYWORDS_MIN} y {KEYWORDS_MAX}, §4.2)."
            )
        for term in terms:
            # (2) sin siglas
            if sigla_re.search(term):
                collector.add_inconsistency(
                    f"[{rel}] %TODO-PLANTILLA palabra clave con sigla en '{site}': "
                    f"'{term}' (usar el concepto desplegado, sin siglas, §4.2)."
                )
            # (3) minúscula inicial salvo nombre propio permitido
            first = term.split()[0] if term.split() else ""
            if first and first[0].isupper() and first not in KEYWORDS_PROPER_NOUNS:
                collector.add_inconsistency(
                    f"[{rel}] %TODO-PLANTILLA palabra clave con mayúscula inicial en "
                    f"'{site}': '{term}' (minúscula inicial salvo nombre propio, §4.2)."
                )


def _check_acronym_index(text: str, rel: str, collector: TodoCollector) -> None:
    """(§4.3) Cobertura del índice de acrónimos.

    (1) Siglas usadas >= ACRONYM_MIN_USES veces en el cuerpo que faltan en el índice.
    (2) Entradas del índice que no aparecen en el cuerpo (índice inflado).

    El índice se parsea del bloque 'Índice de acrónimos' ... primer \\end{description}.
    Las siglas del cuerpo se extraen con el mismo criterio de mayúsculas, filtrando
    ruido (OE/H numéricos, romanos, notación de puertas, códigos internos).
    """
    # --- Entradas del índice: \item[XXX] dentro del bloque del índice.
    idx_start = text.find("Índice de acrónimos")
    if idx_start < 0:
        return  # sin índice de acrónimos: legítimo, no se avisa
    idx_end = text.find("\\end{description}", idx_start)
    if idx_end < 0:
        # El título existe pero no cierra el entorno: el índice está malformado y
        # el chequeo de cobertura NO puede correr. Avisar (no silenciar).
        collector.add_inconsistency(
            f"[{rel}] INFO: chequeo del índice de acrónimos OMITIDO: hay 'Índice de "
            "acrónimos' pero no se encontró su \\end{description}; cobertura NO validada."
        )
        return
    index_block = text[idx_start:idx_end]
    index_keys = set(re.findall(r"\\item\[([^\]]+)\]", index_block))
    if not index_keys:
        collector.add_inconsistency(
            f"[{rel}] INFO: chequeo del índice de acrónimos OMITIDO: el bloque existe "
            "pero no contiene entradas \\item[...]; cobertura NO validada."
        )
        return

    # --- Siglas del cuerpo (desde \mainmatter hasta la bibliografía).
    body = text
    m_main = re.search(r"\\mainmatter", text)
    if m_main:
        body = text[m_main.end() :]
    body = re.split(r"\\begin\{thebibliography\}", body)[0]
    # Quitar comentarios y proteger comandos/etiquetas para no contar siglas de
    # \ref{...}, \cite{...}, \label{...} (identificadores internos, no acrónimos).
    body = _strip_comments(body)
    body = _PROTECT_CMD_RE.sub(" ", body)

    # Candidatas: secuencias de >=2 mayúsculas (admite dígito y guion interno).
    counts: dict[str, int] = defaultdict(int)
    for tok in re.findall(r"\b[A-Z][A-Za-z0-9]*(?:-[A-Z][A-Za-z0-9]*)?\b", body):
        # Debe tener al menos 2 mayúsculas para ser sigla (excluye 'La', 'Para').
        if len(re.findall(r"[A-Z]", tok)) < 2:
            continue
        if ACRONYM_NOISE_RE.match(tok):
            continue
        counts[tok] += 1

    # (1) siglas muy usadas ausentes del índice.
    for sig, n in sorted(counts.items()):
        if n >= ACRONYM_MIN_USES and sig not in index_keys:
            collector.add_inconsistency(
                f"[{rel}] %TODO-PLANTILLA sigla '{sig}' usada {n} veces en el cuerpo "
                f"pero ausente del Índice de acrónimos (§4.3)."
            )

    # (2) entradas del índice sin uso en el cuerpo (índice inflado).
    for key in sorted(index_keys):
        # Buscar el uso literal de la sigla en el cuerpo (palabra completa).
        if not re.search(rf"\b{re.escape(key)}\b", body):
            collector.add_inconsistency(
                f"[{rel}] %TODO-PLANTILLA entrada del índice de acrónimos '{key}' sin "
                "uso en el cuerpo (índice inflado; retirar o citar, §4.3)."
            )


def _check_caption_sources(text: str, rel: str, collector: TodoCollector) -> None:
    """(§4.1) Cada \\caption debe declarar su fuente ('Fuente: ...').

    Recorre los \\caption{...} (el título largo, no el corto entre corchetes) y
    señala los que no incluyan 'Fuente:' (que puede ser 'elaboración propia' o una
    cita). Identifica la tabla/figura por su \\label si es deducible en la cercanía.
    """
    # Localizar cada \caption{ y extraer su cuerpo balanceando llaves.
    for m in re.finditer(r"\\caption(?:\[[^\]]*\])?\{", text):
        start = m.end()
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{" and text[j - 1] != "\\":
                depth += 1
            elif text[j] == "}" and text[j - 1] != "\\":
                depth -= 1
            j += 1
        caption_body = text[start : j - 1]
        if re.search(r"Fuente\s*:", caption_body, re.IGNORECASE):
            continue
        # Deducir el label cercano (dentro de los ~200 chars siguientes).
        near = text[j : j + 200]
        lab = re.search(r"\\label\{([^}]+)\}", near)
        who = f" ({lab.group(1)})" if lab else ""
        collector.add_inconsistency(
            f"[{rel}] %TODO-PLANTILLA caption sin 'Fuente:'{who}: añadir "
            "'Fuente: elaboración propia' o la cita correspondiente (§4.1)."
        )


def _check_inline_arithmetic(text: str, rel: str, collector: TodoCollector) -> None:
    """(Informe §5, erratas §5.2/§4.4) Producto inline 'A x B (x C) = D' que no cuadra.

    Detecta expresiones de multiplicación escritas en prosa con el símbolo de
    factores ('$\\times$', '\\times' o el carácter '×') donde el resultado declarado
    no es el producto de los factores. Caso del informe: '5 topologías × 4
    profundidades × 3–4 semillas = 79' (5·4 = 20 configuraciones; el 79 no
    descompone). Cuando un factor es un rango 'a--b' se acepta si el resultado cae en
    [prod_min, prod_max]. Señal, no fix.

    Preprocesado: se normaliza el separador de factores '$\\times$' al carácter '×'
    y SÓLO ENTONCES se neutraliza el resto del modo matemático. Así el '×' de un
    producto legítimo sobrevive, pero un '$p = 1$' cercano se borra y no se confunde
    con el '= total' (evita el falso positivo de '5 × 4 profundidades ($p=1$--4) ×
    ...', que no declara un total numérico).
    """
    norm = _strip_comments(text)
    norm = re.sub(r"\$\s*\\times\s*\$|\\times", "×", norm)  # unificar separador
    prose = _strip_protected(norm)  # neutraliza el resto del math ($p=1$, etc.)
    num = r"(\d+)(?:\s*(?:--|–)\s*(\d+))?"
    xsep = r"\s*×\s*"  # separador ya normalizado a '×' arriba
    pat = re.compile(
        rf"{num}[^=×\n]{{0,25}}?{xsep}{num}(?:[^=×\n]{{0,25}}?{xsep}{num})?"
        rf"[^=\n]{{0,25}}?=\s*(\d+)"
    )
    for m in pat.finditer(prose):

        def rng(a, b):
            lo = int(a)
            hi = int(b) if b else lo
            return lo, hi

        factors = [rng(m.group(1), m.group(2)), rng(m.group(3), m.group(4))]
        if m.group(5):
            factors.append(rng(m.group(5), m.group(6)))
        total = int(m.group(7))
        prod_min = 1
        prod_max = 1
        for lo, hi in factors:
            prod_min *= lo
            prod_max *= hi
        if not (prod_min <= total <= prod_max):
            ln = _line_number_at(prose, m.start())
            fstr = " × ".join(f"{lo}" if lo == hi else f"{lo}--{hi}" for lo, hi in factors)
            rango = f"{prod_min}" if prod_min == prod_max else f"[{prod_min}, {prod_max}]"
            collector.add_inconsistency(
                f"[{rel}:{ln}] %TODO-CIFRA producto inline no cuadra: "
                f"{fstr} = {rango}, pero se declara {total}. Dar el desglose real "
                "(informe §5, erratas §5.2)."
            )


def _check_speedup_error_ratio(text: str, rel: str, collector: TodoCollector) -> None:
    """(Chequeo F) No confundir el factor de aceleración A con la razón de error R(N).

    El steering (§5, §11, §16, errores #12–#14) fija tres reglas estrictas:

      1. El símbolo de la aceleración es A (o S_ac), NO S: S es la entropía de
         entrelazamiento. Marca "$S$" o "S =" en una cláusula que hable de
         aceleración / speedup.
      2. A (cociente de evaluaciones de circuito) y R(N) (cociente de precisión
         ΔE/gap) son magnitudes distintas y NO se mezclan en el mismo rango. El
         rango prohibido es "2,5×–4400×": mezcla un valor de R (~2,5×) con uno de A
         (~4400×). El rango legítimo de A es "5×–4400×" (p=1 uniforme → 79-dim por
         enlace), que NO debe marcarse. Discriminante: el mínimo del rango.
      3. El rango antiguo "29×–500×" (objetivo general viejo) está prohibido.

    Es señal, no fix: requiere criterio humano. Conservador para no marcar el rango
    legítimo de A (5×–4400×) ni la reducción de |ΔE| con p (15–55×, tercera magnitud).

    NOTA: trabaja sobre el texto CRUDO (solo sin comentarios), no sobre
    _strip_protected, porque los símbolos que analiza ($S$, $\\times$, $2,5$) viven
    en modo matemático y _strip_protected los eliminaría.
    """
    prose = _strip_comments(text)

    # (1) Símbolo S para la aceleración (S está reservado a la entropía).
    #     Solo se marca la 'S' cuando aparece en MODO MATEMÁTICO ($S$) —que es como
    #     se escribe un símbolo— dentro de una cláusula de aceleración/speedup. Se
    #     evita la 'S' de palabras (evaluacioneS, etc.) exigiendo $...$ o 'S ='.
    for m in re.finditer(
        r"(?:factor\s+de\s+)?aceleraci[oó]n[^.]{0,40}?\$S\$|"
        r"\$S\$\s*=[^.]{0,40}?aceleraci[oó]n|"
        r"aceleraci[oó]n[^.]{0,30}?\bS\s*=|"
        r"speedup[^.]{0,25}?\$S\$",
        prose,
        re.IGNORECASE,
    ):
        collector.add_inconsistency(
            f"[{rel}] %TODO-CIFRA símbolo 'S' usado para la aceleración "
            f"('{m.group(0)[:50].strip()}'): S es la entropía de entrelazamiento; "
            "la aceleración es A (steering §11, error #13)."
        )

    # (2) Rango prohibido que mezcla R(N) con A: mínimo pequeño (< 5×) y máximo
    #     grande (> 100×). El rango legítimo de A arranca en ~5×. Se admiten '$'
    #     intercalados en el rango ('$2,5$--$4400$\times').
    for m in re.finditer(
        r"\$?(\d+(?:[.,]\d+)?)\$?\s*(?:--|–|-|a)\s*\$?(\d+(?:[.,]\d+)?)\$?\s*\$?\\?times",
        prose,
    ):
        lo = float(m.group(1).replace(",", "."))
        hi = float(m.group(2).replace(",", "."))
        if lo < 5.0 and hi > 100.0:
            collector.add_inconsistency(
                f"[{rel}] %TODO-CIFRA rango de aceleración '{m.group(0).strip()}' "
                f"mezcla la razón de error R(N) (mínimo {lo:g}×, típico de R) con el "
                f"factor de aceleración A (máximo {hi:g}×): son magnitudes distintas "
                "y no se combinan en un rango (steering §16, error #14). El rango "
                "legítimo de A arranca en ~5×."
            )

    # (2b) Valor ÚNICO grande etiquetado como aceleración/A (error #12): el factor
    #     de aceleración A medido llega a ~100× (79 dim por enlace); un valor >=500×
    #     es la razón de error R(N) (p. ej. 4414×), NO A. Se marca cuando 'A' o
    #     'aceleración'/'speedup' aparece pegado (<=35 chars) a un factor >=500×.
    #     Conservador: NO marca si en la misma cláusula (<=25 chars) hay 'R' o
    #     'razón de error' (etiquetado correcto), ni el rango legítimo de A (parte 2).
    for m in re.finditer(
        r"(aceleraci[oó]n|factor\s+A\b|\bA\s*\\?[≈=]|speedup)[^.]{0,35}?"
        r"(\d{3,5})(?:[.,]\d+)?\s*\$?\\?times|"
        r"(\d{3,5})(?:[.,]\d+)?\s*\$?\\?times[^.]{0,35}?"
        r"(aceleraci[oó]n|factor\s+A\b|speedup)",
        prose,
        re.IGNORECASE,
    ):
        val_s = m.group(2) or m.group(3)
        if val_s is None:
            continue
        val = int(val_s)
        if val < 500:
            continue  # dentro del rango plausible de A
        # ¿la cláusula aclara que es R (razón de error)? entonces es correcto.
        ctx = prose[max(0, m.start() - 25): m.end() + 25]
        if re.search(r"raz[oó]n\s+de\s+error|\bR\b\s*\\?[≈=(]|\$R", ctx):
            continue
        collector.add_inconsistency(
            f"[{rel}] %TODO-CIFRA valor '{val}×' etiquetado como aceleración/A: el "
            f"factor A medido llega a ~100× (79 dim por enlace); {val}× es la razón "
            "de error R(N), no A (steering §16, errores #12/#14). Etiquetar como R."
        )

    # (3) Rango antiguo prohibido "29×–500×" (objetivo general viejo, §16).
    if re.search(r"29\s*(?:--|–|-|a)\s*500\s*\$?\\?times", prose):
        collector.add_inconsistency(
            f"[{rel}] %TODO-CIFRA rango de aceleración antiguo '29×–500×' "
            "(objetivo general obsoleto); usar el factor A canónico con su k̄ y r "
            "(steering §16)."
        )


def _check_cz_budget(text: str, rel: str, collector: TodoCollector) -> None:
    """(Chequeo 1) Conteo de compuertas CZ coherente con el convenio 2 CZ/término.

    Busca menciones de la forma "<N> CZ" / "<N> compuertas CZ" en la prosa y, cuando
    la misma frase declara el modelo, la N y la profundidad p, verifica que el conteo
    salga de una descomposición uniforme (2 CZ por término de dos cuerpos, por capa).
    Es una señal: un conteo que no corresponda a ningún (modelo, N, p) plausible
    (como el "27 CZ" imposible del informe) se marca para revisión.
    """
    prose = _prose_text(text)
    # "<num> (compuertas|puertas) CZ" con contexto ~90 chars alrededor para leer
    # N/p/modelo. Se admiten ambas grafías ('compuertas' y 'puertas', que conviven
    # en el .tex) para no dejar pasar un conteo por la variante léxica.
    cz_re = re.compile(r"(\d{1,4})\s*(?:com)?puertas?\s+CZ\b|(\d{1,4})\s*CZ\b", re.IGNORECASE)
    for m in cz_re.finditer(prose):
        n_cz = int(m.group(1) or m.group(2))
        ctx = prose[max(0, m.start() - 90) : m.end() + 90].lower()
        # Necesitamos N y (opcionalmente) p en el contexto para poder validar.
        n_match = re.search(r"n\s*=\s*(\d{1,3})", ctx)
        if not n_match:
            continue
        N = int(n_match.group(1))
        # p hasta dos dígitos (sondeos a p=5/p=8; no limitar a un solo dígito).
        p_match = re.search(r"p\s*=\s*(\d{1,2})", ctx)
        p = int(p_match.group(1)) if p_match else 1
        # Detectar modelo mencionado en el contexto; por defecto TFIM.
        model = "tfim"
        if "j1" in ctx or "j_1" in ctx or "frustrad" in ctx:
            model = "j1-j2"
        elif "kitaev" in ctx:
            model = "kitaev"
        terms_fn = CZ_TERMS_2BODY.get(model)
        if terms_fn is None:
            continue
        expected = p * 2 * terms_fn(N)
        if n_cz != expected:
            collector.add_inconsistency(
                f"[{rel}] %TODO-CIFRA conteo de CZ '{n_cz}' incoherente con el "
                f"convenio 2 CZ/término para {model} a N={N}, p={p} "
                f"(esperado {expected} = {p}·2·{terms_fn(N)}). Verificar el conteo."
            )


def _check_n_scales(text: str, rel: str, collector: TodoCollector) -> None:
    """(Chequeo 3) N máximo consistente con las tres escalas canónicas (22/40/250).

    Recolecta afirmaciones de escala ("escala hasta N=X", "valida(do) hasta N=X",
    "N máximo X", "hasta N = X"). Señala un N máximo suelto que no sea una de las
    tres escalas canónicas: cada escala tiene un significado distinto (22 = vector
    de estado exacto; 40 = pipeline completo con DMRG; 250 = solo evaluación MPS) y
    mezclarlas sin declarar cuál es lleva a error (steering §16).
    """
    prose = _prose_text(text)
    # Afirmaciones de "N máximo": (a) 'escala/valida... hasta N=X' o 'hasta N<=X';
    # (b) 'N máx(imo) X'; (c) 'escala/valida... hasta X qubits' (sin 'N='). Todas
    # capturan un único entero que debe ser una de las tres escalas canónicas.
    scale_re = re.compile(
        r"(?:escala|valida(?:do|da)?|llega|alcanza)\s+hasta\s+N\s*(?:=|\\leq|<=|≤)\s*(\d{1,3})|"
        r"N\s*m[aá]x(?:imo)?\s*(?:de|=|:)?\s*(\d{1,3})|"
        r"(?:escala|valida(?:do|da)?|llega|alcanza)\s+hasta\s+(\d{1,3})\s+qubits",
        re.IGNORECASE,
    )
    for m in scale_re.finditer(prose):
        val = int(m.group(1) or m.group(2) or m.group(3))
        if val not in CANONICAL_N_SCALES:
            collector.add_inconsistency(
                f"[{rel}] %TODO-CIFRA afirmación de escala 'N={val}' no coincide con "
                f"ninguna de las tres escalas canónicas {CANONICAL_N_SCALES} "
                "(22=vector de estado, 40=pipeline+DMRG, 250=solo MPS); declarar a "
                "cuál escala se refiere (steering §16)."
            )


def _check_h_grid_counts(text: str, rel: str, collector: TodoCollector) -> None:
    """(Chequeo 4) Denominadores de la malla de h coherentes con la Fase 1.

    La malla canónica tiene 52 puntos totales; 39 caen en el régimen válido
    [1,3; 5,0]. Señala menciones de conteos de puntos de la malla (o denominadores
    /NN de tasas de aprobación sobre el barrido) que no correspondan a 39 (válido)
    ni 52 (total) —como el obsoleto "27 puntos"— cuando la frase habla del barrido
    de h / puntos de evaluación.
    """
    prose = _prose_text(text)
    allowed = {H_GRID_TOTAL, H_GRID_VALID}
    # (a) "<n> puntos de (h|evaluación|test|barrido)" o "malla de <n> puntos" — el
    # conteo de la malla escrito como cardinal.
    grid_re = re.compile(
        r"(\d{1,3})\s*puntos\s+(?:de\s+(?:evaluaci[oó]n|test|barrido|h)\b|"
        r"en\s+el\s+barrido)|"
        r"malla\s+(?:de\s+h\s+)?de\s+(\d{1,3})\s*puntos",
        re.IGNORECASE,
    )
    for m in grid_re.finditer(prose):
        n = int(m.group(1) or m.group(2))
        if n not in allowed:
            collector.add_inconsistency(
                f"[{rel}] %TODO-CIFRA conteo de la malla de h '{n} puntos' no coincide "
                f"con la Fase 1 canónica ({H_GRID_VALID} en régimen válido [1,3;5,0] / "
                f"{H_GRID_TOTAL} total). Verificar contra el intervalo declarado."
            )
    # (b) Denominador de tasa de aprobación sobre el barrido: '(n/D)' donde el
    # contexto habla de barrido/malla/régimen válido de h y D no es 39 ni 52. Es el
    # caso del obsoleto '/27'. Conservador: solo dispara si la MISMA cláusula (±60
    # chars) menciona el barrido de h y el denominador cae en el rango de una malla
    # de h (20–60), para no marcar denominadores de otras cuentas (p. ej. '19/22'
    # de tamaños N, o conteos de topologías/semillas).
    denom_ctx_re = re.compile(r"barrido|malla\s+de\s+h|r[eé]gimen\s+v[aá]lido|puntos\s+de\s+h", re.IGNORECASE)
    for m in re.finditer(r"\(\s*\d+\s*/\s*(\d{2})\s*\)", prose):
        D = int(m.group(1))
        if D in allowed or not (20 <= D <= 60):
            continue
        ctx = prose[max(0, m.start() - 60) : m.end() + 60]
        if denom_ctx_re.search(ctx):
            collector.add_inconsistency(
                f"[{rel}] %TODO-CIFRA denominador de barrido '/{D}' no coincide con la "
                f"malla de la Fase 1 ({H_GRID_VALID} en régimen válido / {H_GRID_TOTAL} "
                f"total); ¿es el obsoleto '/27'? Verificar contra el intervalo declarado."
            )


def _check_math_comma_intervals(text: str, rel: str, collector: TodoCollector) -> None:
    """(Chequeo 5, §1.9a) Intervalos con coma separadora en modo matemático.

    Un intervalo escrito como '$[1,3, 3,0]$' se lee como una lista de cuatro números
    (la coma es a la vez separador decimal y de lista). La forma correcta usa punto y
    coma: '$[1{,}3;\\, 3{,}0]$'. Extiende _check_h_range_coherence (que solo mira
    captions de tablas etiquetadas) a TODA la prosa matemática del documento.

    Señal, no fix: la reescritura con ';' es criterio de redacción.
    """
    prose = _strip_comments(text)
    seen: set[str] = set()
    for mm in _MATH_INLINE_RE.finditer(prose):
        for m in _MATH_COMMA_INTERVAL_RE.finditer(mm.group(0)):
            frag = m.group(0)
            if frag in seen:
                continue
            seen.add(frag)
            ln = prose.count("\n", 0, mm.start() + m.start()) + 1
            collector.add_inconsistency(
                f"[{rel}:{ln}] intervalo con coma separadora en modo matemático "
                f"('{frag}') se lee como lista de 4 números; usar punto y coma "
                "('[a{,}b;\\, c{,}d]') para el separador de intervalo (§1.9a)."
            )


def _check_energy_units(text: str, rel: str, collector: TodoCollector) -> None:
    """(Chequeo 6, §4.5) 'energy units' / 'unidades de energía' no es una unidad.

    Los ejes/columnas de |ΔE| se expresan en J = 1 (steering §7, §16), no en un
    literal 'unidades de energía'. Marca cualquier aparición en prosa o rótulos.

    Señal, no fix.
    """
    prose = _strip_comments(text)
    for m in _ENERGY_UNITS_RE.finditer(prose):
        ln = _line_number_at(prose, m.start())
        collector.add_inconsistency(
            f"[{rel}:{ln}] '{m.group(0)}' no es una unidad; usar J = 1 (o "
            "'unidades de $J$') como declara §3.4 y el preámbulo de §5 (§4.5)."
        )


def _check_xref_style(text: str, rel: str, collector: TodoCollector) -> None:
    """(Chequeo 7, §5) Estilo de referencias cruzadas unificado.

    La plantilla escribe 'Sección xx' y 'Ecuación xx' (idealmente con \\ref/\\eqref).
    Conviven a mano tres estilos: 'Ec. 3.3', '§5.4.2', 'Ecuación 2.9'/'Sección 6.1'
    con número literal. Marca las formas abreviadas y los números literales de
    sección/ecuación escritos a mano fuera de comandos, para unificarlos.

    Trabaja sobre prosa protegida (sin math ni comandos) para no marcar los números
    dentro de \\ref{...}/\\eqref{...} ni de fórmulas.

    Señal, no fix (la unificación es criterio de redacción).
    """
    prose = _prose_text(text)
    for label, pat, hint in _XREF_ABBREV_RES:
        seen_lines: set[int] = set()
        for m in pat.finditer(prose):
            ln = _line_number_at(prose, m.start())
            if ln in seen_lines:
                continue
            seen_lines.add(ln)
            collector.add_inconsistency(
                f"[{rel}:{ln}] referencia cruzada '{m.group(0).strip()}' ({label}): {hint} (unificar estilo, §5)."
            )


def _check_entropy_coefficient(text: str, rel: str, collector: TodoCollector) -> None:
    """(Chequeo 8, §1.10) Coeficiente de la entropía en el punto crítico.

    La CFT predice S ~ (c/6) log N (contorno abierto) o (c/3) log N (periódico).
    El informe detectó '6c log N' (coeficiente invertido, 36× mayor). Centinela:
    marca la forma invertida por si reaparece en una futura edición.

    Trabaja sobre el texto crudo (el coeficiente vive en modo matemático).
    """
    prose = _strip_comments(text)
    for m in _ENTROPY_BADCOEF_RE.finditer(prose):
        ln = _line_number_at(prose, m.start())
        collector.add_inconsistency(
            f"[{rel}:{ln}] coeficiente de entropía '{m.group(0).strip()}': la CFT da "
            "S ~ (c/6) log N (contorno abierto) o (c/3) (periódico), no '6c log N' "
            "(§1.10)."
        )


def _check_duplicate_factor_values(text: str, rel: str, collector: TodoCollector) -> None:
    """(Chequeo 9, §1.1/§4.5) La misma cantidad con dos valores en dos sitios.

    El informe detectó '62×' en la Figura 6.1 y '57×' en la Figura 6.2 para N=16:
    la misma cantidad (razón de error R a un mismo N) con dos valores. Heurística:
    recoge todos los factores 'X×' asociados a un 'N=k' en su misma frase y, si un
    mismo N tiene dos factores distintos en el documento, lo señala.

    Conservador: solo agrupa factores que aparecen a <=40 caracteres de un 'N=k'
    explícito, y solo marca cuando el mismo N presenta valores que difieren > 5 %
    (para no marcar A y R legítimamente distintos que estén lejos entre sí en el
    texto). Es una señal para revisión humana.
    """
    prose = _prose_text(text)
    # 'N=k ... Xx' o 'Xx ... N=k' en una ventana corta.
    by_n: dict[int, set[float]] = {}
    factor_re = re.compile(
        r"N\s*=\s*(\d{1,3})[^.]{0,40}?(\d+(?:[.,]\d+)?)\s*(?:×|x\b)|"
        r"(\d+(?:[.,]\d+)?)\s*(?:×|x\b)[^.]{0,40}?N\s*=\s*(\d{1,3})",
        re.IGNORECASE,
    )
    for m in factor_re.finditer(prose):
        if m.group(1):
            n = int(m.group(1))
            fac = float(m.group(2).replace(",", "."))
        else:
            n = int(m.group(4))
            fac = float(m.group(3).replace(",", "."))
        if fac < 1.5:  # ignorar cocientes triviales / cifras que no son factores
            continue
        by_n.setdefault(n, set()).add(round(fac, 2))
    for n, facs in sorted(by_n.items()):
        if len(facs) < 2:
            continue
        lo, hi = min(facs), max(facs)
        if hi > lo * 1.05:  # difieren más de un 5 %
            vals = ", ".join(f"{v:g}×" for v in sorted(facs))
            collector.add_inconsistency(
                f"[{rel}] %TODO-CIFRA a N={n} conviven factores distintos ({vals}): "
                "si son la misma cantidad (p. ej. la razón de error R en dos figuras), "
                "unificar; si son A y R, etiquetarlas distinto (§1.1/§4.5)."
            )


def _check_cross_section_figures(text: str, rel: str, collector: TodoCollector) -> None:
    """(20) Cruce de cifras del mismo concepto entre secciones clave.

    Heurística ligera: para conceptos con una etiqueta reconocible (topología +
    métrica de PassRate, y el rango de speedup), recoge todos los porcentajes y
    señala si un mismo concepto aparece con cifras distintas en capítulos
    distintos (posible contradicción). Es una señal, no una verdad: requiere
    revisión humana.
    """
    chapters = _split_chapters(text)
    # 1) Rango de speedup: solo rangos explícitos "N--M$\times$" o "N--M veces"
    #    (evita capturar números sueltos con \times que no son aceleraciones).
    speedup_by_chap: dict[str, set[str]] = {}
    range_re = re.compile(r"(\d+)\s*--\s*(\d+)\s*\$?\\times")
    for title, body in chapters.items():
        ranges = {f"{a}-{b}" for a, b in range_re.findall(body)}
        if ranges:
            speedup_by_chap[title] = ranges
    all_ranges = {s for v in speedup_by_chap.values() for s in v}
    if len(all_ranges) > 1:
        detail = "; ".join(f"{c}: {sorted(v)}" for c, v in speedup_by_chap.items())
        collector.add_inconsistency(
            f"[{rel}] %TODO-CIFRA rangos de aceleración distintos entre capítulos ({detail}); "
            "unificar la cifra (o eliminarla, steering §5)."
        )

    # 1b) Rango de "puntos de entrenamiento / puntos VQE": debe ser único en todo
    #     el documento (canónico 16--39). El sufijo es OBLIGATORIO ('VQE' o 'de
    #     entrenamiento'): sin él se confunden magnitudes homónimas distintas que
    #     también dicen "puntos" —"+28--33 puntos porcentuales" (mejora de PassRate)
    #     y "10--20 puntos de h" (malla del barrido de h para h_min)— que NO son
    #     puntos de entrenamiento y no deben compararse con el rango canónico.
    pts_by_chap: dict[str, set[str]] = {}
    pts_re = re.compile(r"(\d+)\s*--\s*(\d+)\s*puntos(?:\s+VQE|\s+de\s+entrenamiento)")
    for title, body in chapters.items():
        ranges = {f"{a}-{b}" for a, b in pts_re.findall(body)}
        if ranges:
            pts_by_chap[title] = ranges
    all_pts = {s for v in pts_by_chap.values() for s in v}
    if len(all_pts) > 1:
        detail = "; ".join(f"{c}: {sorted(v)}" for c, v in pts_by_chap.items())
        collector.add_inconsistency(
            f"[{rel}] %TODO-CIFRA rangos de 'puntos de entrenamiento' distintos entre "
            f"capítulos ({detail}); unificar (canónico 16--39, steering §1)."
        )

    # 2) PassRate por topología en la PROSA de resumen/resultados/discusión/conclusiones.
    #
    # Robustez (evita falsos positivos): un mismo nombre de topología aparece muchas
    # veces con porcentajes que NO son tasas de aprobación (un gap del 5%, un error
    # del 10%, una fidelidad del 92%, un umbral <5%). Antes se capturaba el primer %
    # dentro de 60 caracteres del nombre, mezclando magnitudes distintas y marcando
    # falsos positivos. Ahora se exige que:
    #   (a) el % esté MUY cerca del nombre de topología (<=35 chars, misma cláusula);
    #   (b) la ventana contenga una señal de tasa de aprobación ('aprob'/'pasan'/
    #       'PassRate'/'tasa'); y
    #   (c) la ventana NO contenga otra magnitud (gap, error, fidelidad, umbral,
    #       ahorro, iteraciones), que descalifica el % como PassRate.
    # Además, los % SIEMPRE deben provenir de la MISMA sección para poder compararse:
    # un 90% en Resultados y un 79% en 'Resultados complementarios' del apéndice NO
    # son contradicción (son configuraciones distintas), así que solo se marca cuando
    # el mismo concepto difiere entre capítulos DISTINTOS del cuerpo.
    key_chaps = {
        t: b
        for t, b in chapters.items()
        if any(k in t.lower() for k in ("resumen", "resultado", "discus", "conclus"))
        and "complementari" not in t.lower()  # apéndice B: otras configuraciones
    }
    topo_pat = re.compile(
        r"(cadena 1D|heavy-hex|escalera|cuadrada|triangular)(.{0,35}?)(\d{2,3})\\%",
        re.IGNORECASE,
    )
    _passrate_signal = re.compile(r"aprob|pasan|passrate|tasa", re.IGNORECASE)
    _other_magnitude = re.compile(
        r"gap|error|fidelidad|umbral|[<>]|\\leq|\\geq|ahorro|iteracion|"
        r"\\Delta E|precisi[oó]n",
        re.IGNORECASE,
    )
    concept: dict[str, dict[str, set[str]]] = {}
    for title, body in key_chaps.items():
        for m in topo_pat.finditer(body):
            topo = m.group(1).lower()
            # Contexto = ~40 chars ANTES de la topología + la ventana hasta el %.
            # La señal de aprobación puede ir antes ("la tasa de aprobación de
            # {topo} ... 95%") o después ("{topo} ... aprobó el 95%"), por eso se
            # examinan ambos lados. La exclusión de otras magnitudes usa el mismo
            # contexto ampliado.
            pre = body[max(0, m.start() - 40) : m.start()]
            window = pre + m.group(2)
            # (b) debe oler a tasa de aprobación y (c) no a otra magnitud
            if not _passrate_signal.search(window):
                continue
            if _other_magnitude.search(window):
                continue
            concept.setdefault(topo, {}).setdefault(m.group(3), set()).add(title)
    for topo, vals in concept.items():
        # solo es contradicción si el mismo concepto difiere en >1 CAPÍTULO distinto
        chapters_involved = {ch for chs in vals.values() for ch in chs}
        if len(vals) > 1 and len(chapters_involved) > 1:
            detail = "; ".join(f"{pct}\\% en {sorted(ch)}" for pct, ch in vals.items())
            collector.add_inconsistency(
                f"[{rel}] %TODO-CIFRA '{topo}' aparece como tasa de aprobación con "
                f"cifras distintas entre capítulos ({detail}); verificar coherencia."
            )


def _check_hypotheses_coverage(text: str, rel: str, collector: TodoCollector) -> None:
    """(22, señal) Cobertura estructural hipótesis -> conclusiones.

    Detecta las hipótesis Hn definidas en el capítulo de objetivos/hipótesis y
    verifica que cada una se mencione en Conclusiones. NO evalúa si la conclusión
    responde de fondo (eso requiere lectura semántica); solo marca huérfanas.
    """
    chapters = _split_chapters(text)
    obj_body = next((b for t, b in chapters.items() if "hipótesis" in t.lower() or "objetivo" in t.lower()), "")
    concl_body = next((b for t, b in chapters.items() if "conclus" in t.lower()), "")
    if not obj_body or not concl_body:
        return
    hyps = sorted(set(re.findall(r"\bH(\d+)\b", obj_body)), key=int)
    if not hyps:
        return
    missing = [f"H{h}" for h in hyps if not re.search(rf"\bH{h}\b", concl_body)]
    if missing:
        collector.add_inconsistency(
            f"[{rel}] %TODO-HIPOTESIS hipótesis sin mención explícita en Conclusiones: "
            f"{', '.join(missing)} (verificar que las conclusiones respondan a cada una)."
        )
    else:
        # Señal informativa: todas cubiertas estructuralmente (falta juicio de fondo).
        collector.add_inconsistency(
            f"[{rel}] INFO: las {len(hyps)} hipótesis (H1--H{hyps[-1]}) se mencionan en "
            "Conclusiones; revisar manualmente que cada respuesta sea concluyente (§22)."
        )


def _check_editorial(text: str, rel: str, collector: TodoCollector) -> None:
    """(Editorial 1 y 2) Siglas sin definir en primer uso y términos no unificados.

    - Siglas: la primera aparición de cada sigla del glosario debe ir acompañada
      de su expansión (entre paréntesis, en cualquier orden). Se ignora el bloque
      de bibliografía (títulos en inglés) y las líneas de comando.
    - Términos: reporta variantes de grafía conviviendo (steering §6/§11).
    """
    # Recortar la bibliografía para no analizar títulos en inglés.
    body = re.split(r"\\begin\{thebibliography\}", text)[0]

    # Para el chequeo de SIGLAS, arrancar desde el cuerpo real (\mainmatter) y no
    # desde el preámbulo: el \title, \subject y pdfkeywords contienen siglas sin
    # definir de forma legítima (metadatos), y marcarlas es un falso positivo.
    # La primera definición canónica de cada sigla vive en el Resumen/cuerpo.
    m_main = re.search(r"\\mainmatter", body)
    sigla_body = body[m_main.end() :] if m_main else body
    # Offset de líneas del preámbulo recortado, para reportar la línea real.
    sigla_line_offset = body[: m_main.end()].count("\n") if m_main else 0
    sigla_lines = sigla_body.split("\n")

    # (1) Siglas sin definir en primer uso (buscando desde \mainmatter).
    # Se incluyen las siglas base (ACRONYMS) más las adicionales del §4.3
    # (DyPP, NLCE, TREX, DD, PVLS) marcadas sin desplegar en el informe.
    for acr, expansion in {**ACRONYMS, **ACRONYMS_EXTRA}.items():
        first_ln = None
        for idx, line in enumerate(sigla_lines, start=1):
            code = re.sub(r"(?<!\\)%.*$", "", line)
            if re.search(rf"\b{acr}\b", code):
                first_ln = idx + sigla_line_offset  # línea real en el archivo
                first_code = code
                break
        if first_ln is None:
            continue
        # ¿La expansión aparece cerca (misma línea) de la primera aparición?
        # Se acepta tanto la expansión en inglés como su traducción al español.
        first_lower = first_code.lower()
        key_words = expansion.split()[0]  # p. ej. "Variational"
        has_en = key_words.lower() in first_lower or expansion.lower() in first_lower
        has_es = any(kw in first_lower for kw in ACRONYM_ES_KEYWORDS.get(acr, ()))
        if not has_en and not has_es:
            collector.add_inconsistency(
                f"[{rel}:{first_ln}] sigla '{acr}' usada sin definir en su primer uso "
                f"(añadir expansión: '{expansion} ({acr})')."
            )

    # (2) Términos no unificados (variantes conviviendo).
    for canonical, variants in TERM_VARIANTS:
        found: dict[str, int] = {}
        for v in variants:
            n = len(re.findall(v, body))
            if n:
                found[v] = n
        if len(found) > 1:
            detail = ", ".join(f"{v.strip(chr(92) + 'b')}={n}" for v, n in found.items())
            collector.add_inconsistency(
                f"[{rel}] término no unificado (canónico '{canonical}'): variantes "
                f"conviviendo [{detail}]; unificar (steering §6/§11)."
            )


def _check_bibliography(text: str, rel: str, collector: TodoCollector) -> None:
    """(15) Chequeos automatizables de bibliografía, steering §9.

    Detecta (sin acceder a arXiv): números arXiv duplicados entre entradas,
    mismo primer-autor+año sin sufijo a/b, y uso inconsistente de versiones vN.
    La verificación de que autor/título/año coincidan con el registro real de
    arXiv NO es automatizable aquí -> queda como tarea humana (steering §9).
    """
    bibitems = re.findall(r"\\bibitem\[([^\]]*)\]\{([^}]+)\}([^\n]*)", text)
    if not bibitems:
        return
    arxiv_seen: dict[str, list[str]] = {}
    authoryear: dict[str, list[str]] = {}
    has_version = 0
    for label_disp, key, body in bibitems:
        # arXiv duplicado
        for ax in re.findall(r"arXiv:(\d{4}\.\d{4,5})", body):
            arxiv_seen.setdefault(ax, []).append(key)
        # versión vN presente
        if re.search(r"arXiv:\d{4}\.\d{4,5}v\d", body):
            has_version += 1
        # primer autor + año a partir del display [Autor(año)]
        m = re.match(r"([A-Za-zÀ-ÿ]+).*?\((\d{4})[a-z]?\)", label_disp)
        if m:
            ay = f"{m.group(1).lower()}{m.group(2)}"
            has_suffix = bool(re.search(r"\(\d{4}[a-z]\)", label_disp))
            authoryear.setdefault(ay, []).append(key + ("*" if has_suffix else ""))
    for ax, keys in arxiv_seen.items():
        if len(keys) > 1:
            collector.add_inconsistency(
                f"[{rel}] BIBLIO arXiv:{ax} aparece en múltiples entradas ({', '.join(keys)}); "
                "verificar duplicado o número incorrecto (§9)."
            )
    for ay, keys in authoryear.items():
        if len(keys) > 1 and not all(k.endswith("*") for k in keys):
            clean = [k.rstrip("*") for k in keys]
            collector.add_inconsistency(
                f"[{rel}] BIBLIO mismo primer-autor+año sin sufijo a/b: {ay} "
                f"({', '.join(clean)}); usar 2025a, 2025b en texto y bibliografía (§9)."
            )
    n_total = len(bibitems)
    if 0 < has_version < n_total:
        collector.add_inconsistency(
            f"[{rel}] BIBLIO versiones vN inconsistentes: {has_version}/{n_total} entradas "
            "incluyen 'vN'; unificar (todas o ninguna, §9)."
        )

    # (16) Orden alfabético de la bibliografía: el apellido del primer autor del
    # display [Apellido...] debe ir en orden ascendente. Marca la primera entrada
    # que rompe el orden (no reordena; requiere criterio para casos límite).
    surnames = []
    for label_disp, key, _ in bibitems:
        # Las entradas sin apellido de autor (etiqueta '[arXiv:....]' o similar,
        # p. ej. un preprint anónimo o corporativo) no participan del orden por
        # apellido: se omiten para no generar falsos positivos.
        if re.match(r"\s*arXiv:", label_disp, re.IGNORECASE):
            continue
        ms = re.match(r"([A-Za-zÀ-ÿ\\\"'{}]+)", label_disp)
        # Normaliza acentos LaTeX comunes para comparar (\"o -> o, etc.).
        raw = ms.group(1) if ms else label_disp
        norm = re.sub(r'\\[\'"`^~]?\{?([A-Za-z])\}?', r"\1", raw).lower()
        surnames.append((norm, key))
    for idx in range(1, len(surnames)):
        prev_name, prev_key = surnames[idx - 1]
        cur_name, cur_key = surnames[idx]
        if cur_name < prev_name:
            collector.add_inconsistency(
                f"[{rel}] BIBLIO orden alfabético roto: '{cur_key}' ({cur_name}) "
                f"aparece después de '{prev_key}' ({prev_name}); reordenar (§9)."
            )

    # (17) Sufijo a/b huérfano: un display '(YYYYb)' exige que exista '(YYYYa)'
    # del mismo primer autor; y un '(YYYYa)' exige un '(YYYYb)'. Detecta sufijos
    # sueltos (p.ej. '2026b' sin '2026a').
    suffixed: dict[str, set[str]] = {}
    for label_disp, key, _ in bibitems:
        m = re.match(r"([A-Za-zÀ-ÿ]+).*?\((\d{4})([a-z])\)", label_disp)
        if m:
            base = f"{m.group(1).lower()}{m.group(2)}"
            suffixed.setdefault(base, set()).add(m.group(3))
    for base, letters in suffixed.items():
        if "b" in letters and "a" not in letters:
            collector.add_inconsistency(
                f"[{rel}] BIBLIO sufijo huérfano: existe '{base}b' pero no '{base}a'; "
                "quitar el sufijo si hay una sola entrada de ese autor-año (§9)."
            )
        if "a" in letters and "b" not in letters:
            collector.add_inconsistency(
                f"[{rel}] BIBLIO sufijo huérfano: existe '{base}a' pero no '{base}b'; "
                "quitar el sufijo si hay una sola entrada de ese autor-año (§9)."
            )

    # (18) Correspondencia \cite <-> \bibitem (steering §9: toda cita con bibitem
    # y todo bibitem citado). Se recolectan las claves de todas las variantes de
    # cita (\cite, \citep, \citet, \citeauthor, \citeyear), admitiendo el
    # argumento opcional [..] y múltiples claves separadas por coma
    # (\citep{a, b, c}). Los comentarios se eliminan para no contar claves que
    # solo aparecen en notas (p. ej. '% eliminado sumeet2025').
    code_only = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in text.split("\n"))
    bibitem_keys = {key for _, key, _ in bibitems}
    cited_keys: set[str] = set()
    for m in re.finditer(r"\\cite[a-z]*\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", code_only):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                cited_keys.add(k)
    # (a) citas sin \bibitem -> saldrán como [?] y rompen la trazabilidad.
    missing_bibitem = sorted(cited_keys - bibitem_keys)
    if missing_bibitem:
        collector.add_inconsistency(
            f"[{rel}] BIBLIO cita sin \\bibitem (saldrá como [?]): "
            f"{', '.join(missing_bibitem)}; añadir la entrada o corregir la clave (§9)."
        )
    # (b) \bibitem nunca citado -> entrada huérfana que debe eliminarse o citarse.
    uncited = sorted(bibitem_keys - cited_keys)
    if uncited:
        collector.add_inconsistency(
            f"[{rel}] BIBLIO \\bibitem sin cita (entrada huérfana): "
            f"{', '.join(uncited)}; citarla en el texto o eliminarla (§9)."
        )

    # (19) Entradas sin datos de localización (§9, informe §6). Una entrada que
    # nombra una revista/conferencia real pero no da páginas, ni volumen, ni DOI,
    # ni número de arXiv queda incompleta: el tribunal no puede localizarla.
    # Heurística conservadora: se activa solo si (a) el body menciona un lugar de
    # publicación conocido, y (b) no hay ningún localizador (páginas 'NN--MM' o
    # ', NN.'/'NNNNNN.', DOI, arXiv). Los preprints puros de arXiv quedan cubiertos
    # por su propio identificador y no se marcan.
    venue_re = re.compile(
        r"Nature\s+Physics|Nature\s+Communications|Physical\s+Review|npj\b|PRX|PRL|"
        r"Proceedings|ICLR|ICML|NeurIPS|Physics\s+Reports|Science\b",
        re.IGNORECASE,
    )
    # Un libro (editorial universitaria / 'edition') no necesita páginas ni DOI: la
    # editorial es localización suficiente. Se excluye para no marcar libros.
    book_re = re.compile(r"University\s+Press|\bPress\b|edition|editorial", re.IGNORECASE)
    locator_re = re.compile(
        r"\d+\s*--\s*\d+|"  # rango de páginas NN--MM
        r"\d+\s*\(\d+\)|"  # volumen con número: '377(6613)'
        r",\s*[A-Za-z]?\d{2,}|"  # 'volumen/artículo' alfanumérico: ', 106', ', L060401', ', eabk3333'
        r",\s*[a-z]+\d+|"  # artículo alfanumérico: ', eabk3333'
        r"10\.\d{4,9}/|doi|"  # DOI
        r"arXiv:\d{4}\.\d{4,5}|arXiv:[a-z\-]+/\d{7}",  # arXiv (nuevo o viejo)
        re.IGNORECASE,
    )
    for label_disp, key, body in bibitems:
        if venue_re.search(body) and not book_re.search(body) and not locator_re.search(body):
            collector.add_inconsistency(
                f"[{rel}] BIBLIO entrada sin datos de localización: '{key}' nombra un "
                "lugar de publicación pero no da volumen/páginas/DOI/arXiv; completar (§9)."
            )

    # (20) Ancla de reproducibilidad (informe §6): el bloque que da la URL del
    # repositorio debe acompañarse de la versión (tag o hash del commit), la fecha
    # de consulta y la licencia, para que el código sea verificable. Se localiza el
    # \url{...github...} y se comprueba que en su entorno aparezcan esos tres datos.
    m_repo = re.search(r"\\url\{[^}]*github[^}]*\}", code_only)
    if m_repo:
        ventana = code_only[m_repo.start() : m_repo.start() + 700]
        falta = []
        if not re.search(r"\bv\d[\d.]*|commit|hash|tag|[0-9a-f]{7,40}\b", ventana):
            falta.append("versión/hash del commit")
        if not re.search(r"consult|acceso|fecha|202\d", ventana):
            falta.append("fecha de consulta")
        if not re.search(r"licencia|MIT|Apache|GPL|BSD|licen[cs]e", ventana, re.IGNORECASE):
            falta.append("licencia")
        if falta:
            collector.add_inconsistency(
                f"[{rel}] BIBLIO ancla de reproducibilidad incompleta: el repositorio no "
                f"declara {', '.join(falta)}; añadir para que el código sea verificable (§9)."
            )

    # (21) Preprints presentados como publicados (informe §6). Si el cuerpo afirma
    # "N trabajos publicados" pero varias de las entradas citadas son preprints de
    # arXiv, la afirmación es imprecisa. Señal conservadora: solo la frase.
    if re.search(r"trabajos?\s+publicados?", code_only, re.IGNORECASE):
        n_preprints = sum(
            1 for _, _, body in bibitems if re.search(r"arXiv preprint|preprint\s+arXiv", body, re.IGNORECASE)
        )
        if n_preprints >= 3:
            collector.add_inconsistency(
                f"[{rel}] BIBLIO 'trabajos publicados' con {n_preprints} preprints de arXiv en "
                "la bibliografía; usar 'trabajos independientes' o distinguir publicados de "
                "preprints (§9, informe §6)."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-fix: decimales con punto -> coma (con exclusiones seguras)
# ═══════════════════════════════════════════════════════════════════════════════

# Contextos donde un punto decimal NO debe tocarse (valores literales/técnicos).
_VERSION_RE = re.compile(r"[vV]?\d$|Qiskit|Python|arXiv|v\d")


def fix_decimals(tex_path: Path, collector: TodoCollector) -> int:
    """Convierte 'N.M' -> 'N,M' en el .tex, excluyendo contextos técnicos.

    Reglas de exclusión (no se tocan):
      - Dentro de \\texttt{...}, \\url{...}, \\ref/\\cite/\\label, o comentarios.
      - Exponentes (10^{-14}, 1e-3), versiones (Qiskit 2.x), arXiv, URLs.
      - h_c = 1.0 y valores de campo crítico (contexto físico canónico).
      - Números con 3+ partes (1.2.3, IPs, versiones).

    Escribe un backup .bak y registra cada cambio en el TODO log. Devuelve el
    número de sustituciones aplicadas.
    """
    original = tex_path.read_text(encoding="utf-8")
    lines = original.split("\n")
    rel = tex_path.name
    n_fixed = 0
    out_lines: list[str] = []

    # Comandos cuyo argumento no debe tocarse
    protect_cmd = re.compile(r"\\(?:texttt|url|href|ref|eqref|autoref|cite[tp]?|label|input|includegraphics)\{[^}]*\}")
    # Número decimal candidato: entero.decimales, no seguido/precedido de otro punto o dígito extra
    dec_re = re.compile(r"(?<![\w.])(\d+)\.(\d+)(?![\w.])")

    def make_repl(
        code: str,
        idx: int,
        spans: list[tuple[int, int]],
        math_spans: list[tuple[int, int]],
    ):
        """Crea el reemplazador para una línea concreta (bind explícito).

        En prosa: 'N.M' -> 'N,M'. Dentro de math $...$: 'N.M' -> 'N{,}M' (las
        llaves evitan el espacio espurio que LaTeX inserta tras una coma en modo
        matemático, tratándola como separador de lista).
        """
        line_has_url = "arXiv" in code or "http" in code or "github" in code.lower()

        def repl(mm: re.Match) -> str:
            nonlocal n_fixed
            start = mm.start()
            before = code[max(0, start - 12) : start]
            after = code[mm.end() : mm.end() + 14]
            whole = mm.group(0)
            if any(a <= start < b for a, b in spans):  # dentro de comando protegido
                return whole
            if "^" in before[-3:] or "10^" in before or "times 10" in before:
                return whole
            if re.search(r"[eE]$", before) and re.search(r"^\d", after):  # 1e-3
                return whole
            if _VERSION_RE.search(before) or line_has_url:
                return whole
            # Longitudes LaTeX (width=0.90\textwidth, 0.5cm, 2.5pt): no tocar.
            if re.match(r"\s*(\\(?:text|line|column)width|\\height|cm|mm|pt|em|ex|in)\b", after):
                return whole
            in_math = any(a <= start < b for a, b in math_spans)
            sep = "{,}" if in_math else ","
            fixed = mm.group(1) + sep + mm.group(2)
            n_fixed += 1
            collector.add_inconsistency(f"[{rel}:{idx}] FIX aplicado: '{whole}' -> '{fixed}' (decimal a coma).")
            return fixed

        return repl

    # Math inline $...$: los decimales SÍ se convierten, pero con '{,}' para
    # evitar el espaciado espurio de la coma en modo matemático.
    math_re = re.compile(r"(?<!\\)\$[^$]*\$")

    for idx, line in enumerate(lines, start=1):
        m = re.search(r"(?<!\\)%", line)
        comment_pos = m.start() if m else None
        code = line if comment_pos is None else line[:comment_pos]
        tail = "" if comment_pos is None else line[comment_pos:]
        spans = [mm.span() for mm in protect_cmd.finditer(code)]
        math_spans = [mm.span() for mm in math_re.finditer(code)]
        new_code = dec_re.sub(make_repl(code, idx, spans, math_spans), code)
        out_lines.append(new_code + tail)

    if n_fixed == 0:
        return 0

    # Backup no destructivo: no pisar un .bak previo (preserva el original).
    backup = tex_path.with_suffix(tex_path.suffix + ".bak")
    if backup.exists():
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = tex_path.with_suffix(tex_path.suffix + f".{stamp}.bak")
    backup.write_text(original, encoding="utf-8")
    tex_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"  🔧 fix-decimals: {n_fixed} sustituciones (backup en {backup.name})")
    return n_fixed


def fix_tone(tex_path: Path, collector: TodoCollector) -> int:
    """Aplica las sustituciones DETERMINISTAS de tono (TONE_FIXES), steering §6.

    Solo toca frases fijas cuyo reemplazo no depende del contexto (no altera el
    sentido). Casos que requieren criterio (demuestra/garantiza, adjetivos, "dentro
    de las configuraciones evaluadas") NO se auto-corrigen: se dejan como señal.
    Crea backup .bak (no destructivo) y registra cada cambio en el TODO log.
    """
    original = tex_path.read_text(encoding="utf-8")
    text = original
    rel = tex_path.name
    n_fixed = 0
    for needle, repl in TONE_FIXES:
        # Case-insensitive pero preservando: solo aplicamos en minúscula/tal cual.
        for m in list(re.finditer(re.escape(needle), text, re.IGNORECASE)):
            frag = m.group(0)
            n_fixed += text.count(frag)
            text = text.replace(frag, repl)
            collector.add_inconsistency(f"[{rel}] FIX-TONO aplicado: '{frag}' -> '{repl}' (steering §6).")
    if n_fixed == 0:
        return 0
    backup = tex_path.with_suffix(tex_path.suffix + ".bak")
    if backup.exists():
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = tex_path.with_suffix(tex_path.suffix + f".{stamp}.bak")
    backup.write_text(original, encoding="utf-8")
    tex_path.write_text(text, encoding="utf-8")
    print(f"  🔧 fix-tone: {n_fixed} sustituciones (backup en {backup.name})")
    return n_fixed


def fix_anglicisms(tex_path: Path, collector: TodoCollector) -> int:
    """Traduce anglicismos de reemplazo unívoco (ANGLICISM_FIXES), steering §3.

    Opera línea a línea y SOLO en prosa: excluye \\texttt{...}, math $...$,
    comandos con argumento (ref/cite/label/url/includegraphics), comentarios, el
    capítulo Abstract y la bibliografía (inglés legítimo). Para cada línea, calcula
    las zonas protegidas y aplica el reemplazo únicamente en los tramos de prosa,
    preservando mayúscula inicial cuando el original la tenía. Los términos que
    requieren criterio (ANGLICISM_DETECT) o se conservan (ANGLICISM_KEEP) NO se
    tocan. Crea backup .bak (no destructivo) y registra cada cambio.
    """
    original = tex_path.read_text(encoding="utf-8")
    lines = original.split("\n")
    rel = tex_path.name
    n_fixed = 0
    out_lines: list[str] = []
    in_verbatim = False
    in_english = False
    in_acronyms = False

    def _apply_line(code: str, idx: int) -> str:
        nonlocal n_fixed
        # Zonas protegidas de ESTA línea (spans a no tocar).
        spans = [m.span() for m in _PROTECT_CMD_RE.finditer(code)]
        spans += [m.span() for m in _MATH_INLINE_RE.finditer(code)]

        def _protected(pos: int) -> bool:
            return any(a <= pos < b for a, b in spans)

        new = code
        # Reaplicar sobre 'new' desplaza índices; por eso se procesa patrón a
        # patrón reconstruyendo la cadena y recomputando spans tras cada cambio.
        for pat, repl in ANGLICISM_FIXES:
            rebuilt = []
            last = 0
            changed = False
            for m in re.finditer(pat, new):
                if _protected(m.start()):
                    continue
                # Preservar mayúscula inicial del original.
                frag = m.group(0)
                replacement = repl
                if frag[:1].isupper():
                    replacement = repl[:1].upper() + repl[1:]
                rebuilt.append(new[last : m.start()])
                rebuilt.append(replacement)
                last = m.end()
                changed = True
                n_fixed += 1
                collector.add_inconsistency(
                    f"[{rel}:{idx}] FIX-ANGLICISMO aplicado: '{frag}' -> '{replacement}' (steering §3)."
                )
            if changed:
                rebuilt.append(new[last:])
                new = "".join(rebuilt)
                # Recomputar spans protegidos tras la reescritura de la línea.
                spans = [m.span() for m in _PROTECT_CMD_RE.finditer(new)]
                spans += [m.span() for m in _MATH_INLINE_RE.finditer(new)]
        return new

    for idx, line in enumerate(lines, start=1):
        if re.search(r"\\begin\{(verbatim|lstlisting)\}", line):
            in_verbatim = True
        if re.search(r"\\end\{(verbatim|lstlisting)\}", line):
            in_verbatim = False
            out_lines.append(line)
            continue
        if re.search(r"\\chapter\{Abstract\}|\\begin\{thebibliography\}", line):
            in_english = True
        if re.search(r"\\mainmatter|\\end\{thebibliography\}", line):
            in_english = False
        # El índice de acrónimos contiene las expansiones inglesas de las siglas
        # (\emph{Machine Learning}, etc.): son definiciones legítimas, no prosa a
        # traducir. Se protege igual que Abstract/bibliografía.
        if re.search(r"Índice de acrónimos", line):
            in_acronyms = True
        if in_acronyms and re.search(r"\\end\{description\}", line):
            in_acronyms = False
            out_lines.append(line)
            continue
        if in_verbatim or in_english or in_acronyms:
            out_lines.append(line)
            continue
        # Separar comentario para no tocarlo.
        m = re.search(r"(?<!\\)%", line)
        comment_pos = m.start() if m else None
        code = line if comment_pos is None else line[:comment_pos]
        tail = "" if comment_pos is None else line[comment_pos:]
        out_lines.append(_apply_line(code, idx) + tail)

    if n_fixed == 0:
        return 0
    backup = tex_path.with_suffix(tex_path.suffix + ".bak")
    if backup.exists():
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = tex_path.with_suffix(tex_path.suffix + f".{stamp}.bak")
    backup.write_text(original, encoding="utf-8")
    tex_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"  🔧 fix-anglicisms: {n_fixed} sustituciones (backup en {backup.name})")
    return n_fixed


def fix_emdash(tex_path: Path, collector: TodoCollector) -> int:
    """Convierte incisos pareados '---texto---' en incisos con comas, steering §4.

    El usuario pide no usar guiones como separador. Regla SEGURA y determinista:
    solo se tocan PARES '---X---' dentro de una misma línea (un inciso completo),
    reemplazándolos por ', X,' (o ', X' si ya sigue coma/punto). Los '---' sueltos
    (impares en la línea, p.ej. la etiqueta 'Fase 1 --- datos' o un inciso que
    cierra con el punto final) NO se tocan: requieren criterio y se dejan como
    detección. Se excluyen comentarios, \\texttt{...} y math $...$. Crea backup .bak.
    """
    original = tex_path.read_text(encoding="utf-8")
    lines = original.split("\n")
    rel = tex_path.name
    n_fixed = 0
    out_lines: list[str] = []

    # Par de em-dash con contenido en el medio, sin '---' anidado ni fin de línea.
    pair_re = re.compile(r"---(?P<inner>(?:(?!---).)+?)---")

    for idx, line in enumerate(lines, start=1):
        # No tocar líneas de comentario (incluye los separadores %-----).
        stripped = line.lstrip()
        if stripped.startswith("%"):
            out_lines.append(line)
            continue
        m = re.search(r"(?<!\\)%", line)
        comment_pos = m.start() if m else None
        code = line if comment_pos is None else line[:comment_pos]
        tail = "" if comment_pos is None else line[comment_pos:]

        # Zonas protegidas (código/math): no reemplazar pares que las crucen.
        spans = [mm.span() for mm in _PROTECT_CMD_RE.finditer(code)]
        spans += [mm.span() for mm in _MATH_INLINE_RE.finditer(code)]

        def _in_protected(a: int, b: int) -> bool:
            return any(pa <= a and b <= pb for pa, pb in spans)

        def _repl(mm: re.Match) -> str:
            nonlocal n_fixed
            if _in_protected(mm.start(), mm.end()):
                return mm.group(0)
            inner = mm.group("inner")
            # El inciso equivale a comas: ', inner,'. La coma de apertura solo si
            # el carácter previo no es ya un espacio+coma o apertura.
            after_pos = mm.end()
            after = code[after_pos : after_pos + 1]
            close = "" if after in (",", ".", ";", ":", ")") else ","
            n_fixed += 1
            collector.add_inconsistency(f"[{rel}:{idx}] FIX-GUION aplicado: inciso '---...---' -> comas.")
            return f", {inner}{close}"

        new_code = pair_re.sub(_repl, code)
        # Limpiar dobles espacios o ' ,' introducidos por el reemplazo.
        new_code = re.sub(r"\s+,", ",", new_code)
        new_code = re.sub(r",\s*,", ",", new_code)
        out_lines.append(new_code + tail)

    if n_fixed == 0:
        return 0
    backup = tex_path.with_suffix(tex_path.suffix + ".bak")
    if backup.exists():
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = tex_path.with_suffix(tex_path.suffix + f".{stamp}.bak")
    backup.write_text(original, encoding="utf-8")
    tex_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"  🔧 fix-emdash: {n_fixed} incisos pareados convertidos (backup en {backup.name})")
    return n_fixed


def fix_grammar(tex_path: Path, collector: TodoCollector) -> int:
    """Aplica correcciones gramaticales deterministas (GRAMMAR_FIXES), steering §4.

    Solo frases fijas con reemplazo unívoco ("en base a" -> "a partir de";
    "en función a" -> "en función de"). No toca contextos protegidos ni cambia
    concordancia. Crea backup .bak y registra cada cambio.
    """
    original = tex_path.read_text(encoding="utf-8")
    text = original
    rel = tex_path.name
    n_fixed = 0
    for pat, repl in GRAMMAR_FIXES:
        for m in list(re.finditer(pat, text)):
            frag = m.group(0)
            n_fixed += 1
            collector.add_inconsistency(f"[{rel}] FIX-GRAMATICA aplicado: '{frag}' -> '{repl}' (steering §4).")
        text = re.sub(pat, repl, text)
    if n_fixed == 0:
        return 0
    backup = tex_path.with_suffix(tex_path.suffix + ".bak")
    if backup.exists():
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = tex_path.with_suffix(tex_path.suffix + f".{stamp}.bak")
    backup.write_text(original, encoding="utf-8")
    tex_path.write_text(text, encoding="utf-8")
    print(f"  🔧 fix-grammar: {n_fixed} sustituciones (backup en {backup.name})")
    return n_fixed


# ═══════════════════════════════════════════════════════════════════════════════
# Compilación LaTeX + señales visuales (18/19)
# ═══════════════════════════════════════════════════════════════════════════════

_STUB_STY = r"""\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{estilo_unir-1}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage[spanish]{babel}
\usepackage[draft]{graphicx}\usepackage{amsmath,amssymb}\usepackage{natbib}
\usepackage{hyperref}\usepackage{geometry}\usepackage{booktabs}\usepackage{multirow}\usepackage{siunitx}
\newcommand{\subject}[1]{\gdef\@subject{#1}}\newcommand{\profesor}[1]{\gdef\@profesor{#1}}
\providecommand{\@subject}{}\providecommand{\@profesor}{}
\renewcommand{\maketitle}{\begin{titlepage}\centering{\huge\@title\par}\end{titlepage}}
"""


def compile_tex(tex_path: Path, out_dir: Path, collector: TodoCollector, overfull_pt: float = 20.0) -> None:
    """(18) Compila el .tex con pdflatex y (19) reporta señales visuales.

    Usa un stub de ``estilo_unir-1.sty`` (el .sty real vive fuera del repo), copia
    las tablas ``auto_*`` y compila 2 pasadas en un tempdir. Vuelca al TODO log:
      - errores fatales y referencias/citas indefinidas (18),
      - overfull hboxes por encima del umbral y figuras no encontradas (19),
    como señales para revisión visual. Requiere pdflatex en el PATH.
    """
    import shutil
    import subprocess
    import tempfile

    if shutil.which("pdflatex") is None:
        collector.add_inconsistency("[compile] pdflatex no está en el PATH; no se pudo compilar (18/19).")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "tables").mkdir(exist_ok=True)
        shutil.copy(tex_path, tmp_path / tex_path.name)
        for auto in out_dir.glob("auto_*.tex"):
            shutil.copy(auto, tmp_path / "tables" / auto.name)
        (tmp_path / "estilo_unir-1.sty").write_text(_STUB_STY, encoding="utf-8")
        # Stub vacío de icomma: el paquete real solo ajusta el espaciado de la coma
        # decimal en modo matemático (irrelevante para la validación estructural).
        # Sin este shim, la compilación falla si icomma no está instalado en el TeX
        # local, aunque el .tex real de UNIR sí lo tenga disponible.
        (tmp_path / "icomma.sty").write_text("\\NeedsTeXFormat{LaTeX2e}\\ProvidesPackage{icomma}\n", encoding="utf-8")
        # Copiar carpetas de figuras si existen junto al .tex (para señales reales).
        for figdir in ("tesis-figures", "thesis_plots"):
            src = tex_path.parent / figdir
            if src.is_dir():
                shutil.copytree(src, tmp_path / figdir, dirs_exist_ok=True)

        log = ""
        try:
            for _ in range(2):
                proc = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", tex_path.name],
                    cwd=tmp_path,
                    capture_output=True,
                    timeout=180,
                )
                log = proc.stdout.decode("utf-8", "replace") + proc.stderr.decode("utf-8", "replace")
        except subprocess.TimeoutExpired:
            collector.add_inconsistency(
                "[compile] pdflatex superó el timeout (180s); posible espera de entrada "
                "por un error no recuperable. Revisar el .tex manualmente (18)."
            )
            print("  📄 compile: TIMEOUT (pdflatex colgado)")
            return
        except (OSError, ValueError) as e:
            collector.add_inconsistency(f"[compile] fallo al ejecutar pdflatex: {e}")
            return

        rel = tex_path.name
        # Errores fatales
        fatals = re.findall(r"^! (.+)$", log, re.MULTILINE)
        for f in fatals[:10]:
            collector.add_inconsistency(f"[compile] error LaTeX: {f.strip()[:120]}")
        # Referencias/citas indefinidas
        for m in re.findall(r"(?:Reference|Citation) `([^']+)' (?:on page \S+ )?undefined", log):
            collector.add_inconsistency(f"[compile] referencia/cita indefinida: {m} (18).")
        undef_generic = len(re.findall(r"There were undefined references", log))
        if undef_generic and not fatals:
            collector.add_inconsistency("[compile] el log reporta referencias indefinidas; correr otra pasada (18).")
        # (19) Señales visuales: overfull hboxes grandes
        overs = re.findall(r"Overfull \\hbox \((\d+(?:\.\d+)?)pt too wide\)[^\n]*at lines (\d+)", log)
        big = [(float(pt), ln) for pt, ln in overs if float(pt) >= overfull_pt]
        for pt, ln in big[:15]:
            collector.add_inconsistency(
                f"[{rel}:{ln}] SEÑAL-VISUAL overfull hbox {pt:.0f}pt (texto se sale del "
                "margen; revisar visualmente esa página, §19)."
            )
        # (19) Figuras no encontradas (afectan el render)
        missing_figs = sorted(set(re.findall(r"File `([^']+)' not found", log)))
        for fig in missing_figs[:15]:
            collector.add_inconsistency(
                f"[compile] SEÑAL-VISUAL figura no encontrada: {fig} (no renderiza; verificar ruta, §19)."
            )
        pdf_ok = (tmp_path / tex_path.with_suffix(".pdf").name).exists()
        n_over = len(big)
        print(
            f"  📄 compile: {'PDF OK' if pdf_ok else 'SIN PDF'} | "
            f"{len(fatals)} errores | {n_over} overfull>{overfull_pt:.0f}pt | "
            f"{len(missing_figs)} figuras faltantes"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Orquestación
# ═══════════════════════════════════════════════════════════════════════════════

GENERATORS = {
    "auto_scoreboard": ("scoreboard", gen_scoreboard),
    # auto_coverage (coverage matrix, tab:auto_coverage) removed from the thesis
    # — the grade-per-(topology×N) matrix is not used. gen_coverage() is kept for
    # ad-hoc use but no longer emitted.
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
        help="Lista de N (coma) para las columnas de auto_coverage (default: todos los N presentes)",
    )
    parser.add_argument(
        "--check-tex",
        type=Path,
        default=None,
        help="Chequear consistencia de un .tex (refs/labels, decimales, "
        "tablas auto no conectadas) y volcar hallazgos a tesis_todos.txt",
    )
    parser.add_argument(
        "--fix-decimals",
        type=Path,
        default=None,
        help="Convertir decimales con punto a coma en el .tex indicado "
        "(excluye h_c, exponentes, versiones, URLs, arXiv, comandos). "
        "Crea backup .bak y registra cada cambio en tesis_todos.txt.",
    )
    parser.add_argument(
        "--fix-tone",
        type=Path,
        default=None,
        help="Aplicar sustituciones deterministas de tono comercial (frases fijas: "
        "'el pipeline funciona', 'coste cuántico cero', etc.) en el .tex indicado. "
        "Crea backup .bak y registra cada cambio en tesis_todos.txt (steering §6).",
    )
    parser.add_argument(
        "--fix-anglicisms",
        type=Path,
        default=None,
        help="Traducir anglicismos SEGUROS de reemplazo unívoco (noiseless->"
        "simulación ideal, cross-N->entre tamaños, epochs->épocas, etc.) en el .tex "
        "indicado, solo en prosa (excluye código, math, Abstract y bibliografía). "
        "Los términos que cambian concordancia (ground truth, dataset, framework) "
        "solo se detectan. Crea backup .bak y registra cada cambio (steering §3).",
    )
    parser.add_argument(
        "--fix-grammar",
        type=Path,
        default=None,
        help="Corregir gramática determinista ('en base a'->'a partir de', "
        "'en función a'->'en función de') en el .tex indicado. Crea backup .bak "
        "y registra cada cambio (steering §4).",
    )
    parser.add_argument(
        "--fix-emdash",
        type=Path,
        default=None,
        help="Convertir incisos pareados '---texto---' en incisos con comas "
        "(el usuario pide no usar guiones como separador). Solo toca pares en una "
        "misma línea; los '---' sueltos (etiquetas, cierre con punto) se dejan como "
        "detección. Excluye comentarios/código/math. Crea backup .bak (steering §4).",
    )
    parser.add_argument(
        "--compile",
        dest="compile_tex",
        type=Path,
        default=None,
        help="Compilar el .tex con pdflatex (stub .sty) y volcar errores, "
        "citas/refs indefinidas y señales visuales (overfull, figuras faltantes) "
        "a tesis_todos.txt (18/19).",
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
    per_n = load_heavy_hex_per_h(collector) if only & {"auto_heavy_hex_intra_n", "auto_heavy_hex_large_n"} else {}

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
        written.append(_rel(out_path))
        print(f"  ✅ {_rel(out_path)}")

    # Auto-fix de decimales (opcional; se aplica antes del chequeo)
    if args.fix_decimals is not None:
        fix_decimals(args.fix_decimals, collector)

    # Auto-fix de tono determinista (opcional; antes del chequeo)
    if args.fix_tone is not None:
        fix_tone(args.fix_tone, collector)

    # Auto-fix de anglicismos univocos (opcional; antes del chequeo)
    if args.fix_anglicisms is not None:
        fix_anglicisms(args.fix_anglicisms, collector)

    # Auto-fix de gramatica determinista (opcional; antes del chequeo)
    if args.fix_grammar is not None:
        fix_grammar(args.fix_grammar, collector)

    # Auto-fix de incisos con em-dash (opcional; antes del chequeo)
    if args.fix_emdash is not None:
        fix_emdash(args.fix_emdash, collector)

    # Chequeo del documento LaTeX (opcional)
    if args.check_tex is not None:
        check_tex(args.check_tex, out_dir, collector)
        print(f"  🔍 Chequeo LaTeX: {args.check_tex}")

    # Compilación + señales visuales (opcional, 18/19)
    if args.compile_tex is not None:
        compile_tex(args.compile_tex, out_dir, collector)

    # Volcar tesis_todos.txt (informe completo, agrupado por severidad)
    todos_path = out_dir / "tesis_todos.txt"
    todos_path.write_text(collector.render_txt(), encoding="utf-8")
    print(f"  📝 {_rel(todos_path)} ({collector.n_todos()} TODOs, {len(collector.inconsistencies)} inconsistencias)")

    # Volcar checklist accionable de chequeos pendientes (Markdown marcable).
    checklist_path = out_dir / "tesis_checklist_pendientes.md"
    checklist_path.write_text(collector.render_checklist(), encoding="utf-8")
    print(f"  ✅ {_rel(checklist_path)} (checklist accionable por acción)")

    print(f"\n  {len(written)} tablas generadas en {_rel(out_dir)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
