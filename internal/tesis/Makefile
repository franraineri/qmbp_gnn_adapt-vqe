# ─────────────────────────────────────────────────────────────────────────────
# Makefile de la tesis (TFM GNN-HVA) — autocontenido en internal/tesis/
#
# Compila tesis-v4.0.tex a PDF, regenera las tablas auto desde las fuentes de
# verdad del proyecto y valida la consistencia del .tex. Todas las rutas son
# relativas a esta carpeta salvo el generador de tablas, que vive en el repo.
#
# Uso:
#   make            # = make pdf
#   make pdf        # compila el .tex a PDF (2 pasadas, resuelve refs/índice)
#   make tables     # regenera las tablas auto_*.tex desde el proyecto
#   make check      # valida consistencia del .tex (sin compilar)
#   make all        # tables + check + pdf (todo)
#   make view       # abre el PDF (macOS)
#   make overleaf   # regenera tablas y arma el ZIP para subir a Overleaf
#   make clean      # borra auxiliares LaTeX (deja el .pdf)
#   make cleanall   # borra auxiliares y el .pdf
# ─────────────────────────────────────────────────────────────────────────────

TEX      := tesis-v4.0.tex
PDF      := tesis-v4.0.pdf
ZIP      := tesis-v4.0-overleaf.zip
# Archivos que Overleaf necesita, con su ruta relativa (se preservan las carpetas).
ZIP_FILES := $(TEX) estilo_unir-1.sty icomma.sty logo_unir.png \
             tesis-figures/palette.tex \
             tesis-figures/fig_pipeline_tikz.tex \
             tesis-figures/fig_hva_circuit_tikz.tex \
             tesis-figures/fig_topologies_tikz.tex \
             tables/auto_campaign.tex \
             tables/auto_scoreboard.tex \
             tables/auto_heavy_hex_intra_n.tex \
             tables/auto_heavy_hex_large_n.tex
LATEX    := pdflatex -interaction=nonstopmode -file-line-error -halt-on-error
# Raíz del repo (dos niveles arriba de internal/tesis/) para invocar el generador.
# Absoluta, porque el generador se corre con `cd $(REPO)` y una ruta relativa al
# venv dejaría de resolver tras el cambio de directorio.
REPO     := $(abspath ../..)
PYTHON   := $(REPO)/.venv/bin/python
GEN      := scripts/general_project_maintenance/generate_thesis_tables.py
# Verificador de credibilidad numérica (cocientes de tablas, totales de campaña,
# semántica A/R, colisiones de símbolos, recursos faltantes). Complementa a GEN.
VERIFY_NUM := scripts/general_project_maintenance/verify_thesis_numbers.py
# Verificador sobre el PDF renderizado (casos absolutos, matriz de confusión).
VERIFY_PDF := scripts/general_project_maintenance/verify_thesis_pdf.py
# Las tablas auto viven en esta carpeta (tesis/tables), donde el .tex las incluye.
TABLES_DIR := $(CURDIR)/tables

# Auxiliares LaTeX a limpiar.
AUX := $(TEX:.tex=.aux) $(TEX:.tex=.log) $(TEX:.tex=.toc) $(TEX:.tex=.out) \
       $(TEX:.tex=.lof) $(TEX:.tex=.lot) $(TEX:.tex=.fls) $(TEX:.tex=.fdb_latexmk) \
       $(TEX:.tex=.bbl) $(TEX:.tex=.blg) $(TEX:.tex=.synctex.gz)

.PHONY: all pdf tables check verify-numbers verify-pdf verify view overleaf overleaf-zip clean cleanall help

pdf: $(PDF)  ## Compila el .tex a PDF (2 pasadas)

# El PDF depende del .tex, el estilo, el logo y las tablas/figuras incluidas.
$(PDF): $(TEX) estilo_unir-1.sty logo_unir.png $(wildcard tables/*.tex) \
        $(wildcard tesis-figures/*.tex) $(wildcard thesis_plots/*.png)
	@echo "═══ Compilando $(TEX) (pasada 1/2) ═══"
	$(LATEX) $(TEX) </dev/null
	@echo "═══ Compilando $(TEX) (pasada 2/2, resuelve refs e índice) ═══"
	$(LATEX) $(TEX) </dev/null
	@$(MAKE) --no-print-directory clean
	@echo "\n✅ PDF generado: internal/tesis/$(PDF)"

tables:  ## Regenera las tablas auto_*.tex desde las fuentes de verdad del proyecto
	@echo "═══ Regenerando tablas auto en $(TABLES_DIR) ═══"
	cd $(REPO) && $(PYTHON) $(GEN) --out-dir "$(TABLES_DIR)"

check:  ## Valida la consistencia del .tex (refs/labels, cifras, tablas) sin compilar
	@echo "═══ Chequeo de consistencia de $(TEX) ═══"
	cd $(REPO) && $(PYTHON) $(GEN) --no-refresh \
		--out-dir "$(TABLES_DIR)" \
		--check-tex "$(CURDIR)/$(TEX)"

verify-numbers:  ## Verifica credibilidad numérica (cocientes de tablas, total de campaña, semántica A/R, símbolos, recursos)
	@echo "═══ Verificación numérica y de fuentes de $(TEX) ═══"
	cd $(REPO) && $(PYTHON) $(VERIFY_NUM) \
		"$(CURDIR)/$(TEX)" --tables-dir "$(TABLES_DIR)"

verify-pdf: $(PDF)  ## Verifica cifras del PDF renderizado (casos absolutos, matriz de confusión) — requiere pdftotext
	@echo "═══ Verificación del PDF renderizado $(PDF) ═══"
	cd $(REPO) && $(PYTHON) $(VERIFY_PDF) "$(CURDIR)/$(PDF)"

all: tables check verify-numbers pdf  ## Todo: regenera tablas, valida, verifica cifras y compila el PDF

verify:  ## Compila (3 pasadas) y audita el log: refs sin resolver (??), citas indefinidas, labels duplicados, overfull graves
	@echo "═══ Compilando 3 pasadas (conserva el .log para auditar) ═══"
	@$(LATEX) $(TEX) </dev/null >/dev/null 2>&1 || { echo "❌ error de compilación (ver $(TEX:.tex=.log))"; exit 1; }
	@$(LATEX) $(TEX) </dev/null >/dev/null 2>&1
	@$(LATEX) $(TEX) </dev/null >/dev/null 2>&1
	@echo "\n═══ Auditoría del log LaTeX ═══"
	@log=$(TEX:.tex=.log); fail=0; \
	u=$$(grep -cE 'Reference.*undefined|Citation.*undefined' $$log); \
	echo "  Referencias/citas indefinidas: $$u"; [ "$$u" -eq 0 ] || fail=1; \
	m=$$(grep -cE 'multiply.defined|multiply-defined' $$log); \
	echo "  Labels duplicados:             $$m"; [ "$$m" -eq 0 ] || fail=1; \
	e=$$(grep -cE '^!|LaTeX Error|Emergency stop' $$log); \
	echo "  Errores LaTeX:                 $$e"; [ "$$e" -eq 0 ] || fail=1; \
	if command -v pdftotext >/dev/null 2>&1; then \
		q=$$(pdftotext $(PDF) - 2>/dev/null | grep -c '??'); \
		echo "  '??' visibles en el PDF:       $$q"; [ "$$q" -eq 0 ] || fail=1; \
	else echo "  '??' visibles en el PDF:       n/a (pdftotext no instalado)"; fi; \
	o=$$(grep -cE 'Overfull \\hbox \(([3-9][0-9]|[0-9]{3,})\.' $$log); \
	echo "  Overfull graves (>=30pt):      $$o (informativo)"; \
	echo ""; \
	if [ "$$fail" -eq 0 ]; then echo "✅ Auditoría OK: sin refs/citas indefinidas, sin labels duplicados, sin errores, sin ??"; \
	else echo "❌ Auditoría con hallazgos: revisar $$log"; exit 1; fi
	@echo "\n═══ Verificación numérica y de fuentes ═══"
	@cd $(REPO) && $(PYTHON) $(VERIFY_NUM) "$(CURDIR)/$(TEX)" --tables-dir "$(TABLES_DIR)"
	@echo "\n═══ Verificación del PDF renderizado ═══"
	@cd $(REPO) && $(PYTHON) $(VERIFY_PDF) "$(CURDIR)/$(PDF)"

view: $(PDF)  ## Abre el PDF (macOS)
	@open $(PDF)

overleaf: tables overleaf-zip  ## Regenera tablas y arma el ZIP listo para subir a Overleaf

overleaf-zip:  ## (Re)crea el ZIP de Overleaf sobreescribiendo el anterior y limpia temporales
	@echo "═══ Empaquetando $(ZIP) para Overleaf ═══"
	@for f in $(ZIP_FILES); do \
		[ -f "$$f" ] || { echo "❌ falta $$f"; exit 1; }; \
	done
	@rm -f $(ZIP)
	@zip -q -X $(ZIP) $(ZIP_FILES)
	@$(MAKE) --no-print-directory clean
	@echo "\n✅ ZIP actualizado: internal/tesis/$(ZIP) ($$(du -h $(ZIP) | cut -f1))"
	@echo "   Subir a Overleaf: New Project → Upload Project → $(ZIP)"

clean:  ## Borra auxiliares LaTeX (conserva el .pdf)
	@rm -f $(AUX)
	@echo "🧹 auxiliares LaTeX limpiados"

cleanall: clean  ## Borra auxiliares y el .pdf
	@rm -f $(PDF)
	@echo "🧹 PDF eliminado"

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
