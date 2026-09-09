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
#   make clean      # borra auxiliares LaTeX (deja el .pdf)
#   make cleanall   # borra auxiliares y el .pdf
# ─────────────────────────────────────────────────────────────────────────────

TEX      := tesis-v4.0.tex
PDF      := tesis-v4.0.pdf
LATEX    := pdflatex -interaction=nonstopmode -file-line-error -halt-on-error
# Raíz del repo (dos niveles arriba de internal/tesis/) para invocar el generador.
# Absoluta, porque el generador se corre con `cd $(REPO)` y una ruta relativa al
# venv dejaría de resolver tras el cambio de directorio.
REPO     := $(abspath ../..)
PYTHON   := $(REPO)/.venv/bin/python
GEN      := scripts/general_project_maintenance/generate_thesis_tables.py
# Las tablas auto viven en esta carpeta (tesis/tables), donde el .tex las incluye.
TABLES_DIR := $(CURDIR)/tables

# Auxiliares LaTeX a limpiar.
AUX := $(TEX:.tex=.aux) $(TEX:.tex=.log) $(TEX:.tex=.toc) $(TEX:.tex=.out) \
       $(TEX:.tex=.lof) $(TEX:.tex=.lot) $(TEX:.tex=.fls) $(TEX:.tex=.fdb_latexmk) \
       $(TEX:.tex=.bbl) $(TEX:.tex=.blg) $(TEX:.tex=.synctex.gz)

.PHONY: all pdf tables check view clean cleanall help

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

all: tables check pdf  ## Todo: regenera tablas, valida y compila el PDF

view: $(PDF)  ## Abre el PDF (macOS)
	@open $(PDF)

clean:  ## Borra auxiliares LaTeX (conserva el .pdf)
	@rm -f $(AUX)
	@echo "🧹 auxiliares LaTeX limpiados"

cleanall: clean  ## Borra auxiliares y el .pdf
	@rm -f $(PDF)
	@echo "🧹 PDF eliminado"

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
