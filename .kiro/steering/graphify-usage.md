---
inclusion: always
---

# Graphify — code intelligence para este proyecto

Graphify es un grafo de conocimiento (SQLite) de cada símbolo, arista y archivo del
código fuente. Ya está integrado y construido para `qmbp_gnn_adapt-vqe`.

- Índice: `~/.kiro/graphify/qmbp_gnn_adapt-vqe/graph.db`
- Estado actual: ~6949 nodos, ~12344 edges, ~483 communities
- Exclusiones: `.graphifyignore` en la raíz (excluye `.venv`, `data`, `results`, etc.)
- Server MCP: `graphify` en `~/.kiro/settings/mcp.json`, lanzado con
  `koda graphify mcp -d <este-proyecto> -p qmbp_gnn_adapt-vqe` (los flags `-d`/`-p`
  son obligatorios: sin ellos el comando es interactivo y se cuelga)

Este proyecto ya prohíbe grep/find (ver `reuse-workflow`). Graphify es la vía para
buscar código, símbolos y referencias.

## Las mejores utilidades (verificadas)

Ordenadas por utilidad práctica en este repo:

1. `graphify_explore` — Tool PRIMARIA. Query en lenguaje natural o nombres de
   símbolo/archivo. Devuelve el source verbatim con números de línea agrupado por
   archivo, más los módulos/communities y las relaciones. Una llamada suele
   responder toda la pregunta. Trata su salida como Read: no vuelvas a abrir esos
   archivos.

2. `graphify_inspect` — Skeleton de un archivo: todas las firmas, docstrings y
   rangos de línea sin cuerpos. Ideal para archivos god-node grandes como
   `runner_base.py` (148 símbolos) sin traer miles de líneas.

3. `graphify_source` — Cuerpo exacto de un símbolo por nombre o Node ID. Acepta
   `context` para líneas extra arriba/abajo.

4. `graphify_hotspots` — Los archivos más acoplados (god nodes). En este repo los
   top son: `framework/runner_base.py`, `analysis/metrics.py`,
   `predictors/model_zoo.py`. Útil antes de refactorizar.

5. `graphify_impact` — Blast radius de un símbolo (qué se rompe si lo cambiás).
   `depth` hasta 5. Úsalo antes de tocar helpers compartidos del `runner_base`.

6. `graphify_callers` / `graphify_callees` — Aristas entrantes/salientes de un
   símbolo (quién lo llama / qué llama). Requiere Node ID.

7. `graphify_community` — Lista los archivos de un módulo/community por ID o label.

8. `graphify_status` — Salud y frescura del índice.

9. `graphify_reindex` — Fuerza un re-index completo.

## Flujo recomendado

1. Cualquier pregunta sobre código → `graphify_explore`.
2. ¿Necesitás el esqueleto completo de un archivo? → `graphify_inspect`.
3. ¿El cuerpo de una función puntual? → `graphify_source`.
4. Antes de editar un helper compartido → `graphify_impact` para ver qué afecta.
5. Node IDs: se derivan del path, p.ej. `src/qmbp_simulation/framework/runner_base.py`
   → `src_qmbp_simulation_framework_runner_base_py`, y un símbolo dentro añade el
   nombre en minúsculas: `..._runner_base_py_hamiltonianbuilder`.

## Operación (evitar el cuelgue)

El MCP se colgaba antes de esta integración. Reglas para mantenerlo sano:

- El `.graphifyignore` es obligatorio. Sin él, graphify camina `.venv` (decenas de
  miles de `.py` de torch/qiskit) y nunca termina de indexar. No lo borres.
- NUNCA corras `koda graphify` sin argumentos: es interactivo (pide elegir proyecto
  del workspace activo) y se cuelga esperando stdin.
- Para (re)construir el índice desde CLI, apuntá explícitamente al directorio:
  `koda graphify mcp -d <ruta-proyecto> -p qmbp_gnn_adapt-vqe`.
- macOS no trae `timeout`. Para comandos que puedan colgarse, corrélos en background
  y cortalos si se atascan.
- Si un reindex falla con `SQLITE_BUSY` / `database is locked`: hay un proceso koda
  huérfano tocando la db. Solución:
  `pkill -9 -f "koda graphify"` y borrar `graph.db` + `graph.db-journal` antes de
  reintentar.
- Verificar progreso del indexado:
  `sqlite3 ~/.kiro/graphify/qmbp_gnn_adapt-vqe/graph.db "SELECT COUNT(*) FROM nodes;"`
  Con `.graphifyignore` puesto, indexa en ~30s.
- Si las tools del IDE dicen "Not connected", reconectá el server `graphify` desde el
  panel MCP de Kiro (o recargá la ventana). El índice ya existe, así que responde al
  instante.

## Habilitar graphify en un agente

Un agente solo "ve" las tools MCP que declara. Si un agente no detecta graphify:

- Agregá los nombres de tool `graphify_*` a las listas `tools` y `allowedTools` del
  JSON del agente (`~/.kiro/agents/<agente>.json`). Nombres pelados, sin prefijo:
  `graphify_explore`, `graphify_inspect`, `graphify_source`, `graphify_callers`,
  `graphify_callees`, `graphify_impact`, `graphify_community`, `graphify_hotspots`,
  `graphify_status`, `graphify_reindex`.
- Alternativa amplia: `"includeMcpJson": true` expone TODOS los servers MCP (como hace
  `developer.json`). Menos control que la allowlist explícita.
- Opcional: en `toolsSettings` marcá `"graphify_*": { "autoAllow": true }` para las
  read-only (todas menos `reindex`) y evitar confirmaciones.
- El agente `quantum` (usado en este repo) ya quedó configurado con las 10 tools.
