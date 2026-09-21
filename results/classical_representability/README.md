# Classical representability study — resultados

Estudia la **viabilidad de métodos clásicos** (MPS/DMRG, Aer MPS, Haiqu VL) para
representar el ground state del TFIM en función de la **bond dimension** `chi`,
a N = 4, 10, 50, 100. Objetivo: ver hasta dónde escalan y dónde se rompen.

## Cómo se guardan los resultados

Un JSON por corrida, nombrado `study_{method}_{topology}_{timestamp}.json`.
Esquema `classical_representability_v1` (filas planas, estilo `dmrg_vs_exact`):

```json
{
  "schema": "classical_representability_v1",
  "method": "mps_dmrg | aer_mps | haiqu_vl",
  "topology": "chain_1d",
  "generated_utc": "ISO-8601",
  "config": {"N_values": [...], "h_values": [...], "chi_values": [...], "J": 1.0},
  "rows": [
    {
      "method": "mps_dmrg",
      "N": 10, "h": 1.0,
      "chi_requested": 8, "chi_actual": 8,
      "energy": -12.38149,
      "e_ref": -12.38149,          // referencia (exacto si N<=16, else mejor chi)
      "abs_error": 6.3e-11,        // |energy - e_ref|
      "trunc_error": 6.3e-11,      // error de truncamiento del MPS (eps)
      "entanglement_entropy": 0.379,  // S_vN max (corte central)
      "fidelity": null,            // |<psi_ref|psi>|^2 si disponible (N<=16)
      "time_s": 0.83,
      "mem_bytes_est": 40960,      // ~ N * chi^2 * 16 (complex128)
      "status": "ok | truncated | failed"
    }
  ]
}
```

Además, cada corrida agrega una línea a `index.jsonl` (append-only) con
`{method, topology, N_values, h_values, chi_values, path, generated_utc}` para
inventariar sin abrir cada JSON.

## Cómo se analiza

`python scripts/analysis/analyze_representability.py` consolida todos los JSON de
esta carpeta y produce:

- `SUMMARY.md` — tabla de `chi` necesario para converger por (N, h) + veredicto.
- `figures/E_vs_chi.png`, `entropy_vs_N.png`, `chi_needed_vs_N.png`, `mem_vs_chi.png`.

## Métrica clave

La **entropía de entrelazamiento** S_vN gobierna el `chi` necesario (chi ~ e^S).
Cerca del punto crítico (h=1) S crece con N; lejos (h=0.5, h=2.0) satura. Esa es
la firma que decide la viabilidad clásica: estados de bajo S son baratos; los de
alto S (crítico, N grande) requieren chi impracticable.
