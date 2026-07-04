#!/usr/bin/env python3
"""Análisis de errores reales vs mitigados en ejecuciones QESEM.

Compara:
1. E_exact (energía exacta teórica)
2. E_noisy (lo que midió el QPU sin mitigación — error real del hardware)
3. E_mitigated (lo que QESEM corrigió — error residual post-mitigación)

Para los 3 jobs QESEM disponibles.
"""

import numpy as np

# ─── Parámetros del experimento (TFIM N=10 p=1) ───────────────────
E_EXACT_H4 = -40.565690435512735
GAP_H4 = 5.921971082752528

print("=" * 78)
print("  ANÁLISIS DE ERRORES DE HARDWARE vs MITIGACIÓN QESEM")
print("  TFIM 1D N=10, p=1 HVA, ibm_kingston")
print("=" * 78)

# ─── Job 1: 82aa (h=4.0, precision=0.01) ──────────────────────────
print("\n" + "─" * 78)
print("  JOB 82aa33cc — h = 4.0 (deep paramagnetic phase)")
print("─" * 78)

e_exact = E_EXACT_H4
e_noisy = -38.47114285714286
e_mitigated = -40.52391074682875
e_std = 0.2879999853310408

# Errores
error_hardware = abs(e_noisy - e_exact)
error_mitigated = abs(e_mitigated - e_exact)
error_recovered = error_hardware - error_mitigated
gain_pct = (error_recovered / error_hardware) * 100 if error_hardware > 0 else 0

print(f"\n  Energía exacta (diag. exacta):     E_exact     = {e_exact:.6f}")
print(f"  Energía cruda del QPU (sin mitig.): E_noisy     = {e_noisy:.6f}")
print(f"  Energía mitigada por QESEM:         E_mitigated = {e_mitigated:.6f} ± {e_std:.4f}")
print("\n  ─── Desglose de errores ───")
print(f"  Error total del hardware:     |E_noisy - E_exact|     = {error_hardware:.6f}")
print(f"  Error residual (post-QESEM):  |E_mitigated - E_exact| = {error_mitigated:.6f}")
print(f"  Error corregido por QESEM:    {error_recovered:.6f}")
print(f"  Ganancia de mitigación:       {gain_pct:.1f}%")
print(
    f"\n  ΔE/gap (hardware sin mitigar): {error_hardware / GAP_H4:.4f} ({error_hardware / GAP_H4 * 100:.2f}%)"
)
print(
    f"  ΔE/gap (post-QESEM):           {error_mitigated / GAP_H4:.4f} ({error_mitigated / GAP_H4 * 100:.2f}%)"
)
print("  Umbral PASS:                   < 5%")
print(
    f"  Veredicto:                     {'PASS ✅' if error_mitigated / GAP_H4 < 0.05 else 'FAIL ❌'}"
)

# Gate fidelities
print("\n  ─── Calidad del QPU (caracterización QESEM) ───")
print("  Fidelidad 1Q (ID):   0.9990 (99.90%)")
print("  Fidelidad 2Q (RZZ):  0.9972 (99.72%)")
print("  QPU time total:      428 s")
print("  Shots totales:       756,048")
print("  Shots mitigación:    266,000 (35.2%)")

# ─── Observables X (per-site) ──────────────────────────────────────
print("\n  ─── Observables ⟨X_i⟩ por qubit ───")
x_noisy_82 = [0.9374, 0.9357, 0.9457, 0.936, 0.9251, 0.926, 0.96, 0.9683, 0.8874, 0.9466]
x_mitigated_82 = [0.9918, 0.9687, 1.0019, 0.9944, 0.9814, 0.9723, 0.9705, 0.9930, 0.9964, 0.9749]
x_exact = 1.0  # Para h=4.0 >> h_c, ⟨X⟩ → 1

print(
    f"  {'Qubit':<7} {'Noisy':<10} {'QESEM':<10} {'Exact':<8} {'Err_hw':<12} {'Err_mitig':<12} {'Ganancia'}"
)
for i in range(10):
    err_n = abs(x_noisy_82[i] - x_exact)
    err_m = abs(x_mitigated_82[i] - x_exact)
    gain = ((err_n - err_m) / err_n * 100) if err_n > 0 else 0
    print(
        f"  q{i:<5} {x_noisy_82[i]:<10.4f} {x_mitigated_82[i]:<10.4f} {x_exact:<8.4f} {err_n:<12.4f} {err_m:<12.4f} {gain:>6.1f}%"
    )

x_noisy_mean = np.mean(x_noisy_82)
x_mitig_mean = np.mean(x_mitigated_82)
print(f"\n  Promedio ⟨X⟩ noisy:    {x_noisy_mean:.4f} (error: {abs(x_noisy_mean - x_exact):.4f})")
print(f"  Promedio ⟨X⟩ mitigado: {x_mitig_mean:.4f} (error: {abs(x_mitig_mean - x_exact):.4f})")

# ─── Noise scaling analysis (QESEM internal) ──────────────────────
print("\n" + "─" * 78)
print("  NOISE SCALING INTERNO DE QESEM (Job 82aa, observable energía)")
print("─" * 78)
print("\n  QESEM mide a 3 escalas de ruido y extrapola a scale=0:")
print(f"  {'Scale':<8} {'Valor':<16} {'Error bar':<12} {'|Error vs exact|':<18} {'Nota'}")
print(f"  {'─' * 8} {'─' * 16} {'─' * 12} {'─' * 18} {'─' * 30}")
scales = [
    (0.0, -40.5239, 0.288, "QESEM mitigated (extrapolated)"),
    (1.0, -40.0083, 0.058, "hardware native noise + REM"),
    (2.0, -38.9557, 0.484, "2x amplified noise"),
]
for s, v, eb, note in scales:
    err = abs(v - E_EXACT_H4)
    print(f"  {s:<8.1f} {v:<16.4f} {eb:<12.4f} {err:<18.4f} {note}")

print("\n  Interpretación:")
print("  • Scale=1.0: nivel de ruido nativo del QPU + REM (readout error mitigation)")
print("  • Scale=2.0: ruido amplificado ×2 mediante PEA probabilístico")
print("  • Scale=0.0: extrapolación a ruido cero (resultado final QESEM)")
err_scale1 = abs(-40.0083 - E_EXACT_H4)
print(f"  • Error a scale=1 (nativo+REM): {err_scale1:.4f}")
print(f"  • Error a scale=0 (QESEM):      {error_mitigated:.4f}")
print(
    f"  • Mejora por extrapolación:      {(err_scale1 - error_mitigated) / err_scale1 * 100:.1f}%"
)

# ─── Job 2: 4f16 (h=4.0, second run) ─────────────────────────────
print("\n" + "─" * 78)
print("  JOB 4f16e846 — h = 4.0 (second QESEM run, different calibration)")
print("─" * 78)

e_noisy_4f = -37.91257142857144
e_mitigated_4f = -40.19139501057736
e_std_4f = 0.3272458283418606

error_hw_4f = abs(e_noisy_4f - e_exact)
error_mit_4f = abs(e_mitigated_4f - e_exact)
gain_4f = (error_hw_4f - error_mit_4f) / error_hw_4f * 100

print(f"\n  E_exact     = {e_exact:.6f}")
print(f"  E_noisy     = {e_noisy_4f:.6f}")
print(f"  E_mitigated = {e_mitigated_4f:.6f} ± {e_std_4f:.4f}")
print(f"\n  Error hardware:    {error_hw_4f:.4f}  (ΔE/gap = {error_hw_4f / GAP_H4 * 100:.1f}%)")
print(f"  Error mitigado:    {error_mit_4f:.4f}  (ΔE/gap = {error_mit_4f / GAP_H4 * 100:.2f}%)")
print(f"  Ganancia:          {gain_4f:.1f}%")
print("  Gate fid 1Q: 0.9988 | 2Q (RZZ): 0.9978")
print(f"  Veredicto:         {'PASS ✅' if error_mit_4f / GAP_H4 < 0.05 else 'FAIL ❌'}")

# ─── Job 3: d628 (h=3.5) ──────────────────────────────────────────
print("\n" + "─" * 78)
print("  JOB d628a502 — h = 3.5 (QESEM)")
print("─" * 78)

e_noisy_d628 = -33.64114285714285
e_mitigated_d628 = -35.3594159189296
e_std_d628 = 0.27170008251588496

# Para h=3.5 N=10 OBC TFIM: E_exact ≈ -35.52 (estimado analíticamente)
# H = -3.5*sum(X) - sum(ZZ), el ground state energy escala como ~-h*N para h>>1
# Valor más preciso requiere diag. exacta — lo estimamos por la calidad de QESEM
print("\n  [h=3.5: E_exact requiere cálculo numérico separado]")
print(f"  E_noisy     = {e_noisy_d628:.6f}")
print(f"  E_mitigated = {e_mitigated_d628:.6f} ± {e_std_d628:.4f}")
print(f"  Diferencia noisy→mitigated: {abs(e_mitigated_d628 - e_noisy_d628):.4f}")
print("\n  Gate fid 1Q: 0.9983 | 2Q (RZZ): 0.9966 ← peor calibración")
print("  QPU time: 803 s (vs 278 s en job 82aa) — doble de QPU")
print("  Shots: 779,448 | Mitigación: 289,400")

# ─── Comparación PEA-ZNE vs QESEM ─────────────────────────────────
print("\n" + "─" * 78)
print("  COMPARACIÓN: QESEM vs PEA-ZNE (ambos h=4.0, ibm_kingston)")
print("─" * 78)

# PEA-ZNE extrapolated observables (job d8tche5b)
x_pea = [0.9561, 0.9613, 0.9158, 0.9565, 0.9409, 0.9183, 0.9313, 0.9551, 0.9732, 0.9254]
zz_pea = [0.1109, 0.1044, 0.1102, 0.1204, 0.1434, 0.1148, 0.0250, 0.0, 0.0057]

# Reconstruir energía: E = -4.0 * sum(X_i) - sum(ZZ_j)
x_pea_sum = sum(x_pea)
zz_pea_sum = sum(zz_pea)
e_pea = -4.0 * x_pea_sum - zz_pea_sum

err_noisy = abs(e_noisy - e_exact)
err_qesem = abs(e_mitigated - e_exact)
err_pea = abs(e_pea - e_exact)
pea_mean = np.mean(x_pea)

print(f"\n  {'Método':<20} {'E_value':<14} {'|Error|':<10} {'ΔE/gap %':<10} {'⟨X⟩ mean':<10}")
print(f"  {'─' * 20} {'─' * 14} {'─' * 10} {'─' * 10} {'─' * 10}")
print(
    f"  {'Sin mitigación':<20} {e_noisy:<14.4f} {err_noisy:<10.4f} {err_noisy / GAP_H4 * 100:<10.1f} {x_noisy_mean:<10.4f}"
)
print(
    f"  {'QESEM (82aa)':<20} {e_mitigated:<14.4f} {err_qesem:<10.4f} {err_qesem / GAP_H4 * 100:<10.2f} {x_mitig_mean:<10.4f}"
)
print(
    f"  {'PEA-ZNE (d8tc)':<20} {e_pea:<14.4f} {err_pea:<10.4f} {err_pea / GAP_H4 * 100:<10.1f} {pea_mean:<10.4f}"
)
print(f"  {'Exacto':<20} {e_exact:<14.4f} {'0':<10} {'0':<10} {'1.0000':<10}")

print("\n  ─── Ganancia relativa ───")
print(f"  QESEM corrigió:  {(err_noisy - err_qesem) / err_noisy * 100:.1f}% del error del hardware")
print(f"  PEA-ZNE corrigió: {(err_noisy - err_pea) / err_noisy * 100:.1f}% del error del hardware")

# ─── RESUMEN FINAL ────────────────────────────────────────────────
print("\n" + "=" * 78)
print("  RESUMEN FINAL — ERRORES EN LA EJECUCIÓN QESEM")
print("=" * 78)
print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │ TFIM N=10, p=1 HVA, h=4.0 (paramagnetic), ibm_kingston            │
  │ E_exact = {e_exact:.6f}                                       │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  1. ERROR REAL DEL HARDWARE (sin mitigación):                       │
  │     E_noisy = {e_noisy:.4f}                                        │
  │     |Error| = {err_noisy:.4f} Ha                                      │
  │     ΔE/gap  = {err_noisy / GAP_H4 * 100:.1f}% (FAIL: supera el 5%)                       │
  │                                                                     │
  │  2. ERROR QUE QESEM LOGRÓ CORREGIR:                                │
  │     Error corregido = {err_noisy - err_qesem:.4f} Ha                              │
  │     Ganancia = {(err_noisy - err_qesem) / err_noisy * 100:.1f}% del error original                         │
  │                                                                     │
  │  3. ERROR RESIDUAL (post-mitigación):                               │
  │     E_mitigated = {e_mitigated:.4f} ± {e_std:.4f}                     │
  │     |Error| = {err_qesem:.4f} Ha                                       │
  │     ΔE/gap  = {err_qesem / GAP_H4 * 100:.2f}% (PASS: < 5%)                            │
  │                                                                     │
  │  4. FUENTES DEL ERROR RESIDUAL:                                     │
  │     • Incertidumbre estadística: ±{e_std:.4f} (incluye {err_qesem:.4f})   │
  │     • Error compatible con cero dentro de 1σ                        │
  │     • Fidelidad 2Q = 99.72% → error sistemático ~0.28% por gate    │
  │                                                                     │
  │  5. OBSERVABLES LOCALES:                                            │
  │     ⟨X⟩ mejoró de {x_noisy_mean:.4f} → {x_mitig_mean:.4f} (exacto: 1.0)              │
  │     Error ⟨X⟩: {abs(x_noisy_mean - x_exact):.4f} → {abs(x_mitig_mean - x_exact):.4f}  (reducción {(1 - abs(x_mitig_mean - x_exact) / abs(x_noisy_mean - x_exact)) * 100:.0f}%)       │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
""")
