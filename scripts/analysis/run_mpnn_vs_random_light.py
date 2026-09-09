#!/usr/bin/env python3
"""Experimento ligero: predicción MPNN vs VQE con inicialización aleatoria.

Genera, con una configuración ÚNICA y consistente para todos los tamaños
(TFIM estándar, heavy-hex, p=1, misma grilla de h y mismo presupuesto de VQE
aleatorio), los datos reales que sostienen la figura de comparación de la tesis.
Reemplaza al experimento pesado (pipeline completo de 3 fases por N), reutilizando
la infraestructura existente y evitando re-optimizar el warm-start (lo costoso):

  - Estado de referencia exacto (E_0, gap, |ψ_0⟩): ``ClassicalSolver`` con caché en
    disco (``ground_truth_cache.json``) — no se recomputa entre corridas.
  - Predicción MPNN (zero-shot): una inferencia del modelo del zoo por punto h +
    una evaluación de energía. Milisegundos por punto.
  - VQE aleatorio: ``VQEOptimizer.optimize`` desde θ aleatorio con un presupuesto
    ACOTADO y UNIFORME (mismo maxiter/restarts para los cuatro N), que es el único
    cómputo real y es modesto.

Métrica primaria |ΔE| (steering §5); R(N) = (ΔE/gap)_aleatorio / (ΔE/gap)_MPNN.
El JSON de salida usa el mismo esquema ``section_4`` que consumen el generador de
figura (``plot_mpnn_vs_random.py``) y el resto del proyecto.

Uso:
    .venv/bin/python scripts/analysis/run_mpnn_vs_random_light.py
    .venv/bin/python scripts/analysis/run_mpnn_vs_random_light.py --n-values 6 10 16 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation import (  # noqa: E402
    ClassicalSolver,
    HamiltonianBuilder,
    make_lattice,
)
from qmbp_simulation.execution import NoiselessBackend  # noqa: E402
from qmbp_simulation.models.data_models import VQEConfig  # noqa: E402
from qmbp_simulation.optimizers.vqe import VQEOptimizer  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = ROOT / "results" / "experiments" / "exp_mpnn_vs_random_light"
# Hamiltoniano físico: TFIM estándar (H = -J·ZZ - h·X). El ansatz es la variante
# bond-resolved del HVA (una θ por bond y por sitio), que es la que predice el
# modelo del zoo de heavy_hex p=1. En la tesis se reporta como "TFIM, heavy-hex,
# p=1"; el circuito bond-resolved es un detalle del ansatz, no del modelo físico.
MODEL = "tfim"  # etiqueta física (para caption y envelope)
ANSATZ_MODEL = "tfim_bond_resolved"  # spec del circuito (coincide con la MPNN)
TOPOLOGY = "heavy_hex"
P_LAYERS = 1
J = 1.0
# Grilla de h común a los cuatro N (intersección de los rangos con dato verified,
# dentro del régimen válido h >= 1.3). Densa y comparable entre tamaños.
H_GRID = [1.9, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4.0, 4.3, 4.5]
# Presupuesto del VQE aleatorio: ACOTADO y UNIFORME para todos los N (comparación
# a presupuesto igual, steering §5). No busca el óptimo global; mide la calidad
# alcanzable por un optimizador sin información previa a presupuesto fijo.
RANDOM_VQE_MAXITER = 150
RANDOM_VQE_RESTARTS = 2
RANDOM_SEED = 42


def _load_mpnn(n_qubits: int):
    """Carga el mejor modelo del zoo para heavy_hex p=1 (con cross-N si aplica).

    Devuelve (model, descripcion) o (None, motivo) si no hay modelo utilizable.
    """
    try:
        from qmbp_simulation.predictors.model_zoo import load_best_model_for

        model, entry, source = load_best_model_for(TOPOLOGY, p_layers=P_LAYERS)
        if model is None:
            return None, "sin modelo en el zoo"
        desc = f"{getattr(entry, 'checkpoint_file', '?')} (fuente={source})"
        return model, desc
    except Exception as exc:  # noqa: BLE001
        return None, f"error cargando zoo: {exc}"


def _predict_theta(model, lattice, h_value: float) -> np.ndarray:
    """Inferencia zero-shot de θ para (lattice, h) con la MPNN del zoo.

    Usa ``build_graph_for_model`` (elige el grafo según el tipo de modelo:
    legacy MPNNPredictor o UnifiedMPNN bond-resolved) + forward no-grad.
    """
    import torch

    from qmbp_simulation.predictors.unified_graph import build_graph_for_model

    graph = build_graph_for_model(model, lattice, h_value, P_LAYERS)
    model.eval()
    with torch.no_grad():
        theta = model(graph).cpu().numpy().flatten()
    return np.clip(theta, -np.pi, np.pi)


def run_for_n(n_qubits: int, model, backend, solver, builder, spec) -> dict:
    """Ejecuta la comparación para un tamaño N y devuelve el bloque section_4.

    Ambos métodos usan el MISMO ansatz bond-resolved (una θ por bond y por sitio):
    el modelo del zoo predice θ zero-shot, y el VQE aleatorio optimiza ese mismo
    circuito desde θ aleatorio a presupuesto acotado. La única diferencia es la
    inicialización — comparación honesta init-MPNN vs init-aleatoria.
    """
    lattice = make_lattice(TOPOLOGY, n_qubits, J=J, h=H_GRID[0])
    circuit, theta_param = spec.create_circuit(n_qubits, P_LAYERS, lattice)
    n_params = len(theta_param)

    rng = np.random.default_rng(RANDOM_SEED)
    vqe = VQEOptimizer(
        config=VQEConfig(
            method="L-BFGS-B",
            maxiter=RANDOM_VQE_MAXITER,
            n_restarts=RANDOM_VQE_RESTARTS,
        ),
        backend=backend,
        seed=RANDOM_SEED,
    )

    per_point: list[dict] = []
    for h in H_GRID:
        lat_h = make_lattice(TOPOLOGY, n_qubits, J=J, h=h)
        H = builder.build(lat_h)
        gt = solver.solve(H, lat_h)  # cacheado en disco
        e_exact = float(gt.ground_energy)
        gap = float(gt.gap) if gt.gap and gt.gap > 1e-10 else 0.1
        exact_state = getattr(gt, "ground_state", None)

        # ── MPNN zero-shot ──
        theta_mpnn = _predict_theta(model, lat_h, h)
        if theta_mpnn.size != n_params:
            raise ValueError(
                f"θ MPNN ({theta_mpnn.size}) != params del circuito ({n_params}) "
                f"en N={n_qubits}. Revisar topología/p_layers/modelo."
            )
        e_mpnn = float(backend.evaluate(circuit, H, theta_mpnn))
        de_mpnn = abs(e_mpnn - e_exact) / gap
        try:
            fid = (
                float(backend.compute_fidelity(circuit, theta_mpnn, exact_state))
                if exact_state is not None
                else float("nan")
            )
        except Exception:  # noqa: BLE001
            fid = float("nan")

        # ── VQE aleatorio (presupuesto acotado y uniforme) ──
        x0 = rng.uniform(-np.pi, np.pi, n_params)
        res = vqe.optimize(H, circuit, x0, exact_energy=e_exact, exact_state=exact_state)
        e_rand = float(res.energy)
        de_rand = abs(e_rand - e_exact) / gap
        ratio = de_mpnn / de_rand if de_rand > 1e-12 else np.nan  # MPNN/random (<1 gana MPNN)

        per_point.append(
            {
                "h_test": float(h),
                "e_pred": e_mpnn,
                "e_exact": e_exact,
                "de_gap": de_mpnn,
                "de_gap_random_init": de_rand,
                "n_iters_random_init": int(res.n_iterations),
                "mpnn_vs_random_ratio": float(ratio) if np.isfinite(ratio) else None,
                "fidelity": fid,
            }
        )
        print(
            f"    N={n_qubits} h={h:.2f}: |ΔE/gap| MPNN={de_mpnn:.4f} vs "
            f"aleatorio={de_rand:.4f} (R={de_rand / de_mpnn:.1f}×) F={fid:.4f} "
            f"k̄={res.n_iterations}",
            flush=True,
        )

    de_m = [p["de_gap"] for p in per_point]
    de_r = [p["de_gap_random_init"] for p in per_point]
    fids = [p["fidelity"] for p in per_point if np.isfinite(p["fidelity"])]
    wins = sum(1 for p in per_point if p["de_gap"] < p["de_gap_random_init"])
    kbar = [p["n_iters_random_init"] for p in per_point]
    return {
        "pass": float(np.mean([d < 0.05 for d in de_m])) >= 0.8,
        "n_test_points": len(per_point),
        "mean_de_gap": float(np.mean(de_m)),
        "mean_de_gap_random_init": float(np.mean(de_r)),
        "mpnn_wins_vs_random": wins,
        "mean_fidelity": float(np.mean(fids)) if fids else float("nan"),
        "mean_vqe_iters_per_point": float(np.mean(kbar)),
        "per_point": per_point,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", type=int, nargs="+", default=[6, 10, 16, 20])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    from qmbp_simulation.models.model_registry import get_model_spec

    backend = NoiselessBackend()
    solver = ClassicalSolver()
    builder = HamiltonianBuilder()
    spec = get_model_spec(ANSATZ_MODEL)  # circuito bond-resolved (coincide con la MPNN)

    print(f"Experimento ligero MPNN vs VQE aleatorio — {MODEL}, {TOPOLOGY}, p={P_LAYERS}", file=sys.stderr)
    print(
        f"Grilla h={H_GRID} | VQE aleatorio: maxiter={RANDOM_VQE_MAXITER}, "
        f"restarts={RANDOM_VQE_RESTARTS} (uniforme para todo N)",
        file=sys.stderr,
    )

    ok = 0
    for n in args.n_values:
        model, desc = _load_mpnn(n)
        if model is None:
            print(f"  N={n}: SIN modelo MPNN ({desc}) — se omite.", file=sys.stderr)
            continue
        print(f"\n  N={n}: MPNN={desc}", flush=True)
        t0 = time.perf_counter()
        try:
            s4 = run_for_n(n, model, backend, solver, builder, spec)
        except Exception as exc:  # noqa: BLE001
            print(f"  N={n}: FALLÓ ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue
        elapsed = time.perf_counter() - t0
        envelope = {
            "config": {
                "system": {
                    "n_qubits": n,
                    "p_layers": P_LAYERS,
                    "model": MODEL,
                    "topologies": [TOPOLOGY],
                },
                "h_grid": {"h_min": min(H_GRID), "h_max": max(H_GRID), "h_points": len(H_GRID)},
                "vqe": {"maxiter": RANDOM_VQE_MAXITER, "n_restarts": RANDOM_VQE_RESTARTS},
            },
            "results": {"section_4": {"data": s4}},
            "elapsed_s": elapsed,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        out = args.out_dir / f"run_{MODEL}_{TOPOLOGY}_N{n}_p{P_LAYERS}.json"
        out.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        print(
            f"  N={n}: OK en {elapsed:.1f}s → {out.relative_to(ROOT)} "
            f"(MPNN {s4['mean_de_gap']:.4f} vs aleatorio {s4['mean_de_gap_random_init']:.4f}, "
            f"wins {s4['mpnn_wins_vs_random']}/{s4['n_test_points']})",
            flush=True,
        )
        ok += 1

    print(f"\n  {ok}/{len(args.n_values)} tamaños completados.", file=sys.stderr)
    return 0 if ok >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())
