#!/usr/bin/env python3
"""Comprobación rápida y justa de la ventaja de warm-start del MPNN.

Para cada N y cada h del subconjunto de prueba, corre EL MISMO VQE (mismo ansatz
bond-resolved, mismo presupuesto acotado) desde tres inicializaciones:

  - fría        : θ aleatorio ~ U(-π, π)
  - tibia-MPNN  : θ predicho zero-shot por el modelo del zoo (heavy_hex p=1)
  - tibia-verif : θ_opt verificado del NPZ (predictor "ideal" de referencia)

Reporta por N: |ΔE/gap| final, iteraciones hasta converger (k̄) y fidelidad, para
cada init. Objetivo: decidir si el MPNN acelera/mejora la convergencia (figura con
MPNN tal cual) o si conviene usar θ verificado como inicialización informada.

No escribe artefactos de tesis; sólo imprime un resumen. Reutiliza el caché de
ground truth (ClassicalSolver) y los NPZ verified existentes.

Uso:
    .venv/bin/python scripts/analysis/probe_warmstart_advantage.py --n-values 6 16
"""

from __future__ import annotations

import argparse
import json
import sys
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
from qmbp_simulation.models.model_registry import get_model_spec  # noqa: E402
from qmbp_simulation.optimizers.vqe import VQEOptimizer  # noqa: E402

TOPOLOGY = "heavy_hex"  # default; se puede sobreescribir con --topology
P_LAYERS = 1
J = 1.0
ANSATZ_MODEL = "tfim_bond_resolved"
# Subconjunto de h de prueba (dentro del régimen válido). Fallback genérico;
# el default REAL se elige por topología (ver _H_PROBE_BY_TOPOLOGY) cuando no se
# pasa --h-probe, para no medir fuera del régimen interesante de cada red.
H_PROBE = [2.2, 2.8, 3.4, 4.0]  # fallback; sobreescribible con --h-probe
# Grilla de h por defecto por topología. La transición del TFIM cae en h_c≈1
# para chain_1d (grilla que atraviesa la transición hacia el paramagneto);
# las redes quasi-2D/2D tienen h_c efectivo más alto → grilla en el paramagneto
# donde el HVA p=1 es expresivo. Se usa solo si --h-probe no se especifica.
_H_PROBE_BY_TOPOLOGY: dict[str, list[float]] = {
    "chain_1d": [1.3, 1.6, 2.0, 2.5, 3.0],
    "heavy_hex": [2.0, 2.5, 3.0, 3.5, 4.0],
    "ladder": [2.0, 2.5, 3.0, 3.5, 4.0],
    "square": [2.5, 3.0, 3.5, 4.0, 4.5],
    "triangular": [2.5, 3.0, 3.5, 4.0, 4.5],
}
VQE_MAXITER = 150
VQE_RESTARTS = 2
SEED = 42


def _load_mpnn(checkpoint: str | None = None):
    """Carga el modelo del zoo. Si se pasa ``checkpoint``, fija ese específico
    (fuzzy-resuelto); si no, usa el mejor por defecto del zoo para la topología.
    """
    if checkpoint:
        from qmbp_simulation.predictors.model_zoo import (
            _smart_load_checkpoint,
            resolve_checkpoint_fuzzy,
        )

        path = resolve_checkpoint_fuzzy(checkpoint, topology=TOPOLOGY, p_layers=P_LAYERS)
        if path is None:
            raise FileNotFoundError(f"No se pudo resolver el checkpoint {checkpoint!r}")
        model = _smart_load_checkpoint(str(path))
        return model, f"{path.name} (fijado)"

    from qmbp_simulation.predictors.model_zoo import load_best_model_for

    model, entry, source = load_best_model_for(TOPOLOGY, p_layers=P_LAYERS)
    desc = f"{getattr(entry, 'checkpoint_file', '?')} (fuente={source})"
    return model, desc


def _predict_theta(model, lattice, h_value: float) -> np.ndarray:
    import torch

    from qmbp_simulation.predictors.unified_graph import build_graph_for_model

    graph = build_graph_for_model(model, lattice, h_value, P_LAYERS)
    model.eval()
    with torch.no_grad():
        theta = model(graph).cpu().numpy().flatten()
    return np.clip(theta, -np.pi, np.pi)


def _npz_theta_at(npz_path: Path, h_target: float) -> np.ndarray | None:
    """Devuelve el θ_opt verified del NPZ para el h más cercano (tol 0.06)."""
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=True)
    h_all = d["h_values"]
    theta_all = d["theta_opt"]
    tiers = d["quality_tier"] if "quality_tier" in d else None
    i = int(np.argmin(np.abs(h_all - h_target)))
    if abs(float(h_all[i]) - h_target) > 0.06:
        return None
    if tiers is not None and str(tiers[i]) != "verified":
        return None
    return np.asarray(theta_all[i], dtype=float)


def _run_vqe(vqe, H, circuit, x0, e_exact, exact_state):
    res = vqe.optimize(H, circuit, x0, exact_energy=e_exact, exact_state=exact_state)
    return res


def probe_n(n_qubits: int, model, spec, backend, solver, builder) -> dict:
    npz_path = ROOT / "data" / "multi_n_training" / f"{TOPOLOGY}_N{n_qubits}_p{P_LAYERS}.npz"
    rng = np.random.default_rng(SEED)
    vqe = VQEOptimizer(
        config=VQEConfig(method="L-BFGS-B", maxiter=VQE_MAXITER, n_restarts=VQE_RESTARTS),
        backend=backend,
        seed=SEED,
    )
    # VQE de un solo disparo (sin restarts) para medir la aceleración pura del init:
    # los restarts enmascararían la ventaja del punto de partida.
    vqe_single = VQEOptimizer(
        config=VQEConfig(method="L-BFGS-B", maxiter=VQE_MAXITER, n_restarts=1),
        backend=backend,
        seed=SEED,
    )

    rows = []
    for h in H_PROBE:
        lat = make_lattice(TOPOLOGY, n_qubits, J=J, h=h)
        H = builder.build(lat)
        gt = solver.solve(H, lat)
        e_exact = float(gt.ground_energy)
        gap = float(gt.gap) if gt.gap and gt.gap > 1e-10 else 0.1
        exact_state = getattr(gt, "ground_state", None)
        circuit, theta_param = spec.create_circuit(n_qubits, P_LAYERS, lat)
        n_params = len(theta_param)

        # θ del MPNN (zero-shot) y su energía SIN optimizar (para ver el punto de partida)
        theta_mpnn = _predict_theta(model, lat, h)
        e_mpnn0 = float(backend.evaluate(circuit, H, theta_mpnn))
        de_mpnn0 = abs(e_mpnn0 - e_exact) / gap

        # θ verificado del NPZ (predictor ideal) y su energía sin optimizar
        theta_verif = _npz_theta_at(npz_path, h)
        de_verif0 = None
        if theta_verif is not None and theta_verif.size == n_params:
            e_verif0 = float(backend.evaluate(circuit, H, theta_verif))
            de_verif0 = abs(e_verif0 - e_exact) / gap

        # ── VQE de un disparo desde cada init (aceleración pura) ──
        x0_cold = rng.uniform(-np.pi, np.pi, n_params)
        r_cold = _run_vqe(vqe_single, H, circuit, x0_cold, e_exact, exact_state)
        r_warm_mpnn = _run_vqe(vqe_single, H, circuit, theta_mpnn.copy(), e_exact, exact_state)
        r_warm_verif = None
        if theta_verif is not None and theta_verif.size == n_params:
            r_warm_verif = _run_vqe(
                vqe_single, H, circuit, theta_verif.copy(), e_exact, exact_state
            )

        def _init_metrics(theta0):
            """Métricas del punto de partida (sin optimizar)."""
            e0 = float(backend.evaluate(circuit, H, theta0))
            try:
                f0 = float(backend.compute_fidelity(circuit, theta0, exact_state)) \
                    if exact_state is not None else float("nan")
            except Exception:  # noqa: BLE001
                f0 = float("nan")
            return {
                "e_init": e0,
                "abs_delta_e_init": abs(e0 - e_exact),
                "de_gap_init": abs(e0 - e_exact) / gap,
                "fidelity_init": f0,
                "theta_init": np.asarray(theta0, dtype=float).tolist(),
            }

        def _final_metrics(res, theta0):
            """Métricas tras optimizar (init → VQE), con todo lo pedido."""
            rec = _init_metrics(theta0)
            rec.update({
                "e_final": float(res.energy),
                "abs_delta_e_final": float(abs(res.energy - e_exact)),
                "de_gap_final": float(abs(res.energy - e_exact) / gap),
                "fidelity_final": float(res.fidelity),
                "n_iters": int(res.n_iterations),
                "theta_final": np.asarray(res.theta_opt, dtype=float).tolist(),
            })
            return rec

        record = {
            "h": float(h),
            "e_exact": e_exact,
            "gap": gap,
            "n_params": n_params,
            "cold": _final_metrics(r_cold, x0_cold),
            "warm_mpnn": _final_metrics(r_warm_mpnn, theta_mpnn),
            "warm_verif": (
                _final_metrics(r_warm_verif, theta_verif)
                if r_warm_verif is not None else None
            ),
            # Duplicado para lectura rápida (zero-shot = init sin optimizar)
            "de_mpnn_zeroshot": de_mpnn0,
            "de_verif_zeroshot": de_verif0,
        }
        rows.append(record)

        vz = f"{de_verif0:.4f}" if de_verif0 is not None else "n/a"
        wv = record["warm_verif"]
        wv_s = (
            f"de={wv['de_gap_final']:.4f} k={wv['n_iters']} F={wv['fidelity_final']:.3f}"
            if wv else "n/a"
        )
        print(
            f"  N={n_qubits} h={h:.2f} | zero-shot: MPNN de={de_mpnn0:.4f} verif de={vz}\n"
            f"      VQE(1-shot)  frío: de={record['cold']['de_gap_final']:.4f} "
            f"k={record['cold']['n_iters']} F={record['cold']['fidelity_final']:.3f}\n"
            f"      VQE(1-shot)  MPNN: de={record['warm_mpnn']['de_gap_final']:.4f} "
            f"k={record['warm_mpnn']['n_iters']} F={record['warm_mpnn']['fidelity_final']:.3f}\n"
            f"      VQE(1-shot) verif: {wv_s}",
            flush=True,
        )

    return {"n_qubits": n_qubits, "rows": rows}


def _summarize(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("RESUMEN — media sobre los h de prueba (VQE 1-disparo, budget acotado)")
    print("=" * 72)
    header = f"{'N':>3} | {'init':<11} | {'de/gap final':>12} | {'k̄ iters':>9} | {'fidelidad':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        n = r["n_qubits"]
        rows = r["rows"]
        for label, key in [("frío", "cold"), ("MPNN", "warm_mpnn"), ("verif", "warm_verif")]:
            vals = [row[key] for row in rows if row[key] is not None]
            if not vals:
                print(f"{n:>3} | {label:<11} | {'n/a':>12} | {'n/a':>9} | {'n/a':>9}")
                continue
            de = np.mean([v["de_gap_final"] for v in vals])
            k = np.mean([v["n_iters"] for v in vals])
            f = np.mean([v["fidelity_final"] for v in vals])
            print(f"{n:>3} | {label:<11} | {de:>12.4f} | {k:>9.1f} | {f:>9.3f}")
        print("-" * len(header))


def main() -> int:
    global TOPOLOGY, H_PROBE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", type=int, nargs="+", default=[6, 16])
    parser.add_argument("--topology", type=str, default="heavy_hex")
    parser.add_argument("--h-probe", type=float, nargs="+", default=None,
                        help="Grilla de h de prueba (default por topología).")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Fijar un checkpoint específico (fuzzy). Default: mejor del zoo.")
    parser.add_argument("--write-zoo", action="store_true",
                        help="Al terminar, poblar warmstart_by_n del checkpoint en el zoo "
                             "(backfill_warmstart_from_probe sobre el JSON generado).")
    args = parser.parse_args()

    # Sobreescribir globales que consumen las funciones auxiliares.
    TOPOLOGY = args.topology
    if args.h_probe:
        H_PROBE = list(args.h_probe)
    else:
        # Default por topología (régimen válido de cada red); fallback genérico.
        H_PROBE = list(_H_PROBE_BY_TOPOLOGY.get(TOPOLOGY, H_PROBE))
    out_dir = args.out_dir or (
        ROOT / "results" / "experiments" / f"exp_warmstart_probe_{TOPOLOGY}"
    )
    # Resolver a absoluto para que relative_to(ROOT) nunca falle con --out-dir relativo.
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()

    backend = NoiselessBackend()
    solver = ClassicalSolver()
    builder = HamiltonianBuilder()
    spec = get_model_spec(ANSATZ_MODEL)
    model, model_desc = _load_mpnn(args.checkpoint)
    print(f"MPNN: {model_desc}", file=sys.stderr)
    print(f"Ansatz: {ANSATZ_MODEL} | {TOPOLOGY} p={P_LAYERS} | h={H_PROBE} | "
          f"VQE maxiter={VQE_MAXITER}", file=sys.stderr)

    results = []
    for n in args.n_values:
        print(f"\n── N={n} ──", flush=True)
        res = probe_n(n, model, spec, backend, solver, builder)
        results.append(res)
        # Persistencia incremental: escribe el JSON tras cada N (robusto a cortes).
        _save(out_dir, model_desc, results, args.n_values)
    _summarize(results)
    try:
        shown = out_dir.relative_to(ROOT)
    except ValueError:
        shown = out_dir  # out-dir outside the repo → show the absolute path
    print(f"\nRecord completo → {shown}", file=sys.stderr)

    # Optionally push the warm-start metrics straight to the zoo (closes the loop).
    if args.write_zoo:
        try:
            from qmbp_simulation.predictors.model_zoo import backfill_warmstart_from_probe

            probe_json = out_dir / "warmstart_probe.json"
            n = backfill_warmstart_from_probe(probe_json)
            print(f"Zoo warm-start actualizado: {n} entry (desde {probe_json.name})",
                  file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — best-effort, never fail the run
            print(f"⚠️ --write-zoo falló (no crítico): {exc}", file=sys.stderr)

    return 0


def _save(out_dir: Path, model_desc: str, results: list[dict], n_requested: list[int]) -> None:
    """Escribe el record completo (todas las métricas por punto) en JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    envelope = {
        "config": {
            "ansatz_model": ANSATZ_MODEL,
            "physics_model": "tfim",
            "topology": TOPOLOGY,
            "p_layers": P_LAYERS,
            "J": J,
            "h_probe": H_PROBE,
            "vqe": {"method": "L-BFGS-B", "maxiter": VQE_MAXITER,
                    "n_restarts_single_shot": 1, "seed": SEED},
            "mpnn_checkpoint": model_desc,
            "n_values_requested": list(n_requested),
        },
        "results_by_n": results,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    out = out_dir / "warmstart_probe.json"
    out.write_text(json.dumps(envelope, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
