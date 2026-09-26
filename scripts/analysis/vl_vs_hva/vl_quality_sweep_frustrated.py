#!/usr/bin/env python
"""Sweep Haiqu VL quality knobs on the frustrated points where VL degrades.

In the frustrated regime (tfim_frustrated, J2>0) with default settings
(num_layers=2, fine_tuning_iterations=20) VL loading fidelity drops on the
small-gap points: square h=0.5 (0.81), square h=1.0 (0.90), triangular h=1.0
(0.91). This script asks whether that drop is a config limit or a method limit
by sweeping ``num_layers`` × ``fine_tuning_iterations`` on exactly those points
and recording, per config, the VL fidelity and 2q-gate count.

For each (topology, h, num_layers, fine_tuning_iterations) it loads the SAME
exact ground state (reused across configs) via Haiqu VL and records fidelity,
VL energy, energy error |ΔE| and 2q-gate count. Results are persisted
incrementally (crash-safe) to results/hva_vl_study/vl_quality_sweep/.

VL job results are cached in ``data/vl_cache/vl_quality_sweep.json`` (reusing the
project's ``EvalCache``): a config already measured is reused instead of
re-submitting the (slow, paid) Haiqu job. Use ``--force-recompute`` to bypass the
cache read and re-run every job (the cache is still refreshed with new values).

Requires HAIQU_API_KEY (only for configs not already cached).

Usage:
    HAIQU_API_KEY=... .venv/bin/python scripts/analysis/vl_quality_sweep_frustrated.py \
        --num-layers 2 4 8 --fine-tuning 20 50 --n 9 --j2 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hva_vl_study_common import (
    circuit_energy,
    exact_ground_state_vector,
    find_repo_root,
    haiqu_session,
    n_2q,
    run_vl_job,
    save_circuit_qpy,
    save_json,
    study_dir,
)

from qmbp_simulation.execution.eval_cache import EvalCache

# The frustrated points where VL underperforms with default settings.
DEFAULT_POINTS = [("square", 0.5), ("square", 1.0), ("triangular", 1.0)]

# Move-resilient repo root (shared helper, not parents[N] depth counting).
_REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)

# Dedicated VL-result cache (reuses EvalCache; kept apart from the energy cache).
_VL_CACHE_PATH = _REPO_ROOT / "data" / "vl_cache" / "vl_quality_sweep.json"


def _circuits_dir() -> Path:
    """Subdir where VL circuits are saved as QPY for later re-analysis."""
    d = study_dir("vl_quality_sweep") / "circuits"
    d.mkdir(parents=True, exist_ok=True)
    return d

# The three scalar metrics we persist per VL config, each under its own key.
_VL_METRICS = ("fidelity", "energy", "n_2q")


def _cache_key(metric: str, topology: str, n: int, model: str, j2: float,
               h: float, num_layers: int, ft: int) -> str:
    """Namespaced VL cache key. h uses :.2f per the repo h-precision convention."""
    return f"VL{metric}|{model}|{topology}|{n}|J2{j2:.2f}|{h:.2f}|L{num_layers}|F{ft}"


def _cached_row(cache, base: dict, topology, n, model, j2, h, num_layers, ft):
    """Return a fully-populated row from cache, or None if any metric is missing."""
    vals = {}
    for m in _VL_METRICS:
        v = cache.get(_cache_key(m, topology, n, model, j2, h, num_layers, ft))
        if v is None:
            return None
        vals[m] = v
    row = dict(base)
    row["vl_fidelity"] = vals["fidelity"]
    row["vl_energy"] = vals["energy"]
    row["vl_abs_error"] = abs(vals["energy"] - base["E0"])
    row["vl_2q"] = int(vals["n_2q"])
    row["vl_status"] = "DONE"
    row["cached"] = True
    return row


def _store_row(cache, row, topology, n, model, j2, h, num_layers, ft) -> None:
    """Persist the three scalar metrics of a completed VL config into the cache."""
    metric_values = {
        "fidelity": row.get("vl_fidelity"),
        "energy": row.get("vl_energy"),
        "n_2q": row.get("vl_2q"),
    }
    for m, v in metric_values.items():
        if v is not None:
            cache.put(_cache_key(m, topology, n, model, j2, h, num_layers, ft), float(v))


def _persist(rows, n, j2, model, num_layers_grid, fine_tuning_grid) -> None:
    save_json(
        {"rows": rows}, "vl_quality_sweep", "vl_quality_sweep_frustrated.json",
        params={"N": n, "J2": j2, "model": model,
                "num_layers_grid": num_layers_grid, "fine_tuning_grid": fine_tuning_grid},
        description="VL quality-knob sweep (num_layers × fine_tuning) on frustrated points",
    )


def run(points, num_layers_grid, fine_tuning_grid, n, j2, model, timeout_s,
        use_cache: bool = True) -> list[dict]:
    rows: list[dict] = []
    cache = EvalCache(path=_VL_CACHE_PATH)
    session = None  # opened lazily, only if a real Haiqu job is needed

    try:
        for topology, h in points:
            # Solve the exact ground state once per point; reuse for every config.
            psi, e0, gap, H_op = exact_ground_state_vector(
                topology, n, h, model=model, j2=j2, return_hamiltonian=True
            )
            if psi is None:
                rows.append({"topology": topology, "h": h, "error": "no exact ground_state vector"})
                continue
            base = {"topology": topology, "N": n, "h": h, "J2": j2, "gap": gap, "E0": e0}
            for num_layers in num_layers_grid:
                for ft in fine_tuning_grid:
                    row = None
                    if use_cache:
                        row = _cached_row(cache, {**base, "num_layers": num_layers,
                                                  "fine_tuning_iterations": ft},
                                          topology, n, model, j2, h, num_layers, ft)
                    if row is None:
                        row = {**base, "num_layers": num_layers, "fine_tuning_iterations": ft}
                        try:
                            if session is None:  # open the paid session only when needed
                                session = haiqu_session("VL_quality_sweep_frustrated")
                                session.__enter__()
                            fid, status, vl, cid, vl_meta = run_vl_job(
                                psi, n, num_layers=num_layers, timeout_s=timeout_s,
                                fine_tuning_iterations=ft,
                            )
                            row["vl_fidelity"] = fid
                            row["vl_status"] = status
                            row["vl_circuit_id"] = cid
                            row["vl_meta"] = vl_meta
                            row["vl_2q"] = n_2q(vl) if vl is not None else None
                            if vl is not None:
                                vl_energy = circuit_energy(vl, H_op)
                                row["vl_energy"] = vl_energy
                                row["vl_abs_error"] = abs(vl_energy - e0)
                                # Persist the circuit locally (QPY) for later re-analysis
                                # — cloud- and API-key-independent.
                                qpy_path = _circuits_dir() / (
                                    f"{topology}_N{n}_h{h:.2f}_J2{j2:.2f}"
                                    f"_L{num_layers}_F{ft}.qpy"
                                )
                                save_circuit_qpy(vl, qpy_path)
                                row["vl_circuit_qpy"] = str(qpy_path.relative_to(_REPO_ROOT))
                                _store_row(cache, row, topology, n, model, j2, h, num_layers, ft)
                                cache.flush()
                        except Exception as exc:  # noqa: BLE001 — never abort the batch
                            row["error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(row)
                    tag = " [cached]" if row.get("cached") else ""
                    print(f"  {topology} h={h} layers={num_layers} ft={ft}{tag} "
                          f"→ fid={row.get('vl_fidelity')} |dE|={row.get('vl_abs_error')} "
                          f"2q={row.get('vl_2q')}", flush=True)
                    _persist(rows, n, j2, model, num_layers_grid, fine_tuning_grid)
    finally:
        if session is not None:
            session.__exit__(None, None, None)
        cache.flush()
    return rows


def format_rows(rows) -> str:
    out = ["=" * 82,
           f"{'topo':<11}{'h':<6}{'gap':<8}{'layers':<8}{'ft':<6}{'VL fid':<10}"
           f"{'|dE|':<10}{'VL 2q':<7}",
           "-" * 82]
    for r in rows:
        if "error" in r:
            out.append(f"{r['topology']:<11}{r.get('h',''):<6}ERROR: {r['error'][:40]}")
            continue
        fid = r.get("vl_fidelity")
        fs = f"{fid:.4f}" if isinstance(fid, (int, float)) else str(fid)
        de = r.get("vl_abs_error")
        des = f"{de:.4f}" if isinstance(de, (int, float)) else str(de)
        out.append(
            f"{r['topology']:<11}{r['h']:<6}{r['gap']:<8.3f}{r['num_layers']:<8}"
            f"{r['fine_tuning_iterations']:<6}{fs:<10}{des:<10}{str(r.get('vl_2q')):<7}"
        )
    out.append("=" * 82)
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sweep VL quality knobs on frustrated points")
    p.add_argument("--num-layers", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--fine-tuning", type=int, nargs="+", default=[20, 50])
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--model", default="tfim_frustrated")
    p.add_argument("--job-timeout", type=float, default=900.0)
    p.add_argument("--force-recompute", action="store_true",
                   help="Bypass the cache read and re-run every VL job (cache still refreshed)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[vl_quality_sweep] points={DEFAULT_POINTS} layers={args.num_layers} "
          f"ft={args.fine_tuning} N={args.n} J2={args.j2} "
          f"force_recompute={args.force_recompute}", flush=True)
    rows = run(DEFAULT_POINTS, args.num_layers, args.fine_tuning, args.n, args.j2,
               args.model, args.job_timeout, use_cache=not args.force_recompute)
    print("\n" + format_rows(rows))
    print(f"\n→ Resultados: {study_dir('vl_quality_sweep') / 'vl_quality_sweep_frustrated.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
