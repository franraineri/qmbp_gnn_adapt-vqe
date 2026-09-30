"""Reusable query layer over the HVA study result JSONs.

A single, general API for reading back what past runs produced — used both by
runners (to warm-start from the best known point) and by data/results analysis
(to study angles and best fidelities per configuration) from now on. Runners
consume only the slice they care about; analysis can inspect everything.

The result tree (``results/hva_vl_study/hva_nnn_sweep/*.json``) is written by
several experiments (fair-convergence, basin-count, ansatz-variants, sweeps).
Each JSON carries a container (``rows`` / ``runs`` / ``per_h``) of records; every
record now persists ``theta_init`` / ``theta_final`` (see study_runner._run_record
and the standalone runners) plus a fidelity/energy. This module normalizes those
into a uniform ``ResultRecord`` regardless of which experiment wrote them.

Public API
----------
- ``iter_records(...)``           : yield normalized records matching a config filter
- ``query_results(...)``          : list of matching records (sorted by fidelity desc)
- ``best_result(...)``            : the single highest-fidelity record (or None)
- ``load_best_theta(...)``        : best ``theta_final`` for a config → warm-start seed
- ``list_configs()``              : the (topology, N, p, h) configs present on disk
- ``fidelity_summary(...)``       : min/max/mean/count over matching records

All matching is tolerant on ``h`` and optional on ``p_layers`` (None = any), and
can be guarded by ``expected_len`` (the θ vector length = ansatz fingerprint).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Reuse the study's canonical result location without duplicating the constant.
from hva_vl_study_common import study_dir


@dataclass
class ResultRecord:
    """One normalized optimization record harvested from a result JSON."""

    fidelity: float | None
    energy: float | None
    theta_init: list[float] | None
    theta_final: list[float] | None
    source_file: str
    # config context (best-effort, from the file's top level / params)
    topology: str | None = None
    n_qubits: int | None = None
    p_layers: int | None = None
    h: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_params(self) -> int | None:
        if self.theta_final is not None:
            return len(self.theta_final)
        if self.theta_init is not None:
            return len(self.theta_init)
        return None

    @property
    def converged(self) -> bool | None:
        """Convergence flag (explicit or inferred from nit<maxiter); None if unknown."""
        return self.extra.get("converged")


def _iter_result_files(subdir: str = "hva_nnn_sweep") -> Iterator[Path]:
    yield from sorted(study_dir(subdir).glob("*.json"))


def _num(node: dict, *keys):
    """First numeric value among ``keys`` in ``node``, else None."""
    for k in keys:
        v = node.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _infer_converged(node: dict, ctx: dict) -> bool | None:
    """Convergence flag for a run record.

    Prefer an explicit ``converged`` field. Otherwise infer from ``nit`` vs the
    optimizer iteration cap (``maxiter``, propagated in ctx): a run that stopped
    strictly below the cap converged; one that hit the cap is starved. Returns
    None when neither signal is available (caller must not assume convergence).
    """
    if isinstance(node.get("converged"), bool):
        return node["converged"]
    nit = node.get("nit")
    maxiter = ctx.get("maxiter")
    if isinstance(nit, (int, float)) and isinstance(maxiter, (int, float)) and maxiter > 0:
        return int(nit) < int(maxiter)
    return None


# Context keys propagated from an enclosing block (per_h / points / top level)
# down to the run records nested inside it, since a run dict carries only its
# own theta/fidelity/nit and not the config it belongs to.
_CTX_PROPAGATE = {
    "n_qubits": "n",
    "N": "n",
    "p_layers": "p",
    "h": "h",
    "topology": "topology",
    "n_nn": "n_nn",
    "n_nnn": "n_nnn",
    "gap": "gap",
    "maxiter": "maxiter",
}


def _make_record(node: dict, ctx: dict, src: str, theta_final, theta_init, fid):
    return ResultRecord(
        fidelity=(float(fid) if isinstance(fid, (int, float)) else None),
        energy=_num(node, "energy", "e_best", "e_final", "e_vqe"),
        theta_init=theta_init,
        theta_final=theta_final,
        source_file=src,
        topology=ctx.get("topology"),
        n_qubits=ctx.get("n"),
        p_layers=ctx.get("p"),
        h=ctx.get("h"),
        extra={
            "converged": _infer_converged(node, ctx),
            "n_nn": ctx.get("n_nn"),
            "n_nnn": ctx.get("n_nnn"),
            "gap": ctx.get("gap"),
            **{
                k: node[k]
                for k in ("variant", "seed_type", "seed_order", "diagnosis", "k", "sigma", "type", "nit", "restart")
                if k in node
            },
        },
    )


def _harvest(node: Any, ctx: dict, src: str, out: list[ResultRecord]) -> None:
    """Recursively collect angle-bearing records under two naming conventions.

    Convention A (study_runner / basin-count): ``theta_final`` + ``theta_init``
    with sibling ``fidelity``. Convention B (per-run records in per_h/points
    blocks): a bare ``theta`` with sibling ``fidelity`` (per-run optimum), and a
    block-level ``best_theta`` with the block ``fidelity`` (best of the block).

    Config context (N, p, h, n_nn, n_nnn, gap, maxiter) is propagated from each
    enclosing dict so run records nested inside per_h/points inherit it.
    """
    if isinstance(node, dict):
        # extend context with any config fields present at this level
        local = dict(ctx)
        for jkey, ckey in _CTX_PROPAGATE.items():
            if jkey in node and isinstance(node[jkey], (int, float, str)):
                local[ckey] = (
                    int(node[jkey])
                    if ckey in ("n", "p", "n_nn", "n_nnn") and isinstance(node[jkey], (int, float))
                    else (float(node[jkey]) if ckey in ("h", "gap") else node[jkey])
                )
        # Convention A: theta_final / best_theta_final
        for theta_key, fid_key in (("theta_final", "fidelity"), ("best_theta_final", "best_fidelity")):
            if theta_key in node and isinstance(node[theta_key], list):
                fid = node.get(fid_key, node.get("fidelity", node.get("best_fidelity")))
                out.append(
                    _make_record(node, local, src, node[theta_key], node.get(theta_key.replace("final", "init")), fid)
                )
        # Convention B: bare theta (per-run) and best_theta (per-block)
        if "theta" in node and isinstance(node["theta"], list):
            out.append(
                _make_record(node, local, src, node["theta"], None, node.get("fidelity", node.get("best_fidelity")))
            )
        if "best_theta" in node and isinstance(node["best_theta"], list):
            # block-level best: fidelity is the block's own fidelity field
            out.append(
                _make_record(
                    node, local, src, node["best_theta"], None, node.get("fidelity", node.get("best_fidelity"))
                )
            )
        for v in node.values():
            _harvest(v, local, src, out)
    elif isinstance(node, list):
        for v in node:
            _harvest(v, ctx, src, out)


def iter_records(
    *,
    topology: str | None = None,
    n: int | None = None,
    p_layers: int | None = None,
    h: float | None = None,
    expected_len: int | None = None,
    subdir: str = "hva_nnn_sweep",
    tol_h: float = 0.005,
) -> list[ResultRecord]:
    """Return normalized records matching the config filter (any field None = any)."""
    records: list[ResultRecord] = []
    for jf in _iter_result_files(subdir):
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        top_n = data.get("N") or data.get("n_qubits")
        top_p = data.get("p_layers")
        top_h = data.get("h")
        top_topo = data.get("topology")
        # top-level config gate (cheap)
        if n is not None and top_n is not None and int(top_n) != n:
            continue
        if p_layers is not None and top_p is not None and int(top_p) != p_layers:
            continue
        if h is not None and isinstance(top_h, (int, float)) and abs(float(top_h) - h) > tol_h:
            continue
        if topology is not None and top_topo is not None and str(top_topo) != topology:
            continue
        ctx = {
            "topology": top_topo,
            "n": (int(top_n) if top_n is not None else None),
            "p": (int(top_p) if top_p is not None else None),
            "h": (float(top_h) if isinstance(top_h, (int, float)) else None),
        }
        _harvest(data, ctx, jf.name, records)
    # Dedup: a block-level best_theta usually duplicates the top run's theta in
    # the same file. Collapse records that share (file, fidelity, first/last
    # angle) so basin/statistics counts reflect distinct optimization outcomes.
    seen: set = set()
    deduped: list[ResultRecord] = []
    for r in records:
        if r.theta_final is None:
            deduped.append(r)
            continue
        tf = r.theta_final
        sig = (
            r.source_file,
            round(r.fidelity, 9) if r.fidelity is not None else None,
            round(float(tf[0]), 9),
            round(float(tf[-1]), 9),
            len(tf),
        )
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(r)
    records = deduped
    # config filters that also apply to context propagated into nested records
    if n is not None:
        records = [r for r in records if r.n_qubits is None or r.n_qubits == n]
    if p_layers is not None:
        records = [r for r in records if r.p_layers is None or r.p_layers == p_layers]
    if h is not None:
        records = [r for r in records if r.h is None or abs(r.h - h) <= tol_h]
    if topology is not None:
        records = [r for r in records if r.topology is None or r.topology == topology]
    if expected_len is not None:
        records = [r for r in records if r.n_params == expected_len]
    records.sort(key=lambda r: (r.fidelity if r.fidelity is not None else -1.0), reverse=True)
    return records


def query_results(**kw) -> list[ResultRecord]:
    """Alias for iter_records — all matching records, best fidelity first."""
    return iter_records(**kw)


def best_result(**kw) -> ResultRecord | None:
    """The single highest-fidelity record matching the filter, or None."""
    recs = iter_records(**kw)
    return recs[0] if recs else None


def load_best_theta(
    *,
    topology: str,
    n: int,
    p_layers: int | None,
    h: float,
    expected_len: int | None = None,
    subdir: str = "hva_nnn_sweep",
    tol_h: float = 0.005,
):
    """Best ``theta_final`` for a config → warm-start seed.

    Returns ``(fidelity, theta_np, source_file)`` or ``None``. Thin wrapper over
    :func:`best_result` that returns the angle vector ready to seed an optimizer.

    Warm-start guardrail: USE when measuring *achievable fidelity* (production /
    refinement / transfer / scaling). DO NOT use when *characterizing the
    landscape / difficulty* (basin-count, ceiling, fair-convergence, seed
    comparison) — it biases the result. See REPORT_HEURISTIC_warmstart_restarts.md
    § "Warm-start guardrail".
    """
    rec = best_result(
        topology=topology, n=n, p_layers=p_layers, h=h, expected_len=expected_len, subdir=subdir, tol_h=tol_h
    )
    if rec is None or rec.theta_final is None:
        return None
    return (rec.fidelity, np.asarray(rec.theta_final, float), rec.source_file)


def fidelity_summary(**kw) -> dict:
    """min/max/mean/count of fidelity over matching records (for analysis)."""
    recs = [r for r in iter_records(**kw) if r.fidelity is not None]
    if not recs:
        return {"count": 0, "min": None, "max": None, "mean": None}
    fids = np.array([r.fidelity for r in recs])
    return {"count": len(fids), "min": float(fids.min()), "max": float(fids.max()), "mean": float(fids.mean())}


def decompose_theta(
    theta,
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    *,
    rx_final: bool = False,
    rz_final: bool = False,
) -> dict:
    """Split a flat HVA angle vector into per-block arrays by physical role.

    Layout per layer is ``[θ_nn (n_nn) | θ_nnn (n_nnn) | θ_x (n_qubits)]`` repeated
    ``p_layers`` times, with optional trailing ``θ_x_final`` / ``θ_z_final`` blocks.
    Returns a dict of numpy arrays keyed ``theta_nn``, ``theta_nnn``, ``theta_x``
    (concatenated across layers), plus ``theta_x_final`` / ``theta_z_final`` when
    present — so angles are analyzed by role, never as an opaque vector. Raises if
    the length does not match the declared layout.
    """
    theta = np.asarray(theta, float)
    per = n_nn + n_nnn + n_qubits
    expected = per * p_layers + (n_qubits if rx_final else 0) + (n_qubits if rz_final else 0)
    if theta.size != expected:
        raise ValueError(
            f"theta length {theta.size} != layout {expected} "
            f"(n_nn={n_nn} n_nnn={n_nnn} n_q={n_qubits} p={p_layers} "
            f"rx_final={rx_final} rz_final={rz_final})"
        )
    nn, nnn, x = [], [], []
    off = 0
    for _ in range(p_layers):
        nn.append(theta[off : off + n_nn])
        off += n_nn
        nnn.append(theta[off : off + n_nnn])
        off += n_nnn
        x.append(theta[off : off + n_qubits])
        off += n_qubits
    out = {
        "theta_nn": np.concatenate(nn) if nn else np.array([]),
        "theta_nnn": np.concatenate(nnn) if nnn else np.array([]),
        "theta_x": np.concatenate(x) if x else np.array([]),
    }
    if rx_final:
        out["theta_x_final"] = theta[off : off + n_qubits]
        off += n_qubits
    if rz_final:
        out["theta_z_final"] = theta[off : off + n_qubits]
        off += n_qubits
    return out


def block_stats(decomposed: dict) -> dict:
    """Per-block mean/std/absmean of a decomposed theta (for angle-study tables)."""
    return {
        b: {"mean": float(np.mean(v)), "std": float(np.std(v)), "absmean": float(np.mean(np.abs(v))), "n": int(v.size)}
        for b, v in decomposed.items()
        if v.size
    }


def list_configs(subdir: str = "hva_nnn_sweep") -> list[dict]:
    """Distinct (topology, N, p_layers, h) configs present in the result tree."""
    seen = {}
    for jf in _iter_result_files(subdir):
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        key = (data.get("topology"), data.get("N") or data.get("n_qubits"), data.get("p_layers"), data.get("h"))
        seen.setdefault(key, []).append(jf.name)
    return [
        {"topology": k[0], "n_qubits": k[1], "p_layers": k[2], "h": k[3], "files": v}
        for k, v in sorted(seen.items(), key=lambda x: str(x[0]))
    ]
