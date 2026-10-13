#!/usr/bin/env python
"""Aggregate every known solution for a (N, h, topology, model) into one markdown.

Given a configuration, this scans ALL the places the study records solutions —
portable ansatz specs, the bond_ablation result JSONs (variant_topk / ADAPT /
small / extended / geometric), and exact diagonalization — normalizes them into a
single ``Solution`` schema, de-duplicates, computes the fid-per-CX Pareto front,
and writes a self-updating markdown table sorted by fid/2q.

Design
------
Modular + extensible via COLLECTORS: each source is a function
``collect(cfg) -> list[Solution]``. Adding a source = adding one collector to
``COLLECTORS``; nothing else changes. The markdown is regenerated from scratch on
every run, so it always reflects the current state on disk (auto-updating).

Fidelity metric: in a near-degenerate ground manifold (small gap, e.g. square
frustrated at low h) a single-vector fidelity is meaningless, so each solution
carries which metric it used (``vector`` vs ``subspace``); the table flags it.

Reuses: ``AnsatzSpec`` (specs), ``study_core.ground_state`` (exact row + gap/
regime), ``hva_vl_study_common.study_dir`` (paths). No analysis re-implemented.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/generate_ansatz_summary.py \
        --n 20 --h 0.5                     # square tfim_frustrated hardcoded
    .venv/bin/python scripts/analysis/vl_vs_hva/generate_ansatz_summary.py --n 20 --h 1.0
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import STUDY_ROOT, study_dir  # noqa: E402

TOL_H = 0.005


@dataclass
class Config:
    n: int
    h: float
    topology: str = "square"
    model: str = "tfim_frustrated"
    j2: float = 0.5

    def match_h(self, hv) -> bool:
        return hv is not None and abs(float(hv) - self.h) <= TOL_H


@dataclass
class Solution:
    """One normalized solution for the summary table."""

    name: str
    method: str                 # how it was produced (generation / adapt / exact)
    fidelity: float             # the metric value used for ranking
    fidelity_metric: str        # "vector" | "subspace" | "exact"
    n_2q: int
    n_nnn_bonds: int | None
    n_params: int | None
    de_gap: float | None
    source_file: str
    extra: dict = field(default_factory=dict)

    @property
    def fid_per_cx(self) -> float:
        return (self.fidelity / self.n_2q) if self.n_2q and self.n_2q > 0 else 0.0


# ─────────────────────────── collectors ──────────────────────────────────────
# Each collector takes a Config and returns a list[Solution]. Register in COLLECTORS.

def _degenerate_regime(cfg: Config):
    """(e0, phys_gap, manifold_dim, is_degenerate) via the shared src helper.

    Builds the exact ground manifold to decide which fidelity metric is
    meaningful and to give ΔE/gap a physical denominator. Delegates the eigsh +
    manifold detection to :func:`qmbp_simulation.analysis.fidelity.degenerate_ground_manifold`
    (single source of truth; eigvecs discarded here — only the scalars are used).
    """
    from qmbp_simulation.analysis.fidelity import degenerate_ground_manifold
    from qmbp_simulation.models import make_lattice
    from qmbp_simulation.models.model_registry import get_model_spec

    spec = get_model_spec(cfg.model)
    hk = dict(getattr(spec, "hamiltonian_kwargs", {}))
    hk["J2"] = cfg.j2
    lat = make_lattice(cfg.topology, cfg.n, J=1.0, h=cfg.h)
    H = spec.build_hamiltonian(lat, **hk)
    _eigvecs, e0, phys_gap, m = degenerate_ground_manifold(H)
    return e0, phys_gap, m, (m > 1)


def collect_exact(cfg: Config) -> list[Solution]:
    """The exact ground state: fidelity 1, zero gates — the reference ceiling."""
    try:
        e0, phys_gap, m, _deg = _degenerate_regime(cfg)
    except Exception as exc:  # diagonalization may be out of reach at large N
        print(f"  [exact skipped: {exc}]", file=sys.stderr)
        return []
    return [Solution(
        name="exact_diagonalization", method="exact", fidelity=1.0,
        fidelity_metric="exact", n_2q=0, n_nnn_bonds=None, n_params=None,
        de_gap=0.0, source_file="eigsh(k=4)",
        extra={"e0": e0, "phys_gap": phys_gap, "manifold_dim": m})]


def collect_specs(cfg: Config) -> list[Solution]:
    """Portable AnsatzSpec files — the cleanest source (exact nnn edge counts)."""
    from qmbp_simulation.circuits.ansatz_spec import AnsatzSpec

    out: list[Solution] = []
    base = study_dir("ansatz_specs")
    for fp in glob.glob(str(base / f"*_{cfg.topology}_N{cfg.n}_h*.spec.json")):
        try:
            s = AnsatzSpec.load(fp)
        except Exception:
            continue
        if not cfg.match_h(s.h) or s.fidelity is None:
            continue
        n_nnn = len(s.nnn_edges) if s.nnn_edges else None
        out.append(Solution(
            name=s.name, method="generation(spec)", fidelity=float(s.fidelity),
            fidelity_metric="vector", n_2q=int(s.n_2q or 0), n_nnn_bonds=n_nnn,
            n_params=s.n_params, de_gap=None, source_file=Path(fp).name,
            extra={"base_variant": s.base_variant}))
    return out


def collect_bond_ablation(cfg: Config) -> list[Solution]:
    """The bond_ablation result JSONs: variant_topk / small / extended / geometric.

    Normalizes the differing schemas (n_2q_transpiled vs pool_2q, best_fidelity vs
    report_fidelity, n_nnn_bonds vs n_nnn) into Solutions. ADAPT (poolA) files are
    handled by collect_adapt below, so they are skipped here.
    """
    out: list[Solution] = []
    base = study_dir("bond_ablation")
    for fp in glob.glob(str(base / f"*{cfg.topology}_N{cfg.n}_h*.json")):
        name = Path(fp).name
        if name.startswith("adapt_poolA"):
            continue  # handled by collect_adapt
        try:
            d = json.loads(Path(fp).read_text())
        except Exception:
            continue
        rows = d.get("rows", d) if isinstance(d, dict) else d
        if not isinstance(rows, list):
            continue
        for r in rows:
            if not isinstance(r, dict):
                continue
            hv = r.get("h", d.get("h") if isinstance(d, dict) else None)
            if not cfg.match_h(hv):
                continue
            # Metric selection. In a degenerate manifold the subspace fidelity is
            # the correct, member-agnostic metric; prefer it when available. A run
            # signals degeneracy either explicitly (``degenerate``) or implicitly
            # via ``manifold_dim > 1`` (variant_topk's P2 field). Fall back to the
            # vector fidelity (report_ or best_) otherwise.
            sub = r.get("subspace_fidelity")
            degenerate = bool(r.get("degenerate")) or (r.get("manifold_dim") or 1) > 1
            if degenerate and sub is not None:
                fid, metric = sub, "subspace"
            else:
                fid = r.get("report_fidelity")
                metric = "subspace" if (r.get("degenerate") and sub is not None) else "vector"
                if fid is None:
                    fid = r.get("best_fidelity")
                    metric = "vector"
            n2q = r.get("n_2q_transpiled")
            if fid is None or n2q is None:
                continue
            n_nnn = r.get("n_nnn_bonds", r.get("n_nnn"))
            label = r.get("variant") or r.get("label") or name.replace(".json", "")
            # Reject ΔE/gap computed against a near-zero (degenerate) gap — those
            # blow up to hundreds/thousands and are physically meaningless. The
            # physical ΔE/gap only exists where the runner used phys_gap.
            degap = r.get("de_gap")
            if degap is not None and (not np.isfinite(degap) or degap >= 1e3):
                degap = None
            out.append(Solution(
                name=str(label), method="generation(sweep)", fidelity=float(fid),
                fidelity_metric=metric, n_2q=int(n2q), n_nnn_bonds=n_nnn,
                n_params=r.get("n_params"), de_gap=degap,
                source_file=name, extra={}))
    return out


def collect_adapt(cfg: Config) -> list[Solution]:
    """ADAPT (poolA) runs: harvest the max-fid point and the Pareto fid/CX point."""
    out: list[Solution] = []
    base = study_dir("bond_ablation")
    for fp in glob.glob(str(base / f"adapt_poolA_{cfg.topology}_N{cfg.n}*.json")):
        try:
            d = json.loads(Path(fp).read_text())
        except Exception:
            continue
        rows = d.get("rows", d) if isinstance(d, dict) else d
        if not isinstance(rows, list):
            continue
        for r in rows:
            if not isinstance(r, dict) or not cfg.match_h(r.get("h")):
                continue
            name = Path(fp).name
            # max-fid headline
            if r.get("pool_best_fidelity") is not None and r.get("pool_2q"):
                out.append(Solution(
                    name=f"ADAPT({name.replace('.json','').split('_')[-1]})-maxfid",
                    method="adapt_vqe", fidelity=float(r["pool_best_fidelity"]),
                    fidelity_metric="vector", n_2q=int(r["pool_2q"]),
                    n_nnn_bonds=r.get("pool_n_bonds"), n_params=None,
                    de_gap=None, source_file=name, extra={"layers": r.get("pool_n_layers")}))
            # Pareto fid/CX point (if recorded and distinct)
            if r.get("pareto_fid") is not None and r.get("pareto_2q"):
                out.append(Solution(
                    name=f"ADAPT({name.replace('.json','').split('_')[-1]})-pareto",
                    method="adapt_vqe", fidelity=float(r["pareto_fid"]),
                    fidelity_metric="vector", n_2q=int(r["pareto_2q"]),
                    n_nnn_bonds=None, n_params=None, de_gap=None,
                    source_file=name, extra={}))
    return out


COLLECTORS = [collect_exact, collect_specs, collect_bond_ablation, collect_adapt]


# ─────────────────────────── aggregation ─────────────────────────────────────

def _norm_name(name: str) -> str:
    """Canonical structure name for dedup: strip _full suffix and family aliases so
    the same physical ansatz coming from a spec and a sweep collapses to one key."""
    n = name.replace("_full", "")
    # the small-ansatz 'nnn2_half' and the geometric/variant 'full' are the same
    # p2_half_nn full structure at N=20 — map both to a common key by (n_2q) later.
    return n


def _dedup(sols: list[Solution]) -> list[Solution]:
    """Collapse duplicates that are the SAME physical ansatz across sources.

    Key = (normalized-name, n_2q) — NOT method, since the identical circuit is
    recorded both as a portable spec and in a sweep JSON. When merging, prefer the
    record that (1) uses the subspace metric in a degenerate regime, (2) has the
    nnn-bond count populated, (3) has a sane ΔE/gap. Keeps the richest row.
    """
    DEGAP_SANE = 50.0  # ΔE/gap above this came from a degenerate near-zero gap — garbage

    def sane_degap(d):
        return d if (d is not None and np.isfinite(d) and d < DEGAP_SANE) else None

    def richness(s: Solution) -> tuple:
        return (s.fidelity, s.fidelity_metric == "subspace",
                s.n_nnn_bonds is not None, sane_degap(s.de_gap) is not None)

    best: dict[tuple, Solution] = {}
    for s in sols:
        if s.method == "exact":
            best[("__exact__", 0)] = s
            continue
        s.de_gap = sane_degap(s.de_gap)
        # Key by physical identity: same gate count AND same fidelity (to 4 dp) is
        # the same converged ansatz regardless of the name it was saved under
        # (e.g. 'nnn2_half' the small-runner alias == 'full' the geometric ref).
        key = (s.n_2q, round(s.fidelity, 4))
        cur = best.get(key)
        if cur is None:
            best[key] = s
            continue
        keep, other = (s, cur) if richness(s) > richness(cur) else (cur, s)
        if keep.n_nnn_bonds is None and other.n_nnn_bonds is not None:
            keep.n_nnn_bonds = other.n_nnn_bonds
        if keep.de_gap is None and other.de_gap is not None:
            keep.de_gap = other.de_gap
        if keep.n_params is None and other.n_params is not None:
            keep.n_params = other.n_params
        best[key] = keep
    return list(best.values())


def _pareto_front(sols: list[Solution]) -> set[int]:
    """Indices of Pareto-optimal solutions (max fidelity, min 2q — not dominated).

    A solution is dominated if another has ≥ fidelity AND ≤ 2q (strictly better in
    at least one). The non-dominated set is the frontier the user picks from.
    """
    front = set()
    for i, a in enumerate(sols):
        dominated = False
        for j, b in enumerate(sols):
            if i == j:
                continue
            if b.fidelity >= a.fidelity and b.n_2q <= a.n_2q and (b.fidelity > a.fidelity or b.n_2q < a.n_2q):
                dominated = True
                break
        if not dominated:
            front.add(i)
    return front


def aggregate(cfg: Config) -> list[Solution]:
    sols: list[Solution] = []
    for collector in COLLECTORS:
        try:
            sols.extend(collector(cfg))
        except Exception as exc:  # a broken source must not sink the whole report
            print(f"  [collector {collector.__name__} failed: {exc}]", file=sys.stderr)
    sols = _dedup(sols)
    # Sort by fid/CX desc (exact row, fid/cx=0 by zero gates, sinks to the bottom
    # but is pulled to the top as the reference — handled in rendering).
    sols.sort(key=lambda s: s.fid_per_cx, reverse=True)
    return sols


# ─────────────────────────── rendering ───────────────────────────────────────

def render_markdown(cfg: Config, sols: list[Solution]) -> str:
    exact = next((s for s in sols if s.method == "exact"), None)
    ranked = [s for s in sols if s.method != "exact"]
    front = _pareto_front(ranked)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append(f"# Soluciones — {cfg.topology} {cfg.model} | N={cfg.n} h={cfg.h}")
    lines.append("")
    lines.append(f"*Auto-generado: {ts}. Regenerar con `generate_ansatz_summary.py --n {cfg.n} --h {cfg.h}`.*")
    lines.append("")
    if exact is not None:
        e = exact.extra
        reg = "degenerado (doblete)" if e.get("manifold_dim", 1) > 1 else "no degenerado"
        lines.append(f"**Diagonalización exacta:** E0 = {e['e0']:.5f} | gap físico = {e['phys_gap']:.4f} "
                     f"| régimen: {reg} (manifold dim {e.get('manifold_dim')}).")
        lines.append("")
    lines.append(f"**{len(ranked)} soluciones** agregadas de specs + sweeps + ADAPT. "
                 "Ordenadas por fidelidad/2q (mejor eficiencia primero). "
                 "★ = frontera de Pareto (no dominada en fid vs 2q).")
    lines.append("")
    lines.append("| # | Ansatz | Método | Fidelidad | métrica | 2q | nnn bonds | fid/2q | ΔE/gap | Pareto | fuente |")
    lines.append("|---|--------|--------|:---------:|:-------:|:--:|:---------:|:------:|:------:|:------:|--------|")
    for i, s in enumerate(ranked):
        star = "★" if i in front else ""
        nnn = str(s.n_nnn_bonds) if s.n_nnn_bonds is not None else "–"
        deg = f"{s.de_gap:.2f}" if s.de_gap is not None else "–"
        met = {"subspace": "sub", "vector": "vec", "exact": "exact"}.get(s.fidelity_metric, s.fidelity_metric)
        lines.append(f"| {i+1} | {s.name[:34]} | {s.method} | {s.fidelity:.4f} | {met} | "
                     f"{s.n_2q} | {nnn} | {s.fid_per_cx:.3e} | {deg} | {star} | {s.source_file[:30]} |")
    lines.append("")
    lines.append("## Notas")
    lines.append("")
    lines.append("- **métrica**: `vec` = fidelidad al vector de estado fundamental; "
                 "`sub` = fidelidad al subespacio del doblete degenerado (la correcta cuando el gap→0). "
                 "No comparar `vec` vs `sub` directamente.")
    lines.append("- **fid/2q**: fidelidad por compuerta de 2 qubits — la métrica de eficiencia del estudio.")
    lines.append("- **nnn bonds**: cantidad de bonds next-nearest seleccionados (portadores de la frustración J2).")
    lines.append("- **Pareto (★)**: solución no dominada — ninguna otra tiene ≥ fidelidad con ≤ 2q.")
    lines.append("")
    return "\n".join(lines)


def output_path(cfg: Config) -> Path:
    sub = study_dir("ansatz_summary") / f"{cfg.topology}_{cfg.model}"
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"solutions_N{cfg.n}_h{cfg.h:.2f}.md"


# ───────────────────────── overview (ONE general file) ───────────────────────
#
# Design rule: analysis markdown is a VIEW, not a data layer. We emit ONE general
# overview per (topology, model) — a compact best-per-config table across all N/h
# — instead of ~90 per-config files that are noise and tempt other tooling to
# parse markdown. Anything that needs the full per-config detail reads the JSON
# artifacts / specs directly (the real data layer), never this markdown.

_TOPOLOGIES = ("chain_1d", "heavy_hex", "kagome", "square", "triangular", "ladder")
_CFG_RE = re.compile(
    rf"_(?P<topo>{'|'.join(_TOPOLOGIES)})_N(?P<n>\d+)_h(?P<h>[0-9.]+)\.(spec\.json|json)$"
)


def discover_configs(topology: str, model: str, j2: float) -> list[Config]:
    """Find every (N, h) with artifacts for this topology, from spec/JSON names."""
    seen: set[tuple[int, float]] = set()
    for pat in (study_dir("ansatz_specs") / "*.spec.json",
                study_dir("bond_ablation") / "*.json"):
        for fp in glob.glob(str(pat)):
            m = _CFG_RE.search(Path(fp).name)
            if not m or m.group("topo") != topology:
                continue
            try:
                seen.add((int(m.group("n")), round(float(m.group("h")), 2)))
            except ValueError:
                continue
    return [Config(n=n, h=h, topology=topology, model=model, j2=j2)
            for n, h in sorted(seen)]


def overview_path(topology: str, model: str) -> Path:
    """The ONE general overview markdown for a (topology, model)."""
    study_dir("ansatz_summary").mkdir(parents=True, exist_ok=True)
    return study_dir("ansatz_summary") / f"OVERVIEW_{topology}_{model}.md"


def render_overview(topology: str, model: str, per_cfg: list[tuple[Config, list[Solution]]]) -> str:
    """One compact table: the best (Pareto) solutions per (N, h), all configs.

    A single navigable view of the whole study for a topology — not one file per
    config. For each config we surface the Pareto-front solutions (the ones a
    reader actually picks from); full detail lives in the JSON artifacts.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Resumen general — {topology} {model}",
        "",
        f"*Auto-generado: {ts}. Regenerar con `generate_ansatz_summary.py --overview`.*",
        "",
        "Vista única de todas las configuraciones (N, h) con artefactos. Por cada "
        "config se muestra la **frontera de Pareto** (fid vs 2q). El detalle completo "
        "vive en los JSON/specs — este markdown es una vista, no una fuente de datos.",
        "",
        "| N | h | régimen | # sols | mejor fid | 2q@mejor | mejor fid/2q (Pareto) | 2q@fid/2q |",
        "|--:|--:|---------|:------:|----------:|---------:|----------------------:|----------:|",
    ]
    for cfg, sols in per_cfg:
        ranked = [s for s in sols if s.method != "exact"]
        if not ranked:
            continue
        exact = next((s for s in sols if s.method == "exact"), None)
        reg = "—"
        if exact is not None:
            reg = "degen" if exact.extra.get("manifold_dim", 1) > 1 else "no-degen"
        best_fid = max(ranked, key=lambda s: s.fidelity)
        best_fpc = max(ranked, key=lambda s: s.fid_per_cx)
        lines.append(
            f"| {cfg.n} | {cfg.h:.2f} | {reg} | {len(ranked)} | "
            f"{best_fid.fidelity:.4f} | {best_fid.n_2q} | "
            f"{best_fpc.fid_per_cx:.3e} | {best_fpc.n_2q} |"
        )
    lines += [
        "",
        "## Notas",
        "",
        "- **mejor fid**: mayor fidelidad alcanzada en esa config (y su conteo 2q).",
        "- **mejor fid/2q**: solución más eficiente de la frontera de Pareto.",
        "- El detalle por-ansatz (todas las variantes, θ, edges) está en "
        "`bond_ablation/*.json` y `ansatz_specs/*.spec.json` — no en markdown.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Aggregate study solutions into ONE general overview markdown "
        "(default), or print a single config to stdout with --n/--h.")
    p.add_argument("--overview", action="store_true",
                   help="Write the single general OVERVIEW_<topo>_<model>.md (default when no --n/--h).")
    p.add_argument("--n", type=int, default=None, help="Single config N (prints to stdout, no file).")
    p.add_argument("--h", type=float, default=None, help="Single config h (prints to stdout, no file).")
    p.add_argument("--topology", default="square")
    p.add_argument("--model", default="tfim_frustrated")
    p.add_argument("--j2", type=float, default=0.5)
    args = p.parse_args(argv)

    # Single-config mode: PRINT to stdout only — never writes a per-config file
    # (those were ~90 noise files; the overview is the written artifact).
    if args.n is not None and args.h is not None and not args.overview:
        cfg = Config(n=args.n, h=args.h, topology=args.topology, model=args.model, j2=args.j2)
        sols = aggregate(cfg)
        print(render_markdown(cfg, sols))
        return 0

    # Default / --overview: ONE general file across all discovered configs.
    configs = discover_configs(args.topology, args.model, args.j2)
    if not configs:
        print(f"[summary] no artifacts found for {args.topology} {args.model}", flush=True)
        return 0
    print(f"[summary] building overview for {len(configs)} config(s) of {args.topology} {args.model}", flush=True)
    per_cfg = [(cfg, aggregate(cfg)) for cfg in configs]
    out = overview_path(args.topology, args.model)
    out.write_text(render_overview(args.topology, args.model, per_cfg))
    print(f"[summary] overview -> {out.relative_to(STUDY_ROOT.parent.parent)} "
          f"({len(configs)} configs, ONE file)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
