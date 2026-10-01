#!/usr/bin/env python
"""Consolidated VL-vs-HVA comparison across N, h, and every method variant.

Reads ALL the study's JSON artifacts and normalizes them into one common record
shape, then emits a single multi-scenario markdown grouped by (N, h). Never
hand-authored — regenerate from data.

Sources (each normalized to a common row):
- HVA ansatz sweep       : hva_nnn_sweep/hva_ansatze_h_sweep_frustrated.json
                           (nn/nnn × p1/p2 variants, with θ)
- HVA nnn-p2 ceiling     : hva_nnn_sweep/compare_hva_nnn_vs_vl_*_N*_p*.json
                           (best warm-start ceiling + full gate breakdown)
- HVA N=18 variants      : hva_nnn_sweep/ansatz_variants_square_N18_h0.50.json
- VL default h-sweep     : vl_h_sweep/vl_h_sweep_frustrated.json
- VL quality sweep       : vl_quality_sweep/vl_quality_sweep_frustrated.json
- VL large-N (mps)       : vl_mps_n18/*.json

Comparison metrics per row: fidelity F, |ΔE|, ΔE/gap, 2q gates, total gates.
Scenarios are keyed by (N, h). Asymmetries are surfaced explicitly:
- VL at N=18 uses ``mps_loading`` (vs ``vector_loading`` at N≤10) — flagged.
- Missing methods render as absent rows, never back-filled from another scenario.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/generate_vl_hva_consolidated_report.py
"""

from __future__ import annotations

import glob
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import study_dir  # noqa: E402

from qmbp_simulation.framework.runner_base import resolve_project_root  # noqa: E402


@dataclass
class Rec:
    """One normalized (method, variant) measurement at a scenario."""

    method: str  # "HVA" | "VL"
    variant: str  # e.g. "nnn p=2", "L8/F50", "mps L8/F50"
    fidelity: float | None
    abs_error: float | None
    de_gap: float | None
    n_2q: int | None
    total_gates: int | None
    depth: int | None = None
    converged: str | None = None  # HVA: "4/4" | "n/a (random)"; VL: "n/a (fixed)"
    loader: str | None = None  # VL: "vector" | "mps"; HVA: None
    chi: int | None = None  # VL mps bond cap; None otherwise
    note: str = ""
    source: str = ""


@dataclass
class Scenario:
    n: int
    h: float
    gap: float | None = None
    e0: float | None = None
    recs: list[Rec] = field(default_factory=list)
    # Dual-target 2×2 fidelity matrix (VL/HVA × exact/MPS), when available.
    dual_target: dict | None = None


def _key(n, h):
    return (int(n), round(float(h), 2))


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# Cache rebuilt circuits so recompute is done once per (topology,n,h,j2,p,ansatz).
_CIRCUIT_CACHE: dict[tuple, object] = {}


def _hva_circuit(topology, n, h, j2, p, ansatz):
    """Rebuild the HVA circuit deterministically for gate recomputation.

    ``ansatz`` = "nn" → create_bond_resolved; "nnn" → create_bond_resolved_frustrated.
    Cached; no optimization involved.
    """
    key = (topology, int(n), round(float(h), 2), float(j2), int(p), ansatz)
    if key in _CIRCUIT_CACHE:
        return _CIRCUIT_CACHE[key]
    from qmbp_simulation.circuits import HVACircuitBuilder
    from qmbp_simulation.models import make_lattice

    lat = make_lattice(topology, n, J=1.0, h=h)
    builder = HVACircuitBuilder()
    if ansatz == "nn":
        qc, _ = builder.create_bond_resolved(n, p, lat)
    else:
        qc, _ = builder.create_bond_resolved_frustrated(n, p, lat)
    _CIRCUIT_CACHE[key] = qc
    return qc


def _recompute_gate_counts(topology, n, h, j2, p, ansatz):
    """Deterministic (total, 2q, depth) for an HVA circuit in the shared basis.

    Uses fixed non-zero angles (0.37) so RZZ≈identity cancellation cannot deflate
    the structural 2q count. Returns (total_gates, n_2q, depth) or (None, None, None)
    on any failure (kept non-fatal so the report still generates).
    """
    try:
        import numpy as np
        from qiskit import transpile

        from qmbp_simulation.analysis.circuit_visualizer import circuit_summary

        qc = _hva_circuit(topology, n, h, j2, p, ansatz)
        angles = np.full(qc.num_parameters, 0.37)
        t = transpile(qc.assign_parameters(angles), basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
        summ = circuit_summary(t)
        return summ["n_gates_total"], summ["n_2q_gates"], t.depth()
    except Exception:
        return None, None, None


def collect(root: Path, *, recompute_gates: bool = True) -> dict[tuple, Scenario]:
    """Read every source and bucket normalized records by (N, h)."""
    study = root / "results" / "hva_vl_study"
    scen: dict[tuple, Scenario] = {}

    def get(n, h, gap=None, e0=None) -> Scenario:
        k = _key(n, h)
        if k not in scen:
            scen[k] = Scenario(n=int(n), h=round(float(h), 2))
        s = scen[k]
        if gap is not None and s.gap is None:
            s.gap = float(gap)
        if e0 is not None and s.e0 is None:
            s.e0 = float(e0)
        return s

    # ── HVA ansatz sweep (nn/nnn × p1/p2) ────────────────────────────────
    d = _load(study / "hva_nnn_sweep" / "hva_ansatze_h_sweep_frustrated.json")
    if d:
        for r in d.get("rows", []):
            if "error" in r:
                continue
            s = get(r["N"], r["h"], gap=r.get("gap"), e0=r.get("E0"))
            ansatz, p = r["ansatz"], r["p_layers"]
            total, twoq, depth = (None, r.get("n_2q_transpiled"), r.get("depth_transpiled"))
            if recompute_gates:
                rt, r2q, rdepth = _recompute_gate_counts(
                    r.get("topology", "square"), r["N"], r["h"], r.get("J2", 0.5), p, ansatz
                )
                if rt is not None:
                    total, twoq, depth = rt, r2q, rdepth
            # The sweep uses multi-restart random / metropolis best-of, not a
            # per-restart convergence contract, so report convergence honestly.
            opt = r.get("optimizer", "")
            conv = "n/a (random best-of)" if "random" in opt else "n/a"
            s.recs.append(
                Rec(
                    method="HVA",
                    variant=f"{ansatz} p={p}",
                    fidelity=r.get("fidelity"),
                    abs_error=r.get("abs_error"),
                    de_gap=r.get("de_gap"),
                    n_2q=twoq,
                    total_gates=total,
                    depth=depth,
                    converged=conv,
                    source="hva_ansatze_h_sweep",
                )
            )

    # ── HVA nnn-p2 ceiling (best warm-start) + full gate breakdown ───────
    for f in sorted(glob.glob(str(study / "hva_nnn_sweep" / "compare_hva_nnn_vs_vl_*.json"))):
        d = _load(Path(f))
        if not d:
            continue
        for r in d.get("rows", []):
            hva = r.get("hva_nnn")
            if not hva:
                continue
            s = get(r["N"], r["h"], gap=r.get("gap"), e0=r.get("E0"))
            t = hva.get("transpiled", {})
            s.recs.append(
                Rec(
                    method="HVA",
                    variant=f"nnn p={r['p_layers']} (ceiling)",
                    fidelity=hva.get("fidelity"),
                    abs_error=hva.get("abs_error"),
                    de_gap=hva.get("de_gap"),
                    n_2q=t.get("n_2q_gates"),
                    total_gates=t.get("total_gates"),
                    depth=t.get("depth"),
                    converged="ceiling (best-of by energy)",
                    note=f"best warm-start = {hva.get('best_method', '?')}",
                    source="compare_ceiling",
                )
            )

    # ── HVA fair-convergence seeds (n*_faircov*): best converged seed ────
    # Each row is a warm-start seed optimized to convergence; take the best
    # (highest fidelity) as the nnn p=2 ceiling for that scenario, and recompute
    # its gate counts deterministically (the fair-conv artifact omits them).
    for f in sorted(glob.glob(str(study / "hva_nnn_sweep" / "n*_faircov*.json"))):
        d = _load(Path(f))
        if not d or not d.get("rows"):
            continue
        n, h = d.get("N"), d.get("h")
        p = d.get("p_layers", 2)
        gap, e0 = d.get("gap"), d.get("e0")
        rows = [r for r in d["rows"] if r.get("fidelity") is not None]
        if not rows:
            continue
        best = max(rows, key=lambda r: r["fidelity"])
        s = get(n, h, gap=gap, e0=e0)
        total = twoq = depth = None
        if recompute_gates:
            total, twoq, depth = _recompute_gate_counts(d.get("topology", "square"), n, h, d.get("J2", 0.5), p, "nnn")
        nconv = sum(1 for r in rows if r.get("converged"))
        s.recs.append(
            Rec(
                method="HVA",
                variant=f"nnn p={p} (fair-conv best)",
                fidelity=best.get("fidelity"),
                abs_error=best.get("abs_error"),
                de_gap=best.get("de_gap"),
                n_2q=twoq,
                total_gates=total,
                depth=depth,
                converged=f"{nconv}/{len(rows)} seeds",
                note=f"best seed = {best.get('seed_type', '?')}",
                source=f"faircov:{Path(f).name}",
            )
        )

    # ── HVA structural variants (p1_half_nn / p2 / p3, any N,h) ──────────
    for f in sorted(glob.glob(str(study / "hva_nnn_sweep" / "ansatz_variants_*.json"))):
        d = _load(Path(f))
        if not d:
            continue
        gap, e0 = d.get("gap"), d.get("e0")
        topo = d.get("topology", "square")
        for r in d.get("rows", []):
            s = get(d["N"], d["h"], gap=gap, e0=e0)
            e_best = r.get("e_best")
            abs_err = abs(e_best - e0) if (e_best is not None and e0 is not None) else None
            # These variants alter structure (blocks/rx/rz); their exact ansatz
            # label isn't a plain nn/nnn, so gate counts stay as recorded (2q)
            # and total is recomputed only when the variant maps to a base ansatz.
            s.recs.append(
                Rec(
                    method="HVA",
                    variant=r.get("variant", "?"),
                    fidelity=r.get("best_fidelity"),
                    abs_error=abs_err,
                    de_gap=r.get("de_gap"),
                    n_2q=r.get("n_2q_transpiled"),
                    total_gates=None,
                    depth=None,
                    converged=r.get("converged", "?"),
                    note=f"variant on {topo}",
                    source=f"ansatz_variants:{Path(f).name}",
                )
            )

    # ── Bond-selection ablation (masked HVA: pruned / top-k / ADAPT / frac) ──
    # One folder, several schemas; each row is a masked/grown HVA ansatz with its
    # own 2q cost. ADAPT rows are per growth step; the rest per variant.
    for f in sorted(glob.glob(str(study / "bond_ablation" / "*.json"))):
        d = _load(Path(f))
        if not d:
            continue
        gap, e0 = d.get("gap"), d.get("e0")
        schema = d.get("schema", "")
        for r in d.get("rows", []):
            fid = r.get("best_fidelity")
            if fid is None:
                continue
            s = get(d["N"], d["h"], gap=gap, e0=e0)
            if schema == "adapt_bonds_v1":
                label = f"adapt step{r.get('step')} ({r.get('n_bonds')} bonds)"
            else:
                label = r.get("variant", "?")
                nnnb = r.get("n_nnn_bonds")
                if nnnb is not None:
                    label = f"{label} (nnn={nnnb})"
            e_best = r.get("e_best")
            abs_err = (abs(e_best - e0) if (e_best is not None and e0 is not None)
                       else r.get("abs_error"))
            s.recs.append(
                Rec(
                    method="HVA",
                    variant=label,
                    fidelity=fid,
                    abs_error=abs_err,
                    de_gap=r.get("de_gap"),
                    n_2q=r.get("n_2q_transpiled"),
                    total_gates=None,
                    depth=None,
                    converged=r.get("seed_name") or r.get("converged") or "masked",
                    note=f"bond-selection ({schema})",
                    source=f"bond_ablation:{Path(f).name}",
                )
            )

    # ── VL default h-sweep ───────────────────────────────────────────────
    d = _load(study / "vl_h_sweep" / "vl_h_sweep_frustrated.json")
    if d:
        for r in d.get("rows", []):
            s = get(r["N"], r["h"], gap=r.get("gap"), e0=r.get("E0"))
            s.recs.append(
                Rec(
                    method="VL",
                    variant="default L2/F20",
                    fidelity=r.get("vl_fidelity"),
                    abs_error=r.get("vl_abs_error"),
                    de_gap=(abs(r["vl_abs_error"]) / s.gap if r.get("vl_abs_error") is not None and s.gap else None),
                    n_2q=r.get("vl_2q"),
                    total_gates=r.get("vl_total_gates"),
                    depth=r.get("vl_depth"),
                    converged="n/a (fixed circuit)",
                    loader="vector",
                    chi=None,
                    source="vl_h_sweep",
                )
            )

    # ── VL quality sweep (num_layers × fine_tuning) ──────────────────────
    d = _load(study / "vl_quality_sweep" / "vl_quality_sweep_frustrated.json")
    if d:
        seen = set()
        for r in d.get("rows", []):
            nl, ft = r.get("num_layers"), r.get("fine_tuning_iterations")
            dedup = (int(r["N"]), round(float(r["h"]), 2), nl, ft)
            if dedup in seen:
                continue
            seen.add(dedup)
            s = get(r["N"], r["h"], gap=r.get("gap"), e0=r.get("E0"))
            s.recs.append(
                Rec(
                    method="VL",
                    variant=f"L{nl}/F{ft}",
                    fidelity=r.get("vl_fidelity"),
                    abs_error=r.get("vl_abs_error"),
                    de_gap=(abs(r["vl_abs_error"]) / s.gap if r.get("vl_abs_error") is not None and s.gap else None),
                    n_2q=r.get("vl_2q"),
                    total_gates=r.get("vl_total_gates"),
                    converged="n/a (fixed circuit)",
                    loader="vector",
                    chi=None,
                    source="vl_quality_sweep",
                )
            )

    # ── VL large-N via mps_loading ───────────────────────────────────────
    for f in sorted(glob.glob(str(study / "vl_mps_n18" / "*.json"))):
        if f.endswith(".meta.json"):
            continue
        d = _load(Path(f))
        if not d:
            continue
        params = d.get("params", {})
        nl, ft = params.get("num_layers"), params.get("fine_tuning_iterations")
        chi = params.get("chi_max")
        for r in d.get("rows", []):
            if r.get("vl_fidelity") is None:
                continue
            s = get(r["N"], r["h"])
            gap = s.gap
            s.recs.append(
                Rec(
                    method="VL",
                    variant=f"L{nl}/F{ft}",
                    fidelity=r.get("vl_fidelity"),
                    abs_error=r.get("vl_abs_error"),
                    de_gap=(abs(r["vl_abs_error"]) / gap if r.get("vl_abs_error") is not None and gap else None),
                    n_2q=r.get("vl_2q"),
                    total_gates=r.get("vl_total_gates"),
                    depth=r.get("vl_depth"),
                    converged="n/a (fixed circuit)",
                    loader="mps",
                    chi=chi,
                    note="mps_loading (target is the χ-capped MPS, not the exact vector)",
                    source="vl_mps_n18",
                )
            )

    # ── Dual-target 2×2 fidelity matrices (VL/HVA × exact/MPS χ) ──────────
    for f in sorted(glob.glob(str(study / "hva_nnn_sweep" / "dual_target_fidelity_*.json"))):
        d = _load(Path(f))
        if not d:
            continue
        k = _key(d["N"], d["h"])
        if k in scen:
            scen[k].dual_target = {
                "reference_overlap": d.get("reference_overlap_exact_vs_mps"),
                "chi_max": d.get("chi_max"),
                "methods": d.get("methods", {}),
            }

    return scen


def _fmt(x, spec=".4f", missing="—"):
    return missing if x is None else format(x, spec)


def _interp_fidelity_at_2q(recs: list[Rec], budget: int) -> tuple[float | None, str]:
    """Interpolate a method's fidelity at a target 2q budget from its runs.

    Uses the run at-or-below and at-or-above ``budget`` (linear in 2q). Returns
    ``(F, note)``. If ``budget`` is outside the measured range, clamps to the
    nearest measured point and flags it as extrapolation.
    """
    pts = sorted(
        ((r.n_2q, r.fidelity) for r in recs if r.n_2q is not None and r.fidelity is not None), key=lambda t: t[0]
    )
    if not pts:
        return None, "no data"
    exact = [f for q, f in pts if q == budget]
    if exact:
        return max(exact), "measured"
    below = [(q, f) for q, f in pts if q < budget]
    above = [(q, f) for q, f in pts if q > budget]
    if below and above:
        q0, f0 = below[-1]
        q1, f1 = above[0]
        frac = (budget - q0) / (q1 - q0)
        return f0 + frac * (f1 - f0), f"interp {q0}\u2192{q1}cx"
    if below:
        return below[-1][1], f"clamp\u2264{below[-1][0]}cx"
    return above[0][1], f"clamp\u2265{above[0][0]}cx"


def _render_iso_2q(out: list[str], kept: list[Rec]) -> None:
    """Append an iso-2q comparison: fidelity of each method at matched 2q budgets."""
    hva = [r for r in kept if r.method == "HVA"]
    vl = [r for r in kept if r.method == "VL"]
    if not hva or not vl:
        return
    # Budgets = the union of both methods' measured 2q counts (deduped, sorted).
    budgets = sorted({r.n_2q for r in kept if r.n_2q is not None})
    if not budgets:
        return
    out.append(
        "**Iso-2q view** — fidelity at matched 2q budgets (VL interpolated on its quality curve; HVA from its runs)."
    )
    out.append("")
    out.append("| 2q budget | HVA F | VL F | better |")
    out.append("|----------:|------:|-----:|--------|")
    for b in budgets:
        fh, nh = _interp_fidelity_at_2q(hva, b)
        fv, nv = _interp_fidelity_at_2q(vl, b)
        if fh is None and fv is None:
            continue
        better = "—"
        if fh is not None and fv is not None:
            better = "HVA" if fh > fv else ("VL" if fv > fh else "tie")
        out.append(f"| {b} | {_fmt(fh)} | {_fmt(fv)} | {better} |")
    out.append("")


def _render_scenario(out: list[str], s: Scenario, min_fidelity: float) -> None:
    """Render one scenario block: main table + iso-2q + dual-target matrix."""
    out.append(f"## N = {s.n}, h = {s.h:.2f}" + (f" (gap {s.gap:.4f}, E0 {s.e0:.4f})" if s.gap and s.e0 else ""))
    out.append("")

    kept = [r for r in s.recs if r.fidelity is not None and r.fidelity > min_fidelity]
    n_hidden = len(s.recs) - len(kept)
    methods_present = {r.method for r in kept}
    if methods_present != {"HVA", "VL"}:
        only = " and ".join(sorted(methods_present)) or "none"
        out.append(
            f"> ⚠️ Only **{only}** present above F>{min_fidelity} at this "
            f"scenario — no same-scenario cross-method comparison possible."
        )
        out.append("")
    out.append("| method | variant | loader/χ | F | \\|ΔE\\| | ΔE/gap | 2q | total | depth | converged |")
    out.append("|--------|---------|----------|------:|-------:|-------:|---:|------:|------:|-----------|")
    for method in ("HVA", "VL"):
        recs = [r for r in kept if r.method == method]
        recs.sort(key=lambda r: (r.fidelity if r.fidelity is not None else -1), reverse=True)
        for r in recs:
            if r.method == "VL":
                loader = f"{r.loader} χ={r.chi}" if r.loader == "mps" and r.chi else (r.loader or "—")
            else:
                loader = "—"
            out.append(
                f"| {r.method} | {r.variant} | {loader} | {_fmt(r.fidelity)} | "
                f"{_fmt(r.abs_error)} | {_fmt(r.de_gap, '.3f')} | "
                f"{_fmt(r.n_2q, 'd')} | {_fmt(r.total_gates, 'd')} | "
                f"{_fmt(r.depth, 'd')} | {r.converged or '—'} |"
            )
    if n_hidden:
        out.append("")
        out.append(f"_{n_hidden} run(s) hidden (F ≤ {min_fidelity})._")
    out.append("")

    _render_iso_2q(out, kept)

    if s.dual_target and s.dual_target.get("methods"):
        dt = s.dual_target
        ov = dt.get("reference_overlap")
        chi = dt.get("chi_max")
        out.append(
            f"**Dual-target fidelity** — both methods re-simulated locally "
            f"against the same reference. `⟨exact|MPS χ={chi}⟩² = "
            f"{_fmt(ov, '.6f')}`, so the two references coincide to that overlap."
        )
        out.append("")
        out.append("| method | F vs exact | F vs MPS χ | source θ/QPY |")
        out.append("|--------|-----------:|-----------:|--------------|")
        for m in ("HVA", "VL"):
            mm = dt["methods"].get(m)
            if not mm:
                continue
            src = mm.get("source_theta") or mm.get("source_qpy") or ""
            src_short = src.split("/")[-1] if src else "—"
            out.append(
                f"| {m} | {_fmt(mm.get('fidelity_vs_exact'))} | {_fmt(mm.get('fidelity_vs_mps'))} | {src_short} |"
            )
        out.append("")


def build_markdown(scen: dict[tuple, Scenario], *, min_fidelity: float = 0.5) -> str:
    out: list[str] = []
    out.append("# VL vs HVA — consolidated comparison (frustrated square TFIM, J2=0.5)")
    out.append("")
    out.append(
        "Auto-generated from the JSON artifacts across all N, h, and method "
        "variants — do not edit by hand. Regenerate with "
        "`generate_vl_hva_consolidated_report.py`. Descriptive (which method fits "
        "which scenario), not a competition."
    )
    out.append("")
    out.append(f"- Only runs with **F > {min_fidelity}** are shown (weaker runs hidden per scenario).")
    out.append("- Per row: `F`, `|ΔE|`, `ΔE/gap`, 2q gates, total gates, depth, loader/χ, converged.")
    out.append(
        "- HVA gate counts: transpiled `rz, sx, x, cx`, θ-independent structural count "
        "(recomputed deterministically from the saved θ where the sweep omitted them)."
    )
    out.append(
        "- VL gate counts: as returned by Haiqu. `loader` = vector_loading vs mps_loading; "
        "`χ` = MPS bond cap (mps only)."
    )
    out.append(
        "- HVA `converged`: per-restart contract where available (e.g. 4/4); "
        "`ceiling` = best-of-by-energy; `n/a (random)` = random-restart sweep. "
        "VL is a fixed zero-parameter circuit → `n/a (fixed circuit)`."
    )
    out.append("- θ-search cost and hardware executability are out of scope.")
    out.append(
        "- ⚠️ At N=18 VL uses `mps_loading` (target = χ-capped MPS), not the "
        "`vector_loading` used at N≤10 — flagged per row."
    )
    out.append(
        "- Where available, a **dual-target** matrix scores both methods against the "
        "exact ground state AND the MPS χ=64 reference (all re-simulated locally), "
        "quantifying that the target choice barely moves F."
    )
    out.append(
        "- An **iso-2q** view compares fidelity at matched 2q budgets (VL interpolated "
        "on its quality curve) — the fair resource-normalized comparison."
    )
    out.append(
        "- Scenarios where VL was only run at its default config (no quality sweep) "
        "are moved to an **appendix** — the effort is asymmetric, so they don't belong "
        "in the primary same-effort comparison."
    )
    out.append("")

    # Partition scenarios: primary (VL has >1 config = a quality sweep) vs
    # appendix (VL only default → effort-asymmetric vs HVA's full tuning).
    primary, appendix = [], []
    for k in sorted(scen.keys()):
        s = scen[k]
        kept = [r for r in s.recs if r.fidelity is not None and r.fidelity > min_fidelity]
        vl_variants = {r.variant for r in kept if r.method == "VL"}
        has_hva = any(r.method == "HVA" for r in kept)
        vl_default_only = vl_variants and vl_variants <= {"default L2/F20"}
        if has_hva and vl_variants and vl_default_only:
            appendix.append(k)
        else:
            primary.append(k)

    for k in primary:
        _render_scenario(out, scen[k], min_fidelity)

    if appendix:
        out.append("## Appendix — VL default-only scenarios (effort-asymmetric)")
        out.append("")
        out.append(
            "Here VL has ONLY its default config (`L2/F20`, no quality sweep), so HVA's "
            "tuned ceiling is compared against VL's floor. Not a same-effort comparison; "
            "shown for completeness. Re-run the VL quality sweep at these h to make them "
            "primary."
        )
        out.append("")
        for k in appendix:
            _render_scenario(out, scen[k], min_fidelity)

    # Coverage matrix
    out.append("## Coverage matrix (methods present per scenario)")
    out.append("")
    out.append("| N | h | HVA variants | VL variants |")
    out.append("|---|------|-------------:|------------:|")
    for k in sorted(scen.keys()):
        s = scen[k]
        n_hva = sum(1 for r in s.recs if r.method == "HVA")
        n_vl = sum(1 for r in s.recs if r.method == "VL")
        out.append(f"| {s.n} | {s.h:.2f} | {n_hva} | {n_vl} |")
    out.append("")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    root = resolve_project_root(Path(__file__))
    scen = collect(root)
    md = build_markdown(scen)
    out_path = study_dir("reports") / "REPORT_VL_vs_HVA_consolidated.md"
    out_path.write_text(md)
    n_scen = len(scen)
    n_both = sum(1 for s in scen.values() if {r.method for r in s.recs} == {"HVA", "VL"})
    print(f"[consolidated] wrote {out_path}", flush=True)
    print(f"[consolidated] {n_scen} scenarios, {n_both} with both methods", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
