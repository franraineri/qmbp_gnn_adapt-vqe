#!/usr/bin/env python
"""Generate the VL-vs-HVA comparative markdown FROM the JSON artifacts.

This report is never hand-authored: it is regenerated from
``compare_hva_nnn_vs_vl_*.json`` (produced by ``compare_hva_nnn_vs_vl.py``) so the
numbers cannot drift from the data. The point is descriptive — which method
serves which scenario, under an IDENTICAL (N, p, h, J2, topology) setting — not a
competition.

Validations (fail-fast to prevent the recurring documentation bugs):
- Every emitted row must have HVA and VL under the SAME scenario key; a mismatch
  aborts (prevents comparing different N / p / h).
- The HVA ansatz label is surfaced (prevents nn↔nnn confusion).
- Missing VL data is rendered as an explicit ``— (no VL data)`` marker, never
  silently dropped or back-filled from a different scenario.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/generate_vl_hva_comparison_report.py \
        --n 9 --p 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import study_dir  # noqa: E402

from qmbp_simulation.framework.result_io import (  # noqa: E402
    ScenarioMismatch,
    assert_same_scenario,
)

__all__ = ["ScenarioMismatch", "assert_same_scenario", "build_markdown"]


def _load_compare(n: int, p: int, topology: str) -> dict:
    path = study_dir("hva_nnn_sweep") / f"compare_hva_nnn_vs_vl_{topology}_N{n}_p{p}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"compare artifact not found: {path}\n"
            f"Run compare_hva_nnn_vs_vl.py --n {n} --p {p} --topology {topology} first."
        )
    return json.loads(path.read_text())


def _fmt(x, spec=".4f", missing="—"):
    return missing if x is None else format(x, spec)


def _row_table(rows: list[dict]) -> list[str]:
    """Per-h same-scenario comparison table (fidelity + full gate breakdown)."""
    lines = [
        "| h | method | ansatz | F | ΔE/gap | total gates | 2q | 1q | depth |",
        "|------|--------|--------|------:|-------:|------------:|---:|---:|------:|",
    ]
    for row in rows:
        assert_same_scenario(row)
        h = row["h"]
        hva = row["hva_nnn"]
        ht = hva["transpiled"]
        lines.append(
            f"| {h:.2f} | HVA | {hva['ansatz']} | {_fmt(hva['fidelity'])} | "
            f"{_fmt(hva['de_gap'])} | {ht['total_gates']} | {ht['n_2q_gates']} | "
            f"{ht['n_1q_gates']} | {ht['depth']} |"
        )
        vl = row.get("vl")
        if vl is None:
            lines.append(f"| {h:.2f} | VL | — (no VL data) | — | — | — | — | — | — |")
        else:
            lines.append(
                f"| {h:.2f} | VL | {vl.get('settings', 'Haiqu')} | "
                f"{_fmt(vl['fidelity'])} | {_fmt(vl['de_gap'])} | "
                f"{_fmt(vl['total_gates'], 'd')} | {_fmt(vl['n_2q_gates'], 'd')} | "
                f"{_fmt(vl['n_1q_gates'], 'd')} | {_fmt(vl['depth'], 'd')} |"
            )
    return lines


def build_markdown(compare: dict) -> str:
    rows = compare["rows"]
    topo = compare["topology"]
    n = compare["N"]
    p = compare["p_layers"]
    j2 = compare["J2"]

    n_missing_vl = sum(1 for r in rows if r.get("vl") is None)

    out: list[str] = []
    out.append(f"# VL vs HVA — same-scenario comparison ({topo}, N={n}, p={p}, J2={j2})")
    out.append("")
    out.append(
        "Auto-generated from the JSON artifacts — do not edit by hand. Regenerate "
        "with `generate_vl_hva_comparison_report.py`. This is a descriptive "
        "side-by-side (which method fits which scenario), not a competition."
    )
    out.append("")
    out.append(f"- Model: `tfim_frustrated`, `J2 = {j2}`, `J = 1.0`")
    out.append(f"- Lattice: `{topo}`, `N = {n}`, HVA depth `p = {p}`")
    out.append("- HVA fidelity is the best-of-by-energy VQE ceiling (θ captured in the artifact).")
    out.append("- VL numbers are Haiqu-measured (default settings unless noted).")
    out.append("- 2q/1q/total/depth counted on the transpiled circuit (`rz, sx, x, cx`).")
    out.append("- θ-search cost and hardware executability are intentionally out of scope.")
    out.append("")
    out.append(f"- Source artifact: `{compare.get('result_path', 'compare_hva_nnn_vs_vl_*.json')}`")
    if n_missing_vl:
        out.append(f"- ⚠️ {n_missing_vl}/{len(rows)} scenario(s) have no matching VL row (shown as `— (no VL data)`).")
    out.append("")
    out.append("## 1. Same-scenario table")
    out.append("")
    out.extend(_row_table(rows))
    out.append("")
    out.append("## 2. VL quality curve per scenario (num_layers × fine_tuning)")
    out.append("")
    out.append(
        "VL fidelity/2q as its two knobs vary (from the quality sweep). Shows the "
        f"range VL spans at each scenario, next to the single HVA nnn p={p} point."
    )
    out.append("")
    any_quality = False
    for row in rows:
        configs = row.get("vl_quality_configs") or []
        if not configs:
            continue
        any_quality = True
        h = row["h"]
        hva = row["hva_nnn"]
        out.append(f"### h = {h:.2f} (gap {row['gap']:.4f})")
        out.append("")
        out.append("| method | config | F | 2q |")
        out.append("|--------|--------|------:|---:|")
        out.append(f"| HVA nnn | p={p} | {_fmt(hva['fidelity'])} | {hva['transpiled']['n_2q_gates']} |")
        for c in configs:
            out.append(
                f"| VL | L{c['num_layers']}/F{c['fine_tuning_iterations']} | "
                f"{_fmt(c['fidelity'])} | {_fmt(c['n_2q_gates'], 'd')} |"
            )
        out.append("")
    if not any_quality:
        out.append("_No VL quality-sweep data available at these scenarios._")
        out.append("")

    out.append("## 3. Per-scenario read (descriptive)")
    out.append("")
    for row in rows:
        h = row["h"]
        hva = row["hva_nnn"]
        vl = row.get("vl")
        gap = row["gap"]
        out.append(f"### h = {h:.2f} (gap {gap:.4f})")
        out.append("")
        out.append(
            f"- HVA {hva['ansatz']}: F = {_fmt(hva['fidelity'])}, "
            f"{hva['transpiled']['n_2q_gates']} CX, "
            f"{hva['transpiled']['total_gates']} total gates, "
            f"best warm-start = {hva.get('best_method', 'n/a')}."
        )
        if vl is None:
            out.append("- VL: no data at this scenario.")
        else:
            out.append(
                f"- VL ({vl.get('settings', 'Haiqu')}): F = {_fmt(vl['fidelity'])}, "
                f"{_fmt(vl['n_2q_gates'], 'd')} CX, "
                f"{_fmt(vl['total_gates'], 'd')} total gates."
            )
            best = max(
                (c for c in (row.get("vl_quality_configs") or []) if c["fidelity"] is not None),
                key=lambda c: c["fidelity"],
                default=None,
            )
            if best is not None:
                out.append(
                    f"- VL best (L{best['num_layers']}/F{best['fine_tuning_iterations']}): "
                    f"F = {_fmt(best['fidelity'])}, {_fmt(best['n_2q_gates'], 'd')} CX."
                )
        out.append("")
    out.append("## 4. Metric definitions")
    out.append("")
    out.append("- `F` — exact statevector fidelity `|⟨ψ_exact|ψ_prep⟩|²` (measured).")
    out.append("- `ΔE/gap` — energy error of the prepared state normalized by the spectral gap.")
    out.append("- `total/2q/1q/depth` — counted on the transpiled circuit; HVA structural")
    out.append("  count uses fixed non-zero angles (θ-independent), VL as returned by Haiqu.")
    out.append("- `ansatz` — HVA generator set: nnn maps the J2 diagonals; nn is NN-only.")
    out.append("")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate VL-vs-HVA comparison markdown from JSON")
    ap.add_argument("--topology", default="square")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--out", default=None, help="Output markdown path (default: reports/).")
    args = ap.parse_args(argv)

    compare = _load_compare(args.n, args.p, args.topology)
    md = build_markdown(compare)

    out_path = (
        Path(args.out)
        if args.out
        else study_dir("reports") / f"REPORT_VL_vs_HVA_{args.topology}_N{args.n}_p{args.p}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"[report] wrote {out_path} ({len(compare['rows'])} scenarios)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
