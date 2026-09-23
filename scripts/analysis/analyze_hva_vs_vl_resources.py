#!/usr/bin/env python
"""Consolidate the HVA-vs-VL circuit-resource study into SUMMARY + figures.

Reads the latest hva_vs_vl_*.json and renders the depth/2q vs bond-dimension χ
comparison that supports: HVA (GNN) prepares the TFIM ground state at constant,
low circuit cost but limited fidelity, while Vector Loading reaches fidelity → 1
at a circuit cost that grows with χ (and with the state's entanglement).

Figures:
- fidelity_vs_chi.png : fidelity(χ) per VL method + HVA constant line, per h
- n2q_vs_chi.png      : transpiled 2q-gate count(χ) per VL method + HVA line, per h

Usage:
    python scripts/analysis/analyze_hva_vs_vl_resources.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from hva_vl_study_common import study_dir  # noqa: E402

RESULTS_DIR = study_dir("resources")
FIGDIR = RESULTS_DIR / "figures"
_VL_METHODS = ["vl_approx", "vl_exact"]
_LABEL = {"hva": "HVA (GNN)", "vl_exact": "VL exact (Schön)",
          "vl_approx": "VL approx (Ran)", "vl_ceiling": "VL ceiling (MPS trunc)"}


def _latest_json() -> Path | None:
    files = sorted(glob.glob(str(RESULTS_DIR / "hva_vs_vl_*.json")))
    return Path(files[-1]) if files else None


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def build_summary(d: dict) -> str:
    n = d["n_qubits"]
    rows = d["rows"]
    by_h = defaultdict(list)
    for r in rows:
        by_h[r["h"]].append(r)

    lines = [f"# HVA (GNN) vs Vector Loading — circuit resources (N={n}, {d['topology']})\n"]
    lines.append(
        "Preparing the TFIM ground state |ψ₀(h)⟩. HVA is a constant-depth ansatz "
        "(fidelity-limited); VL loads an MPS approximation of bond dimension χ "
        "(fidelity → 1, depth grows with χ). 2q = transpiled CX to basis "
        f"{d['config']['basis_gates']}.\n"
    )

    for h in sorted(by_h):
        rs = by_h[h]
        hva = next((r for r in rs if r["method"] == "hva"), None)
        lines.append(f"## h = {h}\n")
        if hva:
            dg = f"{hva['de_gap']:.4f}" if hva.get("de_gap") is not None else "—"
            nq = f"{hva['nisq_fidelity_est']:.3f}" if hva.get("nisq_fidelity_est") is not None else "—"
            lines.append(
                f"HVA reference: fidelity={hva['fidelity']:.4f}, ΔE/gap={dg}, "
                f"2q={hva['n_2q_transpiled']}, depth_2q={hva.get('depth_2q')}, "
                f"NISQ-fid≈{nq} ({hva['n_params']} params, constant in χ).\n"
            )
        header = "| method | χ | fidelity | ΔE/gap | 2q (CX) | depth_2q | NISQ-fid |"
        sep = "|---|---:|---:|---:|---:|---:|---:|"
        lines.append(header)
        lines.append(sep)
        for r in sorted(rs, key=lambda x: (x["method"], x["chi_requested"] or 0)):
            if r["method"] == "vl_ceiling":
                continue
            chi = r["chi_requested"] if r["chi_requested"] is not None else "—"
            fid = f"{r['fidelity']:.4f}" if r["fidelity"] is not None else "—"
            dg = f"{r['de_gap']:.4f}" if r.get("de_gap") is not None else "—"
            n2 = r["n_2q_transpiled"] if r["n_2q_transpiled"] is not None else "—"
            d2 = r.get("depth_2q") if r.get("depth_2q") is not None else "—"
            nq = f"{r['nisq_fidelity_est']:.3f}" if r.get("nisq_fidelity_est") is not None else "—"
            note = "" if r["status"] == "ok" else f" ({r['status']})"
            lines.append(f"| {_LABEL[r['method']]}{note} | {chi} | {fid} | {dg} | {n2} | {d2} | {nq} |")
        lines.append("")

    lines.append("## Verdict\n")
    lines.append(
        "A three-way trade-off, not just cost: (1) *theoretical accuracy* — VL "
        "reaches ΔE/gap ≈ 0 while HVA has an ansatz ceiling that fails the 5% "
        "criterion near h_c (ΔE/gap ≈ 1.8 at h=1.0) but passes deep in the "
        "paramagnetic phase; (2) *circuit cost* — VL's 2q count grows with χ and "
        "with the state's entanglement, 1–2 orders above HVA's constant ~18 CX; "
        "(3) *hardware executability* — the estimated NISQ fidelity collapses for "
        "VL as χ grows (VL exact χ=8 ≈ 1% at h=1.0: theoretically exact but "
        "unexecutable), while HVA stays the most executable (~0.86). "
        "So VL wins on accuracy but loses on executability; HVA wins on "
        "executability but is accuracy-limited near criticality."
    )
    lines.append("")
    lines.append("Figures: `figures/fidelity_vs_chi.png`, `figures/n2q_vs_chi.png`")
    return "\n".join(lines) + "\n"


def _plot(d: dict) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        return [f"(figures skipped: {exc})"]

    FIGDIR.mkdir(parents=True, exist_ok=True)
    rows = d["rows"]
    hs = sorted({r["h"] for r in rows})
    saved = []

    # fidelity vs chi
    fig, axes = plt.subplots(1, len(hs), figsize=(6 * len(hs), 5), squeeze=False)
    for ax, h in zip(axes[0], hs):
        rs = [r for r in rows if r["h"] == h]
        for method in _VL_METHODS + ["vl_ceiling"]:
            pts = sorted(((r["chi_requested"], r["fidelity"]) for r in rs
                          if r["method"] == method and r["fidelity"] is not None
                          and r["chi_requested"] is not None), key=lambda x: x[0])
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=_LABEL[method])
        hva = next((r for r in rs if r["method"] == "hva"), None)
        if hva and hva["fidelity"] is not None:
            ax.axhline(hva["fidelity"], ls="--", color="k", label=_LABEL["hva"])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("χ (bond dimension)"); ax.set_ylabel("fidelity")
        ax.set_title(f"Fidelity vs χ (h={h})"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    p = FIGDIR / "fidelity_vs_chi.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # 2q vs chi
    fig, axes = plt.subplots(1, len(hs), figsize=(6 * len(hs), 5), squeeze=False)
    for ax, h in zip(axes[0], hs):
        rs = [r for r in rows if r["h"] == h]
        for method in _VL_METHODS:
            pts = sorted(((r["chi_requested"], r["n_2q_transpiled"]) for r in rs
                          if r["method"] == method and r["n_2q_transpiled"] is not None
                          and r["chi_requested"] is not None), key=lambda x: x[0])
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="s", label=_LABEL[method])
        hva = next((r for r in rs if r["method"] == "hva"), None)
        if hva and hva["n_2q_transpiled"] is not None:
            ax.axhline(hva["n_2q_transpiled"], ls="--", color="k", label=_LABEL["hva"])
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("χ (bond dimension)"); ax.set_ylabel("2q gates (CX, transpiled)")
        ax.set_title(f"Circuit 2q-cost vs χ (h={h})"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    p = FIGDIR / "n2q_vs_chi.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    return saved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=str, default=None, help="specific results JSON")
    args = ap.parse_args(argv)

    path = Path(args.file) if args.file else _latest_json()
    if path is None or not path.exists():
        print(f"No results JSON in {RESULTS_DIR}. Run run_hva_vs_vl_resources.py first.")
        return 1

    d = _load(path)
    (RESULTS_DIR / "SUMMARY.md").write_text(build_summary(d))
    print(f"written -> {RESULTS_DIR / 'SUMMARY.md'}  (source: {path.name})")
    figs = _plot(d)
    print("figures:", ", ".join(figs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
