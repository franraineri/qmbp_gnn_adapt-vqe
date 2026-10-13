#!/usr/bin/env python
"""Summarize prerank_micro telemetry (and audit) from an adapt_poolA JSON.

Reads the crash-safe ``adapt_poolA_<topo>_N<n>.json`` and reports, per h-point
and per step, the pre-rank candidate pool, the micro-descent winner, and — when
the run used ``--prerank-micro-audit`` — whether the cheap micro winner matched
the full-converge winner. The audit match-rate is the direct proof of whether the
micro-descent proxy is faithful before trusting it on expensive points.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/analyze_prerank_audit.py \
        results/hva_vl_study/bond_ablation/adapt_poolA_square_N14.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _rows(doc):
    return doc.get("rows", doc) if isinstance(doc, dict) else doc


def summarize(path: Path) -> int:
    doc = json.loads(path.read_text())
    rows = _rows(doc)
    file_status = doc.get("status") if isinstance(doc, dict) else "n/a"
    print(f"file: {path.name}  status={file_status}  points={len(rows)}")

    total_match = total_audit = 0
    for r in rows:
        steps = r.get("pool_steps") or r.get("pool_curve") or []
        pre = [s for s in steps if s.get("prerank")]
        audits = [s["prerank"]["audit"] for s in pre if "audit" in s["prerank"]]
        matches = sum(1 for a in audits if a.get("micro_matches_full"))
        total_match += matches
        total_audit += len(audits)
        best = max((s["best_fidelity"] for s in steps), default=float("nan"))
        print(f"\n  h={r.get('h')} status={r.get('status')} "
              f"steps={len(steps)} prerank={len(pre)} best_fid={best:.4f}")
        if r.get("pareto_fid") is not None:
            print(f"    pareto fid/CX: fid={r['pareto_fid']:.4f} "
                  f"2q={r.get('pareto_2q')} fpc={r.get('pareto_fid_per_cx'):.3e} "
                  f"(step{r.get('pareto_step')})")
        for s in steps:
            line = (f"    step{s.get('step')} {s.get('action_kind'):12s} "
                    f"fid={s.get('best_fidelity'):.4f} 2q={s.get('n_2q_transpiled')} "
                    f"L={s.get('n_layers')} t={s.get('seconds')}s")
            p = s.get("prerank")
            if p:
                cand = ",".join(f"{c['kind']}:{c['micro_score']:+.2e}"
                                for c in p.get("candidates", []))
                line += f"\n        micro[{cand}] chosen={p.get('chosen_kind')}"
                a = p.get("audit")
                if a:
                    line += (f" | full_winner={a.get('full_winner_kind')} "
                             f"MATCH={a.get('micro_matches_full')}")
            print(line)

    if total_audit:
        rate = total_match / total_audit
        verdict = ("PROXY FAITHFUL" if rate >= 0.8 else
                   "PROXY UNRELIABLE — micro mis-ranks vs full")
        print(f"\n=== AUDIT VERDICT === micro==full {total_match}/{total_audit} "
              f"({rate:.0%}) → {verdict}")
    else:
        print("\n(no audit data — run with --prerank-micro-audit to validate)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Summarize prerank_micro audit telemetry")
    ap.add_argument("json_path", type=Path)
    args = ap.parse_args(argv)
    if not args.json_path.exists():
        print(f"not found: {args.json_path}", file=sys.stderr)
        return 1
    return summarize(args.json_path)


if __name__ == "__main__":
    raise SystemExit(main())
