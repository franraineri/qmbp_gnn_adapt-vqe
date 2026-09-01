#!/usr/bin/env python
"""Show the best HVA angle-predictor models in the critical h-window.

Reads the empirical critical-window ranking stored on each ZooEntry
(``critical_ranking``, populated by ``backfill_critical_ranking_from_evals``)
and prints, per model, the aggregated |ΔE|/fidelity over the window and the
per-N metrics at h≈1.0 (the TFIM critical point).

Usage
-----
  scripts/maintenance/show_critical_models.py
  scripts/maintenance/show_critical_models.py --p 1 --n 10 20 30 --h 1.0
  scripts/maintenance/show_critical_models.py --refresh    # repopulate first
"""

from __future__ import annotations

import argparse

from qmbp_simulation.predictors.model_zoo import (
    _critical_window_key,
    _load_manifest,
    get_critical_metrics_at_h,
)


def _fmt(x, prec=3):
    return f"{x:.{prec}f}" if isinstance(x, int | float) else "N/A"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p", type=int, default=1, help="p_layers (default 1)")
    ap.add_argument("--n", type=int, nargs="+", default=[10, 20, 30],
                    help="N values for the per-N h=1.0 table (default 10 20 30)")
    ap.add_argument("--h", type=float, default=1.0, help="target h (default 1.0)")
    ap.add_argument("--refresh", action="store_true",
                    help="repopulate the ranking from eval reports before showing")
    args = ap.parse_args()

    if args.refresh:
        from qmbp_simulation.predictors.model_zoo import backfill_critical_ranking_from_evals

        n = backfill_critical_ranking_from_evals()
        print(f"Refreshed critical ranking: {n} entries updated\n")

    key = _critical_window_key()

    # ── Ranking over the whole critical window (mean |ΔE|, fidelity) ─────
    rows = []
    for e in _load_manifest():
        if e.p_layers != args.p:
            continue
        rec = e.critical_ranking.get(key) if e.critical_ranking else None
        if not rec:
            continue
        rows.append((rec.get("abs_error_mean", float("inf")), rec.get("fidelity_mean"),
                     rec.get("grade", "?"), e.checkpoint_file))

    print(f"=== Critical-window ranking {key} (p={args.p}) — lower |ΔE| is better ===")
    if not rows:
        print("  (no critical_ranking data — run with --refresh, or run experiments first)")
    for ae, fid, grade, ck in sorted(rows, key=lambda r: r[0]):
        print(f"  |ΔE|_mean={_fmt(ae):>8}  fidelity_mean={_fmt(fid):>6}  {grade}  {ck}")

    # ── Per-N metrics at h≈target (default 1.0) ─────────────────────────
    print(f"\n=== Per-N |ΔE| and fidelity at h≈{args.h:.2f} (p={args.p}, N={args.n}) ===")
    metrics = get_critical_metrics_at_h(h=args.h, p_layers=args.p, n_values=args.n)
    if not metrics:
        print("  (no at-h≈1.0 records — run with --refresh)")
        return 0

    ordered = sorted(metrics.items(),
                     key=lambda kv: min((v["abs_error"] for v in kv[1].values()), default=float("inf")))
    for ck, per_n in ordered:
        print(f"\n{ck}")
        for N in args.n:
            if N in per_n:
                r = per_n[N]
                print(f"  N={N:>2}  h={_fmt(r['h'], 2)}  |ΔE|={_fmt(r['abs_error'])}  "
                      f"fidelity={_fmt(r['fidelity'], 4)}")
            else:
                print(f"  N={N:>2}  (no data at h≈{args.h:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
