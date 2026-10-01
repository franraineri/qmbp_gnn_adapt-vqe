#!/usr/bin/env python
"""Fit the bond-sparsity law keep_frac(N) from the bond-ablation corpus.

Line A of the scalability study. The N=10 compression report found two absolute
collapse walls (nnn >= 3, nn >= 7); the open question is whether the *useful*
active fraction of bonds decreases with N (sparser state at larger N), which
would justify pruning more aggressively by default as N grows.

This extractor scans every bond-ablation artifact for a fixed (topology, h) and
builds, per N, the table of active bond counts vs reached fidelity, separating
the nn backbone from the nnn couplings (the corpus shows they obey different
laws). For each N it reports the SPARSEST masked point that is still "useful"
(fidelity >= --fid-threshold) — the empirical keep_frac(N) the default policy
should target — plus the full-reference ceiling for context.

It reads two schemas transparently (both store ``rows`` with ``n_nn_bonds`` /
``n_nnn_bonds`` / ``best_fidelity``):
  - ``variant_topk_*``       (run_variant_topk: full + T1-prune / T2-top-k rows)
  - ``bond_topk_regime_*``   (regime top-k sweep)

Lattice totals (tot_nn, tot_nnn) come from the same builders the runners use,
so fractions are exact. No fit library beyond numpy's lstsq; two closed-form
models are compared: frac(N) = a + b/N and frac(N) = c * N**(-alpha).

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/fit_sparsity_law.py \
        --topology square --h 0.5 --fid-threshold 0.90
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import STUDY_ROOT  # noqa: E402

from qmbp_simulation.models import make_lattice  # noqa: E402
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402

ABLATION_DIR = STUDY_ROOT / "bond_ablation"

# Absolute collapse walls from the N=10 compression report: below these the
# frustrated state is destroyed regardless of N (nnn<3 → 0.977→0.686,
# nn<7 → total collapse). The fraction law must never recommend fewer.
WALL_NN_MIN = 7
WALL_NNN_MIN = 3


def _predict_frac(model, N):
    """Evaluate a fitted fraction model at N. ``model`` is an inv_n or power dict."""
    if "b" in model:          # a + b/N
        return model["a"] + model["b"] / N
    return model["c"] * N ** (-model["alpha"])  # c * N**(-alpha)


def keep_counts_for_N(fit_nn, fit_nnn, n, tot_nn, tot_nnn):
    """Actionable integer keep-counts at N from the fraction laws + walls.

    Returns ``(k_nn, k_nnn, hit_wall)``: the recommended number of nn / nnn bonds
    to keep = ``round(frac(N) * total)`` clamped to ``[WALL_*_MIN, total]``.
    ``hit_wall`` is True when either clamp activated — i.e. the fraction law alone
    would have gone below a collapse wall, so the wall (not the law) governs.
    """
    def _one(fit, tot, wall):
        if not (fit.get("inv_n") or fit.get("power")):
            return min(tot, max(wall, tot)), False
        best = min((m for m in (fit.get("inv_n"), fit.get("power")) if m),
                   key=lambda m: m["rmse"])
        raw = int(round(_predict_frac(best, n) * tot))
        clamped = min(tot, max(wall, raw))
        return clamped, (clamped != raw)

    k_nn, w_nn = _one(fit_nn, tot_nn, WALL_NN_MIN)
    k_nnn, w_nnn = _one(fit_nnn, tot_nnn, WALL_NNN_MIN)
    return k_nn, k_nnn, (w_nn or w_nnn)


def _lattice_totals(topology, n, h):
    """Exact (tot_nn, tot_nnn) for the frustrated square lattice at this N."""
    lat = make_lattice(topology, n, J=1.0, h=h)
    return len(lat.edges), len(HamiltonianBuilder._generate_nnn_edges(lat))


def _iter_rows(topology, h):
    """Yield (N, is_full, nn, nnn, fid) for every usable ablation row.

    Covers both artifact schemas. A row is usable when it carries active bond
    counts and a fidelity. ``is_full`` flags the unmasked reference ceiling
    (variant name ends in ``_full`` or ``_full_ref``).
    """
    patterns = [f"variant_topk_*_{topology}_N*_h{h:.2f}.json",
                f"bond_topk_regime_{topology}_N*_h{h:.2f}.json"]
    seen = set()
    for pat in patterns:
        for f in sorted(glob.glob(str(ABLATION_DIR / pat))):
            if f in seen:
                continue
            seen.add(f)
            try:
                j = json.loads(Path(f).read_text())
            except Exception:
                continue
            m = re.search(r"_N(\d+)_", Path(f).name)
            if not m:
                continue
            n = int(m.group(1))
            for r in j.get("rows", []):
                nn = r.get("n_nn_bonds")
                nnn = r.get("n_nnn_bonds")
                fid = r.get("best_fidelity")
                if nn is None or nnn is None or fid is None:
                    continue
                name = str(r.get("variant", ""))
                is_full = name.endswith("_full") or name.endswith("_full_ref")
                yield n, is_full, int(nn), int(nnn), float(fid)


def build_table(topology, h, fid_threshold):
    """Per-N sparsity table.

    Returns a list of dicts sorted by N, each with the full-reference ceiling and
    the SPARSEST useful masked point (min active fraction with fid >= threshold).
    """
    per_n: dict[int, dict] = {}
    for n, is_full, nn, nnn, fid in _iter_rows(topology, h):
        tot_nn, tot_nnn = _lattice_totals(topology, n, h)
        rec = per_n.setdefault(n, {"N": n, "tot_nn": tot_nn, "tot_nnn": tot_nnn,
                                   "full": None, "masked": []})
        entry = {"nn": nn, "nnn": nnn, "fid": fid,
                 "frac_nn": nn / tot_nn, "frac_nnn": nnn / tot_nnn}
        if is_full:
            if rec["full"] is None or fid > rec["full"]["fid"]:
                rec["full"] = entry
        else:
            rec["masked"].append(entry)

    table = []
    for n in sorted(per_n):
        rec = per_n[n]
        useful = [m for m in rec["masked"] if m["fid"] >= fid_threshold]
        # Sparsest useful point = smallest total active fraction (nn+nnn) that
        # still clears the fidelity bar. Ties: prefer higher fidelity.
        sparsest = None
        if useful:
            sparsest = min(
                useful,
                key=lambda m: ((m["nn"] + m["nnn"]) / (rec["tot_nn"] + rec["tot_nnn"]), -m["fid"]),
            )
        table.append({
            "N": n,
            "tot_nn": rec["tot_nn"],
            "tot_nnn": rec["tot_nnn"],
            "full_fid": (rec["full"] or {}).get("fid"),
            "n_masked": len(rec["masked"]),
            "n_useful": len(useful),
            "sparsest_useful": sparsest,
        })
    return table


def _fit_models(ns, fracs):
    """Fit frac(N)=a+b/N and frac(N)=c*N**(-alpha). Returns dict of params+RMSE.

    Needs >= 2 points. Power law is fit in log space (requires frac>0). Returns
    ``None`` entries for models that cannot be fit.
    """
    ns = np.asarray(ns, float)
    fr = np.asarray(fracs, float)
    out = {"n_points": int(ns.size), "inv_n": None, "power": None}
    if ns.size < 2:
        return out

    # a + b/N via least squares.
    A = np.column_stack([np.ones_like(ns), 1.0 / ns])
    (a, b), *_ = np.linalg.lstsq(A, fr, rcond=None)
    pred = a + b / ns
    out["inv_n"] = {"a": float(a), "b": float(b),
                    "rmse": float(np.sqrt(np.mean((pred - fr) ** 2)))}

    # c * N**(-alpha) in log space (frac must be positive).
    if np.all(fr > 0):
        lx = np.log(ns)
        ly = np.log(fr)
        A2 = np.column_stack([np.ones_like(lx), -lx])
        (lc, alpha), *_ = np.linalg.lstsq(A2, ly, rcond=None)
        c = float(np.exp(lc))
        predp = c * ns ** (-alpha)
        out["power"] = {"c": c, "alpha": float(alpha),
                        "rmse": float(np.sqrt(np.mean((predp - fr) ** 2)))}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fit bond-sparsity law keep_frac(N)")
    p.add_argument("--topology", default="square")
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--fid-threshold", type=float, default=0.90,
                   help="Minimum fidelity for a masked point to count as useful.")
    p.add_argument("--extrapolate", type=int, nargs="+", default=[18, 20, 24],
                   help="N values to extrapolate the fitted laws to.")
    p.add_argument("--save", action="store_true",
                   help="Write the table + fit JSON under reports/.")
    args = p.parse_args(argv)

    table = build_table(args.topology, args.h, args.fid_threshold)
    if not table:
        print(f"No ablation rows found for topology={args.topology} h={args.h}")
        return 1

    print(f"=== Sparsity table ({args.topology}, h={args.h}, "
          f"useful = fid>={args.fid_threshold}) ===", flush=True)
    print(f"{'N':>3} {'tot_nn':>6} {'tot_nnn':>7} {'full_F':>7} "
          f"{'useful/masked':>13} {'sparse_nn':>9} {'sparse_nnn':>10} "
          f"{'frac_nn':>7} {'frac_nnn':>8} {'F':>6}")
    fit_pts = {"nn": ([], []), "nnn": ([], [])}
    for row in table:
        su = row["sparsest_useful"]
        if su:
            print(f"{row['N']:>3} {row['tot_nn']:>6} {row['tot_nnn']:>7} "
                  f"{(row['full_fid'] or 0):>7.4f} "
                  f"{row['n_useful']:>5}/{row['n_masked']:<7} "
                  f"{su['nn']:>9} {su['nnn']:>10} "
                  f"{su['frac_nn']:>7.3f} {su['frac_nnn']:>8.3f} {su['fid']:>6.4f}")
            fit_pts["nn"][0].append(row["N"])
            fit_pts["nn"][1].append(su["frac_nn"])
            fit_pts["nnn"][0].append(row["N"])
            fit_pts["nnn"][1].append(su["frac_nnn"])
        else:
            print(f"{row['N']:>3} {row['tot_nn']:>6} {row['tot_nnn']:>7} "
                  f"{(row['full_fid'] or 0):>7.4f} "
                  f"{row['n_useful']:>5}/{row['n_masked']:<7} "
                  f"{'—':>9} {'—':>10} {'—':>7} {'—':>8} {'—':>6}  (no useful masked point)")

    fits = {}
    for kind in ("nn", "nnn"):
        ns, fr = fit_pts[kind]
        fits[kind] = _fit_models(ns, fr)
        f = fits[kind]
        print(f"\n--- frac_{kind}(N) fit ({f['n_points']} points) ---")
        if f["inv_n"]:
            iv = f["inv_n"]
            print(f"  a + b/N : a={iv['a']:.4f} b={iv['b']:.4f} rmse={iv['rmse']:.4f}")
        if f["power"]:
            pw = f["power"]
            print(f"  c*N^-α  : c={pw['c']:.4f} α={pw['alpha']:.4f} rmse={pw['rmse']:.4f}")

    # Actionable recommendation: wall-aware integer keep-counts at each target N.
    # The fraction law sets the target; WALL_*_MIN clamps it so no recommendation
    # can land below a collapse wall. '*' marks N where a wall overrode the law.
    print(f"\n--- Recommended keep-counts (walls nn>={WALL_NN_MIN}, "
          f"nnn>={WALL_NNN_MIN}; '*' = wall governs) ---")
    recs = {}
    for N in args.extrapolate:
        tot_nn, tot_nnn = _lattice_totals(args.topology, N, args.h)
        k_nn, k_nnn, hit = keep_counts_for_N(fits["nn"], fits["nnn"], N, tot_nn, tot_nnn)
        recs[N] = {"k_nn": k_nn, "k_nnn": k_nnn, "tot_nn": tot_nn,
                   "tot_nnn": tot_nnn, "wall_governs": hit}
        star = " *" if hit else ""
        print(f"  N={N:>2}: keep nn={k_nn}/{tot_nn} nnn={k_nnn}/{tot_nnn}  "
              f"(frac_nn={k_nn/tot_nn:.2f} frac_nnn={k_nnn/tot_nnn:.2f}){star}")
    fits["recommendations"] = recs

    if args.save:
        out = STUDY_ROOT / "reports" / f"sparsity_law_{args.topology}_h{args.h:.2f}.json"
        out.write_text(json.dumps(
            {"topology": args.topology, "h": args.h,
             "fid_threshold": args.fid_threshold, "table": table, "fits": fits,
             "schema": "sparsity_law_v1"}, indent=2, default=float))
        print(f"\nSaved {out}")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
