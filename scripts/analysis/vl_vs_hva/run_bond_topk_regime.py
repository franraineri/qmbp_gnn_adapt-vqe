#!/usr/bin/env python
"""Validate the k=0.5 nnn top-k sweet spot at N=18 with a regime-aware warm-start.

Combines three learnings into one run:
  1. Bond selection (masked HVA): keep all nn + the strongest fraction of nnn
     bonds by |θ| — the fewer-2q lever (bond_mask.top_k_by_weight).
  2. Regime-aware seed (warmstart.select_regime_seed): flat-renorm / second-order
     / so_nn_shrink by h — the sharpest analytic starting point per phase.
  3. θ_x-subspace exploration (study_core.optimize_xspace_bestof): the ZZ angles
     are pinned at their renormalized seed value; the exploration budget is spent
     on the soft θ_x direction (basin-hop on RX), not isotropically.

Ranking of nnn bonds is derived from a converged p2 full reference (so the
"strongest" bonds are meaningful). The masked p2 top-k circuit is then optimized
with the regime seed + θ_x exploration and compared against the p2 full
reference and the known p2_half_nn_rx baseline (0.9268 at N=18 h=0.5).

Reuses only existing pieces: study_core (ground_state, make_cost_fid,
optimize_bestof, optimize_xspace_bestof, cx_and_params), warmstart
(select_regime_seed, theta_x_indices), bond_mask (top_k_by_weight),
hva_variants (make_masked_variant, build_variant, AnsatzVariant), save_json into
the dedicated bond_ablation subfolder.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/run_bond_topk_regime.py \
        --n 18 --h 0.5 --p 2 --nnn-keep-frac 0.5 --restarts 4 --maxiter 2000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import StudyPersister, sync_scoreboard  # noqa: E402

from qmbp_simulation.analysis.theta_patterns import extract_theta_metrics  # noqa: E402
from qmbp_simulation.analysis.warmstart import (  # noqa: E402
    ORDERED_H_MAX,
    PARAMAGNETIC_H_MIN,
    crosses_transition,
    difficulty_index,
    phase_proximity,
    restarts_for_gap,
    select_regime_seed,
    theta_x_indices,
    topk_frac_for_gap,
)
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.bond_mask import (  # noqa: E402
    bond_weights_from_theta,
    top_k_by_weight,
)
from qmbp_simulation.circuits.hva_variants import (  # noqa: E402
    AnsatzVariant,
    build_variant,
    make_masked_variant,
)
from qmbp_simulation.framework.study_core import (  # noqa: E402
    cx_and_params,
    ground_state,
    make_cost_fid,
    optimize_xspace_bestof,
    prepare_warmstart,
)
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402

SUBDIR = "bond_ablation"


def _load_theta_donors(topology, n, h, nnn_edges, *, min_fid=0.85):
    """Scan the study tree for converged high-fidelity θ at (topology, N, h).

    Returns a donor list for ``best_warm_start_seed``: each dict has the θ, its
    layout (n_nn/n_nnn/p inferred from the artifact), a label, and the full-lattice
    ``nnn_edges`` for per-bond transfer alignment. Robust to missing fields.
    """
    import glob
    import json
    from pathlib import Path as _P

    study = _P("results/hva_vl_study/hva_nnn_sweep")
    donors = []
    for f in sorted(glob.glob(str(study / "*.json"))):
        try:
            d = json.loads(_P(f).read_text())
        except Exception:
            continue
        if d.get("h") is not None and abs(d["h"] - h) > 0.01:
            continue
        if (d.get("N") or d.get("n_qubits")) != n:
            continue
        for r in d.get("rows", []):
            th = r.get("best_theta_final") or r.get("theta_final")
            fid = r.get("best_fidelity") or r.get("fidelity")
            if not th or fid is None or fid < min_fid:
                continue
            # Infer p from θ length: len = (n_nn + n_nnn + n) * p. Donors are
            # full-bond ansätze, so n_nnn == len(nnn_edges); solve for n_nn > 0.
            n_nn_full, p_found = None, None
            for p_try in (1, 2, 3):
                if len(th) % p_try:
                    continue
                n_nn_candidate = len(th) // p_try - len(nnn_edges) - n
                if n_nn_candidate > 0:
                    n_nn_full, p_found = n_nn_candidate, p_try
            if n_nn_full is None:
                continue
            donors.append({
                "theta": th, "n_nn": n_nn_full, "n_nnn": len(nnn_edges),
                "p": p_found, "label": f"{r.get('variant', '?')}@{fid:.3f}",
                "nnn_edges": nnn_edges, "_fid": fid,
            })
    # Highest-fidelity donors first; de-dup by (label).
    donors.sort(key=lambda x: -x["_fid"])
    seen, uniq = set(), []
    for d in donors:
        if d["label"] in seen:
            continue
        seen.add(d["label"])
        uniq.append(d)
    return uniq[:4]  # cap: a few best donors is enough


def _load_donor(topology, donor_n, donor_h, p, j2, *, min_fid=0.9):
    """Converged full-ref θ from any (N, h) artifact, as a cascade donor.

    Generalizes the cross-N loader to also serve cross-h donors (same-phase
    neighbor h). transfer_theta handles a differing qubit count via
    ``donor_n_qubits`` (θ_x filled with the donor's N-invariant mean) and aligns
    nn/nnn by bond with regime fill, so a donor from (N', h') transfers onto the
    target layout regardless of whether N or h differ. Returns a donor dict
    carrying its own ``n_qubits`` + ``nnn_edges``, or None if missing / low-fid.
    """
    import json as _json
    from pathlib import Path as _P

    from qmbp_simulation.models import make_lattice

    f = _P(f"results/hva_vl_study/bond_ablation/"
           f"bond_topk_regime_{topology}_N{donor_n}_p{p}_h{donor_h:.2f}.json")
    if not f.exists():
        return None
    try:
        d = _json.loads(f.read_text())
    except Exception:
        return None
    row = next((r for r in d.get("rows", []) if r.get("variant") == "p2_full_ref"), None)
    if row is None:
        return None
    th = row.get("best_theta_final") or row.get("theta_final")
    fid = row.get("best_fidelity")
    if not th or fid is None or fid < min_fid:
        return None
    lat = make_lattice(topology, donor_n, J=1.0, h=donor_h)
    nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
    tag = f"N{donor_n}" if abs(donor_h - 0.0) < 0 else f"N{donor_n}h{donor_h:.2f}"
    return {
        "theta": th, "n_nn": row.get("n_nn_bonds", len(lat.edges)),
        "n_nnn": row.get("n_nnn_bonds", len(nnn_edges)), "p": p,
        "n_qubits": donor_n, "nnn_edges": nnn_edges, "h": donor_h,
        "label": f"donor<{tag}>@{fid:.3f}",
    }


def _autodiscover_donors(topology, N, h, p, j2, *, max_donors=4):
    """Scan the artifact corpus for SAME-PHASE donors (cross-N and cross-h).

    Thin back-compat shim over the shared src discoverer
    :func:`qmbp_simulation.analysis.warmstart.discover_donors` — the duplicated
    bond_ablation scanner that used to live here was removed. It still returns
    the same donor-dict shape (``theta`` as a list is coerced to ndarray by the
    discoverer; ``_fid`` is not populated, which callers do not require) from the
    bond-ablation corpus, now routed through the single source-pluggable
    implementation. Kept so the validate_* study scripts that import it keep
    working; new code should call ``discover_donors`` directly (and may also
    include the NPZ corpus via ``sources=("npz", "bond_ablation")``).
    """
    from qmbp_simulation.analysis.warmstart import discover_donors

    return discover_donors(
        topology, N, h, p,
        model="tfim_frustrated", frustrated=True,
        max_donors=max_donors, same_phase=True,
        sources=("bond_ablation",),
        ablation_variant="p2_full_ref", ablation_min_fid=0.9,
    )


def _preflight_output(out_file, *, force=False):
    """Pre-run guard: surface collisions and load the previous fidelities.

    Smooths the manual loop (no more ad-hoc ``ps``/``cat``):
    - lists any live ``run_bond_topk_regime`` processes so you notice a competing
      run before launching;
    - if the output artifact already exists, reads each variant's previous
      ``best_fidelity`` so the run can print Δ-vs-previous and the row can record
      it. Without ``--force`` it only WARNS (never aborts) — the run proceeds and
      overwrites, matching the existing idempotent-rewrite behavior.

    Returns ``{variant: prev_best_fidelity}`` (empty if no prior artifact).
    """
    import json as _json
    import subprocess as _sp
    from pathlib import Path as _P

    # Live sibling processes (informational).
    try:
        ps = _sp.run(["ps", "ax", "-o", "pid=,command="], capture_output=True,
                     text=True, timeout=5).stdout
        live = [ln.strip() for ln in ps.splitlines()
                if "run_bond_topk_regime.py" in ln and "ps ax" not in ln]
        if live:
            print(f"[preflight] {len(live)} live run_bond_topk_regime process(es) "
                  f"detected — ensure no (N,h) collision:", flush=True)
            for ln in live[:4]:
                print(f"[preflight]   {ln[:140]}", flush=True)
    except Exception:
        pass

    prev = {}
    path = _P("results/hva_vl_study/bond_ablation") / out_file
    if path.exists():
        try:
            d = _json.loads(path.read_text())
            for r in d.get("rows", []):
                v, f = r.get("variant"), r.get("best_fidelity")
                if v and f is not None:
                    prev[v] = float(f)
            tag = "will OVERWRITE (--force)" if force else "will overwrite (idempotent)"
            print(f"[preflight] output exists → {tag}: {out_file} "
                  f"(prev: {', '.join(f'{k}={v:.4f}' for k, v in prev.items())})",
                  flush=True)
        except Exception:
            pass
    return prev





def run(args) -> int:
    # Fail fast on bad inputs rather than mid-run (hours wasted).
    if args.h <= 0:
        raise ValueError(f"--h must be positive, got {args.h}")
    if not 0.0 < args.nnn_keep_frac <= 1.0:
        raise ValueError(f"--nnn-keep-frac must be in (0, 1], got {args.nnn_keep_frac}")
    if args.restarts < 1:
        raise ValueError(f"--restarts must be >= 1, got {args.restarts}")

    lat, _qc, Hop, psi, e0, gap, n_nn, n_nnn = ground_state(
        args.topology, args.n, args.h, args.j2, 1)
    nn = lat.edges
    nnn = HamiltonianBuilder._generate_nnn_edges(lat)
    b = HVACircuitBuilder()
    out_file = f"bond_topk_regime_{args.topology}_N{args.n}_p{args.p}_h{args.h:.2f}.json"
    prev_fids = _preflight_output(out_file, force=getattr(args, "force", False))
    rows: list[dict] = []

    # Donors: converged high-fidelity θ from prior runs at this (N,h), transferred
    # by best_warm_start_seed onto each variant's layout. Their nnn_edges are the
    # FULL lattice nnn (donors are full-bond ansätze), so per-bond alignment works.
    donors = _load_theta_donors(args.topology, args.n, args.h, nnn, min_fid=0.85)
    # Continuation donor (h-sweep): the converged full-ref θ from the previous h
    # step at this N. Prepended so it is tried first — within a phase it is a far
    # better donor than the analytic seed (cross-h study). It is dropped by the
    # sweep driver when the step crosses a phase boundary, so here we just use it.
    cont = getattr(args, "_continuation_donor", None)
    if cont is not None:
        donors = [cont, *donors]
    # Cross-N continuation donor (Fase B): a converged θ from a smaller N at the
    # SAME h, added as an EXTRA donor (not the sole seed) — the micro-descent
    # cascade decides whether it beats the regime seed. The cheap check showed
    # this helps in the transition window; gate on phase_proximity so we don't
    # pay the load/transfer where the regime seed already wins (ordered/paramag).
    cross_dn = getattr(args, "cross_n_donor_n", None)
    if cross_dn and phase_proximity(args.h) > 0.5 and cross_dn < args.n:
        cd = _load_donor(args.topology, cross_dn, args.h, args.p, args.j2)
        if cd is not None:
            donors = [cd, *donors]
            print(f"[topk_regime] cross-N donor from N={cross_dn}: {cd['label']}", flush=True)
    # Auto-discovered same-phase donors (cross-N + cross-h) via the SHARED
    # src-level discoverer (qmbp_simulation.analysis.warmstart.discover_donors),
    # replacing the former private _autodiscover_donors scanner. It draws from
    # BOTH corpora: the bond-ablation study JSONs (this script's historical
    # source, richer N/h coverage e.g. N=9/14/18) AND the NPZ training corpus the
    # pipeline now generates — so the frustrated study and the accelerated runner
    # share one discovery path and the generated data feeds this script too. The
    # cascade's micro-descent safely ignores donors that don't transfer, so this
    # only widens the candidate pool. Gated to the transition where donors help.
    if getattr(args, "auto_donors", True) and phase_proximity(args.h) > 0.5:
        from qmbp_simulation.analysis.warmstart import discover_donors

        auto = discover_donors(
            args.topology, args.n, args.h, args.p,
            model="tfim_frustrated", frustrated=True,
            sources=("npz", "bond_ablation"),
            ablation_variant="p2_full_ref", ablation_min_fid=0.9,
        )
        have = {d["label"] for d in donors}
        for d in auto:
            if d["label"] not in have:
                donors.append(d)
        if auto:
            print(f"[topk_regime] auto-discovered {len(auto)} same-phase donor(s) "
                  f"[shared discover_donors]", flush=True)
    if donors:
        print(f"[topk_regime] loaded {len(donors)} θ donor(s): "
              f"{[d['label'] for d in donors]}", flush=True)

    # Gap-adaptive budget (improvements #2/#3): near-degenerate (small-gap)
    # points get more restarts and a more aggressive top-k mask; large-gap
    # points keep the base budget. Disabled with --no-gap-adaptive.
    eff_restarts = args.restarts
    eff_keep_frac = args.nnn_keep_frac
    if getattr(args, "gap_adaptive", True):
        eff_restarts = restarts_for_gap(gap, args.restarts)
        eff_keep_frac = topk_frac_for_gap(gap, args.nnn_keep_frac)

    # Which regime seed will be used (provenance up-front).
    _, _seed_name = select_regime_seed(n_nn, n_nnn, args.n, args.p, args.h,
                                       J=1.0, J2=args.j2)
    print(f"[topk_regime] N={args.n} h={args.h} p={args.p} e0={e0:.5f} gap={gap:.5f} "
          f"n_nn={n_nn} n_nnn={n_nnn} keep_frac={eff_keep_frac:.2f} "
          f"restarts={eff_restarts} seed={_seed_name}"
          + (" [+continuation donor]" if cont is not None else ""), flush=True)

    persister = StudyPersister(
        subdir=SUBDIR, out_file=out_file,
        fingerprint={"n_qubits": args.n, "p_layers": args.p,
                     "model": "tfim_frustrated", "topology": args.topology,
                     "h": args.h},
        extra={"topology": args.topology, "N": args.n, "h": args.h,
               "p_layers": args.p, "J2": args.j2, "e0": e0, "gap": gap,
               "nnn_keep_frac": args.nnn_keep_frac, "schema": "bond_topk_regime_v1"},
        params={"experiment": "bond_topk_regime", "N": args.n, "h": args.h,
                "p_layers": args.p},
        description="k=0.5 nnn top-k at N=18 with regime-aware seed + "
                    "θ_x-subspace exploration",
        rows_ref=rows)

    # This runner's in-progress row carries extra fields (seed_name, bond counts,
    # fidelity_per_cx), so the per-restart callback stays inline below; the shared
    # persister owns only the payload/fingerprint boilerplate.
    def _persist_with_extra(all_rows, status="partial"):
        persister.persist(all_rows, status=status)

    def _persist(status="partial"):
        persister.persist(rows, status=status)

    def _eval_regime(variant, nn_bonds, nnn_bonds, target_nnn_edges):
        """Converge ``variant`` with the BEST warm-start + θ_x exploration.

        Uses the single reusable ``best_warm_start_seed`` chooser: it transfers
        converged high-fidelity donor θ (e.g. p3_base ~0.91) onto this variant's
        layout AND builds the regime analytic seed, evaluates both with the
        circuit's own ``fid`` and starts from the highest-fidelity one. So the
        reference and the masked top-k both begin IN the good basin (learning
        from prior runs), not from a blind analytic guess that collapses at N=18.

        Persists a crash-safe partial after EVERY restart so a long N=18 run is
        observable mid-flight and never loses completed restarts.
        """
        qc_v, _ = build_variant(b, args.n, lat, variant)
        n2q_v, npar_v = cx_and_params(qc_v)
        cost_v, fid_v, grad_v, _ = make_cost_fid(qc_v, Hop, psi)

        # Single integration point: the full combined warm-start cascade
        # (calibrated + regime + transferred/cross-N donors, selected by a short
        # micro-descent). prepare_warmstart owns the fid/grad/descent plumbing so
        # every runner wires it the same way. --warmstart regime reproduces the
        # pre-cascade behavior (regime seed only) for A/B.
        ws = prepare_warmstart(
            qc_v, Hop, psi, n_nn=nn_bonds, n_nnn=nnn_bonds, n_qubits=args.n,
            p_layers=args.p, h=args.h, J=1.0, J2=args.j2, donors=donors,
            target_nnn_edges=target_nnn_edges, strategy=args.warmstart,
            topology=args.topology, model="tfim_frustrated", gap=gap,
            micro_descent=args.micro_descent, cost=cost_v, fid=fid_v, grad=grad_v)
        seed_v, seed_vn, seed_fid = ws["seed"], ws["provenance"], ws["init_fidelity"]
        ws_report = ws["report"]
        if len(seed_v) != npar_v:
            raise ValueError(
                f"warm-start seed length {len(seed_v)} != circuit params {npar_v} "
                f"for {variant.name} (nn={nn_bonds} nnn={nnn_bonds} p={args.p})")
        x_idx_v = theta_x_indices(nn_bonds, nnn_bonds, args.n, args.p)
        print(f"    [{variant.name}] seed={seed_vn} "
              f"init_fid={seed_fid:.4f}" if seed_fid is not None else
              f"    [{variant.name}] seed={seed_vn}", flush=True)
        tv = time.time()

        def _on_restart(partial_runs, best, _name=variant.name, _n2q=n2q_v,
                        _npar=npar_v, _nnb=nn_bonds, _nnnb=nnn_bonds, _sn=seed_vn):
            # Per-restart crash-safe partial: a provisional row so the artifact
            # (and the console) show progress during the long N=18 convergence.
            fid_now = (best or {}).get("fidelity")
            prog = {
                "variant": _name, "status": "in_progress",
                "n_restarts_done": len(partial_runs),
                "best_fidelity": fid_now, "best_energy": (best or {}).get("energy"),
                "n_2q_transpiled": _n2q, "n_params": _npar,
                "fidelity_per_cx": (fid_now / _n2q) if fid_now is not None else None,
                "n_nn_bonds": _nnb, "n_nnn_bonds": _nnnb, "seed_name": _sn,
                "best_theta_final": (min(partial_runs, key=lambda r: r["energy"])
                                     ["theta_final"] if partial_runs else None),
            }
            _persist_with_extra(rows + [prog])
            print(f"    [{_name}] restart {len(partial_runs)}/{eff_restarts} "
                  f"best_fid={fid_now:.4f}" if fid_now is not None else
                  f"    [{_name}] restart {len(partial_runs)}", flush=True)

        fid_best, e_best_v, runs_v = optimize_xspace_bestof(
            cost_v, fid_v, grad_v, seed_v, x_idx_v, restarts=eff_restarts,
            maxiter=args.maxiter, seed0=args.seed0, sigma_x=args.sigma_x,
            on_restart=_on_restart)
        best = min(runs_v, key=lambda r: r["energy"])
        theta_final = np.asarray(best["theta_final"], float)
        # Derived angle metrics (automation): distance of the converged θ to the
        # regime seed + difficulty index, so downstream analysis never has to
        # recompute them from the raw θ. d_theta_to_seed is the signal that grows
        # with N in the transition (the collapse predictor).
        regime_only, _ = select_regime_seed(nn_bonds, nnn_bonds, args.n, args.p,
                                            args.h, J=1.0, J2=args.j2)
        tm = extract_theta_metrics(theta_final, nn_bonds, nnn_bonds, args.n, args.p,
                                   args.h, J2=args.j2, seed_theta=regime_only)
        D = difficulty_index(args.n, args.h, gap, J2=args.j2)
        return {
            "variant": variant.name, "best_fidelity": fid_best, "e_best": e_best_v,
            "e0": e0, "gap": gap, "n_2q_transpiled": n2q_v, "n_params": npar_v,
            "fidelity_per_cx": fid_best / n2q_v, "n_nn_bonds": nn_bonds,
            "n_nnn_bonds": nnn_bonds, "seed_name": seed_vn,
            "seconds": round(time.time() - tv, 1),
            "best_theta_final": best["theta_final"],
            # ── enriched metrics ──────────────────────────────────────────
            "init_fid": seed_fid,
            "delta_fid_vs_init": (fid_best - seed_fid) if seed_fid is not None else None,
            "restarts_used": eff_restarts,
            "gap_adaptive": bool(getattr(args, "gap_adaptive", True)),
            "difficulty_index": round(float(D), 3),
            "warmstart_report": ws_report,
            "prev_fidelity": prev_fids.get(variant.name),
            "delta_vs_prev": (round(fid_best - prev_fids[variant.name], 4)
                              if variant.name in prev_fids else None),
            "d_theta_to_seed": tm["d_theta_to_seed"],
            "nn_absmean": round(tm["nn_absmean"], 4),
            "nnn_absmean": round(tm["nnn_absmean"], 4),
            "theta_x_mean": round(tm["theta_x_mean"], 4),
            "l2_l1_nn_ratio": round(tm["l2_l1_nn_ratio"], 3),
            "phase": tm["phase"],
        }

    # ── 1) p2 full reference (converge once) → bond ranking ─────────────────
    # Converged via the regime seed + θ_x exploration (NOT cold-start): at
    # N=18 h=0.5 a cold reference fails (~0.46) and the bond ranking derived
    # from it is garbage. Sharing _eval_regime guarantees the reference uses the
    # same warm-start as the top-k run.
    ref = AnsatzVariant("p2_full_ref", "p2 full ref", blocks=["nn", "nnn", "x"] * args.p)
    ref_row = _eval_regime(ref, n_nn, n_nnn, nnn)
    rows.append(ref_row)
    _persist()
    _dv = ref_row.get("delta_vs_prev")
    _dtag = f" (Δ vs prev {_dv:+.4f})" if _dv is not None else ""
    print(f"  p2_full_ref fid={ref_row['best_fidelity']:.4f} "
          f"2q={ref_row['n_2q_transpiled']} seed={ref_row['seed_name']} "
          f"({ref_row['seconds']}s){_dtag}", flush=True)
    theta_ref = np.asarray(ref_row["best_theta_final"], float)

    # ── 2) top-k selection (ranked from the converged reference θ) ──────────
    w_nn, w_nnn = bond_weights_from_theta(theta_ref, n_nn, n_nnn, args.n, args.p)
    k = max(1, int(round(eff_keep_frac * n_nnn)))
    sel = top_k_by_weight(nn, nnn, w_nn, w_nnn, k_nn=None, k_nnn=k)
    v = make_masked_variant(
        f"p{args.p}_topk_regime", f"p{args.p} topk nnn={k}/{n_nnn} + regime seed",
        ["nn", "nnn", "x"] * args.p, sel, tags=("topk", "regime"))
    row = _eval_regime(v, len(sel.nn_edges), len(sel.nnn_edges), sel.nnn_edges)
    rows.append(row)
    _persist(status="final")
    _dv2 = row.get("delta_vs_prev")
    _dtag2 = f" (Δ vs prev {_dv2:+.4f})" if _dv2 is not None else ""
    print(f"  p{args.p}_topk_regime fid={row['best_fidelity']:.4f} "
          f"2q={row['n_2q_transpiled']} nnn={k}/{n_nnn} seed={row['seed_name']} "
          f"({row['seconds']}s){_dtag2}", flush=True)

    print(f"\n=== RESULT (N={args.n}) ===", flush=True)
    for r in rows:
        print(f"  {r['variant']:18s} fid={r['best_fidelity']:.4f} "
              f"2q={r['n_2q_transpiled']:>3} fid/cx={r['fidelity_per_cx']:.2e}", flush=True)
    print("  (baseline p2_half_nn_rx = 0.9268 @ 318 CX)", flush=True)
    print("DONE", flush=True)
    # Return the converged full-ref θ + layout so an h-sweep can reuse it as the
    # continuation donor for the next (descending) h step.
    return {
        "theta_full_ref": theta_ref, "n_nn": n_nn, "n_nnn": n_nnn, "p": args.p,
        "nnn_edges": nnn, "h": args.h, "gap": gap,
        "ref_fid": ref_row["best_fidelity"], "topk_fid": row["best_fidelity"],
    }


def run_sweep(args) -> int:
    """Descending-h sweep with phase-aware continuation (improvement #1).

    Sorts the requested h values DESCENDING (the cross-h study: θ transfers
    better from larger h to smaller within a phase) and runs each point via
    ``run``. The converged full-ref θ of each step is threaded into the next
    step as a continuation donor — UNLESS the step crosses a phase boundary
    (``crosses_transition``), in which case the donor is dropped and the point
    falls back to its regime seed (cross-phase transfer is poor). Gap-adaptive
    budget (#2/#3) and the bond-shape seed (#4, via the donor transfer path)
    apply per point.
    """
    hs = sorted({float(x) for x in args.h_sweep}, reverse=True)
    print(f"[h-sweep] N={args.n} descending h = {hs} "
          f"(gap_adaptive={getattr(args, 'gap_adaptive', True)})", flush=True)
    prev = None
    summary = []
    for h in hs:
        step_args = argparse.Namespace(**vars(args))
        step_args.h = h
        step_args.h_sweep = None
        # Thread the previous full-ref θ as a continuation donor, unless this
        # step crosses a phase boundary (then reset to the regime seed).
        donor = None
        if prev is not None and not crosses_transition(prev["h"], h):
            donor = {
                "theta": prev["theta_full_ref"], "n_nn": prev["n_nn"],
                "n_nnn": prev["n_nnn"], "p": prev["p"],
                "label": f"cont_h{prev['h']:.2f}@{prev['ref_fid']:.3f}",
                "nnn_edges": prev["nnn_edges"],
            }
            print(f"[h-sweep] step h={h}: continuation donor from h={prev['h']:.2f} "
                  f"(same phase)", flush=True)
        elif prev is not None:
            print(f"[h-sweep] step h={h}: crossed phase boundary from "
                  f"h={prev['h']:.2f} → reset to regime seed", flush=True)
        step_args._continuation_donor = donor
        res = run(step_args)
        if isinstance(res, dict):
            prev = res
            summary.append((h, res["gap"], res["ref_fid"], res["topk_fid"]))
    print(f"\n=== H-SWEEP SUMMARY (N={args.n}) ===", flush=True)
    print(f"  {'h':>5} {'gap':>9} {'ref_fid':>8} {'topk_fid':>8}", flush=True)
    for h, gap, rf, tf in summary:
        print(f"  {h:>5} {gap:>9.5f} {rf:>8.4f} {tf:>8.4f}", flush=True)
    print("SWEEP_DONE", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="top-k with regime seed + θ_x explore; "
                                            "optional phase-aware h-sweep")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=18)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--h-sweep", type=float, nargs="+", default=None,
                   help="Run a descending-h sweep over these values with "
                        "phase-aware continuation (overrides --h).")
    p.add_argument("--p", type=int, default=2)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--nnn-keep-frac", type=float, default=0.5)
    p.add_argument("--restarts", type=int, default=1,
                   help="Base restarts. Gap-adaptive scaling raises it for "
                        "small-gap points unless --no-gap-adaptive.")
    p.add_argument("--maxiter", type=int, default=2000)
    p.add_argument("--seed0", type=int, default=70000)
    p.add_argument("--sigma-x", type=float, default=0.3,
                   help="θ_x basin-hop perturbation stddev.")
    p.add_argument("--micro-descent", type=int, default=None,
                   help="L-BFGS-B iters for the warm-start cascade's micro-descent "
                        "(default: auto = difficulty-adaptive from N,h). "
                        "seed selection (0 = select by raw seed fidelity).")
    p.add_argument("--no-gap-adaptive", dest="gap_adaptive", action="store_false",
                   help="Disable gap-proportional restarts and aggressive top-k.")
    p.add_argument("--cross-n-donor-n", type=int, default=None,
                   help="Add a converged θ from this smaller N (same h) as an "
                        "extra cascade donor (Fase B). Only engages near h_c.")
    p.add_argument("--warmstart", choices=["combined", "regime"], default="combined",
                   help="Warm-start strategy: 'combined' = full cascade "
                        "(calibrated + regime + donors, micro-descent select); "
                        "'regime' = regime seed only (pre-cascade A/B baseline).")
    p.add_argument("--no-auto-donors", dest="auto_donors", action="store_false",
                   help="Disable auto-discovery of same-phase donors from the "
                        "artifact corpus (cross-N + cross-h).")
    p.set_defaults(auto_donors=True)
    p.add_argument("--force", action="store_true",
                   help="Acknowledge overwriting an existing output artifact "
                        "(preflight warns either way; run never aborts).")
    p.add_argument("--no-sync", action="store_true",
                   help="Skip the post-run scoreboard refresh.")
    p.set_defaults(gap_adaptive=True)
    args = p.parse_args(argv)
    if args.h_sweep:
        rc = run_sweep(args)
    else:
        rc = 0 if isinstance(run(args), dict) else 1
    if not args.no_sync:
        sync_scoreboard()  # fire-and-forget refresh of the shared scoreboard
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
