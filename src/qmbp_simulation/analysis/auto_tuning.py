"""Auto-tuning feedback loop for the ``--auto`` knob selector.

When a run uses ``--auto`` but the user OVERRIDES one of the auto-chosen knobs
(rank-by / ftol / target-fid / two-pass / micro-descent), that is a signal: the
human disagreed with the heuristic. This module closes the learning loop:

1. :func:`log_override_divergence` — append one JSONL record per (auto vs user)
   divergence, with the detected regime, both choices, the achieved result
   (fidelity, 2q, fid/cx, seconds) and a reference to the saved spec/JSON for
   navigability. Pure observation; changes nothing.
2. :func:`analyze_override_log` — when BOTH the auto choice and the override have
   been measured for the same (regime, knob, value), decide who won by fidelity
   -per-2q, gated by an EVIDENCE THRESHOLD (min cases + min margin) so a single
   noisy run never flips a default. Persists the agreed adjustments to
   ``learned_overrides.json`` and writes a human-readable markdown report.
3. :func:`load_learned_overrides` — read back the learned adjustments so
   ``_apply_auto_defaults`` can bias its choice where evidence says the heuristic
   was wrong.

Design mirrors the training-exclusion registry pattern (``load_training_exclusions``
/ ``auto_detect_exclusions``): a JSON of learned decisions the runtime reads, plus
a disk log of the evidence behind them. Reuses ``json_dump``/``json_serialize``
from utils — no new serialization. All paths live under ``results/hva_vl_study/
auto_tuning/`` for tracking + navigability.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from qmbp_simulation.utils.helpers import json_dump

logger = logging.getLogger(__name__)

# Tunable knobs the --auto selector sets and that a user may override.
TUNABLE_KNOBS = ("rank_by", "ftol", "target_fid", "two_pass", "micro_descent")

# Evidence gate for learning: do not change a default on thin/noisy data.
DEFAULT_MIN_EVIDENCE = 3  # distinct runs favouring the override
DEFAULT_MIN_MARGIN = 0.02  # min fid/cx relative improvement to count as a win


def _repo_root() -> Path:
    """Repo root from this module (src/qmbp_simulation/analysis/ → parents[3])."""
    return Path(__file__).resolve().parents[3]


def auto_tuning_dir() -> Path:
    """Directory holding the override log, learned overrides, and the report."""
    d = _repo_root() / "results" / "hva_vl_study" / "auto_tuning"
    d.mkdir(parents=True, exist_ok=True)
    return d


def override_log_path() -> Path:
    """Append-only JSONL evidence log of auto-vs-user divergences."""
    return auto_tuning_dir() / "override_log.jsonl"


def learned_overrides_path() -> Path:
    """JSON of learned regime→knob adjustments the --auto selector reads."""
    return auto_tuning_dir() / "learned_overrides.json"


def override_report_path() -> Path:
    """Human-readable markdown summary for tracking/navigability."""
    return auto_tuning_dir() / "override_analysis.md"


def study_coherence_report_path() -> Path:
    """Markdown report of the study-artifact coherence check (C1)."""
    return auto_tuning_dir() / "study_coherence.md"


def regime_key(manifold_dim: int | None, gap: float | None) -> str:
    """Canonical regime label used to group evidence.

    The primary degeneracy signal is ``manifold_dim`` (gap is the backup), the
    same split :func:`_apply_auto_defaults` uses. Keep this in sync with the
    selector so logged regimes match the ones the selector reasons about.
    """
    degenerate = (manifold_dim or 1) > 1 or (gap is not None and gap < 1e-3)
    return "degenerate" if degenerate else "healthy_gap"


def log_auto_run(
    *,
    regime: str,
    knobs_effective: dict,
    knobs_auto: dict,
    result: dict,
    context: dict,
    spec_ref: str | None = None,
    run_json_ref: str | None = None,
) -> Path:
    """Append one record per masked-variant run under ``--auto``.

    Logs the EFFECTIVE knob values actually used (``knobs_effective``), what the
    pure heuristic WOULD have chosen (``knobs_auto``), and per-knob which ones the
    user overrode (``overrode[knob] = effective != auto``). Logging every auto run
    — not only divergences — gives :func:`analyze_override_log` both the baseline
    (auto value) and the treatment (override value) to compare fairly.

    ``result`` supplies the achieved metrics (fidelity/subspace_fidelity, 2q,
    fid/cx, seconds). ``context`` carries grouping coordinates (topology, N, h,
    variant, label). ``spec_ref`` / ``run_json_ref`` are repo-relative references
    for navigability. Never raises on an IO/serialization issue.
    """
    import datetime as _dt

    overrode = {
        k: (knobs_effective.get(k) != knobs_auto.get(k))
        for k in TUNABLE_KNOBS
        if k in knobs_effective or k in knobs_auto
    }
    rec = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "regime": regime,
        "knobs_effective": {k: knobs_effective.get(k) for k in TUNABLE_KNOBS},
        "knobs_auto": {k: knobs_auto.get(k) for k in TUNABLE_KNOBS},
        "overrode": overrode,
        "fidelity": result.get("subspace_fidelity") or result.get("best_fidelity"),
        "n_2q": result.get("n_2q_transpiled") or result.get("n_2q"),
        "fidelity_per_cx": result.get("fidelity_per_cx"),
        "seconds": result.get("seconds"),
        "seed_kind": result.get("seed_kind"),
        "topology": context.get("topology"),
        "n_qubits": context.get("n_qubits"),
        "h": context.get("h"),
        "variant": context.get("variant"),
        "label": context.get("label"),
        "spec_ref": _rel(spec_ref),
        "run_json_ref": _rel(run_json_ref),
    }
    path = override_log_path()
    try:
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")
    except OSError as exc:  # pragma: no cover - IO failure must not break a run
        logger.warning("log_auto_run: could not append (%s)", exc)
    return path


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    return str(o)


def read_override_log() -> list[dict]:
    """Load all JSONL evidence records (skips blank/corrupt lines)."""
    path = override_log_path()
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def load_learned_overrides() -> dict:
    """Load the learned regime→knob adjustments the --auto selector reads.

    Shape::

        {"version": 1, "overrides": {"<regime>": {"<knob>": <value>, ...}, ...},
         "updated_at": "...", "evidence": {...}}

    Returns an empty-but-valid dict when the file is absent/corrupt (so the
    selector degrades gracefully to its built-in heuristic).
    """
    path = learned_overrides_path()
    if not path.exists():
        return {"version": 1, "overrides": {}}
    try:
        d = json.loads(path.read_text())
        if not isinstance(d.get("overrides"), dict):
            d["overrides"] = {}
        return d
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("load_learned_overrides: corrupt file (%s), ignoring", exc)
        return {"version": 1, "overrides": {}}


def _rel(path_str: str | None) -> str | None:
    """Make a path relative to the repo root for portable references."""
    if not path_str:
        return None
    try:
        return str(Path(path_str).resolve().relative_to(_repo_root()))
    except (ValueError, OSError):
        return path_str


def analyze_override_log(
    *,
    min_evidence: int = DEFAULT_MIN_EVIDENCE,
    min_margin: float = DEFAULT_MIN_MARGIN,
    write: bool = True,
) -> dict:
    """Compare auto vs override evidence and (optionally) persist learned rules.

    For each ``(regime, knob)``, group the logged runs by the chosen value and
    compare the user-override value against the auto value using MEDIAN
    ``fidelity_per_cx`` (robust to a single lucky run). The override is adopted as
    a learned rule only when ALL hold:

    - both the auto value and the override value have been measured;
    - the override has ``>= min_evidence`` distinct runs;
    - the override's median fid/cx beats the auto's by ``>= min_margin`` (relative).

    Returns a summary dict ``{"regime":{"knob":{...decision...}}}``. When
    ``write`` is True, persists the adopted rules to ``learned_overrides.json``
    and a markdown report for tracking. Pure read of the log; only writes the two
    output artifacts.
    """
    import statistics as _stats

    records = read_override_log()
    # regime -> knob -> {"auto_value": vkey, "by_value": {vkey: [fid/cx, ...]}}
    grouped: dict[str, dict[str, dict]] = {}
    for r in records:
        reg = r.get("regime")
        fpc = r.get("fidelity_per_cx")
        eff = r.get("knobs_effective") or {}
        auto = r.get("knobs_auto") or {}
        if reg is None or fpc is None:
            continue
        for knob in TUNABLE_KNOBS:
            if knob not in eff or eff[knob] is None:
                continue
            slot = grouped.setdefault(reg, {}).setdefault(
                knob, {"auto_value": _vkey(auto.get(knob)), "by_value": {}}
            )
            slot["by_value"].setdefault(_vkey(eff[knob]), []).append(float(fpc))

    summary: dict[str, dict[str, Any]] = {}
    adopted: dict[str, dict[str, Any]] = {}
    for reg, knobs in grouped.items():
        for knob, slot in knobs.items():
            auto_vkey = slot["auto_value"]
            by_val = slot["by_value"]
            decision = {"knob": knob, "regime": reg, "adopted": False, "auto_value": _unkey(auto_vkey)}
            # auto baseline = runs that used the auto value
            auto_fpcs = by_val.get(auto_vkey, [])
            auto_med = _stats.median(auto_fpcs) if auto_fpcs else None
            # best override = highest-median value that is NOT the auto value and
            # has enough evidence
            best = None
            for vkey, fpcs in by_val.items():
                if vkey == auto_vkey or len(fpcs) < min_evidence:
                    continue
                med = _stats.median(fpcs)
                if best is None or med > best[1]:
                    best = (vkey, med, len(fpcs))
            decision.update(
                best_override=best[0] if best else None,
                override_median_fpc=best[1] if best else None,
                override_n=best[2] if best else 0,
                auto_median_fpc=auto_med,
                auto_n=len(auto_fpcs),
            )
            if best and auto_med is not None and auto_med > 0:
                rel_gain = (best[1] - auto_med) / auto_med
                decision["relative_gain"] = rel_gain
                if rel_gain >= min_margin:
                    decision["adopted"] = True
                    adopted.setdefault(reg, {})[knob] = _unkey(best[0])
            summary.setdefault(reg, {})[knob] = decision

    if write:
        _write_learned(adopted, summary, min_evidence, min_margin)
        _write_report(summary, records, min_evidence, min_margin)
    return summary


def _vkey(v) -> str:
    """Stable string key for a knob value (bools/floats/str unified)."""
    if isinstance(v, bool):
        return f"bool:{v}"
    if isinstance(v, (int, float)):
        return f"num:{v}"
    return f"str:{v}"


def _unkey(vkey: str):
    """Inverse of :func:`_vkey`."""
    kind, _, raw = vkey.partition(":")
    if kind == "bool":
        return raw == "True"
    if kind == "num":
        f = float(raw)
        return int(f) if f.is_integer() else f
    return raw


def _write_learned(adopted, summary, min_evidence, min_margin) -> Path:
    import datetime as _dt

    payload = {
        "version": 1,
        "updated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "evidence_gate": {"min_evidence": min_evidence, "min_margin": min_margin},
        "overrides": adopted,
        "analysis": summary,
    }
    path = learned_overrides_path()
    json_dump(payload, path)
    return path


def _write_report(summary, records, min_evidence, min_margin) -> Path:
    lines = [
        "# Auto-tuning — override analysis",
        "",
        f"*Evidence gate: >= {min_evidence} runs favouring the override AND "
        f">= {min_margin:.0%} relative fid/cx gain.*",
        "",
        f"**Total divergence records:** {len(records)} "
        f"(`{_rel(str(override_log_path()))}`)",
        "",
        "| Regime | Knob | Best override | override fid/cx (median, n) | auto fid/cx | rel. gain | adopted |",
        "|--------|------|---------------|-----------------------------|-------------|-----------|:-------:|",
    ]
    for reg in sorted(summary):
        for knob, d in sorted(summary[reg].items()):
            ov = d.get("best_override")
            if ov is None:
                continue
            om = d.get("override_median_fpc")
            on = d.get("override_n", 0)
            am = d.get("auto_median_fpc")
            rg = d.get("relative_gain")
            om_s = f"{om:.3e}" if om is not None else "—"
            am_s = f"{am:.3e}" if am is not None else "—"
            rg_s = f"{rg:+.1%}" if rg is not None else "—"
            lines.append(
                f"| {reg} | {knob} | {_unkey(ov)} | "
                f"{om_s} (n={on}) | {am_s} | {rg_s} | "
                f"{'✓' if d.get('adopted') else ' '} |"
            )
    lines.append("")
    lines.append("Adopted rules are read by `_apply_auto_defaults` via "
                 "`load_learned_overrides()` and bias the auto choice per regime.")
    path = override_report_path()
    path.write_text("\n".join(lines) + "\n")
    return path


# ─────────────────────── study-artifact coherence (C1) ───────────────────────

def _study_root() -> Path:
    """results/hva_vl_study/ (independent of scripts/, same as auto_tuning_dir)."""
    return _repo_root() / "results" / "hva_vl_study"


def validate_study_coherence(
    *,
    max_specs: int | None = 25,
    fid_tol: float = 0.02,
    gap_rel_tol: float = 0.05,
    write: bool = True,
) -> dict:
    """Integrity check for the vl_vs_hva study artifacts (specs + bond_ablation).

    The study analogue of :func:`validate_gt_npz_coherence` (which only covers
    training NPZ). It cross-checks, WITHOUT trusting the stored numbers:

    1. **Spec fidelity** — rebuilds each ``ansatz_specs/*.spec.json`` circuit
       (``AnsatzSpec.build``), re-evaluates its stored θ against the exact ground
       manifold (:func:`degenerate_ground_manifold` + :func:`subspace_fidelity`),
       and flags when the recomputed fidelity differs from the stored one by more
       than ``fid_tol``. This catches specs whose metric drifted or was written
       with a stale θ.
    2. **e0/gap drift** — compares the e0/gap recorded in each bond_ablation JSON
       against the authoritative ``GroundTruthCache`` (each runner re-diagonalizes
       independently via eigsh, so values can drift); flags relative gap
       differences above ``gap_rel_tol``.

    ``max_specs`` caps how many specs are re-evaluated (statevector cost grows as
    2^N); ``None`` = all. Returns a summary dict; when ``write`` is True also
    writes a markdown report under ``auto_tuning/study_coherence.md``. Read-only
    w.r.t. the artifacts — only the report is written. Never raises on a single
    bad artifact (collects it as an issue).
    """
    import glob

    import numpy as np

    from qmbp_simulation.analysis.fidelity import degenerate_ground_manifold, subspace_fidelity
    from qmbp_simulation.circuits.ansatz_spec import AnsatzSpec

    issues: list[dict] = []
    n_specs_checked = 0
    n_rows_checked = 0

    # ── 1) Spec fidelity re-evaluation ──────────────────────────────────
    spec_files = sorted(glob.glob(str(_study_root() / "ansatz_specs" / "*.spec.json")))
    if max_specs is not None:
        spec_files = spec_files[:max_specs]
    _manifold_cache: dict = {}
    for fp in spec_files:
        try:
            s = AnsatzSpec.load(fp)
        except Exception as exc:
            issues.append({"kind": "spec_load_error", "file": _rel(fp), "detail": str(exc)})
            continue
        if s.theta is None or s.fidelity is None:
            continue
        try:
            qc, theta = s.build()
            key = (s.topology, s.n_qubits, round(float(s.h), 4))
            if key not in _manifold_cache:
                from qmbp_simulation.models import make_lattice
                from qmbp_simulation.models.model_registry import get_model_spec

                mspec = get_model_spec("tfim_frustrated")
                hk = dict(getattr(mspec, "hamiltonian_kwargs", {}))
                hk["J2"] = float(getattr(s, "j2", 0.5) or 0.5)
                lat = make_lattice(s.topology, s.n_qubits, J=1.0, h=s.h)
                H = mspec.build_hamiltonian(lat, **hk)
                _manifold_cache[key] = degenerate_ground_manifold(H)
            manifold, _e0, _pg, _md = _manifold_cache[key]
            recomputed = subspace_fidelity(qc, np.asarray(theta, float), manifold)
            n_specs_checked += 1
            if abs(recomputed - float(s.fidelity)) > fid_tol:
                issues.append({
                    "kind": "spec_fidelity_drift", "file": _rel(fp),
                    "stored": round(float(s.fidelity), 4),
                    "recomputed": round(float(recomputed), 4),
                    "delta": round(abs(recomputed - float(s.fidelity)), 4),
                })
        except Exception as exc:
            issues.append({"kind": "spec_eval_error", "file": _rel(fp), "detail": str(exc)})

    # ── 2) e0/gap drift vs GroundTruthCache ─────────────────────────────
    try:
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        gt = GroundTruthCache()
        for fp in sorted(glob.glob(str(_study_root() / "bond_ablation" / "*.json"))):
            try:
                d = json.loads(Path(fp).read_text())
            except Exception:
                continue
            topo = (d.get("extra") or {}).get("topology") or d.get("topology")
            n = (d.get("extra") or {}).get("N") or d.get("N")
            h = (d.get("extra") or {}).get("h") or d.get("h")
            e0 = (d.get("extra") or {}).get("e0") or d.get("e0")
            if topo is None or n is None or h is None or e0 is None:
                continue
            cached = gt.get(topo, int(n), "tfim_frustrated", float(h))
            if not cached:
                continue
            n_rows_checked += 1
            e0_c = float(cached["energy"])
            if abs(float(e0) - e0_c) > max(1e-4, abs(e0_c) * 1e-4):
                issues.append({
                    "kind": "e0_drift_vs_gtcache", "file": _rel(fp),
                    "study_e0": round(float(e0), 5), "gtcache_e0": round(e0_c, 5),
                })
    except Exception as exc:  # pragma: no cover - GT cache optional
        issues.append({"kind": "gtcache_unavailable", "detail": str(exc)})

    summary = {
        "n_specs_checked": n_specs_checked,
        "n_rows_checked": n_rows_checked,
        "n_issues": len(issues),
        "issues": issues,
        "ok": len(issues) == 0,
    }
    if write:
        _write_coherence_report(summary, fid_tol, gap_rel_tol)
    return summary


def _write_coherence_report(summary, fid_tol, gap_rel_tol) -> Path:
    import datetime as _dt

    lines = [
        "# Study-artifact coherence (vl_vs_hva)",
        "",
        f"*Generated {_dt.datetime.now(_dt.timezone.utc).isoformat()}. "
        f"Spec-fidelity tol {fid_tol}, gap rel tol {gap_rel_tol}.*",
        "",
        f"**Checked:** {summary['n_specs_checked']} specs re-evaluated, "
        f"{summary['n_rows_checked']} bond_ablation rows vs GroundTruthCache.",
        f"**Issues:** {summary['n_issues']}",
        "",
    ]
    if not summary["issues"]:
        lines.append("✅ All checked study artifacts are coherent.")
    else:
        lines.append("| Kind | File | Detail |")
        lines.append("|------|------|--------|")
        for it in summary["issues"]:
            detail = {k: v for k, v in it.items() if k not in ("kind", "file")}
            lines.append(f"| {it.get('kind')} | {it.get('file', '—')} | {detail} |")
    path = study_coherence_report_path()
    path.write_text("\n".join(lines) + "\n")
    return path
