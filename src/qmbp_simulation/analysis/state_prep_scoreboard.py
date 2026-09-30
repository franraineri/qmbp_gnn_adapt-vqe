"""Incremental best-results scoreboard for state-preparation comparisons.

Tracks the BEST run per configuration (method, N, h, variant, loader) across
repeated experiments, so newly-found better results are added in an ordered,
deterministic way without losing history. Pure logic — no I/O side effects
beyond the explicit load/save helpers — so it is unit-testable in ``src``.

A "config" is identified by :data:`CONFIG_FIELDS`. An entry is only replaced
when a new run's fidelity exceeds the stored one by more than ``IMPROVE_TOL``
(ties keep the incumbent, so re-runs don't churn the file). Every entry keeps
its full metrics plus provenance (``source`` artifact, ``theta_ref``/``qpy_ref``)
and the ``updated_utc`` timestamp.
"""

from __future__ import annotations

from datetime import UTC, datetime

SCHEMA = "state_prep_scoreboard_v1"

# Fields that uniquely identify a configuration (the upsert key).
CONFIG_FIELDS = ("method", "n_qubits", "h", "variant", "loader")

# Minimum fidelity gain to count as an improvement (ties keep the incumbent).
IMPROVE_TOL = 1e-6


def config_key(entry: dict) -> tuple:
    """Return the hashable config key for an entry."""
    return tuple(entry.get(f) for f in CONFIG_FIELDS)


def _rounded_key(entry: dict) -> tuple:
    """Config key with h rounded to the 2-decimal convention for stable matching."""
    m, n, h, variant, loader = config_key(entry)
    hr = round(float(h), 2) if h is not None else None
    return (m, n, hr, variant, loader)


def make_entry(
    *,
    method: str,
    n_qubits: int,
    h: float,
    variant: str,
    fidelity: float,
    loader: str | None = None,
    chi: int | None = None,
    abs_error: float | None = None,
    de_gap: float | None = None,
    n_2q: int | None = None,
    total_gates: int | None = None,
    depth: int | None = None,
    converged: str | None = None,
    source: str | None = None,
    theta_ref: str | None = None,
    qpy_ref: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Build a normalized scoreboard entry (h rounded to 2 decimals)."""
    entry = {
        "method": method,
        "n_qubits": int(n_qubits),
        "h": round(float(h), 2),
        "variant": variant,
        "loader": loader,
        "chi": chi,
        "fidelity": float(fidelity),
        "abs_error": None if abs_error is None else float(abs_error),
        "de_gap": None if de_gap is None else float(de_gap),
        "n_2q": None if n_2q is None else int(n_2q),
        "total_gates": None if total_gates is None else int(total_gates),
        "depth": None if depth is None else int(depth),
        "converged": converged,
        "source": source,
        "theta_ref": theta_ref,
        "qpy_ref": qpy_ref,
    }
    if extra:
        entry["extra"] = extra
    return entry


def upsert(
    entries: list[dict], candidate: dict, *, tol: float = IMPROVE_TOL, now: str | None = None
) -> tuple[list[dict], str]:
    """Insert or update ``candidate`` into ``entries`` by config key.

    Returns ``(new_entries, action)`` where action is ``"added"`` (new config),
    ``"improved"`` (higher fidelity replaced the incumbent), or ``"kept"`` (the
    incumbent was equal-or-better and was retained). Does not mutate the input
    list. Deterministic: no reliance on dict ordering; ties keep the incumbent.
    """
    stamp = now or datetime.now(UTC).isoformat()
    key = _rounded_key(candidate)
    out: list[dict] = []
    action = "added"
    matched = False
    for e in entries:
        if _rounded_key(e) == key:
            matched = True
            incumbent_f = e.get("fidelity", -1.0) or -1.0
            cand_f = candidate.get("fidelity", -1.0) or -1.0
            if cand_f > incumbent_f + tol:
                new = dict(candidate)
                new["updated_utc"] = stamp
                new["previous_fidelity"] = incumbent_f
                out.append(new)
                action = "improved"
            else:
                out.append(e)
                action = "kept"
        else:
            out.append(e)
    if not matched:
        new = dict(candidate)
        new["updated_utc"] = stamp
        out.append(new)
        action = "added"
    return out, action


def upsert_many(
    entries: list[dict], candidates: list[dict], *, tol: float = IMPROVE_TOL, now: str | None = None
) -> tuple[list[dict], dict]:
    """Upsert a batch; returns ``(entries, counts)`` with per-action tallies."""
    counts = {"added": 0, "improved": 0, "kept": 0}
    for c in candidates:
        entries, action = upsert(entries, c, tol=tol, now=now)
        counts[action] += 1
    return entries, counts


def sort_entries(entries: list[dict]) -> list[dict]:
    """Deterministic ordering: by N, then h, then method, then fidelity desc."""
    return sorted(
        entries,
        key=lambda e: (
            int(e.get("n_qubits", 0)),
            round(float(e.get("h", 0.0)), 2),
            str(e.get("method", "")),
            -(e.get("fidelity") or -1.0),
            str(e.get("variant", "")),
        ),
    )


def build_document(
    entries: list[dict], *, model: str = "tfim_frustrated", j2: float = 0.5, topology: str = "square"
) -> dict:
    """Wrap sorted entries in the persisted JSON document shape."""
    return {
        "schema": SCHEMA,
        "model": model,
        "J2": j2,
        "topology": topology,
        "n_entries": len(entries),
        "entries": sort_entries(entries),
    }


def to_markdown(doc: dict) -> str:
    """Render the scoreboard document as a readable markdown table."""
    lines: list[str] = []
    lines.append(
        f"# Best-results scoreboard — {doc.get('topology', '?')} {doc.get('model', '?')} (J2={doc.get('J2', '?')})"
    )
    lines.append("")
    lines.append(
        "Best run per configuration `(method, N, h, variant, loader)`. "
        "Auto-generated — do not edit by hand. Entries update only when a "
        "new run's fidelity improves on the stored one."
    )
    lines.append("")
    lines.append(f"- Schema: `{doc.get('schema')}` — {doc.get('n_entries', 0)} entries.")
    lines.append("")
    lines.append("| N | h | method | variant | loader/χ | F | \\|ΔE\\| | 2q | total | depth | converged | updated |")
    lines.append(
        "|---|------|--------|---------|----------|------:|-------:|---:|------:|------:|-----------|---------|"
    )

    def fmt(x, spec=".4f"):
        return "—" if x is None else format(x, spec)

    for e in sort_entries(doc.get("entries", [])):
        if e.get("loader") == "mps" and e.get("chi"):
            loader = f"mps χ={e['chi']}"
        elif e.get("loader"):
            loader = e["loader"]
        else:
            loader = "—"
        updated = (e.get("updated_utc") or "")[:10]
        lines.append(
            f"| {e.get('n_qubits')} | {float(e.get('h', 0)):.2f} | {e.get('method')} | "
            f"{e.get('variant')} | {loader} | {fmt(e.get('fidelity'))} | "
            f"{fmt(e.get('abs_error'))} | {fmt(e.get('n_2q'), 'd')} | "
            f"{fmt(e.get('total_gates'), 'd')} | {fmt(e.get('depth'), 'd')} | "
            f"{e.get('converged') or '—'} | {updated} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"
