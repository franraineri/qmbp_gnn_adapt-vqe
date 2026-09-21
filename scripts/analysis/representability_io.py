"""I/O + schema for the classical-representability study (MPS / Aer / Haiqu VL).

Single source of truth for how each method's runner saves results, so the three
methods produce uniform files that `analyze_representability.py` can consolidate.
Schema: classical_representability_v1 (see results/classical_representability/README.md).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "classical_representability_v1"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "classical_representability"
INDEX_PATH = RESULTS_DIR / "index.jsonl"


@dataclass
class Row:
    """One (method, N, h, chi) measurement. Keep flat + JSON-serializable."""

    method: str
    N: int
    h: float
    chi_requested: int
    chi_actual: int | None = None
    energy: float | None = None
    e_ref: float | None = None
    abs_error: float | None = None
    trunc_error: float | None = None
    entanglement_entropy: float | None = None
    fidelity: float | None = None
    circuit_depth: int | None = None       # for VL / circuit methods
    two_qubit_gates: int | None = None      # for VL / circuit methods
    time_s: float | None = None
    mem_bytes_est: int | None = None
    status: str = "ok"                      # ok | truncated | failed
    note: str = ""


def mem_estimate_mps(n: int, chi: int) -> int:
    """Rough memory of an MPS: ~ N * chi^2 complex128 tensors (bytes)."""
    return int(n * chi * chi * 16)


def save_run(
    method: str,
    topology: str,
    rows: list[Row],
    *,
    N_values: list[int],
    h_values: list[float],
    chi_values: list[int],
    J: float = 1.0,
    extra_config: dict | None = None,
) -> Path:
    """Write one run's JSON + append a line to the append-only index.

    Returns the JSON path. Never raises on the index append (best-effort).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"study_{method}_{topology}_{ts}.json"

    payload = {
        "schema": SCHEMA,
        "method": method,
        "topology": topology,
        "generated_utc": datetime.now(UTC).isoformat(),
        "config": {
            "N_values": list(N_values),
            "h_values": list(h_values),
            "chi_values": list(chi_values),
            "J": J,
            **(extra_config or {}),
        },
        "rows": [asdict(r) for r in rows],
    }
    path.write_text(json.dumps(payload, indent=2))

    try:
        line = {
            "method": method, "topology": topology,
            "N_values": list(N_values), "h_values": list(h_values),
            "chi_values": list(chi_values), "path": path.name,
            "generated_utc": payload["generated_utc"], "n_rows": len(rows),
        }
        with open(INDEX_PATH, "a") as f:
            f.write(json.dumps(line) + "\n")
    except Exception:  # noqa: BLE001 - index is a convenience, never fatal
        pass

    return path


def load_all_rows(method: str | None = None) -> list[dict]:
    """Load and flatten every study JSON's rows, optional filter by method.

    Each returned dict carries its parent file's topology + generated_utc so the
    analyzer can group without re-opening files.
    """
    out: list[dict] = []
    for jp in sorted(RESULTS_DIR.glob("study_*.json")):
        try:
            d = json.loads(jp.read_text())
        except Exception:  # noqa: BLE001
            continue
        if d.get("schema") != SCHEMA:
            continue
        if method is not None and d.get("method") != method:
            continue
        for r in d.get("rows", []):
            r = dict(r)
            r.setdefault("topology", d.get("topology"))
            r["_source"] = jp.name
            r["_generated_utc"] = d.get("generated_utc")
            out.append(r)
    return out
