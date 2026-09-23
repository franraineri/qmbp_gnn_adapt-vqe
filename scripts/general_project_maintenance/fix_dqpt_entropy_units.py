#!/usr/bin/env python
"""Fix DQPT trajectory NPZ files whose `entropies` were saved in nats (ln).

A units bug in run_quench_dynamics_study._half_chain_entropy used np.log (nats)
instead of np.log2 (bits) before 2026-09-21. Persisted DQPT NPZs from that
period hold half-chain entropy in nats. This converts them to bits (× 1/ln2)
so they match the repo-wide bit convention (EntanglementAnalyzer).

Detection is exact for N ≤ 16 (recompute S in bits from the exact quench and
compare); for larger N it falls back to a heuristic. Only nats files are
converted. Idempotent: a file already in bits is left untouched. Writes a
`.bak` next to each converted file and stamps `entropy_units="bits"`.

Usage:
    python scripts/general_project_maintenance/fix_dqpt_entropy_units.py --dry-run
    python scripts/general_project_maintenance/fix_dqpt_entropy_units.py --apply
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_DQPT_DIR = _ROOT / "data" / "dqpt_trajectories"
_LN2 = math.log(2.0)
_EXACT_MAX_N = 16  # recompute-verify only where cheap


def _recompute_smax_bits(n: int, h_pre: float, h_post: float, dt: float, n_steps: int) -> float | None:
    """Exact S_max in bits for the quench, or None if too large / on error."""
    if n > _EXACT_MAX_N:
        return None
    try:
        from scipy.sparse.linalg import eigsh, expm_multiply

        from qmbp_simulation import HamiltonianBuilder, make_lattice
        from qmbp_simulation.analysis import EntanglementAnalyzer

        H1 = HamiltonianBuilder().build(make_lattice("chain_1d", n, J=1.0, h=h_pre))
        w, v = eigsh(H1.to_matrix(sparse=True), k=1, which="SA")
        psi = v[:, 0].astype(complex)
        H2s = HamiltonianBuilder().build(make_lattice("chain_1d", n, J=1.0, h=h_post)).to_matrix(sparse=True)
        ana = EntanglementAnalyzer()
        A = -1j * H2s * dt
        s = [ana.compute_half_chain_entropy(psi, n)]
        for _ in range(n_steps):
            psi = expm_multiply(A, psi)
            psi /= np.linalg.norm(psi)
            s.append(ana.compute_half_chain_entropy(psi, n))
        return float(max(s))
    except Exception:  # noqa: BLE001
        return None


def _classify(fp: Path, verify: bool = False) -> str:
    """Return 'bits' or 'nats' for a DQPT NPZ file.

    Primary rule: files stamped `entropy_units="bits"` are bits; every other
    file predates the fix and is nats (the bug was a global ln2 factor). When
    verify=True and the system is small (N ≤ 12, chain_1d), cross-check the
    call against an exact recomputation and warn on any mismatch.
    """
    d = np.load(fp, allow_pickle=True)
    ent = d.get("entropies")
    if ent is None or ent.size == 0:
        return "bits"  # nothing to convert
    stamped = str(d.get("entropy_units", "")) == "bits"
    kind = "bits" if stamped else "nats"

    if verify and not stamped and str(d.get("topology", "")) == "chain_1d" \
            and int(d["n_qubits"]) <= 12:
        bits = _recompute_smax_bits(
            int(d["n_qubits"]), float(d["h_pre"]), float(d["h_post"]),
            float(d["dt"]), int(d["n_steps"]),
        )
        if bits is not None:
            saved = float(np.max(ent))
            if abs(saved - bits * _LN2) > 1e-3 and abs(saved - bits) < 1e-3:
                print(f"  WARN: {fp.name} looks already in bits despite no stamp")
                return "bits"
    return kind


def _convert(fp: Path) -> None:
    """Convert entropies nats→bits in place, with .bak and a units stamp."""
    d = dict(np.load(fp, allow_pickle=True))
    d["entropies"] = np.asarray(d["entropies"], dtype=float) / _LN2
    d["entropy_units"] = "bits"
    shutil.copy2(fp, fp.with_suffix(fp.suffix + ".bak"))
    np.savez(fp, **d)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="report only, no writes")
    g.add_argument("--apply", action="store_true", help="convert nats files in place")
    ap.add_argument("--verify", action="store_true",
                    help="cross-check small chain_1d files against exact recompute")
    args = ap.parse_args(argv)

    files = sorted(_DQPT_DIR.glob("*.npz"))
    if not files:
        print(f"No NPZ files in {_DQPT_DIR}")
        return 1

    counts = {"bits": 0, "nats": 0}
    for fp in files:
        kind = _classify(fp, verify=args.verify)
        counts[kind] += 1
        action = ""
        if kind == "nats":
            action = " -> CONVERT" if args.apply else " -> would convert"
            if args.apply:
                _convert(fp)
        print(f"  [{kind:>4}]{action}  {fp.name}")

    print(f"\nbits={counts['bits']}  nats={counts['nats']}")
    if args.dry_run and counts["nats"]:
        print("Run with --apply to convert the nats files (a .bak is kept).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
