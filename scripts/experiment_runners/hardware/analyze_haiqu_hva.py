#!/usr/bin/env python
"""Analyze REAL quantum-hardware HVA results, grouped by experiment.

Scope is deliberately narrow: ONLY real-QPU executions recovered from the
dashboard (``results/haiqu_recovered/``). Simulator (``fake_*``/``aer_*``) runs
and dry-run cost estimates are out of scope by design — they are never included,
even if present in the input directory. A file is accepted only when it is a
``haiqu_recovered_v1`` record on a real ``ibm_*`` device with a measured energy.

For each accepted file it reports, grouped by (topology, N, h, device):

  * Energy vs exact: energy, |ΔE|, ΔE/gap, pass@5% / pass@10%
  * Error-corrected energy: the mitigated ⟨H⟩ reconstructed from the QPU
    expectation values (this IS the recovered ``energy``)
  * Fidelity + shot uncertainty when the record carries them
  * VQE refinement (optimizer, min_loss, #iterations, restarts) when present

Usage
-----
    python scripts/experiment_runners/hardware/analyze_haiqu_hva.py
    python scripts/experiment_runners/hardware/analyze_haiqu_hva.py --json out.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SIM_PREFIXES = ("fake_", "aer_", "ionq_sim")


def _is_real_hardware_record(d: dict) -> bool:
    """Accept ONLY a recovered real-QPU record with a measured energy.

    Covers both recovered schemas (single-shot ``haiqu_recovered_v1`` and the
    server-side VQE ``haiqu_recovered_vqe_v1``). Guards against a simulator or
    dry-run file: requires a recovered schema, a real ``ibm_*`` device (not a
    ``fake_*`` simulator), and a non-null energy.
    """
    schema = str(d.get("schema", ""))
    device = str(d.get("device", "") or "")
    return (
        schema.startswith("haiqu_recovered")
        and device.startswith("ibm_")
        and not device.startswith(_SIM_PREFIXES)
        and d.get("energy") is not None
    )


def analyze_file(path: str) -> dict | None:
    """Flat summary of one recovered real-hardware JSON, or None if out of scope."""
    d = json.load(open(path))
    if not _is_real_hardware_record(d):
        return None
    is_vqe = str(d.get("schema", "")).startswith("haiqu_recovered_vqe")
    return {
        "file": os.path.basename(path),
        "kind": "VQE" if is_vqe else "single-shot",
        "topology": d.get("topology"),
        "n_qubits": d.get("n_qubits"),
        "h": d.get("h"),
        "device": d.get("device"),
        "job_id": d.get("job_id"),
        "status": d.get("status"),
        "partial": d.get("partial", False),
        "shots": d.get("shots"),
        "energy": d.get("energy"),  # error-corrected (mitigated) ⟨H⟩
        "e_exact": d.get("e_exact"),
        "gap": d.get("gap"),
        "abs_delta_e": d.get("abs_delta_e"),
        "de_gap": d.get("de_gap"),
        "pass_5pct": d.get("pass_5pct"),
        "pass_10pct": d.get("pass_10pct"),
        "fidelity": d.get("fidelity"),
        "uncertainty": d.get("uncertainty"),
        "n_observables": d.get("n_observables"),
        "estimated_qpu_cost": d.get("estimated_qpu_cost"),
        # VQE convergence fields (present only for the VQE schema)
        "vqe_min_loss": d.get("vqe_min_loss"),
        "vqe_n_iterations": d.get("vqe_n_iterations"),
        "vqe_loss_first": d.get("vqe_loss_first"),
        "vqe_loss_last": d.get("vqe_loss_last"),
        "vqe_monotone": d.get("vqe_monotone_decreasing"),
        "vqe_converged": d.get("vqe_converged"),
        "vqe_last_delta": d.get("vqe_last_delta"),
        "variational_bound_ok": d.get("variational_bound_ok"),
    }


def _fmt(v, spec="{:+.4f}"):
    return spec.format(v) if isinstance(v, (int, float)) else "-"


def print_report(rows: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["topology"], r["n_qubits"], r["h"], r["device"]), []).append(r)

    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        topo, n, h, dev = key
        print("=" * 80)
        print(f"{topo} N={n} h={h} | device={dev} (real QPU) | {len(groups[key])} execution(s)")
        for r in groups[key]:
            verdict = "PASS" if r["pass_5pct"] else "FAIL"
            partial = " (PARTIAL/Running)" if r.get("partial") else ""
            elabel = "min_loss (VQE)" if r["kind"] == "VQE" else "E(corrected)"
            print(
                f"  [{r['kind']}]{partial}  {elabel}={_fmt(r['energy'])}  "
                f"E_exact={_fmt(r['e_exact'])}  gap={_fmt(r['gap'], '{:.4f}')}"
            )
            print(
                f"    |ΔE|={_fmt(r['abs_delta_e'], '{:.4f}')}  "
                f"ΔE/gap={_fmt(r['de_gap'], '{:.2%}')}  "
                f"pass@5%={r['pass_5pct']}  pass@10%={r['pass_10pct']}  [{verdict}]"
            )
            extra = []
            if r["fidelity"] is not None:
                extra.append(f"fidelity={_fmt(r['fidelity'], '{:.4f}')}")
            if r["uncertainty"] is not None:
                extra.append(f"uncertainty={_fmt(r['uncertainty'], '{:.4f}')}")
            if extra:
                print("    " + "  ".join(extra))
            if r["kind"] == "VQE":
                print(
                    f"    VQE convergence: iters={r['vqe_n_iterations']}  "
                    f"loss {_fmt(r['vqe_loss_first'])} -> {_fmt(r['vqe_loss_last'])}  "
                    f"monotone={r['vqe_monotone']}  converged={r['vqe_converged']}  "
                    f"last_delta={_fmt(r['vqe_last_delta'], '{:.2e}')}"
                )
                # sanity: a variational energy must be >= exact (bound_ok True)
                sane = "OK" if r["variational_bound_ok"] else "VIOLATED (min_loss < e_exact!)"
                print(f"    sanity: variational bound {sane}")
            print(
                f"    shots={r['shots']}  n_obs={r['n_observables']}  "
                f"qpu_cost={r['estimated_qpu_cost']}"
            )
            print(f"    job={r['job_id']}  status={r.get('status') or 'Done'}")

    print("=" * 80)
    print(
        f"TOTAL: {len(rows)} real-hardware execution(s) in {len(groups)} group(s). "
        f"(simulator + dry-run files are excluded by design.)"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dir",
        default=str(_ROOT / "results" / "haiqu_recovered"),
        help="directory of recovered real-hardware JSONs",
    )
    ap.add_argument(
        "--json", type=Path, default=None, help="also write the flat per-file summary to this JSON"
    )
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    rows = [r for p in paths if (r := analyze_file(p)) is not None]
    skipped = len(paths) - len(rows)
    if not rows:
        print(
            f"no real-hardware recovered results in {args.dir} "
            f"({len(paths)} file(s) scanned, {skipped} out of scope)."
        )
        return 1
    if skipped:
        print(
            f"[note] skipped {skipped} out-of-scope file(s) "
            "(not a real-hardware recovered record).\n"
        )

    print_report(rows)

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, default=str))
        print(f"\nwrote per-file summary -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
