#!/usr/bin/env python
"""Recover real HVA-on-QPU results from already-run Haiqu jobs (Scenario B).

The deployment runner (``run_haiqu_hva_deployment.py``) only persists to JSON the
data it collects DURING a run. Real executions that finished on the dashboard but
were never harvested locally (a run killed after submit, or a ``--dry-run`` that
still triggered a real job) are therefore missing from ``results/haiqu/``. This
tool rebuilds them from the Haiqu dashboard WITHOUT re-submitting (no credits).

What it recovers
----------------
RUN jobs in observable mode that actually executed on hardware: ``Done``,
``dry_run=False``, on a real ``ibm_*`` device. Each such job's result is a list
of Pauli expectation values (one per Hamiltonian term). The energy is
recombined client-side exactly as ``HaiquBackend.evaluate`` does
(``E = Σ coeff_k ⟨P_k⟩``) after rebuilding the SAME Hamiltonian from the
topology/N/h, and compared to the exact ground state. VARIATIONAL jobs are
skipped: their ``result()`` is ``None`` (no retrievable optimized energy).

Because the job carries no topology/N/h metadata, those are supplied via CLI
(defaults match the runner). The observable-count check (rebuilt #Pauli terms ==
returned #values) guards against an N/topology/h mismatch before trusting the
energy.

Output
------
One organized JSON per recovered execution in ``results/haiqu_recovered/``, named
``recovered_<topology>_n<N>_h<h>_<device>_<job-id>.json``, holding the energy,
|ΔE|, ΔE/gap, pass flags, raw expectation values, and full job provenance — the
same fields the deployment JSON carries, so existing analysis works on them.

Usage
-----
    HAIQU_API_KEY=... python scripts/experiment_runners/hardware/\
recover_haiqu_hva_deployment.py --experiment "HVA heavy_hex N10 p1 h1.00" --n-qubits 10 --dry-run

    HAIQU_API_KEY=... python .../recover_haiqu_hva_deployment.py \
        --experiment "HVA heavy_hex N20 p1 h1.00" --n-qubits 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

TOPOLOGY = "heavy_hex"
P_LAYERS = 1
J = 1.0
H_VALUE = 1.0
PASS_5PCT, PASS_10PCT = 0.05, 0.10

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("recover_haiqu_hva")

_ROOT = Path(__file__).resolve().parents[3]


def _status_value(job) -> str:
    st = getattr(job, "status", None)
    return str(getattr(st, "value", st))


_SIMULATOR_PREFIXES = ("fake_", "aer_", "ionq_sim")


def _is_real_hardware_run(job) -> bool:
    """True ONLY for a completed real-QPU RUN that carries retrievable results.

    Every condition below was cross-checked against the live dashboard; the set
    is deliberately strict because ``dry_run`` alone is not enough to tell a real
    execution from a cost estimate or a simulator run:

      * ``job_type == RUN``      — excludes VARIATIONAL (result() is None) and
                                    COMPRESSION jobs.
      * ``run_type == DEVICE_RUN`` — a device execution (not an analytics job).
      * ``status == Done``       — finished with a result.
      * ``dry_run is False``     — not a cost-estimate upload.
      * ``device_id`` is a real ``ibm_*`` QPU, NOT a ``fake_*``/``aer_*``
        simulator — this is the condition that actually separates hardware from
        the fake_torino runs (both are DEVICE_RUN, so run_type alone is not it).
    """
    jt = str(getattr(job, "job_type", "")).replace("JobType.", "")
    rt = str(getattr(job, "run_type", "")).replace("RunJobType.", "")
    dev = str(getattr(job, "device_id", "") or "")
    is_sim = dev.startswith(_SIMULATOR_PREFIXES)
    return (
        jt == "RUN"
        and rt == "DEVICE_RUN"
        and getattr(job, "dry_run", None) is False
        and _status_value(job) == "Done"
        and dev.startswith("ibm_")
        and not is_sim
    )


def _is_real_hardware_vqe(job) -> bool:
    """True for a real-QPU VARIATIONAL job that actually ran the optimizer.

    A server-side VQE (``haiqu.variational_optimization``) exposes its results as
    DIRECT job attributes — ``min_loss`` / ``loss_history`` / ``optimal_parameters``
    — NOT via ``result()`` (which blocks on a still-Running job). A VARIATIONAL job
    that is only a dry-run cost estimate has ``min_loss is None`` and no
    ``loss_history``; that is the discriminator here (``dry_run`` is ``None`` on
    every VARIATIONAL job, so it cannot be used). ``Running`` jobs ARE accepted —
    they carry partial-but-real convergence data already.
    """
    jt = str(getattr(job, "job_type", "")).replace("JobType.", "")
    dev = str(getattr(job, "device_id", "") or "")
    lh = getattr(job, "loss_history", None)
    return (
        jt == "VARIATIONAL"
        and dev.startswith("ibm_")
        and not dev.startswith(_SIMULATOR_PREFIXES)
        and getattr(job, "min_loss", None) is not None
        and isinstance(lh, list)
        and len(lh) > 0
    )


def _expectation_values(result) -> np.ndarray:
    """Flatten a haiqu observable-run result to a 1-D float array of EVs."""
    arr = np.asarray(result, dtype=object)
    return np.array(arr.flatten().tolist(), dtype=float).flatten()


def _recover_job(job, args, hamiltonian, coeffs, e_exact, gap, out_dir) -> dict | None:
    """Rebuild energy + physics verdict for one hardware RUN job."""
    jid = getattr(job, "id", "?")
    try:
        result = job.result()
    except Exception as exc:  # noqa: BLE001
        logger.warning("  %s: result() failed (%s) — skipped", jid, exc)
        return None
    if result is None:
        logger.info("  %s: result is None (not an observable run) — skipped", jid)
        return None

    evs = _expectation_values(result)
    n_obs = len(hamiltonian.paulis)
    if evs.size != n_obs:
        logger.warning(
            "  %s: %d expectation value(s) but the rebuilt Hamiltonian has %d "
            "Pauli terms — N/topology/h mismatch, refusing to guess. Skipped.",
            jid,
            evs.size,
            n_obs,
        )
        return None

    energy = float(np.dot(coeffs, evs))
    abs_de = abs(energy - e_exact)
    de_gap = (abs_de / gap) if gap else None
    record = {
        "schema": "haiqu_recovered_v1",
        "recovered_utc": datetime.now(UTC).isoformat(),
        "job_id": jid,
        "device": getattr(job, "device_id", None),
        "job_name": getattr(job, "name", None),
        "topology": args.topology,
        "n_qubits": args.n_qubits,
        "p_layers": P_LAYERS,
        "h": args.h,
        "shots": getattr(job, "shots", None),
        "energy": energy,
        "e_exact": e_exact,
        "gap": gap,
        "abs_delta_e": abs_de,
        "de_gap": de_gap,
        "pass_5pct": de_gap is not None and de_gap <= PASS_5PCT,
        "pass_10pct": de_gap is not None and de_gap <= PASS_10PCT,
        "n_observables": int(n_obs),
        "expectation_values": evs.tolist(),
        "hamiltonian_coeffs": coeffs.tolist(),
        "pauli_terms": [str(hamiltonian.paulis[k]) for k in range(n_obs)],
        "estimated_qpu_cost": getattr(job, "estimated_qpu_cost_", None),
        "creation_date": str(getattr(job, "creation_date", None)),
        "finish_date": str(getattr(job, "finish_date", None)),
    }
    verdict = "PASS" if record["pass_5pct"] else "FAIL"
    logger.info(
        "  %s: E=%+.4f |ΔE|=%.4f ΔE/gap=%s [%s]",
        jid,
        energy,
        abs_de,
        f"{de_gap:.2%}" if de_gap is not None else "n/a",
        verdict,
    )

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = (
            f"recovered_{args.topology}_n{args.n_qubits}_h{args.h:.2f}_"
            f"{record['device']}_{jid}.json"
        )
        (out_dir / fname).write_text(json.dumps(record, indent=2, default=str))
        logger.info("    saved -> %s", out_dir / fname)
    return record


def _recover_vqe_job(job, args, e_exact, gap, out_dir) -> dict | None:
    """Rebuild the physics + convergence record for a server-side VQE job.

    Reads the optimizer results as DIRECT job attributes (never ``result()``,
    which blocks on a Running job). ``min_loss`` is the mitigated variational
    energy ⟨H⟩_min, compared to the exact ground state. Also captures the
    convergence curve (``loss_history``) and the standard sanity check that a
    variational energy must lie at or above the exact one.
    """
    jid = getattr(job, "id", "?")
    status = _status_value(job)
    min_loss = getattr(job, "min_loss", None)
    loss_history = getattr(job, "loss_history", None) or []
    opt_params = getattr(job, "optimal_parameters", None) or []
    if min_loss is None or not loss_history:
        logger.info("  %s: VARIATIONAL without min_loss/loss_history — skipped", jid)
        return None

    energy = float(min_loss)
    abs_de = abs(energy - e_exact)
    de_gap = (abs_de / gap) if gap else None
    n_iter = len(loss_history)
    monotone = all(loss_history[i + 1] <= loss_history[i] + 1e-9 for i in range(n_iter - 1))
    last_delta = (loss_history[-1] - loss_history[-2]) if n_iter >= 2 else None
    # A variational energy is an upper bound: min_loss < e_exact signals an
    # over-correcting mitigation / sign / noise problem, not a better minimum.
    variational_ok = energy >= e_exact - 1e-6

    record = {
        "schema": "haiqu_recovered_vqe_v1",
        "recovered_utc": datetime.now(UTC).isoformat(),
        "job_id": jid,
        "device": getattr(job, "device_id", None),
        "job_name": getattr(job, "name", None),
        "status": status,
        "partial": status != "Done",  # Running jobs carry partial convergence
        "topology": args.topology,
        "n_qubits": args.n_qubits,
        "p_layers": P_LAYERS,
        "h": args.h,
        "shots": getattr(job, "shots", None),
        "energy": energy,  # = min_loss, the mitigated variational ⟨H⟩
        "vqe_min_loss": energy,
        "e_exact": e_exact,
        "gap": gap,
        "abs_delta_e": abs_de,
        "de_gap": de_gap,
        "pass_5pct": de_gap is not None and de_gap <= PASS_5PCT,
        "pass_10pct": de_gap is not None and de_gap <= PASS_10PCT,
        "vqe_n_iterations": n_iter,
        "vqe_loss_history": [float(x) for x in loss_history],
        "vqe_loss_first": float(loss_history[0]),
        "vqe_loss_last": float(loss_history[-1]),
        "vqe_monotone_decreasing": bool(monotone),
        "vqe_last_delta": float(last_delta) if last_delta is not None else None,
        "vqe_converged": (last_delta is not None and abs(last_delta) < 1e-4),
        "vqe_optimal_parameters": [float(x) for x in opt_params],
        "vqe_n_params": len(opt_params),
        "variational_bound_ok": bool(variational_ok),
        "estimated_qpu_cost": getattr(job, "estimated_qpu_cost_", None),
        "creation_date": str(getattr(job, "creation_date", None)),
        "finish_date": str(getattr(job, "finish_date", None)),
    }
    verdict = "PASS" if record["pass_5pct"] else "FAIL"
    logger.info(
        "  %s: VQE min_loss=%+.4f |ΔE|=%.4f ΔE/gap=%s iters=%d conv=%s bound_ok=%s [%s%s]",
        jid,
        energy,
        abs_de,
        f"{de_gap:.2%}" if de_gap is not None else "n/a",
        n_iter,
        record["vqe_converged"],
        variational_ok,
        verdict,
        " PARTIAL/Running" if record["partial"] else "",
    )

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = (
            f"recovered_vqe_{args.topology}_n{args.n_qubits}_h{args.h:.2f}_"
            f"{record['device']}_{jid}.json"
        )
        (out_dir / fname).write_text(json.dumps(record, indent=2, default=str))
        logger.info("    saved -> %s", out_dir / fname)
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--experiment", required=True, help="Haiqu experiment name (or exp-... id) to recover from"
    )
    ap.add_argument("--topology", default=TOPOLOGY)
    ap.add_argument("--n-qubits", type=int, default=10)
    ap.add_argument("--h", type=float, default=H_VALUE)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--output-dir", default=str(_ROOT / "results" / "haiqu_recovered"))
    ap.add_argument(
        "--dry-run", action="store_true", help="report recoverable executions; write nothing"
    )
    args = ap.parse_args(argv)

    from haiqu.sdk import haiqu as hq

    from qmbp_simulation import ClassicalSolver, HamiltonianBuilder, make_lattice

    if not (getattr(hq, "user", None) and callable(hq.login)):
        pass
    hq.login()

    # Resolve experiment name -> id.
    exp = args.experiment
    if not exp.startswith("exp-"):
        exps = {e.name: str(e.id) for e in hq.list_experiments(widget=False)}
        if exp not in exps:
            raise SystemExit(
                f"no Haiqu experiment named {exp!r}. Available: {', '.join(sorted(exps))}"
            )
        exp_id = exps[exp]
    else:
        exp_id = exp
    logger.info("Recovering from experiment %r (%s)", args.experiment, exp_id)

    # Rebuild the Hamiltonian + ground truth ONCE (same for every job of this cell).
    lattice = make_lattice(args.topology, args.n_qubits, J=J, h=args.h)
    hamiltonian = HamiltonianBuilder().build(lattice)
    coeffs = np.real(np.asarray(hamiltonian.coeffs, dtype=complex))
    gt = ClassicalSolver().solve(hamiltonian, lattice)
    e_exact, gap = gt.ground_energy, gt.gap
    logger.info(
        "Ground truth @ h=%.2f (N=%d %s): E_exact=%+.4f gap=%.4f",
        args.h,
        args.n_qubits,
        args.topology,
        e_exact,
        gap,
    )

    jobs = hq.list_jobs(experiment_id=exp_id, widget=False, limit=args.limit) or []
    real_run = [j for j in jobs if _is_real_hardware_run(j)]
    real_vqe = [j for j in jobs if _is_real_hardware_vqe(j)]
    # Report what was deliberately excluded, so "0 recovered" is never mistaken
    # for a bug — simulator and dry-run jobs are intentionally out of scope.
    n_sim = sum(
        1 for j in jobs if str(getattr(j, "device_id", "") or "").startswith(_SIMULATOR_PREFIXES)
    )
    n_dry = sum(1 for j in jobs if getattr(j, "dry_run", None) is True)
    logger.info(
        "%d job(s) total | %d real-HW single-shot RUN(s), %d real-HW VQE job(s) "
        "| excluded: %d simulator, %d dry-run/cost-estimate",
        len(jobs),
        len(real_run),
        len(real_vqe),
        n_sim,
        n_dry,
    )

    out_dir = Path(args.output_dir)
    recovered = [
        r
        for j in real_run
        if (r := _recover_job(j, args, hamiltonian, coeffs, e_exact, gap, out_dir))
    ]
    recovered_vqe = [r for j in real_vqe if (r := _recover_vqe_job(j, args, e_exact, gap, out_dir))]

    total = len(recovered) + len(recovered_vqe)
    if args.dry_run:
        logger.info(
            "[dry-run] %d single-shot + %d VQE execution(s) recoverable into %s",
            len(recovered),
            len(recovered_vqe),
            out_dir,
        )
    else:
        logger.info(
            "Recovered %d single-shot + %d VQE execution(s) (%d total) -> %s",
            len(recovered),
            len(recovered_vqe),
            total,
            out_dir,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
