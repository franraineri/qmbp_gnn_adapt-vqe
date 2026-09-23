"""Shared paths, traceability, and circuit helpers for the HVA-vs-VL study.

Single source of truth for the analysis scripts comparing the bond-resolved HVA
ansatz against Haiqu Vector Loading (VL). Centralizes:

- The output directory layout under ``results/hva_vl_study/`` (one subfolder per
  experiment), so nothing lands loose in ``results/analysis/``.
- ``save_json``: writes a payload with self-referential traceability
  (``result_path`` + ``source_script`` + ``generated_utc``) so every artifact
  records where it lives and what produced it.
- Circuit helpers (qpy→Qiskit, gate stats) previously duplicated across scripts.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = _REPO_ROOT / "results" / "hva_vl_study"

# One subfolder per experiment (script). Keys are stable logical names.
SUBDIRS = {
    "expressivity": STUDY_ROOT / "expressivity",
    "vl_characterization": STUDY_ROOT / "vl_characterization",
    "circuit_comparison": STUDY_ROOT / "circuit_comparison",
    "vl_h_sweep": STUDY_ROOT / "vl_h_sweep",
    "state_prep_fidelity": STUDY_ROOT / "state_prep_fidelity",
    "hva_nnn_sweep": STUDY_ROOT / "hva_nnn_sweep",
    "resources": STUDY_ROOT / "resources",
}

# 2-qubit gate names counted across native and transpiled bases.
TWO_Q_GATES = ("cx", "cz", "ecr", "rzz", "rxx", "ryy", "cp", "swap", "iswap")

# Hardware-like transpilation target shared by every script, so depth / 2q are
# comparable across HVA and VL circuits.
BASIS_GATES = ["rz", "sx", "x", "cx"]
TRANSPILE_SEED = 42


def study_dir(name: str) -> Path:
    """Return (and create) the study subdirectory for ``name``."""
    d = SUBDIRS.get(name, STUDY_ROOT / name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _caller_script() -> str:
    """Best-effort name of the script that called save_json (traceability)."""
    for frame in inspect.stack()[2:]:
        path = Path(frame.filename)
        if path.name != Path(__file__).name and path.suffix == ".py":
            try:
                return str(path.relative_to(_REPO_ROOT))
            except ValueError:
                return path.name
    return "unknown"


def save_json(payload: dict, subdir: str, filename: str, **meta) -> Path:
    """Write ``payload`` to results/hva_vl_study/<subdir>/<filename> with trace.

    Injects ``result_path`` (repo-relative, self-referential), ``source_script``,
    and ``generated_utc`` into the payload so the artifact is traceable on its
    own. Extra ``meta`` keys are merged in. Returns the absolute path written.
    """
    path = study_dir(subdir) / filename
    enriched = {
        "result_path": str(path.relative_to(_REPO_ROOT)),
        "source_script": _caller_script(),
        "generated_utc": datetime.now(UTC).isoformat(),
        **meta,
        **payload,
    }
    path.write_text(json.dumps(enriched, indent=2, default=str))
    return path


def qiskit_from_qpy(blob):
    """Reconstruct a Qiskit circuit from a (base64 or bytes) QPY blob. None on error."""
    try:
        import base64

        import qiskit.qpy as qpy

        raw = base64.b64decode(blob) if isinstance(blob, str) else blob
        return qpy.load(io.BytesIO(raw))[0]
    except Exception:
        return None


def circuit_stats(qc) -> dict:
    """Depth / gate-count stats for a Qiskit circuit (None-safe)."""
    if qc is None:
        return {"depth": None, "n_2q_gates": None, "gate_counts": None, "num_qubits": None}
    ops = dict(qc.count_ops())
    widths = [len(inst.qubits) for inst in qc.data]
    return {
        "num_qubits": int(qc.num_qubits),
        "num_parameters": int(qc.num_parameters),
        "total_gates": int(len(qc.data)),
        "depth": int(qc.depth()),
        "depth_2q": int(qc.depth(lambda ins: len(ins.qubits) == 2)),
        "n_2q_gates": int(sum(v for k, v in ops.items() if k in TWO_Q_GATES)),
        "n_1q_gates": int(sum(1 for w in widths if w == 1)),
        "gate_counts": dict(sorted(ops.items(), key=lambda kv: -kv[1])),
    }


def exact_ground_state_vector(topology: str, n: int, h: float, *, model: str = "tfim",
                              j2: float = 0.0):
    """Return (psi, e0, gap) for the exact ground state (statevector-exact only).

    Single source of truth for the ground-state solve duplicated across the
    study scripts. Energy and gap come from the shared disk-persistent
    ``GroundTruthCache`` (``data/ground_truth_cache.json``) so repeated runs at
    the same (topology, N, h, model) reuse the cached values instead of
    re-diagonalizing — identical to ``ValidationRunner.exact_ground_state``.

    The state *vector* is not cached (2^N amplitudes are not persisted by
    design), so it is always solved once here via ``ClassicalSolver`` on the
    spec-built Hamiltonian. ``psi`` is normalized; it is ``None`` when the exact
    vector is unavailable (N above the statevector limit).

    Frustrated runs (``j2 > 0``) forward ``J2`` as ``model_kwargs`` so the cache
    key never collides with the ``J2 = 0`` entry for the same model.
    """
    import numpy as np

    from qmbp_simulation import make_lattice
    from qmbp_simulation.models.model_registry import get_model_spec
    from qmbp_simulation.solvers import ClassicalSolver
    from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

    spec = get_model_spec(model)
    ham_kwargs = dict(getattr(spec, "hamiltonian_kwargs", {}))
    model_kwargs = None
    if "J2" in ham_kwargs:
        ham_kwargs["J2"] = j2
        model_kwargs = {"J2": j2} if j2 else None

    solver = ClassicalSolver()
    e0, gap = GroundTruthCache().get_or_compute(
        topology, n, model, h, model_kwargs=model_kwargs, solver=solver
    )

    lat = make_lattice(topology, n, J=1.0, h=h)
    H = spec.build_hamiltonian(lat, **ham_kwargs)
    gt = solver.solve(H, lat)
    psi = None
    if gt.ground_state is not None:
        psi = np.asarray(gt.ground_state, dtype=complex)
        psi /= np.linalg.norm(psi)
    return psi, float(e0), float(gap)


def state_fidelity_exact(circuit, theta, psi_exact) -> float:
    """Exact fidelity |⟨psi_exact|circuit(theta)⟩|² via the shared implementation.

    Thin wrapper over ``qmbp_simulation.analysis.fidelity.compute_exact_fidelity``
    so every study script computes the overlap exactly one way. Returns 0.0 when
    the shared helper returns None (error / infeasible), keeping callers simple.
    """
    import numpy as np

    from qmbp_simulation.analysis.fidelity import compute_exact_fidelity

    fid = compute_exact_fidelity(circuit, np.asarray(theta), psi_exact)
    return float(fid) if fid is not None else 0.0


@contextlib.contextmanager
def haiqu_session(name: str):
    """Context manager that logs in to Haiqu and inits a named session.

    Centralizes the ``login(api_access_key=...) → init(name)`` boilerplate that
    was copy-pasted across the VL scripts. Reads ``HAIQU_API_KEY`` from the
    environment and raises ``RuntimeError`` if it is missing (before any job is
    submitted, so failures are explicit rather than surfacing as an
    ``APIKeyRequiredError`` mid-run). Yields the ``haiqu`` module.
    """
    from haiqu.sdk import haiqu

    key = os.environ.get("HAIQU_API_KEY")
    if not key:
        raise RuntimeError("HAIQU_API_KEY not set in environment.")
    haiqu.login(api_access_key=key)
    haiqu.init(name)
    yield haiqu


def run_vl_job(psi, n_qubits: int, num_layers: int = 2, timeout_s: float = 900.0, poll_s: float = 5.0):
    """Run a Haiqu vector_loading job on ``psi`` and return (fidelity, status, circuit).

    Handles the async lifecycle: submit → poll retrieve_status until terminal →
    refresh via get_job → fetch the CircuitModel and reconstruct the Qiskit
    circuit from qpy. Returns (fidelity|None, status_name, qiskit_circuit|None).
    Requires an active Haiqu session (caller must login/init first).
    """
    import time

    from haiqu.sdk import haiqu
    from haiqu.sdk.schemas import JobStatus

    job = haiqu.vector_loading(psi, num_qubits=n_qubits, num_layers=num_layers)
    t0 = time.time()
    status = None
    while time.time() - t0 < timeout_s:
        status = job.retrieve_status()
        if status in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED):
            break
        time.sleep(poll_s)
    job = haiqu.get_job(job.id)
    fidelity = getattr(job, "fidelity", None)
    status_name = status.name if status else "TIMEOUT"
    circuit = None
    cid = getattr(job, "circuit_id", None)
    if status == JobStatus.DONE and cid:
        cm = haiqu.get_circuit(cid)
        circuit = qiskit_from_qpy(getattr(cm, "qpy", None))
    return fidelity, status_name, circuit


def transpile_hw(qc, optimization_level: int = 2):
    """Transpile to the shared hardware-like basis (rz, sx, x, cx) with fixed seed."""
    from qiskit import transpile

    return transpile(qc, basis_gates=BASIS_GATES, optimization_level=optimization_level,
                     seed_transpiler=TRANSPILE_SEED)


def n_2q(qc) -> int:
    """Count 2-qubit gates (post-transpile CX-like) in a circuit."""
    return sum(v for k, v in qc.count_ops().items() if k in TWO_Q_GATES)


def robust_vqe_fidelity(circuit, hamiltonian, psi_exact, *, backend=None,
                        n_random: int = 6, maxiter: int = 600, include_zeros: bool = False,
                        seed0: int = 0):
    """Best fidelity of an ansatz to |psi_exact| via multi-restart L-BFGS-B.

    Measures the ansatz *expressivity ceiling* (not predictor quality): runs
    scipy L-BFGS-B from several random inits (and optionally the zeros vector),
    keeping the lowest-energy result. Returns (energy, bound_circuit, fidelity).

    Note: the all-zeros init is a trivial stationary point for some ansätze
    (e.g. bond-resolved +NNN) — off by default; enable with include_zeros=True.
    """
    import numpy as np
    from scipy.optimize import minimize

    from qmbp_simulation.execution import NoiselessBackend

    backend = backend or NoiselessBackend()
    n_params = circuit.num_parameters

    def cost(x):
        return backend.evaluate(circuit, hamiltonian, x)

    inits = [np.zeros(n_params)] if include_zeros else []
    inits += [np.random.default_rng(s).uniform(-0.4, 0.4, n_params)
              for s in range(seed0, seed0 + n_random)]

    best_e, best_th = None, None
    for x0 in inits:
        r = minimize(cost, x0, method="L-BFGS-B", bounds=[(-np.pi, np.pi)] * n_params,
                     options={"maxiter": maxiter, "ftol": 1e-12})
        if best_e is None or r.fun < best_e:
            best_e, best_th = r.fun, r.x
    bound = circuit.assign_parameters(best_th)
    fid = state_fidelity_exact(circuit, best_th, psi_exact)
    return float(best_e), bound, fid
