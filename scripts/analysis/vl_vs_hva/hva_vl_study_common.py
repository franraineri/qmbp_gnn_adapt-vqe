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

def _find_repo_root(start: Path) -> Path:
    """Resolve the repo root by walking up for a marker (pyproject.toml / .git).

    Replaces fragile depth-counting (`parents[N]`), which silently broke when this
    module was moved one directory deeper (parents[2] -> parents[3]) and misrouted
    every result to a spurious `scripts/results/`. Walking up for a marker is robust
    to future moves of this file.
    """
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    # Fallback (should not happen in-repo): assume 3 levels up from this file.
    return start.parents[3]


# Public alias — other study scripts should import this instead of counting
# parents[N], so a future directory move never misroutes their outputs.
find_repo_root = _find_repo_root

_REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
STUDY_ROOT = _REPO_ROOT / "results" / "hva_vl_study"

# Regression guard: fail loudly at import if the root ever resolves under scripts/
# (the classic misrouting bug) instead of the repo results tree.
assert "scripts" not in STUDY_ROOT.parts, (
    f"hva_vl_study STUDY_ROOT misresolved to {STUDY_ROOT} (under scripts/). "
    "Repo-root detection failed — check _find_repo_root / repo markers."
)

# One subfolder per experiment (script). Keys are stable logical names.
SUBDIRS = {
    "expressivity": STUDY_ROOT / "expressivity",
    "vl_characterization": STUDY_ROOT / "vl_characterization",
    "circuit_comparison": STUDY_ROOT / "circuit_comparison",
    "vl_h_sweep": STUDY_ROOT / "vl_h_sweep",
    "vl_quality_sweep": STUDY_ROOT / "vl_quality_sweep",
    "vl_mps_n18": STUDY_ROOT / "vl_mps_n18",
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


def save_circuit_qpy(circuit, path) -> Path:
    """Serialize a Qiskit circuit to a local QPY file (cloud- and key-independent).

    QPY is Qiskit's native binary format: it preserves the circuit with full
    fidelity so the VL circuit can be reloaded and re-analyzed later without any
    Haiqu session. Parent directories are created; the write is atomic (tmp +
    rename) to avoid corruption on crash. Returns the path written.
    """
    import qiskit.qpy as qpy

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            qpy.dump(circuit, f)
        tmp.rename(path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return path


def load_circuit_qpy(path):
    """Reload a Qiskit circuit previously saved with ``save_circuit_qpy``.

    Returns the circuit, or None if the file is missing or unreadable.
    """
    import qiskit.qpy as qpy

    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return qpy.load(f)[0]
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
                              j2: float = 0.0, return_hamiltonian: bool = False):
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

    With ``return_hamiltonian=True`` the spec-built Hamiltonian operator is
    appended to the tuple: ``(psi, e0, gap, H)``. Useful when the caller needs
    to evaluate the energy of another circuit against the same H (e.g. the VL
    energy error ``|⟨ψ_VL|H|ψ_VL⟩ − E0|``).
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
    if return_hamiltonian:
        return psi, float(e0), float(gap), H
    return psi, float(e0), float(gap)


def eigsh_ground_state(topology: str, n: int, h: float, *, model: str = "tfim",
                       j2: float = 0.0):
    """Exact ground state via sparse Lanczos (eigsh k=2), for 16 < N ≤ 22.

    Complements ``exact_ground_state_vector`` (capped at STATEVECTOR_MAX_N=16 by a
    conservative guard): here we call ``eigsh`` directly on the sparse
    Hamiltonian, which is tractable up to N≈22 (2^N amplitudes fit in memory;
    N=18 → 262k, ~6 s). Returns ``(psi, e0, gap, H)`` with ``psi`` normalized —
    the *verified exact* ground state, so both exact fidelity and ΔE-to-true-GS
    are available at N=18 (stronger than the DMRG/MPS reference the VL study used
    at the same size).
    """
    import numpy as np
    from scipy.sparse.linalg import eigsh

    from qmbp_simulation import make_lattice
    from qmbp_simulation.models.model_registry import get_model_spec
    from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

    spec = get_model_spec(model)
    ham_kwargs = dict(getattr(spec, "hamiltonian_kwargs", {}))
    if "J2" in ham_kwargs:
        ham_kwargs["J2"] = j2
    lat = make_lattice(topology, n, J=1.0, h=h)
    H = spec.build_hamiltonian(lat, **ham_kwargs)

    # E0/gap are p-independent Hamiltonian properties — reuse the shared
    # GroundTruthCache so repeated N=18 points don't re-diagonalize (~6.5 s each).
    # The state vector is never cached (2^N amplitudes); it is always re-solved.
    cache = GroundTruthCache()
    cached = cache.get(topology, n, model, h)

    evals, evecs = eigsh(H.to_matrix(sparse=True), k=2, which="SA")
    idx = np.argsort(evals)
    psi = evecs[:, idx[0]].astype(complex)
    psi /= np.linalg.norm(psi)
    e0 = float(evals[idx[0]])
    gap = float(evals[idx[1]] - evals[idx[0]])

    if cached is not None:
        # Trust the cached scalars (identical Hamiltonian); return the fresh vector.
        return psi, float(cached["energy"]), float(cached["gap"]), H
    cache.put(topology, n, model, h, energy=e0, gap=gap, method="eigsh_k2")
    cache.flush()
    return psi, e0, gap, H


def circuit_energy(circuit, hamiltonian) -> float:
    """Expectation value ⟨ψ|H|ψ⟩ of a parameter-free circuit via exact statevector.

    Used to score state-preparation circuits that carry no free parameters
    (e.g. a Haiqu VL circuit): the energy of the loaded state under ``H``. Raises
    if the circuit still has unbound parameters.
    """
    import numpy as np

    from qmbp_simulation.execution import NoiselessBackend

    if circuit.num_parameters:
        raise ValueError(
            f"circuit_energy expects a parameter-free circuit, got "
            f"{circuit.num_parameters} free parameters."
        )
    return NoiselessBackend().evaluate(circuit, hamiltonian, np.empty(0))


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


def run_vl_job(psi, n_qubits: int, num_layers: int = 2, timeout_s: float = 900.0,
               poll_s: float = 5.0, fine_tuning_iterations: int = 20,
               truncation_cutoff: float = 1e-6):
    """Run a Haiqu vector_loading job on ``psi``.

    Handles the async lifecycle: submit → poll retrieve_status until terminal →
    refresh via get_job → fetch the CircuitModel and reconstruct the Qiskit
    circuit from qpy. Returns
    ``(fidelity|None, status_name, qiskit_circuit|None, circuit_id|None, vl_meta)``.

    ``circuit_id`` is the Haiqu cloud handle. ``vl_meta`` is a dict of Haiqu-side
    provenance/metrics captured for later analysis (never used for scoring here):
    ``job_time`` (synthesis wall-time), ``circuit_metrics`` and ``circuit_analytics``
    (Haiqu's own circuit metrics), ``circuit_hash`` and ``transpilation_target``.
    Missing fields are simply absent. Requires an active Haiqu session.

    ``num_layers``, ``fine_tuning_iterations`` and ``truncation_cutoff`` are the
    VL quality knobs: more layers / iterations and a smaller cutoff trade cloud
    time (and, for layers, 2q-gate count) for higher loading fidelity — the main
    levers for the small-gap / frustrated regime.
    """
    import time

    from haiqu.sdk import haiqu
    from haiqu.sdk.schemas import JobStatus

    job = haiqu.vector_loading(
        psi, num_qubits=n_qubits, num_layers=num_layers,
        fine_tuning_iterations=fine_tuning_iterations, truncation_cutoff=truncation_cutoff,
    )
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
    vl_meta: dict = {"job_time": getattr(job, "time", None)}
    if status == JobStatus.DONE and cid:
        cm = haiqu.get_circuit(cid)
        circuit = qiskit_from_qpy(getattr(cm, "qpy", None))
        # Capture Haiqu's own circuit metrics for later analysis (not scored here).
        vl_meta["circuit_metrics"] = _jsonable(getattr(cm, "metrics", None))
        vl_meta["circuit_analytics"] = _jsonable(getattr(cm, "analytics", None))
        vl_meta["circuit_hash"] = getattr(cm, "hash", None)
        vl_meta["transpilation_target"] = _jsonable(getattr(cm, "transpilation_target", None))
    return fidelity, status_name, circuit, cid, vl_meta


def _jsonable(obj):
    """Best-effort convert a Haiqu SDK object (pydantic model / nested) to plain
    JSON-serializable data so it can be persisted. Returns None on failure."""
    if obj is None:
        return None
    for attr in ("model_dump", "dict"):  # pydantic v2 / v1
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    if isinstance(obj, (str, int, float, bool, list, dict)):
        return obj
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


def run_vl_mps_job(mps_tensors, *, num_layers: int = 2, timeout_s: float | None = None,
                   poll_s: float = 5.0, fine_tuning_iterations: int = 20,
                   truncation_cutoff: float = 1e-6, shape: str = "plr",
                   max_time: float = 900.0):
    """Run a Haiqu ``mps_loading`` job and return the same 5-tuple as run_vl_job.

    ``mps_tensors`` is a list of rank-3 site tensors in ``shape`` order (default
    ``plr``); for Vidal form, a ``(gammas, lambdas)`` tuple. Site ``i`` → qubit
    ``q_i``. Max bond dimension accepted by Haiqu is 64. Large-N counterpart of
    :func:`run_vl_job`.

    ``max_time`` is Haiqu's compute budget (passed to the job). ``timeout_s`` is
    OUR client-side poll budget; it must exceed ``max_time`` to allow for cloud
    latency/overhead — otherwise the poll gives up while the job is still
    finishing (status stuck at RUNNING). Defaults to ``max_time + 300``. Returns
    ``(fidelity|None, status_name, qiskit_circuit|None, circuit_id|None, vl_meta)``.
    Requires an active Haiqu session.
    """
    import time

    from haiqu.sdk import haiqu
    from haiqu.sdk.schemas import JobStatus

    if timeout_s is None:
        timeout_s = max_time + 300.0  # poll budget must exceed Haiqu compute budget

    job = haiqu.mps_loading(
        mps_tensors, shape=shape, num_layers=num_layers,
        fine_tuning_iterations=fine_tuning_iterations, truncation_cutoff=truncation_cutoff,
        max_time=max_time,
    )
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
    vl_meta: dict = {"job_time": getattr(job, "time", None), "loader": "mps_loading"}
    if status == JobStatus.DONE and cid:
        cm = haiqu.get_circuit(cid)
        circuit = qiskit_from_qpy(getattr(cm, "qpy", None))
        vl_meta["circuit_metrics"] = _jsonable(getattr(cm, "metrics", None))
        vl_meta["circuit_analytics"] = _jsonable(getattr(cm, "analytics", None))
        vl_meta["circuit_hash"] = getattr(cm, "hash", None)
        vl_meta["transpilation_target"] = _jsonable(getattr(cm, "transpilation_target", None))
    return fidelity, status_name, circuit, cid, vl_meta


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


def analytic_warmstart_theta(n_nn: int, n_nnn: int, n_qubits: int, p_layers: int,
                             h: float, *, J: float = 1.0, J2: float = 0.0):
    """Leading-order Trotter/adiabatic warm-start for the frustrated bond-resolved HVA.
    """
    from qmbp_simulation.analysis.warmstart import first_order_warmstart_theta

    return first_order_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2)


def analytic_warmstart_theta_2nd(n_nn: int, n_nnn: int, n_qubits: int, p_layers: int,
                                 h: float, *, J: float = 1.0, J2: float = 0.0,
                                 shrink_coef: float | None = None,
                                 curv_coef: float | None = None):
    """Second-order (frustration-aware) warm-start — promoted from inline scripts.

    Adds the Trotter/BCH ``(J/2h)^2`` corrections to the first-order seed: a
    shrink of the ZZ angles (theta_nn, theta_nnn) and a curvature term on
    theta_x. Confirmed to open a better basin than both the first-order seed and
    Metropolis basin-hopping, but ONLY in a narrow window around the frustrated
    transition (``0.4 <= h <= 0.6``). Deep in the ordered phase it over-shrinks
    and hurts — gate it with :func:`second_order_warmstart_applicable`.

    The calibrated coefficients (``shrink_coef=1/3``, ``curv_coef=1/6``) live in
    the unit-tested core ``qmbp_simulation.analysis.warmstart``; passing ``None``
    uses those defaults. Previously this formula was copy-pasted inline in
    ``confirm_second_order_warmstart.py`` (``_ws_2nd``).
    """
    from qmbp_simulation.analysis.warmstart import (
        DEFAULT_CURV_COEF,
        DEFAULT_SHRINK_COEF,
        second_order_warmstart_theta,
    )

    return second_order_warmstart_theta(
        n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2,
        shrink_coef=DEFAULT_SHRINK_COEF if shrink_coef is None else shrink_coef,
        curv_coef=DEFAULT_CURV_COEF if curv_coef is None else curv_coef,
    )


def second_order_warmstart_applicable(h: float) -> bool:
    """Regime-gate: True when the second-order warm-start is expected to help.

    Selection rule (not an always-on default): the second-order seed is enabled
    only in the confirmed window ``0.4 <= h <= 0.6`` around the frustrated
    transition. Outside it, callers fall back to the first-order seed with
    Metropolis / best-of restarts. Delegates to the unit-tested core.
    """
    from qmbp_simulation.analysis.warmstart import second_order_regime_gate

    return second_order_regime_gate(h)


def select_warmstart_theta(n_nn: int, n_nnn: int, n_qubits: int, p_layers: int,
                           h: float, *, J: float = 1.0, J2: float = 0.0,
                           prefer_second_order: bool = True):
    """Regime-gated warm-start seed: second-order inside the window, else first.

    Returns ``(theta, order)`` where ``order`` is ``"second"`` when the
    second-order seed was selected (``prefer_second_order`` and h in the gate
    window) and ``"first"`` otherwise. Centralizes the selection rule so every
    caller gates the second-order seed identically instead of hardcoding it.
    """
    if prefer_second_order and second_order_warmstart_applicable(h):
        return analytic_warmstart_theta_2nd(n_nn, n_nnn, n_qubits, p_layers, h,
                                            J=J, J2=J2), "second"
    return analytic_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h,
                                    J=J, J2=J2), "first"


def _adjoint_jac(circuit, hamiltonian):
    """Exact adjoint-gradient callable for L-BFGS-B ``jac``, or None (FD fallback).

    Thin re-export of the single source of truth in
    ``qmbp_simulation.framework.study_runner.make_adjoint_gradient`` so every
    warm-start helper here uses the same validated exact gradient.
    """
    from qmbp_simulation.framework.study_runner import make_adjoint_gradient

    return make_adjoint_gradient(circuit, hamiltonian)


def _theta_x_mask(n_nn, n_nnn, n_qubits, p_layers):
    """Boolean mask selecting the transverse-field (theta_x) parameters across layers."""
    import numpy as np

    per = n_nn + n_nnn + n_qubits
    mask = np.zeros(per * p_layers, bool)
    for layer in range(p_layers):
        o = layer * per
        mask[o + n_nn + n_nnn:o + per] = True
    return mask


def warmstart_bestof_vqe(circuit, hamiltonian, psi_exact, *, n_nn, n_nnn, n_qubits,
                         p_layers, h, J=1.0, J2=0.0, backend=None, maxiter=80,
                         sigmas=(0.0, 0.1, 0.1, 0.3), seed0=7000,
                         perturb_subspace="all", use_adjoint_grad=True):
    """Best fidelity via analytic warm-start + perturbed best-of restarts.

    The defensible high-dimension strategy from the restart-heuristic study: seed
    every restart from the analytic warm-start (guaranteed correct basin), then
    perturb by Gaussian noise of scale ``sigma`` and keep the lowest-energy
    result. The default plan ``(0.0, 0.1, 0.1, 0.3)`` = one pure warm-start (the
    floor) + two small-sigma refinements (never collapse) + one large-sigma jump
    (reaches the peak in the small-gap / frustrated regime). Replaces pure-random
    restarts, which collapse ~40% of the time near the transition.

    ``perturb_subspace`` selects where the noise is applied:
    - ``"all"`` (default): isotropic perturbation across all parameters.
    - ``"theta_x"``: perturb only the transverse-field parameters. This looked
      best in a single-seed screen (fid 0.945 at h=0.5) but the 8-seed
      confirmation found it is NOT significant (0.827+/-0.065 vs isotropic
      0.812+/-0.030 at h=0.5 — only 0.21 sigma; see
      REPORT_HEURISTIC_warmstart_restarts.md, confirmation phase). The confirmed
      winner is Metropolis basin-hopping (see ``warmstart_metropolis_vqe``), not
      directed perturbation. Kept here for reproducibility, not recommended.

    Returns ``(energy, bound_circuit, fidelity, runs, best_theta)`` where ``runs``
    is the per-restart record (sigma, energy, fidelity, nit, theta) for full
    reuse/analysis, and ``best_theta`` is the optimal parameter vector (for
    reloading the prepared state or warm-starting another experiment).
    """
    import numpy as np
    from scipy.optimize import minimize

    from qmbp_simulation.execution import NoiselessBackend

    backend = backend or NoiselessBackend()
    n_params = circuit.num_parameters
    th_ws = analytic_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2)
    if perturb_subspace == "theta_x":
        mask = _theta_x_mask(n_nn, n_nnn, n_qubits, p_layers)
    else:
        mask = np.ones(n_params, bool)

    def cost(x):
        return backend.evaluate(circuit, hamiltonian, x)

    # Exact adjoint gradient as L-BFGS-B jac (default) — same optimum, far fewer
    # circuit evals. Falls back to finite-difference if unavailable.
    jac = _adjoint_jac(circuit, hamiltonian) if use_adjoint_grad else None

    best_e, best_th = None, None
    runs = []
    for i, sigma in enumerate(sigmas):
        if sigma == 0.0:
            x0 = th_ws
        else:
            rng = np.random.default_rng(seed0 + i)
            noise = np.zeros(n_params)
            noise[mask] = rng.normal(0, sigma, int(mask.sum()))
            x0 = np.clip(th_ws + noise, -np.pi, np.pi)
        r = minimize(cost, x0, method="L-BFGS-B", jac=jac,
                     bounds=[(-np.pi, np.pi)] * n_params,
                     options={"maxiter": maxiter, "ftol": 1e-12})
        fid_i = state_fidelity_exact(circuit, r.x, psi_exact)
        runs.append({"sigma": float(sigma), "energy": float(r.fun),
                     "fidelity": float(fid_i), "nit": int(r.nit),
                     "theta": r.x.tolist()})
        if best_e is None or r.fun < best_e:
            best_e, best_th = r.fun, r.x
    bound = circuit.assign_parameters(best_th)
    fid = state_fidelity_exact(circuit, best_th, psi_exact)
    return float(best_e), bound, fid, runs, np.asarray(best_th)


def warmstart_metropolis_vqe(circuit, hamiltonian, psi_exact, *, n_nn, n_nnn, n_qubits,
                             p_layers, h, J=1.0, J2=0.0, backend=None, maxiter=80,
                             sigma=0.3, n_hops=8, temperature=None, seed0=13000,
                             use_adjoint_grad=True):
    """Best fidelity via analytic warm-start + Metropolis basin-hopping.

    The confirmed winner of the heuristic study (8-seed confirmation): seed from
    the analytic warm-start, then hop with Gaussian perturbations of scale
    ``sigma``, accepting uphill moves with probability ``exp(-dE / T)`` so the
    search can escape the warm-start basin. Near the frustrated transition this
    beats isotropic perturbed best-of by ~1.9 sigma (0.908+/-0.042 vs
    0.812+/-0.030 at h=0.5, N=9); in the easy regime all methods tie. ``T``
    defaults to the spectral gap (the energy scale of interest).

    Returns ``(energy, bound_circuit, fidelity, runs, best_theta)``; ``runs`` logs
    each hop (energy, fidelity, accepted).
    """
    import numpy as np
    from scipy.optimize import minimize

    from qmbp_simulation.execution import NoiselessBackend

    backend = backend or NoiselessBackend()
    n_params = circuit.num_parameters
    th_ws = analytic_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2)
    T = temperature if temperature is not None else 1.0
    rng = np.random.default_rng(seed0)

    def cost(x):
        return backend.evaluate(circuit, hamiltonian, x)

    jac = _adjoint_jac(circuit, hamiltonian) if use_adjoint_grad else None

    def opt(x0):
        r = minimize(cost, x0, method="L-BFGS-B", jac=jac,
                     bounds=[(-np.pi, np.pi)] * n_params,
                     options={"maxiter": maxiter, "ftol": 1e-12})
        return r.x, float(r.fun)

    x_cur, e_cur = opt(th_ws)
    x_best, e_best = x_cur.copy(), e_cur
    runs = [{"hop": 0, "energy": float(e_cur),
             "fidelity": float(state_fidelity_exact(circuit, x_cur, psi_exact)),
             "accepted": True}]
    for k in range(n_hops):
        x0 = np.clip(x_cur + rng.normal(0, sigma, n_params), -np.pi, np.pi)
        x_new, e_new = opt(x0)
        de = e_new - e_cur
        accepted = de < 0 or rng.random() < np.exp(-de / max(T, 1e-6))
        if accepted:
            x_cur, e_cur = x_new, e_new
        if e_new < e_best:
            x_best, e_best = x_new.copy(), e_new
        runs.append({"hop": k + 1, "energy": float(e_new),
                     "fidelity": float(state_fidelity_exact(circuit, x_new, psi_exact)),
                     "accepted": bool(accepted)})
    bound = circuit.assign_parameters(x_best)
    fid = state_fidelity_exact(circuit, x_best, psi_exact)
    return float(e_best), bound, fid, runs, np.asarray(x_best)


# Note: the second-order warm-start *optimizer* (2nd-order seed + perturbed
# best-of) is provided once by the base runner via
# ``StudyRunner`` / ``run_strategy("second_order")`` in
# qmbp_simulation.framework.study_runner — not duplicated here. Scripts that
# only need the *seed* use ``analytic_warmstart_theta_2nd`` /
# ``select_warmstart_theta`` above.


# ── Checkpoint / persistence helpers (re-exported from the framework core) ────
# The crash-safe checkpoint + resume machinery lives in
# qmbp_simulation.framework.study_checkpoint (pure, unit-tested, no scripts/
# import). Re-exported here so study scripts keep a single import surface.
# (atomic_write_json is intentionally NOT re-exported: callers that need it
# import it from the framework core directly; StudyCheckpoint uses it internally.)
from qmbp_simulation.framework.study_checkpoint import (  # noqa: E402
    StudyCheckpoint,
    read_json,
    resume_ordered_list,
)


def save_json_writer(subdir: str, filename: str, **meta):
    """Return a ``writer(payload) -> Path`` bound to save_json for a subdir/file.

    Adapter so :class:`StudyCheckpoint` can persist through the study's
    traceable ``save_json`` (injecting result_path/source_script/generated_utc)
    while staying agnostic of the results layout. ``meta`` is forwarded to every
    write (e.g. ``params=...``, ``description=...``).
    """
    def _writer(payload: dict):
        return save_json(payload, subdir, filename, **meta)

    return _writer


# ── Artifact organization: canonical taxonomy, naming, tagging, index ─────────
# The naming/tag/status/writer *mechanics* live in the framework core
# (qmbp_simulation.framework.study_artifacts, pure + unit-tested). Here we wire
# in the study's results-tree LAYOUT and re-export the surface so scripts keep a
# single import point.
from qmbp_simulation.framework.study_artifacts import (  # noqa: E402
    META_SCHEMA,
    STATUS_DEPRECATED,
    STATUS_FINAL,
    STATUS_PARTIAL,
    VALID_KINDS,
    VALID_STATUS,
    PathShim,
    StudyArtifactWriter,
    build_artifact_name,
    build_meta,
    infer_status_from_payload,
    meta_sidecar_path,
    promote_status,
    write_meta_sidecar,
)

# Canonical taxonomy (Fase A defines the roots; migration in Fase D populates
# them). NEW runs go under runs/ (single-point) or sweeps/ (multi-point
# aggregates); reports/, cache/, legacy/ complete the tree.
RUNS_ROOT = STUDY_ROOT / "runs"
SWEEPS_ROOT = STUDY_ROOT / "sweeps"
REPORTS_ROOT = STUDY_ROOT / "reports"
CACHE_ROOT = STUDY_ROOT / "cache"
LEGACY_ROOT = STUDY_ROOT / "legacy"
INDEX_JSON = STUDY_ROOT / "index.json"
INDEX_MD = STUDY_ROOT / "INDEX.md"


def run_dir(experiment: str, run_id: str, *, kind: str = "run") -> Path:
    """Return (and create) the canonical directory for a run or sweep.

    ``kind="run"`` -> ``runs/<experiment>/<run_id>/`` (single-point results);
    ``kind="sweep"`` -> ``sweeps/<experiment>/<run_id>/`` (multi-point
    aggregates). Both hold ``final.json`` / ``partials/`` / ``artifacts/`` /
    ``meta.json`` per the taxonomy documented in README.md.
    """
    base = SWEEPS_ROOT if kind == "sweep" else RUNS_ROOT
    d = base / experiment / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def study_artifact_writer(experiment: str, run_id: str, *, kind: str = "run",
                          source_script: str | None = None) -> StudyArtifactWriter:
    """Build a layout-aware :class:`StudyArtifactWriter` for the study tree.

    Resolves ``run_dir`` under the canonical taxonomy and injects the repo root
    (for git provenance) and the caller script (traceability). This is the
    concrete writer the ``StudyRunner`` persistence step plugs into for NEW
    runs; existing scripts keep using ``save_json`` until migrated.
    """
    return StudyArtifactWriter(
        experiment=experiment,
        run_dir=run_dir(experiment, run_id, kind=kind),
        run_id=run_id,
        source_script=source_script or _caller_script(),
        repo_root=_REPO_ROOT,
    )
