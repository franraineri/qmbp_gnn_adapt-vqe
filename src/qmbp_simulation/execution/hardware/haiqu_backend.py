"""Haiqu SDK execution backend for HVA circuits on real QPU.

This module wraps the Haiqu cloud middleware stack (state compression +
automatic error mitigation + QPU execution) behind the project's
``ExecutionBackend`` ABC, so a GNN-predicted HVA circuit can be run on
hardware through the same contract used everywhere else.

Canonical Haiqu workflow (from docs.haiqu.ai, confirmed against SDK v1.4):

    1. haiqu.login(api_access_key=...)     # once per session
    2. haiqu.init("experiment name")        # organizes runs on the dashboard
    3. compressed, quality = haiqu.state_compression(circuit=bound).result()
    4. job = haiqu.run(circuits=..., observables=..., use_mitigation=True)
    5. energy = reconstruct(job.result())

Design decisions grounded in the official API:

- **Observable mode for energy.** For ⟨H⟩ estimation (VQE / ground state),
  the circuit is submitted WITHOUT terminal measurements together with a list
  of ``SparsePauliOp`` observables and ``use_mitigation=True``. Haiqu selects
  the expectation-value mitigation pipeline automatically (2x circuit/shot
  overhead). We reconstruct ⟨H⟩ = Σ_k coef_k · ⟨P_k⟩ from the returned EVs.

- **θ passed as ``parameters`` (no manual bind).** ``haiqu.run`` binds a
  ``ParameterVector`` positionally in vector-index order (θ[0], θ[1], ...),
  which is exactly the ordering produced by ``HVACircuitBuilder``. This lets us
  sweep every h in a single submission via nested ``parameters``.

- **Compression needs a BOUND circuit.** ``state_compression`` compresses the
  action on the all-zero input state, so parameters must be assigned first.
  When compressing, we bind θ then compress per h-value.

- **noise_profile matches the device.** ibm_kingston is a Heron R2 chip, so
  the default ``noise_profile="ibm_heron_r2"`` tunes compression for the right
  noise characteristics.

- **Lazy SDK import.** ``haiqu`` is imported inside methods so the project
  imports cleanly even when ``haiqu-sdk`` is not installed.

References
----------
- https://docs.haiqu.ai/reference/run/run
- https://docs.haiqu.ai/reference/middleware/compression
- https://docs.haiqu.ai/core_features/error_shield
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qmbp_simulation.execution.backends import ExecutionBackend
from qmbp_simulation.utils.helpers import json_dump, json_serialize

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

logger = logging.getLogger(__name__)


# Devices whose native basis / noise profile map to a Haiqu noise_profile.
# Used to auto-select the compression noise profile from a backend name.
_DEVICE_NOISE_PROFILE = {
    "ibm_kingston": "ibm_heron_r2",
    "ibm_torino": "ibm_heron_r1",
    "ibm_fez": "ibm_heron_r2",
    "ibm_marrakesh": "ibm_heron_r2",
    "ibm_brisbane": "ibm_eagle_r3",
    "ibm_sherbrooke": "ibm_eagle_r3",
}


def _scale_qpu_cost(cost: Any, factor: int) -> Any:
    """Scale a Haiqu estimated_qpu_cost payload by an integer factor.

    Multiplies every ``amount`` under the known cost dicts (top-level and the
    ``haiqu``/``standard`` variants) so a multi-start estimate reflects the total
    across all optimization runs. Non-dict / unrecognized payloads pass through.
    """
    if factor == 1 or not isinstance(cost, dict):
        return cost

    def _scale_leaf(d: Any) -> Any:
        if isinstance(d, dict) and "amount" in d and isinstance(d["amount"], (int, float)):
            return {**d, "amount": d["amount"] * factor}
        return d

    out = dict(cost)
    for key in ("native", "converted"):
        if key in out:
            out[key] = _scale_leaf(out[key])
    for variant in ("haiqu", "standard"):
        if isinstance(out.get(variant), dict):
            v = dict(out[variant])
            for key in ("native", "converted"):
                if key in v:
                    v[key] = _scale_leaf(v[key])
            out[variant] = v
    return out


@dataclass
class HaiquConfig:
    """Configuration for the Haiqu execution backend.

    Defaults are tuned for the validated thesis deployment target:
    heavy_hex N=10 p=1 on ibm_kingston (Heron R2).

    Parameters
    ----------
    device_id : str
        Haiqu device identifier. Use ``"fake_torino"`` / ``"aer_simulator"``
        for credential-free development, or a real device such as
        ``"ibm_kingston"`` for hardware runs.
    shots : int
        Shots per circuit execution.
    use_mitigation : bool
        Enable Haiqu Error Shield (automatic multi-layer error mitigation).
        Recommended True for real QPU, False for noiseless simulators.
    use_compression : bool
        Apply ``haiqu.state_compression`` before execution. Recommended for
        deep circuits (p >= 2 or large N); for shallow p=1 N=10 the overhead
        may not be worth it.
    compression_level : {"low", "balanced", "high"}
        Compression aggressiveness. ``balanced`` is the documented default.
    fine_tuning : {"disabled", "low", "heavy"}
        Classical post-compression optimization. ``low`` is the default.
    noise_profile : str | None
        Compression noise profile. When ``None``, auto-selected from
        ``device_id`` via ``_DEVICE_NOISE_PROFILE`` (falls back to
        ``"ibm_heron_r2"``).
    approximation_level : int | None
        Advanced compression knob (1..8). ``None`` = auto from noise_profile.
    error_mitigation_options : dict | None
        Per-component mitigation overrides passed under
        ``options["error_mitigation_options"]``. Only takes effect when
        ``use_mitigation=True``. Keys: ``dynamical_decoupling``,
        ``readout_mitigation``, ``noise_tailoring``, ``advanced_mitigation``.
    experiment_name : str
        Name used for ``haiqu.init`` (dashboard grouping).
    ibm_token : str | None
        IBM Quantum token. When ``None``, read from ``IBM_KEY`` env var and
        passed via the run ``options`` dict (only needed for real IBM devices).
    ibm_instance : str | None
        IBM instance CRN. When ``None``, read from ``IBM_INSTANCE_CRN`` env var.
    api_access_key : str | None
        Haiqu API key. When ``None``, read from ``HAIQU_API_KEY`` env var; if
        that is also unset, ``haiqu.login()`` is called with no argument
        (works inside the pre-configured Haiqu Lab environment).
    """

    device_id: str = "ibm_kingston"
    shots: int = 16384
    use_mitigation: bool = True
    use_compression: bool = False
    compression_level: Literal["low", "balanced", "high"] = "balanced"
    fine_tuning: Literal["disabled", "low", "heavy"] = "low"
    noise_profile: str | None = None
    approximation_level: int | None = None
    error_mitigation_options: dict[str, bool] | None = None
    default_hardware_mitigation: bool = True
    """When ``error_mitigation_options`` is unset and the target is a real QPU,
    apply the twirling-forward preset instead of the SDK default. The SDK leaves
    ``noise_tailoring`` (Pauli twirling) OFF by default, but twirling is what
    unlocks the mitigation gain on hardware; this preset turns it ON while
    keeping readout mitigation OFF (it OOMs the cloud worker at large N)."""
    experiment_name: str = "GNN-HVA hardware deployment"
    run_max_retries: int = 2  # retry a failed haiqu.run job (transient runtime OOM/kill)
    # Server-side variational optimization (haiqu.variational_optimization).
    vqe_optimizer: Literal["nft", "cobyla", "nelder-mead", "powell", "cobyqa"] = "nft"
    vqe_maxfev: int = 200  # max circuit evaluations (server-side)
    vqe_maxiter: int = 100  # max parameter-update iterations (NFT) / scipy iterations
    vqe_restarts: int = 2
    """Extra optimization restarts beyond the GNN warm-start. Total optimizations
    run = 1 (warm-started from the predicted θ) + vqe_restarts (each seeded to a
    random θ in [-0.1π, 0.1π] by the SDK); the lowest-min_loss result wins. Set 0
    to keep the single warm-started run."""
    ibm_token: str | None = None
    ibm_instance: str | None = None
    api_access_key: str | None = None
    max_compression_time_s: int = 1200
    sidecar_path: str | None = None
    """Append-only JSONL sidecar. When set, EVERY operation record (cost
    estimate, run, refinement, evaluation) is flushed here the instant it is
    produced — so a crash mid-run still leaves the job ids and any collected
    energy on disk, recoverable without the final ``save_collected_data``."""

    def resolved_noise_profile(self) -> str:
        """Return the compression noise profile, auto-selecting if unset.

        Falls back to ``"ibm_heron_r2"`` for a device with no explicit mapping;
        ``noise_profile_is_fallback`` reports whether that fallback was used so
        callers can warn.
        """
        if self.noise_profile is not None:
            return self.noise_profile
        return _DEVICE_NOISE_PROFILE.get(self.device_id, "ibm_heron_r2")

    def noise_profile_is_fallback(self) -> bool:
        """True when the compression noise profile is the unmapped-device fallback."""
        return self.noise_profile is None and self.device_id not in _DEVICE_NOISE_PROFILE

    def is_simulator(self) -> bool:
        """True when the target device is a simulator / fake backend."""
        return self.device_id.startswith(("fake_", "aer_", "ionq_sim"))


class HaiquBackend(ExecutionBackend):
    """Execute HVA circuits on QPU via the Haiqu cloud stack.

    Implements the ``ExecutionBackend`` contract so it can be dropped into any
    optimizer or runner that accepts a backend. The energy is estimated in
    Haiqu's observable-mitigation mode by decomposing the Hamiltonian into its
    Pauli terms and reconstructing ⟨H⟩ from mitigated expectation values.

    Notes
    -----
    - ``login``/``init`` happen lazily on first use and only once.
    - Circuits are submitted WITHOUT terminal measurements (observable mode).
    - When ``use_compression=True`` the circuit is bound and compressed per
      call before submission (compression requires a bound circuit).
    """

    def __init__(self, config: HaiquConfig | None = None) -> None:
        self._config = config or HaiquConfig()
        self._session_ready = False
        self._last_uncertainty: float | None = None
        self._last_qpu_cost: Any = None
        # Full data collection: every Haiqu operation appends a rich record.
        self._records: list[dict[str, Any]] = []

    # ─── Session management ──────────────────────────────────────────────

    def _ensure_session(self) -> None:
        """Log in and init the experiment exactly once."""
        if self._session_ready:
            return
        haiqu = self._import_haiqu()
        key = self._config.api_access_key or os.environ.get("HAIQU_API_KEY")
        if key:
            haiqu.login(api_access_key=key.strip())
        else:
            # Pre-configured Haiqu Lab environment: key is implicit.
            haiqu.login()
        haiqu.init(self._config.experiment_name)
        self._session_ready = True
        logger.info("Haiqu session ready (experiment=%r)", self._config.experiment_name)

    @staticmethod
    def _import_haiqu():
        """Lazy import of the Haiqu SDK with a helpful error message."""
        try:
            from haiqu.sdk import haiqu
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "haiqu-sdk is required for HaiquBackend. Install with: pip install haiqu-sdk"
            ) from exc
        return haiqu

    def _run_options(self) -> dict[str, Any]:
        """Build the ``options`` dict for ``haiqu.run`` (credentials + EM).

        Credentials are NEVER stored in the collected records — only key
        *names* are referenced, never their values.
        """
        options: dict[str, Any] = {}
        if not self._config.is_simulator():
            token = self._config.ibm_token or os.environ.get("IBM_KEY")
            instance = self._config.ibm_instance or os.environ.get("IBM_INSTANCE_CRN")
            if token:
                options["ibm_quantum_token"] = token
            if instance:
                options["ibm_quantum_instance"] = instance
        emo = self._resolve_mitigation_options()
        if self._config.use_mitigation and emo:
            options["error_mitigation_options"] = emo
        return options

    def _resolve_mitigation_options(self) -> dict[str, bool] | None:
        """Resolve the Error-Shield options dict actually sent to Haiqu.

        An explicit ``error_mitigation_options`` always wins. Otherwise, on a
        real QPU with ``default_hardware_mitigation`` set, apply the
        twirling-forward preset (noise_tailoring ON, readout_mitigation OFF): the
        SDK default leaves twirling off, but twirling is what unlocks the gain on
        hardware, and readout mitigation OOMs the cloud worker at large N. On
        simulators, return None so the SDK default applies.
        """
        if self._config.error_mitigation_options is not None:
            return dict(self._config.error_mitigation_options)
        if self._config.default_hardware_mitigation and not self._config.is_simulator():
            return {
                "noise_tailoring": True,
                "advanced_mitigation": True,
                "dynamical_decoupling": False,
                "readout_mitigation": False,
            }
        return None

    # ─── Data collection helpers ─────────────────────────────────────────

    @staticmethod
    def _harvest_job_metadata(job: Any) -> dict[str, Any]:
        """Extract EVERY available attribute from a Haiqu job object.

        Best-effort and defensive: Haiqu's job schema may evolve, so every
        field is guarded. Captures job id/status, ``info`` (uncertainty,
        qpu_cost, ...), logs, timing, and any other public scalar attributes.
        """
        meta: dict[str, Any] = {}
        for attr in (
            "id",
            "job_id",
            "status",
            "name",
            "description",
            "device_id",
            "creation_date",
            "run_date",
            "finish_date",
            "time",
            "quality",
            "logs",
            "estimated_qpu_cost",
        ):
            try:
                val = getattr(job, attr, None)
            except Exception:  # noqa: BLE001 - defensive against SDK internals
                val = None
            if val is not None and not callable(val):
                meta[attr] = val
        # ``info`` is a dict of auxiliary metadata (uncertainty, qpu_cost, ...)
        info = getattr(job, "info", None)
        if isinstance(info, dict):
            meta["info"] = dict(info)
        return meta

    @staticmethod
    def _circuit_stats(circuit: Any) -> dict[str, Any]:
        """Capture depth / gate-count stats for provenance.

        Defensive against non-Qiskit circuit objects (e.g. Haiqu's
        ``CircuitModel`` returned by state_compression), which lack the Qiskit
        introspection API — in that case stats are reported as unavailable.
        """
        if not hasattr(circuit, "count_ops") or not hasattr(circuit, "depth"):
            return {
                "n_qubits": getattr(circuit, "num_qubits", None),
                "num_parameters": getattr(circuit, "num_parameters", None),
                "note": f"stats unavailable for {type(circuit).__name__}",
            }
        try:
            depth_2q = circuit.depth(lambda ins: len(ins.qubits) == 2)
        except Exception:  # noqa: BLE001
            depth_2q = None
        try:
            ops = dict(circuit.count_ops())
        except Exception:  # noqa: BLE001
            ops = {}
        n_2q = sum(v for op, v in ops.items() if op in ("cx", "cz", "ecr", "rzz", "rxx", "ryy"))
        # PauliEvolutionGate is not yet decomposed into 2Q gates; flag when the
        # count is pre-transpilation so downstream analysis is not misled.
        has_high_level = any(op in ops for op in ("PauliEvolution", "hamiltonian", "unitary"))
        return {
            "n_qubits": circuit.num_qubits,
            "num_parameters": circuit.num_parameters,
            "depth": circuit.depth(),
            "depth_2q": depth_2q,
            "gate_counts": ops,
            "n_2q_gates": n_2q,
            "pre_transpilation": bool(has_high_level),
        }

    def _record(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Append a timestamped record to the in-memory collection.

        Also flushes the record to the append-only sidecar (if configured) the
        instant it is produced, so job ids and collected energies survive a
        crash before ``save_collected_data``.
        """
        rec = {
            "operation": operation,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "device_id": self._config.device_id,
            **payload,
        }
        self._records.append(rec)
        self._flush_sidecar(rec)
        return rec

    def _flush_sidecar(self, rec: dict[str, Any]) -> None:
        """Append one record to the JSONL sidecar (best-effort, never raises).

        Backup must not break a run: a sidecar I/O error is logged and swallowed
        so the QPU work in progress is never lost to a logging failure.
        """
        path = self._config.sidecar_path
        if not path:
            return
        try:
            import json as _json
            from pathlib import Path as _Path

            p = _Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as f:
                f.write(_json.dumps(rec, default=str) + "\n")
        except Exception as exc:  # noqa: BLE001 - never abort a run over logging
            logger.warning("sidecar flush failed (%s): %s", type(exc).__name__, exc)

    @property
    def records(self) -> list[dict[str, Any]]:
        """All collected operation records (compression, run, cost, evaluate)."""
        return self._records

    def _submit_with_retry(self, label: str, submit):
        """Run a Haiqu job submission with retries + GC between attempts.

        The project's VQE optimizer disables Python GC during optimization
        (mimalloc deadlock workaround on macOS ARM64). Repeated cloud submissions
        under a disabled GC can accumulate memory and get the Haiqu runtime job
        killed ("terminated by the runtime ... OOM or forced kill"). We defend by
        forcing a ``gc.collect()`` before each attempt and retrying transient
        failures. ``submit`` is a zero-arg callable returning the job handle;
        ``label`` names the operation for log/error messages.
        """
        import gc as _gc

        last_exc: Exception | None = None
        attempts = self._config.run_max_retries + 1
        for attempt in range(1, attempts + 1):
            _gc.collect()
            try:
                return submit()
            except Exception as exc:  # noqa: BLE001 - Haiqu raises bare Exception
                last_exc = exc
                logger.warning(
                    "%s failed (attempt %d/%d): %s",
                    label,
                    attempt,
                    attempts,
                    str(exc)[:160],
                )
        raise RuntimeError(f"{label} failed after {attempts} attempts: {last_exc}") from last_exc

    def _submit_run_with_retry(
        self,
        haiqu: Any,
        *,
        circuits: Any,
        observables: list[SparsePauliOp],
        parameters: list[list[float]] | None,
        use_mitigation: bool,
        pack_size: int | None = None,
    ) -> tuple[Any, Any]:
        """Submit a haiqu.run job (observable mode) with retries; returns (job, result).

        When ``pack_size`` is set (>= 2) the run is packed: the SDK replicates the
        circuit onto unused device qubits so several parameter points run in
        parallel while paying shots once, cutting QPU cost.
        """
        packing = {"use_packing": True, "pack_size": pack_size} if pack_size else {}

        def _submit():
            job = haiqu.run(
                circuits=circuits,
                observables=observables,
                parameters=parameters,
                device_id=self._config.device_id,
                shots=self._config.shots,
                options=self._run_options(),
                use_mitigation=use_mitigation,
                **packing,
            )
            return job, job.result()

        return self._submit_with_retry("haiqu.run", _submit)

    def _auto_pack_size(self, n_qubits: int, n_points: int) -> int | None:
        """Auto-select a pack size for a multi-point run, or None if not worth it.

        Packing replicates the circuit on unused device qubits so several
        parameter points run in parallel for the price of one shot budget. We
        pack into at most 2/3 of the device (the SDK's own default ceiling) and
        never more copies than points to evaluate. Returns None for a single
        point or when only one copy fits (no benefit).
        """
        if n_points < 2 or n_qubits < 1:
            return None
        device_qubits = self._device_qubit_count()
        if device_qubits is None:
            return None
        max_copies = int((2 * device_qubits) // (3 * n_qubits))
        pack = min(max_copies, n_points)
        return pack if pack >= 2 else None

    def _device_qubit_count(self) -> int | None:
        """Best-effort qubit count of the target device (None on simulator/unknown)."""
        if self._config.is_simulator():
            return None
        try:
            haiqu = self._import_haiqu()
            self._ensure_session()
            dev = haiqu.get_device(self._config.device_id)
            return getattr(dev, "qubits", None)
        except Exception:  # noqa: BLE001 - device lookup is best-effort
            return None

    # ─── Circuit compression ─────────────────────────────────────────────

    def compress_circuit(
        self,
        circuit: QuantumCircuit,
        params: np.ndarray | None = None,
    ) -> tuple[QuantumCircuit, float]:
        """Compress a circuit via ``haiqu.state_compression``.

        Parameters
        ----------
        circuit : QuantumCircuit
            HVA circuit. If parameterized, ``params`` must be provided (state
            compression requires a bound circuit).
        params : np.ndarray | None
            Parameter values to bind before compression. Required when the
            circuit still has free parameters.

        Returns
        -------
        (compressed_circuit, quality)
            ``quality`` is Haiqu's fidelity-like approximation metric in [0, 1].
        """
        self._ensure_session()
        haiqu = self._import_haiqu()

        bound = circuit
        if circuit.num_parameters > 0:
            if params is None:
                raise ValueError(
                    "compress_circuit: circuit has free parameters but no params "
                    "were provided. State compression requires a bound circuit."
                )
            bound = circuit.assign_parameters(np.asarray(params, dtype=float))

        # State compression tunes for a device noise profile, which is meaningless
        # on a simulator/fake backend — skip it and return the bound circuit so a
        # sim run is not shaped by an inapplicable hardware profile.
        if self._config.is_simulator():
            logger.info(
                "compress_circuit: skipping state_compression on simulator %r "
                "(noise-profile compression is hardware-only).",
                self._config.device_id,
            )
            return bound, float("nan")
        if self._config.noise_profile_is_fallback():
            logger.warning(
                "compress_circuit: device %r has no mapped noise profile; using "
                "fallback %r. Set config.noise_profile explicitly for accurate "
                "compression tuning.",
                self._config.device_id,
                self._config.resolved_noise_profile(),
            )

        t0 = time.perf_counter()

        def _submit_compression():
            job = haiqu.state_compression(
                circuit=bound,
                compression_level=self._config.compression_level,
                noise_profile=self._config.resolved_noise_profile(),
                fine_tuning=self._config.fine_tuning,
                approximation_level=self._config.approximation_level,
                max_time=self._config.max_compression_time_s,
            )
            return job, job.result()

        job, raw = self._submit_with_retry("haiqu.state_compression", _submit_compression)
        wall_s = time.perf_counter() - t0

        # SDK compatibility:
        #   1.4.x: job.result() -> (compressed_circuit, quality)
        #   1.5.x: job.result() -> CircuitModel; quality lives on job.quality
        #          (and job.info: compression_status/percent/time).
        if isinstance(raw, tuple) and len(raw) == 2:
            compressed, quality = raw
        else:
            compressed = raw
            quality = getattr(job, "quality", None)
        quality = float(quality) if quality is not None else float("nan")

        stats_before = self._circuit_stats(bound)
        stats_after = self._circuit_stats(compressed)
        n2q_before = stats_before.get("n_2q_gates") or 0
        n2q_after = stats_after.get("n_2q_gates") or 0
        # cnot_reduction only meaningful when both counts are known (>0).
        cnot_reduction = 1.0 - (n2q_after / n2q_before) if (n2q_before and n2q_after) else None

        self._record(
            "state_compression",
            {
                "compression_level": self._config.compression_level,
                "fine_tuning": self._config.fine_tuning,
                "noise_profile": self._config.resolved_noise_profile(),
                "approximation_level": self._config.approximation_level,
                "quality": quality,
                "cnot_reduction": cnot_reduction,
                "circuit_before": stats_before,
                "circuit_after": stats_after,
                "wall_clock_s": wall_s,
                "job_metadata": self._harvest_job_metadata(job),
            },
        )
        logger.info(
            "state_compression: level=%s fine_tuning=%s profile=%s quality=%s "
            "2Q %s->%s (%.1f%% reduction) in %.2fs",
            self._config.compression_level,
            self._config.fine_tuning,
            self._config.resolved_noise_profile(),
            f"{quality:.4f}",
            n2q_before,
            n2q_after,
            (cnot_reduction or 0.0) * 100,
            wall_s,
        )
        return compressed, quality

    # ─── ExecutionBackend ABC ────────────────────────────────────────────

    def evaluate(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        params: np.ndarray,
    ) -> float:
        """Return the (mitigated) energy ⟨H⟩ for the given parameters.

        The Hamiltonian is decomposed into Pauli terms; each term's
        expectation value is measured in Haiqu observable-mitigation mode and
        recombined as ⟨H⟩ = Σ_k coef_k · ⟨P_k⟩.

        Parameters
        ----------
        circuit : QuantumCircuit
            Parameterized HVA circuit WITHOUT terminal measurements. If it
            contains measurements they are removed before submission.
        hamiltonian : SparsePauliOp
            The Hamiltonian to evaluate.
        params : np.ndarray
            θ values, ordered to match the circuit's ParameterVector.

        Returns
        -------
        float
            Reconstructed energy expectation value.
        """
        self._ensure_session()
        haiqu = self._import_haiqu()

        params = np.asarray(params, dtype=float)

        # Observable mode: submit without terminal measurements.
        exec_circuit = circuit
        if circuit.num_clbits > 0:
            exec_circuit = circuit.remove_final_measurements(inplace=False)

        coeffs, observables = self._hamiltonian_to_observables(hamiltonian)

        # Optional compression (requires bound circuit → recombine params).
        run_parameters: list[list[float]] | None
        compression_quality: float | None = None
        if self._config.use_compression:
            compressed, compression_quality = self.compress_circuit(exec_circuit, params)
            exec_circuit = compressed
            run_parameters = None  # circuit is now bound
        else:
            run_parameters = [params.tolist()] if exec_circuit.num_parameters > 0 else None

        effective_mitigation = self._config.use_mitigation and not self._config.is_simulator()

        t0 = time.perf_counter()
        job, result = self._submit_run_with_retry(
            haiqu,
            circuits=exec_circuit,
            observables=observables,
            parameters=run_parameters,
            use_mitigation=effective_mitigation,
        )
        wall_s = time.perf_counter() - t0

        exp_values = self._extract_expectation_values(result, n_observables=len(observables))
        energy = float(np.dot(coeffs, exp_values))

        # Capture auxiliary metadata for callers that want error bars / cost.
        job_meta = self._harvest_job_metadata(job)
        info = job_meta.get("info", {})
        self._last_uncertainty = info.get("uncertainty") if isinstance(info, dict) else None
        self._last_qpu_cost = info.get("qpu_cost") if isinstance(info, dict) else None

        pauli_labels = [str(hamiltonian.paulis[k]) for k in range(len(observables))]
        self._record(
            "run",
            {
                "energy": energy,
                "uncertainty": self._last_uncertainty,
                "qpu_cost": self._last_qpu_cost,
                "shots": self._config.shots,
                "use_mitigation": effective_mitigation,
                "error_mitigation_options": (
                    self._resolve_mitigation_options() if effective_mitigation else None
                ),
                "use_compression": self._config.use_compression,
                "compression_quality": compression_quality,
                "n_observables": len(observables),
                "hamiltonian_coeffs": coeffs.tolist(),
                "pauli_terms": pauli_labels,
                "expectation_values": exp_values.tolist(),
                "term_contributions": (coeffs * exp_values).tolist(),
                "circuit_submitted": self._circuit_stats(exec_circuit),
                "wall_clock_s": wall_s,
                "job_metadata": job_meta,
            },
        )
        logger.info(
            "HaiquBackend.evaluate: device=%s energy=%.6f uncertainty=%s (%d obs, "
            "mitigation=%s) in %.2fs",
            self._config.device_id,
            energy,
            self._last_uncertainty,
            len(observables),
            effective_mitigation,
            wall_s,
        )
        return energy

    def evaluate_batch(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        params_list: list[np.ndarray] | np.ndarray,
    ) -> list[float]:
        """Evaluate several θ points for the SAME circuit in ONE packed run.

        A multi-point evaluation is exactly the case that maps to multiple QPU
        jobs; here it is submitted as a single ``haiqu.run`` parameter sweep with
        packing turned on automatically, so the points run in parallel on unused
        device qubits for one shot budget instead of one job per point. A
        single-point list falls back to ``evaluate`` (no packing, unchanged).

        Compression is skipped in batch mode because it needs a bound circuit
        (one per point), which would defeat the single-submission batching.

        Returns one reconstructed energy per input θ, in order.
        """
        arr = [np.asarray(p, dtype=float) for p in params_list]
        if len(arr) == 0:
            return []
        if len(arr) == 1:
            return [self.evaluate(circuit, hamiltonian, arr[0])]

        self._ensure_session()
        haiqu = self._import_haiqu()

        exec_circuit = circuit
        if circuit.num_clbits > 0:
            exec_circuit = circuit.remove_final_measurements(inplace=False)
        if exec_circuit.num_parameters == 0:
            raise ValueError("evaluate_batch needs a parameterized circuit to sweep multiple θ.")

        coeffs, observables = self._hamiltonian_to_observables(hamiltonian)
        run_parameters = [p.tolist() for p in arr]
        effective_mitigation = self._config.use_mitigation and not self._config.is_simulator()
        pack_size = self._auto_pack_size(exec_circuit.num_qubits, len(arr))

        t0 = time.perf_counter()
        job, result = self._submit_run_with_retry(
            haiqu,
            circuits=exec_circuit,
            observables=observables,
            parameters=run_parameters,
            use_mitigation=effective_mitigation,
            pack_size=pack_size,
        )
        wall_s = time.perf_counter() - t0

        # Result nesting for one circuit, many observables, many params:
        # [circuit][observable][parameter]; recombine ⟨H⟩ per parameter point.
        res = np.asarray(result, dtype=object)
        ev = np.array(res.flatten().tolist(), dtype=float).flatten()
        n_obs, n_pts = len(observables), len(arr)
        if ev.size != n_obs * n_pts:
            raise ValueError(
                f"Expected {n_obs}×{n_pts}={n_obs * n_pts} expectation values from "
                f"the packed sweep, got {ev.size} (raw shape={res.shape})."
            )
        ev = ev.reshape(n_obs, n_pts)  # [observable][parameter]
        energies = [float(np.dot(coeffs, ev[:, j])) for j in range(n_pts)]

        job_meta = self._harvest_job_metadata(job)
        self._record(
            "run_batch",
            {
                "n_points": n_pts,
                "energies": energies,
                "pack_size": pack_size,
                "use_packing": bool(pack_size),
                "shots": self._config.shots,
                "use_mitigation": effective_mitigation,
                "n_observables": n_obs,
                "circuit_submitted": self._circuit_stats(exec_circuit),
                "wall_clock_s": wall_s,
                "job_metadata": job_meta,
            },
        )
        logger.info(
            "HaiquBackend.evaluate_batch: %d points in ONE run (pack_size=%s, "
            "mitigation=%s) on %s in %.2fs",
            n_pts,
            pack_size,
            effective_mitigation,
            self._config.device_id,
            wall_s,
        )
        return energies

    @property
    def name(self) -> str:
        return f"haiqu_{self._config.device_id}"

    # ─── Cost estimation (dry run) ───────────────────────────────────────

    def estimate_cost(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        params: np.ndarray,
    ) -> Any:
        """Estimate QPU cost without executing (``dry_run=True``).

        Returns Haiqu's ``job.estimated_qpu_cost`` payload. Spends no credits.
        """
        self._ensure_session()
        haiqu = self._import_haiqu()

        params = np.asarray(params, dtype=float)
        exec_circuit = circuit
        if circuit.num_clbits > 0:
            exec_circuit = circuit.remove_final_measurements(inplace=False)
        _coeffs, observables = self._hamiltonian_to_observables(hamiltonian)
        run_parameters = [params.tolist()] if exec_circuit.num_parameters > 0 else None

        job = haiqu.run(
            circuits=exec_circuit,
            observables=observables,
            parameters=run_parameters,
            device_id=self._config.device_id,
            shots=self._config.shots,
            options=self._run_options(),
            use_mitigation=self._config.use_mitigation and not self._config.is_simulator(),
            dry_run=True,
        )
        # Some SDK builds do not populate estimated_qpu_cost; read defensively so
        # a missing field returns None instead of raising on the caller.
        cost = getattr(job, "estimated_qpu_cost", None)
        self._record(
            "dry_run_cost_estimate",
            {
                "estimated_qpu_cost": cost,
                "shots": self._config.shots,
                "n_observables": len(observables),
                "circuit_submitted": self._circuit_stats(exec_circuit),
                "job_metadata": self._harvest_job_metadata(job),
            },
        )
        if cost is None:
            logger.warning(
                "HaiquBackend.estimate_cost: SDK returned no estimated_qpu_cost "
                "(build does not populate it); no QPU-execution job was submitted."
            )
        else:
            logger.info("HaiquBackend.estimate_cost: %s", cost)
        return cost

    def estimate_cost_variational(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        initial_params: np.ndarray,
    ) -> Any:
        """Estimate QPU cost of a SERVER-SIDE VQE refinement (``dry_run=True``).

        When the deployment will refine θ via ``refine_variational``, the real
        cost is the optimization loop (up to ``vqe_maxfev`` evaluations), not a
        single ``run``. This mirrors ``refine_variational``'s submission with
        ``dry_run=True`` so the estimate reflects the optimizer path. Spends no
        credits.
        """
        self._ensure_session()
        haiqu = self._import_haiqu()

        from haiqu.sdk.qml.optimizer import (
            NFTOptimizerOptions,
            ScipyOptimizerOptions,
        )
        from haiqu.sdk.qml.problem import VariationalProblem

        initial_params = np.asarray(initial_params, dtype=float)
        exec_circuit = circuit
        if circuit.num_clbits > 0:
            exec_circuit = circuit.remove_final_measurements(inplace=False)

        problem = VariationalProblem(ansatz=exec_circuit, observable=hamiltonian)
        if self._config.vqe_optimizer == "nft":
            optimizer_options = NFTOptimizerOptions(
                maxfev=self._config.vqe_maxfev,
                maxiter=self._config.vqe_maxiter,
            )
        else:
            optimizer_options = ScipyOptimizerOptions(
                method=self._config.vqe_optimizer,
                maxfev=self._config.vqe_maxfev,
                options={"maxiter": self._config.vqe_maxiter},
            )

        job = haiqu.variational_optimization(
            problem,
            shots=self._config.shots,
            device_id=self._config.device_id,
            options=self._run_options() or None,
            initial_parameters=initial_params.tolist(),
            optimizer_options=optimizer_options,
            use_mitigation=self._config.use_mitigation and not self._config.is_simulator(),
            use_compression=self._config.use_compression,
            dry_run=True,
        )
        cost_per_start = getattr(job, "estimated_qpu_cost", None)
        # Multi-start runs 1 warm + vqe_restarts seeded optimizations, so the total
        # QPU cost is n_starts × the single-optimization estimate.
        n_starts = 1 + max(0, int(self._config.vqe_restarts))
        cost = _scale_qpu_cost(cost_per_start, n_starts)
        self._record(
            "dry_run_cost_estimate_variational",
            {
                "estimated_qpu_cost": cost,
                "estimated_qpu_cost_per_start": cost_per_start,
                "n_starts": n_starts,
                "restarts": self._config.vqe_restarts,
                "optimizer": self._config.vqe_optimizer,
                "maxfev": self._config.vqe_maxfev,
                "maxiter": self._config.vqe_maxiter,
                "shots": self._config.shots,
                "circuit_submitted": self._circuit_stats(exec_circuit),
                "job_metadata": self._harvest_job_metadata(job),
            },
        )
        if cost is None:
            logger.warning(
                "HaiquBackend.estimate_cost_variational: SDK returned no "
                "estimated_qpu_cost; no job submitted."
            )
        else:
            logger.info(
                "HaiquBackend.estimate_cost_variational (optimizer=%s, maxfev=%d, n_starts=%d): %s",
                self._config.vqe_optimizer,
                self._config.vqe_maxfev,
                n_starts,
                cost,
            )
        return cost

    @property
    def last_uncertainty(self) -> float | None:
        """Statistical uncertainty reported by the last observable run."""
        return self._last_uncertainty

    # ─── Full data collection (energy + physics + all Haiqu metadata) ────

    def evaluate_full(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        params: np.ndarray,
        *,
        h: float | None = None,
        e_exact: float | None = None,
        gap: float | None = None,
        exact_state: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run on QPU and return a complete data record for one h-point.

        Executes ``evaluate`` (which already captures every Haiqu run metric)
        and augments it with derived physics: |ΔE|, ΔE/gap, and — when an exact
        statevector is provided and N is tractable — the state fidelity.

        Parameters
        ----------
        circuit, hamiltonian, params
            Same as ``evaluate``.
        h : float | None
            Transverse-field value (for provenance / indexing).
        e_exact : float | None
            Exact ground-state energy. Enables |ΔE| and ΔE/gap.
        gap : float | None
            Spectral gap. Enables ΔE/gap.
        exact_state : np.ndarray | None
            Exact ground-state vector. Enables fidelity |⟨ψ_exact|ψ(θ)⟩|²
            (uses the ExecutionBackend.compute_fidelity default, statevector).

        Returns
        -------
        dict
            Complete record: energy, uncertainty, qpu_cost, wall-clock, all raw
            EVs, plus derived |ΔE|, ΔE/gap, fidelity, and the underlying Haiqu
            run record. Also appended to the collection under
            ``operation="evaluate_full"``.
        """
        params = np.asarray(params, dtype=float)
        energy = self.evaluate(circuit, hamiltonian, params)
        run_record = self._records[-1]  # the "run" record just appended

        abs_delta_e = abs(energy - e_exact) if e_exact is not None else None
        de_gap = abs_delta_e / gap if (abs_delta_e is not None and gap not in (None, 0)) else None

        fidelity: float | None = None
        if exact_state is not None:
            try:
                fidelity = self.compute_fidelity(circuit, params, exact_state)
            except Exception as exc:  # noqa: BLE001
                logger.warning("evaluate_full: fidelity computation failed: %s", exc)

        record = self._record(
            "evaluate_full",
            {
                "h": h,
                "theta_pred": params.tolist(),
                "energy": energy,
                "e_exact": e_exact,
                "gap": gap,
                "abs_delta_e": abs_delta_e,
                "de_gap": de_gap,
                "fidelity": fidelity,
                "uncertainty": self._last_uncertainty,
                "qpu_cost": self._last_qpu_cost,
                "pass_5pct": (de_gap is not None and de_gap < 0.05),
                "run_record": run_record,
            },
        )
        logger.info(
            "evaluate_full: h=%s E=%.6f |ΔE|=%s ΔE/gap=%s fidelity=%s",
            h,
            energy,
            None if abs_delta_e is None else f"{abs_delta_e:.4f}",
            None if de_gap is None else f"{de_gap:.2%}",
            None if fidelity is None else f"{fidelity:.4f}",
        )
        return record

    def refine_variational(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        initial_params: np.ndarray,
        *,
        h: float | None = None,
        job_name: str | None = None,
        job_description: str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Refine θ with Haiqu's SERVER-SIDE variational optimizer.

        Unlike a local VQE loop (one network round-trip per energy evaluation,
        and incompatible with the project's GC-disabled optimizer), this submits
        a single ``variational_optimization`` job. Haiqu runs the whole
        optimization loop in the cloud and streams back the full history — so it
        is tracked as one job on the dashboard, with the warm-start parameters,
        loss trajectory, and QPU cost recorded.

        The default optimizer is NFT (Nakanishi-Fujii-Todo), which is ideal for
        HVA-style ansätze: analytic per-parameter updates, requires each
        parameter to appear in exactly one rotation gate (satisfied by the
        bond-resolved HVA). Set ``config.vqe_optimizer`` to a scipy method
        (cobyla / nelder-mead / powell / cobyqa) to switch.

        Parameters
        ----------
        circuit : QuantumCircuit
            Parameterized ansatz (no terminal measurements needed).
        hamiltonian : SparsePauliOp
            The full Hamiltonian to minimize (single observable).
        initial_params : np.ndarray
            Warm-start θ (the GNN prediction).
        h : float | None
            Transverse-field value, for the record.
        job_name, job_description : str | None
            Passed to ``variational_optimization`` for dashboard tracking.

        Returns
        -------
        (theta_opt, record)
            ``theta_opt`` — optimized parameters (np.ndarray).
            ``record`` — the collected data record (also appended to
            ``self.records``), including loss_history, weights_history,
            min_loss, and qpu_cost.
        """
        self._ensure_session()
        haiqu = self._import_haiqu()

        from haiqu.sdk.qml.optimizer import (
            NFTOptimizerOptions,
            ScipyOptimizerOptions,
        )
        from haiqu.sdk.qml.problem import VariationalProblem

        initial_params = np.asarray(initial_params, dtype=float)

        exec_circuit = circuit
        if circuit.num_clbits > 0:
            exec_circuit = circuit.remove_final_measurements(inplace=False)

        problem = VariationalProblem(ansatz=exec_circuit, observable=hamiltonian)

        if self._config.vqe_optimizer == "nft":
            optimizer_options = NFTOptimizerOptions(
                maxfev=self._config.vqe_maxfev,
                maxiter=self._config.vqe_maxiter,
            )
        else:
            optimizer_options = ScipyOptimizerOptions(
                method=self._config.vqe_optimizer,
                maxfev=self._config.vqe_maxfev,
                options={"maxiter": self._config.vqe_maxiter},
            )

        effective_mitigation = self._config.use_mitigation and not self._config.is_simulator()

        def _run_one(*, warm_start: bool, seed: int | None, label: str):
            """Submit one variational_optimization run (warm-started or seeded)."""
            # initial_parameters and seed are mutually exclusive in the SDK: the
            # warm start passes the GNN θ; each restart passes a seed so the SDK
            # draws a random θ in [-0.1π, 0.1π].
            kw = {"initial_parameters": initial_params.tolist()} if warm_start else {"seed": seed}

            def _submit():
                job = haiqu.variational_optimization(
                    problem,
                    shots=self._config.shots,
                    device_id=self._config.device_id,
                    options=self._run_options() or None,
                    optimizer_options=optimizer_options,
                    use_mitigation=effective_mitigation,
                    use_compression=self._config.use_compression,
                    job_name=(f"{job_name}-{label}" if job_name else None),
                    job_description=job_description,
                    **kw,
                )
                return job, job.result()

            t0 = time.perf_counter()
            job, result = self._submit_with_retry(
                f"haiqu.variational_optimization[{label}]", _submit
            )
            return job, result, time.perf_counter() - t0

        # Multi-start: 1 warm-started run (GNN θ) + vqe_restarts seeded runs; keep
        # the lowest min_loss. A single-start (restarts=0) preserves prior behavior.
        n_restarts = max(0, int(self._config.vqe_restarts))
        attempts: list[dict[str, Any]] = []
        best = None  # (min_loss, theta_opt, job, result, wall_s, label)
        for idx in range(1 + n_restarts):
            warm = idx == 0
            label = "warm" if warm else f"restart{idx}"
            job, result, wall_s = _run_one(warm_start=warm, seed=idx, label=label)
            min_loss = float(result.min_loss)
            theta_i = np.asarray(result.optimal_parameters, dtype=float)
            attempts.append(
                {
                    "label": label,
                    "warm_start": warm,
                    "seed": None if warm else idx,
                    "min_loss": min_loss,
                    "wall_clock_s": wall_s,
                }
            )
            logger.info(
                "variational_optimization[%s]: min_loss=%.6f in %.2fs",
                label,
                min_loss,
                wall_s,
            )
            if best is None or min_loss < best[0]:
                best = (min_loss, theta_i, job, result, wall_s, label)

        min_loss, theta_opt, job, result, wall_s, best_label = best
        job_meta = self._harvest_job_metadata(job)

        record = self._record(
            "variational_optimization",
            {
                "h": h,
                "optimizer": self._config.vqe_optimizer,
                "maxfev": self._config.vqe_maxfev,
                "maxiter": self._config.vqe_maxiter,
                "restarts": n_restarts,
                "n_starts": 1 + n_restarts,
                "best_start": best_label,
                "attempts": attempts,
                "initial_params": initial_params.tolist(),
                "optimal_parameters": theta_opt.tolist(),
                "min_loss": min_loss,
                "loss_history": list(getattr(result, "loss_history", []) or []),
                "weights_history": [
                    list(w) for w in (getattr(result, "weights_history", []) or [])
                ],
                "use_mitigation": effective_mitigation,
                "use_compression": self._config.use_compression,
                "shots": self._config.shots,
                "job_name": job_name,
                "wall_clock_s": wall_s,
                "job_metadata": job_meta,
            },
        )
        logger.info(
            "variational_optimization: optimizer=%s BEST min_loss=%.6f from %r "
            "(%d start(s): 1 warm + %d restart(s), %d params, mitigation=%s)",
            self._config.vqe_optimizer,
            min_loss,
            best_label,
            1 + n_restarts,
            n_restarts,
            theta_opt.size,
            effective_mitigation,
        )
        return theta_opt, record

    def save_collected_data(
        self,
        path: str | Path,
        *,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Persist ALL collected records to a structured JSON file.

        The file contains the full configuration, a UTC timestamp, every
        operation record (compression, dry-run cost, run, evaluate_full) with
        all Haiqu metadata / error metrics / timing, and an aggregate summary
        over ``evaluate_full`` points (pass rate, mean ΔE/gap, mean fidelity,
        total QPU cost).

        Credentials are never written — only key names are referenced.

        Parameters
        ----------
        path : str | Path
            Destination ``.json`` path. Parent directories are created.
        extra : dict | None
            Optional caller-supplied context (topology, N, p, model, seed, ...)
            merged into the top-level payload.

        Returns
        -------
        Path
            The path written.
        """
        from dataclasses import asdict

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        eval_points = [r for r in self._records if r["operation"] == "evaluate_full"]
        payload: dict[str, Any] = {
            "schema": "haiqu_collected_data_v1",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "backend_name": self.name,
            "config": asdict(self._config),
            "noise_profile": self._config.resolved_noise_profile(),
            "n_records": len(self._records),
            "records": self._records,
            "summary": self._build_summary(eval_points),
        }
        # Never persist credential values.
        payload["config"].pop("ibm_token", None)
        payload["config"].pop("ibm_instance", None)
        payload["config"].pop("api_access_key", None)
        if extra:
            payload["context"] = extra

        json_dump(json_serialize(payload), out)
        logger.info("Saved %d Haiqu records → %s", len(self._records), out)
        return out

    @staticmethod
    def _build_summary(eval_points: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate stats over evaluate_full points."""
        if not eval_points:
            return {"n_points": 0}
        de_gaps = [p["de_gap"] for p in eval_points if p.get("de_gap") is not None]
        fids = [p["fidelity"] for p in eval_points if p.get("fidelity") is not None]
        abs_es = [p["abs_delta_e"] for p in eval_points if p.get("abs_delta_e") is not None]
        uncs = [p["uncertainty"] for p in eval_points if p.get("uncertainty") is not None]
        n_pass = sum(1 for p in eval_points if p.get("pass_5pct"))
        return {
            "n_points": len(eval_points),
            "n_pass_5pct": n_pass,
            "pass_rate_5pct": n_pass / len(eval_points),
            "mean_de_gap": float(np.mean(de_gaps)) if de_gaps else None,
            "max_de_gap": float(np.max(de_gaps)) if de_gaps else None,
            "mean_abs_delta_e": float(np.mean(abs_es)) if abs_es else None,
            "mean_fidelity": float(np.mean(fids)) if fids else None,
            "min_fidelity": float(np.min(fids)) if fids else None,
            "mean_uncertainty": float(np.mean(uncs)) if uncs else None,
        }

    # ─── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _hamiltonian_to_observables(
        hamiltonian: SparsePauliOp,
    ) -> tuple[np.ndarray, list[SparsePauliOp]]:
        """Split a SparsePauliOp into (real coeffs, single-term observables).

        Each Pauli term becomes its own unit-coefficient ``SparsePauliOp`` so
        Haiqu returns one expectation value per term; the coefficients are
        applied client-side when recombining ⟨H⟩.
        """
        coeffs = np.real(np.asarray(hamiltonian.coeffs, dtype=complex))
        paulis = hamiltonian.paulis
        observables = [SparsePauliOp(paulis[k], np.array([1.0])) for k in range(len(paulis))]
        return coeffs, observables

    @staticmethod
    def _extract_expectation_values(result: Any, n_observables: int) -> np.ndarray:
        """Flatten a haiqu.run observable result into a 1-D array of EVs.

        For a single circuit with multiple observables and no parameter sweep,
        ``job.result()`` returns ``[[ev_obs1, ev_obs2, ...]]`` (indexed
        ``[circuit][observable]``). With a single-point parameter sweep it may
        be ``[[[ev]], ...]``. This normalizes those shapes to length
        ``n_observables``.
        """
        arr = np.asarray(result, dtype=object)
        flat = np.array(arr.flatten().tolist(), dtype=float).flatten()
        # A blind truncation on a mismatch can silently pick the wrong values out
        # of the [circuit][observable][parameter] nesting, so require the exact
        # count and fail loudly otherwise (mirrors the counts-key guard in the
        # self-contained package).
        if flat.size != n_observables:
            raise ValueError(
                f"Expected {n_observables} expectation values from haiqu.run "
                f"(single circuit, single parameter point), got {flat.size}. "
                f"Raw result shape={arr.shape}. Refusing to guess which values "
                "correspond to the Hamiltonian terms."
            )
        return flat
