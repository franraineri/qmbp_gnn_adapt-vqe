---
inclusion: manual
---

# Haiqu Integration — Running GNN-HVA on QPU (invoke with #haiqu-integration)

Canonical workflow and rules for deploying an HVA circuit with GNN-predicted θ
on real QPU via the Haiqu middleware stack (state compression + Error Shield +
hardware execution).

## What it is

Haiqu is a QPU-agnostic cloud middleware. We wrap it behind the project's
`ExecutionBackend` ABC so a GNN-predicted HVA circuit runs on hardware through
the same contract used everywhere else.

| Purpose | File |
|---------|------|
| Backend + config | `src/qmbp_simulation/execution/hardware/haiqu_backend.py` |
| End-to-end notebook | `notebooks/04_haiqu_hardware_deployment.ipynb` |

## API surface (confirmed against docs.haiqu.ai, SDK v1.4)

| Call | Signature / behavior |
|------|----------------------|
| `haiqu.login(api_access_key=...)` | Once per session. Key from `HAIQU_API_KEY` or implicit in Haiqu Lab. |
| `haiqu.init(name)` | Groups runs on the dashboard. |
| `haiqu.run(circuits, observables, parameters, device_id, shots, options, use_mitigation, dry_run)` | Observable mode returns EVs indexed `[circuit][observable][parameter]`. `job.info` has `uncertainty`, `qpu_cost`. `dry_run=True` → `job.estimated_qpu_cost` (no credits). |
| `haiqu.state_compression(circuit, compression_level, noise_profile, fine_tuning, approximation_level, max_time)` | `job.result()` → `(compressed_circuit, quality)`. Requires a **bound** circuit. |
| `haiqu.save_ibm_credentials(ibm_quantum_token, ibm_quantum_instance)` | Persist IBM creds once; then omit `options`. |

## Canonical flow (energy / ground state)

```python
from qmbp_simulation import HamiltonianBuilder, make_lattice, ClassicalSolver, HVACircuitBuilder
from qmbp_simulation.predictors import predict_theta
from qmbp_simulation.predictors.mpnn import load_mpnn_checkpoint
from qmbp_simulation.execution.hardware.haiqu_backend import HaiquBackend, HaiquConfig

lattice = make_lattice("heavy_hex", 10, J=1.0, h=3.25)
H = HamiltonianBuilder().build(lattice)          # H = -J ΣZZ - h ΣX
gt = ClassicalSolver().solve(H, lattice)         # gt.ground_energy, gt.gap

model = load_mpnn_checkpoint("path/to/mpnn.pt")
theta = predict_theta(model, lattice, [3.25])[3.25]

circuit, _ = HVACircuitBuilder().create_pauli_evolution(10, 1, lattice)

backend = HaiquBackend(HaiquConfig(device_id="ibm_kingston", use_mitigation=True))
cost = backend.estimate_cost(circuit, H, theta)  # dry run first
energy = backend.evaluate(circuit, H, theta)     # observable mode + Error Shield
de_gap = abs(energy - gt.ground_energy) / gt.gap
```

## Key design rules

1. **Observable mode for energy.** Submit the circuit WITHOUT terminal
   measurements + Hamiltonian Pauli terms as `observables` + `use_mitigation=True`.
   `HaiquBackend._hamiltonian_to_observables` splits the `SparsePauliOp` into
   unit-coefficient terms; ⟨H⟩ = Σ_k coef_k·⟨P_k⟩ is recombined client-side.
2. **θ as `parameters`, not manual bind.** `ParameterVector` binds by index
   order (θ[0], θ[1], ...), matching `HVACircuitBuilder`. Compression is the
   exception — it needs a bound circuit, so θ is assigned before compressing.
3. **noise_profile follows the device.** `ibm_kingston` (Heron R2) →
   `ibm_heron_r2`, auto-selected from `device_id` when `noise_profile=None`.
4. **Compress first, then mitigate.** Haiqu's documented recommendation for
   production runs. Enable `use_compression=True` for p≥2 / large N; for shallow
   p=1 N=10 the overhead is usually not worth it.
5. **Always dry-run before real hardware.** `estimate_cost(...)` spends no credits.
6. **Lazy SDK import.** `haiqu.sdk` is imported inside methods — the project
   imports cleanly without `haiqu-sdk` installed.

## Data collection (mandatory)

Every Haiqu operation is captured. `HaiquBackend` accumulates a rich record per
call and persists everything to a single JSON file.

- `backend.records` — list of all operation records (compression, dry-run cost,
  run, evaluate_full).
- `backend.evaluate_full(circuit, H, params, h=, e_exact=, gap=, exact_state=)`
  — runs on QPU and returns a complete record: energy, uncertainty, qpu_cost,
  wall-clock, raw per-observable EVs, coefficients, term contributions, plus
  derived |ΔE|, ΔE/gap, and fidelity.
- `backend.save_collected_data(path, extra=...)` — writes a structured JSON
  (`schema="haiqu_collected_data_v1"`) with full config, all records, and an
  aggregate summary (pass rate, mean ΔE/gap, mean/min fidelity, mean
  uncertainty). Credentials are stripped; only key names are referenced.

What gets captured per operation:

| Operation | Captured |
|-----------|----------|
| `state_compression` | quality, cnot_reduction, circuit before/after (depth, 2Q, gate counts), wall-clock, full job metadata (logs, time, id, status) |
| `dry_run_cost_estimate` | `estimated_qpu_cost`, circuit stats, job metadata |
| `run` | energy, uncertainty, qpu_cost, shots, mitigation flags, compression quality, raw EVs per observable, Pauli terms, coeffs, term contributions, wall-clock, full job metadata |
| `evaluate_full` | everything in `run` + h, θ, e_exact, gap, \|ΔE\|, ΔE/gap, fidelity, pass_5pct |

Output convention: `results/haiqu/haiqu_{topology}_n{N}_p{P}_{device}_{ts}.json`.

## Data loading (`vector_loading` / `mps_loading`)

State preparation, not QPU execution: Haiqu synthesizes a fixed circuit that loads a
known state. Used as a **reference/benchmark** for ground-state preparation (it needs
the ground state as input; it is not a solver). Results and conclusions:
`results/hva_vl_study/reports/REPORT_VL_square_frustrated.md`; handoff:
`internal/documentation/binnacles/binnacle-vl-haiqu-square-frustrated.md`.

| Need                                                     | Use (all in `scripts/analysis/vl_vs_hva/`)                       |
|----------------------------------------------------------|------------------------------------------------------------------|
| Load a dense ground state (N ≤ 20)                       | `hva_vl_study_common.run_vl_job`                                 |
| Load an MPS (large N, bond ≤ 64, standard or Vidal form) | `hva_vl_study_common.run_vl_mps_job`                             |
| Get a bond-capped MPS from any `SparsePauliOp`           | `mps_extraction.dmrg_mps_capped` (+ `capped_mps_vidal`)          |
| Energy error of the loaded circuit (local)               | `hva_vl_study_common.circuit_energy`                             |
| Keep the circuit for later (no cloud, no key)            | `save_circuit_qpy` / `load_circuit_qpy`                          |
| Batch sweep with result cache                            | `vl_quality_sweep_frustrated.py` (`--force-recompute` to bypass) |

Rules learned (see `error-patterns.md` for the failure details):

1. Quality knobs: `num_layers` (2q gates double per step), `fine_tuning_iterations`
   (no gate cost), `truncation_cutoff`. Small-gap states need L8/F50 or more.
2. **Poll longer than the compute budget.** Pass `max_time` to the job and poll for
   `max_time + 300 s`, otherwise long jobs stay `RUNNING`.
3. **Jobs spend credits.** The account hit HTTP 402. Use the cache, persist
   `circuit_id` and the QPY circuit, and prefer `vector_loading_estimates` (unused so
   far) before large batches.
4. `vector_loading` caps at N ≤ 20; `mps_loading` accepts bond ≤ 64 (build the MPS
   from the operator so frustrated couplings are included).
5. Never trust `ClassicalSolver` frustrated energies at N > 16 from before commit
   `6cf7465`. The N = 18 reference is the operator-based DMRG energy.
6. SDK in use: `haiqu-sdk` v1.6.0 (the API table above was confirmed on v1.4).

## Credentials

| Layer | Env var | Needed for |
|-------|---------|-----------|
| Haiqu API key | `HAIQU_API_KEY` | Any Haiqu call (compression, run, mitigation) |
| IBM token | `IBM_KEY` | Real IBM devices only |
| IBM instance CRN | `IBM_INSTANCE_CRN` | Real IBM devices only |

Simulators (`fake_torino`, `aer_simulator`) need only the Haiqu key.

## Development vs hardware

- **Dev / validation:** `device_id="fake_torino"`, `use_mitigation=False`.
  No IBM credentials, no cost. Validates the full flow.
- **Hardware:** `device_id="ibm_kingston"`, `use_mitigation=True`, export IBM
  creds. Compression `noise_profile` auto-switches to `ibm_heron_r2`.

## MCP servers (optional, for AI-assisted work)

`.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "Haiqu": { "type": "http", "url": "https://api.haiqu.ai/mcp",
               "headers": { "Authorization": "HAIQU-API-KEY" } },
    "HaiquDocumentation": { "type": "http", "url": "https://docs.haiqu.ai/mcp" }
  }
}
```

The Docs server needs no key; the API server does.

## References

- https://docs.haiqu.ai/reference/run/run
- https://docs.haiqu.ai/reference/middleware/compression
- https://docs.haiqu.ai/core_features/error_shield
