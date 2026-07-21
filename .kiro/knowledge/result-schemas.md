# Result JSON Schemas — Field Reference

Reference for all result file formats. Use when parsing, validating, or extending results.

## Noiseless Pipeline: `pipeline_run_*.json`

```json
{
  "metadata": {                          // Optional — present in thesis runs
    "timestamp_utc": "ISO-8601",
    "platform": "macOS-...",
    "python_version": "3.12.x",
    "software_versions": { "qiskit": "...", "torch": "...", ... }
  },
  "system": {                            // Physical system description
    "hamiltonian": "TFIM: H = -J Σ Z_iZ_j - h Σ X_i",
    "model_type": "TFIM",
    "topology": "chain_1d|ladder|triangular|kagome",
    "boundary_conditions": "open|periodic",
    "n_qubits": 6,
    "J": 1.0,
    "initial_state": "|+>^N",
    "ansatz": "HVA",
    "p_layers": 2,
    "sweep_direction": "descending"
  },
  "config": {                            // Run configuration
    "n_qubits": 10,
    "topology": "ladder",
    "p_layers": 2,
    "n_restarts": 5,
    "maxiter": 1000,
    "seed": 42,                          // Optional — present in seeded runs
    "mpnn": {
      "hidden_dim": 128,
      "n_layers": 3,
      "n_epochs": 6000,
      "lr": 0.001,
      "patience": 500
    },
    "h_values": [4.0, 3.5, 3.0, 2.5, 2.0],  // Training grid (descending)
    "h_test": [2.5]                           // Deployment test point(s)
  },
  "elapsed_s": 35.6,                    // Total wall-clock time
  "phase12_data": [...],                 // Optional — full Phase 1+2 data
  "phase4_results": [                    // PRIMARY OUTPUT — deployment results
    {
      "h_test": 2.5,
      "predicted_energy": -26.39,
      "delta_e": 0.035,                  // Absolute energy error
      "delta_e_over_gap": 0.017,         // ★ PRIMARY METRIC (< 0.05 = pass)
      "mag_x_pred": 0.935,
      "corr_zz_pred": 0.231,
      "mag_x_error": 0.004,
      "corr_zz_error": 0.011,
      "phase_label": "paramagnetic",
      "metrics_checklist": {
        "delta_e_over_gap_lt_5pct": true,
        "correct_phase": true,
        "mag_x_error_lt_1e2": true,
        "corr_zz_error_lt_1e2": false,
        "observables_computed": true
      }
    }
  ],
  "diagnostics": {                       // Phase-by-phase quality metrics
    "phase1": {
      "n_points": 5,
      "elapsed_s": 5.4,
      "gap_min": 1.105                   // Smallest spectral gap in training set
    },
    "phase2": {
      "per_h_timing_s": [...],
      "per_h_iterations": [...],
      "per_h_restart_spread": [...],     // Variance across restarts
      "per_h_converged": [...],          // Boolean per h-point
      "theta_smoothness": 0.036,         // ★ Should be < 0.1
      "worst_convergence_h": 4.0,
      "total_elapsed_s": 0.0,
      "convergence_rate": 1.0            // ★ Should be 1.0
    },
    "phase3": {
      "per_h_mse": {"4.0": 3.5e-05, ...},
      "theta_zz_mse": 2.5e-05,
      "theta_x_mse": 2.5e-05,
      "generalization_gap": 1.96e-05,    // ★ Should be < 1e-3
      "loss_curve_last100": [],
      "elapsed_s": 21.0
    },
    "phase4": {
      "h_test": 2.5,
      "energy_decomposition": {
        "e_exact": -26.43,
        "e_vqe_ceiling": -26.43,
        "e_mpnn_predicted": -26.39,
        "error_from_circuit": 0.0,       // HVA expressibility error
        "error_from_mpnn": 0.036         // MPNN prediction error
      }
    }
  }
}
```

## Noisy/ZNE: `noisy_*.json`

```json
{
  "metadata": { "timestamp": "ISO-8601", ... },
  "system": { ... },                     // Same as noiseless
  "experiment": "n6_noisy_3mode_comparison",
  "config": {
    "n_qubits": 10,
    "p_layers": 2,
    "J": 1.0,
    "h_values": [4.0, 3.0, 2.5],
    "n_layouts": 3,                      // Number of transpilation layouts
    "shots": 16384,
    "seed": 42,
    "n_restarts": 5,
    "optimizer": "L-BFGS-B",
    "noise_model": {
      "backend": "FakeTorino",
      "optimization_level": 2,
      "n_candidate_layouts": 30
    }
  },
  "vqe_baseline": [...],                 // Noiseless VQE reference
  "results_per_h": [                     // Per-h ZNE results
    {
      "h": 4.0,
      "e_exact": -40.84,
      "gap": 5.15,
      "e_noiseless": -40.83,
      "de_noiseless": 0.0014,            // Noiseless ΔE/gap
      "e_noisy_raw": -30.60,
      "de_noisy_raw": 1.99,              // Raw noisy ΔE/gap (huge)
      "e_zne": -27.64,
      "de_zne": 2.57,                    // ZNE-mitigated ΔE/gap
      "r_squared": 0.957,                // Linear fit quality
      "zne_wins": false,                 // Does ZNE beat raw?
      "good_r2": true,                   // R² > 0.8?
      "ces_values": [0.47, 59.6, 1.23],  // Per-layout CES
      "gain_pct": -28.9                  // % improvement (negative = ZNE hurts)
    }
  ],
  "summary": {                           // ★ AGGREGATE METRICS
    "n_mitigated_wins": 0,
    "n_good_r2": 3,
    "n_total": 3,
    "success_criteria_met": false,       // ★ ZNE beats raw AND R²>0.8
    "mean_de_noiseless": 0.013,
    "mean_de_noisy_raw": 2.69,
    "mean_de_zne": 3.43,
    "mean_r2": 0.954,
    "mean_gain_pct": -27.9,
    "elapsed_s": 43.1
  }
}
```

## BaseExperiment: `run_*.json` (in exp_<id>/)

```json
{
  "config": {
    "experiment_id": "B4",
    "category": "B",
    "description": "Hessian-guided adaptive restarts",
    "hypothesis": "Hessian analysis identifies saddle points...",
    "system": {
      "n_qubits": 6, "p_layers": 2, "topology": "chain_1d",
      "J": 1.0, "h_values": [1.0, 1.25, 1.5, 2.0], "h_test": [1.5]
    },
    "vqe": { "optimizer": "L-BFGS-B", "n_restarts": 5, ... },
    "mpnn": { "hidden_dim": 64, "n_layers": 3, ... },
    "seeds": [42, 43, 44]
  },
  "analysis": {
    "experiment_id": "B4",
    "n_seeds": 3,
    "timestamp": "20260527_004101",
    "per_seed": {
      "42": { "mean_de_gap": 0.03, "n_passing": 3, "n_total": 4, ... },
      "43": { ... },
      "44": { ... }
    },
    "summary": {                         // ★ AGGREGATE — used for verdict
      "mean_de_gap": 0.050,
      "std_de_gap": 0.061,
      "median_de_gap": 0.022,
      "pass_rate": 0.75,                 // ★ Fraction meeting threshold
      "n_total_points": 12,
      "total_time_s": 13.0,
      "convergence_rate": 1.0
    }
  },
  "results": {                           // Per-seed, per-h raw data
    "42": [
      { "h_value": 1.5, "energy": -9.77, "exact_energy": -9.85,
        "relative_error": 0.054, "fidelity": null, ... }
    ]
  }
}
```

## Execution Log: `execution_log_*.json` (in variant folders)

```json
{
  "timestamp": "20260527_040913",
  "topology": "ladder",
  "n_qubits": 10,
  "total_variants": 33,
  "passed": 33,
  "failed": 0,
  "total_elapsed_s": 2085.0,
  "verdicts": { "PASS": 17, "MARGINAL": 2, "FAIL": 4, "SKIP-P3": 0, "ERROR": 0 },
  "results": [
    { "variant_id": "NL-A1", "success": true, "verdict": "PASS",
      "elapsed_s": 44.25, "delta_e_over_gap": 0.0165, ... }
  ],
  "variants": [
    { "id": "NL-A1", "description": "VQE restarts=1 (ladder, hidden=128)",
      "category": "noiseless", "hypothesis": "...", "expected_outcome": "..." }
  ]
}
```

## Simulation Diagnostics Block (auto-injected since 2026-07-13)

Present in ALL `ValidationRunner` results. Old results (pre 2026-07-13) won't have this field.

```json
{
  "simulation_diagnostics": {
    "backend_type": "noiseless_statevector",
    "n_qubits": 10,
    "topology": "heavy_hex",
    "method_exact": true,
    "chi_max": 64,
    "chi_sufficiency_warning": "2D topology 'square' with N=20: chi=64 may be insufficient.",
    "shots": 16384,
    "hardware_mode": "fake_backend",
    "hardware_backend_name": "ibm_kingston",
    "noise_sources": ["gate_error", "readout_error", "decoherence"]
  }
}
```

**Field presence depends on backend type:**

| Backend | backend_type | method_exact | chi_max | shots | hardware_mode |
|---------|:---:|:---:|:---:|:---:|:---:|
| NoiselessBackend | `noiseless_statevector` | true | — | — | — |
| MPSBackend | `mps_aer_mps_chi64_exact` | true | ✓ | — | — |
| FakeTorino | `fake_torino` | false | — | — | — |
| NoisyBackend | `noisy_shots=4096` | false | — | ✓ | — |
| HardwareBackend | `hardware_ibm_kingston` | false | — | ✓ | ✓ |

**VQE per-point new fields (noiseless pipeline Section 2):**
- `variational_violation`: `max(0, E_exact - E_vqe - 1e-8)` — 0 = healthy
- `variational_ok`: `E_vqe >= E_exact - 1e-8` — true = healthy

**Interpretation thresholds:**
- `variational_violation = 0`: Normal
- `variational_violation < 1e-6`: Numerical noise, safe to ignore
- `variational_violation > 1e-4`: Investigate (solver or backend issue)
- `variational_violation > 1e-2`: Critical error (E_exact reference wrong?)
- `>4 consecutive violations`: VQE sweep auto-aborts (since 2026-07-13)

## File Naming Conventions

| Pattern | Meaning |
|---------|---------|
| `pipeline_run_YYYYMMDD_HHMMSS.json` | Noiseless 4-phase pipeline result |
| `noisy_3mode_YYYYMMDD_HHMMSS.json` | 3-mode ZNE comparison (noiseless/raw/mitigated) |
| `run_YYYYMMDD_HHMMSS.json` | BaseExperiment result |
| `log_YYYYMMDD_HHMMSS.json` | Execution log (timing, errors) |
| `execution_log_YYYYMMDD_HHMMSS.json` | Variant runner batch log |
| `diagnostics.json` | Latest diagnostics snapshot for a variant |

## Validation Rules for Integrity Checking

A result file is **valid** if:
1. It parses as valid JSON (no truncation, no encoding errors)
2. Required fields exist and have correct types (see schemas above)
3. `h_values` are in descending order (ascending = wrong sweep)
4. `n_qubits` ∈ {4, 6, 8, 10, 16, 20, 40, 50, 80, 100, 120, 200} (any positive int)
5. `p_layers` ∈ {1, 2, 3, 4, 5, 6} (thesis: p≤2; validation scripts allow more)
6. `delta_e_over_gap` ∈ [0, 50] (>50 = catastrophic failure)
7. `elapsed_s` > 0 (0 = run didn't execute)
8. `convergence_rate` ∈ [0, 1]
9. `mean_r2` ∈ [0, 1] for noisy results

A result is **suspect** (not broken, but investigate) if:
- `delta_e_over_gap` > 0.10 at h well within valid regime
- `theta_smoothness` > 1.0 (warm-start chain likely broke)
- `generalization_gap` > 0.01 (MPNN overfitting)
- Multiple run files exist with decreasing quality (regression)
