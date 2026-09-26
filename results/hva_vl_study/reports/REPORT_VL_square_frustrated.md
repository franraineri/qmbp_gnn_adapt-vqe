# Haiqu Vector Loading on 2D Frustrated Square TFIM — Complete Report

State-preparation study of Haiqu **Vector Loading (VL)** applied to the
**frustrated square-lattice transverse-field Ising model** (`tfim_frustrated`,
next-nearest-neighbour coupling `J2 = 0.5`). All figures below are **primary
measurements** taken directly from the study artifacts — no derived quantities
except infidelity `1 − F`, which is VL's native loading-error metric.

- Model: `tfim_frustrated`, `J2 = 0.5`, `J = 1.0`
- Lattice: `square`, system sizes `N = 9` and `N = 10`
- VL target: the exact ground-state vector (statevector-exact), loaded by Haiqu
  `vector_loading` in the Haiqu cloud
- Fidelity: `F = |⟨ψ_exact | ψ_VL⟩|²` (exact statevector overlap)
- Energy error: `|ΔE| = |⟨ψ_VL|H|ψ_VL⟩ − E0|` (energy of the VL-prepared state)
- Source artifacts:
  - `results/hva_vl_study/circuit_comparison/tfim_frustrated_square_*.json`
  - `results/hva_vl_study/vl_h_sweep/vl_h_sweep_frustrated.json`
  - `results/hva_vl_study/vl_quality_sweep/vl_quality_sweep_frustrated.json`

---

## 1. What VL measures (and what it does not)

VL is a **state-preparation circuit synthesizer**, not a ground-state solver. It
receives the already-known ground-state vector and returns a fixed
(zero-parameter) circuit that approximates it. Its primary error metrics are:

- **Infidelity `1 − F`** — the native loading error.
- **`|ΔE|`** — the energy of the VL-prepared state under `H` versus the exact
  `E0`, computed locally (statevector) from the loaded circuit.

---

## 2. VL parameters studied

Haiqu `vector_loading` exposes three quality knobs. This study varied the first
two; the third was kept at its default.

| Parameter                 | Meaning                                          | Values studied      |
|---------------------------|--------------------------------------------------|---------------------|
| `num_layers`              | Variational layers in the loading circuit        | 2 (default), 4, 8   |
| `fine_tuning_iterations`  | Variational refinement iterations                | 20 (default), 50    |
| `truncation_cutoff`       | MPS truncation threshold                         | 1e-6 (fixed)        |

- **Default configuration**: `num_layers = 2`, `fine_tuning_iterations = 20`.
- **Quality sweep**: full grid `num_layers ∈ {2, 4, 8}` × `fine_tuning_iterations
  ∈ {20, 50}` on the small-gap points (`N = 9`, `h ∈ {0.5, 1.0}`).
- VL native gate basis: `rz`, `ry`, `cx`.

---

## 3. VL at default settings (`num_layers = 2`, `fine_tuning_iterations = 20`)

Per-`h` circuit characterization, all metrics measured. The VL circuit for each
point is saved locally as QPY under `circuit_comparison/circuits/` for
Haiqu-independent re-analysis.

| N  | h    | E0 (exact) | gap    | edges | VL fidelity F | 1−F    | \|ΔE\|  | total gates | 2q gates | 1q gates | depth | 2q depth | status |
|----|------|-----------:|-------:|------:|--------------:|-------:|--------:|------------:|---------:|---------:|------:|---------:|--------|
| 9  | 0.50 |    -7.7532 | 0.2238 |    12 |        0.8086 | 0.1914 |  0.7165 |         207 |       30 |      177 |    53 |       13 | DONE   |
| 9  | 1.00 |   -11.1802 | 0.8601 |    12 |        0.8999 | 0.1001 |  0.5399 |         215 |       30 |      185 |    53 |       13 | DONE   |
| 9  | 2.00 |   -19.3383 | 2.6252 |    12 |        0.9613 | 0.0387 |  0.3887 |         216 |       30 |      186 |    53 |       13 | DONE   |
| 9  | 3.00 |   -27.9787 | 4.5394 |    12 |        0.9870 | 0.0130 |  0.1721 |         216 |       30 |      186 |    53 |       13 | DONE   |
| 10 | 1.00 |   -12.4739 | 0.5023 |    13 |        0.8966 | 0.1034 |  0.6180 |         237 |       34 |      203 |    53 |       13 | DONE   |

VL native gate composition (default settings):

| N | h    | rz  | ry | cx |
|---|------|----:|---:|---:|
| 9 | 0.50 | 112 | 65 | 30 |
| 9 | 1.00 | 116 | 69 | 30 |
| 9 | 2.00 | 118 | 68 | 30 |
| 9 | 3.00 | 118 | 68 | 30 |
| 10| 1.00 | 129 | 75 | 34 |

At default settings the 2q-gate count and depth are flat across `h` (30 CX /
depth 53 for N=9), while fidelity tracks the spectral gap: lowest where the gap
is smallest (`h = 0.5`, gap 0.224 → F = 0.809, `|ΔE|` = 0.717) and near-unity in
the paramagnetic regime (`h = 3.0`, gap 4.54 → F = 0.987).

---

## 4. Quality-knob sweep (N = 9, small-gap points)

Full grid on the two points where default VL underperforms. VL energy `E_VL` and
error `|ΔE|` measured for every configuration.

### 4.1 square, h = 0.50 (gap 0.2238, E0 = -7.7532)

| num_layers | fine_tuning | VL fidelity F | 1−F    | E_VL     | \|ΔE\|  | 2q gates |
|-----------:|------------:|--------------:|-------:|---------:|--------:|---------:|
|          2 |          20 |        0.8093 | 0.1907 |  -7.0367 |  0.7165 |       30 |
|          2 |          50 |        0.8359 | 0.1641 |  -7.1242 |  0.6290 |       30 |
|          4 |          20 |        0.8844 | 0.1156 |  -7.2303 |  0.5229 |       60 |
|          4 |          50 |        0.9085 | 0.0915 |  -7.3322 |  0.4210 |       60 |
|          8 |          20 |        0.9200 | 0.0800 |  -7.3499 |  0.4033 |      120 |
|          8 |          50 |        0.9431 | 0.0569 |  -7.4782 |  0.2750 |      120 |

### 4.2 square, h = 1.00 (gap 0.8601, E0 = -11.1802)

| num_layers | fine_tuning | VL fidelity F | 1−F    | E_VL      | \|ΔE\|  | 2q gates |
|-----------:|------------:|--------------:|-------:|----------:|--------:|---------:|
|          2 |          20 |        0.8999 | 0.1001 |  -10.6403 |  0.5399 |       30 |
|          2 |          50 |        0.9082 | 0.0918 |  -10.7060 |  0.4741 |       30 |
|          4 |          20 |        0.9547 | 0.0453 |  -10.8709 |  0.3093 |       60 |
|          4 |          50 |        0.9665 | 0.0335 |  -10.9369 |  0.2432 |       60 |
|          8 |          20 |        0.9656 | 0.0344 |  -10.9203 |  0.2599 |      120 |
|          8 |          50 |        0.9754 | 0.0246 |  -10.9821 |  0.1981 |      120 |

### 4.3 Best configuration per point

| h    | default (2,20): F / \|ΔE\| | best (8,50): F / \|ΔE\| | 2q (default → best) |
|------|---------------------------:|------------------------:|---------------------|
| 0.50 |            0.8086 / 0.7165  |         0.9431 / 0.2750 | 30 → 120            |
| 1.00 |            0.8999 / 0.5399  |         0.9754 / 0.1981 | 30 → 120            |

---

## 5. Large N via `mps_loading` (N = 18)

At N = 18 the exact ground-state *vector* is no longer available (N > statevector
limit), so the dense `vector_loading` path cannot be used. VL was instead driven
through `mps_loading`: the MPS is obtained from bond-capped DMRG (χ ≤ 64, Haiqu's
cap) built directly from the frustrated `SparsePauliOp`, then loaded. The energy
error `|ΔE|` is measured locally against the DMRG reference energy `E0_dmrg`
(the exact vector is unavailable, so `E0_dmrg` is the reference, not exact diag).

Config: `num_layers = 8`, `fine_tuning_iterations = 50`, `truncation_cutoff = 1e-6`,
standard MPS form. VL native basis `rz`, `ry`, `cx`.

| N  | h    | E0_dmrg   | VL fidelity F | 1−F    | E_VL      | \|ΔE\|  | 2q gates | depth | total gates | Haiqu job time (s) |
|----|------|----------:|--------------:|-------:|----------:|--------:|---------:|------:|------------:|-------------------:|
| 18 | 0.50 | -16.9846  |        0.8625 | 0.1375 | -16.4464  |  0.5382 |      264 |   181 |        1768 |              718.1 |
| 18 | 1.00 | -23.0265  |        0.7924 | 0.2076 | -21.6327  |  1.3939 |      264 |   181 |        1767 |              664.0 |

### 5.1 Input-MPS quality (χ-cap cost)

The bond cap (χ = 64) is not the bottleneck: the DMRG energy at χ = 64 vs a
reference χ = 256 differs negligibly, so the input MPS is faithful and the
fidelity loss happens in the MPS→circuit loading, not in the classical state.

| N  | h    | χ realized | bond saturated | E(χ=64)   | E(χ=256)  | \|ΔE\|_cap |
|----|------|-----------:|:--------------:|----------:|----------:|-----------:|
| 18 | 0.50 |         64 |      yes       | -16.9846  | -16.9850  |    4.3e-04 |
| 18 | 1.00 |         64 |      yes       | -23.0265  | -23.0280  |    1.5e-03 |

### 5.2 Size scaling (fixed config L8/F50)

The same "best" config that reaches F ≈ 0.94–0.98 at N = 9 drops at N = 18, and
the 2q-gate count more than doubles:

| h    | N = 9 (F / 2q) | N = 18 (F / 2q) |
|------|---------------:|----------------:|
| 0.50 |  0.9431 / 120  |   0.8625 / 264  |
| 1.00 |  0.9754 / 120  |   0.7924 / 264  |

Caveat: N = 9 uses `vector_loading` while N = 18 uses `mps_loading`, so part of
this drop reflects the different loading path, not size alone. The gap-limited
trend (§3–4) is the more controlled conclusion.

Provenance: each N = 18 VL circuit is saved as QPY under `vl_mps_n18/circuits/`
with its Haiqu `circuit_id`, for Haiqu-independent re-analysis.

---

## 6. Findings

- **VL fidelity in the frustrated regime is gap-limited at default settings.**
  Lowest fidelity and largest `|ΔE|` coincide with the smallest spectral gap
  (`h = 0.5`: F = 0.809, `|ΔE|` = 0.717); both improve monotonically with `h`.
- **The default-setting drop is a configuration limit, not a hard method limit.**
  Raising `num_layers` to 8 and `fine_tuning_iterations` to 50 lifts the two
  worst points to F = 0.943 / `|ΔE|` = 0.275 (`h = 0.5`) and F = 0.975 / `|ΔE|` =
  0.198 (`h = 1.0`).
- **The two knobs behave differently.** `fine_tuning_iterations` (20 → 50) buys
  ~2-3 pp of fidelity (and lower `|ΔE|`) at **no change in 2q-gate count** (30 →
  30). `num_layers` (2 → 4 → 8) is the strong lever but **doubles the 2q-gate
  count** at each step (30 → 60 → 120).
- **Fidelity and energy error move together**, as expected: every configuration
  that raises F also lowers `|ΔE|`.
- **At N = 18 (via `mps_loading`) fidelity is lower than at N = 9 for the same
  config** (F = 0.79–0.86 vs 0.94–0.98 at L8/F50), while the input MPS remains
  faithful (χ-cap cost ≤ 1.5e-3 in energy). The loss is in the loading step; the
  worst point is the small-gap `h = 1.0` (F = 0.792, `|ΔE|` = 1.394).

---

## 6bis. Not yet measured

Whether pushing `fine_tuning_iterations = 200`, a smaller `truncation_cutoff`
(1e-8), or the Vidal MPS form lifts the N = 18 fidelity above the L8/F50 baseline
is **untested** — those runs were blocked by Haiqu credit exhaustion (HTTP 402).
The tooling (`vl_mps_large_n.py --fine-tuning --truncation-cutoff --mps-form`) is
in place to resume once credits are available.

---

## 7. Reproduction

```bash
# Default per-h characterization (requires HAIQU_API_KEY)
HAIQU_API_KEY=... .venv/bin/python scripts/analysis/vl_h_sweep_frustrated.py \
    --topologies square --n 9 --p 2 --j2 0.5 --h 0.5 1.0 2.0 3.0

# Quality-knob sweep on the small-gap points (cached; --force-recompute to re-run)
HAIQU_API_KEY=... .venv/bin/python scripts/analysis/vl_quality_sweep_frustrated.py \
    --num-layers 2 4 8 --fine-tuning 20 50 --n 9 --j2 0.5

# Large N via mps_loading (bond-capped DMRG → Haiqu mps_loading)
HAIQU_API_KEY=... .venv/bin/python scripts/analysis/vl_mps_large_n.py \
    --topology square --n 18 --j2 0.5 --h 0.5 1.0 \
    --num-layers 8 --fine-tuning 50 --chi-max 64
```

Primary-metric definitions:

- `F` — exact statevector fidelity `|⟨ψ_exact|ψ_VL⟩|²` (measured).
- `1 − F` — loading infidelity.
- `E_VL`, `|ΔE|` — energy of the VL-prepared state under `H` and its absolute
  error vs the exact `E0`, computed locally from the loaded circuit.
- `total gates`, `2q gates`, `1q gates`, `depth`, `2q depth` — counted on the VL
  circuit returned by Haiqu (measured).
- `E0`, `gap` — exact ground-state energy and spectral gap from
  `ClassicalSolver` / `GroundTruthCache` (measured).
- `num_layers`, `fine_tuning_iterations`, `truncation_cutoff` — VL configuration.
