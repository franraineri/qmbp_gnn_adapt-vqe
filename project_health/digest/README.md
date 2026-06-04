# Result Digest Tool

Extracts key knowledge from experiment results by **result kind** — making it easy to take perspective across hundreds of JSON files without reading them individually.

## Quick Start

```bash
# Full digest (all kinds)
python -m project_health.digest

# By kind
python -m project_health.digest --kind noiseless
python -m project_health.digest --kind noisy
python -m project_health.digest --kind experiment
```

## Result Kinds

| Kind | What it shows | Key metrics |
|------|--------------|-------------|
| `noiseless` | 4-phase pipeline runs | ΔE/gap, convergence rate, θ-smoothness, generalization gap |
| `noisy` | ZNE/noise mitigation | R², gain%, mitigation wins, ΔE comparison |
| `experiment` | Hypothesis tests | verdict (confirmed/rejected/failed), pass rate, criteria |

## Filters

```bash
--topology ladder          # chain_1d, ladder, triangular, kagome
--n-qubits 10             # System size
--p-layers 1              # Ansatz depth
--folder variants_N10_ladder  # Specific folder (exact or substring match)
```

## Sorting & Limiting

```bash
--sort delta_e            # Noiseless: delta_e, time, smoothness, gap, folder
--sort r2                 # Noisy: r2, gain, time, folder
--sort verdict            # Experiment: id, verdict, de_gap, pass_rate, folder
--top 10                  # Show only top N results (after sorting)
```

## Grouped Comparisons

```bash
# Noiseless grouping (topology, n_qubits, hidden_dim, n_restarts, p_layers)
python -m project_health.digest --kind noiseless --group-by topology
python -m project_health.digest --kind noiseless --group-by n_restarts --topology ladder

# Noisy grouping (topology, n_qubits, n_layouts, shots, p_layers)
python -m project_health.digest --kind noisy --group-by n_qubits
```

## Analysis

```bash
# Statistical summary (percentiles, distribution, correlations)
python -m project_health.digest --kind noiseless --stats

# Outlier detection (IQR method + automatic diagnosis)
python -m project_health.digest --kind noiseless --outliers

# Side-by-side comparison of two folders
python -m project_health.digest --compare variants_N10_ladder variants_N10_triangular
```

## Output Formats

```bash
# Text table (default, to stdout)
python -m project_health.digest --kind noiseless

# Save to file
python -m project_health.digest --kind noiseless -o results_summary.txt

# Markdown (for documentation)
python -m project_health.digest --markdown -o digest.md

# JSON (for programmatic use)
python -m project_health.digest --json digest.json

# Verbose (extra details per row)
python -m project_health.digest --kind experiment --verbose
```

## Architecture

```
project_health/digest/
├── __init__.py       # Public API exports
├── __main__.py       # CLI entry point + filters + sorting
├── models.py         # Dataclasses + re-exports from framework.criteria
├── scanner.py        # File discovery and JSON parsing
├── formatters.py     # Text, markdown, grouped, stats, outliers, compare
└── README.md
```

Experiment success criteria and verdict logic live in
`src/qmbp_simulation/framework/criteria.py` (single source of truth).
`models.py` re-exports `EXPERIMENT_CRITERIA`, `REJECTION_IS_FINDING`, and
`compute_verdict` for backward compatibility.

## Programmatic Use

```python
from project_health.digest import ResultScanner, NoiselessResult

scanner = ResultScanner(Path("results"))
noiseless, noisy, experiments = scanner.scan_all()

# Filter in code
ladder_n10 = [r for r in noiseless if r.topology == "ladder" and r.n_qubits == 10]
```
