---
inclusion: fileMatch
fileMatchPattern: "**/eval_cache*,**/ground_truth_cache*,**/backends*,**/pipeline/runner*,**/accelerated*,**/run_noiseless*,**/run_accelerated*,**/optimizers/vqe*"
---

# EvalCache — Quick Reference

## What

`CachedBackend` wraps any deterministic `ExecutionBackend` and caches `evaluate(circuit, H, theta)` results keyed by `(model, topology, N, p, h, SHA256(theta))`. Persists to `data/eval_cache.json`. Default-ON in PipelineRunner and AcceleratedVQE.

## When it helps

- Bidirectional sweeps (same h evaluated twice)
- Re-runs with same config (crash recovery)
- AcceleratedVQE refinement (re-evaluates predicted θ)
- Cross-N experiments with overlapping anchors

## When to DISABLE (`eval_cache=False`)

- Noisy/Hardware backends (auto-disabled — stochastic)
- Benchmarking wall-clock time (cache skews timing)
- Debugging circuit bugs (need fresh computation)
- After changing HVA structure (clear cache or disable)

## Architecture

```
PipelineRunner(eval_cache=True)  →  CachedBackend(NoiselessBackend)
AcceleratedVQE(eval_cache=True)  →  CachedBackend(backend)
VQEOptimizer.sweep loop          →  calls backend.set_h(h) per point
```

## Key rules

- `set_h(h)` MUST be called before evaluate() when h changes (VQEOptimizer does this)
- Full float64 precision in hash — no rounding, no false hits during optimization
- Key includes J (coupling constant) — different J = different Hamiltonian = different key
- NaN/Inf never cached; |E| > 1e6 rejected (sanity guard)
- Max 50k entries, LRU eviction
- Noisy/Hardware/Fake backends auto-detected and cache disabled (by name check)
- `cache.validate_entry(key, backend, circuit, H, theta)` for spot-checks (removes stale entries)
- `cache.clear()` after any change to circuit architecture (HVA builder)
- Key format: `{model}|{topology}|{N}|{p}|J{J:.4f}|{h:.8f}|{SHA256(theta)[:32]}`
