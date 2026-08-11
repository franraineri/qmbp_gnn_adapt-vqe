---
inclusion: fileMatch
fileMatchPattern: "scripts/remote/*"
---

# Google Colab Remote Execution

## Tool: `scripts/remote/colab_runner.py`

CLI wrapper around `google-colab-cli` for running qmbp experiments on Colab VMs.

## When to use Colab (vs local Mac)

| Use case | Colab? | Why |
|---|---|---|
| MPNN training | ✅ | GPU-bound, T4 is ~10x faster than CPU |
| Large-N MPS (N>22) | ✅ | Needs 12GB+ RAM, Mac may swap |
| VQE sweeps (many h-points, N≥16) | ✅ | Offload long runs (>30 min) |
| Parallel experiments | ✅ | Run on Colab while Mac does other work |
| Statevector N≤16 | ❌ | Mac CPU is faster (Colab has weak 2-core Xeon) |
| Quick debug/iteration | ❌ | Latency overhead not worth it |

## Colab free tier specs

- **CPU**: Intel Xeon, 2 vCPUs @ 2.2GHz (slow for CPU-bound tasks)
- **RAM**: ~12.7GB system
- **GPU (T4)**: 15GB VRAM, good for torch/MPNN training
- **Disk**: ~100GB ephemeral
- **Session**: 12h max, 90min idle timeout (CLI keep-alive handles this)
- **Quota**: ~30h/week GPU, CPU unlimited

## Commands

```bash
# Setup (once per session)
.venv/bin/python scripts/remote/colab_runner.py setup --gpu T4

# Run experiment
.venv/bin/python scripts/remote/colab_runner.py run <script> [args...] --timeout 7200

# Sync code after git push
.venv/bin/python scripts/remote/colab_runner.py sync

# Download results
.venv/bin/python scripts/remote/colab_runner.py fetch

# Benchmark Colab vs local
.venv/bin/python scripts/remote/colab_runner.py bench

# Stop (preserves GPU quota)
.venv/bin/python scripts/remote/colab_runner.py stop
```

## Important notes

- Session state is **ephemeral** — results must be fetched before `stop` or 12h timeout
- After `sync`, the VM has whatever is on the current git branch (must `git push` first)
- Results land in `results/colab/` locally after `fetch`
- The `--timeout` default is 2h (7200s); increase for very long sweeps
- `jupyter-kernel-client` must stay pinned to `==0.15.0` (v1.0.0 breaks colab-cli)
