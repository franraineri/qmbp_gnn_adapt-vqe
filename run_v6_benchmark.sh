#!/usr/bin/env bash
#
# GNN-HVA v6.0 — Multi-Run Benchmark
#
# Executes the full pipeline N times with different seeds,
# saves raw JSON results, and appends a summary to the binnacle.
#
# Usage:
#   ./run_v6_benchmark.sh           # 3 runs (default)
#   ./run_v6_benchmark.sh 5         # 5 runs
#   ./run_v6_benchmark.sh 3 1.5     # 3 runs at h_test=1.5
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUNS="${1:-3}"
H_TEST="${2:-1.25}"

echo "============================================================"
echo "  GNN-HVA v6.0 — Benchmark (${RUNS} runs, h_test=${H_TEST})"
echo "============================================================"
echo ""

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found."
    exit 1
fi

# Check dependencies
python -c "import torch, torch_geometric, qiskit" 2>/dev/null || {
    echo "ERROR: Missing dependencies. Run: pip install -r requirements.txt"
    exit 1
}

# Run benchmark
python scripts/benchmark_v6.py --runs "$RUNS" --h-test "$H_TEST"
