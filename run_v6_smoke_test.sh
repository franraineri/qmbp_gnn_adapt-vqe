#!/usr/bin/env bash
#
# GNN-HVA v6.0 — Quick Smoke Test Runner
#
# Activates the virtual environment and runs the reduced end-to-end pipeline
# to verify all v6 modules work correctly.
#
# Usage:
#   ./run_v6_smoke_test.sh
#
# Expected runtime: ~2 minutes
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  GNN-HVA v6.0 — Smoke Test"
echo "============================================================"
echo ""

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating .venv..."
    source .venv/bin/activate
else
    echo "ERROR: .venv not found. Create it first:"
    echo "  python3.12 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Verify key dependencies
echo "Checking dependencies..."
python -c "
import qiskit, torch, torch_geometric, sklearn, numpy, scipy
print(f'  Python:           {__import__(\"sys\").version.split()[0]}')
print(f'  Qiskit:           {qiskit.__version__}')
print(f'  PyTorch:          {torch.__version__}')
print(f'  PyTorch Geometric: {torch_geometric.__version__}')
print(f'  scikit-learn:     {sklearn.__version__}')
print(f'  NumPy:            {numpy.__version__}')
print(f'  SciPy:            {scipy.__version__}')
" || {
    echo ""
    echo "ERROR: Missing dependencies. Install them with:"
    echo "  pip install -r requirements.txt"
    exit 1
}

echo ""

# Run the smoke test
python src/poc/v6/smoke_test.py
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Smoke test completed successfully."
else
    echo ""
    echo "❌ Smoke test failed (exit code $EXIT_CODE)."
fi

exit $EXIT_CODE
