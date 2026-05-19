#!/usr/bin/env bash
# Pre-push hook: run fast test suite before pushing.
# Prevents broken code from reaching the remote.
#
# Install: cp scripts/hooks/pre-push-tests.sh .git/hooks/pre-push && chmod +x .git/hooks/pre-push
# Or: make hooks-install (which also sets up pre-commit hooks)

set -e

# Use project venv if available, otherwise system python3
REPO_ROOT="$(git rev-parse --show-toplevel)"
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
else
    PYTHON="python3"
fi

echo "🧪 Running test suite before push..."
echo ""

# Run fast tests only (excludes @pytest.mark.slow)
$PYTHON -m pytest tests/ -x -q -m "not slow" --tb=short 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ Tests failed. Push aborted."
    echo "   Fix the failing tests or use 'git push --no-verify' to skip."
    exit 1
fi

echo ""
echo "✅ All tests passed. Pushing..."
exit 0
