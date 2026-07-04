# Test Fixtures

Representative result JSONs for integration testing without running actual experiments.

## Files

| File | Description | Use case |
|------|-------------|----------|
| `complete_run.json` | Full 4-section noiseless pipeline result (N=4, p=1, chain_1d) | Test result loading, inspection, index |
| `interrupted_run.json` | Run interrupted during Section 3 (Sections 1+2 complete) | Test resume, partial save |
| `failed_run.json` | Run where Section 2 (VQE) returned pass=False | Test failure detail, analysis |

## Usage in tests

```python
from pathlib import Path
FIXTURES = Path(__file__).parent / "fixtures"

def test_inspect_complete_run():
    data = load_result(FIXTURES / "complete_run.json")
    assert data["summary"]["all_passed"] is True
```
