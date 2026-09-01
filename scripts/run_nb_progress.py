#!/usr/bin/env python
"""Run a notebook in-place printing live per-cell progress.

Usage: python scripts/run_nb_progress.py <notebook.ipynb> [timeout_seconds]
"""
import sys
import time

import nbformat
from nbclient import NotebookClient


def main() -> int:
    path = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 1800

    nb = nbformat.read(path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    total = len(code_cells)
    print(f"[run] {path} — {total} code cells, timeout={timeout}s/cell", flush=True)

    client = NotebookClient(nb, timeout=timeout, kernel_name="python3")
    idx = {"n": 0}
    t0 = time.time()

    orig = client.execute_cell

    def traced(cell, cell_index, *a, **kw):
        if cell.cell_type == "code":
            idx["n"] += 1
            preview = cell.source.strip().splitlines()[0][:70] if cell.source.strip() else "<empty>"
            print(f"[cell {idx['n']:>2}/{total}] +{time.time()-t0:6.1f}s  {preview}", flush=True)
        result = orig(cell, cell_index, *a, **kw)
        nbformat.write(nb, path)  # persist after each cell
        return result

    client.execute_cell = traced
    try:
        with client.setup_kernel():
            client.execute()
    finally:
        nbformat.write(nb, path)
    print(f"[done] {idx['n']} cells in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
