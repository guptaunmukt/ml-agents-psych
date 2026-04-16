# Tests

This directory contains the standalone `ml-agents-psych` package tests.

## Current Local Validation Lane

Create an isolated Python 3.10 environment with `uv` and run the standalone tests:

```bash
uv venv --python python3.10 tests/.venv
uv pip install --python tests/.venv/bin/python -e . pytest
tests/.venv/bin/python -m pytest -q tests/test_psych_stats_writer.py tests/test_modal_curiosity_selection.py
```

## Coverage Focus
- stats-writer registration/import behavior
- resume-behavior correctness
- malformed-key safety
- modal curiosity branch resolution and weight normalization
