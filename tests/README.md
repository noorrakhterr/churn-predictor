# Test Suite

## Overview

| File | What it covers |
|---|---|
| `conftest.py` | Shared pytest fixtures: three customer dicts (low/medium/high risk) and cached model/preprocessor artifacts |
| `test_preprocess.py` | Column classification, stratified splitting, NaN imputation, unknown-category handling, and raw data shape validation |
| `test_model.py` | Model loading, probability range, high-vs-low sanity check, single-customer inference path |
| `test_explain.py` | `explain_prediction` output shape and SHAP ordering, `make_recommendation` coverage and fallback behaviour |

## Running locally

```bash
# Full suite
pytest tests/ -v

# Single file
pytest tests/test_model.py -v

# Stop on first failure
pytest tests/ -x
```

Or via Make:

```bash
make test
```

## Prerequisites

The tests that hit the model require pre-built artifacts. Run the full pipeline first if artifacts are missing:

```bash
make all   # data → preprocess → train → explain
```

## Code quality

```bash
make lint      # ruff check only (no changes)
make format    # black + ruff --fix
```

Configuration lives in `pyproject.toml` at the project root (line-length 100, rules E/W/F/I/N/UP).
