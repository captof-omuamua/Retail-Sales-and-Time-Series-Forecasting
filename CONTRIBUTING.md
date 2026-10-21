# Contributing

This repository is a small forecasting experiment and learning project. Contributions that improve reproducibility, validation, documentation, or baseline comparisons are welcome. It is not a production forecasting service.

## Reproduce the checks

Use Python 3.14 and install the pinned retail environment in a virtual environment:

```text
python -m venv .venv
python -m pip install -r requirements-retail.txt
python -m pytest -q
python scripts/smoke_test.py --work-dir outputs/contributor-smoke-1
```

Activate the virtual environment before the last three commands, or use its Python executable directly. On Windows that is `.venv\Scripts\python`; on Linux/macOS it is `.venv/bin/python`.

The smoke check executes the actual retail notebook in a fresh kernel with deterministic synthetic inputs. Use a **new work directory for each run**. Synthetic scores establish execution only; they do not measure forecasting quality. No competition data, credentials, paid APIs, or model training downloads are needed for these checks.

## Before opening a pull request

1. Describe the concrete problem and provide a minimal example.
2. Keep the change focused. Add a regression test when changing data validation, metrics, splitting, selection, or submission mapping.
3. Run regression tests and the notebook smoke check when changing the retail workflow.
4. Explain changes to the evaluation protocol before running a real-data comparison. Freeze model selection before viewing final holdout outcomes.
5. State exactly which checks ran and distinguish synthetic checks from real-data results.

Preserve the official holiday flags and original finite targets. Keep all rows for a date in the same chronological split, learn preprocessing only from training data, and match submission predictions by ID. Never adjust the method after inspecting final holdout errors and then present those errors as an untouched evaluation.

## Reporting a bug

Include the Git commit, Python/package versions, command, expected and actual behavior, and the smallest synthetic input that reproduces the issue. Redact local usernames and paths when unnecessary.

Do not attach raw competition CSVs, row-level targets/predictions, credentials, or private documents. Aggregate output and a synthetic reproduction are usually sufficient.

## Useful contribution areas

- Add a synthetic regression case for a validation edge case, with a documented expected outcome.
- Improve setup instructions after reproducing them in a clean environment.
- Propose a separately versioned protocol for a different dataset or forecasting horizon; preserve the published study evidence.

Check existing issues and pull requests before starting. Make sure you have the rights to submit your contribution and retain any applicable third-party notices.

