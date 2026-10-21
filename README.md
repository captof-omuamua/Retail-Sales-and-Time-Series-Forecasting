# Retail Sales and Time Series Forecasting

[![Retail checks](https://github.com/captof-omuamua/Retail-Sales-and-Time-Series-Forecasting/actions/workflows/retail-checks.yml/badge.svg)](https://github.com/captof-omuamua/Retail-Sales-and-Time-Series-Forecasting/actions/workflows/retail-checks.yml)

We reran the retail forecasting code with improvements to error calculation, chronological validation, baseline comparison, and submission alignment.

| Method | Pooled development WMAE | Final holdout WMAE |
| --- | ---: | ---: |
| Random Forest | 1912.519432 | **1607.077553** |
| Seasonal naive | 2032.328260 | 1805.616138 |
| Last observed | 3745.397354 | 3201.241475 |
| Histogram gradient boosting | 3616.818659 | Not evaluated (not selected) |

Random Forest was selected using development results before evaluation on the final 39-week holdout (2012-02-03 through 2012-10-26). These are local chronological measurements, not Kaggle scores. See the [result and verification record](reports/real-run-v1/README.md), [protocol](reports/real-run-v1/protocol.json), and [data provenance](reports/real-run-v1/source_manifest.json).

| Notebook | What it covers | Stage |
| --- | --- | --- |
| [02_retail_sales_forecasting.ipynb](02_retail_sales_forecasting.ipynb) | Audited inputs, chronological development/final evaluation, and ID-matched submission. | Project experiment |
| [64_lstm_sequence_prediction.ipynb](64_lstm_sequence_prediction.ipynb) | A small Keras LSTM experiment for predicting the next value of a numeric sequence. | Learning experiment |

## Retail setup and regression checks

Verified with **Python 3.14.0** and the versions in [requirements-retail.txt](requirements-retail.txt). TensorFlow/Keras is optional and separate for the unchanged LSTM notebook. The retail workflow downloads no data and installs no packages on execution.

From the repository root:

```text
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-retail.txt
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts/smoke_test.py
```

On Linux/macOS use `.venv/bin/python`. The smoke command generates clearly marked synthetic CSVs and executes **every cell of the actual notebook** in a fresh kernel using that interpreter. It tests the full protocol with 143 labeled and 39 external weeks, sparse/unseen series, negative and zero targets, official holidays, separate Easter features, and reversed submission IDs. A seasonal baseline wins this fixture and remains the primary submission. Synthetic scores verify code execution only. Generated evidence goes under ignored `outputs/synthetic-smoke/`; `--work-dir PATH` changes that location. Each study needs a **fresh output directory** to preserve prior evidence.

## Data and provenance

Supply these five CSVs in an ignored local directory such as `data/`:

| File | Required columns |
| --- | --- |
| `train.csv` | Store, Dept, Date, IsHoliday, Weekly_Sales |
| `test.csv` | Store, Dept, Date, IsHoliday |
| `features.csv` | Store, Date, IsHoliday, Temperature, Fuel_Price, CPI, Unemployment, MarkDown1–MarkDown5 |
| `stores.csv` | Store, Type (A/B/C), Size |
| `sampleSubmission.csv` | Id (`Store_Dept_YYYY-MM-DD`), Weekly_Sales |

Dates use `YYYY-MM-DD`. Unique panel keys are Store/Dept/Date; feature keys are Store/Date; stores key is Store. The workflow validates many-to-one left joins without dropping rows, official holiday agreement, finite selected inputs/targets, exact weekly dates and matching submission IDs. Unused covariate missingness is audited rather than used to discard data. Every valid finite target, including negative/zero sales and large values, is retained.

The fixed real-study protocol expects 421,570 labeled rows, 115,064 external rows, 8,190 feature rows, 45 stores, **143 labeled weeks** (2010-02-05 through 2012-10-26) and **39 external weeks** (2012-11-02 through 2013-07-26). Mismatches fail instead of silently changing the experiment.

The original competition is [Walmart Recruiting: Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting). The fresh study uses public-mirror files whose actual URLs, pinned repository commits, sizes and SHA-256 hashes are recorded in the run's `source_manifest.json`. Train/test/features/stores were byte-identical across two independently retrieved mirrors. No authenticated official-file checksums were available: mirror agreement corroborates the data but does **not** prove official authenticity. No raw competition CSVs or row-level targets are committed. Copy the files identified by the supplied manifest to a local directory and retain that manifest; real runs reject hash changes.

## Reproduce the real study

Run one explicit command against those locally supplied CSVs and their provenance manifest:

```text
python -m retail_study --data-dir data --provenance data/source_manifest.json --input-kind real_public_mirror --output-dir outputs/real-run-1
```

Use `--input-kind real_official` only for documented official inputs, or `synthetic` for fixtures. Real input kinds require a provenance manifest; the runner never downloads inputs or silently substitutes synthetic data. Python/package versions, file/code hashes, Git revision/dirty diff hash, seed, threads, and exact resolved date/row splits are captured in the run directory before fitting.

The notebook calls this **same** study implementation. Open it from the repository root in a Jupyter-compatible editor using the installed Python kernel. Set its `DATA_DIR`, fresh `OUTPUT_DIR`, `INPUT_KIND`, and real-data `PROVENANCE_PATH` variables, or set `RETAIL_DATA_DIR`, `RETAIL_OUTPUT_DIR`, `RETAIL_INPUT_KIND`, and `RETAIL_PROVENANCE` before kernel startup. The cell headed “Run the predeclared study” reveals final holdout outcomes; earlier plots use only the first 52 weeks.

## Methodology

| Stage | Training weeks | Later scoring weeks | Purpose |
| --- | --- | --- | --- |
| Development 1 | First 52 | Next 26 | Candidate comparison |
| Development 2 | First 78 | Next 26 | Candidate comparison |
| Final holdout | First 104 | Final 39: 2012-02-03–2012-10-26 | One-time final measurement |
| External forecast | All 143 | 39 supplied external weeks | Unscored keyed submission |

All rows for one date stay together, including sparse series. The final 39-week holdout participates in neither development fitting nor scoring. Selection minimizes **pooled** development weighted mean absolute error: `sum(weight * abs(actual - prediction)) / sum(weight)` using original official holiday flags (5 for holidays, 1 otherwise), full precision and positional arrays. The pooled denominator accounts for unequal fold sizes/holiday counts. Ties within 1e-10 prefer seasonal naive, last observed, histogram boosting, then Random Forest. `selection.json` freezes the winner before final scoring. Final holdout evaluates that winner plus the two declared baseline references; no unselected learned model is reconsidered there.

Four fixed candidates are compared:

- **Last observed:** latest training sale for each Store/Dept; unseen series use the global training median.
- **Seasonal naive:** training sale exactly 52 weeks (364 days) earlier; missing lags use the latest series sale, then the training median.
- **Random Forest:** 56 trees, max depth 20, minimum leaf 2, minimum split 3, all features per split, squared-error fitting, official holiday sample weights, seed 42, 2 threads. It estimates conditional means despite absolute-error evaluation. Numeric threshold splits for nominal Store/Dept/Type are a baseline limitation.
- **Histogram gradient boosting:** absolute-error loss, 100 iterations, learning rate 0.1, 31 leaves, minimum leaf 20, L2 1.0, **no early stopping**, holiday sample weights, seed 42, 2-thread cap. Store/Dept/Type categories are fitted only on each training prefix; unseen values become missing categories.

Both learned models use Store, Dept, Type, Size, ISO week, calendar year, official IsHoliday and separate IsEaster. The fixed historical Easter mapping covers 2010–2013 only and never changes official score weights. Future external covariates/target-derived statistics are excluded. No random split, KFold or tuning search is used, and there is no requirement for a learned model to beat the baselines.

After final evaluation, the selected method is reconstructed/refitted on **all labeled rows**, persisted/reloaded and checked for identical external predictions. If a baseline wins, it produces the primary submission. Export maps predictions by `Store_Dept_YYYY-MM-DD` into the sample template's original order, rejecting duplicate/missing/extra IDs.

## Run evidence and interpretation

The run directory contains `data_audit.json`, `source_manifest.json`, `protocol.json`, `selection.json`, `run_status.json`, fold/pooled/final metrics, final week-level errors, row-level development/final predictions, independent metrics recomputed from the saved final CSV, `final_fit.json`, `selected_model.pkl`, `Submission.csv`, and a concise `report.md`. Full precision is retained in machine-readable files; rounding is only for display. Row-level artifacts stay local because they contain original targets. Subgroup MAE is null when no matching rows exist.

Git attributes preserve notebook/evidence bytes and use LF for Python/requirements so recorded SHA-256 hashes survive checkout across platforms.

The scores above come from the improved rerun. The final score is a **local chronological holdout** measurement, not an official hidden-test/leaderboard score. External labels are unavailable, so official competition performance remains unknown.

Limitations include short history, only two 26-week development blocks versus a 39-week task, one final holdout, and correlated weeks/stores/departments. The final Feb–Oct holdout includes Super Bowl and Labor Day weeks but **no Thanksgiving or Christmas**; the external Nov–Jul period contains both. Sparse/unseen series and future calendar/economic changes may behave differently. Week-level error spread is descriptive; no narrow confidence interval assuming independent panel rows or forecast-quality guarantee is claimed.

## About this collection

These notebooks are organised from an existing Colab archive. They remain experiments and learning exercises, not a production forecasting service. Dataset files and model weights are supplied separately. The unrelated LSTM notebook has not been changed or rerun.

[Browse all AI, ML and automation topics](https://github.com/captof-omuamua/Colab-AI-ML-Projects).

## Contributing and maintenance

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, regression checks, notebook smoke verification, and how to report reproducible bugs. [Retail checks](https://github.com/captof-omuamua/Retail-Sales-and-Time-Series-Forecasting/actions/workflows/retail-checks.yml) run the regression suite and execute the actual notebook with synthetic inputs on relevant pushes and pull requests. These automated checks do not rerun the real-data study or establish forecast quality.

## License

Repository code is available under the [MIT License](LICENSE). External datasets, models, dependencies, and any separately attributed third-party components retain their own licenses and access conditions. The license does not grant rights to the competition data supplied separately.

