"""Reproducible fixed-protocol retail study; no downloads or synthetic fallback."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from retail_forecasting import (
    MODEL_FEATURES, RF_SETTINGS, build_submission, make_random_forest,
    prepare_sales, seasonal_naive, weighted_mae,
)

ROOT = Path(__file__).resolve().parent
CANDIDATES = ["seasonal_naive", "last_observed", "hist_gradient_boosting", "random_forest"]
CATEGORICAL = ["Store", "Dept", "Type"]
HGB_SETTINGS = dict(loss="absolute_error", max_iter=100, learning_rate=0.1,
                    max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
                    early_stopping=False, random_state=42,
                    categorical_features=[name in CATEGORICAL for name in MODEL_FEATURES])
FILES = {"train": "train.csv", "test": "test.csv", "features": "features.csv",
         "stores": "stores.csv", "template": "sampleSubmission.csv"}
REQUIRED = {
    "train": {"Store", "Dept", "Date", "IsHoliday", "Weekly_Sales"},
    "test": {"Store", "Dept", "Date", "IsHoliday"},
    "features": {"Store", "Date", "IsHoliday", "Temperature", "Fuel_Price", "CPI", "Unemployment",
                 "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"},
    "stores": {"Store", "Type", "Size"}, "template": {"Id", "Weekly_Sales"},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_inputs(data_dir):
    directory = Path(data_dir)
    missing = [filename for filename in FILES.values() if not (directory / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing CSVs in {directory}: {missing}")
    tables = {name: pd.read_csv(directory / filename) for name, filename in FILES.items()}
    for name, columns in REQUIRED.items():
        absent = columns - set(tables[name])
        if absent:
            raise ValueError(f"{FILES[name]} is missing required columns: {sorted(absent)}")
    training = prepare_sales(tables["train"], tables["features"], tables["stores"])
    external = prepare_sales(tables["test"], tables["features"], tables["stores"])
    # Validates ID uniqueness/equality before any expensive model fitting.
    build_submission(external, np.zeros(len(external)), tables["template"])
    return tables, training, external


def study_splits(training, external):
    """Exact global weekly partitions, fixed before looking at model errors."""
    dates = pd.DatetimeIndex(sorted(training.Date.unique()))
    future_dates = pd.DatetimeIndex(sorted(external.Date.unique()))
    if len(dates) != 143 or len(future_dates) != 39:
        raise ValueError("Study protocol requires exactly 143 labeled and 39 external weekly dates")
    for values in [dates, future_dates]:
        if not (np.diff(values.to_numpy()) == np.timedelta64(7, "D")).all():
            raise ValueError("Study dates must be consecutive weekly dates")
    if future_dates[0] - dates[-1] != pd.Timedelta(weeks=1):
        raise ValueError("External dates must begin one week after all labeled dates")
    result = {}
    for name, start, end in [("dev_1", 52, 78), ("dev_2", 78, 104), ("final_holdout", 104, 143)]:
        train = training.loc[training.Date < dates[start]].copy()
        validation = training.loc[training.Date.between(dates[start], dates[end - 1])].copy()
        assert train.Date.max() < validation.Date.min()
        result[name] = (train, validation)
    return result


def split_record(name, earlier, later):
    known = pd.MultiIndex.from_frame(earlier[["Store", "Dept"]].drop_duplicates())
    requested = pd.MultiIndex.from_frame(later[["Store", "Dept"]])
    return {
        "split": name, "training_rows": len(earlier), "validation_rows": len(later),
        "training_weeks": earlier.Date.nunique(), "validation_weeks": later.Date.nunique(),
        "training_start": earlier.Date.min().date().isoformat(),
        "training_end": earlier.Date.max().date().isoformat(),
        "validation_start": later.Date.min().date().isoformat(),
        "validation_end": later.Date.max().date().isoformat(),
        "unseen_series": len(requested[~requested.isin(known)].unique()),
        "unseen_series_rows": int((~requested.isin(known)).sum()),
    }


class Forecaster:
    """Training-owned state for both learned candidates and both simple baselines."""

    def __init__(self, name):
        if name not in CANDIDATES:
            raise ValueError(f"Unknown candidate {name}")
        self.name = name

    def _encode(self, frame):
        result = frame[MODEL_FEATURES].astype(float).copy()
        if self.name == "hist_gradient_boosting":
            for column, categories in self.categories_.items():
                # Training-owned codes; unknowns map to NaN and HGB's missing route.
                result[column] = frame[column].map({value: code for code, value in enumerate(categories)})
        return result.to_numpy()

    def fit(self, training):
        self.training_rows_ = len(training)
        self.training_end_ = training.Date.max().isoformat()
        self.history_ = training[["Store", "Dept", "Date", "Weekly_Sales"]].copy()
        if self.name in ["seasonal_naive", "last_observed"]:
            return self
        self.categories_ = {column: sorted(training[column].unique().tolist()) for column in CATEGORICAL}
        self.model_ = make_random_forest() if self.name == "random_forest" else HistGradientBoostingRegressor(**HGB_SETTINGS)
        self.model_.fit(self._encode(training), training.Weekly_Sales.to_numpy(),
                        sample_weight=np.where(training.IsHoliday.to_numpy(), 5.0, 1.0))
        return self

    def predict(self, rows):
        if self.name in ["seasonal_naive", "last_observed"]:
            return seasonal_naive(self.history_, rows, use_seasonal_lag=self.name == "seasonal_naive")
        return self.model_.predict(self._encode(rows))


def metrics(actual, predicted, holidays):
    score = weighted_mae(actual, predicted, holidays)
    errors = np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))
    flags = np.asarray(holidays, dtype=bool)
    weights = np.where(flags, 5.0, 1.0)
    return {"WMAE": score, "MAE": float(errors.mean()),
            "holiday_MAE": float(errors[flags].mean()) if flags.any() else None,
            "nonholiday_MAE": float(errors[~flags].mean()) if (~flags).any() else None,
            "rows": len(errors), "holiday_rows": int(flags.sum()),
            "sum_weights": float(weights.sum()),
            "weighted_absolute_error": float(np.dot(errors, weights))}


def pooled_metrics(predictions):
    """Pool rows, not per-fold scores; holiday/row counts may differ by fold."""
    result = []
    for name in CANDIDATES:
        rows = predictions.loc[predictions.model == name]
        record = {"model": name, **metrics(rows.Weekly_Sales, rows.prediction, rows.IsHoliday)}
        record["weeks"] = rows.Date.nunique()
        result.append(record)
    return pd.DataFrame(result)


def select_candidate(pooled):
    """Only development evidence enters selection. First predeclared tie wins."""
    scores = pooled.set_index("model").WMAE
    if len(scores) != len(CANDIDATES) or scores.index.duplicated().any() or set(scores.index) != set(CANDIDATES) or not np.isfinite(scores.to_numpy()).all():
        raise ValueError("Selection requires one finite development score for each candidate")
    best = scores.min()
    return next(name for name in CANDIDATES if scores[name] <= best + 1e-10)


def predict_block(name, earlier, later, split):
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        model = Forecaster(name).fit(earlier)
        fitted = time.perf_counter()
        predictions = model.predict(later)
        finished = time.perf_counter()
    record = {"model": name, **split_record(split, earlier, later),
              **metrics(later.Weekly_Sales, predictions, later.IsHoliday),
              "fit_seconds": fitted - started, "predict_seconds": finished - fitted}
    rows = later[["Store", "Dept", "Date", "IsHoliday", "Weekly_Sales"]].copy()
    rows["model"], rows["split"], rows["prediction"] = name, split, predictions
    return record, rows


def audit_inputs(tables, training, external, directory, input_kind, provenance):
    audit = {"input_kind": input_kind, "provenance": provenance, "files": {}}
    for name, frame in tables.items():
        record = {"filename": FILES[name], "sha256": sha256(directory / FILES[name]),
                  "bytes": (directory / FILES[name]).stat().st_size, "rows": len(frame),
                  "columns": list(frame), "dtypes": {c: str(t) for c, t in frame.dtypes.items()},
                  "missing": {c: int(n) for c, n in frame.isna().sum().items()}}
        key = {"train": ["Store", "Dept", "Date"], "test": ["Store", "Dept", "Date"],
               "features": ["Store", "Date"], "stores": ["Store"], "template": ["Id"]}[name]
        record["duplicate_keys"] = int(frame.duplicated(key).sum())
        if "Store" in frame:
            record["stores"] = frame.Store.nunique()
        if "Dept" in frame:
            record["departments"] = frame.Dept.nunique()
        if "Date" in frame:
            dates = pd.to_datetime(frame.Date, format="ISO8601")
            record.update(date_start=dates.min().date().isoformat(), date_end=dates.max().date().isoformat(),
                          weeks=dates.nunique())
        if name == "train":
            record["targets"] = {"negative": int((frame.Weekly_Sales < 0).sum()),
                                 "zero": int((frame.Weekly_Sales == 0).sum()),
                                 "minimum": float(frame.Weekly_Sales.min()), "maximum": float(frame.Weekly_Sales.max())}
        audit["files"][FILES[name]] = record
        if provenance and "files" in provenance and FILES[name] in provenance["files"]:
            if provenance["files"][FILES[name]]["sha256"] != record["sha256"]:
                raise ValueError(f"Input hash differs from supplied provenance: {FILES[name]}")
    if input_kind != "synthetic":
        expected = {"train": 421570, "test": 115064, "features": 8190, "stores": 45, "template": 115064}
        if any(len(tables[name]) != count for name, count in expected.items()):
            raise ValueError("Real inputs do not match the predeclared dataset dimensions; review protocol before modeling")
    audit["row_preservation"] = {"train_before": len(tables["train"]), "train_after": len(training),
                                 "test_before": len(tables["test"]), "test_after": len(external)}
    audit["overlapping_train_test_keys"] = 0  # strict disjoint global dates are checked by study_splits
    audit["joins"] = "validated many-to-one; no missing matches; official holiday flags agree"
    return audit


def code_manifest():
    paths = ["retail_forecasting.py", "retail_study.py", "02_retail_sales_forecasting.ipynb", "requirements-retail.txt"]
    info = {"sha256": {name: sha256(ROOT / name) for name in paths}}
    try:
        info["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
        info["git_diff_sha256"] = hashlib.sha256(diff).hexdigest()
        info["git_status"] = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        info["git_revision"] = "unavailable; use file hashes"
    return info


def independent_recompute(path):
    """Read persisted CSV; calculate directly without weighted_mae/metrics helpers."""
    rows = pd.read_csv(path, parse_dates=["Date"])
    records = []
    for name, group in rows.groupby("model", sort=False):
        actual = group.Weekly_Sales.to_numpy(dtype=float)
        predicted = group.prediction.to_numpy(dtype=float)
        holidays = group.IsHoliday.to_numpy(dtype=bool)
        errors = np.abs(actual - predicted)
        weights = 1.0 + 4.0 * holidays.astype(float)
        records.append({"model": name, "WMAE": float((errors * weights).sum() / weights.sum()),
                        "MAE": float(errors.mean()),
                        "holiday_MAE": float(errors[holidays].mean()) if holidays.any() else None,
                        "nonholiday_MAE": float(errors[~holidays].mean()) if (~holidays).any() else None,
                        "rows": len(group), "holiday_rows": int(holidays.sum()), "sum_weights": float(weights.sum())})
    return pd.DataFrame(records)


def run_study(data_dir, output_dir, input_kind, provenance_path=None):
    if input_kind not in ["real_public_mirror", "real_official", "synthetic"]:
        raise ValueError("Explicit input_kind must be real_public_mirror, real_official or synthetic")
    directory, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("Run directory already contains a protocol; use a fresh directory to preserve its evidence")
    status = {"started_utc": utcnow(), "input_kind": input_kind, "status": "running"}
    write_json(output / "run_status.json", status)
    try:
        provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8-sig")) if provenance_path else None
        if input_kind != "synthetic" and provenance is None:
            raise ValueError("Real-data runs require an explicit provenance JSON manifest")
        tables, training, external = load_inputs(directory)
        splits = study_splits(training, external)
        audit = audit_inputs(tables, training, external, directory, input_kind, provenance)
        write_json(output / "data_audit.json", audit)
        if provenance:
            write_json(output / "source_manifest.json", provenance)
        protocol = {
            "declared_before_model_fitting_utc": utcnow(), "input_kind": input_kind,
            "selection": "lowest pooled development WMAE; ties within 1e-10 follow candidate order",
            "candidate_tie_order": CANDIDATES, "features": MODEL_FEATURES,
            "random_forest": RF_SETTINGS, "hist_gradient_boosting": HGB_SETTINGS,
            "baseline": "training-only 364-day lag / last series sale / global training median",
            "holiday_weights": {"original_official_holiday": 5, "other": 1},
            "target_handling": "all finite targets retained, including negative and zero sales",
            "hgb_categories": "training-only Store/Dept/Type categories; unknown becomes missing",
            "rf_category_limitation": "numeric-code threshold splits for nominal Store/Dept/Type",
            "easter": "separate fixed ISO-week mapping 2010:13,2011:16,2012:14,2013:13; false outside range",
            "thread_cap": 2, "splits": [split_record(name, *frames) for name, frames in splits.items()],
            "external": split_record("external_unscored", training, external),
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
            "code": code_manifest(),
        }
        write_json(output / "protocol.json", protocol)
        print("Input audit and protocol saved before model fitting.", flush=True)
        development_metrics, development_rows = [], []
        for split in ["dev_1", "dev_2"]:
            for name in CANDIDATES:
                print(f"Development {split}: {name}", flush=True)
                record, predictions = predict_block(name, *splits[split], split)
                development_metrics.append(record)
                development_rows.append(predictions)
                print(f"  WMAE={record['WMAE']:.6f}; fit={record['fit_seconds']:.2f}s", flush=True)
        development_rows = pd.concat(development_rows, ignore_index=True)
        development_rows.to_csv(output / "development_predictions.csv", index=False, float_format="%.17g")
        development_metrics = pd.DataFrame(development_metrics)
        development_metrics.to_csv(output / "development_fold_metrics.csv", index=False)
        pooled = pooled_metrics(development_rows)
        pooled.to_csv(output / "development_pooled_metrics.csv", index=False)
        winner = select_candidate(pooled)
        selection = {"selected_before_final_scoring_utc": utcnow(), "model": winner,
                     "pooled_development_WMAE": float(pooled.set_index("model").loc[winner, "WMAE"]),
                     "protocol_sha256": sha256(output / "protocol.json"),
                     "parameters": RF_SETTINGS if winner == "random_forest" else HGB_SETTINGS if winner == "hist_gradient_boosting" else protocol["baseline"],
                     "features": MODEL_FEATURES, "code": protocol["code"]}
        write_json(output / "selection.json", selection)
        print(f"Frozen development winner: {winner}. Revealing final holdout now.", flush=True)
        final_metrics, final_rows = [], []
        for name in dict.fromkeys([winner, "seasonal_naive", "last_observed"]):
            record, predictions = predict_block(name, *splits["final_holdout"], "final_holdout")
            final_metrics.append(record)
            final_rows.append(predictions)
        final_rows = pd.concat(final_rows, ignore_index=True)
        final_rows.to_csv(output / "final_holdout_predictions.csv", index=False, float_format="%.17g")
        final_metrics = pd.DataFrame(final_metrics)
        final_metrics.to_csv(output / "final_holdout_metrics.csv", index=False)
        recomputed = independent_recompute(output / "final_holdout_predictions.csv")
        for name in recomputed.model:
            left, right = final_metrics.set_index("model").loc[name], recomputed.set_index("model").loc[name]
            for metric in ["WMAE", "MAE", "holiday_MAE", "nonholiday_MAE", "rows", "holiday_rows", "sum_weights"]:
                if pd.isna(left[metric]) and pd.isna(right[metric]):
                    continue
                assert np.isclose(left[metric], right[metric], rtol=1e-12, atol=1e-8), f"Persisted metric mismatch: {name}/{metric}"
        recomputed.to_csv(output / "independent_recomputed_metrics.csv", index=False)
        weekly = []
        for (name, date), group in final_rows.groupby(["model", "Date"]):
            weekly.append({"model": name, "Date": date.date().isoformat(),
                           **metrics(group.Weekly_Sales, group.prediction, group.IsHoliday)})
        weekly = pd.DataFrame(weekly)
        weekly.to_csv(output / "final_weekly_metrics.csv", index=False)
        print(f"Refitting selected {winner} on all {len(training)} labeled rows.", flush=True)
        start = time.perf_counter()
        with threadpool_limits(limits=2):
            final_model = Forecaster(winner).fit(training)
            external_predictions = final_model.predict(external)
        with (output / "selected_model.pkl").open("wb") as stream:
            pickle.dump(final_model, stream)
        with (output / "selected_model.pkl").open("rb") as stream:
            loaded = pickle.load(stream)
        with threadpool_limits(limits=2):
            np.testing.assert_array_equal(external_predictions, loaded.predict(external))
        submission = build_submission(external, external_predictions, tables["template"])
        submission.to_csv(output / "Submission.csv", index=False, float_format="%.17g")
        write_json(output / "final_fit.json", {"model": winner, "training_rows": final_model.training_rows_,
                   "training_end": final_model.training_end_, "test_rows": len(external),
                   "fit_predict_seconds": time.perf_counter() - start, "save_reload_identical": True,
                   "submission_sha256": sha256(output / "Submission.csv")})
        selected = final_metrics.set_index("model").loc[winner]
        holiday_dates = splits["final_holdout"][1].loc[lambda x: x.IsHoliday, "Date"].drop_duplicates().sort_values()
        report = ["# Retail forecasting: fresh chronological study", "",
                  f"Input kind: {input_kind}. Run started {status['started_utc']}.",
                  "Public-mirror agreement corroborates inputs; it does not authenticate official Kaggle originals." if input_kind == "real_public_mirror" else "See data audit and supplied source manifest for provenance.",
                  "", "The selection protocol and input/code hashes were saved before fitting. No final holdout scores were used for selection.",
                  f"Development winner: **{winner}**, pooled development WMAE **{selection['pooled_development_WMAE']:.6f}**.",
                  f"Final 39-week local holdout ({selected['validation_start']} through {selected['validation_end']}): WMAE **{selected['WMAE']:.6f}**.",
                  "This is a local measurement on labeled data, not an official hidden-test or leaderboard score.", "",
                  "Final holdout comparisons (same rows and original holiday weights):", "",
                  final_metrics[["model", "WMAE", "MAE", "holiday_MAE", "nonholiday_MAE"]].to_csv(index=False),
                  "Official holiday weeks represented: " + ", ".join(date.date().isoformat() for date in holiday_dates) + ".",
                  "For the standard Walmart dates this final Feb–Oct period includes Super Bowl and Labor Day weeks, but no Thanksgiving or Christmas. The external Nov–Jul horizon contains Thanksgiving and Christmas, so holiday coverage differs materially.",
                  f"All {len(training)} labeled rows were retained for final refitting; submission contains {len(external)} keyed external predictions.",
                  "", "Limitations: short history, two 26-week development folds versus a 39-week forecast horizon, one final holdout, correlated stores/departments/weeks, sparse and unseen series, and differing calendar/holiday composition. The external period may behave differently. Week-level spread is descriptive, not an independent-row confidence interval. RF uses numeric threshold splits for nominal IDs; the Easter map has finite 2010–2013 coverage. No guarantee of forecasting quality or allegation of fabrication follows from these results.",
                  "", "Audit, protocol, selection, fold/pooled/final metrics, weekly errors, persisted row predictions and independent recomputation are saved alongside this report."]
        (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        status.update(status="complete", completed_utc=utcnow(), selected_model=winner,
                      final_holdout_WMAE=float(selected.WMAE))
        write_json(output / "run_status.json", status)
        print(f"Complete: {winner}; final local holdout WMAE={selected.WMAE:.6f}", flush=True)
        return {"protocol": protocol, "selection": selection, "development_metrics": development_metrics,
                "pooled_metrics": pooled, "final_metrics": final_metrics, "weekly_metrics": weekly,
                "submission": submission, "audit": audit}
    except Exception as exc:
        status.update(status="failed", failed_utc=utcnow(), error_type=type(exc).__name__, error=str(exc))
        write_json(output / "run_status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-kind", choices=["real_public_mirror", "real_official", "synthetic"], required=True)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()
    run_study(args.data_dir, args.output_dir, args.input_kind, args.provenance)


if __name__ == "__main__":
    # Ensure persisted objects use retail_study.Forecaster, even under python -m.
    from retail_study import main as canonical_main
    canonical_main()
