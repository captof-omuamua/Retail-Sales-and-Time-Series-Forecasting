"""Small, tested helpers for the retail forecasting experiment."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

KEYS = ["Store", "Dept", "Date"]
MODEL_FEATURES = ["Store", "Dept", "IsHoliday", "IsEaster", "Size", "Week", "Type", "Year"]
RF_SETTINGS = dict(
    n_estimators=56, max_depth=20, max_features=1.0, criterion="squared_error",
    min_samples_split=3, min_samples_leaf=2, random_state=42, n_jobs=2,
)


def _vector(values, name):
    result = np.asarray(values)
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    return result


def _finite_vector(values, name):
    result = _vector(values, name)
    try:
        result = result.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _holiday_vector(values):
    result = _vector(values, "is_holiday")
    if pd.isna(result).any() or not np.isin(result, [False, True, 0, 1]).all():
        raise ValueError("is_holiday must contain only nonmissing boolean or 0/1 values")
    return result.astype(bool)


def weighted_mae(y_true, y_pred, is_holiday):
    """Unrounded WMAE, with official holidays weighted 5 and other rows 1.

    Inputs describe the SAME row order. Pandas index labels are deliberately
    ignored; predictions and targets must come from one corresponding frame.
    """
    actual = _finite_vector(y_true, "y_true")
    predicted = _finite_vector(y_pred, "y_pred")
    holidays = _holiday_vector(is_holiday)
    if not (len(actual) == len(predicted) == len(holidays)):
        raise ValueError("targets, predictions and holiday flags must have equal lengths")
    weights = np.where(holidays, 5.0, 1.0)
    return float(np.average(np.abs(actual - predicted), weights=weights))


def _dates(values):
    try:
        result = pd.to_datetime(values, errors="raise", format="ISO8601")
    except (TypeError, ValueError) as exc:
        raise ValueError("Date must contain valid dates") from exc
    if result.isna().any():
        raise ValueError("Date must not contain missing dates")
    if result.dt.tz is not None or (result != result.dt.normalize()).any():
        raise ValueError("Date must contain timezone-naive dates without a time component")
    return result


def _keyed_frame(frame, name):
    if not set(KEYS).issubset(frame.columns) or frame.empty:
        raise ValueError(f"{name} requires nonempty Store/Dept/Date columns")
    result = frame.copy()
    result["Date"] = _dates(result["Date"])
    for key in ["Store", "Dept"]:
        values = _finite_vector(result[key], key)
        if (values <= 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"{key} must contain positive integer identifiers")
        result[key] = values.astype(np.int64)
    if result.duplicated(KEYS).any():
        raise ValueError(f"{name} contains duplicate Store/Dept/Date keys")
    return result


def chronological_holdout(frame, validation_fraction=0.2):
    """Reserve the latest ceil(fraction * distinct dates) globally; keep all rows."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    dated = frame.copy()
    dated["Date"] = _dates(dated["Date"])
    dates = np.sort(dated["Date"].unique())
    if len(dates) < 2:
        raise ValueError("chronological validation requires at least two distinct dates")
    count = max(1, int(np.ceil(len(dates) * validation_fraction)))
    if count >= len(dates):
        raise ValueError("validation_fraction leaves no training dates")
    cutoff = dates[-count]
    earlier = dated.loc[dated["Date"] < cutoff].copy()
    later = dated.loc[dated["Date"] >= cutoff].copy()
    assert earlier["Date"].max() < later["Date"].min()
    return earlier, later


def prepare_sales(sales, features, stores):
    """Keep every sales row; fail on ambiguous or missing feature/store joins."""
    result = _keyed_frame(sales, "sales")
    result["IsHoliday"] = _holiday_vector(result["IsHoliday"])
    if "Weekly_Sales" in result:
        result["Weekly_Sales"] = _finite_vector(result["Weekly_Sales"], "Weekly_Sales")
    feature_rows = features.copy()
    feature_rows["Date"] = _dates(feature_rows["Date"])
    feature_rows["IsHoliday"] = _holiday_vector(feature_rows["IsHoliday"])
    feature_keys = ["Store", "Date"]
    if feature_rows[feature_keys].isna().any().any() or feature_rows.duplicated(feature_keys).any():
        raise ValueError("features require unique nonmissing Store/Date keys")
    if stores["Store"].isna().any() or stores.duplicated("Store").any():
        raise ValueError("stores require unique nonmissing Store keys")
    result = result.merge(
        feature_rows, on=feature_keys, how="left", validate="many_to_one",
        suffixes=("", "_feature"), indicator=True,
    )
    if (result["_merge"] != "both").any():
        raise ValueError("some sales rows have no matching features")
    if (result["IsHoliday"] != result["IsHoliday_feature"]).any():
        raise ValueError("sales and features disagree about official IsHoliday flags")
    result = result.drop(columns=["_merge", "IsHoliday_feature"])
    result = result.merge(stores, on="Store", how="left", validate="many_to_one", indicator=True)
    if (result["_merge"] != "both").any():
        raise ValueError("some sales rows have no matching store")
    result = result.drop(columns="_merge")
    result["Week"] = result["Date"].dt.isocalendar().week.astype(int)
    result["Year"] = result["Date"].dt.year
    # Finite historical mapping only; preserve the official scoring flags.
    easter_weeks = {2010: 13, 2011: 16, 2012: 14, 2013: 13}
    result["IsEaster"] = result["Week"].eq(result["Year"].map(easter_weeks))
    result["Type"] = result["Type"].map({"A": 3, "B": 2, "C": 1})
    if result["Type"].isna().any():
        raise ValueError("store Type must be A, B or C")
    for column in MODEL_FEATURES:
        _finite_vector(result[column], column)
    return result.sort_values(KEYS).reset_index(drop=True)


def seasonal_naive(training, requested, use_seasonal_lag=True):
    """52-week (364-day) lag from training only, then last series sale, then median.

    All requested dates must follow all training dates. No recursive forecasts or
    requested targets are used, including for horizons longer than one year.
    """
    history = _keyed_frame(training, "training")
    future = _keyed_frame(requested, "requested")
    history["Weekly_Sales"] = _finite_vector(history["Weekly_Sales"], "Weekly_Sales")
    if history["Date"].max() >= future["Date"].min():
        raise ValueError("baseline requests must be strictly after all training dates")
    lag_lookup = history.set_index(KEYS)["Weekly_Sales"]
    lag_keys = pd.MultiIndex.from_arrays([
        future["Store"], future["Dept"], future["Date"] - pd.Timedelta(weeks=52),
    ], names=KEYS)
    predictions = lag_lookup.reindex(lag_keys).to_numpy() if use_seasonal_lag else np.full(len(future), np.nan)
    latest = history.sort_values("Date").drop_duplicates(["Store", "Dept"], keep="last")
    last_lookup = latest.set_index(["Store", "Dept"])["Weekly_Sales"]
    series_keys = pd.MultiIndex.from_frame(future[["Store", "Dept"]])
    fallback = last_lookup.reindex(series_keys).to_numpy()
    predictions = np.where(np.isnan(predictions), fallback, predictions)
    return np.where(np.isnan(predictions), history["Weekly_Sales"].median(), predictions)


def make_random_forest():
    """Predeclared practical settings, not optimized on the final holdout."""
    return RandomForestRegressor(**RF_SETTINGS)


def submission_ids(rows):
    keys = _keyed_frame(rows, "prediction rows")
    return (keys["Store"].astype(str) + "_" + keys["Dept"].astype(str)
            + "_" + keys["Date"].dt.strftime("%Y-%m-%d"))


def build_submission(rows, predictions, template):
    """Map finite predictions by Id while retaining the template's exact order."""
    ids = submission_ids(rows)
    predicted = _finite_vector(predictions, "predictions")
    if len(ids) != len(predicted):
        raise ValueError("prediction rows and predictions must have equal lengths")
    if "Id" not in template or template.empty or template["Id"].isna().any():
        raise ValueError("template requires nonmissing Id values")
    if template["Id"].duplicated().any():
        raise ValueError("template contains duplicate Id values")
    if set(ids) != set(template["Id"]):
        raise ValueError("prediction and template Id sets differ (missing or extra IDs)")
    lookup = pd.Series(predicted, index=ids.to_numpy())
    result = template.copy()
    result["Weekly_Sales"] = result["Id"].map(lookup)
    return result
