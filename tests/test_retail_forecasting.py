import numpy as np
import pandas as pd
import pytest

from retail_forecasting import (
    build_submission, chronological_holdout, prepare_sales,
    seasonal_naive, weighted_mae,
)


def test_wmae_ignores_series_labels_and_keeps_precision():
    actual = pd.Series([100, 200], index=[10, 11])
    predicted = pd.Series([0, 0], index=[0, 1])
    assert weighted_mae(actual, predicted, [False, False]) == 150.0
    assert weighted_mae(actual, [90, 180], [False, True]) == pytest.approx(110 / 6)
    assert weighted_mae([0], [1 / 3], [0]) == pytest.approx(1 / 3, abs=1e-15)


@pytest.mark.parametrize("actual,predicted,holidays", [
    ([], [], []), ([1, 2], [1], [0, 0]), ([1], [1], [0, 1]),
    ([[1]], [1], [0]), ([1], [[1]], [0]), ([1], [1], [[0]]),
    ([np.nan], [1], [0]), ([1], [np.inf], [0]), ([1], [1], [None]),
    ([1], [1], [2]), ([1], [1], ["False"]), ([1], [1], [pd.NA]),
])
def test_wmae_rejects_malformed_inputs(actual, predicted, holidays):
    with pytest.raises(ValueError):
        weighted_mae(actual, predicted, holidays)


def test_split_is_global_chronological_and_preserves_sparse_rows():
    dates = pd.date_range("2011-01-07", periods=10, freq="7D")
    rows = pd.DataFrame([
        (store, dept, date) for store, dept in [(1, 1), (2, 1)] for date in dates
    ] + [(3, 7, dates[2]), (3, 7, dates[-1])], columns=["Store", "Dept", "Date"])
    rows["row_id"] = np.arange(len(rows))
    rows = rows.sample(frac=1, random_state=17)
    training, validation = chronological_holdout(rows)
    assert training.Date.max() == dates[7]
    assert validation.Date.min() == dates[8]
    assert set(training.Date).isdisjoint(validation.Date)
    assert set(training.row_id).isdisjoint(validation.row_id)
    assert sorted(pd.concat([training, validation]).row_id) == list(range(len(rows)))
    assert set(pd.concat([training, validation]).query("Store == 3").row_id) == {20, 21}


@pytest.mark.parametrize("dates", [[], ["2011-01-07"], ["2011-01-07", None], ["bad", "2011-01-14"]])
def test_split_rejects_insufficient_or_invalid_dates(dates):
    with pytest.raises(ValueError):
        chronological_holdout(pd.DataFrame({"Date": dates}))


def test_split_uses_ceiling_for_distinct_dates():
    frame = pd.DataFrame({"Date": pd.date_range("2011-01-07", periods=6, freq="7D")})
    earlier, later = chronological_holdout(frame)
    assert len(earlier) == 4
    assert len(later) == 2


def history_and_future():
    history = pd.DataFrame([
        (1, 1, "2010-01-08", 10.0),
        (1, 1, "2010-12-31", 30.0),
        (2, 1, "2010-12-31", 50.0),
    ], columns=["Store", "Dept", "Date", "Weekly_Sales"])
    future = pd.DataFrame([
        (1, 1, "2011-01-07", 99999.0),
        (2, 1, "2011-01-07", -99999.0),
        (3, 1, "2011-01-07", 70000.0),
    ], columns=history.columns)
    return history, future


def test_baseline_lag_last_sale_and_training_median_ignore_validation_targets():
    history, future = history_and_future()
    np.testing.assert_array_equal(seasonal_naive(history, future), [10, 50, 30])
    future["Weekly_Sales"] = [np.nan, np.inf, -np.inf]
    np.testing.assert_array_equal(seasonal_naive(history, future), [10, 50, 30])
    with pytest.raises(ValueError, match="duplicate"):
        seasonal_naive(pd.concat([history, history.iloc[:1]]), future)
    with pytest.raises(ValueError, match="strictly after"):
        seasonal_naive(history, history)


def test_submission_maps_distinct_predictions_to_reversed_template():
    _, rows = history_and_future()
    rows = rows.iloc[[1, 2, 0]]
    template = pd.DataFrame({"Id": ["1_1_2011-01-07", "3_1_2011-01-07", "2_1_2011-01-07"], "Weekly_Sales": 0})
    actual = build_submission(rows, [50, 30, 10], template)
    assert actual.Id.tolist() == template.Id.tolist()
    assert actual.Weekly_Sales.tolist() == [10, 30, 50]


@pytest.mark.parametrize("failure", ["duplicate_rows", "duplicate_ids", "missing_id", "extra_id", "length", "nonfinite"])
def test_submission_rejects_ambiguous_or_incomplete_mapping(failure):
    _, rows = history_and_future()
    template = pd.DataFrame({"Id": ["1_1_2011-01-07", "2_1_2011-01-07", "3_1_2011-01-07"]})
    predictions = [10, 50, 30]
    if failure == "duplicate_rows":
        rows = pd.concat([rows.iloc[:1], rows.iloc[:1], rows.iloc[2:]])
    elif failure == "duplicate_ids":
        template.loc[1, "Id"] = template.loc[0, "Id"]
    elif failure == "missing_id":
        template = template.iloc[:2]
    elif failure == "extra_id":
        template.loc[0, "Id"] = "9_9_2011-01-07"
    elif failure == "length":
        predictions = [10, 50]
    else:
        predictions = [10, np.nan, 30]
    with pytest.raises(ValueError):
        build_submission(rows, predictions, template)


def test_preparation_preserves_official_flags_and_rejects_bad_joins():
    sales = pd.DataFrame({"Store": [1, 1], "Dept": [1, 2], "Date": ["2012-04-06"] * 2,
                          "IsHoliday": [False, False], "Weekly_Sales": [10, 20]})
    features = pd.DataFrame({"Store": [1], "Date": ["2012-04-06"], "IsHoliday": [False]})
    stores = pd.DataFrame({"Store": [1], "Type": ["A"], "Size": [1000]})
    result = prepare_sales(sales, features, stores)
    assert result.IsEaster.all()
    assert not result.IsHoliday.any()
    assert sales.IsHoliday.tolist() == [False, False]
    assert len(result) == len(sales)
    with pytest.raises(ValueError, match="unique"):
        prepare_sales(sales, pd.concat([features, features]), stores)
    with pytest.raises(ValueError, match="matching features"):
        prepare_sales(sales, features.assign(Store=2), stores)
    with pytest.raises(ValueError, match="matching store"):
        prepare_sales(sales, features, stores.assign(Store=2))
    with pytest.raises(ValueError, match="disagree"):
        prepare_sales(sales, features.assign(IsHoliday=True), stores)
