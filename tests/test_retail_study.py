import numpy as np
import pandas as pd
import pytest

from retail_study import (
    CANDIDATES, Forecaster, metrics, pooled_metrics, select_candidate, study_splits,
)


def panels():
    dates = pd.date_range("2010-02-05", periods=182, freq="7D")
    training = pd.DataFrame({"Store": 1, "Dept": 1, "Date": dates[:143],
                             "Weekly_Sales": np.arange(143), "IsHoliday": False})
    sparse = training.iloc[[1, 60, 140]].assign(Store=2)
    training = pd.concat([training, sparse]).sample(frac=1, random_state=42)
    external = training.iloc[:39].copy()
    external["Date"] = dates[143:]
    return training, external


def test_exact_study_splits_keep_sparse_rows_and_final_out_of_development():
    training, external = panels()
    splits = study_splits(training, external)
    assert [(a.Date.nunique(), b.Date.nunique()) for a, b in splits.values()] == [(52, 26), (78, 26), (104, 39)]
    final_dates = set(splits["final_holdout"][1].Date)
    for name in ["dev_1", "dev_2"]:
        a, b = splits[name]
        assert a.Date.max() < b.Date.min()
        assert final_dates.isdisjoint(set(a.Date) | set(b.Date))
    a, b = splits["final_holdout"]
    assert len(a) + len(b) == len(training)
    assert len(pd.concat([a, b]).query("Store == 2")) == 3
    mutated = training.copy()
    mutated.loc[mutated.Date.isin(final_dates), "Weekly_Sales"] = -99999999
    changed = study_splits(mutated, external)
    for name in ["dev_1", "dev_2"]:
        for original, altered in zip(splits[name], changed[name]):
            pd.testing.assert_frame_equal(original, altered)


def test_study_refuses_to_silently_change_protocol():
    training, external = panels()
    with pytest.raises(ValueError, match="143"):
        study_splits(training.iloc[1:], external.iloc[:38])
    external.loc[external.index[0], "Date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="weekly"):
        study_splits(training, external)


def test_pooled_wmae_uses_weighted_error_denominators_and_baseline_can_win():
    records = []
    for name in CANDIDATES:
        offset = 0 if name == "last_observed" else 10
        # Unequal fold sizes and holiday counts: (2 + 5*4 + 6) / (1+5+1) = 4.
        for date, holiday, error in [("2011-01-07", False, 2), ("2011-07-08", True, 4), ("2011-07-15", False, 6)]:
            records.append(dict(model=name, Date=date, IsHoliday=holiday, Weekly_Sales=0, prediction=error + offset))
    pooled = pooled_metrics(pd.DataFrame(records))
    assert pooled.set_index("model").loc["last_observed", "WMAE"] == 4.0
    assert select_candidate(pooled) == "last_observed"
    assert select_candidate(pooled.assign(WMAE=4.0)) == "seasonal_naive"
    absent = metrics([1, 2], [0, 0], [False, False])
    assert absent["holiday_MAE"] is None


def test_hgb_categories_are_training_owned_and_unknowns_become_missing():
    frame = pd.DataFrame({"Store": [1, 2] * 20, "Dept": 1, "Type": [3, 2] * 20,
                          "Size": 1000, "Week": np.arange(40) % 52 + 1, "Year": 2010,
                          "IsHoliday": False, "IsEaster": False,
                          "Date": pd.date_range("2010-01-01", periods=40, freq="7D"),
                          "Weekly_Sales": np.arange(40) - 3})
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        model = Forecaster("hist_gradient_boosting").fit(frame)
        future = frame.iloc[:1].assign(Store=99, Dept=99, Weekly_Sales=999999)
        encoded = model._encode(future)
        assert np.isnan(encoded[0, 0]) and np.isnan(encoded[0, 1])
        assert model.categories_["Store"] == [1, 2]
        assert np.isfinite(model.predict(future)).all()
