"""Execute the actual notebook in a fresh kernel on explicitly synthetic CSVs."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retail_forecasting import submission_ids  # noqa: E402


def create_synthetic_inputs(directory):
    """Full schema, 143 labeled + 39 external weeks, sparse/unseen series."""
    directory.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2010-02-05", periods=182, freq="7D")
    holidays = pd.to_datetime([
        "2010-02-12", "2010-09-10", "2010-11-26", "2010-12-31",
        "2011-02-11", "2011-09-09", "2011-11-25", "2011-12-30",
        "2012-02-10", "2012-09-07", "2012-11-23", "2012-12-28",
    ])
    stores = pd.DataFrame({"Store": [1, 2, 3], "Type": ["A", "B", "C"], "Size": [100000, 70000, 30000]})
    feature_rows = []
    for store in stores.Store:
        for i, date in enumerate(dates):
            feature_rows.append({
                "Store": store, "Date": date, "IsHoliday": date in holidays,
                "Temperature": 60 + 20 * np.sin(i / 8), "Fuel_Price": 2.5 + i / 200,
                "CPI": 180 + i / 10, "Unemployment": 7.5 - i / 200,
                **{f"MarkDown{n}": np.nan if i % 3 else 10 * n for n in range(1, 6)},
            })
    features = pd.DataFrame(feature_rows)
    train_rows, test_rows = [], []
    for store, dept in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 9), (3, 10)]:
        for i, date in enumerate(dates):
            if i < 143 and store == 3:
                keep = [2, 57, 100, 142] if dept == 9 else [141, 142]
                if i not in keep:
                    continue
            row = dict(Store=store, Dept=dept, Date=date, IsHoliday=date in holidays)
            if i < 143:
                row["Weekly_Sales"] = float(500 * store + 100 * dept + 40 * np.sin(2 * np.pi * i / 52)
                                            + 75 * row["IsHoliday"])
                if (store, dept, i % 52) == (1, 1, 3):
                    row["Weekly_Sales"] = -10.0
                if (store, dept, i % 52) == (2, 2, 5):
                    row["Weekly_Sales"] = 0.0
                train_rows.append(row)
            else:
                test_rows.append(row)
    train = pd.DataFrame(train_rows).sample(frac=1, random_state=11).reset_index(drop=True)
    test = pd.DataFrame(test_rows).sample(frac=1, random_state=13).reset_index(drop=True)
    sorted_test = test.sort_values(["Store", "Dept", "Date"])
    template = pd.DataFrame({"Id": submission_ids(sorted_test).iloc[::-1].to_numpy(), "Weekly_Sales": 0.0})
    for filename, frame in [("train.csv", train), ("test.csv", test), ("features.csv", features),
                            ("stores.csv", stores), ("sampleSubmission.csv", template)]:
        frame.to_csv(directory / filename, index=False, date_format="%Y-%m-%d")
    (directory / "SYNTHETIC_DATA.txt").write_text(
        "Generated deterministic integration fixture. Not real Walmart observations or performance evidence.\n",
        encoding="utf-8",
    )


VERIFY_CELL = '''
# Extra integration assertions, appended in memory; source notebook stays clear.
import json
from retail_forecasting import submission_ids
from retail_study import Forecaster, independent_recompute
for training_detail, validation_detail in splits.values():
    assert training_detail.Date.max() < validation_detail.Date.min()
assert (train_detail.Store.eq(3)).sum() == 6  # sparse series were retained
assert (train_detail.Weekly_Sales < 0).sum() == 3
assert (train_detail.Weekly_Sales == 0).sum() == 3
for raw_name, prepared in [("train", train_detail), ("test", test_detail)]:
    raw = tables[raw_name].copy()
    raw["Date"] = pd.to_datetime(raw.Date)
    expected_flags = raw.set_index(["Store", "Dept", "Date"]).IsHoliday.sort_index()
    actual_flags = prepared.set_index(["Store", "Dept", "Date"]).IsHoliday.sort_index()
    pd.testing.assert_series_equal(expected_flags, actual_flags)
assert (train_detail.IsEaster & ~train_detail.IsHoliday).any()
assert study["selection"]["model"] == "seasonal_naive"  # a baseline winner is the primary output
fit = json.loads((OUTPUT_DIR / "final_fit.json").read_text())
assert fit["training_rows"] == len(tables["train"])
assert fit["save_reload_identical"]
sample_submission = tables["template"]
pd.testing.assert_series_equal(submission.Id, sample_submission.Id)
assert len(submission) == len(test_detail)
assert np.isfinite(submission.Weekly_Sales).all()
assert submission.Weekly_Sales.nunique() > 1
test_predictions = Forecaster(study["selection"]["model"]).fit(train_detail).predict(test_detail)
lookup = pd.Series(test_predictions, index=submission_ids(test_detail))
np.testing.assert_array_equal(submission.Weekly_Sales, sample_submission.Id.map(lookup))
assert submission.Id.tolist() == submission_ids(test_detail).iloc[::-1].tolist()
for filename in ["Submission.csv", "final_holdout_metrics.csv", "selected_model.pkl", "protocol.json", "selection.json"]:
    assert (OUTPUT_DIR / filename).is_file()
print("SYNTHETIC INTEGRATION ASSERTIONS PASSED")
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs" / "synthetic-smoke")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    input_dir, output_dir = work / "synthetic-data", work / "synthetic-results"
    create_synthetic_inputs(input_dir)
    for key, path in {
        "RETAIL_DATA_DIR": input_dir, "RETAIL_OUTPUT_DIR": output_dir,
        "MPLCONFIGDIR": work / "matplotlib", "JUPYTER_RUNTIME_DIR": work / "jupyter-runtime",
        "IPYTHONDIR": work / "ipython", "TEMP": work / "tmp", "TMP": work / "tmp",
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["RETAIL_INPUT_KIND"] = "synthetic"
    # Use this exact interpreter; do not silently execute a system Python kernel.
    kernel_dir = work / "kernels" / "retail-smoke"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Retail synthetic smoke", "language": "python",
    }), encoding="utf-8")
    notebook = nbformat.read(ROOT / "02_retail_sales_forecasting.ipynb", as_version=4)
    assert all(not cell.get("outputs") and cell.get("execution_count") is None
               for cell in notebook.cells if cell.cell_type == "code")
    original_cells = len(notebook.cells)
    notebook.cells.append(nbformat.v4.new_code_cell(VERIFY_CELL))
    manager = KernelManager(kernel_name="retail-smoke", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]))
    client = NotebookClient(notebook, km=manager, timeout=180, resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook, work / "synthetic-executed.ipynb")
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    results = pd.read_csv(output_dir / "final_holdout_metrics.csv")
    assert np.isfinite(results.WMAE).all()
    print(f"Executed every cell of the actual notebook ({original_cells} source cells) in a fresh kernel.")
    print("SYNTHETIC integration verification passed; these scores do not measure real forecasting quality.")
    print(results[["model", "WMAE", "training_rows", "validation_rows"]].to_string(index=False))
    print(f"Generated artifacts: {work}")


if __name__ == "__main__":
    main()
