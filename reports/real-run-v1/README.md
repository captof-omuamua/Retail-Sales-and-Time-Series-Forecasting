# Newly executed retail study: 2026-10-19

The predeclared development comparison selected **Random Forest**. It achieved
**1607.0775528171794 WMAE** on the final **39-week local holdout**, 2012-02-03
through 2012-10-26: 115,588 rows, including 5,967 official holiday rows. Training
used only the earlier 305,982 rows through 2012-01-27. This is a local observable
result; official hidden-test/leaderboard performance remains unknown.

| Candidate | Pooled development WMAE | Final holdout WMAE |
| --- | ---: | ---: |
| Random Forest | 1912.519432 | **1607.077553** |
| Seasonal naive | 2032.328260 | 1805.616138 |
| Last observed | 3745.397354 | 3201.241475 |
| Histogram gradient boosting | 3616.818659 | Not evaluated; not selected |

Development validation covers two nonoverlapping 26-week blocks totaling 153,608
rows; its WMAE is pooled by weighted error numerator and denominator, rather than
averaging fold scores. The method and settings were frozen in
[selection.json](selection.json) before final holdout evaluation. No method was
changed after seeing final errors. The final RF error is 10.996% lower than the
seasonal reference on this particular holdout. This is a descriptive comparison,
without a significance or forecast-quality guarantee.

The final RF ordinary MAE was 1591.668534, official-holiday MAE 1681.700380, and
nonholiday MAE 1586.767831. Across the 39 individual weekly WMAEs, its minimum,
median and maximum were 1227.928371, 1546.196243 and 2452.505230 (sample standard
deviation 208.827695). These weeks and panel rows are correlated; that spread is
descriptive, not an independent-row confidence interval. See
[final weekly errors](final_weekly_metrics.csv).

All 421,570 finite targets were preserved, including 1,285 negative and 73 zero
sales (range -4988.94 through 693099.36). No clipping or row deletion was used.
There were 25 unseen Store/Dept series (112 rows) on the final holdout. Simple
baselines fall back only to their training history; HGB categories are learned
from the corresponding training prefix and unknown categories become missing.
Official IsHoliday flags stayed unchanged; Easter was a separate fixed feature.

After evaluation, a fresh selected RF fitted **all 421,570 labeled rows** and
produced **115,064** keyed external predictions, preserving sample-template order.
Fresh-process loading of its persisted model and exact submission mapping were
independently checked. The real run completed in approximately 40 seconds using
Python 3.14.0, NumPy 2.5.4, pandas 3.0.6 and scikit-learn 1.9.1 with seed 42 and
two-thread caps. [Final fit evidence](final_fit.json) records counts and artifact
hashes; the external submission remains unscored.

The five input files came from a pinned public GitHub mirror. Train, test,
features and stores were byte-identical to files retrieved from a second pinned
mirror. The [source manifest](source_manifest.json) records actual source URLs,
commits, file sizes and SHA-256 hashes. Authenticated official-file checksums were
not available: this is structurally audited public-mirror data, and agreement
between mirrors cannot establish official authenticity. The observed bugs and
withdrawn historical claims do not establish deliberate fabrication.

The final Feb–Oct period contains Super Bowl and Labor Day holiday weeks
(2012-02-10 and 2012-09-07), but **no Thanksgiving or Christmas**. The external
Nov–Jul horizon contains both. This and the short history, only two development
folds, shorter development horizons, one final holdout, sparse/new series and
changing future conditions limit generalization. The lower final error than
development error does not imply that the external period is easier.

Evidence retained here:

- [Data audit](data_audit.json), [predeclared protocol/code hashes](protocol.json),
  [frozen selection](selection.json), and [start/completion status](run_status.json).
- [Development fold metrics](development_fold_metrics.csv),
  [pooled comparison](development_pooled_metrics.csv), and
  [final metrics](final_holdout_metrics.csv), all at full precision.
- [Independent saved-row recomputation](independent_recomputed_metrics.csv) and
  [separate reviewer verification](independent_review.json). Both recomputations
  matched reported metrics; the separate reviewer also checked original
  targets/holiday flags and expected panel keys.
- [Artifact hashes/sizes](artifact_manifest.json),
  [captured run output](run.log), and the runner's [generated report](report.md).

Raw CSVs, row-level target/prediction tables, model binary and large external
submission remain local run artifacts. Their SHA-256 hashes and sizes are in the
artifact manifest. They are not included in this repository. Reproduce them using
the source manifest and the explicit `python -m retail_study` command in the main
README, choosing a fresh output directory.

Verification: **32 regression tests passed**, the actual notebook completed in a
fresh kernel on clearly labeled synthetic full-schema data, Python compilation
and notebook JSON validation passed, and `git diff --check` passed. The unrelated
LSTM notebook remained byte-for-byte unchanged (SHA-256
`209255f7a4e5872440dc595ab43193c9b96bda98e69782cd6eae9b777e2e0224`).
