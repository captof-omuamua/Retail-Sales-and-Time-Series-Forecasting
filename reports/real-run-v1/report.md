# Retail forecasting: fresh chronological study

Input kind: real_public_mirror. Run started 2026-10-19T02:44:59.088820+00:00.
Public-mirror agreement corroborates inputs; it does not authenticate official Kaggle originals.

The selection protocol and input/code hashes were saved before fitting. No final holdout scores were used for selection.
Development winner: **random_forest**, pooled development WMAE **1912.519432**.
Final 39-week local holdout (2012-02-03 through 2012-10-26): WMAE **1607.077553**.
This is a local measurement on labeled data, not an official hidden-test or leaderboard score.

Final holdout comparisons (same rows and original holiday weights):

model,WMAE,MAE,holiday_MAE,nonholiday_MAE
random_forest,1607.0775528171794,1591.6685343305883,1681.7003795654655,1586.7678308110387
seasonal_naive,1805.6161376348095,1804.0669547876944,1813.1185235461708,1803.574250736629
last_observed,3201.241475160624,3128.45886977887,3553.7132277526393,3105.310998896197

Official holiday weeks represented: 2012-02-10, 2012-09-07.
For the standard Walmart dates this final Feb–Oct period includes Super Bowl and Labor Day weeks, but no Thanksgiving or Christmas. The external Nov–Jul horizon contains Thanksgiving and Christmas, so holiday coverage differs materially.
All 421570 labeled rows were retained for final refitting; submission contains 115064 keyed external predictions.

Limitations: short history, two 26-week development folds versus a 39-week forecast horizon, one final holdout, correlated stores/departments/weeks, sparse and unseen series, and differing calendar/holiday composition. The external period may behave differently. Week-level spread is descriptive, not an independent-row confidence interval. RF uses numeric threshold splits for nominal IDs; the Easter map has finite 2010–2013 coverage. No guarantee of forecasting quality or allegation of fabrication follows from these results.

Audit, protocol, selection, fold/pooled/final metrics, weekly errors, persisted row predictions and independent recomputation are saved alongside this report.
