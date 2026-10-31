# Model and data audit report

Generated from held-out predictions; model and threshold choices use earlier periods only.

## Measured test performance

| Horizon | Delay model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Throughput RMSE | Baseline RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3h | gradient_boosting | 0.780 | 0.372 | 0.435 | 0.401 | 0.727 | 0.372 | 0.128 | 0.340 | 0.468 |
| 6h | gradient_boosting_small | 0.777 | 0.359 | 0.402 | 0.380 | 0.716 | 0.323 | 0.131 | 0.340 | 0.468 |
| 9h | random_forest | 0.778 | 0.365 | 0.425 | 0.393 | 0.713 | 0.348 | 0.131 | 0.340 | 0.468 |
| 12h | logistic_regression | 0.772 | 0.346 | 0.390 | 0.367 | 0.704 | 0.304 | 0.132 | 0.339 | 0.468 |

## Airport results

| Horizon | Airport | Delay ROC-AUC | Delay PR-AUC | Throughput RMSE | Baseline RMSE |
|---|---|---:|---:|---:|---:|
| 3h | ATL | 0.737 | 0.410 | 0.346 | 0.499 |
| 3h | DFW | 0.716 | 0.364 | 0.334 | 0.436 |
| 6h | ATL | 0.723 | 0.296 | 0.346 | 0.499 |
| 6h | DFW | 0.713 | 0.348 | 0.334 | 0.436 |
| 9h | ATL | 0.711 | 0.326 | 0.347 | 0.499 |
| 9h | DFW | 0.715 | 0.372 | 0.333 | 0.436 |
| 12h | ATL | 0.714 | 0.280 | 0.346 | 0.499 |
| 12h | DFW | 0.701 | 0.332 | 0.333 | 0.436 |

## Interpretation

Accuracy is paired with precision, recall and PR-AUC. The majority baseline is competitive on accuracy because severe delays are uncommon. The review policy trades recall for fewer false alarms. The dashboard exposes this trade-off interactively.
ROC-AUC/PR-AUC measure ranking, while Brier score measures probability error. Calibration is fitted on the first half of a later calibration block and accepted only if it improves Brier score on its second half. Earlier validation/calibration predictions use the uncalibrated model to avoid applying a future-fitted transform to earlier forecasts.
Throughput improvement includes planned demand. It does not establish a causal weather effect or weather-forecast skill. The CRS schedule snapshot is assumed available at issue time; schedule revisions were not supplied.
The 80% empirical residual intervals are measured on a separate calibration period. Test coverage is recorded in evaluation_results.json; nominal coverage is not guaranteed during drift.

## Data corrections and cleaning

- Flight source: 148,273 rows and 123 columns; 0 duplicate rows.
- Date coverage: 2023-01-01 to 2025-07-31; unresolved scheduled arrivals after clock validation: 0.
- CIG ceiling and VIS visibility now use their own measurement QC fields. AJ1 depth is in centimetres and uses the snow-depth QC position.
- Only one-hour precipitation amounts are combined. Six-/24-hour accumulation depths are excluded from the one-hour feature.
- Missing observations and weather indicators stay missing. Training-only imputers add explicit missingness indicators.
- Individual weather values expire after three hours; latest snapshots and 3h/6h lags are matched by airport/time, with an assumed 20-minute publication latency.
- Actual arrivals are reconstructed from a timezone-validated scheduled arrival plus ARR_DELAY, checked against ARR_TIME. This handles midnight crossings and delays exceeding 24 hours.
- Scheduled arrivals/departures include cancelled records in planned demand, because future cancellation outcomes cannot be used as schedule inputs.

## Limits and remaining data needs

- Official AAR is unavailable. Sampled counts cannot identify true airport-wide capacity or official capacity reductions.
- NOAA observations are persistence inputs. Time-versioned TAF/NWP forecasts are needed for a genuine weather-forecast experiment.
- Severe weather cases and the flight sample are limited. More models cannot recover unobserved traffic, network disruption and runway configuration.
- Before/after prototype figures use corrected timestamp/target construction and slightly different test cohorts; compare current candidates/baselines on the common current split for fair selection.

## Inspectable examples

- 3h correct: ATL 2025-03-15 20:25:00+00:00, p=0.235, threshold=0.200, actual=1.
- 3h incorrect: DFW 2025-03-14 03:52:00+00:00, p=0.203, threshold=0.200, actual=0.
- 3h borderline: ATL 2025-05-27 00:45:00+00:00, p=0.200, threshold=0.200, actual=0.
- 6h correct: DFW 2025-03-15 23:35:00+00:00, p=0.399, threshold=0.200, actual=1.
- 6h incorrect: DFW 2025-03-14 03:52:00+00:00, p=0.270, threshold=0.200, actual=0.
- 6h borderline: ATL 2025-06-23 19:30:00+00:00, p=0.200, threshold=0.200, actual=0.
- 9h correct: DFW 2025-03-15 23:35:00+00:00, p=0.325, threshold=0.200, actual=1.
- 9h incorrect: DFW 2025-03-15 01:55:00+00:00, p=0.329, threshold=0.200, actual=0.
- 9h borderline: ATL 2025-06-13 03:00:00+00:00, p=0.200, threshold=0.200, actual=1.
- 12h correct: DFW 2025-03-15 23:35:00+00:00, p=0.275, threshold=0.200, actual=1.
- 12h incorrect: DFW 2025-03-15 01:55:00+00:00, p=0.357, threshold=0.200, actual=0.
- 12h borderline: DFW 2025-06-20 23:38:00+00:00, p=0.200, threshold=0.200, actual=0.
