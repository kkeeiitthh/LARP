# AeroCast — Autonomous SG 2026 Airbus Challenge #5

A reproducible historical prototype for ATL/DFW arrival throughput and severe flight-delay risk, 3/6/9/12 hours ahead.

## Start the application

From the cloned repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

This branch includes the source CSVs and evaluation reports. Run `python train.py` after cloning to generate the model artifacts before starting the app. Retrain when input data, features, or model code changes. Restart any running Streamlit process after upgrading dependencies.

```powershell
python -m pytest -p no:cacheprovider
python evaluation.py
```

Tests keep temporary files inside `outputs/pytest-temp`. Parsed input caching uses source-file and parser-code fingerprints. The dashboard reloads artifacts when the evaluation file changes.

## What was repaired

The previous parser read the CIG ceiling flag from the wrong position and discarded nearly all ceiling observations. It also used the wrong snow-depth scale/QC field and mixed precipitation accumulation periods. The parser now checks each field's documented QC position, preserves missing values, keeps one-hour rain separate, and converts units explicitly.

The previous model used only a small set of weather/calendar predictors, lacked scheduled demand, and displayed class-balanced scores as probabilities. Current candidates use planned carrier/route/duration/distance, local calendar cycles, scheduled hourly arrival/departure demand, runway wind, past weather summaries and actual timestamped weather lags. Candidates include logistic regression, random forest, gradient boosting, and ridge/Poisson regression for throughput.

The dashboard previously averaged nearby historical flights when a time was selected. It now runs the fitted models on the exact selected flight and hour. Custom scheduled-flight scenarios are supported inside the historical data range.

## Dataset findings

- Flight CSV: 148,273 rows, 123 source columns, zero duplicate rows, 2023-01-01 to 2025-07-31.
- ATL origins: 8,415; DFW origins: 7,736. ATL destinations: 5,429; DFW destinations: 4,032.
- All six NOAA files are under `datasets/`. ATL station: 72219013874; DFW: 72259003927.
- Weather extends through August 2025, beyond the available July flight outcomes. Training/evaluation uses only overlapping flight coverage.
- Official AAR, time-versioned weather forecasts, and historical schedule revisions are unavailable.
- Flight-source attached weather columns lack observation-time metadata and remain excluded.

Full column lists, missingness, station checks, per-file date coverage and duplicate counts are in `outputs/data_audit.json`. Airport headings/timezones are read from `airport_config.csv`.

## Targets and time semantics

**Severe delay:** arrival delay strictly greater than 45 minutes for completed, non-diverted ATL/DFW-origin flights. The forecast event is scheduled departure; the issue time is 3/6/9/12 hours earlier. An origin-weather model cannot isolate arrival-destination weather effects.

**Observed hourly arrivals:** completed arrivals at ATL/DFW destinations, counted by their actual arrival UTC hour. Scheduled arrival is reconstructed from origin local scheduled departure, origin timezone, and scheduled elapsed time, then checked against destination scheduled clock time. Actual arrival is scheduled arrival plus ARR_DELAY, checked against ARR_TIME. This handles overnight crossings and delays exceeding 24 hours. Actual clocks and delays enter target construction only.

Zero hours mean no arrivals in the supplied flight sample. They do not mean the airport had zero traffic. First/last partial source-date boundaries are conservatively excluded. Fractional model outputs are expected counts.

**Proxy reduction:** the dashboard compares expected sample throughput with an airport/hour/weekday historical baseline. This cannot establish official runway-capacity reductions because total demand and official AAR are absent.

## Forecast features and leakage controls

- Airport-local calendar fields; sine/cosine hour/month cycles.
- Scheduled carrier, destination, elapsed time and distance for flight predictions.
- Scheduled arrivals/departures in the target hour and adjacent hours. Cancelled records remain in schedule demand because future cancellation outcomes are unavailable at issue time.
- Latest available NOAA observation, wind/gust, temperature/dew point, visibility, ceiling, pressure, hourly rain, snow and supported thunderstorm indication.
- Signed/absolute runway wind, direction sine/cosine, gust difference and low-visibility/rain/severe-weather indicators.
- Past 3h/6h weather rolling summaries and true earlier snapshots, rather than shifting flight rows.
- Weather values expire after 3 hours; a conservative assumed publication latency of 20 minutes is applied.
- Unknown weather and its binary indicators stay missing. Imputers and missing indicators are fitted on training data only.
- No actual delay/time, delay causes, cancellation outcomes, actual elapsed time, tail number or target-derived predictor enters the feature whitelist.

The archived CRS schedule is assumed known at issue time. Historical revisions were not supplied, so strict real-world schedule availability cannot be proven. NOAA publication/revision history is also unavailable; reporting latency is an explicit assumption.

Wind is the direction the wind comes **from**. With angle = direction − challenge runway heading, crosswind = speed × sin(angle), headwind = speed × cos(angle). Positive headwind means headwind; negative means tailwind. Calm wind produces zero components even when direction is missing. Speed units are knots.

## Evaluation and calibration

All tasks/horizons share chronological boundaries:

| Partition | Event dates |
|---|---|
| Train | 2023 through 2024-10-21 |
| Model selection | 2024-10-24 through 2025-01-14 |
| Calibration/check | 2025-01-17 through 2025-03-11 |
| Final test | From 2025-03-14 |

Two-day gaps are purged; labels must mature before the next partition's earliest issue time. Gradient boosting's internal random validation is disabled.

Classification selection uses validation PR-AUC. Regression selection uses validation RMSE. Sigmoid calibration is fitted on the first calibration half and accepted only if it improves probability error on its later half. Earlier validation/calibration predictions use the raw model, preserving temporal availability. Final test data never selects model, threshold or calibration.

The alert threshold uses validation F0.5, with a 20% minimum probability review threshold. The dashboard lets reviewers inspect precision/recall and confusion matrices at other thresholds. Majority-class accuracy is shown because accuracy alone rewards missed severe cases.

Baselines use training-only airport prevalence for delay and smoothed airport/hour/weekday throughput. Regression also includes ridge and Poisson candidates. Reported probability quality includes Brier score and log loss. Throughput intervals use calibration residuals and report measured test coverage.

Read **[MODEL_REPORT.md](MODEL_REPORT.md)** for current measured results, airport/horizon comparisons, limits and correct/incorrect/borderline cases. Complete artifacts are:

- `outputs/evaluation_results.json`: validation, calibration and test metrics, candidate comparisons, feature importance.
- `outputs/predictions.csv`: event/issue/weather availability timestamps, outcomes, predictions and baselines.
- `outputs/case_studies.json`: inspectable examples.
- `outputs/*_Nh.joblib`: selected model bundles.
- Model binaries, caches, generated prediction CSVs, and local previous-run backups are excluded from Git. Training regenerates the required artifacts.

Before/after headline figures use corrected targets/timestamps and slightly different cohorts. Use the current shared split and baselines for fair model selection.

## Dashboard

Forecast, Model evaluation, and Data quality tabs provide airport/horizon/date/flight selection, exact inference, custom schedule scenarios, issue-time weather with units/age, wind components, throughput and delay risk, proxy reduction, uncertainty intervals, controller review prompts, historical charts, reliability/precision–recall plots, actual/predicted/baseline charts, confusion matrices, feature importance, airport/horizon tables and examples.

Risk labels describe probability bands: Low below 10%, Moderate from 10% to below 30%, Severe at 30% or higher. These display bands are distinct from the validation-based binary review threshold.

## Project layout

| File | Purpose |
|---|---|
| app.py | Streamlit dashboard and exact scenario inference |
| data_loading.py | Flight cleaning, schedules and target timestamps |
| noaa_parser.py | NOAA encoded fields, quality flags and units |
| features.py | Weather availability, lags/rolling features, calendar and wind |
| modeling.py | Candidate models, temporal splits, calibration, baselines |
| evaluation.py | Artifact audit, examples and human-readable report |
| train.py | Cached full training/evaluation workflow |
| airport_config.csv | Coordinates, headings, timezones and station IDs |
| tests/ | Parsing, missingness, time, wind, target and leakage regressions |
| pipeline.py | Legacy schedule-only compatibility helpers; unused by current app/training |

## Limitations and next data needed

The prototype still has moderate predictive skill. Severe delays depend on traffic network state and operational decisions absent from the sample. NOAA observations support weather persistence, not a true future-weather forecast. Full movement coverage, FAA ASPM AAR, time-versioned TAF/NWP forecasts and historical schedule snapshots are the highest-value next data sources. More complex models cannot recover missing observations.

The dashboard supports human review and does not make official safety or traffic-control decisions.

## Three-minute presentation

“We built AeroCast to give controllers a clearer view of weather-related airport disruption three to twelve hours ahead.

Our first step was an audit: 148,273 flight records and six NOAA weather files for ATL and DFW. We found no official AAR. That led to two honest targets: expected completed arrivals in the supplied sample, and probability of an arrival delay over 45 minutes.

Data quality mattered more than swapping algorithms. We corrected NOAA quality-flag decoding, separated precipitation periods, converted local schedules to UTC, and handled midnight and multi-day delays. We calculate crosswind and headwind against the challenge runway headings. Every weather input must have been available at the issue time.

We then added planned demand, flight schedule context, and past weather trends. Logistic regression, random forest, gradient boosting and regression candidates compete on earlier validation dates. A separate later block checks probability calibration, and the latest dates remain the final test.

The Forecast tab predicts the exact selected flight and airport hour. The Evaluation tab shows measured airport and horizon performance, baselines, false alarms, missed delays, probability reliability and explanatory feature importance. We show a correct case, an incorrect case and a borderline case.

The results and their limits are visible in the report. Our throughput model benefits substantially from planned demand. Delay discrimination improves, but false alarms and missed disruptions remain. High overall accuracy alone would be misleading because most flights are not severely delayed.

Our next step is to replace weather persistence with time-versioned forecasts, and the sampled throughput proxy with full movement and official AAR data. AeroCast supports earlier human review of arrival plans while keeping the uncertainty and data limitations visible.”

## Challenge #5 mapping

Both targets and all four horizons are implemented; historical weather and flight performance support timestamp-aware forecasting; challenge runway wind calculations have tests; leakage controls use explicit feature whitelists and chronological purges; simple baselines and multiple model families are evaluated; airport/horizon metrics and errors are inspectable; the dashboard provides review prompts and limitations. The missing official-capacity and genuine-weather-forecast data prevent a defensible claim of operational runway-capacity forecasting.
