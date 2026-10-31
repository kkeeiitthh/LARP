# Stage 1 — TAF and BTS enrichment

This file is the shared handoff for Codex and Claude. Read it before changing the
pipeline or consuming its features. Stage 1 owns data preparation; the next stage
owns model training, chronological evaluation and the dashboard.

## Run

Use Python 3.12 (tested with pandas 2.2.3 and NumPy 2.3.5). From the repo root:

```sh
python -m pip install -r requirements.txt
python -m unittest -v test_stage1_pipeline
python stage1_pipeline.py --start 2023-01-01 --end 2023-01-03
python stage1_pipeline.py
python stage1_pipeline.py --offline
```

`--end` is exclusive. Dates filter the original flight's local scheduled departure
date, preserving both in-scope endpoints. The default covers January 2023–July
2025. The full download includes adjacent source months for lookbacks and UTC
rollovers and is approximately 1 GB compressed. A first run needs internet access;
offline mode requires a complete validated cache for the selected range.

The pipeline starts from the tracked `data/clean/flights_clean.csv` and
`data/weather/hourly_atl_dfw.csv`. The original flight CSV is absent; do not run
`data_cleaning.ipynb` expecting it to regenerate these inputs. Do not replace the
25,299 original IDs/labels with BTS rows. BTS supplies airport context only.

## Outputs and model contract

Results are in `data/stage1/START_END/`, with large caches in `data/external/`.
Both are ignored by Git; regenerate them rather than committing them.

- `flight_features.csv.gz`: one row per `(flight_id, airport, event_type,
  lead_hours)`. Departure and arrival endpoints are separate; ATL–DFW flights have
  two endpoints. Full input: 25,299 distinct IDs and 76,836 rows across 3/6/12 hours.
- `airport_hour_context.csv.gz`: BTS scheduled airport-hour counts.
- `taf_segments.csv.gz`: normalized forecasts and source/product identifiers.
- `coverage.csv.gz`: missing joins by airport, endpoint and horizon.
- `manifest.json`: successful-run source URLs, SHA-256 hashes, coverage, checks
  and **the explicit `predictors` list**. `last_failure.json` records failed runs
  without replacing a previously successful manifest.
- `STAGE1_VALIDATION.md`: small, tracked summary of the most recent successful run.

Load the feature CSV with `pd.read_csv(path, low_memory=False)` because some
forecast flags are nullable. Build model inputs using only `manifest['predictors']`. For departure risk, filter
`event_type == 'departure'`, use nullable `severe_delay` (>45 minutes), and exclude
unknown labels and `timestamp_valid == False`. Keep cancellations/diversions available for audit. Arrival rows
carry arrival-delay labels, not AAR labels. Do not use identifiers, tail numbers,
actual times, taxi/air times, delay-cause columns, cancellations or diversions as
predictors. Do not turn missing outcomes into negative labels.

Split flight groups chronologically in UTC: train on 2023, validate on 2024, test
on January–July 2025. Keep every endpoint/horizon of a flight in one split; purge
flights whose endpoints span a split boundary. Fit preprocessing only on
training data. Report results by airport and horizon. Changing the schema requires
updating this contract and tests in the same change.

## Time and units

- `event_utc` is the scheduled endpoint time; `prediction_utc = event_utc - lead`.
- METAR uses a backward join on the last contributing observation, at or before
  the **exact** prediction cutoff, with a two-hour maximum age. The notebook imports
  the shared attachment function; do not restore its old `lead_hours - 1` check.
- TAF selects the latest issuance before/equal to prediction, then the prevailing
  segment covering the future event. Expired/not-yet-valid TAFs remain missing.
  TEMPO/PROB30/PROB40/BECMG qualifiers stay separate from prevailing conditions.
- BTS recent operations cover `(prediction - 3h, prediction]`, using wheels-off/on
  timestamps; future operations cannot affect a row. Severe departure-delay rate
  uses completed departures in that same window; an empty denominator is unknown.
- All joins use UTC. Ambiguous/nonexistent local times and inconsistent schedules
  stay unavailable, with rows retained. Airport time zones come from OpenFlights
  IANA identifiers, not fixed UTC offsets; ATL/DFW zones are explicit.
- Wind: knots, direction FROM true north. Runway headings: ATL 80°, DFW 180° as
  assumed by the challenge. Crosswind is absolute; tailwind is nonnegative.
  Forecast gust components use sustained direction because separate gust direction
  is unavailable. Variable wind keeps a speed but no invented direction.
- Visibility: statute miles; ceiling: feet; temperature: °C; pressure: hPa;
  precipitation: mm. TAF P6SM is encoded as 6.01, not a precise visibility reading.
- A missing ceiling may mean no broken/overcast ceiling. Preserve missingness and
  the TAF no-ceiling indicator; do not replace all missing weather with zero.

## Sources and limits

- TAF archive/API: https://mesonet.agron.iastate.edu/request/taf.php
- TAF issuance/validity metadata:
  https://mesonet.agron.iastate.edu/api/1/nws/taf_overview.json
- BTS monthly reporting-carrier archives: https://transtats.bts.gov/PREZIP/
- Airport metadata: https://openflights.org/data.php (OpenFlights/ODbL attribution;
  the cached metadata retains its source URL).

BTS counts represent reporting-carrier traffic, **not all airport traffic and not
Arrival Acceptance Rate**. Real hourly AAR labels remain missing. Final BTS
schedules are retrospective demand proxies, not archived point-in-time schedules.
Observation/operation timestamps proxy availability: publication/ingestion latency
and later archive corrections are not fully reconstructed. A live service would
need timestamped operational feeds. Stage 1 makes no model-accuracy claims.

The live pilot found malformed month-boundary TAF validity metadata (for example,
a December 31, 2022 issuance with December 2 expiry). These issuances are flagged
`validity_invalid` in the normalized source table and remain unavailable to the
model. The manifest/report counts them; do not silently repair source dates.

## Coordination and status

Branch: `codex/stage-1-data-pipeline`. Keep unrelated user changes untouched.
Keep source, feature logic and instructions together in reviewable commits. Do not
merge this branch or change another teammate's model/notebook work implicitly.

Validated on 31 October 2026:

- All 22 standard-library tests passed, including exact-cutoff regression,
  forecast amendments/expiry, DST/midnight, traffic windows and download recovery.
- The live pilot (January 1–2, 2023) produced 135 rows from 44 flights. Its offline
  replay reproduced all four compressed output hashes.
- The full run validated 166 public source files, including 33 BTS months and
  66 station-month TAF CSVs with their issuance metadata. It used 22,164 TAF
  issuances and 3,459,187 BTS records to produce 76,836 rows for 25,299 flight IDs.
- Full offline replay reproduced all four output hashes. An independent CSV
  comparison confirmed every original flight column and label was preserved.
- Two original flights (IDs 10059 and 83353) have inconsistent schedules; their
  six horizon rows are retained with unavailable features. There are 197 BTS
  timestamp anomalies and 358 invalid TAF validity intervals; these are excluded
  from counts/forecast joins as applicable.
- Missing feature joins: 13 METAR, 1,149 TAF and 6 BTS rows. See the manifest and
  coverage table for airport/endpoint/horizon breakdowns.

The generated data and caches are local and ignored by Git. Run the commands
above after checking out this branch to reproduce them. Model training has not
been performed. Record new validation outcomes when changing the pipeline.
