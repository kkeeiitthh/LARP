# Stage 1 validation

Window: 2023-01-01 to 2025-08-01 (exclusive).

- Flights: 25,299; feature rows: 76,836.
- TAF issuances: 22,164; BTS flights: 3,459,187.
- Missing METAR/TAF/BTS joins: 13 / 1,149 / 6.
- Inconsistent input flight timestamps: 2; retained with unavailable features.
- Invalid BTS timestamps excluded from counts: 197.
- Invalid TAF issuance validity intervals left unavailable: 358.
- Row counts, identifiers, unique keys, exact cutoffs and TAF validity passed.

See the generated manifest for hashes, source URLs, predictor columns and detailed coverage.
Model training and accuracy evaluation are outside Stage 1.

## Handoff checks — 31 October 2026

- All 22 standard-library unit tests passed.
- The notebook attachment cell ran successfully against the real cleaned data,
  retaining all 16,151 departure rows at each requested horizon.
- Live pilot: 44 flights, 135 endpoint/horizon rows. Its offline replay matched
  the SHA-256 hashes of all four compressed output files.
- Full backfill: all 166 source files validated, including 33 BTS archives and
  66 monthly station TAF downloads with their metadata.
- Full offline replay matched all four compressed output hashes exactly.
- Independent comparison of exported rows against the original cleaned table
  confirmed every original flight column and label was preserved.
- Inconsistent original flight IDs: 10059 and 83353 (six horizon rows retained).

Outputs are under `data/stage1/2023-01-01_2025-08-01/`. Raw caches and generated
tables are ignored by Git; the manifest records source and output hashes.
The full feature table is approximately 10.7 MB compressed.

Known limits: BTS counts are reporting-carrier traffic proxies, not AAR labels;
final schedules are retrospective; publication latency is not reconstructed.
Malformed TAF validity metadata remains unavailable. Read `agent.md` before modelling.
