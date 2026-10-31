"""Reproducible pipeline: audit, cache, compare, calibrate, evaluate, persist."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from data_loading import load_flights, delay_events, throughput_events, attach_schedule_features, arrival_schedule
from noaa_parser import load_noaa_directory
from features import prepare_weather, asof_weather
from modeling import evaluate

ROOT = Path(__file__).parent
DATA = ROOT / "datasets"
OUT = ROOT / "outputs"
VERSION = "2026-10-31-v2"


def fingerprint():
    paths = sorted(DATA.glob("*.csv")) + [ROOT/p for p in ("features.py", "noaa_parser.py", "data_loading.py", "airport_config.csv")]
    return hashlib.sha256("|".join(f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in paths).encode()).hexdigest()[:20]


def load_inputs():
    cache = OUT / "cache"
    cache.mkdir(exist_ok=True, parents=True)
    key = fingerprint()
    path = cache / f"inputs-{key}.joblib"
    if path.exists():
        return joblib.load(path)
    flights = load_flights(DATA / "Flight_Data_With_Weather_Final.csv")
    weather = prepare_weather(load_noaa_directory(DATA))
    joblib.dump((flights, weather), path, compress=3)
    return flights, weather


def audit(flights, weather):
    source = pd.read_csv(DATA / "Flight_Data_With_Weather_Final.csv", low_memory=False)
    schedule = arrival_schedule(flights)
    weather_files = []
    for path in sorted(DATA.glob("*_20*.csv")):
        raw = pd.read_csv(path, low_memory=False)
        weather_files.append({"file": path.name, "rows": len(raw), "columns": raw.columns.tolist(),
                              "date_min": raw.DATE.min(), "date_max": raw.DATE.max(),
                              "stations": raw.STATION.astype(str).unique().tolist(),
                              "duplicate_rows": int(raw.duplicated().sum())})
    return {"version": VERSION, "flight_rows": len(source), "flight_columns": source.columns.tolist(),
            "source_duplicate_rows": int(source.duplicated().sum()),
            "date_min": str(flights.FL_DATE.min().date()), "date_max": str(flights.FL_DATE.max().date()),
            "source_missing_values": source.isna().sum().to_dict(),
            "origin_counts": flights.loc[flights.ORIGIN.isin(["ATL","DFW"]), "ORIGIN"].value_counts().to_dict(),
            "destination_counts": flights.loc[flights.DEST.isin(["ATL","DFW"]), "DEST"].value_counts().to_dict(),
            "unresolved_scheduled_arrivals": int(schedule.scheduled_arrival_utc.isna().sum()),
            "official_aar_available": False, "weather_files": weather_files,
            "parsed_weather_rows": len(weather),
            "parsed_weather_missing_fraction": weather.select_dtypes("number").isna().mean().to_dict()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", type=int, default=[3,6,9,12], choices=[3,6,9,12])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    flights, weather = load_inputs()
    data_audit = audit(flights, weather)
    (OUT/"data_audit.json").write_text(json.dumps(data_audit, indent=2), encoding="utf-8")
    results = {"version": VERSION, "source_fingerprint": fingerprint(),
        "target_definition": {"throughput": "completed sampled arrivals per UTC hour; observed throughput proxy",
                              "delay": "ARR_DELAY > 45 on completed, non-diverted ATL/DFW-origin flights; relative to scheduled departure"},
        "weather_method": "Past observations with 20-minute assumed reporting latency and 3-hour staleness limit; persistence context, no true weather forecasts.",
        "schedule_assumption": "Archived CRS schedules are assumed known at issue time. No historical schedule revisions were supplied.",
        "horizons": {}}
    predictions = []
    for horizon in args.horizons:
        results["horizons"][str(horizon)] = {}
        for name, builder, target, kind in [
            ("delay", delay_events, "severe_delay", "classification"),
            ("throughput", throughput_events, "observed_hourly_arrivals", "regression")]:
            print(f"Training {name}: {horizon}h", flush=True)
            events = attach_schedule_features(builder(flights, horizon), flights)
            joined = asof_weather(events, weather, prepared=True)
            model, metrics, pred = evaluate(joined, target, kind)
            metrics["weather_unmatched_fraction"] = float(joined.observation_time_utc.isna().mean())
            results["horizons"][str(horizon)][name] = metrics
            pred["task"] = name
            pred["forecast_horizon_hours"] = horizon
            keep = ["task", "forecast_horizon_hours", "partition", "airport", "event_time_utc", "issue_time_utc",
                    "observation_time_utc", "available_time_utc", "actual", "prediction", "baseline_prediction",
                    "decision_threshold", "interval_low", "interval_high"]
            if name == "delay":
                keep += ["carrier", "destination", "planned_elapsed_minutes", "distance_miles"]
            predictions.append(pred[keep])
            joblib.dump(model, OUT/f"{name}_{horizon}h.joblib", compress=3)
            print(f"Selected {model.name}; test {metrics['test']['model']}", flush=True)
    pd.concat(predictions, ignore_index=True).to_csv(OUT/"predictions.csv", index=False)
    joblib.dump(weather, OUT/"weather_prepared.joblib", compress=3)
    (OUT/"evaluation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    from evaluation import write_report
    write_report()
    print("Complete. Artifacts saved under outputs/.", flush=True)


if __name__ == "__main__":
    main()
