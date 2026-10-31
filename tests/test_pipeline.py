from datetime import time

import pandas as pd
import pytest

from pipeline import (
    FEATURE_COLUMNS,
    _chronological_date_splits,
    _features_for_flights,
    audit_dataset,
    load_flights,
    parse_hhmm,
    prepare_cohort,
)


def test_parse_hhmm_accepts_valid_values_and_midnight_rollover() -> None:
    assert parse_hhmm(1530) == time(15, 30)
    assert parse_hhmm(0) == time(0, 0)
    assert parse_hhmm(2400) == time(0, 0)


@pytest.mark.parametrize("value", [None, "bad", 1260, 2500, 123.5])
def test_parse_hhmm_rejects_missing_or_invalid_values(value) -> None:
    assert parse_hhmm(value) is None


def _sample_frame() -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2024-01-01", periods=30, freq="D"):
        for airport, destination in (("ATL", "BOS"), ("DFW", "LAX")):
            rows.append(
                {
                    "FL_DATE": day,
                    "ORIGIN": airport,
                    "DEST": destination,
                    "OP_UNIQUE_CARRIER": "ZZ",
                    "CRS_DEP_TIME": 900,
                    "CRS_ARR_TIME": 1100,
                    "ARR_DELAY": 50 if len(rows) % 4 == 0 else 0,
                    "CANCELLED": 0,
                    "DIVERTED": 0,
                    "CRS_ELAPSED_TIME": 120,
                    "DISTANCE": 900,
                    "temp_ORIGIN": 30,
                    "wspd_ORIGIN": 12,
                }
            )
    return pd.DataFrame(rows)


def test_load_flights_reports_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        load_flights(pd.DataFrame({"ORIGIN": ["ATL"]}).to_csv(index=False).encode())


def test_features_exclude_weather_and_outcome_columns() -> None:
    flights = _sample_frame()
    parsed = load_flights(flights.to_csv(index=False).encode("utf-8"))
    cohort = prepare_cohort(parsed)
    features = _features_for_flights(cohort, horizon=3)
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert not any("weather" in column.lower() for column in features.columns)
    assert not any(
        token in column.lower()
        for column in features.columns
        for token in ("delay", "cancel", "divert", "actual", "temp", "wspd")
    )


def test_audit_counts_original_source_columns_only() -> None:
    source = _sample_frame()
    parsed = load_flights(source.to_csv(index=False).encode("utf-8"))
    audit = audit_dataset(parsed)
    assert audit["columns"] == len(source.columns)
    assert "scheduled_departure" not in audit["column_names"]
    assert audit["aar_column_present"] is False
    assert audit["wind_direction_present"] is False


def test_chronological_split_keeps_each_date_in_a_single_partition() -> None:
    flights = prepare_cohort(load_flights(_sample_frame().to_csv(index=False).encode()))
    split, boundaries = _chronological_date_splits(flights)
    grouped = pd.DataFrame(
        {"date": flights["FL_DATE"].dt.date, "partition": split}
    ).groupby("date")["partition"].nunique()
    assert grouped.max() == 1
    assert boundaries["train_end"] < boundaries["validation_end"]
    assert boundaries["validation_end"] < boundaries["test_end"]
