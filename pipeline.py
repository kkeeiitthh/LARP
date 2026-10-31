"""Legacy schedule-only compatibility helpers. Active training uses train.py and modeling.py."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

AIRPORTS = ("ATL", "DFW")
FORECAST_HORIZONS = (3, 6, 9, 12)
RUNWAY_HEADINGS = {"ATL": 80.0, "DFW": 180.0}
WEATHER_COLUMNS = (
    "temp",
    "prcp",
    "wspd",
    "wpgt",
    "pres",
    "snow",
    "coco",
)
REQUIRED_COLUMNS = (
    "FL_DATE",
    "ORIGIN",
    "DEST",
    "OP_UNIQUE_CARRIER",
    "CRS_DEP_TIME",
    "ARR_DELAY",
    "CANCELLED",
    "DIVERTED",
)
CATEGORICAL_FEATURES = ("airport", "destination", "carrier", "season")
NUMERIC_FEATURES = (
    "scheduled_hour",
    "scheduled_weekday",
    "scheduled_month",
    "issue_hour",
    "issue_weekday",
    "issue_month",
    "planned_elapsed_minutes",
    "distance_miles",
)
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _season_name(month: int | float) -> str:
    if pd.isna(month):
        return "unknown"
    month_number = int(month)
    if month_number in (12, 1, 2):
        return "winter"
    if month_number in (3, 4, 5):
        return "spring"
    if month_number in (6, 7, 8):
        return "summer"
    return "fall"


@dataclass
class Evaluation:
    """Models, chronological results, and feature data for one horizon."""

    horizon: int
    model: Pipeline
    baseline: "AirportPriorBaseline"
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    feature_names: list[str]
    coefficients: np.ndarray
    split_dates: dict[str, str]


class AirportPriorBaseline:
    """Predict each airport's train-period severe-delay prevalence."""

    def fit(self, airports: pd.Series, labels: np.ndarray) -> "AirportPriorBaseline":
        rows = pd.DataFrame(
            {"airport": airports.astype(str).to_numpy(), "label": labels}
        )
        self.global_prior = float(np.mean(labels))
        self.airport_priors = rows.groupby("airport")["label"].mean().to_dict()
        return self

    def predict_proba(self, airports: pd.Series) -> np.ndarray:
        positive = airports.astype(str).map(self.airport_priors).fillna(
            self.global_prior
        ).to_numpy(dtype=float)
        return np.column_stack((1.0 - positive, positive))


def _read_csv(source: str | Path | BinaryIO | bytes) -> pd.DataFrame:
    if isinstance(source, bytes):
        from io import BytesIO

        return pd.read_csv(BytesIO(source), low_memory=False)
    if hasattr(source, "read"):
        return pd.read_csv(source, low_memory=False)
    return pd.read_csv(source, low_memory=False)


def parse_hhmm(value: Any) -> time | None:
    """Parse airline HHMM values, including 2400, without guessing invalid times."""
    if pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not numeric.is_integer():
        return None
    hhmm = int(numeric)
    if hhmm == 2400:
        return time(0, 0)
    hour, minute = divmod(hhmm, 100)
    if hour > 23 or minute > 59 or hhmm < 0:
        return None
    return time(hour, minute)


def _scheduled_datetime(flight_date: Any, hhmm: Any) -> pd.Timestamp:
    parsed_time = parse_hhmm(hhmm)
    parsed_date = pd.to_datetime(flight_date, errors="coerce")
    if pd.isna(parsed_date) or parsed_time is None:
        return pd.NaT
    timestamp = pd.Timestamp.combine(parsed_date.date(), parsed_time)
    try:
        if float(hhmm) == 2400:
            timestamp += pd.Timedelta(days=1)
    except (TypeError, ValueError):
        return pd.NaT
    return timestamp


def load_flights(source: str | Path | BinaryIO | bytes) -> pd.DataFrame:
    """Load and minimally validate the supplied flight-level CSV."""
    flights = _read_csv(source)
    missing = [column for column in REQUIRED_COLUMNS if column not in flights.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )

    flights["FL_DATE"] = pd.to_datetime(flights["FL_DATE"], errors="coerce")
    flights["ORIGIN"] = flights["ORIGIN"].astype("string").str.strip().str.upper()
    flights["DEST"] = flights["DEST"].astype("string").str.strip().str.upper()
    flights["OP_UNIQUE_CARRIER"] = (
        flights["OP_UNIQUE_CARRIER"].astype("string").str.strip()
    )
    flights["ARR_DELAY"] = pd.to_numeric(flights["ARR_DELAY"], errors="coerce")
    flights["CANCELLED"] = pd.to_numeric(flights["CANCELLED"], errors="coerce")
    flights["DIVERTED"] = pd.to_numeric(flights["DIVERTED"], errors="coerce")
    flights["CRS_DEP_TIME"] = pd.to_numeric(
        flights["CRS_DEP_TIME"], errors="coerce"
    )
    flights["scheduled_departure"] = [
        _scheduled_datetime(day, hhmm)
        for day, hhmm in zip(flights["FL_DATE"], flights["CRS_DEP_TIME"])
    ]
    return flights


def audit_dataset(flights: pd.DataFrame) -> dict[str, Any]:
    """Summarize schema, inferred non-empty value types, missingness, and duplicates."""
    source_columns = [
        column for column in flights.columns if column != "scheduled_departure"
    ]
    inferred_types: dict[str, str] = {}
    for column in source_columns:
        values = flights[column].dropna()
        if values.empty:
            inferred_types[column] = "all_missing"
        elif pd.api.types.is_numeric_dtype(values):
            inferred_types[column] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(values):
            inferred_types[column] = "datetime"
        else:
            inferred_types[column] = "string_or_mixed"

    missing = flights[source_columns].isna().sum()
    return {
        "rows": int(len(flights)),
        "columns": len(source_columns),
        "column_names": source_columns,
        "inferred_types": inferred_types,
        "missing_values": {
            column: int(count) for column, count in missing.items() if count
        },
        "duplicate_rows": int(flights[source_columns].duplicated().sum()),
        "date_min": (
            flights["FL_DATE"].min().date().isoformat()
            if flights["FL_DATE"].notna().any()
            else None
        ),
        "date_max": (
            flights["FL_DATE"].max().date().isoformat()
            if flights["FL_DATE"].notna().any()
            else None
        ),
        "distinct_flight_dates": int(flights["FL_DATE"].dt.date.nunique()),
        "origin_airports": sorted(flights["ORIGIN"].dropna().unique().tolist()),
        "destination_airports": sorted(flights["DEST"].dropna().unique().tolist()),
        "arr_delay_available": int(flights["ARR_DELAY"].notna().sum()),
        "arr_delay_over_45": int((flights["ARR_DELAY"] > 45).sum()),
        "aar_column_present": any(
            column.lower() in {"aar", "arrival_acceptance_rate", "aar_per_hour"}
            for column in flights.columns
        ),
        "weather_fields": [
            column
            for column in flights.columns
            if any(column.lower().startswith(f"{prefix}_") for prefix in WEATHER_COLUMNS)
        ],
        "weather_observation_timestamp_present": any(
            any(token in column.lower() for token in ("weather_time", "weather_timestamp", "valid_time", "observation_time"))
            for column in source_columns
        ),
        "wind_direction_present": any(
            (
                "wind" in column.lower() and "dir" in column.lower()
            )
            or "wdir" in column.lower()
            for column in source_columns
        ),
    }


def prepare_cohort(flights: pd.DataFrame) -> pd.DataFrame:
    """Keep labeled, completed ATL/DFW-origin flights for the delay-risk target."""
    cohort = flights.loc[
        flights["ORIGIN"].isin(AIRPORTS)
        & flights["ARR_DELAY"].notna()
        & (flights["CANCELLED"].fillna(0) == 0)
        & (flights["DIVERTED"].fillna(0) == 0)
        & flights["scheduled_departure"].notna()
    ].copy()
    cohort["severe_delay"] = (cohort["ARR_DELAY"] > 45).astype("int8")
    cohort["airport"] = cohort["ORIGIN"].astype(str)
    cohort["destination"] = cohort["DEST"].fillna("UNKNOWN").astype(str)
    cohort["carrier"] = (
        cohort["OP_UNIQUE_CARRIER"].fillna("UNKNOWN").astype(str)
    )
    return cohort.sort_values("scheduled_departure").reset_index(drop=True)


def _features_for_flights(
    flights: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """Build only flight-plan/calendar features known at the issue time."""
    scheduled = pd.to_datetime(flights["scheduled_departure"])
    issue = scheduled - pd.to_timedelta(int(horizon), unit="h")
    planned_elapsed = pd.to_numeric(
        flights.get("CRS_ELAPSED_TIME", pd.Series(np.nan, index=flights.index)),
        errors="coerce",
    )
    distance = pd.to_numeric(
        flights.get("DISTANCE", pd.Series(np.nan, index=flights.index)),
        errors="coerce",
    )
    features = pd.DataFrame(index=flights.index)
    features["airport"] = flights["ORIGIN"].astype(str)
    features["destination"] = flights["DEST"].fillna("UNKNOWN").astype(str)
    features["carrier"] = flights["OP_UNIQUE_CARRIER"].fillna("UNKNOWN").astype(str)
    features["season"] = scheduled.dt.month.map(_season_name)
    features["scheduled_hour"] = scheduled.dt.hour
    features["scheduled_weekday"] = scheduled.dt.day_of_week
    features["scheduled_month"] = scheduled.dt.month
    features["issue_hour"] = issue.dt.hour
    features["issue_weekday"] = issue.dt.day_of_week
    features["issue_month"] = issue.dt.month
    features["planned_elapsed_minutes"] = planned_elapsed
    features["distance_miles"] = distance
    return features.loc[:, FEATURE_COLUMNS]


def _chronological_date_splits(cohort: pd.DataFrame) -> tuple[pd.Series, dict[str, str]]:
    dates = pd.Series(cohort["FL_DATE"].dt.normalize().dropna().unique()).sort_values()
    if len(dates) < 10:
        raise ValueError(
            "At least 10 distinct flight dates are required for chronological "
            "train/validation/test evaluation."
        )
    train_end_index = max(0, int(len(dates) * 0.70) - 1)
    validation_end_index = max(train_end_index + 1, int(len(dates) * 0.85) - 1)
    validation_end_index = min(validation_end_index, len(dates) - 2)
    train_end = pd.Timestamp(dates.iloc[train_end_index])
    validation_end = pd.Timestamp(dates.iloc[validation_end_index])
    masks = pd.Series(index=cohort.index, dtype="object")
    normalized = cohort["FL_DATE"].dt.normalize()
    masks.loc[normalized <= train_end] = "train"
    masks.loc[(normalized > train_end) & (normalized <= validation_end)] = "validation"
    masks.loc[normalized > validation_end] = "test"
    dates_report = {
        "train_end": train_end.date().isoformat(),
        "validation_end": validation_end.date().isoformat(),
        "test_start": pd.Timestamp(
            dates.iloc[validation_end_index + 1]
        ).date().isoformat(),
        "test_end": normalized.max().date().isoformat(),
    }
    return masks, dates_report


def _classification_metrics(
    actual: np.ndarray, probability: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    predicted = probability >= threshold
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    return {
        "rows": int(len(actual)),
        "positive_rate": float(np.mean(actual)) if len(actual) else None,
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(actual, probability))
            if len(np.unique(actual)) == 2
            else None
        ),
        "confusion_matrix": matrix.tolist(),
    }


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                list(CATEGORICAL_FEATURES),
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(NUMERIC_FEATURES),
            ),
        ],
        remainder="drop",
    )


def evaluate_horizon(cohort: pd.DataFrame, horizon: int) -> Evaluation:
    """Fit schedule-only logistic regression and an airport-prior baseline."""
    if horizon not in FORECAST_HORIZONS:
        raise ValueError(
            f"Unsupported horizon {horizon}; choose one of {FORECAST_HORIZONS}."
        )
    ordered = cohort.sort_values("scheduled_departure").reset_index(drop=True)
    split, split_dates = _chronological_date_splits(ordered)
    features = _features_for_flights(ordered, horizon)
    labels = ordered["severe_delay"].to_numpy()
    train_mask = split.eq("train").to_numpy()
    validation_mask = split.eq("validation").to_numpy()
    test_mask = split.eq("test").to_numpy()
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("Chronological split produced an empty partition.")

    model = Pipeline(
        steps=[
            ("preprocessor", _preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1500,
                    random_state=17,
                ),
            ),
        ]
    )
    model.fit(features.loc[train_mask], labels[train_mask])

    baseline = AirportPriorBaseline().fit(
        features.loc[train_mask, "airport"], labels[train_mask]
    )

    records: list[dict[str, Any]] = []
    for partition, mask in (
        ("validation", validation_mask),
        ("test", test_mask),
    ):
        part_features = features.loc[mask]
        part_labels = labels[mask]
        part = ordered.loc[mask]
        model_probability = model.predict_proba(part_features)[:, 1]
        baseline_probability = baseline.predict_proba(part_features["airport"])[:, 1]
        for row_index, (_, flight) in enumerate(part.iterrows()):
            records.append(
                {
                    "partition": partition,
                    "airport": str(flight["ORIGIN"]),
                    "destination": str(flight["DEST"]),
                    "carrier": str(flight["OP_UNIQUE_CARRIER"]),
                    "scheduled_departure": flight["scheduled_departure"],
                    "actual_delay_minutes": float(flight["ARR_DELAY"]),
                    "actual_severe": int(part_labels[row_index]),
                    "probability": float(model_probability[row_index]),
                    "predicted_severe": int(model_probability[row_index] >= 0.5),
                    "baseline_probability": float(baseline_probability[row_index]),
                }
            )
    predictions = pd.DataFrame(records)
    validation = predictions.loc[predictions["partition"] == "validation"]
    test = predictions.loc[predictions["partition"] == "test"]

    def report_partition(frame: pd.DataFrame) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for model_name, probability_column in (
            ("logistic_regression", "probability"),
            ("airport_prior_baseline", "baseline_probability"),
        ):
            result[model_name] = _classification_metrics(
                frame["actual_severe"].to_numpy(),
                frame[probability_column].to_numpy(),
            )
            result[model_name]["by_airport"] = {}
            for airport in AIRPORTS:
                airport_rows = frame.loc[frame["airport"] == airport]
                if airport_rows.empty:
                    result[model_name]["by_airport"][airport] = None
                else:
                    result[model_name]["by_airport"][airport] = (
                        _classification_metrics(
                            airport_rows["actual_severe"].to_numpy(),
                            airport_rows[probability_column].to_numpy(),
                        )
                    )
        return result

    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out().tolist()
    coefficients = model.named_steps["classifier"].coef_[0].copy()
    return Evaluation(
        horizon=horizon,
        model=model,
        baseline=baseline,
        metrics={
            "validation": report_partition(validation),
            "test": report_partition(test),
        },
        predictions=predictions,
        feature_names=feature_names,
        coefficients=coefficients,
        split_dates=split_dates,
    )


def evaluate_all_horizons(flights: pd.DataFrame) -> dict[int, Evaluation]:
    """Evaluate every requested horizon on shared chronological partitions."""
    cohort = prepare_cohort(flights)
    if cohort.empty:
        raise ValueError(
            "No completed, labeled flights originating at ATL or DFW were found."
        )
    return {
        horizon: evaluate_horizon(cohort, horizon)
        for horizon in FORECAST_HORIZONS
    }


def build_prediction_features(
    airport: str,
    destination: str,
    carrier: str,
    scheduled_departure: datetime,
    horizon: int,
    planned_elapsed_minutes: float | None = None,
    distance_miles: float | None = None,
) -> pd.DataFrame:
    """Build one inference row using planned metadata only."""
    if airport not in AIRPORTS:
        raise ValueError(f"Airport must be one of {AIRPORTS}.")
    if horizon not in FORECAST_HORIZONS:
        raise ValueError(f"Horizon must be one of {FORECAST_HORIZONS}.")
    scheduled = pd.Timestamp(scheduled_departure)
    issue = scheduled - pd.to_timedelta(int(horizon), unit="h")
    return pd.DataFrame(
        [
            {
                "airport": airport,
                "destination": destination,
                "carrier": carrier,
                "season": _season_name(scheduled.month),
                "scheduled_hour": scheduled.hour,
                "scheduled_weekday": scheduled.dayofweek,
                "scheduled_month": scheduled.month,
                "issue_hour": issue.hour,
                "issue_weekday": issue.dayofweek,
                "issue_month": issue.month,
                "planned_elapsed_minutes": planned_elapsed_minutes,
                "distance_miles": distance_miles,
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def predict_risk(
    evaluation: Evaluation,
    airport: str,
    destination: str,
    carrier: str,
    scheduled_departure: datetime,
    planned_elapsed_minutes: float | None = None,
    distance_miles: float | None = None,
) -> float:
    """Return schedule-only severe-arrival-delay probability for a flight plan."""
    row = build_prediction_features(
        airport=airport,
        destination=destination,
        carrier=carrier,
        scheduled_departure=scheduled_departure,
        horizon=evaluation.horizon,
        planned_elapsed_minutes=planned_elapsed_minutes,
        distance_miles=distance_miles,
    )
    return float(evaluation.model.predict_proba(row)[0, 1])


def historical_weather_summary(
    flights: pd.DataFrame, airport: str, selected_date: date
) -> dict[str, float | None]:
    """Describe attached same-date weather values; these are never model inputs."""
    result: dict[str, float | None] = {}
    day_rows = flights.loc[
        (flights["ORIGIN"] == airport)
        & (flights["FL_DATE"].dt.date == selected_date)
    ]
    for prefix in WEATHER_COLUMNS:
        column = f"{prefix}_ORIGIN"
        if column not in flights.columns or day_rows.empty:
            result[prefix] = None
            continue
        values = pd.to_numeric(day_rows[column], errors="coerce").dropna()
        result[prefix] = float(values.median()) if not values.empty else None
    return result


def summarize_weather_availability(flights: pd.DataFrame) -> dict[str, Any]:
    """Inspect fields without asserting that their values were available at issue time."""
    weather_columns = [
        column
        for column in flights.columns
        if any(column.startswith(f"{prefix}_") for prefix in WEATHER_COLUMNS)
    ]
    return {
        "fields": weather_columns,
        "timestamp_fields": [
            column
            for column in flights.columns
            if any(token in column.lower() for token in ("weather_time", "weather_timestamp", "valid_time"))
        ],
        "wind_direction_fields": [
            column
            for column in flights.columns
            if "wind" in column.lower() and "dir" in column.lower()
        ],
        "visibility_fields": [
            column for column in flights.columns if "visib" in column.lower()
        ],
    }
