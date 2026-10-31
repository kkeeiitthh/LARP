"""Model comparison, separate temporal calibration, and a held-out final test."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                            f1_score, fbeta_score, log_loss, mean_absolute_error, mean_squared_error, precision_score,
                            recall_score, roc_auc_score, r2_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from features import WEATHER_BASE

FORBIDDEN = {"ARR_DELAY", "DEP_DELAY", "ARR_TIME", "DEP_TIME", "CARRIER_DELAY", "WEATHER_DELAY",
             "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY", "actual", "severe_delay",
             "observed_hourly_arrivals", "label_available_utc"}
WEATHER = WEATHER_BASE + ["crosswind_kt", "headwind_kt", "absolute_crosswind_kt",
    "gust_difference_kt", "wind_direction_sin", "wind_direction_cos", "precipitation_indicator",
    "low_visibility_indicator", "severe_weather_indicator", "dewpoint_spread_c", "weather_age_minutes"]
WEATHER += [f"{c}_past{h}h_mean" for c in ["wind_speed_kt", "visibility_m", "precipitation_mm",
    "sea_level_pressure_hpa", "absolute_crosswind_kt"] for h in (3, 6)]
WEATHER += [f"{c}_lag{h}h" for c in ("wind_speed_kt", "visibility_m") for h in (3, 6)]
SCHEDULE = [f"{c}_offset{o}" for c in ("scheduled_arrivals", "scheduled_departures") for o in (-1, 0, 1)]
NUMERIC = ["hour", "weekday", "month", "hour_sin", "hour_cos", "month_sin", "month_cos", "horizon"] + SCHEDULE + WEATHER
SPLIT_DATES = {"train_end": "2024-10-22", "validation_start": "2024-10-24",
               "validation_end": "2025-01-15", "calibration_start": "2025-01-17",
               "calibration_end": "2025-03-12", "test_start": "2025-03-14"}


def chronological_split(df, time_col="event_time_utc"):
    t = df[time_col]
    # Fixed shared boundaries for both tasks and all horizons; two-day gaps are purged.
    if t.min().year <= 2023 and t.max().year >= 2025:
        s = pd.Series("purged", index=df.index)
        def utc(value): return pd.Timestamp(value, tz="UTC")
        s.loc[t < utc(SPLIT_DATES["train_end"])] = "train"
        s.loc[t.between(utc(SPLIT_DATES["validation_start"]), utc(SPLIT_DATES["validation_end"]), inclusive="left")] = "validation"
        s.loc[t.between(utc(SPLIT_DATES["calibration_start"]), utc(SPLIT_DATES["calibration_end"]), inclusive="left")] = "calibration"
        s.loc[t >= utc(SPLIT_DATES["test_start"])] = "test"
        return s
    # Small synthetic datasets for tests, with whole-day boundaries.
    days = t.dt.normalize().sort_values().unique()
    a, b = days[max(0, int(len(days)*.7)-1)], days[max(1, int(len(days)*.85)-1)]
    return pd.Series(np.select([t.dt.normalize() <= a, t.dt.normalize() <= b], ["train", "validation"], default="test"), index=df.index)


def classification_metrics(y, p, threshold=.5):
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-7, 1-1e-7)
    pred = p >= threshold
    return {"rows": int(len(y)), "positive_rate": float(np.mean(y)), "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)), "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None,
        "brier_score": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def regression_metrics(y, p):
    return {"rows": int(len(y)), "rmse": float(mean_squared_error(y, p)**.5),
            "mae": float(mean_absolute_error(y, p)), "r2": float(r2_score(y, p)),
            "mean_actual": float(np.mean(y)), "mean_predicted": float(np.mean(p))}


class HistoricalBaseline:
    def __init__(self, kind):
        self.kind = kind
    def fit(self, x, y):
        d = x[["airport", "hour", "weekday"]].copy()
        d["y"] = np.asarray(y)
        self.mean = float(np.mean(y))
        self.airport = d.groupby("airport").y.mean().to_dict()
        stat = d.groupby(["airport", "hour", "weekday"]).y.agg(["mean", "count"])
        self.groups = ((stat["mean"]*stat["count"] + 20*self.mean)/(stat["count"]+20)).to_dict()
        return self
    def predict(self, x):
        if self.kind == "classification":
            return x.airport.map(self.airport).fillna(self.mean).to_numpy()
        return np.array([self.groups.get(k, self.mean) for k in x[["airport", "hour", "weekday"]].itertuples(index=False, name=None)])


@dataclass
class ModelBundle:
    model: object
    columns: list
    kind: str
    name: str
    threshold: float = .5
    calibrator: object = None
    interval: tuple = (0., 0.)
    review_threshold: float = .2

    def raw_predict(self, x):
        selected = x.loc[:, self.columns]
        if self.kind == "classification":
            return self.model.predict_proba(selected)[:, 1]
        return np.maximum(0, self.model.predict(selected))

    def predict(self, x):
        p = self.raw_predict(x)
        if self.kind == "classification" and self.calibrator is not None:
            logit = np.log(np.clip(p, 1e-6, 1-1e-6)/(1-np.clip(p, 1e-6, 1-1e-6)))
            p = self.calibrator.predict_proba(logit.reshape(-1, 1))[:, 1]
        return p


def _pipeline(estimator, numeric, categorical):
    prep = ColumnTransformer([
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                          ("scale", StandardScaler())]), numeric)])
    return Pipeline([("preprocessor", prep), ("model", estimator)])


def evaluate(frame, target, kind):
    data = frame.dropna(subset=[target, "event_time_utc", "issue_time_utc"]).copy()
    split = chronological_split(data)
    # Labels must have matured before the next partition's earliest issue.
    for part, following in (("train", "validation"), ("validation", "calibration"), ("calibration", "test")):
        next_issue = data.loc[split.eq(following), "issue_time_utc"].min()
        if "label_available_utc" in data:
            split.loc[split.eq(part) & (data.label_available_utc.isna() | (data.label_available_utc >= next_issue))] = "purged"
    numeric = NUMERIC + (["planned_elapsed_minutes", "distance_miles"] if kind == "classification" else [])
    categorical = ["airport"] + (["carrier", "destination"] if kind == "classification" else [])
    columns = categorical + numeric
    if FORBIDDEN.intersection(columns):
        raise AssertionError("Outcome columns configured as predictors")
    x, y = data[columns], data[target]
    train, val, cal, test = [split.eq(p) for p in ("train", "validation", "calibration", "test")]
    if min(map(sum, (train, val, cal, test))) == 0:
        raise ValueError("Empty temporal partition; inspect dataset date coverage")
    if kind == "classification":
        estimators = {
            "logistic_regression": LogisticRegression(C=.15, max_iter=1200),
            "gradient_boosting_small": HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=7, learning_rate=.05, l2_regularization=15, min_samples_leaf=45, early_stopping=False, random_state=17),
            "gradient_boosting": HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=15, learning_rate=.04, l2_regularization=20, min_samples_leaf=60, early_stopping=False, random_state=17),
            "random_forest": RandomForestClassifier(n_estimators=140, max_depth=10, min_samples_leaf=25, max_features=.7, n_jobs=2, random_state=17)}
    else:
        estimators = {
            "ridge_regression": Ridge(alpha=100),
            "poisson_boosting": HistGradientBoostingRegressor(loss="poisson", max_iter=140, max_leaf_nodes=7, learning_rate=.05, l2_regularization=20, min_samples_leaf=60, early_stopping=False, random_state=17),
            "gradient_boosting": HistGradientBoostingRegressor(max_iter=140, max_leaf_nodes=7, learning_rate=.05, l2_regularization=30, min_samples_leaf=80, early_stopping=False, random_state=17)}
    candidates, scores = {}, {}
    metric = classification_metrics if kind == "classification" else regression_metrics
    baseline = HistoricalBaseline(kind).fit(x.loc[train], y.loc[train])
    baseline_val = baseline.predict(x.loc[val])
    scores["historical_baseline"] = metric(y.loc[val], baseline_val)
    for name, estimator in estimators.items():
        pipe = _pipeline(estimator, numeric, categorical)
        pipe.fit(x.loc[train], y.loc[train])
        bundle = ModelBundle(pipe, columns, kind, name)
        scores[name] = metric(y.loc[val], bundle.predict(x.loc[val]))
        candidates[name] = bundle
    if kind == "classification":
        choice = max(candidates, key=lambda name: scores[name]["pr_auc"])
    else:
        choice = min(candidates, key=lambda name: scores[name]["rmse"])
    bundle = candidates[choice]
    if kind == "classification":
        cal_order = data.loc[cal].sort_values("event_time_utc").index
        midpoint = len(cal_order)//2
        cal_fit, cal_check = cal_order[:midpoint], cal_order[midpoint:]
        if "label_available_utc" in data:
            check_issue = data.loc[cal_check, "issue_time_utc"].min()
            cal_fit = cal_fit[data.loc[cal_fit, "label_available_utc"].lt(check_issue)]
        raw = np.clip(bundle.raw_predict(x.loc[cal_fit]), 1e-6, 1-1e-6)
        logits = np.log(raw/(1-raw)).reshape(-1, 1)
        sigmoid = LogisticRegression(C=1., max_iter=1000).fit(logits, y.loc[cal_fit])
        # Check calibration on a later, untouched half of the calibration block.
        check_raw = bundle.raw_predict(x.loc[cal_check])
        check_logits = np.log(np.clip(check_raw,1e-6,1-1e-6)/(1-np.clip(check_raw,1e-6,1-1e-6))).reshape(-1,1)
        if brier_score_loss(y.loc[cal_check], sigmoid.predict_proba(check_logits)[:,1]) < brier_score_loss(y.loc[cal_check], check_raw):
            bundle.calibrator = sigmoid
        pval = bundle.raw_predict(x.loc[val])
        thresholds = np.linspace(.05, .60, 112)
        # F0.5 values precision twice as strongly as recall, limiting false alarms.
        # The threshold is selected strictly before the held-out test period.
        bundle.review_threshold = max(.2, float(max(thresholds, key=lambda t: fbeta_score(y.loc[val], pval >= t, beta=.5, zero_division=0))))
        bundle.threshold = bundle.review_threshold
    else:
        residuals = y.loc[cal] - bundle.predict(x.loc[cal])
        bundle.interval = tuple(map(float, np.quantile(residuals, [.1, .9])))
    report = {"selected_model": choice, "selection_rule": "validation PR-AUC" if kind=="classification" else "validation RMSE",
              "candidate_validation_metrics": scores, "split_dates": SPLIT_DATES,
              "partition_rows": split.value_counts().to_dict(), "features": columns,
              "calibrated": bundle.calibrator is not None, "decision_threshold": bundle.threshold,
              "review_threshold": bundle.review_threshold,
              "threshold_rule": "validation F0.5 with a 20% minimum; precision-oriented review policy"}
    records = []
    for part, mask in (("validation", val), ("calibration", cal), ("test", test)):
        # Earlier reported predictions must not use a calibrator fitted later in time.
        p = bundle.predict(x.loc[mask]) if part=="test" else bundle.raw_predict(x.loc[mask])
        bp = baseline.predict(x.loc[mask])
        kwargs = {"threshold": bundle.threshold} if kind == "classification" else {}
        report[part] = {"model": metric(y.loc[mask], p, **kwargs), "baseline": metric(y.loc[mask], bp),
                        "by_airport": {}}
        if kind == "classification":
            report[part]["review_policy"] = metric(y.loc[mask], p, threshold=bundle.review_threshold)
        for airport in ("ATL", "DFW"):
            sel = data.loc[mask].airport.eq(airport).to_numpy()
            report[part]["by_airport"][airport] = {
                "model": metric(y.loc[mask].to_numpy()[sel], p[sel], **kwargs),
                "baseline": metric(y.loc[mask].to_numpy()[sel], bp[sel])}
        d = data.loc[mask].copy()
        d["partition"], d["actual"], d["prediction"], d["baseline_prediction"] = part, y.loc[mask], p, bp
        d["decision_threshold"] = bundle.threshold
        d["interval_low"] = np.maximum(0, p + bundle.interval[0]) if kind=="regression" else np.nan
        d["interval_high"] = np.maximum(0, p + bundle.interval[1]) if kind=="regression" else np.nan
        records.append(d)
    sample = x.loc[val].sample(min(600, sum(val)), random_state=17)
    importance = permutation_importance(bundle.model, sample, y.loc[sample.index], n_repeats=2,
            random_state=17, scoring="average_precision" if kind=="classification" else "neg_mean_squared_error")
    report["feature_importance"] = sorted([{"feature": c, "importance": float(v)}
        for c, v in zip(columns, importance.importances_mean)], key=lambda row: row["importance"], reverse=True)[:15]
    return bundle, report, pd.concat(records, ignore_index=True)
