"""Features available at a declared issue time; no future observations or outcomes."""
from pathlib import Path
import numpy as np
import pandas as pd

CONFIG = pd.read_csv(Path(__file__).parent / "airport_config.csv").set_index("airport")
HORIZONS = (3, 6, 9, 12)
HEADINGS = CONFIG.runway_heading.to_dict()
TIMEZONES = CONFIG.timezone.to_dict()
WEATHER_BASE = ["wind_speed_kt", "wind_gust_kt", "temperature_c", "dew_point_c",
                "visibility_m", "sea_level_pressure_hpa", "precipitation_mm",
                "snow_depth_cm", "ceiling_m", "thunderstorm_indicator"]
LATENCY_MINUTES = 20


def hhmm_minutes(values):
    v = pd.to_numeric(values, errors="coerce")
    h, m = v // 100, v % 100
    valid = v.eq(v.round()) & ((h.between(0, 23) & m.between(0, 59)) | v.eq(2400))
    return (h * 60 + m).where(valid)


def local_to_utc(dates, hhmm, airports):
    base = pd.to_datetime(dates, errors="coerce") + pd.to_timedelta(hhmm_minutes(hhmm), unit="m")
    result = pd.Series(pd.NaT, index=base.index, dtype="datetime64[ns, UTC]")
    for airport, tz in TIMEZONES.items():
        mask = airports.eq(airport) & base.notna()
        # Ambiguous/nonexistent wall-clock times are unresolvable without more metadata.
        result.loc[mask] = base.loc[mask].dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    return result


def add_wind_features(df):
    out = df.copy()
    angle = np.deg2rad(out.wind_direction_deg - out.airport.map(HEADINGS))
    out["crosswind_kt"] = out.wind_speed_kt * np.sin(angle)
    out["headwind_kt"] = out.wind_speed_kt * np.cos(angle)
    # A calm report has zero components even if direction is unspecified.
    calm = out.wind_speed_kt.eq(0)
    out.loc[calm, ["crosswind_kt", "headwind_kt"]] = 0.
    out["absolute_crosswind_kt"] = out.crosswind_kt.abs()
    out["absolute_headwind_kt"] = out.headwind_kt.abs()
    radians = np.deg2rad(out.wind_direction_deg)
    out["wind_direction_sin"], out["wind_direction_cos"] = np.sin(radians), np.cos(radians)
    out["gust_difference_kt"] = out.wind_gust_kt - out.wind_speed_kt
    out["crosswind_category"] = pd.cut(out.absolute_crosswind_kt, [-np.inf, 10, 20, np.inf], labels=["low", "moderate", "high"])
    out["precipitation_indicator"] = out.precipitation_mm.gt(0).astype(float).where(out.precipitation_mm.notna())
    out["low_visibility_indicator"] = out.visibility_m.lt(5000).astype(float).where(out.visibility_m.notna())
    severe = pd.concat([out.absolute_crosswind_kt.ge(20).where(out.absolute_crosswind_kt.notna()),
                        out.visibility_m.lt(1600).where(out.visibility_m.notna()),
                        out.precipitation_mm.ge(10).where(out.precipitation_mm.notna())], axis=1)
    out["severe_weather_indicator"] = severe.max(axis=1).astype(float)
    out["dewpoint_spread_c"] = out.temperature_c - out.dew_point_c if "temperature_c" in out else np.nan
    return out


def prepare_weather(weather):
    """Build past-only rolling summaries at observation timestamps."""
    groups = []
    for airport, frame in weather.groupby("airport"):
        g = frame.sort_values("observation_time_utc").copy()
        g["observation_time_utc"] = g.observation_time_utc.astype("datetime64[ns, UTC]")
        g = g.set_index("observation_time_utc")
        # Last valid value per variable with an explicit 3h expiry.
        for c in WEATHER_BASE + ["wind_direction_deg"]:
            if c not in g:
                g[c] = np.nan
            seen = pd.Series(g.index, index=g.index).where(g[c].notna()).ffill()
            g[f"{c}_observed_at"] = seen
            g[c] = g[c].ffill().where((pd.Series(g.index, index=g.index) - seen) <= pd.Timedelta("3h"))
        g = add_wind_features(g)
        for c in ["wind_speed_kt", "visibility_m", "precipitation_mm", "sea_level_pressure_hpa", "absolute_crosswind_kt"]:
            for window in (3, 6):
                g[f"{c}_past{window}h_mean"] = g[c].rolling(f"{window}h", min_periods=1).mean()
        g = g.reset_index()
        g["available_time_utc"] = g.observation_time_utc + pd.Timedelta(minutes=LATENCY_MINUTES)
        groups.append(g)
    return pd.concat(groups, ignore_index=True)


def asof_weather(events, weather, prepared=False):
    left = events.sort_values("issue_time_utc").copy()
    right = (weather.copy() if prepared else prepare_weather(weather)).sort_values("available_time_utc")
    for f in (left, right):
        f["airport"] = f.airport.astype(object)
    left["issue_time_utc"] = left.issue_time_utc.astype("datetime64[ns, UTC]")
    right["available_time_utc"] = right.available_time_utc.astype("datetime64[ns, UTC]")
    joined = pd.merge_asof(left, right, left_on="issue_time_utc", right_on="available_time_utc",
                           by="airport", direction="backward", tolerance=pd.Timedelta("3h"))
    joined["weather_age_minutes"] = (joined.issue_time_utc - joined.observation_time_utc).dt.total_seconds() / 60
    assert (joined.available_time_utc.dropna() <= joined.loc[joined.available_time_utc.notna(), "issue_time_utc"]).all()
    for c in WEATHER_BASE + ["wind_direction_deg"]:
        joined[c] = joined[c].where((joined.issue_time_utc - joined[f"{c}_observed_at"]) <= pd.Timedelta("3h"))
    joined = add_wind_features(joined)
    # Retrieve true earlier weather snapshots, never shift flight rows.
    for lag in (3, 6):
        queries = left[["airport", "issue_time_utc"]].copy()
        queries["query"] = queries.issue_time_utc - pd.Timedelta(hours=lag)
        prior = pd.merge_asof(queries, right[["airport", "available_time_utc", "wind_speed_kt", "visibility_m", "wind_speed_kt_observed_at", "visibility_m_observed_at"]],
                              left_on="query", right_on="available_time_utc", by="airport",
                              direction="backward", tolerance=pd.Timedelta("3h"))
        for c in ("wind_speed_kt", "visibility_m"):
            values = prior[c].where((prior["query"] - prior[f"{c}_observed_at"]) <= pd.Timedelta("3h"))
            joined[f"{c}_lag{lag}h"] = values.to_numpy()
    return joined


def calendar_features(d):
    out = d.copy()
    for c in ("hour", "weekday", "month"):
        out[c] = np.nan
    for airport, timezone in TIMEZONES.items():
        mask = out.airport.eq(airport)
        local = out.loc[mask, "event_time_utc"].dt.tz_convert(timezone)
        out.loc[mask, "hour"] = local.dt.hour
        out.loc[mask, "weekday"] = local.dt.day_of_week
        out.loc[mask, "month"] = local.dt.month
    out["hour_sin"] = np.sin(2*np.pi*out.hour/24)
    out["hour_cos"] = np.cos(2*np.pi*out.hour/24)
    out["month_sin"] = np.sin(2*np.pi*out.month/12)
    out["month_cos"] = np.cos(2*np.pi*out.month/12)
    return out
