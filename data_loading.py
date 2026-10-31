"""Validated flight schedules and targets; source outcomes never enter feature columns."""
from pathlib import Path
import numpy as np
import pandas as pd
from features import calendar_features, hhmm_minutes, local_to_utc, TIMEZONES

STATE_ZONES = {}
for timezone, states in {
    "America/New_York": "CT DE FL GA IN KY MA MD ME MI NC NH NJ NY OH PA RI SC VA VT WV",
    "America/Chicago": "AL AR IA IL KS LA MN MO MS ND NE OK SD TN TX WI",
    "America/Denver": "CO ID MT NM UT WY",
    "America/Los_Angeles": "CA NV OR WA",
    "America/Phoenix": "AZ", "America/Anchorage": "AK", "Pacific/Honolulu": "HI",
    "America/Puerto_Rico": "PR VI", "Pacific/Guam": "GU", "Pacific/Saipan": "MP"
}.items():
    STATE_ZONES.update(dict.fromkeys(states.split(), timezone))
ZONE_EXCEPTIONS = {
    **dict.fromkeys(["PNS", "VPS", "ECP", "EVV", "PAH", "OWB"], "America/Chicago"),
    **dict.fromkeys(["TYS", "TRI", "CHA"], "America/New_York"),
    **dict.fromkeys(["ELP", "BFF", "CDR", "DIK", "RAP"], "America/Denver"),
    **dict.fromkeys(["LWS", "PUW"], "America/Los_Angeles"),
    "ADK": "America/Adak",
}


def load_flights(path):
    f = pd.read_csv(path, low_memory=False)
    required = {"FL_DATE", "ORIGIN", "DEST", "ORIGIN_STATE_ABR", "CRS_DEP_TIME", "CRS_ARR_TIME",
                "CRS_ELAPSED_TIME", "ARR_DELAY", "ARR_TIME", "CANCELLED", "DIVERTED"}
    if missing := required - set(f):
        raise ValueError(f"Missing required flight columns: {sorted(missing)}")
    f["FL_DATE"] = pd.to_datetime(f.FL_DATE, errors="coerce")
    for c in ("ORIGIN", "DEST", "ORIGIN_STATE_ABR"):
        f[c] = f[c].astype("string").str.upper().str.strip()
    for c in ("CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "ARR_DELAY", "ARR_TIME",
              "CANCELLED", "DIVERTED", "DISTANCE", "DEP_DELAY", "ACTUAL_ELAPSED_TIME"):
        if c in f:
            f[c] = pd.to_numeric(f[c], errors="coerce")
    return f.drop_duplicates().copy()


def departure_utc(f):
    zones = f.ORIGIN_STATE_ABR.map(STATE_ZONES)
    overrides = f.ORIGIN.map(ZONE_EXCEPTIONS)
    zones = overrides.combine_first(zones)
    base = f.FL_DATE + pd.to_timedelta(hhmm_minutes(f.CRS_DEP_TIME), unit="m")
    result = pd.Series(pd.NaT, index=f.index, dtype="datetime64[ns, UTC]")
    for timezone in zones.dropna().unique():
        mask = zones.eq(timezone) & base.notna()
        result.loc[mask] = base.loc[mask].dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    return result


def scheduled_arrival_utc(f):
    # Elapsed time fixes overnight and timezone crossings; ARR_DELAY fixes >24h delays later.
    arrival = departure_utc(f) + pd.to_timedelta(f.CRS_ELAPSED_TIME, unit="m")
    valid = pd.Series(False, index=f.index)
    for airport, timezone in TIMEZONES.items():
        mask = f.DEST.eq(airport) & arrival.notna()
        wall = arrival.loc[mask].dt.tz_convert(timezone)
        minutes = wall.dt.hour * 60 + wall.dt.minute
        reported = hhmm_minutes(f.loc[mask, "CRS_ARR_TIME"]) % 1440
        circular_error = ((minutes - reported + 720) % 1440 - 720).abs()
        valid.loc[mask] = circular_error.le(2)
    return arrival.where(valid)


def _event_frame(airport, times, horizon):
    d = pd.DataFrame({"airport": airport, "event_time_utc": times})
    d["issue_time_utc"] = d.event_time_utc - pd.Timedelta(hours=horizon)
    d["horizon"] = horizon
    return calendar_features(d.dropna(subset=["event_time_utc"]))


def delay_events(f, horizon):
    mask = f.ORIGIN.isin(TIMEZONES) & f.ARR_DELAY.notna() & f.CANCELLED.eq(0) & f.DIVERTED.eq(0)
    source = f.loc[mask]
    times = local_to_utc(source.FL_DATE, source.CRS_DEP_TIME, source.ORIGIN)
    d = _event_frame(source.ORIGIN, times, horizon)
    d["severe_delay"] = source.loc[d.index, "ARR_DELAY"].gt(45).astype("int8")
    d["carrier"] = source.loc[d.index, "OP_UNIQUE_CARRIER"].astype(str)
    d["destination"] = source.loc[d.index, "DEST"].astype(str)
    d["planned_elapsed_minutes"] = source.loc[d.index, "CRS_ELAPSED_TIME"]
    d["distance_miles"] = source.loc[d.index, "DISTANCE"]
    if "OP_CARRIER_FL_NUM" in source:
        d["flight_number"] = source.loc[d.index, "OP_CARRIER_FL_NUM"].astype(str)
    # Only used to audit label availability at split boundaries, never a model input.
    d["label_available_utc"] = times + pd.to_timedelta(
        source.DEP_DELAY + source.ACTUAL_ELAPSED_TIME + 60, unit="m")
    return d.reset_index(drop=True)


def arrival_schedule(f):
    source = f.loc[f.DEST.isin(TIMEZONES)].copy()
    source["scheduled_arrival_utc"] = scheduled_arrival_utc(source)
    return source


def throughput_events(f, horizon):
    a = arrival_schedule(f)
    completed = a.CANCELLED.eq(0) & a.DIVERTED.eq(0) & a.ARR_TIME.notna() & a.ARR_DELAY.notna()
    # Use delay relative to reconstructed schedule, so even >24h delays keep the correct date.
    actual = a.scheduled_arrival_utc + pd.to_timedelta(a.ARR_DELAY, unit="m")
    consistent = pd.Series(False, index=a.index)
    for airport, timezone in TIMEZONES.items():
        mask = a.DEST.eq(airport) & actual.notna()
        local = actual.loc[mask].dt.tz_convert(timezone)
        difference = ((local.dt.hour*60 + local.dt.minute - hhmm_minutes(a.loc[mask, "ARR_TIME"]) + 720) % 1440 - 720).abs()
        consistent.loc[mask] = difference.le(2)
    arrivals = pd.DataFrame({"airport": a.DEST, "event_time_utc": actual.dt.floor("h")})
    counts = arrivals.loc[completed & consistent].groupby(["airport", "event_time_utc"]).size().rename("observed_hourly_arrivals").reset_index()
    start = pd.Timestamp(f.FL_DATE.min(), tz="UTC") + pd.Timedelta(days=1)
    end = pd.Timestamp(f.FL_DATE.max(), tz="UTC") - pd.Timedelta(hours=1)
    grids = []
    for airport in TIMEZONES:
        grid = pd.DataFrame({"airport": airport, "event_time_utc": pd.date_range(start, end, freq="h")})
        grid = grid.merge(counts, on=["airport", "event_time_utc"], how="left")
        grid["observed_hourly_arrivals"] = grid.observed_hourly_arrivals.fillna(0).astype(int)
        grids.append(grid)
    out = pd.concat(grids, ignore_index=True)
    out["issue_time_utc"] = out.event_time_utc - pd.Timedelta(hours=horizon)
    out["horizon"] = horizon
    out["label_available_utc"] = out.event_time_utc + pd.Timedelta(hours=2)
    return calendar_features(out)


def attach_schedule_features(events, flights):
    """Scheduled demand is available only under the documented frozen-schedule assumption."""
    a = arrival_schedule(flights)
    arr = pd.DataFrame({"airport": a.DEST, "hour_utc": a.scheduled_arrival_utc.dt.floor("h")})
    departures = flights.loc[flights.ORIGIN.isin(TIMEZONES)]
    dep = pd.DataFrame({"airport": departures.ORIGIN,
                        "hour_utc": local_to_utc(departures.FL_DATE, departures.CRS_DEP_TIME, departures.ORIGIN).dt.floor("h")})
    out = events.copy()
    for name, source in (("scheduled_arrivals", arr), ("scheduled_departures", dep)):
        counts = source.dropna().groupby(["airport", "hour_utc"]).size()
        for offset in (-1, 0, 1):
            keys = pd.MultiIndex.from_arrays([out.airport, out.event_time_utc.dt.floor("h") + pd.Timedelta(hours=offset)])
            out[f"{name}_offset{offset}"] = counts.reindex(keys, fill_value=0).to_numpy()
    return out
