"""Stage 1: reproducible, cutoff-aware TAF and BTS enrichment. See agent.md."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
AIRPORT_TZ = {"ATL": "America/New_York", "DFW": "America/Chicago"}
RWY_HEADING = {"ATL": 80, "DFW": 180}
KEY = ["flight_id", "airport", "event_type", "lead_hours"]
METAR_COLUMNS = [
    "temp_c", "dewpt_c", "slp_hpa", "wind_dir_deg", "wind_kt", "gust_kt",
    "vis_mi", "ceiling_ft", "headwind_kt", "crosswind_kt", "peak_wind_kt",
    "xwind_max_kt", "tailwind_max_kt", "vis_min_mi", "ceiling_min_ft",
    "precip_mm", "thunder", "precip_type", "flight_category", "n_obs",
]
BTS_COLUMNS = [
    "FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline",
    "Origin", "Dest", "CRSDepTime", "CRSArrTime", "CRSElapsedTime",
    "DepDelay", "ArrDelay", "TaxiOut", "TaxiIn", "Cancelled", "Diverted",
]
IEM = "https://mesonet.agron.iastate.edu"
BTS = "https://transtats.bts.gov/PREZIP/"
AIRPORT_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", compresslevel=1, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False)
    temporary.replace(path)


def validate_download(path, kind):
    if kind == "bts":
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            require(len(members) == 1, "BTS archive must contain one CSV")
            with archive.open(members[0]) as handle:
                header = pd.read_csv(handle, nrows=0).columns
            require(set(BTS_COLUMNS) <= set(header), "Unexpected BTS columns")
    elif kind == "taf":
        header = pd.read_csv(path, nrows=0).columns
        require({"station", "valid", "fx_valid", "raw", "ftype", "product_id"} <= set(header),
                "Unexpected TAF columns")
    elif kind == "metadata":
        value = json.loads(path.read_text())
        require(isinstance(value.get("data"), list), "Unexpected TAF metadata")
        if value["data"]:
            require({"station", "product_id", "utc_issued", "utc_issue", "utc_expire"}
                    <= set(value["data"][0]), "TAF validity metadata is missing")
    else:
        require(pd.read_csv(path, header=None, nrows=1).shape[1] == 14,
                "Unexpected airport metadata")


def fetch_cached(path, url, kind, offline=False):
    """Only promote validated responses; existing caches must match their receipt."""
    path = Path(path)
    receipt = path.with_name(path.name + ".source.json")
    if path.exists():
        validate_download(path, kind)
        digest = sha256(path)
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            require(previous["url"] == url and previous["sha256"] == digest,
                    f"Cache mismatch: remove and refetch {path}")
        return {"path": str(path), "url": url, "sha256": digest, "bytes": path.stat().st_size}
    if offline:
        raise FileNotFoundError(f"Offline cache missing: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    partial_receipt = path.with_name(path.name + ".part.json")
    for attempt in range(3):
        try:
            print(f"Downloading {path.name}", flush=True)
            headers = {"User-Agent": "LARP-Stage1/1.0"}
            offset = temporary.stat().st_size if kind == "bts" and temporary.exists() else 0
            if offset and partial_receipt.exists():
                partial = json.loads(partial_receipt.read_text())
                require(partial["url"] == url, "Partial download URL mismatch")
                headers.update({"Range": f"bytes={offset}-", "If-Range": partial["etag"]})
            else:
                offset = 0
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                response_headers = getattr(response, "headers", {})
                resumed = offset and getattr(response, "status", 200) == 206
                if resumed:
                    require(response_headers.get("Content-Range", "").startswith(f"bytes {offset}-"),
                            "Unexpected resumed byte range")
                etag = response_headers.get("ETag") or response_headers.get("Last-Modified")
                if kind == "bts" and etag:
                    write_json(partial_receipt, {"url": url, "etag": etag})
                with temporary.open("ab" if resumed else "wb") as out:
                    shutil.copyfileobj(response, out, length=1024 * 1024)
                length = response_headers.get("Content-Length")
                if length:
                    require(temporary.stat().st_size == int(length) + (offset if resumed else 0),
                            "Incomplete response; retained bytes for resume")
            validate_download(temporary, kind)
            temporary.replace(path)
            partial_receipt.unlink(missing_ok=True)
            record = {"path": str(path), "url": url, "sha256": sha256(path),
                      "bytes": path.stat().st_size}
            write_json(receipt, record)
            return record
        except (OSError, ValueError, zipfile.BadZipFile, pd.errors.ParserError) as error:
            if kind != "bts" or isinstance(error, zipfile.BadZipFile):
                temporary.unlink(missing_ok=True)
                partial_receipt.unlink(missing_ok=True)
            print(f"Attempt {attempt + 1} failed for {path.name}: {error}", flush=True)
            if attempt == 2:
                raise RuntimeError(f"Download failed: {url}: {error}") from error
            time.sleep(2 ** attempt)


def months_between(start, end):
    return pd.date_range(pd.Timestamp(start).to_period("M").start_time,
                         pd.Timestamp(end).to_period("M").start_time, freq="MS")


def download_sources(cache, start, end, offline=False):
    jobs = [(cache / "airports.dat", AIRPORT_URL, "airports", offline)]
    for month in months_between(start, end):
        following = month + pd.offsets.MonthBegin()
        name = month.strftime("%Y-%m")
        filename = ("On_Time_Reporting_Carrier_On_Time_Performance_1987_present_"
                    f"{month.year}_{month.month}.zip")
        jobs.append((cache / "bts" / filename, BTS + filename, "bts", offline))
        for airport in AIRPORT_TZ:
            station = "K" + airport
            params = {"station": station, "sts": month.strftime("%Y-%m-%dT00:00Z"),
                      "ets": following.strftime("%Y-%m-%dT00:00Z"), "fmt": "csv", "tz": "UTC"}
            jobs.append((cache / "taf" / f"{airport}_{name}.csv",
                         IEM + "/cgi-bin/request/taf.py?" + urllib.parse.urlencode(params), "taf", offline))
            params = {"station": station, "sts": month.strftime("%Y-%m-%d"),
                      "ets": following.strftime("%Y-%m-%d")}
            jobs.append((cache / "taf" / f"{airport}_{name}.json",
                         IEM + "/api/1/nws/taf_overview.json?" + urllib.parse.urlencode(params),
                         "metadata", offline))
    iem_slots = threading.Semaphore(2)
    def download(job):
        if job[2] in ("taf", "metadata"):
            with iem_slots:
                return fetch_cached(*job)
        return fetch_cached(*job)
    with ThreadPoolExecutor(max_workers=6) as workers:
        futures = [workers.submit(download, job) for job in jobs]
        try:
            records = [future.result() for future in as_completed(futures)]
        except Exception:
            for future in futures:
                future.cancel()
            raise
    return sorted(records, key=lambda record: record["path"])


def local_to_utc(values, timezone):
    local = pd.to_datetime(values, errors="coerce")
    return local.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")


def airport_utc(values, airports, zones):
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    for timezone in airports.map(zones).dropna().unique():
        mask = airports.map(zones).eq(timezone)
        result.loc[mask] = local_to_utc(values.loc[mask], timezone)
    return result


def airport_zones(path):
    data = pd.read_csv(path, header=None, na_values=[r"\N"])
    data = data.dropna(subset=[4, 11]).drop_duplicates(4)
    return {**dict(zip(data[4], data[11])), **AIRPORT_TZ}


def event_keys(flights, side="ORIGIN", lead_hours=3):
    require(side in ("ORIGIN", "DEST") and lead_hours >= 1, "Invalid side or lead time")
    column = "sched_dep" if side == "ORIGIN" else "sched_arr"
    parts = []
    for airport, zone in AIRPORT_TZ.items():
        sub = flights.loc[flights[side].eq(airport), ["flight_id", column]].copy()
        sub["airport"] = airport
        sub["event_utc"] = local_to_utc(sub[column], zone)
        parts.append(sub.drop(columns=column))
    result = pd.concat(parts, ignore_index=True)
    result["event_type"] = "departure" if side == "ORIGIN" else "arrival"
    result["lead_hours"] = lead_hours
    result["prediction_utc"] = result.event_utc - pd.Timedelta(hours=lead_hours)
    return result


def build_events(flights, zones):
    require(flights.flight_id.is_unique, "Duplicate flight_id in input")
    departure = airport_utc(flights.sched_dep, flights.ORIGIN, zones)
    arrival = airport_utc(flights.sched_arr, flights.DEST, zones)
    inconsistent = ((arrival - departure).dt.total_seconds() / 60 - flights.CRS_ELAPSED_TIME).abs().gt(1)
    bad_ids = set(flights.loc[inconsistent, "flight_id"])
    parts = []
    for side in ("ORIGIN", "DEST"):
        for lead in (3, 6, 12):
            part = event_keys(flights, side, lead).merge(flights, on="flight_id", validate="many_to_one")
            part["timestamp_valid"] = part.event_utc.notna() & ~part.flight_id.isin(bad_ids)
            part.loc[~part.timestamp_valid, ["event_utc", "prediction_utc"]] = pd.NaT
            delay = part.DEP_DELAY if side == "ORIGIN" else part.ARR_DELAY
            part["delay_minutes"] = delay
            part["severe_delay"] = delay.gt(45).astype("boolean").mask(delay.isna() | part.CANCELLED.eq(1))
            local = pd.to_datetime(part.sched_dep if side == "ORIGIN" else part.sched_arr)
            part["event_hour_local"] = local.dt.hour
            part["event_day_of_week"] = local.dt.dayofweek
            part["event_month"] = local.dt.month
            parts.append(part)
    return pd.concat(parts, ignore_index=True), sorted(bad_ids)


def attach_metar(events, hourly):
    require(not hourly.duplicated(["airport", "hour_utc"]).any(), "Duplicate METAR hour keys")
    weather = hourly.copy()
    weather["known_at_utc"] = pd.to_datetime(weather.known_at_utc, utc=True)
    weather.loc[weather.vis_mi.isna() & weather.ceiling_ft.isna(), "flight_category"] = pd.NA
    weather = weather.loc[weather.n_obs.gt(0) & weather.known_at_utc.notna()]
    require(not weather.duplicated(["airport", "known_at_utc"]).any(), "Duplicate METAR observation times")
    columns = ["known_at_utc"] + METAR_COLUMNS
    parts = []
    for airport in AIRPORT_TZ:
        left = events.loc[events.airport.eq(airport), ["prediction_utc"]].copy()
        left["_row"] = left.index
        right = weather.loc[weather.airport.eq(airport), columns].sort_values("known_at_utc")
        # shortcut: observation time proxies availability; use receipt timestamps with a live feed.
        valid = pd.merge_asof(left.dropna(subset=["prediction_utc"]).sort_values("prediction_utc"), right,
                              left_on="prediction_utc", right_on="known_at_utc", direction="backward",
                              tolerance=pd.Timedelta(hours=2))
        invalid = left.loc[left.prediction_utc.isna()]
        parts.append(pd.concat([valid, invalid], ignore_index=True).set_index("_row")[columns])
    features = pd.concat(parts).reindex(events.index).add_prefix("metar_")
    features["metar_missing"] = features.metar_known_at_utc.isna()
    return events.join(features)


def attach_weather(flights, hourly, side="ORIGIN", lead_hours=3):
    """Notebook-compatible join with the exact prediction cutoff and a 2h age limit."""
    keys = event_keys(flights, side, lead_hours)
    out = attach_metar(keys, hourly)
    result = out[["flight_id", "event_utc", "metar_known_at_utc"]].rename(
        columns={"metar_known_at_utc": "known_at_utc"})
    for column in METAR_COLUMNS:
        result[f"{side.lower()}_{column}_lag{lead_hours}h"] = out["metar_" + column]
    return result


def array_value(value):
    if not isinstance(value, str) or not value.strip():
        return []
    parsed = ast.literal_eval(value)
    require(isinstance(parsed, list), "Expected a TAF array")
    return parsed


def normalize_taf(segments, metadata):
    metadata = metadata.rename(columns={"utc_issued": "issued_utc", "utc_issue": "valid_from_utc",
                                         "utc_expire": "valid_to_utc"}).copy()
    for column in ("issued_utc", "valid_from_utc", "valid_to_utc"):
        metadata[column] = pd.to_datetime(metadata[column], utc=True)
    metadata["airport"] = metadata.station.str.removeprefix("K")
    reports = metadata[["airport", "station", "product_id", "issued_utc", "valid_from_utc",
                        "valid_to_utc", "is_amendment"]].drop_duplicates()
    require(not reports.duplicated(["station", "product_id"]).any(), "Conflicting TAF metadata")
    reports["validity_invalid"] = (reports[["issued_utc", "valid_from_utc", "valid_to_utc"]].isna().any(axis=1)
                                   | reports.valid_to_utc.le(reports.valid_from_utc))
    data = segments.drop(columns="is_amendment", errors="ignore").merge(
        reports, on=["station", "product_id"], how="left", validate="many_to_one")
    require(data.issued_utc.notna().all(), "TAF segments lack issuance metadata")
    require(pd.to_datetime(data.valid, utc=True).eq(data.issued_utc).all(),
            "TAF segment issuance differs from metadata")
    data["segment_from_utc"] = pd.to_datetime(data.fx_valid, utc=True)
    data["segment_to_utc"] = pd.to_datetime(data.fx_valid_end, utc=True)
    data["prevailing"] = data.ftype.isin(["Observation", "Forecast"])
    observation = data.ftype.eq("Observation")
    data.loc[observation, "segment_from_utc"] = data.loc[observation, "valid_from_utc"]
    data = data.sort_values(["station", "product_id", "segment_from_utc", "raw"]).drop_duplicates()
    prevailing = data.loc[data.prevailing]
    next_start = prevailing.groupby(["station", "product_id"]).segment_from_utc.shift(-1)
    data.loc[prevailing.index, "segment_to_utc"] = next_start.fillna(prevailing.valid_to_utc)
    data["segment_to_utc"] = data.segment_to_utc.fillna(data.valid_to_utc)
    for column in ("sknt", "drct", "gust", "visibility"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    # IEM's numeric parser omits VRB winds; retain the speed without inventing direction.
    variable = data.raw.str.extract(r"\bVRB(\d{2,3})(?:G(\d{2,3}))?KT")
    data["sknt"] = data.sknt.fillna(pd.to_numeric(variable[0], errors="coerce"))
    data["gust"] = data.gust.fillna(pd.to_numeric(variable[1], errors="coerce"))
    ceilings, clear, thunder, freezing = [], [], [], []
    for row in data.itertuples():
        sky, levels, wx = array_value(row.skyc), array_value(row.skyl), array_value(row.presentwx)
        ceiling = [float(level) for cover, level in zip(sky, levels)
                   if cover in ("BKN", "OVC", "VV") and level is not None]
        ceilings.append(min(ceiling) if ceiling else np.nan)
        clear.append(bool(sky) and all(cover in ("FEW", "SCT", "SKC", "CLR", "NSC", "NCD") for cover in sky))
        thunder.append(any("TS" in code for code in wx))
        freezing.append(any("FZRA" in code or "FZDZ" in code for code in wx))
    data["ceiling_ft"], data["no_ceiling"], data["thunder"], data["freezing_precip"] = ceilings, clear, thunder, freezing
    return data, reports.sort_values(["airport", "issued_utc", "product_id"]).reset_index(drop=True)


def attach_taf(events, segments, reports):
    columns = ["taf_product_id", "taf_issued_utc", "taf_valid_from_utc", "taf_valid_to_utc",
               "taf_segment_from_utc", "taf_segment_to_utc", "taf_is_amendment", "taf_wind_kt",
               "taf_wind_dir_deg", "taf_gust_kt", "taf_vis_mi", "taf_ceiling_ft", "taf_no_ceiling",
               "taf_thunder", "taf_freezing_precip", "taf_tempo", "taf_prob30", "taf_prob40", "taf_becmg",
               "taf_conditional_thunder", "taf_conditional_vis_min_mi", "taf_conditional_ceiling_min_ft"]
    groups = {(airport, product): group.to_dict("records")
              for (airport, product), group in segments.groupby(["airport", "product_id"])}
    by_airport = {airport: group for airport, group in reports.groupby("airport")}
    records = []
    for event in events[["airport", "event_utc", "prediction_utc"]].itertuples(index=False):
        out = dict.fromkeys(columns)
        available = by_airport.get(event.airport)
        if pd.notna(event.prediction_utc) and available is not None:
            position = available.issued_utc.array.asi8.searchsorted(event.prediction_utc.value, side="right") - 1
            if position >= 0:
                report = available.iloc[position]
                if report.valid_from_utc <= event.event_utc < report.valid_to_utc:
                    matching = [s for s in groups.get((event.airport, report.product_id), [])
                                if s["segment_from_utc"] <= event.event_utc < s["segment_to_utc"]]
                    base = [s for s in matching if s["prevailing"]]
                    if base:
                        require(len(base) == 1, "Overlapping prevailing TAF segments")
                        selected = base[0]
                        out.update(zip(columns[:7], [report.product_id, report.issued_utc,
                                   report.valid_from_utc, report.valid_to_utc, selected["segment_from_utc"],
                                   selected["segment_to_utc"], report.is_amendment]))
                        for source, target in [("sknt", "wind_kt"), ("drct", "wind_dir_deg"),
                                               ("gust", "gust_kt"), ("visibility", "vis_mi"),
                                               ("ceiling_ft", "ceiling_ft"), ("no_ceiling", "no_ceiling"),
                                               ("thunder", "thunder"), ("freezing_precip", "freezing_precip")]:
                            out["taf_" + target] = selected[source]
                        conditional = [s for s in matching if not s["prevailing"]]
                        for token, name in [("TEMPO", "tempo"), ("PROB30", "prob30"),
                                            ("PROB40", "prob40"), ("BECMG", "becmg")]:
                            out["taf_" + name] = any(s["raw"].startswith(token) for s in conditional)
                        out["taf_conditional_thunder"] = any(s["thunder"] for s in conditional)
                        for source, target in [("visibility", "vis_min_mi"), ("ceiling_ft", "ceiling_min_ft")]:
                            values = [s[source] for s in conditional if pd.notna(s[source])]
                            out["taf_conditional_" + target] = min(values) if values else np.nan
        records.append(out)
    features = pd.DataFrame(records, index=events.index, columns=columns)
    for column in [c for c in columns if c.endswith("_utc")]:
        features[column] = pd.to_datetime(features[column], utc=True)
    for column in ("taf_wind_kt", "taf_wind_dir_deg", "taf_gust_kt"):
        features[column] = pd.to_numeric(features[column], errors="coerce")
    angle = np.deg2rad(features.taf_wind_dir_deg - events.airport.map(RWY_HEADING))
    # shortcut: gusts use sustained direction; upgrade when directional gust forecasts are available.
    speed = np.fmax(features.taf_wind_kt, features.taf_gust_kt)
    features["taf_crosswind_kt"] = np.where(speed == 0, 0, (speed * np.sin(angle)).abs())
    features["taf_tailwind_kt"] = np.where(speed == 0, 0, (-speed * np.cos(angle)).clip(lower=0))
    features["taf_missing"] = features.taf_product_id.isna()
    return events.join(features)


def normalize_bts(frame, zones):
    data = frame.copy()
    day = pd.to_datetime(data.FlightDate, errors="coerce")
    clock = pd.to_numeric(data.CRSDepTime, errors="coerce")
    valid_clock = clock.between(0, 2400) & clock.mod(100).lt(60) & clock.mod(1).eq(0)
    local = day + pd.to_timedelta((clock // 100 * 60 + clock % 100).where(valid_clock), unit="m")
    data["sched_dep_utc"] = airport_utc(local, data.Origin, zones)
    elapsed = pd.to_numeric(data.CRSElapsedTime, errors="coerce").where(lambda s: s.gt(0))
    data["sched_arr_utc"] = data.sched_dep_utc + pd.to_timedelta(elapsed, unit="m")
    destination = data.Dest.map(zones)
    mismatch = pd.Series(False, index=data.index)
    for zone in destination.dropna().unique():
        mask = destination.eq(zone)
        actual_clock = data.loc[mask, "sched_arr_utc"].dt.tz_convert(zone)
        expected = pd.to_numeric(data.loc[mask, "CRSArrTime"], errors="coerce") % 2400
        mismatch.loc[mask] = (actual_clock.dt.hour * 100 + actual_clock.dt.minute).ne(expected)
    data["timestamp_invalid"] = data.sched_dep_utc.isna() | destination.isna() | mismatch
    data.loc[data.timestamp_invalid, ["sched_dep_utc", "sched_arr_utc"]] = pd.NaT
    for side, delay in [("dep", "DepDelay"), ("arr", "ArrDelay")]:
        minutes = pd.to_numeric(data[delay], errors="coerce")
        data[f"actual_{side}_utc"] = data[f"sched_{side}_utc"] + pd.to_timedelta(minutes, unit="m")
    # Cancelled gate returns are not counted as completed airport operations.
    operated = data.Cancelled.eq(0)
    data["wheels_off_utc"] = (data.actual_dep_utc + pd.to_timedelta(data.TaxiOut, unit="m")).where(operated)
    data["wheels_on_utc"] = (data.actual_arr_utc - pd.to_timedelta(data.TaxiIn, unit="m")).where(operated & data.Diverted.eq(0))
    return data


def load_bts(records, zones, cache):
    parts = []
    for record in records:
        path = Path(record["path"])
        if path.suffix != ".zip":
            continue
        selected_path = cache / "bts_selected" / (path.stem + ".csv.gz")
        receipt_path = selected_path.with_name(selected_path.name + ".inputs.json")
        dependencies = {"source": record["sha256"], "airports": sha256(cache / "airports.dat"),
                        "transform": sha256(__file__)}
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        reusable = (selected_path.exists() and previous.get("inputs") == dependencies
                    and previous.get("sha256") == sha256(selected_path))
        if not reusable:
            print(f"Processing {path.name}", flush=True)
            selected = []
            with zipfile.ZipFile(path) as archive:
                name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
                with archive.open(name) as handle:
                    for chunk in pd.read_csv(handle, usecols=BTS_COLUMNS, chunksize=100000, low_memory=False):
                        chunk = chunk.loc[chunk.Origin.isin(AIRPORT_TZ) | chunk.Dest.isin(AIRPORT_TZ)].copy()
                        selected.append(normalize_bts(chunk, zones))
            write_csv(selected_path, pd.concat(selected, ignore_index=True))
            write_json(receipt_path, {"inputs": dependencies, "sha256": sha256(selected_path)})
        parts.append(pd.read_csv(selected_path, parse_dates=["sched_dep_utc", "sched_arr_utc", "actual_dep_utc",
                                                          "actual_arr_utc", "wheels_off_utc", "wheels_on_utc"]))
    data = pd.concat(parts, ignore_index=True)
    key = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest", "CRSDepTime"]
    data = data.drop_duplicates()
    require(not data.duplicated(key).any(), "Conflicting BTS flight keys")
    for column in [c for c in data if c.endswith("_utc")]:
        data[column] = pd.to_datetime(data[column], utc=True)
    return data


def attach_bts(events, bts):
    features = pd.DataFrame(index=events.index)
    for name in ("scheduled_departures", "scheduled_arrivals", "departures_previous_3h",
                 "arrivals_previous_3h", "severe_departure_rate_previous_3h"):
        features["bts_" + name] = np.nan
    features["bts_observed_through_utc"] = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns, UTC]")
    context_parts = []
    for airport in AIRPORT_TZ:
        departures = bts.loc[bts.Origin.eq(airport)]
        arrivals = bts.loc[bts.Dest.eq(airport)]
        scheduled_dep = departures.groupby(departures.sched_dep_utc.dt.floor("h")).size()
        scheduled_arr = arrivals.groupby(arrivals.sched_arr_utc.dt.floor("h")).size()
        context = pd.concat([scheduled_dep.rename("scheduled_departures"),
                             scheduled_arr.rename("scheduled_arrivals")], axis=1).fillna(0).astype(int)
        context.index.name = "hour_utc"
        context["airport"] = airport
        context_parts.append(context.reset_index())
        mask = events.airport.eq(airport) & events.prediction_utc.notna()
        target_hours = events.loc[mask, "event_utc"].dt.floor("h")
        # shortcut: final schedules proxy advance demand; replace with archived schedule snapshots for deployment.
        features.loc[mask, "bts_scheduled_departures"] = target_hours.map(scheduled_dep).fillna(0)
        features.loc[mask, "bts_scheduled_arrivals"] = target_hours.map(scheduled_arr).fillna(0)
        for subset, column, name in [(departures, "wheels_off_utc", "departures"),
                                      (arrivals, "wheels_on_utc", "arrivals")]:
            observed = subset.dropna(subset=[column]).sort_values(column)
            times = observed[column].array.asi8
            require(len(times) > 0, f"No BTS observations for {airport} {name}")
            mask = events.airport.eq(airport) & events.prediction_utc.notna()
            cutoffs = events.loc[mask, "prediction_utc"].array.asi8
            lower = times.searchsorted(cutoffs - pd.Timedelta(hours=3).value, side="right")
            upper = times.searchsorted(cutoffs, side="right")
            features.loc[mask, f"bts_{name}_previous_3h"] = upper - lower
            if name == "departures":
                cumulative = np.r_[0, observed.DepDelay.gt(45).cumsum().to_numpy()]
                count = upper - lower
                features.loc[mask, "bts_severe_departure_rate_previous_3h"] = np.divide(
                    cumulative[upper] - cumulative[lower], count,
                    out=np.full(len(count), np.nan), where=count > 0)
            latest = pd.Series(pd.to_datetime(times[np.maximum(upper - 1, 0)], utc=True), index=events.index[mask])
            latest.loc[upper == lower] = pd.NaT
            previous = features.loc[mask, "bts_observed_through_utc"]
            features.loc[mask, "bts_observed_through_utc"] = previous.where(previous >= latest, latest).fillna(previous)
    features["bts_missing"] = features.bts_scheduled_departures.isna()
    return events.join(features), pd.concat(context_parts, ignore_index=True).sort_values(["hour_utc", "airport"])


def predictor_columns(frame):
    base = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "DISTANCE", "CRS_ELAPSED_TIME", "airport",
            "event_type", "lead_hours", "event_hour_local", "event_day_of_week", "event_month"]
    provenance = {"taf_product_id", "taf_is_amendment"}
    return base + [c for c in frame if c.startswith(("metar_", "taf_", "bts_"))
                   and not c.endswith("_utc") and c not in provenance]


def validate_features(frame, expected_rows):
    require(len(frame) == expected_rows, "Feature join changed event row count")
    require(not frame.duplicated(KEY).any(), "Duplicate feature keys")
    for column in ("metar_known_at_utc", "taf_issued_utc", "bts_observed_through_utc"):
        require(not frame[column].gt(frame.prediction_utc).any(), f"Future information in {column}")
    have_taf = ~frame.taf_missing
    require((frame.loc[have_taf, "taf_valid_from_utc"] <= frame.loc[have_taf, "event_utc"]).all()
            and (frame.loc[have_taf, "event_utc"] < frame.loc[have_taf, "taf_valid_to_utc"]).all(),
            "TAF does not cover event time")
    require((frame.loc[have_taf, "taf_segment_from_utc"] <= frame.loc[have_taf, "event_utc"]).all()
            and (frame.loc[have_taf, "event_utc"] < frame.loc[have_taf, "taf_segment_to_utc"]).all(),
            "TAF segment does not cover event time")
    forbidden = {"flight_id", "DEP_DELAY", "ARR_DELAY", "delay_minutes", "severe_delay", "CANCELLED",
                 "DIVERTED", "TAXI_OUT", "TAXI_IN", "AIR_TIME", "CARRIER_DELAY", "WEATHER_DELAY",
                 "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY", "CANCELLATION_CODE", "TAIL_NUM"}
    require(not forbidden.intersection(predictor_columns(frame)), "Outcome in predictor list")


def run(start="2023-01-01", end="2025-08-01", offline=False, root=ROOT):
    root = Path(root)
    require(pd.Timestamp(start) < pd.Timestamp(end), "--end must be after --start (end is exclusive)")
    output = root / "data" / "stage1" / f"{start}_{end}"
    cache = root / "data" / "external"
    flights_path = root / "data" / "clean" / "flights_clean.csv"
    weather_path = root / "data" / "weather" / "hourly_atl_dfw.csv"
    flights = pd.read_csv(flights_path, parse_dates=["sched_dep", "sched_arr"])
    flights = flights.loc[flights.sched_dep.ge(start) & flights.sched_dep.lt(end)].copy()
    require(len(flights) > 0, "No input flights in requested date range")
    manifest = {"stage": 1, "status": "running", "start": start, "end_exclusive": end, "offline": offline,
                "input_flights": len(flights), "inputs": {str(p.relative_to(root)): sha256(p)
                for p in [flights_path, weather_path]}, "sources": []}
    manifest["inputs"]["stage1_pipeline.py"] = sha256(__file__)
    try:
        # Pad the source window for forecast issuance, previous operations and UTC date rollovers.
        records = download_sources(cache, pd.Timestamp(start) - pd.Timedelta(days=2),
                                   pd.Timestamp(end) + pd.Timedelta(days=1), offline)
        manifest["sources"] = [{**r, "path": str(Path(r["path"]).relative_to(root))} for r in records]
        zones = airport_zones(cache / "airports.dat")
        events, invalid = build_events(flights, zones)
        print(f"Built {len(events)} event/horizon rows", flush=True)
        hourly = pd.read_csv(weather_path, parse_dates=["hour_utc", "known_at_utc"])
        enriched = attach_metar(events, hourly)
        segment_frames, metadata_frames = [], []
        for record in records:
            path = Path(record["path"])
            if path.parent.name == "taf" and path.suffix == ".csv":
                segment_frames.append(pd.read_csv(path))
            elif path.parent.name == "taf" and path.suffix == ".json":
                metadata_frames.append(pd.DataFrame(json.loads(path.read_text())["data"]))
        segments, reports = normalize_taf(pd.concat(segment_frames, ignore_index=True),
                                          pd.concat(metadata_frames, ignore_index=True))
        print(f"Joining {len(reports)} TAF issuances", flush=True)
        enriched = attach_taf(enriched, segments, reports)
        bts = load_bts(records, zones, cache)
        print(f"Joining context from {len(bts)} BTS flights", flush=True)
        enriched, context = attach_bts(enriched, bts)
        expected = int((flights.ORIGIN.isin(AIRPORT_TZ).sum() + flights.DEST.isin(AIRPORT_TZ).sum()) * 3)
        validate_features(enriched, expected)
        require(set(enriched.flight_id) == set(flights.flight_id), "Flight IDs lost during enrichment")
        enriched = enriched.sort_values(KEY).reset_index(drop=True)
        coverage = enriched.groupby(["airport", "event_type", "lead_hours"]).agg(
            rows=("flight_id", "size"), metar_missing=("metar_missing", "sum"),
            taf_missing=("taf_missing", "sum"), bts_missing=("bts_missing", "sum")).reset_index()
        write_csv(output / "flight_features.csv.gz", enriched)
        write_csv(output / "airport_hour_context.csv.gz", context)
        write_csv(output / "taf_segments.csv.gz", segments)
        write_csv(output / "coverage.csv.gz", coverage)
        manifest.update(status="complete", feature_rows=len(enriched), distinct_flight_ids=enriched.flight_id.nunique(),
                        invalid_input_timestamp_ids=invalid, bts_flights=len(bts),
                        bts_invalid_timestamps=int(bts.timestamp_invalid.sum()),
                        taf_issuances=len(reports), taf_invalid_validity_issuances=int(reports.validity_invalid.sum()),
                        predictors=predictor_columns(enriched),
                        coverage=coverage.to_dict("records"),
                        checks={"unique_keys": True, "ids_preserved": True, "no_future_timestamps": True,
                                "taf_validity": True, "outcomes_excluded_from_predictors": True},
                        outputs={p.name: sha256(p) for p in sorted(output.glob("*.csv.gz"))},
                        limitations=["BTS is reporting-carrier throughput, not AAR.",
                                     "BTS final schedules are retrospective demand proxies.",
                                     "Observation/operation timestamps proxy availability; publication latency is not reconstructed.",
                                     "METAR observations older than two hours are excluded.",
                                     "Original flight CSV is absent; existing cleaned labels are preserved."])
        write_json(output / "manifest.json", manifest)
        summary = (f"# Stage 1 validation\n\nWindow: {start} to {end} (exclusive).\n\n"
                   f"- Flights: {len(flights):,}; feature rows: {len(enriched):,}.\n"
                   f"- TAF issuances: {len(reports):,}; BTS flights: {len(bts):,}.\n"
                   f"- Missing METAR/TAF/BTS joins: {int(enriched.metar_missing.sum()):,} / "
                   f"{int(enriched.taf_missing.sum()):,} / {int(enriched.bts_missing.sum()):,}.\n"
                   f"- Inconsistent input flight timestamps: {len(invalid)}; retained with unavailable features.\n"
                   f"- Invalid BTS timestamps excluded from counts: {int(bts.timestamp_invalid.sum()):,}.\n"
                   f"- Invalid TAF issuance validity intervals left unavailable: {int(reports.validity_invalid.sum()):,}.\n"
                   "- Row counts, identifiers, unique keys, exact cutoffs and TAF validity passed.\n\n"
                   "See the generated manifest for hashes, source URLs, predictor columns and detailed coverage.\n"
                   "Model training and accuracy evaluation are outside Stage 1.\n")
        (root / "STAGE1_VALIDATION.md").write_text(summary)
        print(summary, flush=True)
        return manifest
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "last_failure.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-01-01", help="Inclusive local departure date")
    parser.add_argument("--end", default="2025-08-01", help="Exclusive local departure date")
    parser.add_argument("--offline", action="store_true", help="Require cached sources; do not use the network")
    args = parser.parse_args()
    run(args.start, args.end, args.offline)


if __name__ == "__main__":
    main()
