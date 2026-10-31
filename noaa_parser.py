"""NOAA ISD decoding with field-specific quality flags and explicit units.

Reference: https://www.ncei.noaa.gov/pub/data/noaa/isd-format-document.pdf
"""
from pathlib import Path

import numpy as np
import pandas as pd

GOOD_QUALITY = {"0", "1", "4", "5", "9", "A", "C", "I", "M", "P", "R", "U"}
KNOTS_PER_MPS = 1.943844


def _parts(raw, column):
    return raw.get(column, pd.Series(pd.NA, index=raw.index, dtype="string")).astype("string").str.split(",")


def _value(parts, value_index, quality_index, scale=1, missing=()):
    text = parts.str[value_index]
    values = pd.to_numeric(text, errors="coerce")
    return (values.where(~text.isin(missing) & parts.str[quality_index].isin(GOOD_QUALITY)) / scale).astype(float)


def parse_noaa_file(path, airport=None):
    path = Path(path)
    raw = pd.read_csv(path, dtype={"STATION": "string"}, low_memory=False)
    airport = airport or path.stem.split("_")[0]
    expected = {"ATL": "72219013874", "DFW": "72259003927"}[airport]
    if not raw.STATION.dropna().eq(expected).all():
        raise ValueError(f"Unexpected station in {path}")
    out = pd.DataFrame(index=raw.index)
    out["airport"] = airport
    out["observation_time_utc"] = pd.to_datetime(raw.DATE, utc=True, errors="coerce")
    wnd = _parts(raw, "WND")
    out["wind_direction_deg"] = _value(wnd, 0, 1, missing=("999",))
    out["wind_speed_kt"] = _value(wnd, 3, 4, 10, ("9999",)) * KNOTS_PER_MPS
    for column, name in (("TMP", "temperature_c"), ("DEW", "dew_point_c"), ("SLP", "sea_level_pressure_hpa")):
        out[name] = _value(_parts(raw, column), 0, 1, 10, ("+9999", "-9999", "99999"))
    out["visibility_m"] = _value(_parts(raw, "VIS"), 0, 1, missing=("999999",))
    # CIG is height, height QC, determination method, CAVOK. The last field is NOT QC.
    out["ceiling_m"] = _value(_parts(raw, "CIG"), 0, 1, missing=("99999",))
    out["wind_gust_kt"] = _value(_parts(raw, "OC1"), 0, 1, 10, ("9999",)) * KNOTS_PER_MPS
    # AJ1 depth is already centimetres; its QC is field 2, not the water-depth QC.
    out["snow_depth_cm"] = _value(_parts(raw, "AJ1"), 0, 2, 1, ("9999",))
    precip = []
    for col in ("AA1", "AA2", "AA3", "AA4"):
        p = _parts(raw, col)
        # Do not mix 1/6/24-hour accumulation depths. Only 1-hour values feed the model.
        depth = _value(p, 1, 3, 10, ("9999",)).where(pd.to_numeric(p.str[0], errors="coerce").eq(1))
        depth = depth.where(~p.str[2].isin(["2", "5", "6", "7"]))
        precip.append(depth)
    out["precipitation_mm"] = pd.concat(precip, axis=1).max(axis=1)
    manual = _parts(raw, "MW1")
    out["present_weather"] = _value(manual, 0, 1, missing=("99",))
    # WMO manual weather 95-99: thunderstorm. Automated AW codes use a different codebook.
    out["thunderstorm_indicator"] = out.present_weather.between(95, 99).astype(float).where(out.present_weather.notna())
    out["report_type"] = raw.get("REPORT_TYPE", "")
    out = out.dropna(subset=["observation_time_utc"])
    values = [c for c in out if c not in ("airport", "observation_time_utc", "report_type")]
    out["_complete"] = out[values].notna().sum(axis=1)
    out = out.sort_values(["observation_time_utc", "_complete"], kind="stable").drop_duplicates(["airport", "observation_time_utc"], keep="last")
    return out.drop(columns="_complete").reset_index(drop=True)


def load_noaa_directory(directory):
    files = sorted(Path(directory).glob("ATL_20*.csv")) + sorted(Path(directory).glob("DFW_20*.csv"))
    if len(files) != 6:
        raise FileNotFoundError(f"Expected six NOAA files; found {len(files)} in {directory}")
    return pd.concat([parse_noaa_file(p) for p in files], ignore_index=True).sort_values(["airport", "observation_time_utc"])
