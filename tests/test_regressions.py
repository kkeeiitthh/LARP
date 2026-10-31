"""Regression tests for cleaning, timestamps, forecast availability and target semantics."""
import numpy as np
import pandas as pd
import pytest
from data_loading import load_flights, scheduled_arrival_utc, throughput_events, attach_schedule_features
from features import add_wind_features, asof_weather, hhmm_minutes
from noaa_parser import parse_noaa_file


def test_field_specific_noaa_quality_and_units(tmp_path):
    path = tmp_path / "ATL_2024.csv"
    pd.DataFrame({
        "STATION": ["72219013874"]*2, "DATE": ["2024-01-01T00:00:00","2024-01-01T01:00:00"],
        "WND": ["080,1,N,0100,1","999,9,N,9999,9"],
        "TMP": ["+0123,1","+9999,9"], "DEW": ["+0050,1","+9999,9"],
        "SLP": ["10132,1","99999,9"], "CIG": ["00300,5,9,N","00200,3,9,N"],
        "VIS": ["001000,1,N,9","999999,9,N,9"],
        "OC1": ["0150,1",None], "AJ1": ["0012,9,1,999999,9,3",None],
        "AA1": ["06,0120,9,1","01,0015,9,1"], "AA2": ["01,0005,9,1",None],
    }).to_csv(path, index=False)
    got = parse_noaa_file(path)
    assert got.ceiling_m.iloc[0] == 300  # QC=5; last token N is CAVOK, not QC.
    assert np.isnan(got.ceiling_m.iloc[1])
    assert got.visibility_m.iloc[0] == 1000  # Uses distance QC, not variability QC.
    assert got.snow_depth_cm.iloc[0] == 12  # Already cm; QC token 2.
    assert got.precipitation_mm.iloc[0] == .5  # Reject the six-hour amount.
    assert got.temperature_c.iloc[0] == 12.3
    assert np.isclose(got.wind_speed_kt.iloc[0],19.43844)
    assert got.iloc[1][["temperature_c","visibility_m","wind_speed_kt"]].isna().all()


def test_missing_weather_indicators_stay_missing():
    d=pd.DataFrame({"airport":["ATL"],"wind_direction_deg":[np.nan],"wind_speed_kt":[np.nan],
        "wind_gust_kt":[np.nan],"visibility_m":[np.nan],"precipitation_mm":[np.nan]})
    got=add_wind_features(d)
    assert got[["precipitation_indicator","low_visibility_indicator","severe_weather_indicator"]].isna().all().all()


@pytest.mark.parametrize("value", [2401, 2460, -1, 123.5, 1260])
def test_invalid_airline_times(value):
    assert pd.isna(hhmm_minutes(pd.Series([value])).iloc[0])


def test_reporting_latency_and_future_mutation():
    e=pd.DataFrame({"airport":["ATL"],"issue_time_utc":pd.to_datetime(["2024-01-01T05:00Z"])})
    w=pd.DataFrame({"airport":["ATL"]*3,"observation_time_utc":pd.to_datetime(["2024-01-01T04:00Z","2024-01-01T04:55Z","2024-01-01T06:00Z"]),
        "wind_direction_deg":[80]*3,"wind_speed_kt":[5.,50.,80.],"wind_gust_kt":[np.nan]*3,
        "precipitation_mm":[np.nan]*3,"visibility_m":[10000]*3})
    before=asof_weather(e,w)
    w.loc[1:,"wind_speed_kt"]=9999
    after=asof_weather(e,w)
    assert before.wind_speed_kt.iloc[0]==after.wind_speed_kt.iloc[0]==5.
    assert before.available_time_utc.iloc[0] <= e.issue_time_utc.iloc[0]


def _flights():
    return pd.DataFrame({
        "FL_DATE":pd.to_datetime(["2024-01-01","2024-01-04"]), "ORIGIN":["DFW","DFW"],
        "DEST":["ATL","ATL"],"ORIGIN_STATE_ABR":["TX","TX"],"CRS_DEP_TIME":[2330,1200],
        "CRS_ARR_TIME":[230,1500],"CRS_ELAPSED_TIME":[120,120],"ARR_DELAY":[1500.,np.nan],
        "ARR_TIME":[330.,np.nan],"CANCELLED":[0,1],"DIVERTED":[0,0],"DISTANCE":[700,700]})


def test_overnight_and_more_than_24hour_delay_target():
    f=_flights()
    assert scheduled_arrival_utc(f).iloc[0]==pd.Timestamp("2024-01-02T07:30Z")
    got=throughput_events(f,3)
    target=got[(got.airport=="ATL")&(got.event_time_utc==pd.Timestamp("2024-01-03T08:00Z"))]
    assert target.observed_hourly_arrivals.iloc[0]==1
    assert got.observed_hourly_arrivals.sum()==1  # cancelled flight excluded


def test_scheduled_demand_independent_of_actual_outcomes():
    f=_flights()
    e=pd.DataFrame({"airport":["ATL"],"event_time_utc":pd.to_datetime(["2024-01-02T07:00Z"])})
    a=attach_schedule_features(e,f)
    f["ARR_DELAY"]=9999; f["ARR_TIME"]=9999; f["CANCELLED"]=1
    b=attach_schedule_features(e,f)
    pd.testing.assert_frame_equal(a,b)
    assert a.scheduled_arrivals_offset0.iloc[0]==1
