import numpy as np
import pandas as pd
from features import add_wind_features,hhmm_minutes,asof_weather
from modeling import chronological_split,FORBIDDEN,NUMERIC

def test_hhmm_and_2400():
    assert hhmm_minutes(pd.Series([726,1550,2400,1260])).tolist()[:3]==[446,950,1440]
    assert pd.isna(hhmm_minutes(pd.Series([1260])).iloc[0])

def test_runway_wind_sign_convention():
    frame=pd.DataFrame({"airport":["ATL","ATL","ATL"],"wind_direction_deg":[80.,170.,260.],"wind_speed_kt":[10.,10.,10.],"wind_gust_kt":[np.nan]*3,"precipitation_mm":[np.nan]*3,"visibility_m":[10000.]*3})
    got=add_wind_features(frame)
    assert np.isclose(got.headwind_kt.iloc[0],10)  # wind from runway heading is headwind
    assert np.isclose(got.crosswind_kt.iloc[1],10)
    assert np.isclose(got.headwind_kt.iloc[2],-10) # opposite direction is tailwind

def test_asof_never_uses_future_weather():
    e=pd.DataFrame({"airport":["ATL"],"issue_time_utc":pd.to_datetime(["2024-01-01T05:00Z"])})
    w=pd.DataFrame({"airport":["ATL","ATL"],"observation_time_utc":pd.to_datetime(["2024-01-01T04:00Z","2024-01-01T06:00Z"]),"wind_direction_deg":[80,90],"wind_speed_kt":[5,50],"wind_gust_kt":[np.nan]*2,"precipitation_mm":[np.nan]*2,"visibility_m":[10000]*2})
    assert asof_weather(e,w).wind_speed_kt.iloc[0]==5

def test_chronological_split_and_leakage_list():
    d=pd.DataFrame({"t":pd.date_range("2024-01-01",periods=100,freq="h")}); s=chronological_split(d,"t")
    assert d.loc[s.eq("train"),"t"].max()<d.loc[s.eq("test"),"t"].min()
    assert not FORBIDDEN.intersection(NUMERIC)
