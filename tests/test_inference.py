"""Exercise fitted artifacts and every airport/horizon UI path after a full training run."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from data_loading import attach_schedule_features
from features import asof_weather, calendar_features
from train import load_inputs

OUT=Path(__file__).resolve().parents[1]/"outputs"


@pytest.mark.skipif(not (OUT/"weather_prepared.joblib").exists(),reason="Run python train.py to generate artifacts")
@pytest.mark.parametrize("airport",["ATL","DFW"])
@pytest.mark.parametrize("horizon",[3,6,9,12])
def test_saved_forecast_matches_exact_single_row_inference(airport,horizon):
    pred=pd.read_csv(OUT/"predictions.csv",low_memory=False,parse_dates=["event_time_utc","issue_time_utc"])
    flights,weather=load_inputs()
    for task in ("delay","throughput"):
        r=pred[(pred.airport==airport)&(pred.forecast_horizon_hours==horizon)&(pred.task==task)&(pred.partition=="test")].iloc[20]
        event=pd.DataFrame({"airport":[airport],"event_time_utc":[r.event_time_utc],
                            "issue_time_utc":[r.issue_time_utc],"horizon":[horizon]})
        if task=="delay":
            for c in ("carrier","destination","planned_elapsed_minutes","distance_miles"):
                event[c]=r[c]
        x=asof_weather(attach_schedule_features(calendar_features(event),flights),weather,prepared=True)
        model=joblib.load(OUT/f"{task}_{horizon}h.joblib")
        assert np.isclose(model.predict(x)[0],r.prediction,rtol=1e-10,atol=1e-10)
