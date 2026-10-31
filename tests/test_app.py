"""Streamlit interaction checks: airport/horizon changes and custom scenario inference."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.skipif(not (ROOT/"outputs"/"weather_prepared.joblib").exists(),
                            reason="Full training artifacts are required")


@pytest.fixture(scope="module")
def dashboard():
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=60).run()
    assert not app.exception
    return app


@pytest.mark.parametrize("airport",["ATL","DFW"])
@pytest.mark.parametrize("horizon",[3,6,9,12])
def test_every_airport_horizon(dashboard,airport,horizon):
    dashboard.selectbox[0].select(airport)
    dashboard.select_slider[0].set_value(horizon)
    dashboard.run()
    assert not dashboard.exception
    assert len(dashboard.metric)>=4
    assert all(m.value.lower()!="nan" for m in dashboard.metric[:2])


def test_custom_schedule(dashboard):
    dashboard.radio[0].set_value("Custom schedule").run()
    assert not dashboard.exception
    assert dashboard.time_input
    assert all(m.value.lower()!="nan" for m in dashboard.metric[:2])
