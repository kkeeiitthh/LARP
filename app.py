"""Forecast scenarios and transparent held-out evaluation for ATL/DFW."""
import json
from datetime import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from data_loading import attach_schedule_features
from features import TIMEZONES, asof_weather, calendar_features
from modeling import classification_metrics
from train import load_inputs

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
st.set_page_config(page_title="AeroCast | Weather & airport flow", page_icon="✈", layout="wide")


@st.cache_resource
def artifacts(version):
    predictions = pd.read_csv(OUT/"predictions.csv", low_memory=False, parse_dates=["event_time_utc","issue_time_utc","observation_time_utc","available_time_utc"])
    metrics = json.loads((OUT/"evaluation_results.json").read_text(encoding="utf-8"))
    flights, weather = load_inputs()
    models = {(task,h): joblib.load(OUT/f"{task}_{h}h.joblib") for task in ("delay","throughput") for h in (3,6,9,12)}
    return predictions, metrics, flights, weather, models


def metric_table(metrics):
    rows = []
    for h, block in metrics["horizons"].items():
        for task, report in block.items():
            for airport in ("Overall", "ATL", "DFW"):
                item = report["test"] if airport=="Overall" else report["test"]["by_airport"][airport]
                row = {"Horizon": int(h), "Airport": airport, "Task": task, "Model": report["selected_model"]}
                row.update(item["model"])
                row["baseline_rmse"] = item["baseline"].get("rmse")
                row["baseline_pr_auc"] = item["baseline"].get("pr_auc")
                rows.append(row)
    return pd.DataFrame(rows)


def forecast_row(airport, event, horizon, flights, weather, metadata=None):
    row = pd.DataFrame({"airport":[airport], "event_time_utc":[event],
                        "issue_time_utc":[event-pd.Timedelta(hours=horizon)], "horizon":[horizon]})
    if metadata:
        for key, value in metadata.items():
            row[key] = value
    return asof_weather(attach_schedule_features(calendar_features(row), flights), weather, prepared=True)


def performance_tab(pred, metrics, airport, horizon):
    st.subheader("Measured performance on later, unseen dates")
    st.caption("Train: Jan 2023–Oct 2024 · validation: Oct 2024–Jan 2025 · calibration: Jan–Mar 2025 · test: Mar–Jul 2025. Two-day gaps protect issue-time availability.")
    block = metrics["horizons"][str(horizon)]
    delay = pred[(pred.task=="delay") & (pred.airport==airport) & (pred.forecast_horizon_hours==horizon) & (pred.partition=="test")]
    threshold = st.slider("Evaluate the alert trade-off at probability threshold", .05, .80, round(block["delay"]["decision_threshold"],2), .01, key="evaluation_threshold")
    score = classification_metrics(delay.actual, delay.prediction, threshold)
    majority_accuracy = max(delay.actual.mean(),1-delay.actual.mean())
    cols = st.columns(5)
    for col, label, value in zip(cols, ["Accuracy","Precision","Recall","ROC-AUC","PR-AUC"],
            [score["accuracy"],score["precision"],score["recall"],score["roc_auc"],score["pr_auc"]]):
        col.metric(label, f"{value:.3f}")
    st.caption(f"Predicting 'no severe delay' for every flight would achieve {majority_accuracy:.1%} accuracy here. Compare recall and PR-AUC too. Test prevalence: {delay.actual.mean():.1%}.")
    st.write(f"Selected delay model: **{block['delay']['selected_model']}**. Throughput model: **{block['throughput']['selected_model']}**.")
    a,b = st.columns(2)
    with a:
        precision, recall, _ = precision_recall_curve(delay.actual, delay.prediction)
        fig=px.line(x=recall,y=precision,labels={"x":"Recall","y":"Precision"},title="Precision–recall trade-off")
        fig.add_hline(y=float(delay.actual.mean()),line_dash="dash",annotation_text="Prevalence baseline")
        st.plotly_chart(fig, width="stretch")
    with b:
        true, predicted = calibration_curve(delay.actual, delay.prediction,n_bins=8,strategy="quantile")
        fig=px.line(x=predicted,y=true,markers=True,labels={"x":"Predicted probability","y":"Observed severe-delay rate"},title="Probability reliability")
        fig.add_shape(type="line",x0=0,y0=0,x1=1,y1=1,line={"dash":"dash","color":"gray"})
        st.plotly_chart(fig,width="stretch")
    st.write("Confusion matrix: rows = actual, columns = predicted; order = non-severe, severe.")
    st.dataframe(pd.DataFrame(score["confusion_matrix"], index=["Actual non-severe","Actual severe"], columns=["Predicted non-severe","Predicted severe"]),width="stretch")
    tp=pred[(pred.task=="throughput") & (pred.airport==airport) & (pred.forecast_horizon_hours==horizon) & (pred.partition=="test")]
    daily=tp.set_index("event_time_utc")[["actual","prediction","baseline_prediction"]].resample("D").sum().reset_index()
    st.plotly_chart(px.line(daily,x="event_time_utc",y=["actual","prediction","baseline_prediction"],title="Test-set daily sampled arrivals: actual, model and historical baseline"),width="stretch")
    st.subheader("All airports and horizons")
    table=metric_table(metrics)
    st.dataframe(table.drop(columns=["confusion_matrix"],errors="ignore"),width="stretch",hide_index=True)
    st.download_button("Download complete evaluation JSON",json.dumps(metrics,indent=2),"evaluation_results.json","application/json")
    st.subheader("What the model uses")
    importance=pd.DataFrame(block["delay"]["feature_importance"]).head(10).sort_values("importance")
    st.plotly_chart(px.bar(importance,x="importance",y="feature",orientation="h",title="Validation permutation importance: change in PR-AUC"),width="stretch")
    st.caption("These are model associations. Negative importance means shuffling helped on this validation sample.")
    example=delay.copy()
    example["predicted_class"]=(example.prediction>=threshold).astype(int)
    correct=example[example.predicted_class==example.actual]
    wrong=example[example.predicted_class!=example.actual]
    borderline=example.iloc[(example.prediction-threshold).abs().argsort()[:1]]
    st.subheader("Cases to inspect")
    for label, rows in [("Correct",correct.head(1)),("Incorrect",wrong.head(1)),("Borderline",borderline)]:
        if not rows.empty:
            r=rows.iloc[0]
            st.write(f"**{label}:** {r.event_time_utc} · {r.carrier} to {r.destination} · probability {r.prediction:.1%} · actual severe delay: {bool(r.actual)}")
        else:
            st.write(f"{label}: no example at this threshold.")


def main():
    st.markdown("""<style>
    .block-container {max-width:1400px;padding-top:2rem}
    [data-testid="stMetric"] {border:1px solid #243650;border-radius:12px;padding:16px}
    div[data-testid="stSidebarContent"] {padding-top:1rem}
    </style>""",unsafe_allow_html=True)
    st.title("AeroCast")
    st.write("Weather-aware flight risk and airport arrival flow")
    st.caption("Autonomous SG 2026 · Airbus Challenge #5 · historical prototype")
    path=OUT/"evaluation_results.json"
    if not path.exists():
        st.error("Train the models first: python train.py")
        st.stop()
    pred,metrics,flights,weather,models=artifacts(path.stat().st_mtime_ns)
    if metrics.get("version")!="2026-10-31-v2":
        st.error("Artifacts need updating. Run python train.py.")
        st.stop()
    st.sidebar.header("Forecast scenario")
    airport=st.sidebar.selectbox("Airport",["ATL","DFW"])
    horizon=st.sidebar.select_slider("Hours ahead",options=[3,6,9,12],value=3)
    tz=TIMEZONES[airport]
    delay_rows=pred[(pred.task=="delay") & (pred.airport==airport) & (pred.forecast_horizon_hours==horizon) & (pred.partition=="test")].copy()
    local=delay_rows.event_time_utc.dt.tz_convert(tz)
    default_date=(pd.Timestamp(local.dt.date.max())-pd.Timedelta(days=2)).date()
    date=st.sidebar.date_input("Scheduled departure date (airport local time)",value=default_date,min_value=local.dt.date.min(),max_value=local.dt.date.max())
    mode=st.sidebar.radio("Flight scenario",["Recorded flight","Custom schedule"])
    recorded=None
    if mode=="Recorded flight":
        options=delay_rows[local.dt.date.eq(date)].sort_values("event_time_utc").reset_index(drop=True)
        if options.empty:
            st.warning("No sampled flights on this date. Choose another date or enter a custom schedule.")
            st.stop()
        labels=[f"{r.event_time_utc.tz_convert(tz).strftime('%H:%M')} · {r.carrier} → {r.destination} · row {i+1}" for i,r in options.iterrows()]
        selected=st.sidebar.selectbox("Scheduled flight",range(len(options)),format_func=lambda i:labels[i])
        recorded=options.iloc[selected]
        event=recorded.event_time_utc
        metadata={c:recorded[c] for c in ("carrier","destination","planned_elapsed_minutes","distance_miles")}
    else:
        clock=st.sidebar.time_input("Scheduled departure (local)",time(12))
        event=pd.Timestamp.combine(date,clock).tz_localize(tz,ambiguous="NaT",nonexistent="NaT")
        if pd.isna(event):
            st.warning("This local time is ambiguous or nonexistent due to daylight saving time.")
            st.stop()
        event=event.tz_convert("UTC")
        carrier=st.sidebar.selectbox("Carrier",sorted(flights.loc[flights.ORIGIN.eq(airport),"OP_UNIQUE_CARRIER"].unique()))
        routes=flights[flights.ORIGIN.eq(airport)&flights.OP_UNIQUE_CARRIER.eq(carrier)]
        destination=st.sidebar.selectbox("Destination",sorted(routes.DEST.unique()))
        route=routes[routes.DEST.eq(destination)]
        duration=st.sidebar.number_input("Scheduled elapsed minutes",min_value=1.,value=float(route.CRS_ELAPSED_TIME.median()))
        distance=st.sidebar.number_input("Distance (miles)",min_value=1.,value=float(route.DISTANCE.median()))
        metadata={"carrier":carrier,"destination":destination,"planned_elapsed_minutes":duration,"distance_miles":distance}
    issue=event-pd.Timedelta(hours=horizon)
    st.sidebar.caption(f"Departure: {event.tz_convert(tz):%d %b %H:%M %Z}\n\nIssue: {issue.tz_convert(tz):%d %b %H:%M %Z}")
    forecast,performance,quality=st.tabs(["Forecast","Model evaluation","Data quality & limits"])
    with forecast:
        event_hour=event.floor("h")
        risk_input=forecast_row(airport,event,horizon,flights,weather,metadata)
        flow_input=forecast_row(airport,event_hour,horizon,flights,weather)
        delay_model=models["delay",horizon]
        p=float(delay_model.predict(risk_input)[0])
        flow_model=models["throughput",horizon]
        arrivals=float(flow_model.predict(flow_input)[0])
        baseline_rows=pred[(pred.task=="throughput")&(pred.airport==airport)&(pred.forecast_horizon_hours==horizon)&pred.event_time_utc.eq(event_hour)]
        baseline=float(baseline_rows.baseline_prediction.iloc[0]) if len(baseline_rows) else np.nan
        scheduled_demand=int(flow_input.scheduled_arrivals_offset0.iloc[0])
        reduction=(1-arrivals/baseline)*100 if baseline>0 and scheduled_demand>0 else np.nan
        risk="Low" if p<.1 else "Moderate" if p<.3 else "Severe"
        a,b,c,d=st.columns(4)
        a.metric("Predicted hourly arrival throughput",f"{arrivals:.2f}")
        b.metric("Arrival-delay risk >45 min",f"{p:.1%}")
        c.metric("Risk category",risk)
        d.metric("Relative to historical baseline",f"{-reduction:+.1f}%" if np.isfinite(reduction) else "No scheduled demand" if scheduled_demand==0 else "Unavailable")
        st.caption(f"Scheduled arrivals in the sample for this hour: {scheduled_demand}. Low planned demand can explain low throughput without a capacity restriction.")
        st.caption("Throughput counts the supplied flight sample. A negative change versus baseline is a proxy reduction, not a declared runway-capacity reduction.")
        st.caption(f"Flight prediction: {event.tz_convert(tz):%d %b %H:%M %Z}. Throughput interval: {event_hour.tz_convert(tz):%H:%M}–{(event_hour+pd.Timedelta(hours=1)).tz_convert(tz):%H:%M %Z}.")
        lower=max(0,arrivals+flow_model.interval[0]); upper=max(0,arrivals+flow_model.interval[1])
        coverage=metrics["horizons"][str(horizon)]["throughput"]["test"].get("empirical_80pct_interval_coverage",float("nan"))
        st.caption(f"Nominal 80% throughput interval: {lower:.2f}–{upper:.2f} sampled arrivals; estimated on the calibration period. Measured pooled test coverage: {coverage:.1%}.")
        if recorded is not None:
            st.caption(f"Recorded test outcome for this exact flight: {'severe arrival delay' if recorded.actual else 'arrival delay ≤45 minutes'}. Outcomes are shown for review and excluded from prediction inputs.")
        w=risk_input.iloc[0]
        st.subheader("Weather known at issue time")
        if pd.isna(w.observation_time_utc):
            st.warning("No weather observation satisfies the availability and freshness limits. The model uses learned missing-value handling; confidence is reduced.")
        else:
            st.caption(f"Observation: {w.observation_time_utc:%d %b %H:%M UTC} · age {w.weather_age_minutes:.0f} minutes · assumed reporting latency: 20 minutes.")
            fields={"Wind direction (°)":w.wind_direction_deg,"Wind (kt)":w.wind_speed_kt,"Gust (kt)":w.wind_gust_kt,
                    "Crosswind magnitude (kt)":w.absolute_crosswind_kt,"Headwind + / tailwind − (kt)":w.headwind_kt,
                    "Visibility (m)":w.visibility_m,"Ceiling (m)":w.ceiling_m,"1h rain (mm)":w.precipitation_mm,
                    "Temperature (°C)":w.temperature_c,"Dew point (°C)":w.dew_point_c}
            st.dataframe(pd.DataFrame({"Measurement":fields.keys(),"Value":[f"{v:.1f}" if pd.notna(v) else "Not reported" for v in fields.values()]}),hide_index=True,width="stretch")
        st.subheader("Controller review prompt")
        concerns=[]
        if pd.notna(w.absolute_crosswind_kt) and w.absolute_crosswind_kt>=15: concerns.append("elevated crosswind")
        if pd.notna(w.visibility_m) and w.visibility_m<5000: concerns.append("reduced visibility")
        if pd.notna(w.precipitation_mm) and w.precipitation_mm>0: concerns.append("reported precipitation")
        if scheduled_demand==0:
            text="No arrivals are scheduled in this sample for the selected hour. Low predicted throughput alone does not indicate reduced runway capacity."
            if p>=delay_model.review_threshold:
                text+=" The selected flight's delay probability warrants separate review."
        elif np.isfinite(reduction) and reduction>=20:
            text=f"Predicted sampled throughput is {reduction:.0f}% below its historical baseline. Review the arrival plan against current operational forecasts."
        elif p>=delay_model.review_threshold:
            text="Flight-delay probability exceeds the review threshold. Review sequencing and contingency plans against authoritative weather and traffic forecasts."
        else:
            text="Continue monitoring weather and traffic conditions. This model does not currently trigger the review threshold."
        if concerns: text+=" Available observations show "+", ".join(concerns)+"."
        st.info(text)
        st.caption("Recommendations support human review. They are not official operational or safety decisions.")
        left,right=st.columns(2)
        hist=weather[(weather.airport==airport)&weather.available_time_utc.le(issue)&weather.observation_time_utc.ge(issue-pd.Timedelta(days=3))]
        with left:
            st.plotly_chart(px.line(hist,x="observation_time_utc",y=["wind_speed_kt","absolute_crosswind_kt"],title="Wind history available at issue time"),width="stretch")
        with right:
            # Conservative 48h maturation protects the chart from exposing unfinished flights.
            history=delay_rows[delay_rows.event_time_utc.le(issue-pd.Timedelta(days=2))]
            daily=history.set_index("event_time_utc").actual.resample("D").mean().reset_index()
            st.plotly_chart(px.line(daily,x="event_time_utc",y="actual",title="Matured historical severe-delay rate"),width="stretch")
    with performance:
        performance_tab(pred,metrics,airport,horizon)
    with quality:
        audit=json.loads((OUT/"data_audit.json").read_text(encoding="utf-8"))
        st.subheader("Source coverage")
        st.write(f"{audit['flight_rows']:,} flight rows · {len(audit['flight_columns'])} columns · {audit['date_min']} through {audit['date_max']}.")
        st.write(f"Unresolved scheduled arrivals: {audit['unresolved_scheduled_arrivals']}. Duplicate source rows: {audit['source_duplicate_rows']}.")
        missing=pd.DataFrame({"feature":audit["parsed_weather_missing_fraction"].keys(),"missing_fraction":audit["parsed_weather_missing_fraction"].values()})
        st.dataframe(missing.sort_values("missing_fraction",ascending=False),hide_index=True,width="stretch")
        st.warning("Official AAR was not supplied. Sparse sample counts cannot establish airport-wide capacity.")
        st.markdown("- NOAA inputs are historical observations with assumed reporting latency. Genuine time-versioned weather forecasts are needed to assess severe-weather forecasting skill.\n- Schedule-demand inputs assume the archived schedules were known at issue time. Historical revisions are unavailable.\n- Missing weather expires after three hours. Missing values and missing indicators are handled using training data only.\n- Runway headings simplify changing runway configurations.\n- Seasonal shifts and the small flight sample limit performance. Accuracy alone rewards predicting the majority class.\n- Calibration and prediction intervals can drift and require monitoring.")
        st.caption("NOAA encoding reference: https://www.ncei.noaa.gov/pub/data/noaa/isd-format-document.pdf")


if __name__=="__main__":
    main()
