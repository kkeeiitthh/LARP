"""Post-training audit and human-readable evidence report, generated from real artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT/"outputs"


def write_report():
    results=json.loads((OUT/"evaluation_results.json").read_text(encoding="utf-8"))
    audit=json.loads((OUT/"data_audit.json").read_text(encoding="utf-8"))
    pred=pd.read_csv(OUT/"predictions.csv",low_memory=False,
                     parse_dates=["event_time_utc","issue_time_utc","observation_time_utc","available_time_utc"])
    matched=pred.available_time_utc.notna()
    if not (pred.loc[matched,"available_time_utc"] <= pred.loc[matched,"issue_time_utc"]).all():
        raise AssertionError("Future/unavailable weather found in saved predictions")
    cases=[]
    lines=["# Model and data audit report", "",
           "Generated from held-out predictions; model and threshold choices use earlier periods only.", "",
           "## Measured test performance", "",
           "| Horizon | Delay model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Throughput RMSE | Baseline RMSE |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    summary=[]
    for h, block in results["horizons"].items():
        d=block["delay"]["test"]["model"]; t=block["throughput"]["test"]
        lines.append(f"| {h}h | {block['delay']['selected_model']} | {d['accuracy']:.3f} | {d['precision']:.3f} | {d['recall']:.3f} | {d['f1']:.3f} | {d['roc_auc']:.3f} | {d['pr_auc']:.3f} | {d['brier_score']:.3f} | {t['model']['rmse']:.3f} | {t['baseline']['rmse']:.3f} |")
        flow=pred[(pred.task=="throughput")&(pred.partition=="test")&(pred.forecast_horizon_hours==int(h))]
        coverage=float(((flow.actual>=flow.interval_low)&(flow.actual<=flow.interval_high)).mean())
        block["throughput"]["test"]["empirical_80pct_interval_coverage"]=coverage
        summary.append({"horizon":int(h),"delay":d,"throughput":t,"interval_coverage":coverage})
        flights=pred[(pred.task=="delay")&(pred.partition=="test")&(pred.forecast_horizon_hours==int(h))].copy()
        flights["predicted_class"]=(flights.prediction>=block["delay"]["decision_threshold"]).astype(int)
        correct=flights[(flights.actual==1)&flights.predicted_class.eq(1)]
        if correct.empty: correct=flights[flights.actual.eq(flights.predicted_class)]
        wrong=flights[flights.actual.ne(flights.predicted_class)]
        border=flights.iloc[(flights.prediction-block["delay"]["decision_threshold"]).abs().argsort()[:1]]
        for name,frame in (("correct",correct),("incorrect",wrong),("borderline",border)):
            if len(frame):
                row=frame.iloc[0]
                cases.append({"horizon":int(h),"case":name,"airport":row.airport,
                    "event_time_utc":str(row.event_time_utc),"carrier":row.carrier,"destination":row.destination,
                    "actual_severe":int(row.actual),"probability":float(row.prediction),
                    "threshold":float(block["delay"]["decision_threshold"])})
    lines += ["", "## Airport results", "",
              "| Horizon | Airport | Delay ROC-AUC | Delay PR-AUC | Throughput RMSE | Baseline RMSE |",
              "|---|---|---:|---:|---:|---:|"]
    for h,block in results["horizons"].items():
        for airport in ("ATL","DFW"):
            d=block["delay"]["test"]["by_airport"][airport]["model"]
            t=block["throughput"]["test"]["by_airport"][airport]
            lines.append(f"| {h}h | {airport} | {d['roc_auc']:.3f} | {d['pr_auc']:.3f} | {t['model']['rmse']:.3f} | {t['baseline']['rmse']:.3f} |")
    lines += ["", "## Interpretation", "",
        "Accuracy is paired with precision, recall and PR-AUC. The majority baseline is competitive on accuracy because severe delays are uncommon. The review policy trades recall for fewer false alarms. The dashboard exposes this trade-off interactively.",
        "ROC-AUC/PR-AUC measure ranking, while Brier score measures probability error. Calibration is fitted on the first half of a later calibration block and accepted only if it improves Brier score on its second half. Earlier validation/calibration predictions use the uncalibrated model to avoid applying a future-fitted transform to earlier forecasts.",
        "Throughput improvement includes planned demand. It does not establish a causal weather effect or weather-forecast skill. The CRS schedule snapshot is assumed available at issue time; schedule revisions were not supplied.",
        "The 80% empirical residual intervals are measured on a separate calibration period. Test coverage is recorded in evaluation_results.json; nominal coverage is not guaranteed during drift.",
        "", "## Data corrections and cleaning", "",
        f"- Flight source: {audit['flight_rows']:,} rows and {len(audit['flight_columns'])} columns; {audit['source_duplicate_rows']} duplicate rows.",
        f"- Date coverage: {audit['date_min']} to {audit['date_max']}; unresolved scheduled arrivals after clock validation: {audit['unresolved_scheduled_arrivals']}.",
        "- CIG ceiling and VIS visibility now use their own measurement QC fields. AJ1 depth is in centimetres and uses the snow-depth QC position.",
        "- Only one-hour precipitation amounts are combined. Six-/24-hour accumulation depths are excluded from the one-hour feature.",
        "- Missing observations and weather indicators stay missing. Training-only imputers add explicit missingness indicators.",
        "- Individual weather values expire after three hours; latest snapshots and 3h/6h lags are matched by airport/time, with an assumed 20-minute publication latency.",
        "- Actual arrivals are reconstructed from a timezone-validated scheduled arrival plus ARR_DELAY, checked against ARR_TIME. This handles midnight crossings and delays exceeding 24 hours.",
        "- Scheduled arrivals/departures include cancelled records in planned demand, because future cancellation outcomes cannot be used as schedule inputs.",
        "", "## Limits and remaining data needs", "",
        "- Official AAR is unavailable. Sampled counts cannot identify true airport-wide capacity or official capacity reductions.",
        "- NOAA observations are persistence inputs. Time-versioned TAF/NWP forecasts are needed for a genuine weather-forecast experiment.",
        "- Severe weather cases and the flight sample are limited. More models cannot recover unobserved traffic, network disruption and runway configuration.",
        "- Before/after prototype figures use corrected timestamp/target construction and slightly different test cohorts; compare current candidates/baselines on the common current split for fair selection.",
        "", "## Inspectable examples", ""]
    for case in cases:
        lines.append(f"- {case['horizon']}h {case['case']}: {case['airport']} {case['event_time_utc']}, p={case['probability']:.3f}, threshold={case['threshold']:.3f}, actual={case['actual_severe']}.")
    (OUT/"case_studies.json").write_text(json.dumps(cases,indent=2),encoding="utf-8")
    (OUT/"evaluation_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    (ROOT/"MODEL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    write_report()
