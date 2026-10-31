"""Run with python -m unittest -v test_stage1_pipeline. No network required."""

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import numpy as np
import pandas as pd

import stage1_pipeline as pipeline


def utc(value):
    return pd.Timestamp(value, tz="UTC")


def metar_fixture():
    rows = []
    for hour in (6, 7):
        row = dict.fromkeys(pipeline.METAR_COLUMNS, 1.0)
        row.update(airport="ATL", hour_utc=utc(f"2023-01-01 {hour}:00"),
                   known_at_utc=utc(f"2023-01-01 {hour}:52"), n_obs=1,
                   flight_category="VFR", wind_kt=hour)
        rows.append(row)
    return pd.DataFrame(rows)


def taf_fixture(invalid_expiry=False):
    metadata = pd.DataFrame([
        dict(station="KATL", product_id="original", utc_issued="2023-01-01 06:00Z",
             utc_issue="2023-01-01 06:00Z", utc_expire="2023-01-01 18:00Z", is_amendment=False),
        dict(station="KATL", product_id="amendment", utc_issued="2023-01-01 08:00Z",
             utc_issue="2023-01-01 08:00Z", utc_expire="2023-01-01 18:00Z", is_amendment=True),
    ])
    if invalid_expiry:
        metadata.loc[metadata.product_id.eq("amendment"), "utc_expire"] = "2022-12-02 00:00Z"
    rows = []
    for product, issued, start, end, kind, raw, wind, sky, levels, wx in [
        ("original", "06:00", "06:00", None, "Observation", "08005KT P6SM SCT020", 5, "['SCT']", "[2000]", "[]"),
        ("original", "06:00", "12:00", None, "Forecast", "FM011200 08010KT", 10, "['BKN']", "[1000]", "[]"),
        ("original", "06:00", "08:00", "11:00", "Temporary", "TEMPO 0108/0111 TSRA BKN003", np.nan, "['BKN']", "[300]", "['TSRA']"),
        ("original", "06:00", "08:00", "11:00", "Probability 30", "PROB30 0108/0111 TSRA", np.nan, "[]", "[]", "['TSRA']"),
        ("amendment", "08:00", "08:00", None, "Observation", "08030KT P6SM SCT020", 30, "['SCT']", "[2000]", "[]"),
    ]:
        rows.append(dict(station="KATL", product_id=product, valid=f"2023-01-01 {issued}Z",
                         fx_valid=f"2023-01-01 {start}Z", fx_valid_end=f"2023-01-01 {end}Z" if end else None,
                         ftype=kind, raw=raw, sknt=wind, drct=80, gust=0, visibility=6.01,
                         skyc=sky, skyl=levels, presentwx=wx))
    return pipeline.normalize_taf(pd.DataFrame(rows), metadata)


def event(target="10:00", cutoff="07:00", airport="ATL"):
    return pd.DataFrame([dict(airport=airport, event_utc=utc("2023-01-01 " + target),
                              prediction_utc=utc("2023-01-01 " + cutoff))])


class WeatherTests(unittest.TestCase):
    def test_exact_cutoff_excludes_0752_report(self):
        out = pipeline.attach_metar(event(), metar_fixture())
        self.assertEqual(out.loc[0, "metar_known_at_utc"], utc("2023-01-01 06:52"))
        self.assertEqual(out.loc[0, "metar_wind_kt"], 6)

    def test_stale_and_missing_events_are_retained(self):
        events = pd.concat([event("15:00", "12:00"), event()], ignore_index=True)
        events.loc[1, ["event_utc", "prediction_utc"]] = pd.NaT
        out = pipeline.attach_metar(events, metar_fixture())
        self.assertEqual(len(out), 2)
        self.assertTrue(out.metar_missing.all())

    def test_duplicate_weather_key_rejected(self):
        weather = metar_fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate METAR"):
            pipeline.attach_metar(event(), pd.concat([weather, weather]))

    def test_unknown_weather_not_vfr(self):
        weather = metar_fixture()
        weather.loc[0, ["vis_mi", "ceiling_ft"]] = np.nan
        out = pipeline.attach_metar(event(), weather)
        self.assertTrue(pd.isna(out.loc[0, "metar_flight_category"]))

    def test_legacy_notebook_join_uses_same_cutoff(self):
        flights = pd.DataFrame([dict(flight_id=7, ORIGIN="ATL", sched_dep=pd.Timestamp("2023-01-01 05:00"))])
        out = pipeline.attach_weather(flights, metar_fixture())
        self.assertEqual(out.loc[0, "known_at_utc"], utc("2023-01-01 06:52"))
        self.assertEqual(out.loc[0, "origin_wind_kt_lag3h"], 6)


class TafTests(unittest.TestCase):
    def setUp(self):
        self.segments, self.reports = taf_fixture()

    def test_later_amendment_excluded(self):
        out = pipeline.attach_taf(event(), self.segments, self.reports)
        self.assertEqual(out.loc[0, "taf_product_id"], "original")
        self.assertEqual(out.loc[0, "taf_wind_kt"], 5)
        self.assertFalse(out.loc[0, "taf_thunder"])
        self.assertTrue(out.loc[0, "taf_tempo"])
        self.assertTrue(out.loc[0, "taf_prob30"])
        self.assertTrue(out.loc[0, "taf_conditional_thunder"])
        self.assertEqual(out.loc[0, "taf_conditional_ceiling_min_ft"], 300)

    def test_amendment_at_cutoff_is_eligible(self):
        out = pipeline.attach_taf(event(cutoff="08:00"), self.segments, self.reports)
        self.assertEqual(out.loc[0, "taf_product_id"], "amendment")
        self.assertEqual(out.loc[0, "taf_wind_kt"], 30)

    def test_forecast_selected_for_event_not_cutoff(self):
        out = pipeline.attach_taf(event(target="14:00"), self.segments, self.reports)
        self.assertEqual(out.loc[0, "taf_wind_kt"], 10)
        self.assertFalse(out.loc[0, "taf_tempo"])

    def test_expiry_is_exclusive_and_missing_station_retained(self):
        events = pd.concat([event(target="18:00"), event(airport="DFW")], ignore_index=True)
        out = pipeline.attach_taf(events, self.segments, self.reports)
        self.assertEqual(len(out), 2)
        self.assertTrue(out.taf_missing.all())

    def test_newest_unusable_report_does_not_fall_back(self):
        reports = self.reports.copy()
        reports.loc[reports.product_id.eq("amendment"), "valid_from_utc"] = utc("2023-01-01 12:00")
        out = pipeline.attach_taf(event(cutoff="08:00"), self.segments, reports)
        self.assertTrue(out.loc[0, "taf_missing"])

    def test_segment_arrays_cannot_execute_code(self):
        with self.assertRaises((ValueError, SyntaxError)):
            pipeline.array_value("__import__('os').getcwd()")

    def test_bad_archive_expiry_is_flagged_and_unavailable(self):
        segments, reports = taf_fixture(invalid_expiry=True)
        self.assertEqual(int(reports.validity_invalid.sum()), 1)
        out = pipeline.attach_taf(event(cutoff="08:00"), segments, reports)
        self.assertTrue(out.loc[0, "taf_missing"])


class TimestampTests(unittest.TestCase):
    def test_dst_ambiguity_and_nonexistent_time_are_unknown(self):
        times = pd.Series(pd.to_datetime(["2024-11-03 01:30", "2024-03-10 02:30", "2024-03-10 03:30"]))
        out = pipeline.local_to_utc(times, "America/New_York")
        self.assertTrue(out.iloc[:2].isna().all())
        self.assertEqual(out.iloc[2], utc("2024-03-10 07:30"))

    def test_bts_2400_and_overnight_arrival(self):
        row = dict(FlightDate="2023-01-31", Origin="ATL", Dest="DFW", CRSDepTime=2400,
                   CRSArrTime=200, CRSElapsedTime=180, DepDelay=5, ArrDelay=10,
                   TaxiOut=15, TaxiIn=5, Cancelled=0, Diverted=0)
        out = pipeline.normalize_bts(pd.DataFrame([row]), pipeline.AIRPORT_TZ)
        self.assertEqual(out.loc[0, "sched_dep_utc"], utc("2023-02-01 05:00"))
        self.assertEqual(out.loc[0, "sched_arr_utc"], utc("2023-02-01 08:00"))
        self.assertFalse(out.loc[0, "timestamp_invalid"])
        self.assertEqual(out.loc[0, "wheels_on_utc"], utc("2023-02-01 08:05"))

    def test_inconsistent_schedule_is_flagged(self):
        row = dict(FlightDate="2023-01-31", Origin="ATL", Dest="DFW", CRSDepTime=2400,
                   CRSArrTime=100, CRSElapsedTime=180, DepDelay=5, ArrDelay=10,
                   TaxiOut=15, TaxiIn=5, Cancelled=0, Diverted=0)
        out = pipeline.normalize_bts(pd.DataFrame([row]), pipeline.AIRPORT_TZ)
        self.assertTrue(out.loc[0, "timestamp_invalid"])
        self.assertTrue(pd.isna(out.loc[0, "wheels_on_utc"]))

    def test_full_input_ids_rows_and_labels(self):
        flights = pd.read_csv(pipeline.ROOT / "data/clean/flights_clean.csv", parse_dates=["sched_dep", "sched_arr"])
        out, _ = pipeline.build_events(flights, pipeline.AIRPORT_TZ)
        self.assertEqual(len(out), 76836)
        self.assertEqual(out.flight_id.nunique(), 25299)
        self.assertFalse(out.duplicated(pipeline.KEY).any())
        self.assertTrue(out.loc[out.CANCELLED.eq(1), "severe_delay"].isna().all())
        predictors = pipeline.predictor_columns(out)
        self.assertNotIn("DEP_DELAY", predictors)
        self.assertNotIn("flight_id", predictors)


class CacheTests(unittest.TestCase):
    content = b"station,valid,fx_valid,raw,ftype,product_id\nKATL,2023-01-01,2023-01-01,08005KT,Observation,id\n"

    def test_cache_reuse_and_offline_no_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "taf.csv"
            with patch.object(pipeline.urllib.request, "urlopen", return_value=io.BytesIO(self.content)) as request:
                first = pipeline.fetch_cached(path, "https://example.test/taf", "taf")
            request.assert_called_once()
            with patch.object(pipeline.urllib.request, "urlopen", side_effect=AssertionError("network used")):
                second = pipeline.fetch_cached(path, "https://example.test/taf", "taf", offline=True)
            self.assertEqual(first, second)
            path.write_bytes(self.content + b"\n")
            with self.assertRaisesRegex(ValueError, "Cache mismatch"):
                pipeline.fetch_cached(path, "https://example.test/taf", "taf", offline=True)

    def test_missing_cache_and_interrupted_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "taf.csv"
            with self.assertRaises(FileNotFoundError):
                pipeline.fetch_cached(path, "https://example.test/taf", "taf", offline=True)
            with patch.object(pipeline.urllib.request, "urlopen", side_effect=urllib.error.URLError("offline")), patch.object(pipeline.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "Download failed"):
                    pipeline.fetch_cached(path, "https://example.test/taf", "taf")
            self.assertFalse(path.exists())
            self.assertFalse(path.with_name("taf.csv.part").exists())

    def test_compressed_outputs_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.csv.gz"
            frame = pd.DataFrame({"value": [1, 2]})
            pipeline.write_csv(path, frame)
            first = path.read_bytes()
            pipeline.write_csv(path, frame)
            self.assertEqual(first, path.read_bytes())
            pd.testing.assert_frame_equal(pd.read_csv(path), frame)

    def test_bts_download_resumes_short_response(self):
        first = io.BytesIO(b"12345")
        first.status = 200
        first.headers = {"Content-Length": "10", "ETag": '"version1"'}
        second = io.BytesIO(b"67890")
        second.status = 206
        second.headers = {"Content-Length": "5", "Content-Range": "bytes 5-9/10", "ETag": '"version1"'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bts.zip"
            with patch.object(pipeline.urllib.request, "urlopen", side_effect=[first, second]) as request, patch.object(pipeline, "validate_download"), patch.object(pipeline.time, "sleep"):
                pipeline.fetch_cached(path, "https://example.test/bts", "bts")
            self.assertEqual(path.read_bytes(), b"1234567890")
            resumed = request.call_args_list[1].args[0]
            self.assertEqual(resumed.get_header("Range"), "bytes=5-")
            self.assertEqual(resumed.get_header("If-range"), '"version1"')


class BtsContextTests(unittest.TestCase):
    def test_window_uses_only_completed_operations_and_empty_rate_is_missing(self):
        rows = []
        for airport, other in [("ATL", "DFW"), ("DFW", "ATL")]:
            for clock, delay in [("06:00", 60), ("08:00", 60), ("09:00", 0), ("09:01", 60)]:
                stamp = utc("2023-01-01 " + clock)
                rows.append(dict(Origin=airport, Dest=other, sched_dep_utc=stamp, sched_arr_utc=stamp,
                                 wheels_off_utc=stamp, wheels_on_utc=stamp, DepDelay=delay))
        bts = pd.DataFrame(rows)
        events = pd.concat([event("12:00", "09:00"), event("06:00", "03:00")], ignore_index=True)
        out, context = pipeline.attach_bts(events, bts)
        self.assertEqual(out.loc[0, "bts_departures_previous_3h"], 2)
        self.assertEqual(out.loc[0, "bts_arrivals_previous_3h"], 2)
        self.assertEqual(out.loc[0, "bts_severe_departure_rate_previous_3h"], 0.5)
        self.assertEqual(out.loc[0, "bts_observed_through_utc"], utc("2023-01-01 09:00"))
        self.assertEqual(out.loc[1, "bts_departures_previous_3h"], 0)
        self.assertTrue(pd.isna(out.loc[1, "bts_severe_departure_rate_previous_3h"]))
        self.assertFalse(context.duplicated(["airport", "hour_utc"]).any())

    def test_future_operations_cannot_change_past_features(self):
        stamp = utc("2023-01-01 08:00")
        rows = [dict(Origin=a, Dest=b, sched_dep_utc=stamp, sched_arr_utc=stamp,
                     wheels_off_utc=stamp, wheels_on_utc=stamp, DepDelay=0)
                for a, b in [("ATL", "DFW"), ("DFW", "ATL")]]
        bts = pd.DataFrame(rows)
        events = event("12:00", "09:00")
        before, _ = pipeline.attach_bts(events, bts)
        future = bts.copy()
        future[["wheels_off_utc", "wheels_on_utc"]] += pd.Timedelta(hours=2)
        future.DepDelay = 120
        after, _ = pipeline.attach_bts(events, pd.concat([bts, future], ignore_index=True))
        columns = ["bts_departures_previous_3h", "bts_arrivals_previous_3h",
                   "bts_severe_departure_rate_previous_3h", "bts_observed_through_utc"]
        pd.testing.assert_frame_equal(before[columns], after[columns])


if __name__ == "__main__":
    unittest.main()
