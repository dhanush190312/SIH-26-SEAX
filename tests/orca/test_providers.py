import json
import unittest
from dataclasses import replace
from datetime import timedelta
from unittest import mock
from common import NOW, at, isolated_data_dir, make_cfg
from orca.data import EmptyProvider, make_provider, save_cache
from orca.data.providers import _demo_docs, freeze
from orca.data.live import build_live_docs, parse_open_meteo


class DemoProviderTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.p = make_provider(self.cfg)

    def test_demo_is_labelled_everywhere(self):
        obs = self.p.weather(10.765, 79.87, at(6)) + self.p.ocean(11.0, 80.12, at(6))
        self.assertTrue(obs)
        for o in obs:
            self.assertEqual(o.data_mode, "DEMO")
            self.assertIn("synthetic", o.source_label.lower())

    def test_observation_keeps_three_clocks_apart(self):
        o = next(x for x in self.p.weather(10.765, 79.87, at(6)) if x.variable == "wave_height")
        self.assertEqual(o.time_kind, "forecast")
        self.assertIsNone(o.observed_at)
        self.assertTrue(o.forecast_for.startswith("2026-10-01T06:00"))
        self.assertTrue(o.issued_at and o.retrieved_at)
        self.assertEqual(o.freshness_hours, 2.1)
        oc = next(x for x in self.p.ocean(11.0, 80.12, at(6)) if x.variable == "chlorophyll")
        self.assertEqual(oc.time_kind, "observed")
        self.assertIsNotNone(oc.observed_at)
        self.assertIsNone(oc.forecast_for)

    def test_source_priority_data_has_two_sources_for_the_conflict_demo(self):
        fams = {o.source_family for o in self.p.weather(10.765, 79.87, at(10)) if o.variable == "wave_height"}
        self.assertEqual(fams, {"incois", "open_model"})

    def test_far_point_yields_explicit_missing_not_the_nearest_value(self):
        obs = self.p.weather(13.08, 80.30, at(6))  # Chennai: >40 km from any demo point
        self.assertTrue(obs)
        self.assertTrue(all(o.value is None and o.quality == "missing" for o in obs))
        self.assertTrue(all("no observation point" in o.note for o in obs))

    def test_time_outside_window_is_missing(self):
        obs = self.p.weather(10.765, 79.87, at(6, day=5))
        self.assertTrue(all(o.value is None for o in obs))

    def test_pfz_only_valid_on_its_dates(self):
        self.assertTrue(self.p.pfz("2026-10-01").available)
        r = self.p.pfz("2026-10-09")
        self.assertFalse(r.available)
        self.assertEqual(r.zones, ())

    def test_demo_reanchors_to_the_clock(self):
        """DEMO data is regenerated relative to 'now' (so 'tomorrow' always exists); only OFFLINE ages."""
        later = replace(make_cfg(), now=NOW + timedelta(hours=40))
        w = next(o for o in make_provider(later).weather(10.765, 79.87, at(6, day=2)) if o.variable == "wave_height")
        self.assertEqual(w.quality, "available")


class OfflineTests(unittest.TestCase):
    def test_no_cache_means_empty_provider_with_reason(self):
        cfg = replace(make_cfg(), mode="OFFLINE", data_dir=isolated_data_dir())
        p = make_provider(cfg)
        self.assertIsInstance(p, EmptyProvider)
        self.assertFalse(any(p.capabilities().values()))
        self.assertIn("cache", p.errors[0])

    def test_cache_round_trip_is_honestly_labelled(self):
        dd = isolated_data_dir()
        cfg = replace(make_cfg(), mode="OFFLINE", data_dir=dd)
        save_cache(freeze(_demo_docs(replace(cfg, data_dir=dd)), NOW), dd / "cache" / "bundle.json", origin_mode="DEMO", retrieved_at=NOW)
        p = make_provider(cfg)
        self.assertTrue(p.capabilities()["weather"])
        self.assertIn("OFFLINE", p.label())
        self.assertIn("CACHED DEMO DATA", p.label(), "cached synthetic data must not be presented as real cached data")
        self.assertIn("last updated", p.label())
        o = p.weather(10.765, 79.87, at(6))[0]
        self.assertEqual(o.data_mode, "OFFLINE")
        self.assertIn("cached", o.note)

    def test_cache_goes_stale_as_the_clock_advances(self):
        dd = isolated_data_dir()
        cfg = replace(make_cfg(), mode="OFFLINE", data_dir=dd)
        save_cache(freeze(_demo_docs(replace(cfg, data_dir=dd)), NOW), dd / "cache" / "bundle.json", origin_mode="DEMO", retrieved_at=NOW)
        old = replace(cfg, now=NOW + timedelta(hours=30))
        p = make_provider(old)
        wave = next(o for o in p.weather(10.765, 79.87, at(6)) if o.variable == "wave_height")
        self.assertEqual(wave.quality, "stale")
        self.assertGreater(wave.freshness_hours, 12)
        self.assertEqual(p.advisories().quality, "stale")


# A minimal payload in the documented Open-Meteo shape (times in IST because timezone=Asia/Kolkata is requested)
def _om(n_points, hours=6):
    times = [f"2026-10-01T{h:02d}:00" for h in range(hours)]
    marine = [{"latitude": 10.0 + i, "longitude": 80.0, "hourly": {"time": times, "wave_height": [1.0 + i] * hours}} for i in range(n_points)]
    fore = [{"latitude": 10.0 + i, "longitude": 80.0, "hourly": {"time": times, "wind_speed_10m": [12.0] * hours, "precipitation": [0.0] * hours}} for i in range(n_points)]
    return marine, fore


class LiveTests(unittest.TestCase):
    def test_parse_fixture(self):
        m, f = _om(2)
        pts, anchor, hours = parse_open_meteo(m, f, [(10.0, 80.0), (11.0, 80.0)])
        self.assertEqual(hours, 6)
        self.assertEqual(anchor.hour, 0)
        self.assertEqual(pts[1]["series"]["open_model"]["wave_height"][0], 2.0)

    def test_parse_rejects_malformed_payload(self):
        m, f = _om(2)
        del f[0]["hourly"]["wind_speed_10m"]
        with self.assertRaises(ValueError):
            parse_open_meteo(m, f, [(10.0, 80.0), (11.0, 80.0)])
        m, f = _om(1)
        with self.assertRaises(ValueError):
            parse_open_meteo(m, f, [(10.0, 80.0), (11.0, 80.0)])

    def test_live_failure_is_reported_and_never_filled_from_demo(self):
        cfg = replace(make_cfg(), mode="LIVE", data_dir=isolated_data_dir())
        gaz = json.loads((cfg.data_dir / "demo" / "gazetteer.json").read_text(encoding="utf-8"))
        with mock.patch("orca.data.live._get_json", side_effect=OSError("network unreachable")):
            docs, errors = build_live_docs(cfg, (10.765, 79.87), gaz)
        self.assertNotIn("weather", docs)
        self.assertNotIn("pfz", docs)
        self.assertNotIn("advisories", docs)
        self.assertTrue(any("Open-Meteo unavailable" in e for e in errors))
        self.assertTrue(any("PFZ" in e for e in errors))
        with mock.patch("orca.data.live._get_json", side_effect=OSError("network unreachable")):
            p = make_provider(cfg)
        self.assertFalse(p.capabilities()["weather"])
        self.assertFalse(p.capabilities()["pfz"])
        self.assertIn("LIVE", p.label())

    def test_live_success_path_with_fixture(self):
        cfg = replace(make_cfg(), mode="LIVE", data_dir=isolated_data_dir())
        gaz = json.loads((cfg.data_dir / "demo" / "gazetteer.json").read_text(encoding="utf-8"))
        def fake(url, params):
            n = len(str(params["latitude"]).split(","))
            m, f = _om(n, 72)
            return m if "marine" in url else f
        with mock.patch("orca.data.live._get_json", side_effect=fake):
            p = make_provider(cfg, focus=(10.765, 79.87))
        self.assertTrue(p.capabilities()["weather"])
        self.assertFalse(p.capabilities()["pfz"], "no public PFZ API is used")
        obs = p.weather(10.765, 79.87, at(6))
        self.assertTrue(obs and all(o.source_family == "open_model" and o.data_mode == "LIVE" for o in obs if o.value is not None))
        self.assertTrue((cfg.data_dir / "cache" / "bundle.json").is_file(), "a successful LIVE fetch refreshes the offline cache")
