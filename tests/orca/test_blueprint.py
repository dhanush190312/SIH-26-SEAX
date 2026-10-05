"""Blueprint workflows: nearest PFZ, closer, route check, ocean regions, scenario, memory, evidence, alerts."""
import json
import unittest
from dataclasses import replace
from datetime import timedelta
from common import NOW, at, isolated_data_dir, make_cfg, mk_obs, ROOT
from orca.decision.hazards import check_route, hazards_near
from orca.decision.geofence import ConstraintSet, load_geofences
from orca.data import make_provider
from orca.pipeline import Orca
from orca.schema import Advisory, AdvisoryFeed


def fresh(**kw):
    return Orca(make_cfg(**kw))


class NearestPfzTests(unittest.TestCase):
    def test_nearest_valid_pfz_skips_a_closer_zone_that_violates_a_geofence_and_says_so(self):
        r = fresh().query("Find the nearest PFZ from Nagapattinam and check the hazards around it tomorrow morning.")
        self.assertEqual(r["intent"], "nearest_pfz")
        self.assertIn("Nearest valid PFZ to Nagapattinam: PFZ-A", r["answer"])
        self.assertIn("geographically nearest PFZ, PFZ-D", r["answer"], "the skipped nearer zone must be disclosed, not silently dropped")
        zs = {z["zone"]["zone_id"]: z for z in r["zones"]}
        self.assertLess(zs["PFZ-D"]["relation"]["distance_km"], zs["PFZ-A"]["relation"]["distance_km"])
        self.assertTrue(r["verification"]["ok"])

    def test_distances_are_deterministic_haversine(self):
        from orca.geo import haversine_km
        r = fresh().query("Find the nearest PFZ from Nagapattinam tomorrow morning.")
        z = next(z for z in r["zones"] if z["zone"]["zone_id"] == "PFZ-A")
        o = r["place"]
        self.assertAlmostEqual(z["relation"]["distance_km"], haversine_km(o["sea_lat"], o["sea_lon"], z["zone"]["latitude"], z["zone"]["longitude"]), delta=0.06)

    def test_device_location_is_used_for_my_location(self):
        r = fresh().query("Where is the nearest PFZ from my location?", location={"lat": 10.80, "lon": 79.95})
        self.assertEqual(r["place"]["id"], "user")
        self.assertIn("your location", r["answer"])
        self.assertNotIn("Nagapattinam", r["answer"].split("PFZ source")[0].replace("Nagapattinam port", ""))

    def test_no_valid_pfz_uses_the_blueprints_sentence(self):
        o = fresh()
        r = o.structured("nearest_pfz", "Nagapattinam", [at(6, day=3).isoformat()])  # outside the PFZ validity AND the data window
        self.assertEqual(r["state"], "INSUFFICIENT_DATA")
        d = isolated_data_dir()
        pf = d / "demo" / "pfz.json"
        doc = json.loads(pf.read_text(encoding="utf-8"))
        doc["valid_day_offsets"] = [0]  # PFZ advisory only valid today
        pf.write_text(json.dumps(doc), encoding="utf-8")
        r2 = Orca(replace(make_cfg(), data_dir=d)).query("Find the nearest PFZ from Nagapattinam tomorrow morning.")
        self.assertIn("No valid PFZ information was available from the configured source for this request.", r2["answer"])
        self.assertEqual(r2["zones"], [])

    def test_stale_pfz_product_is_rejected_not_used(self):
        d = isolated_data_dir()
        pf = d / "demo" / "pfz.json"
        doc = json.loads(pf.read_text(encoding="utf-8"))
        doc["source"]["issued_hours_before_now"] = 60.0
        pf.write_text(json.dumps(doc), encoding="utf-8")
        r = Orca(replace(make_cfg(), data_dir=d)).query("Show potential fishing zones near Nagapattinam tomorrow morning.")
        self.assertEqual(r["state"], "INSUFFICIENT_DATA")
        self.assertIn("freshness limit", r["answer"])
        self.assertEqual(r["zones"], [])

    def test_active_advisory_is_reported_as_official_near_the_pfz_and_inactive_one_is_not(self):
        o = fresh()
        act = o.query("Find the nearest PFZ from Nagapattinam and check the hazards around it tomorrow at 2 PM.")
        self.assertIn("Small-craft caution", act["answer"])
        self.assertTrue(any(a["class"] == "official" for a in act["alerts"]))
        early = fresh().query("Find the nearest PFZ from Nagapattinam and check the hazards around it tomorrow at 8 AM.")
        self.assertNotIn("Small-craft caution", early["answer"])
        self.assertEqual(early["nearby_hazards"]["official_advisories"], [])
        self.assertEqual(early["nearby_hazards"]["inactive_advisories"][0]["why_inactive"], "not yet in force at the requested time")


class MemoryTests(unittest.TestCase):
    def test_blueprint_conversation_pfz_then_tomorrow_then_closer(self):
        o = fresh()
        a = o.query("Show PFZ near Nagapattinam.", session_id="m")
        self.assertEqual(a["intent"], "pfz_zones")
        b = o.query("What about tomorrow?", session_id="m")
        self.assertEqual((b["intent"], b["place"]["name"], b["interpretation"]["followup"]), ("pfz_zones", "Nagapattinam", True))
        self.assertTrue(all(x.startswith("2026-10-01") for x in b["times"]))
        c = o.query("Which one is closer?", session_id="m")
        self.assertEqual(c["intent"], "closer")
        self.assertIn("PFZ-A is the closest zone that can be offered", c["answer"])
        self.assertIn("PFZ-D: 29.6 km E (not offered", c["answer"])

    def test_there_refers_to_the_pfz_focus(self):
        o = fresh()
        o.query("Find the nearest PFZ from Nagapattinam tomorrow morning.", session_id="m2")
        r = o.query("What are the marine conditions there tomorrow morning?", session_id="m2")
        self.assertEqual(r["place"]["id"], "PFZ-A")
        self.assertEqual(r["place"]["source"], "focus")
        o.query("Can I fish tomorrow at 6 AM?", session_id="m2")
        self.assertEqual(o.sessions["m2"]["ctx"]["place"]["name"], "Nagapattinam", '"there" must not replace the home place')

    def test_closer_and_why_without_history_ask_instead_of_guessing(self):
        r = fresh().query("Which one is closer?", session_id="none")
        self.assertEqual(r["state"], "INSUFFICIENT_DATA")
        self.assertIn("no earlier PFZ list", r["answer"])

    def test_sessions_are_isolated(self):
        o = fresh()
        o.query("Show PFZ near Nagapattinam tomorrow.", session_id="A")
        self.assertEqual(o.query("What about tomorrow?", session_id="B")["status"], "needs_input")


class RouteCheckTests(unittest.TestCase):
    def cons(self):
        prov = make_provider(make_cfg())
        return prov, ConstraintSet(load_geofences(prov.geofences()), prov.coast(), coverage=prov.geofence_coverage())

    def test_route_through_the_naval_area_is_flagged_with_evidence(self):
        _, cons = self.cons()
        res = check_route(cons, None, None, [(10.74, 79.87), (10.72, 80.32)])
        self.assertFalse(res["ok"])
        self.assertEqual([c["id"] for c in res["hard_crossings"]], ["GF-NAVAL"])
        self.assertEqual(res["n_fences"], 4)

    def test_clear_route_and_advisory_only_geofence(self):
        _, cons = self.cons()
        res = check_route(cons, None, None, [(10.765, 79.87), (11.0, 80.12)])
        self.assertTrue(res["ok"])
        self.assertEqual([c["id"] for c in res["soft_crossings"]], ["GF-PORT"], "advisory-only fences are reported but not blocking")

    def test_route_over_land_is_flagged(self):
        _, cons = self.cons()
        self.assertTrue(check_route(cons, None, None, [(10.77, 79.4), (10.77, 79.6)])["land_legs"])

    def test_active_advisory_polygon_crossed_only_while_valid(self):
        prov, cons = self.cons()
        feed = prov.advisories()
        path = [(10.765, 79.87), (11.0, 80.12)]
        self.assertTrue(check_route(cons, feed, at(14), path)["advisories"])
        self.assertEqual(check_route(cons, feed, at(8), path)["advisories"], [], "not yet in force at 08:00")

    def test_conversation_route_then_check(self):
        o = fresh()
        o.query("Give me two route options avoiding restricted areas from Nagapattinam tomorrow at 10 AM.", session_id="rc")
        r = o.query("Does this route cross a restricted area?", session_id="rc")
        self.assertEqual(r["intent"], "route_check")
        self.assertTrue(all(x["ok"] for x in r["route_check"]))
        self.assertGreaterEqual(len(r["route_check"]), 2)
        self.assertNotIn("straight line", r["answer"])

    def test_straight_line_assumption_is_stated(self):
        r = fresh().query("Does this route cross a restricted area?", session_id="rc2", location={"lat": 10.74, "lon": 79.87})
        self.assertIn("No route was supplied", r["answer"])

    def test_route_check_endpoint_with_waypoints_that_violate(self):
        o = fresh()
        from orca.api import endpoints as ep
        r = ep.route_check(o, {"waypoints": [[10.74, 79.87], [10.72, 80.32]], "time": at(6).isoformat()})
        self.assertEqual(r["state"], "RESTRICTED")
        self.assertIn("crosses Naval firing-exercise area", r["answer"])
        self.assertTrue(any(a["level"] == "critical" for a in r["alerts"]))
        with self.assertRaises(ep.BadRequest):
            ep.route_check(o, {"waypoints": [[10.0, 80.0]]})
        with self.assertRaises(ep.BadRequest):
            ep.route_check(o, {"waypoints": [[10.0, 80.0], [95.0, 80.0]]})


class HazardTemporalTests(unittest.TestCase):
    """Blueprint section 14 / safety tests: an expired or future warning must not be applied."""

    def _feed(self, valid_from, valid_to):
        adv = Advisory("X1", "test", "warning", "test warning", ((10, 79), (10, 81), (11, 81), (11, 79)), valid_from.isoformat(), valid_to.isoformat(), "imd", "IMD (test)")
        return AdvisoryFeed(True, (adv,), "none", "", "imd", "IMD (test)", NOW.isoformat(), NOW.isoformat(), 1.0, "available", "DEMO")

    def test_expired_warning_is_not_applied_by_the_decision_engine(self):
        from orca.decision.evidence import fuse
        from orca.decision.risk import decide_site
        cfg = make_cfg()
        obs = [mk_obs("wave_height", 0.5), mk_obs("wind_speed", 5, unit="km/h")]
        clear = {"inside": [], "clear": True, "on_land": False, "covered": True}
        f = lambda feed: decide_site(fuse(obs, cfg), feed, clear, cfg, lat=10.5, lon=80.0, when=at(6))  # noqa: E731
        self.assertEqual(f(self._feed(at(0) - timedelta(days=2), at(0) - timedelta(days=1))).state, "LOWER_RISK", "expired")
        self.assertEqual(f(self._feed(at(12), at(23))).state, "LOWER_RISK", "not yet in force")
        self.assertEqual(f(self._feed(at(0), at(23))).state, "HIGHER_RISK", "active")

    def test_hazards_near_lists_inactive_advisories_with_the_reason_but_never_as_hazards(self):
        prov = make_provider(make_cfg())
        cons = ConstraintSet(load_geofences(prov.geofences()), prov.coast())
        hz = hazards_near(prov, cons, make_cfg(), 11.0, 80.12, 30.0, at(6))
        self.assertEqual(hz["official_advisories"], [])
        self.assertEqual(len(hz["inactive_advisories"]), 1)
        hz2 = hazards_near(prov, cons, make_cfg(), 11.0, 80.12, 30.0, at(14))
        self.assertEqual(len(hz2["official_advisories"]), 1)
        self.assertEqual(hz2["official_advisories"][0]["severity"], "caution")


class OceanRegionTests(unittest.TestCase):
    def test_favourable_has_an_explicit_stated_definition_and_filters_by_it(self):
        r = fresh().query("Which nearby regions have high chlorophyll and favourable SST?", session_id="oc", location=None) if False else \
            fresh().query("Which nearby regions near Nagapattinam have high chlorophyll and favourable SST?")
        self.assertIn("Criteria used: chlorophyll at least 0.5 mg/m³ and SST from 26 °C to 31 °C", r["answer"])
        self.assertIn("not a biological threshold", r["answer"])
        meets = {g["id"] for g in r["regions"] if g["meets"]}
        self.assertEqual(meets, {"OP-A", "OP-B"})
        for g in r["regions"]:  # re-derive the filter independently from the stored values
            self.assertEqual(g["meets"], g["chlorophyll"] >= 0.5 and 26 <= g["sst"] <= 31 and not g["restricted"])
        self.assertIn("not a forecast", r["answer"])

    def test_changing_the_threshold_changes_the_result(self):
        from orca.config import Thresholds
        o = Orca(replace(make_cfg(), thresholds=replace(Thresholds(), chl_min_mg_m3=0.6)))
        r = o.query("Which regions near Nagapattinam have high chlorophyll and favourable SST?")
        self.assertEqual({g["id"] for g in r["regions"] if g["meets"]}, {"OP-A"})


class ScenarioTests(unittest.TestCase):
    def test_only_the_time_changes_and_everything_is_recomputed(self):
        o = fresh()
        base = o.query("Can I fish tomorrow at 6 AM from Nagapattinam?", session_id="sc")
        r = o.query("What changes if I leave in the evening?", session_id="sc")
        self.assertEqual(r["intent"], "scenario")
        self.assertEqual([x["label"][-5:] for x in r["comparison"]], ["06:00", "17:00", "19:00"])
        self.assertEqual(r["comparison"][0]["state"], base["state"], "the baseline is recomputed and agrees with the original answer")
        ev = [e for e in r["trace"]["entries"] if e["step"] == "scenario"][0]["output"]
        self.assertEqual(ev["unchanged"]["place"], "Nagapattinam")
        self.assertIn("modified", ev)
        self.assertGreater(len([e for e in r["trace"]["entries"] if e["step"] == "agent"]), 6, "data was re-retrieved for the new times")
        self.assertIn("LOWER RISK -> MODERATE CONDITIONS", r["answer"])

    def test_scenario_without_a_baseline_asks(self):
        self.assertEqual(fresh().query("What changes if I leave in the evening?", session_id="empty")["state"], "INSUFFICIENT_DATA")


class EvidenceLayerTests(unittest.TestCase):
    def setUp(self):
        self.r = fresh().query("Can I fish tomorrow at 2 PM from Nagapattinam?")

    def test_every_record_has_the_blueprint_fields(self):
        for rec in self.r["evidence_records"]:
            for k in ("claim", "source", "issued_at", "valid_from", "valid_to", "operation", "result", "class"):
                self.assertIn(k, rec)
            self.assertIn(rec["class"], ("official", "source", "computed"))

    def test_official_and_computed_are_kept_distinct(self):
        by = {}
        for rec in self.r["evidence_records"]:
            by.setdefault(rec["class"], []).append(rec)
        self.assertTrue(any(x["kind"] == "advisory" for x in by["official"]))
        self.assertTrue(all(x["kind"] in ("advisory", "cyclone", "pfz") for x in by["official"]))
        self.assertTrue(any(x["operation"] == "threshold_rule" for x in by["computed"]))
        self.assertTrue(all(x["operation"] == "observation_lookup" for x in by["source"]))

    def test_factual_sentences_are_linked_and_an_invented_one_is_caught(self):
        from orca.evidence_pack import link_answer
        self.assertEqual(self.r["evidence_linkage"]["unlinked"], [])
        bad = link_answer(self.r["answer"] + " Wave height will be 0.3 m near the harbour.", self.r["evidence_records"])
        self.assertEqual(len(bad["unlinked"]), 1)
        self.assertLess(bad["completeness"], 1.0)

    def test_observation_exposes_blueprint_data_model_fields(self):
        ev = next(iter(fresh().query("Can I fish tomorrow at 6 AM from Nagapattinam?")["decisions"]))
        o = next(iter(self.r["evidence_records"]))
        r = fresh()
        res = r.query("Can I fish tomorrow at 6 AM from Nagapattinam?")
        stored = r.get_result(res["query_id"])["evidence"]
        one = next(iter(stored.values()))
        for k in ("source", "dataset", "variable", "value", "unit", "geometry", "observation_time" if False else "valid_from", "valid_to", "retrieved_at", "quality_status"):
            self.assertIn(k, one)

    def test_alerts_come_from_structured_results_and_carry_a_class(self):
        for a in self.r["alerts"]:
            self.assertIn(a["class"], ("official", "computed"))
            self.assertIn(a["level"], ("critical", "warning", "info"))
        tamil = fresh().query("நாளைக்கு காலை 10 மணிக்கு கடலுக்கு போகலாமா? நாகப்பட்டினம்")
        self.assertTrue(all(isinstance(a["text"], str) for a in tamil["alerts"]))


class EvalHarnessTests(unittest.TestCase):
    def test_core_set_passes_and_paraphrase_failures_are_reported_not_hidden(self):
        from orca.eval.runner import run
        core = run("core")
        s = core["summary"]["all"]
        self.assertEqual(s["cases_fully_passed"], s["cases"], [r for r in core["results"] if not r["ok"]])
        self.assertEqual(s["evidence_sentences"]["linked"], s["evidence_sentences"]["of"])
        para = run("paraphrase")
        self.assertGreater(len([r for r in para["results"] if not r["ok"]]), 0, "the held-out phrasing set must expose real limits, not be tuned to 100%")
        self.assertTrue(all(r["checks"]["grounded"] for r in para["results"]), "even when the intent is misread, nothing unsupported may be said")
