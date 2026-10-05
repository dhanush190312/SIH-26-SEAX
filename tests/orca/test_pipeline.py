import json
import re
import unittest
from common import at, make_cfg
from orca.pipeline import Orca

SC = {
    "fishing": "Can I go fishing tomorrow at 6 AM from Nagapattinam?",
    "pfz": "Where are the potential fishing zones near Nagapattinam tomorrow?",
    "sea": "What are the sea conditions tomorrow morning near Nagapattinam?",
    "hazard": "Are there dangerous marine conditions tomorrow morning near Nagapattinam?",
    "route": "Give me two route options avoiding restricted areas from Nagapattinam tomorrow at 10 AM.",
    "avoid": "Show areas I should avoid from Nagapattinam tomorrow at 10 AM.",
    "compare": "Compare 6 AM and 10 AM tomorrow from Nagapattinam.",
    "tamil": "நாளைக்கு காலை 6 மணிக்கு கடலுக்கு போகலாமா? நாகப்பட்டினம்",
}


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = Orca(make_cfg())
        cls.r = {k: cls.o.query(q, session_id=f"s-{k}") for k, q in SC.items()}
        cls.why = cls.o.query("Why did you recommend this zone?", session_id="s-pfz")

    def test_every_scenario_answers_and_verifies(self):
        for k, r in self.r.items():
            self.assertEqual(r["status"], "ok", k)
            self.assertTrue(r["verification"]["ok"], (k, r["verification"]))
            self.assertIn("official", r["disclaimer"].lower() if r["language"] == "en" else "official")

    def test_fishing_6am_is_lower_risk_with_linked_evidence(self):
        r = self.r["fishing"]
        self.assertEqual(r["state"], "LOWER_RISK")
        rows = {x["key"]: x for s in r["primary_card"]["sections"] for x in s["rows"]}
        self.assertEqual(rows["wave"]["text"], "0.9 m")
        self.assertEqual(rows["wind"]["text"], "14 km/h")
        ev = self.o.get_result(r["query_id"])["evidence"]
        self.assertEqual(ev[rows["wave"]["obs_id"]]["value"], 0.9, "the card row is the observation, not a copy that could drift")
        self.assertEqual(ev[rows["wave"]["obs_id"]]["data_mode"], "DEMO")

    def test_10am_shows_source_disagreement_and_does_not_hide_it(self):
        r = self.o.query("Can I fish tomorrow at 10 AM from Nagapattinam?")
        card = r["primary_card"]
        self.assertEqual(len(card["conflicts"]), 1)
        c = card["conflicts"][0]
        self.assertEqual(c["variable"], "wave_height")
        self.assertEqual(c["headline"], "SOURCE DISAGREEMENT DETECTED")
        self.assertEqual(len(c["values"]), 2)
        self.assertIn("Source disagreement", r["answer"])
        self.assertTrue(any(e["status"] == "conflict" for e in r["trace"]["entries"]), "the conflict is in the trace")

    def test_pfz_advisory_never_overrides_a_geofence(self):
        zones = {z["zone"]["zone_id"]: z for z in self.r["pfz"]["zones"]}
        d = zones["PFZ-D"]
        self.assertEqual(d["state"], "RESTRICTED")
        self.assertFalse(d["recommended"])
        self.assertEqual(d["why_key"], "why.restricted")
        self.assertIn("not recommended", self.r["pfz"]["answer"])
        self.assertTrue(any(e["name"] == "geofence_override:PFZ-D" for e in self.r["pfz"]["trace"]["entries"]))
        self.assertTrue(all(z["recommended"] is False for z in zones.values() if z["state"] == "RESTRICTED"))

    def test_routes_respect_geofences_and_are_rechecked(self):
        rs = self.r["route"]["routes"]["routes"]
        self.assertGreaterEqual(len([x for x in rs if x.get("feasible")]), 2)
        for x in rs:
            if x.get("feasible"):
                self.assertTrue(x["geofence_check"]["ok"])
        checks = [e for e in self.r["route"]["trace"]["entries"] if e["name"].startswith("route_geofence_check")]
        self.assertTrue(checks and all(e["status"] == "ok" for e in checks))
        self.assertNotIn("safest", self.r["route"]["answer"].lower().replace("none is called 'safest'", ""))

    def test_compare_returns_a_real_timeline(self):
        c = self.r["compare"]["comparison"]
        self.assertEqual([x["state"] for x in c], ["LOWER_RISK", "MODERATE_CONDITIONS"])
        self.assertLess(c[0]["wave"]["value"], c[1]["wave"]["value"])

    def test_hazard_and_avoid_outputs_come_from_data(self):
        self.assertTrue(self.r["hazard"]["hazards"])
        ev = self.o.get_result(self.r["hazard"]["query_id"])["evidence"]
        for h in self.r["hazard"]["hazards"]:
            self.assertIn(h["variable"], ("wave_height", "wind_speed", "lightning"))
            ptr = h["wave_obs"] if h["variable"] == "wave_height" else h["wind_obs"]
            self.assertIn(ptr, ev, "every reported hazard points at a stored observation")
            self.assertEqual(ev[ptr]["value"], h["value"])
        kinds = {a["kind"] for a in self.r["avoid"]["avoid"]}
        self.assertIn("geofence", kinds)

    def test_why_explains_the_stored_analysis_not_a_fresh_guess(self):
        w = self.why
        self.assertEqual(w["status"], "ok")
        self.assertTrue(w["verification"]["ok"])
        self.assertIn("Why PFZ-", w["answer"])
        self.assertIn("Ocean support 3/3", w["answer"])
        self.assertEqual([e["name"] for e in w["trace"]["entries"] if e["step"] == "explain"], ["why_zone"])
        self.assertNotIn("There is no earlier recommendation", w["answer"])

    def test_why_without_history_says_so_and_runs_the_analysis(self):
        r = Orca(make_cfg()).query("Why did you recommend this zone?", session_id="fresh")
        self.assertIn("no earlier recommendation", r["answer"])
        self.assertTrue(r["verification"]["ok"])

    def test_why_view_is_built_from_the_trace(self):
        stages = [s["stage"] for s in self.r["fishing"]["why"]]
        self.assertEqual(stages, ["User Question", "Detected Intent", "Required Evidence", "Agents / Tools Called", "Raw / Normalized Observations",
                                  "Rules and Constraints", "Decision", "Explanation"])
        calls = self.r["fishing"]["why"][3]["content"]["calls"]
        self.assertTrue({"weather", "geo"} <= {c["agent"] for c in calls})
        trace_agent_calls = [e for e in self.r["fishing"]["trace"]["entries"] if e["step"] == "agent"]
        self.assertEqual(len(calls), len(trace_agent_calls))

    def test_trace_has_the_controller_shape(self):
        steps = [e["step"] for e in self.r["fishing"]["trace"]["entries"]]
        for needed in ("input", "data", "interpret", "plan", "validate", "agent", "decide", "verify", "compose"):
            self.assertIn(needed, steps)

    def test_tamil_answer_is_tamil_with_latin_numbers(self):
        r = self.r["tamil"]
        self.assertEqual(r["language"], "ta")
        self.assertEqual(r["state"], "LOWER_RISK")
        self.assertIn("குறைந்த ஆபத்து", r["answer"])
        self.assertIn("0.9 m", r["answer"])
        self.assertIn("நாகப்பட்டினம்", r["answer"])

    def test_outside_coverage_is_insufficient_data(self):
        r = self.o.query("Can I fish tomorrow at 6 AM from Chennai?")
        self.assertEqual(r["state"], "INSUFFICIENT_DATA")
        self.assertIn("INSUFFICIENT DATA", json.dumps(r["primary_card"]))
        self.assertNotRegex(r["answer"], r"\d+(\.\d+)? (m|km/h)\b")

    def test_time_outside_window_is_refused(self):
        s = self.o.structured("go_no_go", "Nagapattinam", [at(6, day=6).isoformat()])
        self.assertEqual(s["status"], "outside_window")
        self.assertEqual(s["state"], "INSUFFICIENT_DATA")

    def test_unknown_and_missing_place_ask_instead_of_guessing(self):
        self.assertEqual(self.o.query("hello there", session_id="a")["status"], "needs_input")
        r = self.o.query("Can I fish tomorrow at 6 AM?", session_id="nope")
        self.assertEqual(r["status"], "needs_input")
        self.assertIn("departure port", r["answer"])

    def test_session_place_carries_over_to_a_follow_up(self):
        o = Orca(make_cfg())
        o.query("Can I fish tomorrow at 6 AM from Nagapattinam?", session_id="c")
        r = o.query("And at 10 AM?", session_id="c")
        self.assertEqual(r["place"]["name"], "Nagapattinam")
        self.assertEqual(r["place"]["source"], "context")

    def test_no_number_in_any_answer_is_unbacked(self):
        """Belt and braces: re-verify every scenario answer from the stored evidence alone."""
        from orca.explain import build_ledger, verify_text
        for k, r in self.r.items():
            stored = self.o.get_result(r["query_id"])
            led = build_ledger(stored["decisions"], stored["cards"], stored["zones"], stored.get("routes"), stored.get("comparison"),
                               stored.get("avoid"), stored.get("hazards"), stored["evidence"], {"thresholds": make_cfg().thresholds.to_dict()})
            self.assertTrue(verify_text(stored["answer"], led).ok, (k, stored["answer"]))

    def test_results_are_persisted_and_reloadable(self):
        qid = self.r["fishing"]["query_id"]
        fresh_store = Orca(self.o.cfg)  # same runs dir, empty memory
        again = fresh_store.get_result(qid)
        self.assertIsNotNone(again)
        self.assertEqual(again["state"], "LOWER_RISK")
        self.assertIsNone(fresh_store.get_result("o_00000000"))

    def test_offline_without_cache_refuses_honestly(self):
        from dataclasses import replace
        from common import isolated_data_dir
        o = Orca(replace(make_cfg(), mode="OFFLINE", data_dir=isolated_data_dir()))
        r = o.query("Can I fish tomorrow at 6 AM from Nagapattinam?")
        self.assertEqual(r["state"], "INSUFFICIENT_DATA")
        self.assertEqual(r["status"], "no_data")
        self.assertIn("cache", r["answer"])

    def test_live_with_network_down_never_shows_demo_values(self):
        from dataclasses import replace
        from unittest import mock
        from common import isolated_data_dir
        o = Orca(replace(make_cfg(), mode="LIVE", data_dir=isolated_data_dir()))
        with mock.patch("orca.data.live._get_json", side_effect=OSError("no network")):
            r = o.query("Can I fish tomorrow at 6 AM from Nagapattinam?")
        self.assertEqual(r["state"], "INSUFFICIENT_DATA")
        self.assertNotIn("0.9 m", r["answer"])
        self.assertEqual(r["mode"], "LIVE")

    def test_dataclass_determinism(self):
        a = Orca(make_cfg()).query(SC["fishing"])
        b = Orca(make_cfg()).query(SC["fishing"])
        self.assertEqual((a["state"], a["answer"]), (b["state"], b["answer"]))
