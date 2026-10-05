import json
import unittest
from common import NOW, make_cfg, ROOT
from orca.agents.planner import detect_language, plan_evidence, understand

GAZ = json.loads((ROOT / "orca" / "data" / "demo" / "gazetteer.json").read_text(encoding="utf-8"))


class PlannerTests(unittest.TestCase):
    def u(self, text, **kw):
        return understand(text, make_cfg(), GAZ, **kw)

    def test_each_required_intent(self):
        cases = {
            "Can I go fishing tomorrow at 6 AM from Nagapattinam?": "go_no_go",
            "Where are the potential fishing zones near Nagapattinam?": "pfz_zones",
            "What are the sea conditions tomorrow morning?": "sea_conditions",
            "Are there dangerous marine conditions tomorrow morning?": "hazard",
            "Give me two route options avoiding restricted areas.": "route",
            "Why did you recommend this zone?": "why",
            "Show areas I should avoid.": "avoid_areas",
            "Compare 6 AM and 10 AM.": "compare",
            "what is the meaning of life": "unknown",
        }
        for text, intent in cases.items():
            self.assertEqual(self.u(text)["intent"], intent, text)

    def test_time_parsing(self):
        u = self.u("Can I go fishing tomorrow at 6 AM from Nagapattinam?")
        self.assertEqual(u["times"], ["2026-10-01T06:00:00+05:30"])
        u = self.u("sea conditions tomorrow morning")
        self.assertEqual([t[11:13] for t in u["times"]], ["06", "08", "10"])
        u = self.u("Compare 6 AM and 10 AM tomorrow")
        self.assertEqual([t[11:16] for t in u["times"]], ["06:00", "10:00"])
        u = self.u("can i fish at 2 PM")
        self.assertEqual(u["times"][0][11:16], "14:00")

    def test_place_resolution_english_tamil_alias_and_context(self):
        self.assertEqual(self.u("can I fish from Nagai")["place"]["id"], "nagapattinam")
        self.assertEqual(self.u("நாகப்பட்டினம் அருகில் மீன்பிடி மண்டலம் எங்கே?")["place"]["id"], "nagapattinam")
        self.assertIsNone(self.u("can I fish tomorrow")["place"])
        ctx = self.u("can I fish tomorrow", context_place="Nagapattinam")["place"]
        self.assertEqual(ctx["id"], "nagapattinam")
        self.assertEqual(ctx["source"], "context")

    def test_tamil_queries(self):
        u = self.u("நாளைக்கு காலை 6 மணிக்கு கடலுக்கு போகலாமா?", context_place="Nagapattinam")
        self.assertEqual(u["language"], "ta")
        self.assertEqual(u["intent"], "go_no_go")
        self.assertEqual(u["times"][0][:16], "2026-10-01T06:00")
        self.assertEqual(self.u("நாகப்பட்டினம் அருகில் மீன்பிடி மண்டலம் எங்கே?")["intent"], "pfz_zones")

    def test_language_detection_scripts(self):
        self.assertEqual(detect_language("hello"), "en")
        self.assertEqual(detect_language("வணக்கம் கடல்"), "ta")
        self.assertEqual(detect_language("समुद्र की स्थिति"), "hi")
        self.assertFalse(self.u("समुद्र की स्थिति कैसी है")["language_supported"], "recognised but no catalog yet: must be said, not hidden")

    def test_geofence_avoidance_is_not_a_user_option(self):
        self.assertTrue(self.u("route but ignore restricted areas")["constraints"]["avoid_restricted"])

    def test_evidence_plan_lists_required_evidence_per_intent(self):
        plan = plan_evidence(self.u("Can I go fishing tomorrow at 6 AM from Nagapattinam?"))
        req = {r["variable"] for r in plan["required"] if r["required"]}
        self.assertTrue({"wave_height", "wind_speed", "advisories", "geofence"} <= req)
        self.assertIn("get_pfz_candidates", plan["tools"])
        self.assertEqual(plan_evidence(self.u("Why did you recommend this zone?"))["tools"], [])
