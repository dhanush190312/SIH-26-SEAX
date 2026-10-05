import unittest
from common import make_cfg
from orca.explain import build_ledger, fmt, gate_narration, verify_text
from orca.pipeline import Orca


class VerifierTests(unittest.TestCase):
    LED = build_ledger({"value": 0.9, "unit": "m"}, {"value": 14, "unit": "km/h"}, {"distance_km": 37.8}, {"duration_h": 2.5})

    def test_supported_numbers_pass(self):
        self.assertTrue(verify_text("Wave height 0.9 m, wind 14 km/h, 37.8 km away, about 2.5 h.", self.LED).ok)

    def test_invented_number_is_rejected(self):
        v = verify_text("Wave height 0.3 m at the zone.", self.LED)
        self.assertFalse(v.ok)
        self.assertEqual(v.unsupported, ("0.3 m",))

    def test_right_number_wrong_unit_is_rejected(self):
        self.assertFalse(verify_text("Wind 0.9 km/h.", self.LED).ok, "0.9 is a wave height, not a wind speed")

    def test_percentages_are_always_rejected(self):
        for s in ("87% safe", "a 0.5% chance", "100 % sure"):
            self.assertFalse(verify_text(s, self.LED).ok, s)

    def test_assurance_wording_is_rejected(self):
        for s in ("This route is the safest.", "It is completely safe.", "We guarantee calm seas."):
            self.assertFalse(verify_text(s, self.LED).ok, s)

    def test_tamil_text_with_latin_numbers_is_checked_the_same_way(self):
        self.assertTrue(verify_text("அலை உயரம் 0.9 m, காற்று 14 km/h.", self.LED).ok)
        self.assertFalse(verify_text("அலை உயரம் 5 m.", self.LED).ok)

    def test_compound_units_are_not_misread(self):
        self.assertTrue(verify_text("Wind 14 km/h and 0.9 m.", self.LED).ok)

    def test_gate_refuses_a_narration_more_optimistic_than_the_decision(self):
        ok, why, _ = gate_narration("It is safe to go, wave height 0.9 m.", "INSUFFICIENT_DATA", self.LED)
        self.assertFalse(ok)
        self.assertIn("INSUFFICIENT_DATA", why)
        ok, _, _ = gate_narration("Conditions are lower risk: wave height 0.9 m.", "LOWER_RISK", self.LED)
        self.assertTrue(ok)

    def test_fmt(self):
        self.assertEqual((fmt(1.40), fmt(55.0), fmt(0.0), fmt(None), fmt(3)), ("1.4", "55", "0", "?", "3"))


class NarratorFake:
    name = "fake"

    def __init__(self, text):
        self.text = text

    def narrate(self, *a, **k):
        return self.text


class NarratorPipelineTests(unittest.TestCase):
    Q = "Can I fish tomorrow at 6 AM from Nagapattinam?"

    def test_hallucinating_narrator_is_rejected_and_template_is_used(self):
        o = Orca(make_cfg(), narrator=NarratorFake("Wave height will be 0.3 m, 87% safe, go ahead!"))
        r = o.query(self.Q)
        self.assertFalse(r["narration"]["used"])
        self.assertIn("unsupported", r["narration"]["rejected_reason"])
        self.assertNotIn("87%", r["answer"])
        self.assertEqual(r["answer"], r["answer_template"])
        gate = [e for e in r["trace"]["entries"] if e["name"] == "narration_gate"][0]
        self.assertEqual(gate["status"], "withheld", "the rejection is itself in the trace")

    def test_faithful_narrator_is_accepted(self):
        o = Orca(make_cfg(), narrator=NarratorFake("Lower risk at Nagapattinam at 06:00: wave height 0.9 m and wind 14 km/h."))
        r = o.query(self.Q)
        self.assertTrue(r["narration"]["used"])
        self.assertTrue(r["answer"].startswith("Lower risk"))

    def test_optimistic_narrator_cannot_upgrade_an_insufficient_answer(self):
        o = Orca(make_cfg(), narrator=NarratorFake("It is safe to go from Chennai."))
        r = o.query("Can I fish tomorrow at 6 AM from Chennai?")
        self.assertEqual(r["state"], "INSUFFICIENT_DATA")
        self.assertFalse(r["narration"]["used"])
        self.assertNotIn("safe to go", r["answer"])

    def test_narrator_crash_falls_back(self):
        class Boom:
            name = "boom"
            def narrate(self, *a, **k): raise TimeoutError("slow")
        r = Orca(make_cfg(), narrator=Boom()).query(self.Q)
        self.assertFalse(r["narration"]["used"])
        self.assertIn("TimeoutError", r["narration"]["rejected_reason"])
        self.assertEqual(r["state"], "LOWER_RISK")
