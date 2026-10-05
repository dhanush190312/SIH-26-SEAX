import unittest
from dataclasses import replace
from common import at, make_cfg, mk_obs
from orca.decision.evidence import fuse
from orca.decision.risk import decide_site
from orca.schema import Advisory, AdvisoryFeed

CLEAR = {"inside": [], "clear": True, "on_land": False, "covered": True}


def feed(items=(), cyclone="none", quality="available", available=True, fresh=1.0):
    return AdvisoryFeed(available, tuple(items), cyclone, "", "imd", "IMD (test)", "2026-09-30T08:00:00+05:30", "2026-09-30T10:00:00+05:30",
                        fresh, quality, "DEMO", coverage=None)


def adv(sev):
    return Advisory("A1", "test", sev, f"{sev} text", ((10, 79), (10, 81), (11, 81), (11, 79)), at(0).isoformat(), at(23).isoformat(), "imd", "IMD (test)")


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()

    def decide(self, wave=0.9, wind=14.0, f=None, gs=CLEAR, extra=(), **kw):
        obs = [mk_obs("wave_height", wave), mk_obs("wind_speed", wind, unit="km/h"), *extra]
        obs = [o for o in obs if o.value is not None or True]
        fr = fuse(obs, self.cfg, wanted=["wave_height", "wind_speed"])
        return decide_site(fr, feed() if f is None else f, gs, self.cfg, lat=10.765, lon=80.0, when=at(6), **kw)

    def test_lower_risk(self):
        self.assertEqual(self.decide().state, "LOWER_RISK")

    def test_moderate_by_wave_and_by_wind(self):
        self.assertEqual(self.decide(wave=1.5).state, "MODERATE_CONDITIONS")
        self.assertEqual(self.decide(wind=25.0).state, "MODERATE_CONDITIONS")

    def test_higher_by_wave_wind_lightning(self):
        self.assertEqual(self.decide(wave=2.5).state, "HIGHER_RISK")
        self.assertEqual(self.decide(wind=40.0).state, "HIGHER_RISK")
        self.assertEqual(self.decide(extra=[mk_obs("lightning", 1.0, unit="flag")]).state, "HIGHER_RISK")

    def test_threshold_boundaries_are_inclusive_and_exact(self):
        self.assertEqual(self.decide(wave=1.49).state, "LOWER_RISK")
        self.assertEqual(self.decide(wave=1.5).state, "MODERATE_CONDITIONS")
        self.assertEqual(self.decide(wave=2.49).state, "MODERATE_CONDITIONS")

    def test_hard_geofence_is_restricted_even_in_calm_water(self):
        gs = {"inside": [{"id": "GF-X", "name": "Test zone", "kind": "restricted"}], "clear": False, "on_land": False, "covered": True}
        d = self.decide(wave=0.2, wind=3, gs=gs)
        self.assertEqual(d.state, "RESTRICTED")
        self.assertTrue(any(r.rule_id == "R-GEO-01" for r in d.reasons))

    def test_official_warning_forces_higher_and_ban_forces_restricted(self):
        self.assertEqual(self.decide(wave=0.2, wind=3, f=feed([adv("warning")])).state, "HIGHER_RISK")
        self.assertEqual(self.decide(wave=0.2, wind=3, f=feed([adv("ban")])).state, "RESTRICTED")
        self.assertEqual(self.decide(wave=0.2, wind=3, f=feed(cyclone="warning")).state, "HIGHER_RISK")
        self.assertEqual(self.decide(wave=0.2, wind=3, f=feed([adv("caution")])).state, "MODERATE_CONDITIONS")

    def test_missing_required_data_is_insufficient_never_lower(self):
        d = self.decide(wave=None)
        self.assertEqual(d.state, "INSUFFICIENT_DATA")
        self.assertIn("wave_height", d.missing)

    def test_missing_advisory_feed_is_insufficient(self):
        d = self.decide(f=feed(available=False))
        self.assertEqual(d.state, "INSUFFICIENT_DATA")
        self.assertIn("advisories", d.missing)

    def test_stale_advisory_feed_is_insufficient(self):
        self.assertEqual(self.decide(f=feed(quality="stale", fresh=99.0)).state, "INSUFFICIENT_DATA")

    def test_stale_observation_is_not_used(self):
        obs = [mk_obs("wave_height", 0.5, quality="stale"), mk_obs("wind_speed", 5, unit="km/h")]
        fr = fuse(obs, self.cfg, wanted=["wave_height", "wind_speed"])
        d = decide_site(fr, feed(), CLEAR, self.cfg, lat=10.7, lon=80, when=at(6))
        self.assertEqual(d.state, "INSUFFICIENT_DATA")
        self.assertIn("wave_height", d.stale)

    def test_proven_danger_beats_missing_data(self):
        d = self.decide(wave=None, wind=45.0)
        self.assertEqual(d.state, "HIGHER_RISK", "a missing wave value must not hide a proven wind hazard")

    def test_uncovered_location_is_insufficient_not_clear(self):
        d = self.decide(gs={"inside": [], "clear": True, "on_land": False, "covered": False})
        self.assertEqual(d.state, "INSUFFICIENT_DATA")
        self.assertIn("geofence", d.missing)

    def test_every_reason_is_machine_readable_and_has_a_rule_id(self):
        d = self.decide(wave=2.6, wind=30)
        for r in d.reasons:
            self.assertTrue(r.rule_id.startswith("R-"))
            self.assertIn(r.outcome, ("LOWER_RISK", "MODERATE_CONDITIONS", "HIGHER_RISK", "RESTRICTED", "INSUFFICIENT_DATA"))
        self.assertEqual(d.thresholds["wave_higher_m"], 2.5, "thresholds travel with the decision")

    def test_no_probabilities_anywhere_in_a_decision(self):
        import json
        s = json.dumps(self.decide(wave=2.6).to_dict())
        self.assertNotIn("probab", s.lower())
        self.assertNotIn("confidence", s.lower())

    def test_source_sensitivity_flags_a_decision_that_flips(self):
        a = [mk_obs("wave_height", 1.4, family="incois", uid="i"), mk_obs("wind_speed", 10, unit="km/h", family="incois", uid="iw")]
        b = [mk_obs("wave_height", 2.7, family="open_model", uid="o"), mk_obs("wind_speed", 10, unit="km/h", family="open_model", uid="ow")]
        fr = fuse(a + b, self.cfg, wanted=["wave_height", "wind_speed"])
        d = decide_site(fr, feed(), CLEAR, self.cfg, lat=10.7, lon=80, when=at(6))
        self.assertEqual(d.state, "LOWER_RISK", "priority selects INCOIS (1.4 m)")
        self.assertIn("SOURCE_SENSITIVE", d.flags)
        s = d.sensitivity[0]
        self.assertTrue(s["changes_decision"])
        self.assertEqual((s["state_with_selected"], s["state_with_alternative"]), ("LOWER_RISK", "HIGHER_RISK"))

    def test_sensitivity_quiet_when_states_agree(self):
        a = [mk_obs("wave_height", 1.0, family="incois", uid="i"), mk_obs("wind_speed", 10, unit="km/h", family="incois", uid="iw")]
        b = [mk_obs("wave_height", 1.4, family="open_model", uid="o"), mk_obs("wind_speed", 10, unit="km/h", family="open_model", uid="ow")]
        d = decide_site(fuse(a + b, self.cfg), feed(), CLEAR, self.cfg, lat=10.7, lon=80, when=at(6))
        self.assertNotIn("SOURCE_SENSITIVE", d.flags)
        self.assertFalse(d.sensitivity[0]["changes_decision"])
