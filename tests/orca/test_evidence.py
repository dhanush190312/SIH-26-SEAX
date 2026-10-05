import unittest
from dataclasses import replace
from common import make_cfg, mk_obs
from orca.decision.evidence import fuse


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()

    def two(self, a, b):
        return [mk_obs("wave_height", a, family="incois", uid="i"), mk_obs("wave_height", b, family="open_model", uid="o")]

    def test_disagreement_is_reported_with_every_value(self):
        fr = fuse(self.two(1.4, 2.1), self.cfg)
        self.assertEqual(len(fr.conflicts), 1)
        c = fr.conflicts[0]
        self.assertEqual({v["source_family"] for v in c.values}, {"incois", "open_model"})
        self.assertAlmostEqual(c.difference, 0.7)
        self.assertEqual(c.selected_source, "incois")
        self.assertEqual(fr.value("wave_height"), 1.4)
        self.assertEqual(fr.variables["wave_height"].status, "conflict")

    def test_within_tolerance_is_not_a_conflict(self):
        self.assertEqual(fuse(self.two(1.4, 1.6), self.cfg).conflicts, ())

    def test_priority_is_configuration_not_hardcoded(self):
        cfg = replace(self.cfg, source_priority=("open_model", "incois"))
        fr = fuse(self.two(1.4, 2.1), cfg)
        self.assertEqual(fr.conflicts[0].selected_source, "open_model")
        self.assertEqual(fr.value("wave_height"), 2.1)

    def test_unlisted_source_ranks_last(self):
        obs = [mk_obs("wave_height", 3.0, family="mystery", uid="m"), mk_obs("wave_height", 1.0, family="incois", uid="i")]
        self.assertEqual(fuse(obs, self.cfg).value("wave_height"), 1.0)

    def test_missing_is_reported_not_filled(self):
        fr = fuse([mk_obs("wave_height", None)], self.cfg, wanted=["wave_height", "wind_speed"])
        self.assertIn("wave_height", fr.missing)
        self.assertIn("wind_speed", fr.missing, "a wanted variable with no observation at all must still be listed")
        self.assertIsNone(fr.value("wave_height"))

    def test_stale_is_excluded_unless_explicitly_allowed(self):
        stale = [mk_obs("wave_height", 1.0, quality="stale")]
        self.assertIn("wave_height", fuse(stale, self.cfg).stale)
        allowed = fuse(stale, replace(self.cfg, allow_stale=True))
        self.assertEqual(allowed.value("wave_height"), 1.0)
        self.assertIn("wave_height", allowed.stale, "even when allowed, staleness stays flagged")

    def test_stale_source_cannot_win_over_fresh_lower_priority_source(self):
        obs = [mk_obs("wave_height", 0.1, family="incois", quality="stale", uid="i"), mk_obs("wave_height", 2.0, family="open_model", uid="o")]
        self.assertEqual(fuse(obs, self.cfg).value("wave_height"), 2.0)
