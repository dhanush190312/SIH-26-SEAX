import random
import unittest
from common import at, make_cfg
from orca.data import make_provider
from orca.decision.evidence import fuse
from orca.decision.geofence import ConstraintSet, Geofence, load_geofences
from orca.decision.route import OBJECTIVES, ExposureField, plan_route, plan_route_set
from orca.geo import haversine_km


def build(cfg=None, mpa_hard=True):
    cfg = cfg or make_cfg()
    prov = make_provider(cfg)
    cons = ConstraintSet(load_geofences(prov.geofences()), prov.coast(), mpa_hard=mpa_hard, coverage=prov.geofence_coverage())
    return cfg, prov, cons


def field_at(cfg, prov, when):
    W, V = [], []
    for p in prov.weather_points():
        fr = fuse(prov.weather(p["lat"], p["lon"], when), cfg)
        W.append((p["lat"], p["lon"], fr.value("wave_height")))
        V.append((p["lat"], p["lon"], fr.value("wind_speed")))
    return ExposureField(W, V, cfg)


class GeofenceTests(unittest.TestCase):
    def test_mpa_is_binding_by_default_and_switchable(self):
        _, _, on = build(mpa_hard=True)
        _, _, off = build(mpa_hard=False)
        self.assertIn("GF-MPA", [f.fence_id for f in on.binding])
        self.assertNotIn("GF-MPA", [f.fence_id for f in off.binding])

    def test_soft_port_geofence_does_not_bind(self):
        _, _, c = build()
        self.assertNotIn("GF-PORT", [f.fence_id for f in c.binding])

    def test_point_status_inside_naval_area(self):
        _, _, c = build()
        st = c.status_at(10.74, 80.14)
        self.assertFalse(st["clear"])
        self.assertEqual(st["inside"][0]["id"], "GF-NAVAL")

    def test_verify_path_catches_a_violating_path(self):
        _, _, c = build()
        bad = [(10.74, 79.87), (10.72, 80.32)]  # passes straight through the naval polygon
        chk = c.verify_path(bad)
        self.assertFalse(chk["ok"])
        self.assertIn("GF-NAVAL", chk["violations"])

    def test_land_counts_as_a_violation(self):
        _, _, c = build()
        self.assertTrue(c.on_land(10.77, 79.5))
        self.assertFalse(c.verify_path([(10.77, 79.5), (10.77, 79.6)])["ok"])

    def test_coverage_gate(self):
        _, _, c = build()
        self.assertTrue(c.covered(10.7, 80.0))
        self.assertFalse(c.covered(13.08, 80.30), "Chennai is outside the demo dataset's coverage")


class RouteTests(unittest.TestCase):
    def test_direct_route_blocked_by_naval_area_is_rerouted(self):
        cfg, prov, cons = build()
        f = field_at(cfg, prov, at(6))
        r = plan_route((10.765, 79.87), (10.72, 80.32), cons, f, cfg, objective="shortest_distance", dest_id="PFZ-B", dest_name="B")
        self.assertTrue(r["feasible"])
        self.assertIn("GF-NAVAL", r["direct_blocked_by"])
        self.assertGreater(r["detour_km"], 0)
        self.assertTrue(cons.verify_path([tuple(p) for p in r["path"]])["ok"])

    def test_property_no_route_ever_enters_a_hard_geofence(self):
        """24 random destinations x 3 objectives: every returned route passes the post-check."""
        cfg, prov, cons = build()
        f = field_at(cfg, prov, at(10))
        rng = random.Random(7)
        feasible = 0
        for _ in range(24):
            dest = (rng.uniform(10.3, 11.1), rng.uniform(79.95, 80.45))
            for obj in OBJECTIVES:
                r = plan_route((10.765, 79.87), dest, cons, f, cfg, objective=obj)
                if r["feasible"]:
                    feasible += 1
                    self.assertTrue(cons.verify_path([tuple(p) for p in r["path"]])["ok"], (dest, obj, r["path"]))
        self.assertGreater(feasible, 40, "the property test must actually exercise routes")

    def test_destination_inside_a_hard_geofence_is_not_reached(self):
        cfg, prov, cons = build()
        r = plan_route((10.765, 79.87), (10.74, 80.14), cons, None, cfg, objective="shortest_distance")
        if r["feasible"]:
            self.assertTrue(cons.verify_path([tuple(p) for p in r["path"]])["ok"], "a route may stop at the fence edge, never inside")

    def test_lower_exposure_never_worse_than_shortest_on_its_own_metric(self):
        cfg, prov, cons = build()
        f = field_at(cfg, prov, at(10))
        a = plan_route((10.765, 79.87), (10.72, 80.32), cons, f, cfg, objective="shortest_distance")
        b = plan_route((10.765, 79.87), (10.72, 80.32), cons, f, cfg, objective="lower_exposure")
        self.assertLessEqual(b["mean_exposure"], a["mean_exposure"] + 1e-9)

    def test_route_set_excludes_restricted_and_higher_risk_zones_and_never_says_safest(self):
        cfg, prov, cons = build()
        zones = [dict(id="A", name="A", lat=11.0, lon=80.12, support=3, state="LOWER_RISK", restricted=False),
                 dict(id="D", name="D", lat=10.74, lon=80.14, support=3, state="RESTRICTED", restricted=True),
                 dict(id="X", name="X", lat=10.4, lon=80.0, support=3, state="HIGHER_RISK", restricted=False)]
        rs = plan_route_set((10.765, 79.87), zones, cons, None, cfg)
        dests = {r["destination"]["id"] for r in rs["routes"]}
        self.assertEqual(dests, {"A"})
        import json
        self.assertNotIn("safest", json.dumps(rs["routes"]).lower(), "no route may be labelled 'safest'")

    def test_route_set_with_no_legal_zone_offers_nothing(self):
        cfg, prov, cons = build()
        rs = plan_route_set((10.765, 79.87), [dict(id="D", name="D", lat=10.74, lon=80.14, support=3, state="RESTRICTED", restricted=True)], cons, None, cfg)
        self.assertEqual(rs["routes"], [])

    def test_no_legal_path_is_reported_not_faked(self):
        cfg, prov, _ = build()
        wall = Geofence("WALL", "wall", "restricted", True, ((9.0, 79.0), (9.0, 82.0), (12.5, 82.0), (12.5, 79.0)))
        cons = ConstraintSet([wall], [], mpa_hard=True)
        r = plan_route((10.0, 80.0), (11.0, 81.0), cons, None, cfg, objective="shortest_distance")
        self.assertFalse(r["feasible"])
        self.assertNotIn("path", r)
