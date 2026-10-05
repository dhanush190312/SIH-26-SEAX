import unittest
from common import ROOT  # noqa: F401
from orca import geo


class GeoTests(unittest.TestCase):
    SQ = [(10.0, 80.0), (10.0, 81.0), (11.0, 81.0), (11.0, 80.0)]

    def test_haversine_known_distance(self):
        # one degree of latitude is ~111.2 km
        self.assertAlmostEqual(geo.haversine_km(10, 80, 11, 80), 111.2, delta=0.3)

    def test_point_in_polygon_inside_outside_edge(self):
        self.assertTrue(geo.point_in_polygon(10.5, 80.5, self.SQ))
        self.assertFalse(geo.point_in_polygon(9.9, 80.5, self.SQ))
        self.assertTrue(geo.point_in_polygon(10.0, 80.5, self.SQ), "boundary counts as inside (conservative)")

    def test_segment_intersection_cases(self):
        self.assertTrue(geo.segment_intersects_polygon((9.0, 80.5), (12.0, 80.5), self.SQ), "crossing through")
        self.assertTrue(geo.segment_intersects_polygon((10.4, 80.4), (10.6, 80.6), self.SQ), "fully inside")
        self.assertFalse(geo.segment_intersects_polygon((9.0, 79.0), (9.5, 79.9), self.SQ), "clear miss")
        self.assertTrue(geo.segment_intersects_polygon((9.0, 80.0), (10.0, 80.0), self.SQ), "touching a vertex counts")

    def test_distance_to_polygon(self):
        self.assertEqual(geo.distance_to_polygon_km(10.5, 80.5, self.SQ), 0.0)
        self.assertAlmostEqual(geo.distance_to_polygon_km(9.0, 80.5, self.SQ), 111.2, delta=1.0)  # local planar approx vs spherical: <1%

    def test_bearing_and_compass(self):
        self.assertEqual(geo.compass(geo.bearing_deg(10, 80, 11, 80)), "N")
        self.assertEqual(geo.compass(geo.bearing_deg(10, 80, 10, 81)), "E")

    def test_idw_exact_hit_and_none(self):
        self.assertEqual(geo.idw(10, 80, [(10, 80, 3.5), (11, 81, 9.0)]), 3.5)
        self.assertIsNone(geo.idw(10, 80, []))
