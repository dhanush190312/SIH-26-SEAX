"""Deterministic geospatial primitives (stdlib only).

Everything the LLM must never do -- distance, bearing, containment, intersection --
lives here as plain functions with tests. Polygons are lists of ``(lat, lon)``.
Intersection tests run in planar (lon, lat) space, which is exact for topology
(an affine map preserves intersection); only *distances* use spherical/local maths.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

EARTH_RADIUS_KM = 6371.0088

Point = tuple[float, float]  # (lat, lon)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


_COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


def compass(bearing: float) -> str:
    return _COMPASS[int((bearing % 360) / 22.5 + 0.5) % 16]


def destination_point(lat: float, lon: float, bearing: float, km: float) -> Point:
    d = km / EARTH_RADIUS_KM
    b = math.radians(bearing)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540.0) % 360.0 - 180.0


def polyline_length_km(points: Sequence[Point]) -> float:
    return sum(haversine_km(a[0], a[1], b[0], b[1]) for a, b in zip(points, points[1:]))


def densify(a: Point, b: Point, step_km: float) -> list[Point]:
    """Points from ``a`` to ``b`` inclusive, at most ``step_km`` apart (linear in lat/lon)."""
    d = haversine_km(a[0], a[1], b[0], b[1])
    n = max(1, int(math.ceil(d / max(step_km, 1e-6))))
    return [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n + 1)]


def point_in_polygon(lat: float, lon: float, poly: Sequence[Point]) -> bool:
    """Ray casting. Points exactly on an edge count as inside (conservative for geofences)."""
    inside = False
    n = len(poly)
    for i in range(n):
        y1, x1 = poly[i]
        y2, x2 = poly[(i + 1) % n]
        if _on_segment((lon, lat), (x1, y1), (x2, y2)):
            return True
        if (y1 > lat) != (y2 > lat):
            xint = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < xint:
                inside = not inside
    return inside


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(p, a, b, eps: float = 1e-12) -> bool:
    if abs(_orient(a, b, p)) > eps:
        return False
    return min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Do segments p1-p2 and q1-q2 (each ``(lat, lon)``) touch or cross?"""
    a, b, c, d = (p1[1], p1[0]), (p2[1], p2[0]), (q1[1], q1[0]), (q2[1], q2[0])
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)) and o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0:
        return True
    return _on_segment(c, a, b) or _on_segment(d, a, b) or _on_segment(a, c, d) or _on_segment(b, c, d)


def segment_intersects_polygon(a: Point, b: Point, poly: Sequence[Point]) -> bool:
    """True if the segment enters, touches, or lies inside the polygon."""
    if point_in_polygon(a[0], a[1], poly) or point_in_polygon(b[0], b[1], poly):
        return True
    n = len(poly)
    return any(segments_intersect(a, b, poly[i], poly[(i + 1) % n]) for i in range(n))


def _local_xy(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    return (lon - lon0) * 111.320 * math.cos(math.radians(lat0)), (lat - lat0) * 110.574


def point_segment_distance_km(p: Point, a: Point, b: Point) -> float:
    px, py = 0.0, 0.0
    ax, ay = _local_xy(p[0], p[1], a[0], a[1])
    bx, by = _local_xy(p[0], p[1], b[0], b[1])
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(ax + t * dx - px, ay + t * dy - py)


def distance_to_polygon_km(lat: float, lon: float, poly: Sequence[Point]) -> float:
    """0 inside; otherwise the distance to the nearest edge."""
    if point_in_polygon(lat, lon, poly):
        return 0.0
    n = len(poly)
    return min(point_segment_distance_km((lat, lon), poly[i], poly[(i + 1) % n]) for i in range(n))


def nearest_vertex_bearing(lat: float, lon: float, poly: Iterable[Point]) -> float:
    best = min(poly, key=lambda v: haversine_km(lat, lon, v[0], v[1]))
    return bearing_deg(lat, lon, best[0], best[1])


def idw(lat: float, lon: float, samples: Sequence[tuple[float, float, float]], power: float = 2.0) -> float | None:
    """Inverse-distance-weighted value at (lat, lon) from ``(lat, lon, value)`` samples.

    Used only for the route *exposure field*; an exact hit returns that sample."""
    if not samples:
        return None
    num = den = 0.0
    for slat, slon, v in samples:
        d = haversine_km(lat, lon, slat, slon)
        if d < 0.05:
            return v
        w = 1.0 / d ** power
        num += w * v
        den += w
    return num / den
