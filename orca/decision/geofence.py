"""Geofences as hard constraints.

A :class:`Geofence` with ``hard=True`` (restricted areas, boundary buffers, and -- by
default -- marine protected areas) can never be entered by a route or recommended as a
destination. This module is plain geometry: the LLM never sees a polygon and cannot
override a verdict. :meth:`ConstraintSet.verify_path` is the last line of defence --
every route is re-checked against it after construction, and a violating route is
rejected rather than shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..geo import (Point, bearing_deg, compass, distance_to_polygon_km, point_in_polygon,
                   segment_intersects_polygon)

__all__ = ["Geofence", "ConstraintSet", "load_geofences"]


@dataclass(frozen=True)
class Geofence:
    fence_id: str
    name: str
    kind: str                            # restricted | boundary | mpa | port
    hard: bool
    polygon: tuple[Point, ...]
    source_note: str = ""

    def to_geojson(self) -> dict[str, Any]:
        ring = [[lon, lat] for lat, lon in self.polygon] + [[self.polygon[0][1], self.polygon[0][0]]]
        return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {"id": self.fence_id, "name": self.name, "kind": self.kind, "hard": self.hard,
                               "note": self.source_note}}


def load_geofences(raw: Iterable[Mapping[str, Any]]) -> list[Geofence]:
    return [Geofence(g["id"], g["name"], g["kind"], bool(g.get("hard", True)),
                     tuple((float(a), float(b)) for a, b in g["polygon"]), g.get("source_note", "")) for g in raw]


class ConstraintSet:
    """Hard geofences (+ optionally land) as one queryable object."""

    def __init__(self, fences: Sequence[Geofence], land: Sequence[Point] = (), *, mpa_hard: bool = True,
                 coverage: Sequence[float] | None = None) -> None:
        self.coverage = tuple(coverage) if coverage else None   # (south, west, north, east); None = unrestricted
        self.fences = list(fences)
        self.land = tuple(land)
        self.mpa_hard = mpa_hard

    # -- which fences bind --------------------------------------------------- #

    def is_binding(self, f: Geofence) -> bool:
        if f.kind == "mpa":
            return self.mpa_hard and f.hard
        return f.hard

    @property
    def binding(self) -> list[Geofence]:
        return [f for f in self.fences if self.is_binding(f)]

    def covered(self, lat: float, lon: float) -> bool:
        """Does the geofence dataset make any claim about this location? Outside its
        coverage, "no geofence" means "no data", not "clear"."""
        if self.coverage is None:
            return True
        s, w, n, e = self.coverage
        return s <= lat <= n and w <= lon <= e

    # -- point / segment / path --------------------------------------------- #

    def on_land(self, lat: float, lon: float) -> bool:
        return bool(self.land) and point_in_polygon(lat, lon, self.land)

    def point_violations(self, lat: float, lon: float) -> list[Geofence]:
        return [f for f in self.binding if point_in_polygon(lat, lon, f.polygon)]

    def segment_violations(self, a: Point, b: Point) -> list[Geofence]:
        return [f for f in self.binding if segment_intersects_polygon(a, b, f.polygon)]

    def segment_hits_land(self, a: Point, b: Point) -> bool:
        return bool(self.land) and segment_intersects_polygon(a, b, self.land)

    def blocked_segment(self, a: Point, b: Point) -> list[str]:
        ids = [f.fence_id for f in self.segment_violations(a, b)]
        if self.segment_hits_land(a, b):
            ids.append("LAND")
        return ids

    def verify_path(self, path: Sequence[Point]) -> dict[str, Any]:
        """Deterministic post-check: does *any* leg touch a binding geofence or land?"""
        hits: dict[str, int] = {}
        for a, b in zip(path, path[1:]):
            for fid in self.blocked_segment(a, b):
                hits[fid] = hits.get(fid, 0) + 1
        return {"ok": not hits, "violations": hits}

    # -- status text for the evidence card ----------------------------------- #

    def status_at(self, lat: float, lon: float) -> dict[str, Any]:
        inside = self.point_violations(lat, lon)
        fences = self.binding
        nearest = None
        if fences:
            nearest = min(((distance_to_polygon_km(lat, lon, f.polygon), f) for f in fences), key=lambda t: t[0])
        info: dict[str, Any] = {
            "inside": [{"id": f.fence_id, "name": f.name, "kind": f.kind} for f in inside],
            "clear": not inside,
            "on_land": self.on_land(lat, lon),
            "covered": self.covered(lat, lon),
        }
        if nearest:
            d, f = nearest
            cx = sum(p[0] for p in f.polygon) / len(f.polygon)
            cy = sum(p[1] for p in f.polygon) / len(f.polygon)
            info["nearest"] = {"id": f.fence_id, "name": f.name, "kind": f.kind, "distance_km": round(d, 1),
                               "direction": compass(bearing_deg(lat, lon, cx, cy))}
        return info

    def to_geojson(self) -> dict[str, Any]:
        return {"type": "FeatureCollection", "features": [f.to_geojson() for f in self.fences]}
