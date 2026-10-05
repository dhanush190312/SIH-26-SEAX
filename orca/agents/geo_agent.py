"""Geospatial / PFZ / Route Agent: geofences (hard), PFZ candidates, distances, routes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..decision.route import ExposureField
from ..geo import bearing_deg, compass, distance_to_polygon_km, haversine_km
from ..schema import PFZZone
from .base import Agent


class GeoAgent(Agent):
    def __init__(self, ctx, trace, evidence):
        super().__init__("geo", ctx, trace, evidence)

    def geofence(self, lat: float, lon: float) -> dict[str, Any]:
        return self.call("check_geofence", latitude=lat, longitude=lon).data

    def pfz(self, date_iso: str):
        return self.call("get_pfz_candidates", date=date_iso).data

    def screen_zone(self, z: PFZZone) -> dict[str, Any]:
        """A PFZ advisory never overrides a geofence: a zone whose centre is inside, or whose
        radius overlaps, a binding geofence is RESTRICTED before any weather is looked at."""
        cons = self.ctx.constraints
        status = cons.status_at(z.latitude, z.longitude)
        overlaps = []
        for f in cons.binding:
            d = distance_to_polygon_km(z.latitude, z.longitude, f.polygon)
            if d < z.radius_km and not any(i["id"] == f.fence_id for i in status["inside"]):
                overlaps.append({"id": f.fence_id, "name": f.name, "kind": f.kind, "gap_km": round(d, 1)})
        return {"status": status, "overlaps": overlaps, "restricted": bool(status["inside"] or overlaps or status["on_land"])}

    def relation(self, origin: tuple[float, float], lat: float, lon: float) -> dict[str, Any]:
        b = bearing_deg(origin[0], origin[1], lat, lon)
        return {"distance_km": round(haversine_km(origin[0], origin[1], lat, lon), 1), "direction": compass(b), "bearing_deg": round(b)}

    def check_route(self, path, when) -> dict[str, Any]:
        res = self.call("check_route", waypoints=[list(p) for p in path], time=when.isoformat(timespec="seconds") if when else None)
        return res.data if res.ok else {"ok": False, "error": res.note}

    def hazards_near(self, lat: float, lon: float, radius_km: float, when) -> dict[str, Any] | None:
        res = self.call("hazards_near", latitude=lat, longitude=lon, radius_km=radius_km, time=when.isoformat(timespec="seconds"))
        return res.data if res.ok else None

    def routes(self, origin: tuple[float, float], zones: Sequence[Mapping[str, Any]], field: ExposureField | None) -> dict[str, Any]:
        res = self.call("plan_routes", origin={"lat": origin[0], "lon": origin[1]}, zones=list(zones), field=field)
        return res.data if res.ok else {"routes": [], "note": res.note or "route planning unavailable"}
