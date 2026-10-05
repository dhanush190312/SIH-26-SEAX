"""Deterministic hazard-proximity and route-restriction checks (blueprint sections 13, 18, 19).

Pure functions over the provider, the constraint set and the thresholds -- no LLM, no randomness.
Official advisory polygons stay labelled as official; threshold breaches found here are
SEAX-computed interpretations of source data, and are returned in a separate list so the two can
never be conflated downstream.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from ..config import OrcaConfig
from ..geo import Point, bearing_deg, compass, distance_to_polygon_km, haversine_km, polyline_length_km, segment_intersects_polygon
from .evidence import fuse
from .geofence import ConstraintSet


def _active(a, when: datetime) -> bool:
    return datetime.fromisoformat(a.valid_from) <= when <= datetime.fromisoformat(a.valid_to)


def hazards_near(provider, cons: ConstraintSet, cfg: OrcaConfig, lat: float, lon: float, radius_km: float, when: datetime) -> dict[str, Any]:
    """What is within ``radius_km`` of (lat, lon) at ``when``?

    * official advisories whose polygon is within the radius AND valid at ``when`` (expired / not-yet-valid
      ones are listed separately, with the reason, and never counted as hazards);
    * geofences within the radius (hard ones flagged);
    * weather-grid points within the radius that breach a SEAX threshold (computed, not official).
    """
    th = cfg.thresholds
    feed = provider.advisories() if provider.capabilities()["advisory"] else None
    official, inactive = [], []
    if feed is not None and feed.available:
        for a in feed.items:
            d = distance_to_polygon_km(lat, lon, a.polygon)
            if d > radius_km:
                continue
            rec = {"advisory_id": a.advisory_id, "kind": a.kind, "severity": a.severity, "text": a.text, "distance_km": round(d, 1),
                   "valid_from": a.valid_from, "valid_to": a.valid_to, "source_label": a.source_label, "issued_at": feed.issued_at}
            if _active(a, when):
                official.append(rec)
            else:
                rec["why_inactive"] = "expired before the requested time" if datetime.fromisoformat(a.valid_to) < when else "not yet in force at the requested time"
                inactive.append(rec)
    fences = []
    for f in cons.fences:
        d = distance_to_polygon_km(lat, lon, f.polygon)
        if d <= radius_km:
            cx, cy = sum(p[0] for p in f.polygon) / len(f.polygon), sum(p[1] for p in f.polygon) / len(f.polygon)
            fences.append({"id": f.fence_id, "name": f.name, "kind": f.kind, "hard": cons.is_binding(f), "distance_km": round(d, 1),
                           "direction": compass(bearing_deg(lat, lon, cx, cy))})
    computed = []
    for p in provider.weather_points():
        if haversine_km(lat, lon, p["lat"], p["lon"]) > radius_km:
            continue
        fr = fuse(provider.weather(p["lat"], p["lon"], when), cfg)
        for var, unit, mod, hi in (("wave_height", "m", th.wave_moderate_m, th.wave_higher_m), ("wind_speed", "km/h", th.wind_moderate_kmh, th.wind_higher_kmh)):
            o = fr.observation(var)
            if o is not None and o.value is not None and o.value >= mod:
                level = "HIGHER_RISK" if o.value >= hi else "MODERATE_CONDITIONS"
                computed.append({"point": p["id"], "lat": p["lat"], "lon": p["lon"], "variable": var, "value": o.value, "unit": unit,
                                 "level": level, "threshold": hi if level == "HIGHER_RISK" else mod, "obs_id": o.obs_id,
                                 "source_label": o.source_label, "distance_km": round(haversine_km(lat, lon, p["lat"], p["lon"]), 1)})
    computed.sort(key=lambda c: (c["level"] != "HIGHER_RISK", -c["value"]))
    return {"center": [lat, lon], "radius_km": radius_km, "time": when.isoformat(timespec="minutes"),
            "official_advisories": official, "inactive_advisories": inactive, "geofences": fences, "computed_breaches": computed[:6],
            "n_breaches_total": len(computed), "any": bool(official or computed or any(f["hard"] for f in fences))}


def check_route(cons: ConstraintSet, feed, when: datetime | None, path: Sequence[Point]) -> dict[str, Any]:
    """Intersect every leg with every geofence, land, and (if valid at ``when``) every advisory polygon."""
    legs = list(zip(path, path[1:]))
    crossings, soft = [], []
    for f in cons.fences:
        idx = [i for i, (a, b) in enumerate(legs) if segment_intersects_polygon(a, b, f.polygon)]
        if idx:
            rec = {"id": f.fence_id, "name": f.name, "kind": f.kind, "legs": idx, "hard": cons.is_binding(f)}
            (crossings if rec["hard"] else soft).append(rec)
    land = [i for i, (a, b) in enumerate(legs) if cons.segment_hits_land(a, b)]
    adv = []
    if feed is not None and feed.available and when is not None:
        for a in feed.items:
            if _active(a, when):
                idx = [i for i, (p, q) in enumerate(legs) if segment_intersects_polygon(p, q, a.polygon)]
                if idx:
                    adv.append({"advisory_id": a.advisory_id, "severity": a.severity, "text": a.text, "legs": idx, "source_label": a.source_label,
                                "issued_at": feed.issued_at, "valid_from": a.valid_from, "valid_to": a.valid_to})
    return {"n_legs": len(legs), "n_fences": len(cons.fences), "distance_km": round(polyline_length_km(path), 1),
            "hard_crossings": crossings, "soft_crossings": soft, "land_legs": land, "advisories": adv,
            "ok": not crossings and not land, "path": [[round(p[0], 4), round(p[1], 4)] for p in path]}
