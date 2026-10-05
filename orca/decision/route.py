"""Route engine: deterministic, constraint-first, no route is ever called "safest".

Three objectives, each with its criteria written out:

``shortest_distance``  minimise length. Direct great-circle leg if it is legal, otherwise
                       the shortest legal grid path around hard constraints.
``lower_exposure``     minimise  length x (1 + a*e^2)  with a=8, where e is the *marine
                       exposure index* (below) interpolated at each point at departure time.
``pfz_oriented``       same cost with a=2 (mild exposure aversion) to the best-supported PFZ
                       candidate rather than the nearest one.

Marine exposure index  e = max(wave_height / wave_higher_threshold,
                               wind_speed  / wind_higher_threshold)
so e = 1.0 means "at the HIGHER_RISK threshold". It is an ORCA-defined, transparent index --
not a physical quantity and not a probability. The field is an inverse-distance
interpolation of the selected observations; it is evaluated at the departure hour only
(a documented simplification: the route is not time-stepped).

Legality is enforced twice: the grid search never uses a blocked node/edge, and every
finished route is re-verified with :meth:`ConstraintSet.verify_path`. A route that fails
the second check is dropped, never shown.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..config import OrcaConfig
from ..geo import Point, bearing_deg, compass, densify, haversine_km, idw, polyline_length_km
from .geofence import ConstraintSet

__all__ = ["ExposureField", "plan_route", "plan_route_set", "OBJECTIVES"]

STEP_DEG = 0.02
MARGIN_DEG = 0.35
_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

OBJECTIVES: Mapping[str, dict[str, Any]] = {
    "shortest_distance": {"alpha": 0.0, "criteria": "minimum distance; hard geofences and land excluded"},
    "lower_exposure": {"alpha": 8.0, "criteria": "minimum distance x (1 + 8 x exposure^2); exposure = max(wave/higher-wave threshold, wind/higher-wind threshold) at departure time; hard geofences and land excluded"},
    "pfz_oriented": {"alpha": 2.0, "criteria": "goes to the best-supported PFZ candidate; distance x (1 + 2 x exposure^2); hard geofences and land excluded"},
}


class ExposureField:
    """Interpolated wave / wind / exposure from selected point observations."""

    def __init__(self, waves: Sequence[tuple[float, float, float]], winds: Sequence[tuple[float, float, float]], cfg: OrcaConfig) -> None:
        self._w, self._v = list(waves), list(winds)
        self._hw, self._hv = cfg.thresholds.wave_higher_m, cfg.thresholds.wind_higher_kmh

    @property
    def available(self) -> bool:
        return bool(self._w) and bool(self._v)

    def wave(self, lat: float, lon: float) -> float | None:
        return idw(lat, lon, self._w)

    def wind(self, lat: float, lon: float) -> float | None:
        return idw(lat, lon, self._v)

    def exposure(self, lat: float, lon: float) -> float:
        w, v = self.wave(lat, lon), self.wind(lat, lon)
        parts = [x for x in (None if w is None else w / self._hw, None if v is None else v / self._hv) if x is not None]
        return max(parts) if parts else 0.0


def _seg_cost(a: Point, b: Point, alpha: float, field: ExposureField | None) -> float:
    d = haversine_km(a[0], a[1], b[0], b[1])
    if alpha == 0 or field is None or not field.available:
        return d
    pts = densify(a, b, 2.0)
    e = sum(field.exposure(p[0], p[1]) for p in pts) / len(pts)
    return d * (1.0 + alpha * e * e)


def _grid_path(origin: Point, dest: Point, cons: ConstraintSet, alpha: float, field: ExposureField | None) -> list[Point] | None:
    lat_lo = min(origin[0], dest[0]) - MARGIN_DEG
    lat_hi = max(origin[0], dest[0]) + MARGIN_DEG
    lon_lo = min(origin[1], dest[1]) - MARGIN_DEG
    lon_hi = max(origin[1], dest[1]) + MARGIN_DEG
    ni = int((lat_hi - lat_lo) / STEP_DEG) + 1
    nj = int((lon_hi - lon_lo) / STEP_DEG) + 1
    pos = lambda i, j: (lat_lo + i * STEP_DEG, lon_lo + j * STEP_DEG)  # noqa: E731
    node_ok: dict[tuple[int, int], bool] = {}

    def ok(i: int, j: int) -> bool:
        if not (0 <= i < ni and 0 <= j < nj):
            return False
        if (i, j) not in node_ok:
            p = pos(i, j)
            node_ok[(i, j)] = not cons.point_violations(*p) and not cons.on_land(*p)
        return node_ok[(i, j)]

    def snap(p: Point) -> tuple[int, int] | None:
        ci, cj = round((p[0] - lat_lo) / STEP_DEG), round((p[1] - lon_lo) / STEP_DEG)
        best = None
        for di in range(-3, 4):
            for dj in range(-3, 4):
                if ok(ci + di, cj + dj):
                    d = haversine_km(p[0], p[1], *pos(ci + di, cj + dj))
                    if best is None or d < best[0]:
                        best = (d, (ci + di, cj + dj))
        return best[1] if best else None

    s, g = snap(origin), snap(dest)
    if s is None or g is None:
        return None
    dist: dict[tuple[int, int], float] = {s: 0.0}
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    heap: list[tuple[float, tuple[int, int]]] = [(0.0, s)]
    edge_ok: dict[tuple[tuple[int, int], tuple[int, int]], bool] = {}
    while heap:
        d, u = heapq.heappop(heap)
        if u == g:
            break
        if d > dist.get(u, math.inf):
            continue
        for di, dj in _NEIGHBOURS:
            v = (u[0] + di, u[1] + dj)
            if not ok(*v):
                continue
            key = (u, v) if u < v else (v, u)
            if key not in edge_ok:
                edge_ok[key] = not cons.blocked_segment(pos(*u), pos(*v))
            if not edge_ok[key]:
                continue
            nd = d + _seg_cost(pos(*u), pos(*v), alpha, field)
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if g not in dist:
        return None
    nodes = [g]
    while nodes[-1] != s:
        nodes.append(prev[nodes[-1]])
    nodes.reverse()
    path = [origin] + [pos(*n) for n in nodes] + [dest]
    return _smooth(path, cons, alpha, field)


def _smooth(path: list[Point], cons: ConstraintSet, alpha: float, field: ExposureField | None) -> list[Point]:
    """Greedy line-of-sight shortcutting, accepted only when legal *and* not costlier."""
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        best = i + 1
        acc = _seg_cost(path[i], path[i + 1], alpha, field)
        run = acc
        for j in range(i + 2, len(path)):
            run += _seg_cost(path[j - 1], path[j], alpha, field)
            if not cons.blocked_segment(path[i], path[j]) and _seg_cost(path[i], path[j], alpha, field) <= run + 1e-9:
                best = j
        out.append(path[best])
        i = best
    return out


def _metrics(path: Sequence[Point], field: ExposureField | None, cfg: OrcaConfig) -> dict[str, Any]:
    dist = polyline_length_km(path)
    m: dict[str, Any] = {"distance_km": round(dist, 1), "duration_h": round(dist / cfg.vessel_speed_kmh, 1)}
    if field is None or not field.available:
        m.update(max_exposure=None, mean_exposure=None, max_wave_m=None, max_wind_kmh=None, exposure_data=False)
        return m
    samples: list[Point] = []
    for a, b in zip(path, path[1:]):
        samples.extend(densify(a, b, 2.0))
    ex = [field.exposure(*p) for p in samples]
    m.update(max_exposure=round(max(ex), 2), mean_exposure=round(sum(ex) / len(ex), 2),
             max_wave_m=round(max(field.wave(*p) or 0 for p in samples), 2),
             max_wind_kmh=round(max(field.wind(*p) or 0 for p in samples), 1), exposure_data=True)
    return m


def plan_route(origin: Point, dest: Point, cons: ConstraintSet, field: ExposureField | None, cfg: OrcaConfig, *,
               objective: str, dest_name: str = "", dest_id: str = "") -> dict[str, Any]:
    spec = OBJECTIVES[objective]
    alpha = spec["alpha"]
    direct_block = cons.blocked_segment(origin, dest)
    direct_km = haversine_km(origin[0], origin[1], dest[0], dest[1])
    if objective == "shortest_distance" and not direct_block:
        path, ptype = [origin, dest], "direct_great_circle"
    else:
        path = _grid_path(origin, dest, cons, alpha, field)
        ptype = "constraint_aware_grid_search"
        if path is None:
            return {"objective": objective, "destination": {"id": dest_id, "name": dest_name}, "feasible": False,
                    "criteria": spec["criteria"], "reason": "no legal path exists between origin and destination"}
    check = cons.verify_path(path)
    if not check["ok"]:  # never show an illegal route
        return {"objective": objective, "destination": {"id": dest_id, "name": dest_name}, "feasible": False,
                "criteria": spec["criteria"], "reason": f"rejected by post-verification: {check['violations']}"}
    m = _metrics(path, field, cfg)
    r: dict[str, Any] = {
        "objective": objective, "feasible": True, "criteria": spec["criteria"],
        "destination": {"id": dest_id, "name": dest_name, "lat": dest[0], "lon": dest[1]},
        "path": [[round(p[0], 4), round(p[1], 4)] for p in path], "path_type": ptype,
        "direct_blocked_by": direct_block, "direct_distance_km": round(direct_km, 1),
        "detour_km": round(m["distance_km"] - direct_km, 1),
        "initial_bearing": compass(bearing_deg(origin[0], origin[1], path[1][0], path[1][1])),
        "geofence_check": {"ok": True, "checked_legs": len(path) - 1, "binding_geofences": [f.fence_id for f in cons.binding]},
        **m,
    }
    return r


def plan_route_set(origin: Point, zones: Sequence[Mapping[str, Any]], cons: ConstraintSet, field: ExposureField | None,
                   cfg: OrcaConfig) -> dict[str, Any]:
    """``zones``: dicts with id, name, lat, lon, support (0..3), state, restricted (bool). Picks
    destinations by stated rules and returns the three routes plus how each was chosen."""
    legal = [z for z in zones if not z.get("restricted") and z.get("state") not in ("RESTRICTED", "HIGHER_RISK")]
    if not legal:
        return {"routes": [], "note": "no PFZ candidate is currently legal and below HIGHER_RISK, so no route is offered",
                "excluded": [{"id": z["id"], "state": z.get("state"), "restricted": z.get("restricted")} for z in zones]}
    nearest = min(legal, key=lambda z: haversine_km(origin[0], origin[1], z["lat"], z["lon"]))
    best = sorted(legal, key=lambda z: (-z.get("support", 0), _state_rank(z.get("state")),
                                        haversine_km(origin[0], origin[1], z["lat"], z["lon"])))[0]
    routes = [
        {**plan_route(origin, (nearest["lat"], nearest["lon"]), cons, field, cfg, objective="shortest_distance", dest_name=nearest["name"], dest_id=nearest["id"]), "role": "shortest"},
        {**plan_route(origin, (nearest["lat"], nearest["lon"]), cons, field, cfg, objective="lower_exposure", dest_name=nearest["name"], dest_id=nearest["id"]), "role": "lower_exposure"},
        {**plan_route(origin, (best["lat"], best["lon"]), cons, field, cfg, objective="pfz_oriented", dest_name=best["name"], dest_id=best["id"]), "role": "pfz_oriented"},
    ]
    others = [z for z in legal if z["id"] not in (nearest["id"], best["id"])]
    if others:  # a genuinely different second option: next-best legal PFZ by the same ranking, lower-exposure criteria
        alt = sorted(others, key=lambda z: (-z.get("support", 0), _state_rank(z.get("state")), haversine_km(origin[0], origin[1], z["lat"], z["lon"])))[0]
        routes.append({**plan_route(origin, (alt["lat"], alt["lon"]), cons, field, cfg, objective="lower_exposure", dest_name=alt["name"], dest_id=alt["id"]), "role": "alternative"})
    for i, r in enumerate(routes):  # say so when two routes are the same path rather than pretending they differ
        if r.get("feasible"):
            r["identical_to"] = next((q["role"] for q in routes[:i] if q.get("feasible") and q["path"] == r["path"]), None)
    return {
        "routes": routes,
        "destination_rules": {
            "shortest_distance/lower_exposure": f"nearest legal PFZ candidate below HIGHER_RISK -> {nearest['id']}",
            "pfz_oriented": f"most ocean-support criteria met, ties by lower risk state then distance -> {best['id']}",
        },
        "note": "No route is labelled 'safest': each is optimal only for the criteria stated on it.",
    }


def _state_rank(s: str | None) -> int:
    return {"LOWER_RISK": 0, "MODERATE_CONDITIONS": 1, "INSUFFICIENT_DATA": 2}.get(s or "", 3)
