"""Direct tool endpoints (ocean / weather / geo) and structured route / compare requests.

These mirror SatQuery's ``/tools/v1/*`` idea: an assessor can call a specialist directly
and see the same observations an answer cited. They go through the same providers, tools
and fusion as a full query -- there is no second code path.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping

from satquery.agent.trace import Trace

from ..agents.base import compact
from ..config.settings import IST
from ..decision.evidence import fuse
from ..decision.geofence import ConstraintSet, load_geofences
from ..pipeline import Orca
from ..tools import ToolContext, run_tool


class BadRequest(ValueError):
    pass


def parse_time(v: Any, default: datetime | None = None) -> datetime:
    if v in (None, ""):
        if default is None:
            raise BadRequest("a time is required (ISO 8601, e.g. 2026-10-01T06:00:00+05:30; naive times are read as IST)")
        return default
    try:
        d = datetime.fromisoformat(str(v))
    except ValueError as exc:
        raise BadRequest(f"unreadable time {v!r}: {exc}") from None
    return d.replace(tzinfo=IST) if d.tzinfo is None else d.astimezone(IST)


def _num(body: Mapping[str, Any], key: str, lo: float, hi: float) -> float:
    try:
        v = float(body[key])
    except (KeyError, TypeError, ValueError):
        raise BadRequest(f"'{key}' is required and must be a number") from None
    if not lo <= v <= hi:
        raise BadRequest(f"'{key}' must be between {lo} and {hi}")
    return v


def _ctx(orca: Orca, body: Mapping[str, Any]):
    cfg = orca.cfg_for(body.get("mode"))
    prov = orca.provider(cfg)
    cons = ConstraintSet(load_geofences(prov.geofences()), prov.coast(), mpa_hard=cfg.mpa_hard, coverage=prov.geofence_coverage())
    return cfg, prov, ToolContext(prov, cfg, cons)


def _fused_view(fr) -> dict[str, Any]:
    return {v: {"status": fv.status, "selected": fv.selected.to_dict() if fv.selected else None,
                "candidates": [o.to_dict() for o in fv.candidates], "rejected": [o.to_dict() for o in fv.rejected],
                "conflict": fv.conflict.to_dict() if fv.conflict else None} for v, fv in fr.variables.items()}


def _point_tool(orca: Orca, tool: str, body: Mapping[str, Any]) -> dict[str, Any]:
    cfg, prov, ctx = _ctx(orca, body)
    lat, lon = _num(body, "latitude", -90, 90), _num(body, "longitude", -180, 180)
    start = parse_time(body.get("start_time"))
    end = parse_time(body.get("end_time"), start)
    if end < start:
        raise BadRequest("'end_time' is before 'start_time'")
    tr = Trace("", f"direct:{tool}")
    with tr.step("agent", tool, params={"latitude": lat, "longitude": lon, "start": start.isoformat(), "end": end.isoformat()}) as st:
        res = run_tool(tool, ctx, latitude=lat, longitude=lon, start_time=start.isoformat(), end_time=end.isoformat())
        st.output = compact(res)
        st.status = "ok" if res.ok else "error"
    by_time: dict[str, list] = {}
    for o in res.observations:
        by_time.setdefault(o.valid_time or "n/a", []).append(o)
    slots = []
    for t_, obs in sorted(by_time.items()):
        fr = fuse(obs, cfg)
        slots.append({"valid_time": t_, "fused": _fused_view(fr), "conflicts": [c.to_dict() for c in fr.conflicts],
                      "missing": list(fr.missing), "stale": list(fr.stale)})
    return {"mode": cfg.mode, "mode_label": prov.label(), "tool": tool, "ok": res.ok, "note": res.note,
            "observations": [o.to_dict() for o in res.observations], "slots": slots, "trace": tr.to_dict()}


def ocean(orca: Orca, body: Mapping[str, Any]) -> dict[str, Any]:
    return _point_tool(orca, "get_ocean_conditions", body)


def weather(orca: Orca, body: Mapping[str, Any]) -> dict[str, Any]:
    return _point_tool(orca, "get_marine_weather", body)


def geo(orca: Orca, body: Mapping[str, Any]) -> dict[str, Any]:
    cfg, prov, ctx = _ctx(orca, body)
    lat, lon = _num(body, "latitude", -90, 90), _num(body, "longitude", -180, 180)
    out: dict[str, Any] = {"mode": cfg.mode, "mode_label": prov.label(), "geofence_status": ctx.constraints.status_at(lat, lon),
                           "geofences": ctx.constraints.to_geojson(), "binding": [f.fence_id for f in ctx.constraints.binding],
                           "geofence_note": prov.geofence_origin()}
    date = body.get("date")
    if date:
        r = run_tool("get_pfz_candidates", ctx, date=str(date))
        out["pfz"] = r.data.to_dict() if r.data is not None else {"available": False, "note": r.note}
    return out


def route(orca: Orca, body: Mapping[str, Any]) -> dict[str, Any]:
    place = str(body.get("place") or "Nagapattinam")
    t = parse_time(body.get("time"), orca.cfg.local_midnight() + timedelta(days=1, hours=6))
    return orca.structured("route", place, [t.isoformat()], mode=body.get("mode"), session_id=str(body.get("session_id", "default")),
                           lang=str(body.get("lang", "en")), text=f"[route] {place} @ {t.strftime('%d %b %H:%M')}")


def parse_location(v: Any) -> dict[str, float] | None:
    if v in (None, ""):
        return None
    if not isinstance(v, Mapping):
        raise BadRequest("'location' must be an object {lat, lon}")
    return {"lat": _num(v, "lat", -90, 90), "lon": _num(v, "lon", -180, 180)}


def route_check(orca: Orca, body: Mapping[str, Any]) -> dict[str, Any]:
    wps = body.get("waypoints")
    if not isinstance(wps, list) or len(wps) < 2 or len(wps) > 60:
        raise BadRequest("'waypoints' must be a list of 2 to 60 [lat, lon] pairs")
    path = []
    for w in wps:
        try:
            la, lo = float(w[0]), float(w[1])
        except (TypeError, ValueError, IndexError, KeyError):
            raise BadRequest("each waypoint must be [lat, lon] numbers") from None
        if not (-90 <= la <= 90 and -180 <= lo <= 180):
            raise BadRequest("waypoint coordinates out of range")
        path.append([la, lo])
    t = parse_time(body.get("time"), orca.cfg.clock().replace(minute=0, second=0, microsecond=0))
    place = {"id": "user", "name": "your location", "lat": path[0][0], "lon": path[0][1], "sea_lat": path[0][0], "sea_lon": path[0][1], "matched": "first waypoint", "source": "waypoint"}
    return orca.structured("route_check", "route", [t.isoformat()], mode=body.get("mode"), session_id=str(body.get("session_id", "default")),
                           lang=str(body.get("lang", "en")), text=f"[route check] {len(path)} waypoints @ {t.strftime('%d %b %H:%M')}", extra={"route_waypoints": path}, place_dict=place)


def compare(orca: Orca, body: Mapping[str, Any]) -> dict[str, Any]:
    place = str(body.get("place") or "Nagapattinam")
    raw = body.get("times")
    if not isinstance(raw, list) or len(raw) < 2:
        raise BadRequest("'times' must be a list of at least two ISO times")
    ts = [parse_time(x) for x in raw[:3]]
    return orca.structured("compare", place, [t.isoformat() for t in ts], mode=body.get("mode"), session_id=str(body.get("session_id", "default")),
                           lang=str(body.get("lang", "en")), text=f"[compare] {place} @ " + " vs ".join(t.strftime("%d %b %H:%M") for t in ts))
