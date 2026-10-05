"""Tool contracts + registry + capability-pruned menu.

Reuses SatQuery's design: every tool is a :class:`ToolSpec` (name, summary, JSON-schema
parameters, a ``requires`` token checked against what the active data provider can
actually serve). :func:`build_menu` prunes the menu *before* any agent runs and records
what was withheld and why -- so "the agent chose not to" and "the agent could not" stay
distinguishable in the trace (SatQuery's invariant 02, carried over).

Tools are deterministic functions of (provider, constraints, parameters). They return
schema Observations or plain geometry results -- never prose, never an LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Sequence

from ..config import OrcaConfig
from ..config.settings import IST
from ..data.providers import BundleProvider
from ..decision.geofence import ConstraintSet
from ..geo import bearing_deg, compass, haversine_km
from ..schema import Observation
from satquery.agent.trace import jsonable

__all__ = ["ToolSpec", "ToolContext", "ToolResult", "TOOLS", "build_menu", "run_tool"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    agent: str                      # ocean | weather | geo
    summary: str
    requires: tuple[str, ...]       # provider capability tokens
    parameters: Mapping[str, Any] = field(default_factory=dict)
    required_params: tuple[str, ...] = ()

    def schema(self) -> dict[str, Any]:
        """Function-calling schema (same shape SatQuery hands its router)."""
        return {"type": "function", "function": {"name": self.name, "description": self.summary,
                "parameters": {"type": "object", "properties": dict(self.parameters), "required": list(self.required_params)}}}


_LL = {"latitude": {"type": "number"}, "longitude": {"type": "number"},
       "start_time": {"type": "string", "format": "date-time"}, "end_time": {"type": "string", "format": "date-time"}}

TOOLS: dict[str, ToolSpec] = {t.name: t for t in (
    ToolSpec("get_ocean_conditions", "ocean", "SST, chlorophyll-a, SST gradient, current speed and tide at a point/time.",
             ("ocean",), _LL, ("latitude", "longitude", "start_time", "end_time")),
    ToolSpec("get_marine_weather", "weather", "Wave height, wind, precipitation and lightning flag at a point/time.",
             ("weather",), _LL, ("latitude", "longitude", "start_time", "end_time")),
    ToolSpec("get_advisories", "weather", "Active marine advisories and cyclone status (official feed).",
             ("advisory",), {"start_time": _LL["start_time"], "end_time": _LL["end_time"]}, ("start_time", "end_time")),
    ToolSpec("get_pfz_candidates", "geo", "Potential Fishing Zone candidates from the PFZ advisory for a date.",
             ("pfz",), {"date": {"type": "string", "format": "date"}}, ("date",)),
    ToolSpec("check_geofence", "geo", "Hard-constraint status of a point: inside/outside, nearest restricted zone.",
             ("geofence",), {"latitude": _LL["latitude"], "longitude": _LL["longitude"]}, ("latitude", "longitude")),
    ToolSpec("check_route", "geo", "Intersect a route with every geofence, land and active advisory area.",
             ("geofence",), {"waypoints": {"type": "array"}, "time": {"type": "string"}}, ("waypoints",)),
    ToolSpec("hazards_near", "geo", "Active advisories, geofences and threshold breaches within a radius of a point.",
             ("geofence", "weather"), {**{k: _LL[k] for k in ("latitude", "longitude")}, "radius_km": {"type": "number"}, "time": {"type": "string"}},
             ("latitude", "longitude", "radius_km", "time")),
    ToolSpec("plan_routes", "geo", "Shortest / lower-exposure / PFZ-oriented routes that respect hard geofences.",
             ("geofence",), {"origin": {"type": "object"}, "zones": {"type": "array"}, "time": {"type": "string"}}, ("origin", "zones")),
)}


@dataclass
class ToolContext:
    provider: BundleProvider
    cfg: OrcaConfig
    constraints: ConstraintSet


@dataclass
class ToolResult:
    name: str
    params: dict[str, Any]
    ok: bool
    observations: list[Observation] = field(default_factory=list)
    data: Any = None
    note: str = ""

    def summary(self) -> dict[str, Any]:
        """JSON-safe digest for the trace."""
        return {"ok": self.ok, "n_observations": len(self.observations), "note": self.note,
                "observations": [o.to_dict() for o in self.observations],
                "data": jsonable(self.data) if self.data is not None else None}


def build_menu(capabilities: Mapping[str, bool], wanted: Sequence[str] | None = None) -> dict[str, Any]:
    """Split tools into available / withheld (with reasons) given provider capabilities."""
    names = list(wanted) if wanted is not None else list(TOOLS)
    avail: list[str] = []
    withheld: list[dict[str, str]] = []
    for n in names:
        spec = TOOLS[n]
        lacking = [r for r in spec.requires if not capabilities.get(r)]
        if lacking:
            withheld.append({"tool": n, "reason": f"data source not available in this mode: {', '.join(lacking)}"})
        else:
            avail.append(n)
    return {"available": avail, "withheld": withheld}


def _slots(start: str, end: str, limit: int = 12) -> list[datetime]:
    a = datetime.fromisoformat(start).astimezone(IST).replace(minute=0, second=0, microsecond=0)
    b = datetime.fromisoformat(end).astimezone(IST).replace(minute=0, second=0, microsecond=0)
    out = [a]
    while out[-1] < b and len(out) < limit:
        out.append(out[-1] + timedelta(hours=1))
    return out


def run_tool(name: str, ctx: ToolContext, **p: Any) -> ToolResult:
    spec = TOOLS.get(name)
    if spec is None:
        return ToolResult(name, p, False, note="unknown tool")
    missing = [r for r in spec.requires if not ctx.provider.capabilities().get(r)]
    if missing:
        return ToolResult(name, p, False, note=f"capability unavailable: {', '.join(missing)}")
    try:
        if name == "get_marine_weather":
            obs = [o for t in _slots(p["start_time"], p["end_time"]) for o in ctx.provider.weather(p["latitude"], p["longitude"], t)]
            return ToolResult(name, p, True, obs)
        if name == "get_ocean_conditions":
            obs = [o for t in _slots(p["start_time"], p["end_time"]) for o in ctx.provider.ocean(p["latitude"], p["longitude"], t)]
            return ToolResult(name, p, True, obs)
        if name == "get_advisories":
            feed = ctx.provider.advisories()
            return ToolResult(name, p, feed.available, data=feed, note="" if feed.available else feed.note)
        if name == "get_pfz_candidates":
            res = ctx.provider.pfz(p["date"])
            return ToolResult(name, p, res.available, data=res, note=res.note)
        if name == "check_geofence":
            return ToolResult(name, p, True, data=ctx.constraints.status_at(p["latitude"], p["longitude"]))
        if name == "check_route":
            from ..decision.hazards import check_route
            feed = ctx.provider.advisories() if ctx.provider.capabilities()["advisory"] else None
            when = datetime.fromisoformat(p["time"]).astimezone(IST) if p.get("time") else None
            return ToolResult(name, p, True, data=check_route(ctx.constraints, feed, when, [tuple(x) for x in p["waypoints"]]))
        if name == "hazards_near":
            from ..decision.hazards import hazards_near
            return ToolResult(name, p, True, data=hazards_near(ctx.provider, ctx.constraints, ctx.cfg, p["latitude"], p["longitude"],
                                                              p["radius_km"], datetime.fromisoformat(p["time"]).astimezone(IST)))
        if name == "plan_routes":
            from ..decision.route import plan_route_set
            o = p["origin"]
            out = plan_route_set((o["lat"], o["lon"]), p["zones"], ctx.constraints, p.get("field"), ctx.cfg)
            return ToolResult(name, {k: v for k, v in p.items() if k != "field"}, True, data=out)
    except Exception as exc:  # a tool that fails leaves a receipt, never a crash
        return ToolResult(name, {k: v for k, v in p.items() if k != "field"}, False, note=f"{type(exc).__name__}: {exc}")
    return ToolResult(name, p, False, note="not implemented")


def place_relation(origin: tuple[float, float], lat: float, lon: float) -> dict[str, Any]:
    """Deterministic distance/bearing from the origin to a point (used for zone listings)."""
    b = bearing_deg(origin[0], origin[1], lat, lon)
    return {"distance_km": round(haversine_km(origin[0], origin[1], lat, lon), 1), "bearing_deg": round(b), "direction": compass(b)}
