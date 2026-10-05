"""Weather & Hazard Agent: wind, waves, precipitation, lightning, advisories, cyclone status."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from ..decision.evidence import FusionResult, fuse
from ..schema import AdvisoryFeed, Observation
from .base import Agent


class WeatherAgent(Agent):
    def __init__(self, ctx, trace, evidence):
        super().__init__("weather", ctx, trace, evidence)

    def marine_weather(self, lat: float, lon: float, when: datetime) -> list[Observation]:
        iso = when.isoformat(timespec="seconds")
        return self.call("get_marine_weather", latitude=lat, longitude=lon, start_time=iso, end_time=iso).observations

    def advisories(self, start: datetime, end: datetime) -> AdvisoryFeed | None:
        res = self.call("get_advisories", start_time=start.isoformat(timespec="seconds"), end_time=end.isoformat(timespec="seconds"))
        return res.data if res.ok else None

    def scan_hazards(self, times: Sequence[datetime]) -> dict[str, Any]:
        """Deterministic sweep of every weather grid point at the given times: which breach
        a threshold? (No tool call per point: one recorded sweep with the full list.)"""
        th = self.ctx.cfg.thresholds
        cfg = self.ctx.cfg
        rows: list[dict[str, Any]] = []
        with self.trace.step("agent", "weather.scan_hazards", params={"times": [t.isoformat(timespec="minutes") for t in times]}) as st:
            for p in self.ctx.provider.weather_points():
                for t in times:
                    obs = self.ctx.provider.weather(p["lat"], p["lon"], t)
                    fr = fuse(obs, cfg)
                    for o in obs:
                        self.evidence[o.obs_id] = o.to_dict()
                    wave, wind = fr.observation("wave_height"), fr.observation("wind_speed")
                    lt = fr.observation("lightning")
                    rows.append({"point": p["id"], "lat": p["lat"], "lon": p["lon"], "time": t.isoformat(timespec="minutes"),
                                 "wave_height": wave.value if wave else None, "wind_speed": wind.value if wind else None,
                                 "lightning": lt.value if lt else None,
                                 "wave_obs": wave.obs_id if wave else None, "wind_obs": wind.obs_id if wind else None,
                                 "conflict": [c.variable for c in fr.conflicts]})
            breaches = []
            for r in rows:
                if r["wave_height"] is not None and r["wave_height"] >= th.wave_moderate_m:
                    breaches.append({**r, "variable": "wave_height", "value": r["wave_height"], "unit": "m",
                                     "level": "HIGHER_RISK" if r["wave_height"] >= th.wave_higher_m else "MODERATE_CONDITIONS",
                                     "threshold": th.wave_higher_m if r["wave_height"] >= th.wave_higher_m else th.wave_moderate_m})
                if r["wind_speed"] is not None and r["wind_speed"] >= th.wind_moderate_kmh:
                    breaches.append({**r, "variable": "wind_speed", "value": r["wind_speed"], "unit": "km/h",
                                     "level": "HIGHER_RISK" if r["wind_speed"] >= th.wind_higher_kmh else "MODERATE_CONDITIONS",
                                     "threshold": th.wind_higher_kmh if r["wind_speed"] >= th.wind_higher_kmh else th.wind_moderate_kmh})
                if r["lightning"] is not None and r["lightning"] >= 1:
                    breaches.append({**r, "variable": "lightning", "value": 1, "unit": "flag", "level": "HIGHER_RISK", "threshold": 1})
            st.output = {"ok": True, "n_points": len(self.ctx.provider.weather_points()), "n_rows": len(rows), "n_breaches": len(breaches)}
        return {"rows": rows, "breaches": breaches}

    def grid(self, when: datetime) -> list[dict[str, Any]]:
        """Fused wave/wind at every weather grid point at ``when`` (one recorded sweep).
        Feeds the route exposure field, the map's hazard layer and the avoid-areas list."""
        cfg, th = self.ctx.cfg, self.ctx.cfg.thresholds
        out: list[dict[str, Any]] = []
        with self.trace.step("agent", "weather.grid", params={"time": when.isoformat(timespec="minutes")}) as st:
            for p in self.ctx.provider.weather_points():
                obs = self.ctx.provider.weather(p["lat"], p["lon"], when)
                fr = fuse(obs, cfg)
                for o in obs:
                    self.evidence[o.obs_id] = o.to_dict()
                w, v, lt = fr.observation("wave_height"), fr.observation("wind_speed"), fr.observation("lightning")
                wave = None if w is None else w.value
                wind = None if v is None else v.value
                level = "INSUFFICIENT_DATA"
                if wave is not None and wind is not None:
                    level = "LOWER_RISK"
                    if wave >= th.wave_moderate_m or wind >= th.wind_moderate_kmh:
                        level = "MODERATE_CONDITIONS"
                    if wave >= th.wave_higher_m or wind >= th.wind_higher_kmh or (lt is not None and lt.value and lt.value >= 1):
                        level = "HIGHER_RISK"
                out.append({"id": p["id"], "lat": p["lat"], "lon": p["lon"], "wave_height": wave, "wind_speed": wind,
                            "lightning": None if lt is None else lt.value, "level": level,
                            "conflict": [c.variable for c in fr.conflicts],
                            "wave_obs": None if w is None else w.obs_id, "wind_obs": None if v is None else v.obs_id})
            st.output = {"ok": True, "n_points": len(out), "levels": {lv: sum(1 for g in out if g["level"] == lv) for lv in sorted({g["level"] for g in out})}}
        return out
