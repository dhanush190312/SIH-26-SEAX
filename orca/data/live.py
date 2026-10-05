"""LIVE adapters -- documented public endpoints only.

**Status: written against the public Open-Meteo API documentation and parse-tested on a
fixture of that shape, but NOT exercised against the live service in the build
environment (it had no network).** First-run verification is item 1 in the README's
"Known limitations".

What LIVE can and cannot provide (see docs/DATA_SOURCES.md):

* wave height, wind speed, precipitation -- Open-Meteo Marine + Forecast APIs (free,
  no key; an *open model* forecast, source family ``open_model``). The API does not
  report the model-run time, so freshness is measured from retrieval and every
  observation says so in its ``note``.
* INCOIS PFZ, IMD advisories, MOSDAC/ISRO ocean colour: **no documented public JSON API
  is used, and none is invented.** They are reported unavailable unless the operator
  drops an official copy into ``orca/data/imports/{pfz,advisories}.json`` (same schema
  as the demo documents, absolute timestamps). Decisions that need them then say
  INSUFFICIENT_DATA instead of guessing.
* Currents, chlorophyll, lightning: unavailable in LIVE.

Nothing here ever falls back to DEMO data.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from ..config import OrcaConfig
from ..config.settings import IST

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT_S = 12
_UA = "ORCA-prototype/0.1 (SIH26176; decision-support; non-commercial)"


def _get_json(url: str, params: Mapping[str, Any]) -> Any:
    q = urllib.parse.urlencode(params, safe=",")
    req = urllib.request.Request(f"{url}?{q}", headers={"User-Agent": _UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310 - fixed https hosts above
        return json.loads(resp.read().decode("utf-8"))


def _as_list(payload: Any) -> list[dict[str, Any]]:
    return payload if isinstance(payload, list) else [payload]


def parse_open_meteo(marine: Any, forecast: Any, points: list[tuple[float, float]]) -> tuple[list[dict[str, Any]], datetime, int]:
    """Merge marine + forecast payloads into ORCA weather points. Pure function (tested)."""
    m, f = _as_list(marine), _as_list(forecast)
    if len(m) != len(points) or len(f) != len(points):
        raise ValueError("Open-Meteo returned a different number of locations than requested")
    out: list[dict[str, Any]] = []
    anchor: datetime | None = None
    hours = 0
    for i, (lat, lon) in enumerate(points):
        mh, fh = m[i].get("hourly", {}), f[i].get("hourly", {})
        times = mh.get("time") or []
        if not times or times != (fh.get("time") or []):
            raise ValueError("marine and forecast hourly time axes differ")
        t0 = datetime.fromisoformat(times[0]).replace(tzinfo=IST)
        if anchor is None:
            anchor, hours = t0, len(times)
        elif t0 != anchor or len(times) != hours:
            raise ValueError("grid points have different time axes")
        series = {
            "wave_height": mh.get("wave_height"),
            "wind_speed": fh.get("wind_speed_10m"),
            "precipitation": fh.get("precipitation"),
        }
        if any(v is None for v in series.values()):
            raise ValueError("an expected hourly variable is missing from the response")
        out.append({"id": f"L{i}", "lat": float(m[i].get("latitude", lat)), "lon": float(m[i].get("longitude", lon)),
                    "series": {"open_model": series}})
    assert anchor is not None
    return out, anchor, hours


def build_live_docs(cfg: OrcaConfig, focus: tuple[float, float], gazetteer: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    now = cfg.clock()
    docs: dict[str, Any] = {"gazetteer": dict(gazetteer)}
    demo = cfg.data_dir / "demo"
    for name in ("geofences", "coast"):  # static layers: illustrative shapes, labelled as such in their meta
        docs[name] = json.loads((demo / f"{name}.json").read_text(encoding="utf-8"))

    pts = [(round(focus[0] + dy, 3), round(focus[1] + dx, 3)) for dy in (-0.3, 0.0, 0.3) for dx in (0.03, 0.25, 0.5)]
    lat_s, lon_s = ",".join(str(p[0]) for p in pts), ",".join(str(p[1]) for p in pts)
    try:
        marine = _get_json(MARINE_URL, {"latitude": lat_s, "longitude": lon_s, "hourly": "wave_height",
                                        "timezone": "Asia/Kolkata", "forecast_days": 3})
        fore = _get_json(FORECAST_URL, {"latitude": lat_s, "longitude": lon_s, "hourly": "wind_speed_10m,precipitation",
                                        "wind_speed_unit": "kmh", "timezone": "Asia/Kolkata", "forecast_days": 3})
        wpoints, anchor, hours = parse_open_meteo(marine, fore, pts)
        docs["weather"] = {
            "meta": {"kind": "weather", "anchor": anchor.isoformat(timespec="seconds"), "hours": hours,
                     "origin": "Open-Meteo Marine + Forecast APIs (open model forecast)"},
            "sources": {"open_model": {"id": "open-meteo", "label": "Open-Meteo model forecast (live)",
                                       "issued_at": now.isoformat(timespec="seconds"),
                                       "note": "model-run time not provided by the API; freshness measured from retrieval"}},
            "points": wpoints,
        }
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
        errors.append(f"Open-Meteo unavailable: {type(exc).__name__}: {exc}")

    imports = cfg.data_dir / "imports"
    for key, fname in (("pfz", "pfz.json"), ("advisories", "advisories.json")):
        p: Path = imports / fname
        if p.is_file():
            try:
                docs[key] = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"could not read import {fname}: {exc}")
    if "pfz" not in docs:
        errors.append("PFZ: no documented public API used; place an official INCOIS copy at orca/data/imports/pfz.json")
    if "advisories" not in docs:
        errors.append("advisories: no documented public API used; place an official IMD copy at orca/data/imports/advisories.json")
    return docs, errors
