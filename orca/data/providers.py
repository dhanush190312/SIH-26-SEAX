"""Data providers. One interface, three modes, and never a silent substitution.

``DEMO``     bundled synthetic files (``orca/data/demo``), every value labelled DEMO DATA.
``OFFLINE``  the last cache snapshot (``orca/data/cache/bundle.json``); its age is real,
             so it goes stale honestly and the decision engine says so.
``LIVE``     documented public endpoints only (see :mod:`orca.data.live`); whatever cannot
             be fetched is reported as unavailable -- it is never filled from DEMO.

All three funnel into :class:`BundleProvider`, which turns compact source documents into
:class:`~orca.schema.Observation` objects. That single conversion point is where the
three clocks (observed / forecast / retrieved), freshness and staleness are computed.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from satquery.log import get_logger

from ..config import OrcaConfig
from ..config.settings import IST
from ..geo import haversine_km
from ..schema import VARIABLES, Advisory, AdvisoryFeed, Observation, PFZZone

_LOG = get_logger("orca.data")

__all__ = ["BundleProvider", "EmptyProvider", "PFZResult", "ProviderUnavailable", "freeze", "make_provider", "save_cache"]


class ProviderUnavailable(RuntimeError):
    """A mode's data cannot be provided at all (e.g. OFFLINE with no cache file)."""


def _iso(dt: datetime) -> str:
    return dt.astimezone(IST).isoformat(timespec="seconds")


def _parse(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(IST)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class PFZResult:
    available: bool
    zones: tuple[PFZZone, ...]
    note: str
    source_label: str
    issued_at: str | None
    freshness_hours: float | None
    quality: str
    data_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {"available": self.available, "zones": [z.to_dict() for z in self.zones], "note": self.note,
                "source_label": self.source_label, "issued_at": self.issued_at,
                "freshness_hours": self.freshness_hours, "quality": self.quality, "data_mode": self.data_mode}


# --------------------------------------------------------------------------- #
# freezing: relative demo documents -> absolute documents (also the cache format)
# --------------------------------------------------------------------------- #

def freeze(docs: Mapping[str, Any], now: datetime) -> dict[str, Any]:
    """Make every relative time absolute, anchored on ``now`` (IST)."""
    d = copy.deepcopy(dict(docs))
    now = now.astimezone(IST)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if "weather" in d:
        d["weather"]["meta"]["anchor"] = _iso(midnight)
        for s in d["weather"]["sources"].values():
            if "issued_hours_before_now" in s:
                s["issued_at"] = _iso(now - timedelta(hours=s.pop("issued_hours_before_now")))
    if "ocean" in d:
        d["ocean"]["meta"]["anchor"] = _iso(midnight)
        for s in d["ocean"]["sources"].values():
            if "observed_hours_before_now" in s:
                s["observed_at"] = _iso(now - timedelta(hours=s.pop("observed_hours_before_now")))
        t = d["ocean"].get("tide")
        if t and "issued_hours_before_now" in t:
            t["issued_at"] = _iso(now - timedelta(hours=t.pop("issued_hours_before_now")))
    if "pfz" in d:
        s = d["pfz"]["source"]
        if "issued_hours_before_now" in s:
            s["issued_at"] = _iso(now - timedelta(hours=s.pop("issued_hours_before_now")))
        offs = d["pfz"].pop("valid_day_offsets", None) or ([d["pfz"].pop("valid_day_offset")] if "valid_day_offset" in d["pfz"] else None)
        if offs is not None:
            d["pfz"]["valid_dates"] = [(midnight + timedelta(days=o)).date().isoformat() for o in offs]
    if "advisories" in d:
        s = d["advisories"]["source"]
        if "issued_hours_before_now" in s:
            s["issued_at"] = _iso(now - timedelta(hours=s.pop("issued_hours_before_now")))
        for it in d["advisories"].get("items", []):
            if "from_h" in it:
                it["valid_from"] = _iso(midnight + timedelta(hours=it.pop("from_h")))
                it["valid_to"] = _iso(midnight + timedelta(hours=it.pop("to_h")))
    return d


def save_cache(docs: Mapping[str, Any], path: Path, *, origin_mode: str, retrieved_at: datetime) -> Path:
    """Persist frozen documents as the OFFLINE cache. ``retrieved_at`` is what "Last updated" shows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {"retrieved_at": _iso(retrieved_at), "origin_mode": origin_mode, "format": 1}, "docs": dict(docs)}
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# the provider
# --------------------------------------------------------------------------- #

class BundleProvider:
    """Serve absolute documents as schema objects. Never invents a value."""

    def __init__(self, docs: Mapping[str, Any], cfg: OrcaConfig, *, mode: str, retrieved_at: datetime,
                 origin: str = "", errors: Sequence[str] = (), origin_mode: str | None = None) -> None:
        self.docs = dict(docs)
        self.origin_mode = origin_mode or mode
        self.cfg = cfg
        self.mode = mode
        self.retrieved_at = retrieved_at.astimezone(IST)
        self.origin = origin
        self.errors = tuple(errors)
        self._now = cfg.clock()

    # -- introspection ------------------------------------------------------- #

    def capabilities(self) -> dict[str, bool]:
        d = self.docs
        adv = d.get("advisories")
        pfz = d.get("pfz")
        return {
            "weather": bool(d.get("weather", {}).get("points")),
            "ocean": bool(d.get("ocean", {}).get("points")),
            "tide": bool(d.get("ocean", {}).get("tide")),
            "pfz": bool(pfz and pfz.get("zones") is not None and pfz.get("source")),
            "advisory": bool(adv and adv.get("feed_ok")),
            "geofence": bool(d.get("geofences", {}).get("geofences")),
            "coast": bool(d.get("coast", {}).get("land")),
        }

    def info(self) -> dict[str, Any]:
        return {"mode": self.mode, "retrieved_at": _iso(self.retrieved_at), "origin": self.origin,
                "capabilities": self.capabilities(), "errors": list(self.errors),
                "origin_mode": self.origin_mode,
                "last_updated": _iso(self.retrieved_at) if self.mode == "OFFLINE" else None,
                "label": self.label()}

    def label(self) -> str:
        if self.mode == "OFFLINE":
            what = "CACHED DEMO DATA (synthetic)" if self.origin_mode == "DEMO" else "CACHED DATA"
            return f"OFFLINE / {what} - last updated {_iso(self.retrieved_at)}"
        return {"DEMO": "DEMO DATA (synthetic)", "LIVE": "LIVE DATA"}[self.mode]

    # -- helpers ------------------------------------------------------------- #

    def _age_h(self, ref_iso: str | None) -> float | None:
        if not ref_iso:
            return None
        return round(max(0.0, (self._now - _parse(ref_iso)).total_seconds() / 3600.0), 2)

    def _quality(self, value: float | None, freshness: float | None, cls: str) -> str:
        if value is None:
            return "missing"
        limit = self.cfg.thresholds.max_age_hours.get(cls)
        if freshness is not None and limit is not None and freshness > limit:
            return "stale"
        return "available"

    def _mk(self, *, variable: str, value: float | None, lat: float, lon: float, req: tuple[float, float],
            kind: str, valid: datetime | None, issued_at: str | None, family: str, source: str, label: str,
            offset_km: float | None, t_off: float | None = None, note: str = "", uid: str = "") -> Observation:
        unit, cls, _ = VARIABLES[variable]
        ref = valid.isoformat(timespec="seconds") if (kind == "observed" and valid) else issued_at
        freshness = self._age_h(ref if kind == "observed" else issued_at)
        obs_note = note
        if self.mode == "OFFLINE" and not obs_note:
            obs_note = f"cached copy of {self.origin_mode} data; last updated {_iso(self.retrieved_at)}"
        return Observation(
            obs_id=f"{uid or source}:{variable}:{lat:.3f},{lon:.3f}:{valid.strftime('%d%H') if valid else 'na'}",
            variable=variable, value=value, unit=unit, latitude=lat, longitude=lon, time_kind=kind,
            observed_at=_iso(valid) if (kind == "observed" and valid) else None,
            forecast_for=_iso(valid) if (kind == "forecast" and valid) else None,
            issued_at=issued_at, retrieved_at=_iso(self.retrieved_at), source=source, source_family=family,
            source_label=label, quality=self._quality(value, freshness, cls), freshness_hours=freshness,
            data_mode=self.mode, requested_lat=req[0], requested_lon=req[1], offset_km=offset_km,
            time_offset_min=t_off, note=obs_note)

    def _missing(self, variable: str, family: str, source: str, label: str, req, kind: str, when: datetime | None, note: str) -> Observation:
        return self._mk(variable=variable, value=None, lat=req[0], lon=req[1], req=req, kind=kind, valid=when,
                        issued_at=None, family=family, source=source, label=label, offset_km=None, note=note,
                        uid=f"{source}-missing")

    # -- weather ------------------------------------------------------------- #

    def weather(self, lat: float, lon: float, when: datetime) -> list[Observation]:
        doc = self.docs.get("weather")
        if not doc or not doc.get("points"):
            return []
        anchor = _parse(doc["meta"]["anchor"])
        hours = int(doc["meta"]["hours"])
        idx_f = (when.astimezone(IST) - anchor).total_seconds() / 3600.0
        idx = int(round(idx_f))
        out: list[Observation] = []
        for fam, src in doc["sources"].items():
            pts = [p for p in doc["points"] if fam in p["series"]]
            if not pts:
                continue
            near = min(pts, key=lambda p: haversine_km(lat, lon, p["lat"], p["lon"]))
            dist = round(haversine_km(lat, lon, near["lat"], near["lon"]), 2)
            if dist > self.cfg.snap_radius_km:
                for v in ("wave_height", "wind_speed", "precipitation", "lightning"):
                    out.append(self._missing(v, fam, src["id"], src["label"], (lat, lon), "forecast", when,
                                             f"no observation point within {self.cfg.snap_radius_km:g} km (nearest {dist:g} km)"))
                continue
            if not (0 <= idx < hours):
                for v in ("wave_height", "wind_speed", "precipitation", "lightning"):
                    out.append(self._missing(v, fam, src["id"], src["label"], (lat, lon), "forecast", when,
                                             "requested time is outside the covered forecast window"))
                continue
            valid = anchor + timedelta(hours=idx)
            t_off = round((when.astimezone(IST) - valid).total_seconds() / 60.0, 1)
            for v, arr in near["series"][fam].items():
                raw = arr[idx]
                out.append(self._mk(variable=v, value=None if raw is None else float(raw), lat=near["lat"], lon=near["lon"],
                                    req=(lat, lon), kind="forecast", valid=valid, issued_at=src.get("issued_at"),
                                    family=fam, source=src["id"], label=src["label"], offset_km=dist, t_off=t_off,
                                    note=src.get("note", ""), uid=f"{src['id']}@{near['id']}"))
        return out

    def weather_window(self) -> tuple[datetime, datetime] | None:
        doc = self.docs.get("weather")
        if not doc or not doc.get("points"):
            return None
        a = _parse(doc["meta"]["anchor"])
        return a, a + timedelta(hours=int(doc["meta"]["hours"]) - 1)

    def ocean_points(self) -> list[dict[str, Any]]:
        return [{"id": p["id"], "lat": p["lat"], "lon": p["lon"]} for p in self.docs.get("ocean", {}).get("points", [])]

    def weather_points(self) -> list[dict[str, Any]]:
        return list(self.docs.get("weather", {}).get("points", []))

    # -- ocean --------------------------------------------------------------- #

    def ocean(self, lat: float, lon: float, when: datetime) -> list[Observation]:
        doc = self.docs.get("ocean")
        if not doc or not doc.get("points"):
            return []
        anchor = _parse(doc["meta"]["anchor"])
        hours = int(doc["meta"]["hours"])
        idx = int(round((when.astimezone(IST) - anchor).total_seconds() / 3600.0))
        out: list[Observation] = []
        fam, src = next(iter(doc["sources"].items()))
        near = min(doc["points"], key=lambda p: haversine_km(lat, lon, p["lat"], p["lon"]))
        dist = round(haversine_km(lat, lon, near["lat"], near["lon"]), 2)
        observed = _parse(src["observed_at"])
        if dist > self.cfg.snap_radius_km:
            for v in ("sst", "chlorophyll", "sst_gradient", "current_speed"):
                out.append(self._missing(v, fam, src["id"], src["label"], (lat, lon), "observed" if v != "current_speed" else "forecast",
                                         observed if v != "current_speed" else when,
                                         f"no ocean product point within {self.cfg.snap_radius_km:g} km (nearest {dist:g} km)"))
        else:
            for v in ("sst", "chlorophyll", "sst_gradient"):
                raw = near.get(v)
                out.append(self._mk(variable=v, value=None if raw is None else float(raw), lat=near["lat"], lon=near["lon"],
                                    req=(lat, lon), kind="observed", valid=observed, issued_at=None, family=fam,
                                    source=src["id"], label=src["label"], offset_km=dist,
                                    note="latest composite; not time-resolved", uid=f"{src['id']}@{near['id']}"))
            cur = near.get("current_speed")
            if cur is not None and 0 <= idx < hours:
                out.append(self._mk(variable="current_speed", value=float(cur[idx]), lat=near["lat"], lon=near["lon"],
                                    req=(lat, lon), kind="forecast", valid=anchor + timedelta(hours=idx),
                                    issued_at=src.get("observed_at"), family=fam, source=src["id"], label=src["label"],
                                    offset_km=dist, uid=f"{src['id']}@{near['id']}"))
        tide = doc.get("tide")
        if tide:
            d = round(haversine_km(lat, lon, tide["lat"], tide["lon"]), 2)
            if d <= self.cfg.snap_radius_km and 0 <= idx < hours:
                out.append(self._mk(variable="tide_height", value=float(tide["series"][idx]), lat=tide["lat"], lon=tide["lon"],
                                    req=(lat, lon), kind="forecast", valid=anchor + timedelta(hours=idx),
                                    issued_at=tide.get("issued_at"), family=fam, source=src["id"] + "-tide",
                                    label=tide["station"], offset_km=d, uid=f"{src['id']}-tide"))
        return out

    # -- PFZ ----------------------------------------------------------------- #

    def pfz(self, date_iso: str) -> PFZResult:
        doc = self.docs.get("pfz")
        if not doc or not doc.get("source"):
            return PFZResult(False, (), "no PFZ source is available in this mode", "none", None, None, "missing", self.mode)
        src = doc["source"]
        fresh = self._age_h(src.get("issued_at"))
        quality = "available"
        limit = self.cfg.thresholds.max_age_hours["pfz"]
        if fresh is not None and fresh > limit:
            quality = "stale"
        valid = doc.get("valid_dates") or ([doc["valid_for"]] if doc.get("valid_for") else [])
        if date_iso not in valid:
            return PFZResult(False, (), f"the PFZ advisory on file is valid for {', '.join(valid) or 'no date'}, not {date_iso}",
                             src["label"], src.get("issued_at"), fresh, quality, self.mode)
        zones = tuple(PFZZone(
            zone_id=z["id"], name=z["name"], latitude=z["lat"], longitude=z["lon"], radius_km=z["radius_km"],
            source=src["id"], source_family=src["family"], source_label=src["label"], valid_for=date_iso,
            issued_at=src.get("issued_at") or "", retrieved_at=_iso(self.retrieved_at), freshness_hours=fresh or 0.0,
            quality=quality, data_mode=self.mode) for z in doc["zones"])
        return PFZResult(True, zones, "", src["label"], src.get("issued_at"), fresh, quality, self.mode)

    # -- advisories ---------------------------------------------------------- #

    def advisories(self) -> AdvisoryFeed:
        doc = self.docs.get("advisories")
        if not doc or not doc.get("feed_ok"):
            return AdvisoryFeed(False, (), "unknown", "", "none", "no advisory feed", None, _iso(self.retrieved_at), None,
                                "missing", self.mode, note="no advisory feed is available in this mode")
        src = doc["source"]
        fresh = self._age_h(src.get("issued_at"))
        limit = self.cfg.thresholds.max_age_hours["advisory"]
        items = tuple(Advisory(
            advisory_id=i["id"], kind=i["kind"], severity=i["severity"], text=i["text"],
            polygon=tuple((float(a), float(b)) for a, b in i["polygon"]), valid_from=i["valid_from"], valid_to=i["valid_to"],
            source=src["id"], source_label=src["label"]) for i in doc.get("items", []))
        cy = doc.get("cyclone", {})
        return AdvisoryFeed(True, items, cy.get("status", "unknown"), cy.get("text", ""), src["id"], src["label"],
                            src.get("issued_at"), _iso(self.retrieved_at), fresh,
                            "stale" if (fresh is not None and fresh > limit) else "available", self.mode,
                            coverage=tuple(doc["coverage"]) if doc.get("coverage") else None)

    # -- static layers ------------------------------------------------------- #

    def geofences(self) -> list[dict[str, Any]]:
        doc = self.docs.get("geofences", {})
        note = doc.get("meta", {}).get("origin", "")
        return [{**g, "source_note": note} for g in doc.get("geofences", [])]

    def geofence_coverage(self):
        c = self.docs.get("geofences", {}).get("meta", {}).get("coverage")
        return tuple(c) if c else None

    def geofence_origin(self) -> str:
        return self.docs.get("geofences", {}).get("meta", {}).get("origin", "")

    def coast(self) -> list[tuple[float, float]]:
        return [(float(a), float(b)) for a, b in self.docs.get("coast", {}).get("land", [])]

    def gazetteer(self) -> dict[str, Any]:
        return self.docs.get("gazetteer", {"ports": []})


class EmptyProvider(BundleProvider):
    """Nothing available (e.g. OFFLINE without a cache). Every capability is False,
    so the pipeline reports INSUFFICIENT_DATA rather than guessing."""

    def __init__(self, cfg: OrcaConfig, *, mode: str, reason: str, gazetteer: Mapping[str, Any] | None = None) -> None:
        super().__init__({"gazetteer": dict(gazetteer or {"ports": []})}, cfg, mode=mode, retrieved_at=cfg.clock(),
                         origin="none", errors=(reason,))

    def label(self) -> str:
        return f"{self.mode}: NO DATA AVAILABLE"


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #

def _demo_docs(cfg: OrcaConfig) -> dict[str, Any]:
    d = cfg.data_dir / "demo"
    return {name: _load(d / f"{name}.json") for name in ("weather", "ocean", "pfz", "advisories", "geofences", "coast", "gazetteer")}


def _apply_official_overrides(docs: dict[str, Any], cfg: OrcaConfig) -> None:
    """User-supplied official layers (never shipped): geofences_official.json etc."""
    g = cfg.data_dir / "geofences_official.json"
    if g.is_file():
        docs["geofences"] = _load(g)


def make_provider(cfg: OrcaConfig, *, focus: tuple[float, float] | None = None) -> BundleProvider:
    now = cfg.clock()
    if cfg.mode == "DEMO":
        docs = freeze(_demo_docs(cfg), now)
        _apply_official_overrides(docs, cfg)
        return BundleProvider(docs, cfg, mode="DEMO", retrieved_at=now, origin="bundled synthetic demo datasets")
    if cfg.mode == "OFFLINE":
        path = cfg.data_dir / "cache" / "bundle.json"
        gaz = _load(cfg.data_dir / "demo" / "gazetteer.json")
        if not path.is_file():
            return EmptyProvider(cfg, mode="OFFLINE", gazetteer=gaz,
                                 reason=f"no cache file at {path}; run `python -m orca cache-refresh` while in DEMO or LIVE mode")
        payload = _load(path)
        docs = payload["docs"]
        docs.setdefault("gazetteer", gaz)
        _apply_official_overrides(docs, cfg)
        return BundleProvider(docs, cfg, mode="OFFLINE", retrieved_at=_parse(payload["meta"]["retrieved_at"]),
                              origin=f"cache written from {payload['meta'].get('origin_mode', '?')} mode",
                              origin_mode=payload["meta"].get("origin_mode", "?"))
    # LIVE
    from .live import build_live_docs
    gaz = _load(cfg.data_dir / "demo" / "gazetteer.json")
    docs, errors = build_live_docs(cfg, focus or (10.7650, 79.8700), gaz)
    _apply_official_overrides(docs, cfg)
    prov = BundleProvider(docs, cfg, mode="LIVE", retrieved_at=now, origin="public endpoints (see docs/DATA_SOURCES.md)", errors=errors)
    if prov.capabilities()["weather"]:
        try:
            save_cache(docs, cfg.data_dir / "cache" / "bundle.json", origin_mode="LIVE", retrieved_at=now)
        except OSError as exc:  # a read-only checkout must not break a live query
            _LOG.warning("could not write cache: %s", exc)
    return prov
