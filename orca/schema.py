"""The common evidence schema: what the Fact Engine is allowed to hold.

An :class:`Observation` is the only way a marine number enters ORCA. It carries its
own provenance and keeps three clocks apart, because mixing them is how a stale
forecast gets presented as a current reading:

``observed_at``   when a sensor/satellite saw it   (time_kind == "observed")
``forecast_for``  the instant a forecast predicts  (time_kind == "forecast")
``retrieved_at``  when ORCA fetched it             (always)

``issued_at`` (the forecast run / product time) is what *freshness* is measured
against for forecasts. There is deliberately no numeric "confidence" field:
availability is a state (``available`` / ``stale`` / ``missing``), and ORCA never
emits a probability it cannot derive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

# variable -> (unit, freshness class, display key)
VARIABLES: dict[str, tuple[str, str, str]] = {
    "wave_height": ("m", "weather", "var.wave_height"),
    "wind_speed": ("km/h", "weather", "var.wind_speed"),
    "precipitation": ("mm/h", "weather", "var.precipitation"),
    "lightning": ("flag", "weather", "var.lightning"),
    "sst": ("°C", "ocean", "var.sst"),
    "chlorophyll": ("mg/m³", "ocean", "var.chlorophyll"),
    "sst_gradient": ("°C/10km", "ocean", "var.sst_gradient"),
    "current_speed": ("m/s", "ocean", "var.current_speed"),
    "tide_height": ("m", "tide", "var.tide_height"),
}

QUALITIES = ("available", "stale", "missing")
TIME_KINDS = ("observed", "forecast")


@dataclass(frozen=True)
class Observation:
    obs_id: str
    variable: str
    value: float | None
    unit: str
    latitude: float
    longitude: float
    time_kind: str                      # observed | forecast
    observed_at: str | None
    forecast_for: str | None
    issued_at: str | None
    retrieved_at: str
    source: str                         # concrete feed id, e.g. "demo-incois"
    source_family: str                  # priority key, e.g. "incois"
    source_label: str                   # human label incl. DEMO/LIVE marking
    quality: str                        # available | stale | missing
    freshness_hours: float | None
    data_mode: str                      # DEMO | LIVE | OFFLINE
    requested_lat: float | None = None
    requested_lon: float | None = None
    offset_km: float | None = None      # distance from the requested point to this sample
    time_offset_min: float | None = None
    note: str = ""

    @property
    def valid_time(self) -> str | None:
        return self.observed_at if self.time_kind == "observed" else self.forecast_for

    @property
    def geometry(self) -> dict[str, Any]:
        return {"type": "Point", "coordinates": [self.longitude, self.latitude]}

    @property
    def usable(self) -> bool:
        return self.value is not None and self.quality == "available"

    @property
    def valid_from(self) -> str | None:
        return self.valid_time

    @property
    def valid_to(self) -> str | None:
        """A forecast step is valid for one hour; an observation is an instant."""
        if self.time_kind == "forecast" and self.forecast_for:
            from datetime import datetime, timedelta
            return (datetime.fromisoformat(self.forecast_for) + timedelta(hours=1)).isoformat(timespec="seconds")
        return self.valid_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.source, "quality_status": self.quality, "valid_from": self.valid_from, "valid_to": self.valid_to,
            "obs_id": self.obs_id, "variable": self.variable, "value": self.value, "unit": self.unit,
            "latitude": self.latitude, "longitude": self.longitude, "geometry": self.geometry,
            "time_kind": self.time_kind, "observed_at": self.observed_at, "forecast_for": self.forecast_for,
            "valid_time": self.valid_time, "issued_at": self.issued_at, "retrieved_at": self.retrieved_at,
            "source": self.source, "source_family": self.source_family, "source_label": self.source_label,
            "quality": self.quality, "freshness_hours": self.freshness_hours, "data_mode": self.data_mode,
            "offset_km": self.offset_km, "time_offset_min": self.time_offset_min, "note": self.note,
        }


@dataclass(frozen=True)
class PFZZone:
    zone_id: str
    name: str
    latitude: float
    longitude: float
    radius_km: float
    source: str
    source_family: str
    source_label: str
    valid_for: str                      # ISO date the advisory applies to
    issued_at: str
    retrieved_at: str
    freshness_hours: float
    quality: str
    data_mode: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class Advisory:
    advisory_id: str
    kind: str                           # small_craft_caution | cyclone | swell | fishing_ban | ...
    severity: str                       # info | caution | warning | nogo | ban
    text: str
    polygon: tuple[tuple[float, float], ...]
    valid_from: str
    valid_to: str
    source: str
    source_label: str

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["polygon"] = [list(p) for p in self.polygon]
        return d


@dataclass(frozen=True)
class AdvisoryFeed:
    """``available=False`` means the feed could not be read -- *not* "no advisories"."""

    available: bool
    items: tuple[Advisory, ...]
    cyclone_status: str                 # none | watch | warning | unknown
    cyclone_text: str
    source: str
    source_label: str
    issued_at: str | None
    retrieved_at: str
    freshness_hours: float | None
    quality: str
    data_mode: str
    note: str = ""
    coverage: tuple[float, float, float, float] | None = None   # (south, west, north, east)

    def covers(self, lat: float, lon: float) -> bool:
        if self.coverage is None:
            return True
        s, w, n, e = self.coverage
        return s <= lat <= n and w <= lon <= e

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage": list(self.coverage) if self.coverage else None,
            "available": self.available, "items": [a.to_dict() for a in self.items],
            "cyclone_status": self.cyclone_status, "cyclone_text": self.cyclone_text,
            "source": self.source, "source_label": self.source_label, "issued_at": self.issued_at,
            "retrieved_at": self.retrieved_at, "freshness_hours": self.freshness_hours,
            "quality": self.quality, "data_mode": self.data_mode, "note": self.note,
        }


@dataclass(frozen=True)
class Conflict:
    """Two or more sources disagree. Never resolved silently."""

    variable: str
    latitude: float
    longitude: float
    valid_time: str | None
    values: tuple[Mapping[str, Any], ...]      # [{source_family, source_label, value, unit, obs_id}]
    selected_source: str
    selected_value: float
    difference: float
    unit: str
    tolerance: float
    priority: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["values"] = [dict(v) for v in self.values]
        d["priority"] = list(self.priority)
        return d


@dataclass(frozen=True)
class Reason:
    """One machine-readable reason behind a decision."""

    rule_id: str
    outcome: str                        # the state this rule pushed toward
    variable: str | None
    value: Any
    unit: str | None
    threshold: Any
    comparison: str                     # e.g. ">=", "inside", "missing", "stale"
    obs_id: str | None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


STATES = ("LOWER_RISK", "MODERATE_CONDITIONS", "HIGHER_RISK", "RESTRICTED", "INSUFFICIENT_DATA")


@dataclass(frozen=True)
class Decision:
    state: str
    scope: Mapping[str, Any]
    reasons: tuple[Reason, ...]
    missing: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    sensitivity: tuple[Mapping[str, Any], ...] = ()
    thresholds: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state, "scope": dict(self.scope),
            "reasons": [r.to_dict() for r in self.reasons],
            "missing": list(self.missing), "stale": list(self.stale), "flags": list(self.flags),
            "sensitivity": [dict(s) for s in self.sensitivity], "thresholds": dict(self.thresholds),
        }
