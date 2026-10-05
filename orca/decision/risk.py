"""The deterministic Decision Engine.

Input: a fused evidence slice, an advisory feed, a geofence status, thresholds.
Output: one of five states with machine-readable reasons -- never a probability.

    RESTRICTED         a hard geofence binds, or an official fishing ban is active
    HIGHER_RISK        a threshold or an official warning/no-go says so
    INSUFFICIENT_DATA  required evidence is missing/stale and nothing already proves danger
    MODERATE_CONDITIONS
    LOWER_RISK         all required evidence present, every rule below its moderate line

Precedence: RESTRICTED > HIGHER_RISK > INSUFFICIENT_DATA > MODERATE > LOWER.
Missing data can never *clear* a location (INSUFFICIENT beats MODERATE/LOWER) but also never
*hides* danger that is already proven (HIGHER beats INSUFFICIENT). "Lower risk" means
"below the moderate thresholds", not "safe", and the wording everywhere reflects that.

:func:`decide_site` is a pure function of its inputs, which is what makes the
source-choice sensitivity check possible: re-run it with each conflicting value and see
whether the state changes.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any, Mapping, Sequence

from ..config import OrcaConfig
from ..schema import STATES, AdvisoryFeed, Conflict, Decision, Reason
from ..geo import point_in_polygon
from .evidence import FusionResult

__all__ = ["decide_site", "REQUIRED_SITE", "worst", "advisories_active_at"]

#: evidence a go/no-go site decision cannot be made without
REQUIRED_SITE = ("wave_height", "wind_speed")

_PRECEDENCE = {"RESTRICTED": 4, "HIGHER_RISK": 3, "INSUFFICIENT_DATA": 2, "MODERATE_CONDITIONS": 1, "LOWER_RISK": 0}


def worst(states: Sequence[str]) -> str:
    return max(states, key=lambda s: _PRECEDENCE[s]) if states else "INSUFFICIENT_DATA"


def advisories_active_at(feed: AdvisoryFeed, lat: float, lon: float, when: datetime) -> list:
    out = []
    for a in feed.items:
        if datetime.fromisoformat(a.valid_from) <= when <= datetime.fromisoformat(a.valid_to) and point_in_polygon(lat, lon, a.polygon):
            out.append(a)
    return out


def decide_site(
    fusion: FusionResult,
    feed: AdvisoryFeed | None,
    geofence_status: Mapping[str, Any],
    cfg: OrcaConfig,
    *,
    lat: float,
    lon: float,
    when: datetime,
    scope_id: str = "site",
    required: Sequence[str] = REQUIRED_SITE,
    require_advisory: bool = True,
    _skip_sensitivity: bool = False,
) -> Decision:
    th = cfg.thresholds
    reasons: list[Reason] = []
    contributions: list[str] = []
    flags: list[str] = []

    def add(rule: str, outcome: str, variable: str | None, value: Any, unit: str | None, threshold: Any,
            comparison: str, obs_id: str | None, detail: str = "") -> None:
        reasons.append(Reason(rule, outcome, variable, value, unit, threshold, comparison, obs_id, detail))
        contributions.append(outcome)

    # -- R-GEO: hard constraints -------------------------------------------- #
    missing: list[str] = []
    stale: list[str] = []
    if not geofence_status.get("covered", True):
        missing.append("geofence")
        add("R-GEO-03", "INSUFFICIENT_DATA", "geofence", None, None, "dataset coverage", "missing", None,
            "no geofence dataset covers this location, so restricted areas cannot be ruled out")
    for f in geofence_status.get("inside", []):
        add("R-GEO-01", "RESTRICTED", "geofence", f["id"], None, "hard geofence", "inside", None,
            f"{f['name']} ({f['kind']}) is a hard constraint")
    if geofence_status.get("on_land"):
        add("R-GEO-02", "RESTRICTED", "geofence", "LAND", None, "sea point required", "inside", None,
            "the point is on land, not at sea")
    if geofence_status.get("covered", True) and not geofence_status.get("inside") and not geofence_status.get("on_land"):
        reasons.append(Reason("R-GEO-00", "LOWER_RISK", "geofence", "clear", None, "hard geofence", "outside", None,
                              "point is outside every binding geofence"))

    # -- R-ADV: official advisories ----------------------------------------- #
    if feed is not None and feed.available and not feed.covers(lat, lon):
        if require_advisory:
            missing.append("advisories")
            add("R-ADV-08", "INSUFFICIENT_DATA", "advisories", None, None, "feed coverage", "missing", None,
                "the advisory feed does not cover this location, so official warnings cannot be ruled out")
    elif feed is None or not feed.available:
        if require_advisory:
            missing.append("advisories")
            add("R-ADV-00", "INSUFFICIENT_DATA", "advisories", None, None, "feed required", "missing", None,
                "no advisory feed is available, so official warnings cannot be ruled out")
    else:
        if feed.quality == "stale":
            stale.append("advisories")
            add("R-ADV-00", "INSUFFICIENT_DATA", "advisories", feed.freshness_hours, "h", th.max_age_hours["advisory"], "stale", None,
                "the advisory feed is older than the freshness limit")
        active = advisories_active_at(feed, lat, lon, when)
        for a in active:
            if a.severity == "ban":
                add("R-ADV-04", "RESTRICTED", "advisory", a.advisory_id, None, "ban", "active", None, a.text)
            elif a.severity in ("warning", "nogo"):
                add("R-ADV-03", "HIGHER_RISK", "advisory", a.advisory_id, None, a.severity, "active", None, a.text)
            elif a.severity == "caution":
                add("R-ADV-02", "MODERATE_CONDITIONS", "advisory", a.advisory_id, None, "caution", "active", None, a.text)
        if feed.cyclone_status in ("warning",):
            add("R-ADV-05", "HIGHER_RISK", "cyclone_status", feed.cyclone_status, None, "warning", "active", None, feed.cyclone_text)
        elif feed.cyclone_status == "watch":
            add("R-ADV-06", "MODERATE_CONDITIONS", "cyclone_status", feed.cyclone_status, None, "watch", "active", None, feed.cyclone_text)
        elif feed.cyclone_status == "unknown" and require_advisory:
            missing.append("cyclone_status")
            add("R-ADV-07", "INSUFFICIENT_DATA", "cyclone_status", None, None, "status required", "missing", None,
                "cyclone status is not reported by the feed")
        if not active and feed.cyclone_status == "none" and feed.quality == "available":
            reasons.append(Reason("R-ADV-01", "LOWER_RISK", "advisory", "none", None, "active advisory", "none", None,
                                  "no active advisory or cyclone warning at this point and time"))

    # -- R-WAVE / R-WIND / R-RAIN / R-LTG ----------------------------------- #
    def val(v: str):
        o = fusion.observation(v)
        return o

    for var in required:
        if var not in fusion.variables or fusion.variables[var].selected is None:
            (stale if fusion.variables.get(var) and fusion.variables[var].status == "stale" else missing).append(var)
            add(f"R-DATA-{var}", "INSUFFICIENT_DATA", var, None, None, "required", "missing" if var not in stale else "stale", None,
                f"required evidence '{var}' is not available")

    o = val("wave_height")
    if o and o.value is not None:
        if o.value >= th.wave_higher_m:
            add("R-WAVE-03", "HIGHER_RISK", "wave_height", o.value, "m", th.wave_higher_m, ">=", o.obs_id)
        elif o.value >= th.wave_moderate_m:
            add("R-WAVE-02", "MODERATE_CONDITIONS", "wave_height", o.value, "m", th.wave_moderate_m, ">=", o.obs_id)
        else:
            add("R-WAVE-01", "LOWER_RISK", "wave_height", o.value, "m", th.wave_moderate_m, "<", o.obs_id)
    o = val("wind_speed")
    if o and o.value is not None:
        if o.value >= th.wind_higher_kmh:
            add("R-WIND-03", "HIGHER_RISK", "wind_speed", o.value, "km/h", th.wind_higher_kmh, ">=", o.obs_id)
        elif o.value >= th.wind_moderate_kmh:
            add("R-WIND-02", "MODERATE_CONDITIONS", "wind_speed", o.value, "km/h", th.wind_moderate_kmh, ">=", o.obs_id)
        else:
            add("R-WIND-01", "LOWER_RISK", "wind_speed", o.value, "km/h", th.wind_moderate_kmh, "<", o.obs_id)
    o = val("precipitation")
    if o and o.value is not None:
        if o.value >= th.precip_moderate_mmh:
            add("R-RAIN-02", "MODERATE_CONDITIONS", "precipitation", o.value, "mm/h", th.precip_moderate_mmh, ">=", o.obs_id,
                "heavy rain reduces visibility")
        else:
            add("R-RAIN-01", "LOWER_RISK", "precipitation", o.value, "mm/h", th.precip_moderate_mmh, "<", o.obs_id)
    o = val("lightning")
    if o and o.value is not None:
        if o.value >= 1:
            add("R-LTG-02", "HIGHER_RISK", "lightning", int(o.value), "flag", 1, ">=", o.obs_id, "lightning risk at sea")
        else:
            add("R-LTG-01", "LOWER_RISK", "lightning", 0, "flag", 1, "<", o.obs_id)

    state = worst(contributions) if contributions else "INSUFFICIENT_DATA"
    # A lone LOWER_RISK verdict is only legitimate when nothing required is missing.
    if state == "LOWER_RISK" and (missing or stale):
        state = "INSUFFICIENT_DATA"
    for f in fusion.stale:
        if f not in stale:
            stale.append(f)
            flags.append("STALE_DATA_USED")

    decision = Decision(state=state, scope={"id": scope_id, "lat": lat, "lon": lon, "time": when.isoformat(timespec="seconds")},
                        reasons=tuple(reasons), missing=tuple(dict.fromkeys(missing)), stale=tuple(dict.fromkeys(stale)),
                        flags=tuple(flags), thresholds=th.to_dict())
    if _skip_sensitivity or not fusion.conflicts:
        return decision
    return replace(decision, sensitivity=tuple(_sensitivity(decision, fusion, feed, geofence_status, cfg, lat, lon, when,
                                                              scope_id, required, require_advisory)),
                   flags=tuple(decision.flags) + (("SOURCE_SENSITIVE",) if _sensitive(decision, fusion, feed, geofence_status, cfg,
                                                                                       lat, lon, when, scope_id, required, require_advisory) else ()))


def _alt_fusions(fusion: FusionResult):
    """Yield (conflict, source_family, FusionResult) with that source's value swapped in."""
    for c in fusion.conflicts:
        for v in c.values:
            if v["source_family"] == c.selected_source:
                continue
            fv = fusion.variables[c.variable]
            alt = next((o for o in fv.candidates if o.obs_id == v["obs_id"]), None)
            if alt is None:
                continue
            new = dict(fusion.variables)
            new[c.variable] = replace(fv, selected=alt)
            yield c, v["source_family"], replace(fusion, variables=new)


def _sensitivity(decision, fusion, feed, gs, cfg, lat, lon, when, scope_id, required, require_advisory):
    out = []
    for c, fam, alt_fusion in _alt_fusions(fusion):
        d2 = decide_site(alt_fusion, feed, gs, cfg, lat=lat, lon=lon, when=when, scope_id=scope_id, required=required,
                         require_advisory=require_advisory, _skip_sensitivity=True)
        alt_val = next(v["value"] for v in c.values if v["source_family"] == fam)
        out.append({"variable": c.variable, "selected_source": c.selected_source, "selected_value": c.selected_value,
                    "alternative_source": fam, "alternative_value": alt_val, "unit": c.unit,
                    "state_with_selected": decision.state, "state_with_alternative": d2.state,
                    "changes_decision": d2.state != decision.state})
    return out


def _sensitive(decision, fusion, feed, gs, cfg, lat, lon, when, scope_id, required, require_advisory) -> bool:
    return any(s["changes_decision"] for s in _sensitivity(decision, fusion, feed, gs, cfg, lat, lon, when, scope_id, required, require_advisory))
