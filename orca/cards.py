"""Evidence cards: the central UI artefact.

A card row is built from exactly one schema object (an Observation, a feed, a geofence
status). If that object is missing or unusable the row says ``INSUFFICIENT DATA`` -- rows are
never defaulted, interpolated, or filled from another source.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .decision.evidence import FusionResult
from .explain import fmt
from .i18n import state_label, t
from .schema import AdvisoryFeed, Decision, Observation

INSUFFICIENT = "INSUFFICIENT DATA"
_STATUS = {"LOWER_RISK": "ok", "MODERATE_CONDITIONS": "warn", "HIGHER_RISK": "bad", "RESTRICTED": "bad", "INSUFFICIENT_DATA": "missing"}


def _worst_status(decision: Decision | None, variable: str) -> str:
    if decision is None:
        return "info"
    order = ["ok", "warn", "missing", "bad"]
    found = [_STATUS[r.outcome] for r in decision.reasons if r.variable == variable]
    return max(found, key=order.index) if found else "info"


def obs_row(lang: str, key: str, label_key: str, obs: Observation | None, decision: Decision | None = None, *,
            status: str | None = None, extra: str = "") -> dict[str, Any]:
    label = t(lang, label_key)
    if obs is None or obs.value is None:
        why = ""
        return {"key": key, "label": label, "text": INSUFFICIENT, "status": "missing", "obs_id": obs.obs_id if obs else None,
                "source": obs.source_label if obs else None, "valid_time": None, "time_kind": None, "freshness_hours": None,
                "note": (obs.note if obs else "") or why}
    text = f"{fmt(obs.value)} {obs.unit}"
    if obs.variable == "lightning":
        text = t(lang, "a.lightning_yes") if obs.value >= 1 else t(lang, "a.lightning_none")
    if extra:
        text += f" ({extra})"
    if obs.quality == "stale":
        text += "  [STALE]"
    return {"key": key, "label": label, "text": text, "status": status or _worst_status(decision, obs.variable),
            "obs_id": obs.obs_id, "source": obs.source_label, "valid_time": obs.valid_time, "time_kind": obs.time_kind,
            "freshness_hours": obs.freshness_hours, "offset_km": obs.offset_km, "note": obs.note}


def advisory_rows(lang: str, feed: AdvisoryFeed | None, active: Sequence[Any], decision: Decision | None) -> list[dict[str, Any]]:
    rows = []
    if feed is None or not feed.available:
        rows.append({"key": "advisory", "label": t(lang, "var.advisory"), "text": INSUFFICIENT, "status": "missing", "source": None,
                     "note": "advisory feed unavailable - official warnings cannot be ruled out"})
        rows.append({"key": "cyclone", "label": t(lang, "var.cyclone"), "text": INSUFFICIENT, "status": "missing", "source": None})
        return rows
    if active:
        for a in active:
            rows.append({"key": "advisory", "label": t(lang, "var.advisory"), "text": f"{a.severity.upper()}: {a.text}",
                         "status": "bad" if a.severity in ("warning", "nogo", "ban") else "warn", "source": a.source_label,
                         "valid_time": f"{a.valid_from} -> {a.valid_to}", "freshness_hours": feed.freshness_hours})
    else:
        rows.append({"key": "advisory", "label": t(lang, "var.advisory"), "text": t(lang, "a.adv_none"),
                     "status": "ok" if feed.quality == "available" else "missing", "source": feed.source_label,
                     "freshness_hours": feed.freshness_hours})
    cy_status = "ok" if feed.cyclone_status == "none" else ("warn" if feed.cyclone_status == "watch" else ("bad" if feed.cyclone_status == "warning" else "missing"))
    cy_text = (t(lang, "a.cyclone_none") if feed.cyclone_status == "none" else
               (INSUFFICIENT if feed.cyclone_status == "unknown" else t(lang, "a.cyclone_other", status=feed.cyclone_status)))
    rows.append({"key": "cyclone", "label": t(lang, "var.cyclone"), "text": cy_text, "status": cy_status,
                 "source": feed.source_label, "note": feed.cyclone_text, "freshness_hours": feed.freshness_hours})
    return rows


def geofence_row(lang: str, gs: Mapping[str, Any] | None) -> dict[str, Any]:
    if gs is None:
        return {"key": "geofence", "label": t(lang, "var.geofence"), "text": INSUFFICIENT, "status": "missing"}
    if not gs.get("covered", True):
        return {"key": "geofence", "label": t(lang, "var.geofence"), "text": INSUFFICIENT, "status": "missing",
                "note": "no geofence dataset covers this location - restricted areas cannot be ruled out"}
    if gs.get("on_land"):
        return {"key": "geofence", "label": t(lang, "var.geofence"), "text": "ON LAND", "status": "bad"}
    if gs["inside"]:
        return {"key": "geofence", "label": t(lang, "var.geofence"), "text": t(lang, "a.geo_inside", name=gs["inside"][0]["name"]), "status": "bad"}
    n = gs.get("nearest")
    note = f"nearest hard constraint: {n['name']}, {fmt(n['distance_km'])} km {n['direction']}" if n else ""
    return {"key": "geofence", "label": t(lang, "var.geofence"), "text": t(lang, "a.geo_clear"), "status": "ok", "note": note}


def freshness_block(lang: str, *, weather: Observation | None, ocean: Observation | None, pfz=None, feed: AdvisoryFeed | None = None) -> dict[str, Any]:
    def one(freshness, issued, label):
        if freshness is None:
            return {"text": INSUFFICIENT, "hours": None, "source": label}
        return {"text": f"{fmt(freshness)} h old", "hours": freshness, "issued_at": issued, "source": label}
    return {
        "weather": one(*( (weather.freshness_hours, weather.issued_at, weather.source_label) if weather else (None, None, None) )),
        "ocean": one(*( (ocean.freshness_hours, ocean.observed_at, ocean.source_label) if ocean else (None, None, None) )),
        "pfz": one(*( (pfz.freshness_hours, pfz.issued_at, pfz.source_label) if pfz and pfz.available else (None, None, None) )),
        "advisory": one(*( (feed.freshness_hours, feed.issued_at, feed.source_label) if feed and feed.available else (None, None, None) )),
    }


def conflict_block(lang: str, fusion: FusionResult) -> list[dict[str, Any]]:
    out = []
    for c in fusion.conflicts:
        out.append({**c.to_dict(), "headline": "SOURCE DISAGREEMENT DETECTED",
                    "lines": [f"{v['source_label']}: {fmt(v['value'])} {v['unit']}" for v in c.values],
                    "selected_line": f"Selected source: {c.selected_source} (configured priority: {' > '.join(c.priority)})",
                    "difference_line": f"Difference: {fmt(c.difference)} {c.unit}"})
    return out


def sources_of(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: list[str] = []
    for r in rows:
        s = r.get("source")
        if s and s not in seen:
            seen.append(s)
    return seen


def make_card(lang: str, *, title: str, decision: Decision | None, headline: str, sections: Sequence[Mapping[str, Any]],
              freshness: Mapping[str, Any], conflicts: Sequence[Mapping[str, Any]], mode_label: str, kind: str = "site") -> dict[str, Any]:
    rows = [r for s in sections for r in s["rows"]]
    state = decision.state if decision else "INSUFFICIENT_DATA"
    return {
        "kind": kind, "title": title,
        "recommendation": {"state": state, "label": state_label(lang, state), "headline": headline},
        "sections": [dict(s) for s in sections], "freshness": dict(freshness), "sources": sources_of(rows),
        "conflicts": list(conflicts),
        "sensitivity": [dict(s) for s in (decision.sensitivity if decision else ())],
        "missing": list(decision.missing) if decision else [], "stale": list(decision.stale) if decision else [],
        "flags": list(decision.flags) if decision else [], "mode_label": mode_label,
    }
