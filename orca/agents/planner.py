"""Query understanding + evidence planning. Deterministic, bilingual (English / Tamil).

The rule-based understanding here is what makes the demo work with no key and no network.
An LLM may *assist* intent extraction in a deployment, but nothing downstream depends on
one: the output of this module is a plain dict (intent, place, times, constraints) that
the validator checks and the trace records. A misparse therefore shows up as a wrong
intent in the trace, not as an invented marine fact.

Adding a language = add a keyword table below and a catalog in :mod:`orca.i18n`.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from ..config import OrcaConfig
from ..config.settings import IST

__all__ = ["detect_language", "understand", "plan_evidence", "INTENTS"]

INTENTS = ("go_no_go", "pfz_zones", "nearest_pfz", "sea_conditions", "hazard", "route", "route_check", "avoid_areas", "ocean_regions",
           "compare", "scenario", "closer", "why", "unknown")

#: intents that can be re-run with a new time when the user only says "what about tomorrow?"
RERUNNABLE = ("go_no_go", "pfz_zones", "nearest_pfz", "sea_conditions", "hazard", "route", "avoid_areas", "ocean_regions")

_SCRIPTS = {  # unicode block -> language code
    "ta": (0x0B80, 0x0BFF), "hi": (0x0900, 0x097F), "bn": (0x0980, 0x09FF), "te": (0x0C00, 0x0C7F),
    "kn": (0x0C80, 0x0CFF), "ml": (0x0D00, 0x0D7F),
}
SUPPORTED_LANGS = ("en", "ta")

_TAMIL_DIGITS = str.maketrans("௦௧௨௩௪௫௬௭௮௯", "0123456789")


def detect_language(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    counts = {lang: 0 for lang in _SCRIPTS}
    for c in letters:
        o = ord(c)
        for lang, (lo, hi) in _SCRIPTS.items():
            if lo <= o <= hi:
                counts[lang] += 1
    lang, n = max(counts.items(), key=lambda kv: kv[1])
    return lang if n / len(letters) >= 0.3 else "en"


# --- keyword tables ---------------------------------------------------------- #
_KW: dict[str, dict[str, tuple[str, ...]]] = {
    "scenario": {"en": ("what changes if", "what if i", "what happens if i", "if i leave", "if i go later", "if i leave later", "what would change"), "ta": ("என்ன மாறும்",)},
    "closer": {"en": ("which one is closer", "which is closer", "which one is nearer", "which is nearer", "closer one", "which one is closest", "which is closest"), "ta": ("எது அருகில்", "எது அருகிலுள்ள")},
    "nearest": {"en": ("nearest pfz", "closest pfz", "nearest potential fishing", "closest potential fishing", "nearest fishing zone", "closest fishing zone", "pfz nearest", "nearest zone"),
                "ta": ("மிக அருகில்", "அருகிலுள்ள மீன்பிடி", "அருகில் உள்ள மீன்பிடி")},
    "route_check": {"en": ("cross a restricted", "cross restricted", "cross any restricted", "enter a restricted", "enter restricted", "pass through restricted", "route cross", "does this route", "is this route", "intersect a restricted", "route intersect", "does the route"),
                    "ta": ("தடைசெய்யப்பட்ட பகுதியைக் கடக்க", "தடை எல்லையைக் கடக்க")},
    "regions": {"en": ("high chlorophyll", "favourable sst", "favorable sst", "chlorophyll and", "which nearby regions", "which regions", "chlorophyll"), "ta": ("குளோரோபில்",)},
    "compare": {"en": ("compare", " versus ", " vs ", "comparison"), "ta": ("ஒப்பிட",)},
    "why": {"en": ("why did", "why is", "why was", "why this", "why do you", "why recommend", "explain why", "reason for"), "ta": ("ஏன்",)},
    "route": {"en": ("route", "path to", "way to", "navigate"), "ta": ("வழி", "பாதை")},
    "avoid": {"en": ("avoid", "restricted", "no-go", "off-limits", "forbidden"), "ta": ("தவிர்", "தடை")},
    "pfz": {"en": ("potential fishing zone", "fishing zone", "pfz", "where to fish", "where can i fish", "where should i fish", "fish near", "good fishing"),
            "ta": ("மீன்பிடி மண்டல", "மீன்பிடிப் பகுதி", "மீன்பிடி பகுதி", "மீன் கிடைக்கும்", "பிஎஃப்இசட்")},
    "hazard": {"en": ("dangerous", "hazard", "warning", "cyclone", "alert", "storm", "unsafe"), "ta": ("ஆபத்து", "எச்சரிக்கை", "புயல்")},
    "sea": {"en": ("sea condition", "conditions", "wave", "wind", "weather", "swell", "sea state", "how rough"), "ta": ("அலை", "காற்று", "கடல் நிலை", "வானிலை")},
    "go": {"en": ("can i", "should i", "go fishing", "go out", "fish tomorrow", "fish today", "safe to", "okay to", "ok to", "fish at", "fish from"),
           "ta": ("போகலாமா", "செல்லலாமா", "மீன்பிடிக்க", "கடலுக்கு")},
}
_DAY = {
    "en": (("day after tomorrow", 2), ("tomorrow", 1), ("today", 0), ("tonight", 0), ("now", 0)),
    "ta": (("நாளை மறுநாள்", 2), ("நாளைக்கு", 1), ("நாளை", 1), ("இன்றைக்கு", 0), ("இன்று", 0), ("இப்போது", 0)),
}
_PART = {  # part of day -> representative hours
    "en": (("morning", (6, 8, 10)), ("afternoon", (12, 14, 16)), ("evening", (17, 19)), ("night", (21,))),
    "ta": (("காலை", (6, 8, 10)), ("மதியம்", (12, 14, 16)), ("மாலை", (17, 19)), ("இரவு", (21,))),
}
_TA_PERIOD_OFFSET = {"காலை": 0, "மதியம்": 12, "மாலை": 12, "இரவு": 12}


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(_TAMIL_DIGITS)


def _has(low: str, lang: str, key: str) -> list[str]:
    hits = []
    for k in _KW[key]["en"] + (_KW[key].get(lang, ()) if lang != "en" else ()):
        if k in low:
            hits.append(k.strip())
    if lang != "en":  # a Tamil query may still contain an English word ("PFZ")
        pass
    return hits


def _parse_places(low: str, gaz: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    best: tuple[int, dict[str, Any], str] | None = None
    for p in gaz.get("ports", []):
        names = [p["en"].lower(), *[a.lower() for a in p.get("aliases", [])], *[t.lower() for t in p.get("ta", [])]]
        for n in names:
            if n and n in low and (best is None or len(n) > best[0]):
                best = (len(n), p, n)
    return (best[1], best[2]) if best else (None, "")


def _parse_times(text: str, low: str, lang: str, now: datetime, compare: bool) -> dict[str, Any]:
    day_offset, day_hit = None, ""
    for table in (_DAY["en"], _DAY.get(lang, ())):
        for k, off in table:
            if k in low or k in text:
                if day_offset is None or len(k) > len(day_hit):
                    day_offset, day_hit = off, k
    hours: list[tuple[int, int]] = []
    hit_terms: list[str] = []
    # 6 AM / 10:30 pm
    for m in re.finditer(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)", low):
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3).replace(".", "")
        if 1 <= h <= 12:
            h = (h % 12) + (12 if ap == "pm" else 0)
            hours.append((h, mi)); hit_terms.append(m.group(0))
    # Tamil: காலை 6 மணி
    for m in re.finditer(r"(காலை|மதியம்|மாலை|இரவு)?\s*(\d{1,2})(?::(\d{2}))?\s*மணி", text):
        h, mi = int(m.group(2)), int(m.group(3) or 0)
        off = _TA_PERIOD_OFFSET.get(m.group(1) or "", None)
        if off is not None and h < 12:
            h = (h % 12) + off if off else h
        hours.append((h, mi)); hit_terms.append(m.group(0).strip())
    # 24h "at 14:00" / "at 6"
    if not hours:
        for m in re.finditer(r"\bat\s+(\d{1,2})(?::(\d{2}))?\b(?!\s*(?:am|pm))", low):
            h = int(m.group(1))
            if 0 <= h <= 23:
                hours.append((h, int(m.group(2) or 0))); hit_terms.append(m.group(0))
    # compare "6 and 10" with no am/pm: read as morning hours
    if compare and len(hours) < 2:
        nums = re.findall(r"\b(\d{1,2})\b", low)
        if len(nums) >= 2 and not hours:
            hours = [(int(n), 0) for n in nums[:2] if 0 <= int(n) <= 23]
            hit_terms += nums[:2]
    part_hours: tuple[int, ...] = ()
    part_label = ""
    if not hours:
        for table in (_PART["en"], _PART.get(lang, ())):
            for k, hrs in table:
                if k in low or k in text:
                    part_hours, part_label = hrs, k
                    break
            if part_hours:
                break
    return {"day_offset": day_offset, "hours": hours, "part_hours": part_hours, "part_label": part_label, "terms": hit_terms + ([day_hit] if day_hit else [])}


_THERE = re.compile(r"\b(there|that zone|that pfz|that area|the same place|around it|near it|the zone)\b")
_MYLOC = re.compile(r"\b(my location|my position|my current|near me|from here|around me|where i am|my boat)\b")
_FOLLOWUP = re.compile(r"^\s*(and|what about|how about|then|also)\b")


def understand(text: str, cfg: OrcaConfig, gazetteer: Mapping[str, Any], *, context_place: str | None = None,
               forced_lang: str | None = None, context: Mapping[str, Any] | None = None,
               user_location: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Free text -> a normalised, language-independent intent record."""
    raw = text.strip()
    norm = _norm(raw)
    low = norm.lower()
    lang = forced_lang or detect_language(norm)
    now = cfg.clock()

    # ---- intent (priority order) ------------------------------------------------ #
    times_probe = _parse_times(norm, low, lang, now, compare=False)
    hits = {k: _has(low, lang, k) for k in _KW}
    if lang != "en":
        for k in _KW:  # Tamil keywords are matched against the un-lowercased normalised text too
            hits[k] += [t for t in _KW[k].get(lang, ()) if t in norm and t not in hits[k]]
    two_times = len(times_probe["hours"]) >= 2 or (re.search(r"\b\d{1,2}\s*(am|pm)?\s*(and|&|,|vs)\s*\d{1,2}\s*(am|pm)", low) is not None)
    intent = "unknown"
    matched: list[str] = []
    for name, key in (("scenario", "scenario"), ("closer", "closer"), ("route_check", "route_check"), ("ocean_regions", "regions"),
                      ("compare", "compare"), ("why", "why"), ("nearest_pfz", "nearest"), ("route", "route"), ("avoid_areas", "avoid"),
                      ("pfz_zones", "pfz"), ("hazard", "hazard"), ("sea_conditions", "sea"), ("go_no_go", "go")):
        if hits[key]:
            intent, matched = name, hits[key]
            break
    if intent in ("unknown", "go_no_go", "sea_conditions") and two_times and "compare" not in (intent,):
        if hits["compare"] or re.search(r"\b\d{1,2}\s*(am|pm)?\s*(and|&|vs)\s*\d{1,2}\s*(am|pm)", low):
            intent, matched = "compare", matched or ["two times"]
    if intent == "unknown" and re.search(r"\bfish", low):
        intent, matched = "go_no_go", ["fish"]

    ctx = dict(context or {})
    if context_place and not ctx.get("place"):
        ctx["place_name"] = context_place
    # ---- place ------------------------------------------------------------------ #
    port, port_hit = _parse_places(low + " " + norm, gazetteer)
    place_source = "query"
    place = None
    if port:
        place = {"id": port["id"], "name": port["en"], "lat": port["lat"], "lon": port["lon"],
                 "sea_lat": port["sea_lat"], "sea_lon": port["sea_lon"], "matched": port_hit, "source": place_source}
    elif user_location and (_MYLOC.search(low) or not (ctx.get("place") or ctx.get("place_name"))):
        la, lo = float(user_location["lat"]), float(user_location["lon"])
        place = {"id": "user", "name": "your location", "lat": la, "lon": lo, "sea_lat": la, "sea_lon": lo, "matched": "device location", "source": "user_location"}
    elif ctx.get("focus") and (_THERE.search(low) or "அங்கு" in norm or "அங்கே" in norm) and intent in ("go_no_go", "sea_conditions", "hazard", "unknown", "route", "route_check"):
        f = ctx["focus"]
        place = {"id": f["id"], "name": f["name"], "lat": f["lat"], "lon": f["lon"], "sea_lat": f["lat"], "sea_lon": f["lon"], "matched": "there", "source": "focus"}
    elif ctx.get("place"):
        place = {**ctx["place"], "matched": "(from conversation)", "source": "context"}
    elif context_place:
        port, port_hit = _parse_places(context_place.lower(), gazetteer)
        if port:
            place = {"id": port["id"], "name": port["en"], "lat": port["lat"], "lon": port["lon"], "sea_lat": port["sea_lat"],
                     "sea_lon": port["sea_lon"], "matched": port_hit, "source": "context"}

    # ---- time ------------------------------------------------------------------- #
    t = _parse_times(norm, low, lang, now, compare=(intent == "compare"))
    # follow-up: "What about tomorrow?" / "And at 10 AM?" keep the previous topic and change only the time
    followup = False
    time_given = t["day_offset"] is not None or bool(t["hours"]) or bool(t["part_hours"])
    if intent == "unknown" and ctx.get("intent") in RERUNNABLE and time_given and (_FOLLOWUP.search(low) or len(low.split()) <= 6):
        intent, matched, followup = ctx["intent"], ["follow-up: previous topic, new time"], True
    if intent == "scenario":
        # the scenario engine changes ONLY the time; the baseline comes from the conversation
        pass
    day_offset = t["day_offset"]
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    instants: list[datetime] = []
    label = ""
    if t["hours"]:
        d = day_offset if day_offset is not None else (0 if (t["hours"][0][0] > now.hour) else 1)
        assumed_day = day_offset is None
        for h, mi in t["hours"]:
            instants.append(midnight + timedelta(days=d, hours=h, minutes=mi))
        label = ", ".join(i.strftime("%H:%M") for i in instants)
        if assumed_day:
            label += f" (day assumed: {'today' if d == 0 else 'tomorrow'})"
    elif t["part_hours"]:
        d = day_offset if day_offset is not None else 0
        instants = [midnight + timedelta(days=d, hours=h) for h in t["part_hours"]]
        label = f"{t['part_label']} ({instants[0].strftime('%H:%M')}-{instants[-1].strftime('%H:%M')})"
    else:
        d = day_offset if day_offset is not None else 0
        base = now.replace(minute=0, second=0, microsecond=0) + (timedelta(hours=1) if now.minute >= 30 else timedelta())
        instants = [base] if day_offset in (None, 0) else [midnight + timedelta(days=d, hours=6)]
        label = "now" if day_offset in (None, 0) else "06:00 (default morning departure)"

    constraints = {
        "avoid_restricted": True,  # always: geofences are hard constraints, not a user option
        "objective": "lower_exposure" if re.search(r"lower (marine )?exposure|less exposure|calmer|குறைந்த", low + norm) else None,
        "n_routes_requested": 2 if re.search(r"\btwo\b|\b2\b", low) and intent == "route" else None,
    }
    return {
        "text": raw, "language": lang, "language_supported": lang in SUPPORTED_LANGS, "intent": intent, "followup": followup, "day_given": day_offset is not None,
        "matched_terms": matched, "place": place, "times": [i.isoformat(timespec="seconds") for i in instants],
        "time_label": label, "time_terms": t["terms"], "constraints": constraints,
    }


_SITE_WX = ("wave_height", "wind_speed", "precipitation", "lightning")
_OCEAN = ("sst", "chlorophyll", "sst_gradient", "current_speed", "tide_height")


def plan_evidence(u: Mapping[str, Any]) -> dict[str, Any]:
    """What evidence does this question need, from which agent/tool, and why."""
    it = u["intent"]
    req: list[dict[str, str]] = []

    def need(variable: str, agent: str, tool: str, why: str, required: bool = True) -> None:
        req.append({"variable": variable, "agent": agent, "tool": tool, "why": why, "required": required})

    def weather(required: bool = True):
        need("wave_height", "weather", "get_marine_weather", "sea-state limit for a small boat", required)
        need("wind_speed", "weather", "get_marine_weather", "wind limit for a small boat", required)
        need("precipitation", "weather", "get_marine_weather", "visibility", False)
        need("lightning", "weather", "get_marine_weather", "lightning risk at sea", False)

    def advisories():
        need("advisories", "weather", "get_advisories", "official warnings always override ORCA thresholds")
        need("geofence", "geo", "check_geofence", "hard constraint: restricted / boundary / protected areas")

    if it in ("go_no_go", "compare"):
        weather(); advisories()
        need("pfz_zones", "geo", "get_pfz_candidates", "where is fishing effort likely to pay off", False)
        for v in ("sst", "chlorophyll", "sst_gradient"):
            need(v, "ocean", "get_ocean_conditions", "ocean support for the PFZ candidate", False)
    elif it == "pfz_zones":
        need("pfz_zones", "geo", "get_pfz_candidates", "the PFZ advisory is the source of zones")
        for v in ("sst", "chlorophyll", "sst_gradient"):
            need(v, "ocean", "get_ocean_conditions", "ocean support criteria at each zone", True)
        weather(); advisories()
    elif it in ("sea_conditions", "hazard"):
        weather(); advisories()
        for v in ("sst", "current_speed", "tide_height"):
            need(v, "ocean", "get_ocean_conditions", "sea-state context", False)
    elif it == "nearest_pfz":
        need("pfz_zones", "geo", "get_pfz_candidates", "the PFZ product is the only source of zones")
        need("hazards_around", "geo", "hazards_near", "advisories, geofences and threshold breaches around the nearest PFZ", False)
        for v in ("sst", "chlorophyll", "sst_gradient"):
            need(v, "ocean", "get_ocean_conditions", "ocean support for each PFZ candidate", False)
        weather(); advisories()
    elif it == "route_check":
        need("route", "geo", "check_route", "intersect the route with every hard geofence and land")
        advisories()
        need("wave_height", "weather", "get_marine_weather", "conditions at the departure", False)
    elif it == "ocean_regions":
        for v in ("chlorophyll", "sst"):
            need(v, "ocean", "get_ocean_conditions", "explicit criteria: chlorophyll minimum and SST band", True)
        need("sst_gradient", "ocean", "get_ocean_conditions", "optional front criterion", False)
        weather(False)
        need("advisories", "weather", "get_advisories", "departure card: official warnings", False)
        need("geofence", "geo", "check_geofence", "regions inside a hard geofence are excluded", False)
    elif it == "scenario":
        weather(); advisories()
        for v in ("sst", "chlorophyll", "sst_gradient"):
            need(v, "ocean", "get_ocean_conditions", "PFZ ocean support at each time", False)
        need("pfz_zones", "geo", "get_pfz_candidates", "PFZ conditions re-evaluated at each time", False)
    elif it == "closer":
        need("previous_result", "geo", "-", "distances come from the stored PFZ list of this conversation")
    elif it == "route":
        need("pfz_zones", "geo", "get_pfz_candidates", "route destinations")
        need("route", "geo", "plan_routes", "constraint-aware route search")
        weather(); advisories()
        for v in ("sst", "chlorophyll", "sst_gradient"):
            need(v, "ocean", "get_ocean_conditions", "ranking PFZ destinations", False)
    elif it == "avoid_areas":
        advisories(); weather(False)
    elif it == "why":
        need("previous_result", "geo", "-", "the explanation must come from a stored decision, not be regenerated freehand")
    return {"intent": it, "required": req, "tools": sorted({r["tool"] for r in req if r["tool"] != "-"})}
