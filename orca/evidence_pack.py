"""Evidence layer (blueprint section 20) and alerts (section 7: chat / map / alerts).

Turns the structured results of a run into **claim records** -- each says what is claimed, where it
came from, which clock it is valid on, what operation produced it and what that operation returned --
and links every factual sentence of the final answer to the records that support it.

Three provenance classes keep official information distinguishable from SEAX's own interpretation
(blueprint sections 18 and 35.11):

``official``  advisory / cyclone / PFZ products as issued by the configured official source
``source``    observations and forecasts from a data source (not official warnings)
``computed``  anything SEAX derived: threshold rules, distances, intersections, rankings, decisions

"No evidence -> no factual claim": :func:`link_answer` reports any factual sentence it cannot tie
to a record (``unlinked``); the evaluation counts them as an evidence-completeness failure.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from .explain import _NUM_UNIT, fmt
from .i18n import t

OFFICIAL, SOURCE, COMPUTED = "official", "source", "computed"

# words that make a sentence a factual claim about a category, and the record kinds that can support it
_CATEGORY = {
    "advisory": (("advisory", "எச்சரிக்கை", "warning"), ("advisory",)),
    "cyclone": (("cyclone", "புயல்"), ("cyclone",)),
    "geofence": (("geofence", "hard constraint", "restricted", "தடை", "naval", "land"), ("geofence", "route_check", "zone_screen", "hazard_near")),
    "pfz": (("pfz", "மண்டல", "fishing zone"), ("pfz", "distance", "zone_screen", "ranking")),
    "conflict": (("disagreement", "முரண்", "depends on which source", "எந்த மூலத்தை"), ("conflict",)),
    "decision": (("risk", "conditions", "restricted", "insufficient data", "ஆபத்து", "நிலைமைகள்", "போதிய தரவு", "தடைசெய்யப்பட்ட"), ("decision", "rule")),
    "route": (("route", "வழித்தட"), ("route_check", "route", "geofence")),
    "ocean_region": (("region", "chlorophyll", "குளோரோபில்", "criteria"), ("ocean_region", "rule")),
    "lightning": (("lightning", "மின்னல்"), ("rule", "observation")),
}


def _rec(rec_id: str, kind: str, cls: str, claim: str, *, source: str | None, operation: str, result: Any, issued_at=None, valid_from=None,
         valid_to=None, location=None, inputs: Sequence[str] = (), values: Sequence[tuple[float, str]] = ()) -> dict[str, Any]:
    return {"id": rec_id, "kind": kind, "class": cls, "claim": claim, "source": source, "issued_at": issued_at, "valid_from": valid_from,
            "valid_to": valid_to, "operation": operation, "result": result, "location": location, "inputs": list(inputs),
            "values": [[v, u] for v, u in values]}


def build_records(run: Any) -> list[dict[str, Any]]:
    """Claim records from a finished :class:`orca.pipeline.Run` (reads only structured results)."""
    recs: list[dict[str, Any]] = []
    n = 0

    def nid() -> str:
        nonlocal n
        n += 1
        return f"E{n}"

    ev = run.evidence
    seen_obs: set[str] = set()
    for d in run.decisions:
        scope = d["scope"]
        loc = {"lat": scope["lat"], "lon": scope["lon"], "id": scope["id"]}
        for r in d["reasons"]:
            oid = r.get("obs_id")
            if oid and oid in ev and oid not in seen_obs:
                seen_obs.add(oid)
                o = ev[oid]
                recs.append(_rec(nid(), "observation", SOURCE, f"{o['variable']} was {fmt(o['value'])} {o['unit']} ({o['time_kind']}) at {o['latitude']:.3f}, {o['longitude']:.3f}",
                                 source=o["source_label"], operation="observation_lookup", result=o["value"], issued_at=o.get("issued_at"), valid_from=o.get("valid_from"),
                                 valid_to=o.get("valid_to"), location={"lat": o["latitude"], "lon": o["longitude"]}, inputs=[oid], values=[(o["value"], o["unit"])]))
            if r["variable"] in ("geofence", "advisory", "cyclone_status", "advisories"):
                continue  # covered by the dedicated records below
            if r.get("value") is not None and r.get("unit"):
                recs.append(_rec(nid(), "rule", COMPUTED, f"Rule {r['rule_id']}: {r['variable']} {fmt(r['value']) if isinstance(r['value'], (int, float)) else r['value']} {r['unit']} {r['comparison']} {fmt(r['threshold']) if isinstance(r['threshold'], (int, float)) else r['threshold']} -> {r['outcome']}",
                                 source="SEAX threshold rule (prototype default)", operation="threshold_rule", result=r["outcome"], location=loc, inputs=[oid] if oid else [],
                                 values=[(r["value"], r["unit"])] + ([(r["threshold"], r["unit"])] if isinstance(r["threshold"], (int, float)) else [])))
        recs.append(_rec(nid(), "decision", COMPUTED, f"Decision at {scope['id']} for {scope['time']}: {d['state']}", source="SEAX decision engine", operation="rule_precedence",
                         result=d["state"], location=loc, inputs=[r["rule_id"] for r in d["reasons"]]))
        for c in d.get("sensitivity", []):
            recs.append(_rec(nid(), "conflict", COMPUTED, f"Decision sensitivity for {c['variable']}: {c['state_with_selected']} with {c['selected_source']}, {c['state_with_alternative']} with {c['alternative_source']}",
                             source="SEAX decision engine", operation="decision_recomputation", result=c["changes_decision"], location=loc,
                             values=[(c["selected_value"], c["unit"]), (c["alternative_value"], c["unit"])]))
    for card in run.cards:
        for cf in card["conflicts"]:
            recs.append(_rec(nid(), "conflict", COMPUTED, f"Sources disagree on {cf['variable']}: " + "; ".join(f"{v['source_label']} {fmt(v['value'])} {v['unit']}" for v in cf["values"]) +
                             f"; selected {cf['selected_source']} by configured priority", source=", ".join(v["source_label"] for v in cf["values"]), operation="evidence_fusion", result=cf["difference"],
                             valid_from=cf.get("valid_time"), location={"lat": cf["latitude"], "lon": cf["longitude"]}, inputs=[v["obs_id"] for v in cf["values"]],
                             values=[(v["value"], v["unit"]) for v in cf["values"]] + [(cf["difference"], cf["unit"]), (cf["tolerance"], cf["unit"])]))
    for card in run.cards:  # every value a card shows must have a record, not only the ones a rule used
        for sec in card["sections"]:
            for row in sec["rows"]:
                oid = row.get("obs_id")
                if oid and oid in ev and oid not in seen_obs and ev[oid]["value"] is not None:
                    seen_obs.add(oid)
                    o = ev[oid]
                    recs.append(_rec(nid(), "observation", SOURCE, f"{o['variable']} was {fmt(o['value'])} {o['unit']} ({o['time_kind']}) at {o['latitude']:.3f}, {o['longitude']:.3f}",
                                     source=o["source_label"], operation="observation_lookup", result=o["value"], issued_at=o.get("issued_at"), valid_from=o.get("valid_from"),
                                     valid_to=o.get("valid_to"), location={"lat": o["latitude"], "lon": o["longitude"]}, inputs=[oid], values=[(o["value"], o["unit"])]))
    for e in run.trace.of_step("validate"):  # a refusal is itself evidence: why the system declined to assess
        if e.status in ("refused", "withheld") and e.detail:
            recs.append(_rec(nid(), "decision", COMPUTED, f"Assessment not possible: {e.detail}", source="SEAX validators", operation="validation", result="refused",
                             values=[]))
    if run.provider.errors:
        recs.append(_rec(nid(), "decision", COMPUTED, f"Data unavailable in {run.provider.mode} mode: {'; '.join(run.provider.errors)[:300]}", source=run.provider.label(),
                         operation="capability_check", result="unavailable"))
    feed = run.feed
    if feed is not None and feed.available:
        for a in feed.items:
            recs.append(_rec(nid(), "advisory", OFFICIAL, f"Official advisory {a.advisory_id} ({a.severity}): {a.text}", source=a.source_label, operation="temporal_validity+spatial_extent",
                             result={"valid_from": a.valid_from, "valid_to": a.valid_to}, issued_at=feed.issued_at, valid_from=a.valid_from, valid_to=a.valid_to, inputs=[a.advisory_id]))
        recs.append(_rec(nid(), "cyclone", OFFICIAL, f"Official cyclone status: {feed.cyclone_status}. {feed.cyclone_text}", source=feed.source_label, operation="feed_lookup",
                         result=feed.cyclone_status, issued_at=feed.issued_at))
    elif "get_advisories" in [w["tool"] for w in run.menu["withheld"]] or (run.plan and any(r["variable"] == "advisories" for r in run.plan["required"])):
        recs.append(_rec(nid(), "advisory", OFFICIAL, "Official advisory feed unavailable: official warnings cannot be ruled out", source=None, operation="feed_lookup", result="unavailable"))
    for gs_key, gs in run._gs.items():
        inside = [i["id"] for i in gs["inside"]]
        near = gs.get("nearest")
        recs.append(_rec(nid(), "geofence", COMPUTED, (f"Point {gs_key[0]:.3f}, {gs_key[1]:.3f} is inside hard geofence {', '.join(inside)}" if inside else
                         (f"Point {gs_key[0]:.3f}, {gs_key[1]:.3f} is outside every hard geofence" + (f"; nearest {near['name']} at {fmt(near['distance_km'])} km {near['direction']}" if near else "")) if gs.get("covered", True)
                         else f"Point {gs_key[0]:.3f}, {gs_key[1]:.3f} is outside the geofence dataset's coverage"),
                         source=run.provider.geofence_origin()[:60] or "configured geofences", operation="point_in_polygon", result=not inside if gs.get("covered", True) else None,
                         location={"lat": gs_key[0], "lon": gs_key[1]}, values=[(near["distance_km"], "km")] if near else []))
    pfz = run.pfz_res
    if pfz is not None and pfz.available:
        recs.append(_rec(nid(), "pfz", OFFICIAL, f"PFZ advisory from {pfz.source_label}: {len(pfz.zones)} zone(s) valid for {pfz.zones[0].valid_for if pfz.zones else '?'}", source=pfz.source_label,
                         operation="validity_check", result="valid", issued_at=pfz.issued_at, valid_from=pfz.zones[0].valid_for if pfz.zones else None, valid_to=pfz.zones[0].valid_for if pfz.zones else None,
                         values=[(pfz.freshness_hours, "h")] if pfz.freshness_hours is not None else []))
    for z in run.extra.get("zones", []) or []:
        zn = z["zone"]
        recs.append(_rec(nid(), "distance", COMPUTED, f"{zn['name']} is {fmt(z['relation']['distance_km'])} km {z['relation']['direction']} of the origin", source="SEAX GIS engine (haversine)", operation="haversine_distance",
                         result=z["relation"]["distance_km"], location={"lat": zn["latitude"], "lon": zn["longitude"]}, values=[(z["relation"]["distance_km"], "km")]))
        recs.append(_rec(nid(), "zone_screen", COMPUTED, f"{zn['name']} geofence screen: {'REJECTED' if z['screen']['restricted'] else 'clear'}; ocean support {z['support']['met']}/3; state {z['state']}",
                         source="SEAX GIS engine + ocean criteria", operation="spatial_intersection", result=z["screen"]["restricted"], location={"lat": zn["latitude"], "lon": zn["longitude"]},
                         inputs=[i["obs_id"] for i in z["support"]["items"] if i["obs_id"]],
                         values=[(i["value"], i["unit"]) for i in z["support"]["items"] if i["value"] is not None and i.get("unit")] + [(zn["radius_km"], "km")]))
        for i in z["support"]["items"]:
            if i["value"] is None:
                continue
            if i["obs_id"] and i["obs_id"] in ev and i["obs_id"] not in seen_obs:
                seen_obs.add(i["obs_id"]); o = ev[i["obs_id"]]
                recs.append(_rec(nid(), "observation", SOURCE, f"{o['variable']} was {fmt(o['value'])} {o['unit']} (observed composite) near {zn['name']}", source=o["source_label"], operation="observation_lookup",
                                 result=o["value"], valid_from=o.get("valid_from"), location={"lat": o["latitude"], "lon": o["longitude"]}, inputs=[i["obs_id"]], values=[(o["value"], o["unit"])]))
            key = (zn["zone_id"], i["variable"])
            if key in seen_obs:
                continue
            seen_obs.add(key)
            thr = i["threshold"]
            thr_vals = [(float(x), i["unit"]) for x in str(thr).split("-")] if isinstance(thr, str) else [(thr, i["unit"])]
            recs.append(_rec(nid(), "rule", COMPUTED, f"PFZ support criterion {i['criterion']} ({thr}): {'met' if i['met'] else 'not met'} ({fmt(i['value'])} {i['unit']})",
                             source="SEAX ocean-support criteria (prototype default)", operation="threshold_rule", result=i["met"], inputs=[i["obs_id"]] if i["obs_id"] else [],
                             values=[(i["value"], i["unit"])] + thr_vals))
    for z in run.extra.get("zones_summary", []) or []:
        recs.append(_rec(nid(), "distance", COMPUTED, f"{z['name']} is {fmt(z['distance_km'])} km {z['direction']} (stored from the earlier PFZ list)", source="SEAX GIS engine (haversine)",
                         operation="haversine_distance", result=z["distance_km"], location={"lat": z["lat"], "lon": z["lon"]}, values=[(z["distance_km"], "km")]))
    hz = run.extra.get("nearby_hazards")
    if hz:
        recs.append(_rec(nid(), "hazard_near", COMPUTED, f"Hazard scan within {fmt(hz['radius_km'])} km of {fmt(hz['center'][0])}, {fmt(hz['center'][1])} at {hz['time']}: {len(hz['official_advisories'])} official advisory area(s), "
                         f"{len(hz['geofences'])} geofence(s), {hz['n_breaches_total']} threshold breach(es)", source="SEAX GIS engine", operation="buffer_intersection", result=hz["any"],
                         valid_from=hz["time"], values=[(hz["radius_km"], "km")] + [(f["distance_km"], "km") for f in hz["geofences"]] + [(c["value"], c["unit"]) for c in hz["computed_breaches"]] + [(c["threshold"], c["unit"]) for c in hz["computed_breaches"]]))
        for a in hz["official_advisories"]:
            recs.append(_rec(nid(), "advisory", OFFICIAL, f"Official advisory {a['advisory_id']} overlaps the analysis area: {a['text']}", source=a["source_label"], operation="temporal_validity+spatial_intersection", result=True,
                             issued_at=a.get("issued_at"), valid_from=a["valid_from"], valid_to=a["valid_to"]))
    for r in run.extra.get("route_check", []) or []:
        recs.append(_rec(nid(), "route_check", COMPUTED, f"{r['label']}: {len(r['hard_crossings'])} hard geofence crossing(s), land legs {len(r['land_legs'])}, {len(r['advisories'])} active advisory crossing(s); {r['n_legs']} leg(s) vs {r['n_fences']} geofences",
                         source="SEAX GIS engine", operation="spatial_intersection", result=r["ok"], location=None, values=[(r["n_legs"], "legs"), (r["n_fences"], "geofences"), (r["distance_km"], "km")]))
    if run.extra.get("regions") is not None and run.u["intent"] == "ocean_regions":
        th = run.cfg.thresholds
        recs.append(_rec(nid(), "rule", COMPUTED, "Explicit 'favourable' definition: chlorophyll minimum and SST band (prototype defaults, not a biological threshold)", source="SEAX ocean-region filter",
                         operation="criteria_definition", result={"chl_min": th.chl_min_mg_m3, "sst_band": [th.sst_min_c, th.sst_max_c]},
                         values=[(th.chl_min_mg_m3, "mg/m³"), (th.sst_min_c, "°C"), (th.sst_max_c, "°C"), (th.sst_gradient_min_c_per_10km, "°C")]))
    for g in run.extra.get("regions", []) or []:
        recs.append(_rec(nid(), "ocean_region", COMPUTED, f"{g['name']}: chlorophyll {fmt(g['chlorophyll'])} mg/m³, SST {fmt(g['sst'])} °C -> {'meets' if g['meets'] else 'does not meet'} the criteria",
                         source="SEAX ocean-region filter", operation="explicit_criteria_filter", result=g["meets"], inputs=[o for o in g["obs"] if o],
                         values=[(g["chlorophyll"], "mg/m³"), (g["sst"], "°C"), (g["distance_km"], "km")] if g["chlorophyll"] is not None else [(g["distance_km"], "km")]))
    rs = run.extra.get("routes")
    if rs:
        for r in rs.get("routes", []):
            if r.get("feasible") and r.get("max_exposure") is not None:
                recs.append(_rec(nid(), "route", COMPUTED, f"{r['role']} route to {r['destination']['name']}: {fmt(r['distance_km'])} km, about {fmt(r['duration_h'])} h, max exposure {fmt(r['max_exposure'])}",
                                 source="SEAX route engine", operation="constraint_aware_search+post_verification", result=r["geofence_check"]["ok"],
                                 values=[(r["distance_km"], "km"), (r["duration_h"], "h"), (r["max_exposure"], "*"), (r.get("detour_km", 0), "km"), (r.get("mean_exposure") or 0, "*")]))
    for c in run.extra.get("comparison", []) or []:
        recs.append(_rec(nid(), "decision", COMPUTED, f"Comparison row {c['label']}: {c['state']}", source="SEAX decision engine", operation="rule_precedence", result=c["state"], valid_from=c["time"],
                         values=[(x["value"], x["unit"]) for x in (c["wind"], c["wave"], c["precipitation"]) if x]))
    for a in run.extra.get("avoid", []) or []:
        if a["kind"] == "geofence":
            recs.append(_rec(nid(), "geofence", COMPUTED, f"{a['name']} ({a['fence_kind']}) is {fmt(a['distance_km'])} km {a['direction']} of the departure point", source=run.provider.geofence_origin()[:60] or "configured geofences",
                             operation="distance_to_polygon", result=a["distance_km"], values=[(a["distance_km"], "km")]))
        elif a["kind"] == "hotspot":
            recs.append(_rec(nid(), "rule", COMPUTED, f"High-exposure area near {fmt(a['lat'])}, {fmt(a['lon'])}: wave {fmt(a['wave_height'])} m, wind {fmt(a['wind_speed'])} km/h ({a['level']})",
                             source="SEAX threshold rule (prototype default)", operation="threshold_rule", result=a["level"], location={"lat": a["lat"], "lon": a["lon"]},
                             inputs=[x for x in (a.get("wave_obs"), a.get("wind_obs")) if x], values=[(a["wave_height"], "m"), (a["wind_speed"], "km/h")]))
        elif a["kind"] == "advisory":
            recs.append(_rec(nid(), "advisory", OFFICIAL, f"Official advisory {a['id']}: {a['name']}", source=run.feed.source_label if run.feed else None, operation="temporal_validity",
                             result=True, issued_at=run.feed.issued_at if run.feed else None))
    for h in run.extra.get("hazards", []) or []:
        recs.append(_rec(nid(), "rule", COMPUTED, f"Hazard sweep: {h['variable']} {fmt(h['value'])} {h['unit']} at {h['point']} reaches {h['level']}", source="SEAX threshold rule (prototype default)",
                         operation="threshold_rule", result=h["level"], location={"lat": h["lat"], "lon": h["lon"]}, values=[(h["value"], h["unit"]), (h["threshold"], h["unit"])]))
    return recs


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\u0B80-\u0BFF(])")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def link_answer(answer: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Tie each factual sentence to the records that support it (numbers first, then category)."""
    out, unlinked, factual = [], [], 0
    for s in _sentences(answer):
        nums = [(float(m.group(1)), m.group(2)) for m in _NUM_UNIT.finditer(s) if m.group(2) != "%"]
        low = s.lower()
        cats = [c for c, (words, _) in _CATEGORY.items() if any(w in low or w in s for w in words)]
        if not nums and not cats:
            out.append({"sentence": s, "factual": False, "evidence": []})
            continue
        factual += 1
        ids: list[str] = []
        for r in records:
            hit = any(abs(v - rv) <= 0.051 and (u == ru or ru == "*") for v, u in nums for rv, ru in r["values"]) if nums else False
            if not hit and not nums:
                # category words are supported by a record of that kind, or by a record stating the data was unavailable / the question refused
                hit = any(r["kind"] in _CATEGORY[c][1] or r["operation"] in ("validation", "capability_check") for c in cats)
            if hit:
                ids.append(r["id"])
        # a sentence with numbers must have its numbers supported; category words alone need a record of that category
        ok = bool(ids)
        if nums:  # every number must be backed, not just one
            ok = all(any(abs(v - rv) <= 0.051 and (u == ru or ru == "*") for r in records for rv, ru in r["values"]) for v, u in nums)
        if not ok:
            unlinked.append(s)
        out.append({"sentence": s, "factual": True, "evidence": ids[:8], "linked": ok})
    return {"sentences": out, "n_factual": factual, "n_linked": factual - len(unlinked), "unlinked": unlinked,
            "completeness": 1.0 if factual == 0 else round((factual - len(unlinked)) / factual, 4)}


def build_alerts(run: Any, lang: str) -> list[dict[str, Any]]:
    """Alerts from the same structured results (never from prose). ``class`` keeps official apart from computed."""
    L = lambda k, **kw: t(lang, k, **kw)  # noqa: E731
    al: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(level: str, cls: str, text: str, key: str) -> None:
        if key not in seen:
            seen.add(key)
            al.append({"level": level, "class": cls, "text": text})

    feed = run.feed
    focus_id = (run.extra.get("focus") or {}).get("id")
    relevant = [d for d in run.decisions if not d["scope"]["id"].startswith("zone:") or d["scope"]["id"] == f"zone:{focus_id}"]
    for z in run.extra.get("zones", []) or []:   # a PFZ the advisory lists but a hard constraint rejects: say so, as information
        if z["screen"]["restricted"]:
            nm = (z["screen"]["status"]["inside"] or z["screen"]["overlaps"] or [{"name": "hard geofence"}])[0]["name"]
            add("info", COMPUTED, L("alert.zone_rejected", zone=z["zone"]["name"].split(" (")[0], name=nm), f"zr:{z['zone']['zone_id']}")
    for d in relevant:
        for r in d["reasons"]:
            if r["rule_id"] in ("R-GEO-01", "R-GEO-02"):
                add("critical", COMPUTED, L("alert.restricted", name=r["detail"].split(" (")[0] if r["detail"] else r["value"]), f"geo:{r['value']}")
            elif r["rule_id"] == "R-ADV-04":
                add("critical", OFFICIAL, L("alert.official", text=r["detail"]), f"adv:{r['value']}")
            elif r["rule_id"] == "R-ADV-03":
                add("critical", OFFICIAL, L("alert.official", text=r["detail"]), f"adv:{r['value']}")
            elif r["rule_id"] == "R-ADV-02":
                add("warning", OFFICIAL, L("alert.official", text=r["detail"]), f"adv:{r['value']}")
            elif r["rule_id"] in ("R-ADV-05", "R-ADV-06"):
                add("critical" if r["rule_id"] == "R-ADV-05" else "warning", OFFICIAL, L("alert.cyclone", status=r["value"]), "cy")
            elif r["outcome"] == "HIGHER_RISK" and r.get("unit") and r["variable"] in ("wave_height", "wind_speed"):
                add("warning", COMPUTED, L("alert.higher", label=t(lang, f"var.{r['variable']}"), val=fmt(r["value"]), unit=r["unit"], thr=fmt(r["threshold"])), f"hi:{r['variable']}:{d['scope']['id']}")
            elif r["rule_id"] == "R-LTG-02":
                add("warning", COMPUTED, L("alert.lightning"), "ltg")
        if d.get("missing"):
            add("info", COMPUTED, L("alert.missing", what=", ".join(d["missing"])), f"miss:{d['scope']['id']}")
        if d.get("stale"):
            add("info", COMPUTED, L("alert.stale", what=", ".join(d["stale"])), f"stale:{d['scope']['id']}")
        for s in d.get("sensitivity", []):
            if s["changes_decision"]:
                add("warning", COMPUTED, L("alert.sensitive", var=t(lang, f"var.{s['variable']}")), f"sens:{s['variable']}")
    for r in run.extra.get("route_check", []) or []:
        for x in r["hard_crossings"]:
            add("critical", COMPUTED, L("alert.route_cross", name=x["name"]), f"rc:{x['id']}")
        for a in r["advisories"]:
            add("warning", OFFICIAL, L("alert.official", text=a["text"]), f"adv:{a['advisory_id']}")
    hz = run.extra.get("nearby_hazards")
    if hz:
        zone = (run.extra.get("focus") or {}).get("name", "")
        for a in hz["official_advisories"]:
            add("warning", OFFICIAL, L("alert.hazard_near", zone=zone, text=a["text"]), f"hz:{a['advisory_id']}")
    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(al, key=lambda a: order[a["level"]])
