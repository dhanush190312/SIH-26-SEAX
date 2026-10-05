"""The ORCA controller: interpret -> plan -> validate -> specialists -> fuse -> decide -> explain.

Same shape as SatQuery's ``Controller`` (validate -> interpret -> route -> execute -> compose),
specialised for marine evidence. Every hop writes a :class:`satquery.agent.trace.Trace` entry,
including refusals, so the WHY THIS RESULT? view is a rendering of what actually ran.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from satquery.agent.trace import Trace, jsonable
from satquery.log import get_logger

from . import cards as C
from .agents.geo_agent import GeoAgent
from .agents.ocean_agent import OceanAgent
from .agents.planner import plan_evidence, understand
from .agents.weather_agent import WeatherAgent
from .config import OrcaConfig, get_config
from .config.settings import IST
from .data.providers import BundleProvider, make_provider
from .decision.evidence import FusionResult, fuse
from .decision.geofence import ConstraintSet, load_geofences
from .decision.risk import REQUIRED_SITE, advisories_active_at, decide_site, worst
from .decision.route import ExposureField
from .evidence_pack import build_alerts, build_records, link_answer
from .explain import MistralNarrator, build_ledger, fmt, gate_narration, verify_text
from .geo import haversine_km
from .i18n import CATALOGS, state_label, t
from .schema import AdvisoryFeed, Decision

_LOG = get_logger("orca.pipeline")
__all__ = ["Orca"]

_RANK = {"LOWER_RISK": 0, "MODERATE_CONDITIONS": 1, "INSUFFICIENT_DATA": 2, "HIGHER_RISK": 3, "RESTRICTED": 4}


def when_label(dt: datetime) -> str:
    return dt.astimezone(IST).strftime("%d %b %H:%M")


@dataclass
class Site:
    lat: float
    lon: float
    when: datetime
    scope: str
    fusion: FusionResult
    decision: Decision
    gs: dict[str, Any]
    active: list
    feed: AdvisoryFeed | None
    adv_covered: bool = True


class ResultStore:
    """In-memory LRU plus JSON on disk (results survive a restart, like SatQuery sessions)."""

    def __init__(self, root: Path, cap: int = 200) -> None:
        self.root, self.cap = Path(root), cap
        self._mem: dict[str, dict[str, Any]] = {}

    def put(self, qid: str, result: dict[str, Any]) -> None:
        self._mem[qid] = result
        while len(self._mem) > self.cap:
            self._mem.pop(next(iter(self._mem)))
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / f"{qid}.json").write_text(json.dumps(jsonable(result), ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            _LOG.warning("could not persist %s: %s", qid, exc)

    def get(self, qid: str) -> dict[str, Any] | None:
        if qid in self._mem:
            return self._mem[qid]
        p = self.root / f"{qid}.json"
        if p.is_file() and p.resolve().parent == self.root.resolve():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        return None


class Orca:
    """Service object: owns config, result store, sessions, provider cache, optional narrator."""

    def __init__(self, cfg: OrcaConfig | None = None, narrator=None) -> None:
        self.cfg = cfg or get_config()
        self.store = ResultStore(self.cfg.runs_dir)
        self.sessions: dict[str, dict[str, Any]] = {}
        self._live: tuple[float, BundleProvider] | None = None
        self.narrator = narrator
        if narrator is None and self.cfg.llm == "mistral":
            n = MistralNarrator(self.cfg.llm_model)
            self.narrator = n if n.configured else None

    # -- helpers ------------------------------------------------------------- #

    def gazetteer(self) -> dict[str, Any]:
        return json.loads((self.cfg.data_dir / "demo" / "gazetteer.json").read_text(encoding="utf-8"))

    def cfg_for(self, mode: str | None) -> OrcaConfig:
        if mode and mode.upper() != self.cfg.mode:
            if mode.upper() not in ("DEMO", "LIVE", "OFFLINE"):
                raise ValueError(f"unknown mode {mode!r}")
            return replace(self.cfg, mode=mode.upper())
        return self.cfg

    def provider(self, cfg: OrcaConfig, focus: tuple[float, float] | None = None) -> BundleProvider:
        if cfg.mode == "LIVE":
            if self._live and time.time() - self._live[0] < 900:
                return self._live[1]
            prov = make_provider(cfg, focus=focus)
            self._live = (time.time(), prov)
            return prov
        return make_provider(cfg)

    def health(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": "ok", "service": "ORCA", "version": "0.1.0", "config": self.cfg.public(),
                               "languages": sorted(CATALOGS), "narrator": getattr(self.narrator, "name", None),
                               "modes": {}}
        for m in ("DEMO", "OFFLINE", "LIVE"):
            if m == "LIVE":
                out["modes"][m] = {"note": "not probed by /health (would need network); select LIVE to try it"}
                continue
            prov = self.provider(self.cfg_for(m))
            out["modes"][m] = {"label": prov.label(), "capabilities": prov.capabilities(), "errors": prov.errors[:2]}
        return out

    def query(self, text: str, *, session_id: str = "default", mode: str | None = None, lang: str | None = None,
              now: datetime | None = None, location: Mapping[str, float] | None = None) -> dict[str, Any]:
        cfg = self.cfg_for(mode)
        if now is not None:
            cfg = replace(cfg, now=now)
        gaz = self.gazetteer()
        sess = self.sessions.get(session_id, {})
        u = understand(text, cfg, gaz, context_place=sess.get("place"), forced_lang=lang, context=sess.get("ctx"), user_location=location)
        return self.run(u, cfg, session_id)

    def run(self, u: dict[str, Any], cfg: OrcaConfig, session_id: str = "default") -> dict[str, Any]:
        return Run(self, cfg, u, session_id).execute()

    def structured(self, intent: str, place: str, times: Sequence[str], *, mode: str | None = None, session_id: str = "default",
                   lang: str = "en", text: str | None = None, extra: Mapping[str, Any] | None = None,
                   place_dict: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Programmatic entry (route / compare endpoints): skips text parsing, keeps everything else."""
        cfg = self.cfg_for(mode)
        gaz = self.gazetteer()
        u = understand(f"{place}", cfg, gaz, forced_lang=lang)  # resolves the place through the same gazetteer
        u = {**u, "intent": intent, "text": text or f"[{intent}] {place}", "matched_terms": ["structured request"], "language": lang,
             "language_supported": lang in CATALOGS,
             "times": [datetime.fromisoformat(x).astimezone(IST).isoformat(timespec="seconds") for x in times],
             "time_label": ", ".join(datetime.fromisoformat(x).astimezone(IST).strftime("%H:%M") for x in times)}
        if place_dict is not None:
            u["place"] = dict(place_dict)
        u.update(extra or {})
        return self.run(u, cfg, session_id)

    def get_result(self, qid: str) -> dict[str, Any] | None:
        return self.store.get(qid)


class Run:
    def __init__(self, orca: Orca, cfg: OrcaConfig, u: dict[str, Any], session_id: str) -> None:
        self.orca, self.cfg, self.u, self.sid = orca, cfg, u, session_id
        self.qid = "o_" + uuid.uuid4().hex[:8]
        self.trace = Trace(self.qid, u["text"])
        self.evidence: dict[str, dict[str, Any]] = {}
        self.lang = u["language"] if u.get("language") in CATALOGS else "en"
        self.warnings: list[str] = []
        self.ledger_structs: list[Any] = []
        self.decisions: list[dict[str, Any]] = []
        self.cards: list[dict[str, Any]] = []
        self.map_layers: dict[str, Any] = {}
        self.lines: list[str] = []
        self._gs: dict[tuple[float, float], dict[str, Any]] = {}
        self.feed: AdvisoryFeed | None = None
        self.extra: dict[str, Any] = {}
        self.pfz_res = None

    # -- utilities ----------------------------------------------------------- #

    def L(self, key: str, **kw: Any) -> str:
        return t(self.lang, key, **kw)

    def lab(self, var: str) -> str:
        return t(self.lang, f"var.{var}")

    def st(self, state: str) -> str:
        return state_label(self.lang, state)

    def note(self, *structs: Any) -> None:
        self.ledger_structs.extend(structs)

    def place_name(self) -> str:
        p = self.u.get("place")
        if not p:
            return ""
        if p.get("id") == "user":
            return self.L("place.user")
        if self.lang != "en":
            for g in self.orca.gazetteer()["ports"]:
                if g["id"] == p["id"] and g.get(self.lang):
                    return g[self.lang][0]
        return p["name"]

    def gs(self, lat: float, lon: float) -> dict[str, Any]:
        k = (round(lat, 5), round(lon, 5))
        if k not in self._gs:
            self._gs[k] = self.geo.geofence(lat, lon)
        return self._gs[k]

    # -- the sequence -------------------------------------------------------- #

    def execute(self) -> dict[str, Any]:
        u, cfg, tr = self.u, self.cfg, self.trace
        tr.add("input", "received", params={"text": u["text"], "session": self.sid, "mode_requested": cfg.mode})
        place = u.get("place")
        focus = (place["sea_lat"], place["sea_lon"]) if place else None
        self.provider = self.orca.provider(cfg, focus)
        info = self.provider.info()
        tr.add("data", "provider", output=info, detail=info["label"])
        self.cons = ConstraintSet(load_geofences(self.provider.geofences()), self.provider.coast(), mpa_hard=cfg.mpa_hard,
                                  coverage=self.provider.geofence_coverage())
        from .tools import ToolContext
        self.ctx = ToolContext(self.provider, cfg, self.cons)
        self.weather = WeatherAgent(self.ctx, tr, self.evidence)
        self.ocean = OceanAgent(self.ctx, tr, self.evidence)
        self.geo = GeoAgent(self.ctx, tr, self.evidence)
        tr.add("interpret", "query_understanding", output={k: u[k] for k in ("language", "language_supported", "intent", "matched_terms", "place", "times", "time_label", "time_terms", "constraints")})
        if not u.get("language_supported", True):
            self.warnings.append(self.L("a.unsupported_lang", lang=u["language"]))
        plan = plan_evidence(u)
        tr.add("plan", "evidence_planner", output=plan)
        from .tools import build_menu
        menu = build_menu(self.provider.capabilities(), plan["tools"])
        tr.add("validate", "capability_validator", status="ok" if not menu["withheld"] else "withheld", output=menu,
               detail="; ".join(f"{w['tool']}: {w['reason']}" for w in menu["withheld"]))
        self.plan, self.menu = plan, menu

        refusal = self._validate_context()
        if refusal:
            return self._finish(refusal["state"], refusal["answer"], status=refusal["status"])
        handler = getattr(self, f"_do_{u['intent']}")
        try:
            state, answer = handler()
        except Exception as exc:  # a crash must still leave a receipt and a truthful answer
            _LOG.exception("handler failed")
            tr.add("error", "handler", status="error", detail=f"{type(exc).__name__}: {exc}")
            return self._finish("INSUFFICIENT_DATA", f"Internal error while assessing: {type(exc).__name__}. No conclusion is offered.", status="error")
        return self._finish(state, answer)

    def _validate_context(self) -> dict[str, Any] | None:
        u, tr = self.u, self.trace
        if not u["text"].strip():
            tr.add("validate", "context", status="refused", detail="empty question")
            return {"state": "INSUFFICIENT_DATA", "answer": self.L("a.unknown"), "status": "needs_input"}
        if u["intent"] == "unknown":
            tr.add("validate", "context", status="refused", detail="intent not recognised")
            return {"state": "INSUFFICIENT_DATA", "answer": self.L("a.unknown"), "status": "needs_input"}
        if u["intent"] in ("why", "closer"):
            tr.add("validate", "context", detail=f"{u['intent']}-question: uses the stored result of this conversation, no place needed")
            return None
        if not u.get("place"):
            tr.add("validate", "context", status="refused", detail="no departure place resolved")
            return {"state": "INSUFFICIENT_DATA", "answer": self.L("a.needs_place"), "status": "needs_input"}
        caps = self.provider.capabilities()
        if not caps["weather"]:
            reason = "; ".join(self.provider.errors) or "no weather data"
            tr.add("validate", "context", status="refused", detail=f"no weather data in {self.cfg.mode}: {reason}")
            return {"state": "INSUFFICIENT_DATA", "answer": self.L("a.no_data", mode=self.cfg.mode, reason=reason), "status": "no_data"}
        win = self.provider.weather_window()
        times = [datetime.fromisoformat(x) for x in u["times"]]
        if win and u["intent"] not in ("avoid_areas",):
            bad = [x for x in times if not (win[0] <= x <= win[1])]
            if bad:
                tr.add("validate", "context", status="refused", detail=f"time outside the data window {win[0].isoformat(timespec='minutes')}..{win[1].isoformat(timespec='minutes')}",
                       output={"requested": [b.isoformat(timespec="minutes") for b in bad]})
                return {"state": "INSUFFICIENT_DATA", "status": "outside_window",
                        "answer": self.L("a.outside_window", when=when_label(bad[0]), start=when_label(win[0]), end=when_label(win[1]))}
        tr.add("validate", "context", output={"place": u["place"]["name"], "times": u["times"], "window": [x.isoformat(timespec="minutes") for x in win] if win else None})
        return None

    # -- building blocks ----------------------------------------------------- #

    @property
    def times(self) -> list[datetime]:
        ts = [datetime.fromisoformat(x).astimezone(IST) for x in self.u["times"]]
        return ts if len(ts) <= 3 else [ts[0], ts[len(ts) // 2], ts[-1]]

    def _feed(self) -> AdvisoryFeed | None:
        if self.feed is None and "get_advisories" in self.menu["available"]:
            ts = self.times
            self.feed = self.weather.advisories(min(ts), max(ts))
            if self.feed is not None:
                self.note(self.feed.to_dict())
        return self.feed

    def site_eval(self, lat: float, lon: float, when: datetime, scope: str, *, ocean: bool = False) -> Site:
        feed = self._feed()
        obs = list(self.weather.marine_weather(lat, lon, when))
        if ocean and "get_ocean_conditions" in self.menu["available"]:
            obs += self.ocean.ocean_conditions(lat, lon, when)
        fusion = fuse(obs, self.cfg, wanted=list(REQUIRED_SITE))
        gs = self.gs(lat, lon)
        dec = decide_site(fusion, feed, gs, self.cfg, lat=lat, lon=lon, when=when, scope_id=scope)
        active = advisories_active_at(feed, lat, lon, when) if feed and feed.available else []
        site = Site(lat, lon, when, scope, fusion, dec, gs, active, feed, adv_covered=bool(feed and feed.available and feed.covers(lat, lon)))
        self.decisions.append(dec.to_dict())
        self.trace.add("decide", f"decision_engine:{scope}", output={"state": dec.state, "reasons": [r.to_dict() for r in dec.reasons],
                                                                      "missing": dec.missing, "stale": dec.stale, "flags": dec.flags,
                                                                      "sensitivity": dec.sensitivity, "time": when.isoformat(timespec="minutes")},
                       detail=f"{scope} @ {when.isoformat(timespec='minutes')} -> {dec.state}")
        if fusion.conflicts:
            self.trace.add("fuse", f"source_conflict:{scope}", status="conflict", output={"conflicts": [c.to_dict() for c in fusion.conflicts]},
                           detail="; ".join(f"{c.variable}: {[(v['source_family'], v['value']) for v in c.values]} -> selected {c.selected_source}" for c in fusion.conflicts))
        self.note(dec.to_dict(), [c.to_dict() for c in fusion.conflicts],
                  [fusion.variables[v].selected.to_dict() for v in fusion.variables if fusion.variables[v].selected])
        return site

    def _site_rows(self, s: Site) -> list[dict[str, Any]]:
        lg, f, d = self.lang, s.fusion, s.decision
        rows = [C.obs_row(lg, "wind", "var.wind_speed", f.observation("wind_speed"), d),
                C.obs_row(lg, "wave", "var.wave_height", f.observation("wave_height"), d)]
        if f.observation("precipitation") or "precipitation" in f.variables:
            rows.append(C.obs_row(lg, "rain", "var.precipitation", f.observation("precipitation"), d))
        if "lightning" in f.variables:
            rows.append(C.obs_row(lg, "lightning", "var.lightning", f.observation("lightning"), d))
        for v, key in (("current_speed", "var.current_speed"), ("tide_height", "var.tide_height"), ("sst", "var.sst")):
            if v in f.variables and f.observation(v):
                rows.append(C.obs_row(lg, v, key, f.observation(v), d, status="info"))
        rows += C.advisory_rows(lg, s.feed if s.adv_covered else None, s.active, d)
        rows.append(C.geofence_row(lg, s.gs))
        return rows

    def _zone_rows(self, z: dict[str, Any]) -> list[dict[str, Any]]:
        lg = self.lang
        rows = []
        for it in z["support"]["items"]:
            key = {"sst": "var.sst", "chlorophyll": "var.chlorophyll", "sst_gradient": "var.sst_gradient"}[it["variable"]]
            o = z["_fusion"].observation(it["variable"])
            status = "ok" if it["met"] else ("warn" if it["met"] is False else "missing")
            rows.append(C.obs_row(lg, it["variable"], key, o, None, status=status, extra=f"criterion: {it['criterion']}" if o else ""))
        pf = z["zone"]
        rows.append({"key": "pfz", "label": self.lab("pfz"), "status": "ok" if z["recommended"] else ("bad" if z["state"] == "RESTRICTED" else "warn"),
                     "text": f"{pf['name']}: ocean support {z['support']['met']}/3", "source": pf["source_label"],
                     "valid_time": pf["valid_for"], "freshness_hours": pf["freshness_hours"], "time_kind": "forecast"})
        return rows

    def make_site_card(self, s: Site, title: str, zone: dict[str, Any] | None = None, *, kind: str = "site") -> dict[str, Any]:
        lg = self.lang
        sections = [{"title": title, "rows": self._site_rows(s)}]
        if zone is not None:
            sections.append({"title": f"PFZ: {zone['zone']['name']}", "rows": self._zone_rows(zone)})
        st = s.decision.state
        if st in ("HIGHER_RISK", "RESTRICTED"):
            head = self.L("card.hold")
        elif st == "INSUFFICIENT_DATA":
            head = self.L("card.insufficient")
        else:
            head = self.L("card.opportunity") if (zone is not None and zone["recommended"]) else self.L("card.no_opportunity")
        if kind == "zone":
            head = self.L("card.opportunity") if (zone and zone["recommended"]) else self.L("card.no_opportunity")
        ocean_obs = (zone["_fusion"].observation("sst") if zone else None) or s.fusion.observation("sst")
        fresh = C.freshness_block(lg, weather=s.fusion.observation("wave_height") or s.fusion.observation("wind_speed"),
                                  ocean=ocean_obs, pfz=self.pfz_res, feed=s.feed)
        card = C.make_card(lg, title=title, decision=s.decision, headline=head, sections=sections, freshness=fresh,
                           conflicts=C.conflict_block(lg, s.fusion), mode_label=self.provider.label(), kind=kind)
        card["scope"] = {"lat": s.lat, "lon": s.lon, "time": s.when.isoformat(timespec="minutes"), "id": s.scope}
        self.cards.append(card)
        return card

    def eval_zones(self, when: datetime, origin: tuple[float, float]) -> list[dict[str, Any]]:
        date_iso = when.date().isoformat()
        if "get_pfz_candidates" not in self.menu["available"] and "get_pfz_candidates" not in [x["tool"] for x in self.menu["withheld"]]:
            pass
        res = self.geo.pfz(date_iso) if self.provider.capabilities()["pfz"] else None
        self.pfz_res = res
        if res is not None and res.available and res.quality == "stale" and not self.cfg.allow_stale:
            res_note = f"the PFZ advisory is older than the freshness limit ({fmt(res.freshness_hours)} h old, limit {fmt(self.cfg.thresholds.max_age_hours['pfz'])} h), so it was rejected"
            self.extra["pfz_note"] = res_note
            self.note(res.to_dict(), {"pfz_limit_h": self.cfg.thresholds.max_age_hours["pfz"]})
            self.trace.add("validate", "pfz_freshness", status="refused", detail=res_note)
            return []
        if res is None or not res.available:
            note = res.note if res is not None else "PFZ source not available in this mode"
            self.extra["pfz_note"] = note
            return []
        self.note(res.to_dict())
        zones = []
        in_range = []
        for z in res.zones:
            d_km = self.geo.relation(origin, z.latitude, z.longitude)["distance_km"]
            if d_km <= self.cfg.pfz_radius_km:
                in_range.append(z)
            else:
                self.trace.add("rank", f"pfz_out_of_range:{z.zone_id}", status="skipped", detail=f"{d_km} km from the departure point (limit {self.cfg.pfz_radius_km:g} km)")
        if not in_range:
            self.extra["pfz_note"] = f"no PFZ candidate lies within {self.cfg.pfz_radius_km:g} km of the departure point"
            return []
        for z in in_range:
            screen = self.geo.screen_zone(z)
            oobs = self.ocean.ocean_conditions(z.latitude, z.longitude, when) if self.provider.capabilities()["ocean"] else []
            fo = fuse(oobs, self.cfg, wanted=["sst", "chlorophyll", "sst_gradient"])
            support = self.ocean.pfz_support(fo)
            site = self.site_eval(z.latitude, z.longitude, when, f"zone:{z.zone_id}")
            rel = self.geo.relation(origin, z.latitude, z.longitude)
            state = "RESTRICTED" if screen["restricted"] else site.decision.state
            why_key, why_kw = None, {}
            if screen["restricted"]:
                name = (screen["status"]["inside"] or screen["overlaps"] or [{"name": "land"}])[0]["name"]
                why_key, why_kw = "why.restricted", {"name": name}
                self.trace.add("constraint", f"geofence_override:{z.zone_id}", status="withheld",
                               detail=f"PFZ advisory zone {z.zone_id} rejected: {name}", output={"screen": screen})
            elif state in ("HIGHER_RISK",):
                why_key, why_kw = "why.higher", {"state": self.st(state)}
            elif state == "INSUFFICIENT_DATA":
                why_key = "why.insufficient"
            elif support["met"] < 2:
                why_key, why_kw = "why.support", {"n": support["met"]}
            recommended = why_key is None and state in ("LOWER_RISK", "MODERATE_CONDITIONS")
            zd = {"zone": z.to_dict(), "relation": rel, "screen": screen, "support": support, "state": state,
                  "conditions_state": site.decision.state, "recommended": recommended, "why_key": why_key, "why_kw": why_kw,
                  "decision": site.decision.to_dict(), "_site": site, "_fusion": fo}
            zones.append(zd)
        zones.sort(key=lambda z: (not z["recommended"], -z["support"]["met"], _RANK[z["state"]], z["relation"]["distance_km"]))
        for i, z in enumerate(zones):
            z["rank"] = i + 1
        self.note([{k: v for k, v in z.items() if not k.startswith("_")} for z in zones])
        self.trace.add("rank", "pfz_candidates", output={"order": [(z["zone"]["zone_id"], z["recommended"], z["support"]["met"], z["state"]) for z in zones]},
                       detail="rule: recommended first, more ocean-support criteria, lower risk state, shorter distance")
        return zones

    @staticmethod
    def clean_zone(z: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in z.items() if not k.startswith("_")}

    # -- intents ------------------------------------------------------------- #

    def _origin(self) -> tuple[float, float]:
        p = self.u["place"]
        return p["sea_lat"], p["sea_lon"]

    def _header(self, s: Site) -> str:
        return self.L("a.go", state=self.st(s.decision.state), place=self.place_name(), when=when_label(s.when))

    def _detail_lines(self, s: Site, ocean: bool = False) -> list[str]:
        f, d, out = s.fusion, s.decision, []
        w, v, rn, lt = f.observation("wave_height"), f.observation("wind_speed"), f.observation("precipitation"), f.observation("lightning")
        if w and v:
            out.append(self.L("a.values", wave=fmt(w.value), wind=fmt(v.value)))
        if rn:
            out.append(self.L("a.rain", rain=fmt(rn.value)))
        if ocean:
            sst, cur, td = f.observation("sst"), f.observation("current_speed"), f.observation("tide_height")
            bits = [x for x in (sst and f"{self.lab('sst')} {fmt(sst.value)} {sst.unit}", cur and f"{self.lab('current_speed')} {fmt(cur.value)} {cur.unit}", td and f"{self.lab('tide_height')} {fmt(td.value)} {td.unit}") if x]
            if bits:
                out.append(", ".join(bits) + ".")
        if lt:
            out.append(self.L("a.lightning_yes") if lt.value >= 1 else self.L("a.lightning_none"))
        if s.adv_covered and s.feed:
            out.append(" ".join(self.L("a.adv_active", text=a.text) for a in s.active) if s.active else self.L("a.adv_none"))
            out.append(self.L("a.cyclone_none") if s.feed.cyclone_status == "none" else
                       (self.L("a.cyclone_other", status=s.feed.cyclone_status) if s.feed.cyclone_status != "unknown" else ""))
        if s.gs.get("covered", True):
            out.append(self.L("a.geo_inside", name=s.gs["inside"][0]["name"]) if s.gs["inside"] else self.L("a.geo_clear"))
        if any("no observation point" in (o.note or "") for v in ("wave_height", "wind_speed") if v in f.variables for o in f.variables[v].rejected):
            out.append(self.L("a.outside", place=self.place_name()))
        for c in f.conflicts:
            out.append(self.L("a.conflict", var=self.lab(c.variable), list="; ".join(f"{x['source_label']}: {fmt(x['value'])} {x['unit']}" for x in c.values),
                              src=c.selected_source, diff=fmt(c.difference), unit=c.unit))
        for sv in d.sensitivity:
            if sv["changes_decision"]:
                out.append(self.L("a.sensitive", var=self.lab(sv["variable"]), a=sv["selected_source"], va=fmt(sv["selected_value"]), unit=sv["unit"],
                                  sa=self.st(sv["state_with_selected"]), b=sv["alternative_source"], vb=fmt(sv["alternative_value"]), sb=self.st(sv["state_with_alternative"])))
        if d.missing:
            names = {"geofence": "var.geofence", "advisories": "var.advisory", "cyclone_status": "var.cyclone"}
            out.append(self.L("a.missing", missing=", ".join(self.L(names[m]) if m in names else self.lab(m) if f"var.{m}" in CATALOGS["en"] else m for m in d.missing)))
        if d.stale:
            out.append(self.L("a.stale", stale=", ".join(d.stale)))
        return [x for x in out if x]

    def _pfz_best_line(self, zones: list[dict[str, Any]]) -> str:
        best = next((z for z in zones if z["recommended"]), None)
        if not best:
            return self.L("a.pfz_none") if not zones else ""
        return self.L("a.pfz_best", zone=best["zone"]["name"], dist=fmt(best["relation"]["distance_km"]), dir=best["relation"]["direction"],
                      n=best["support"]["met"], state=self.st(best["state"]))

    def _zone_line(self, z: dict[str, Any]) -> str:
        if z["why_key"]:
            return self.L("a.pfz_excluded", zone=z["zone"]["name"], why=self.L(z["why_key"], **z["why_kw"]))
        it = {i["variable"]: i for i in z["support"]["items"]}
        g = lambda k: fmt(it[k]["value"]) if it[k]["value"] is not None else "?"  # noqa: E731
        return self.L("a.pfz_line", zone=z["zone"]["name"], dist=fmt(z["relation"]["distance_km"]), dir=z["relation"]["direction"], n=z["support"]["met"],
                      sst=g("sst"), chl=g("chlorophyll"), grad=g("sst_gradient"), state=self.st(z["state"]))

    def _do_go_no_go(self) -> tuple[str, str]:
        o = self._origin()
        sites = [self.site_eval(o[0], o[1], w, "departure") for w in self.times]
        zones = self.eval_zones(self.times[0], o)
        best = next((z for z in zones if z["recommended"]), None)
        for s in sites:
            self.make_site_card(s, self.L("card.title_site", place=self.place_name(), when=when_label(s.when)), best)
        self._map_common(self.times[0], zones, sites[0])
        self.extra["zones"] = [self.clean_zone(z) for z in zones]
        worst_site = max(sites, key=lambda s: _RANK[s.decision.state])
        lines = [self._header(s) for s in sites] if len(sites) > 1 else []
        if len(sites) == 1:
            lines = [self._header(sites[0])]
        lines += self._detail_lines(worst_site)
        if worst_site.decision.state not in ("RESTRICTED",):
            lines.append(self._pfz_best_line(zones) or "")
        self._remember(zones)
        return worst(s.decision.state for s in sites), " ".join(x for x in lines if x)

    def _do_pfz_zones(self) -> tuple[str, str]:
        o = self._origin()
        when = self.times[0]
        zones = self.eval_zones(when, o)
        site = self.site_eval(o[0], o[1], when, "departure")
        if not zones:
            self._map_common(when, zones, site)
            return "INSUFFICIENT_DATA", f"{self.L('a.pfz_none')} {self.extra.get('pfz_note', '')}".strip()
        for z in zones:
            self.make_site_card(z["_site"], f"{z['zone']['name']} - {when_label(when)}", z, kind="zone")
        self._map_common(when, zones, site)
        self.extra["zones"] = [self.clean_zone(z) for z in zones]
        self.note(self.pfz_res.to_dict())
        head = self.L("a.pfz_head", date=when.date().isoformat(), source=self.pfz_res.source_label)
        lines = [head] + [f"{i + 1}. {self._zone_line(z)}" for i, z in enumerate(zones)]
        self._remember(zones)
        rec = [z for z in zones if z["recommended"]]
        state = worst(z["state"] for z in rec) if rec else "INSUFFICIENT_DATA" if any(z["state"] == "INSUFFICIENT_DATA" for z in zones) else "HIGHER_RISK"
        if not rec and all(z["state"] == "RESTRICTED" for z in zones):
            state = "RESTRICTED"
        return state, " ".join(lines)

    def _do_sea_conditions(self) -> tuple[str, str]:
        o = self._origin()
        sites = [self.site_eval(o[0], o[1], w, "departure", ocean=True) for w in self.times]
        for s in sites:
            self.make_site_card(s, self.L("card.title_site", place=self.place_name(), when=when_label(s.when)))
        self._map_common(self.times[0], [], sites[0])
        lines = []
        for s in sites:
            lines.append(self._header(s))
            lines += self._detail_lines(s, ocean=True)[:3]
        w = max(sites, key=lambda s: _RANK[s.decision.state])
        lines += self._detail_lines(w)[3:]
        return worst(s.decision.state for s in sites), " ".join(x for x in lines if x)

    def _do_hazard(self) -> tuple[str, str]:
        o = self._origin()
        sites = [self.site_eval(o[0], o[1], w, "departure") for w in self.times]
        for s in sites:
            self.make_site_card(s, self.L("card.title_site", place=self.place_name(), when=when_label(s.when)))
        scan = self.weather.scan_hazards(self.times)
        br = scan["breaches"]
        best: dict[tuple[str, str], dict[str, Any]] = {}
        for b in br:
            k = (b["point"], b["variable"])
            if k not in best or b["value"] > best[k]["value"]:
                best[k] = b
        top = sorted(best.values(), key=lambda b: (b["level"] != "HIGHER_RISK", -b["value"] / (b["threshold"] or 1)))[:5]
        self.extra["hazards"] = top
        self.note(top)
        self.trace.add("decide", "hazard_sweep", output={"n_breaches": len(br), "top": top})
        self._map_common(self.times[0], [], sites[0])
        lines = [self._header(sites[0]) if len(sites) == 1 else ""]
        if len(sites) > 1:
            lines = [self._header(s) for s in sites]
        lines += self._detail_lines(max(sites, key=lambda s: _RANK[s.decision.state]))
        if top:
            lines.append(self.L("a.hazard_area"))
            for b in top:
                lines.append(self.L("a.hazard_line", label=self.lab(b["variable"]), val=fmt(b["value"]), unit=b["unit"],
                                    where=f"{b['point']} ({fmt(b['lat'])}, {fmt(b['lon'])})", level=self.st(b["level"]), thr=fmt(b["threshold"])))
        else:
            lines.append(self.L("a.hazard_none", when=self.u["time_label"] or when_label(self.times[0]), place=self.place_name()))
        st = worst([s.decision.state for s in sites] + [b["level"] for b in top])
        return st, " ".join(x for x in lines if x)

    def _do_route(self) -> tuple[str, str]:
        o = self._origin()
        when = self.times[0]
        site = self.site_eval(o[0], o[1], when, "departure")
        zones = self.eval_zones(when, o)
        grid = self.weather.grid(when)
        self.extra["grid"] = grid
        field = ExposureField([(g["lat"], g["lon"], g["wave_height"]) for g in grid if g["wave_height"] is not None],
                              [(g["lat"], g["lon"], g["wind_speed"]) for g in grid if g["wind_speed"] is not None], self.cfg)
        zin = [{"id": z["zone"]["zone_id"], "name": z["zone"]["name"], "lat": z["zone"]["latitude"], "lon": z["zone"]["longitude"],
                "support": z["support"]["met"], "state": z["state"], "restricted": z["screen"]["restricted"]} for z in zones]
        rs = self.geo.routes(o, zin, field) if zin else {"routes": [], "note": self.extra.get("pfz_note", "no PFZ destinations available")}
        routes = [r for r in rs.get("routes", [])]
        self.extra["routes"] = rs
        self.note(rs, zin)
        self.make_site_card(site, self.L("card.title_site", place=self.place_name(), when=when_label(when)), next((z for z in zones if z["recommended"]), None))
        self._map_common(when, zones, site)
        for r in routes:  # the last line of defence, recorded per route
            chk = self.cons.verify_path([tuple(p) for p in r["path"]]) if r.get("feasible") else {"ok": False, "violations": {"n/a": 0}}
            self.trace.add("constraint", f"route_geofence_check:{r['role']}", status="ok" if chk["ok"] or not r.get("feasible") else "error",
                           output={"objective": r["objective"], "feasible": r.get("feasible"), "check": chk, "criteria": r.get("criteria")})
        lines = [self._header(site), self.L("a.route_head", place=self.place_name())]
        if not routes:
            lines.append(self.L("a.route_none", note=rs.get("note", "")))
            return ("INSUFFICIENT_DATA" if site.decision.state != "RESTRICTED" else "RESTRICTED"), " ".join(lines)
        names = {"shortest": "Shortest", "lower_exposure": "Lower-exposure", "pfz_oriented": "PFZ-oriented", "alternative": "Alternative destination (lower-exposure)"}
        for r in routes:
            if not r.get("feasible"):
                lines.append(self.L("a.route_none", note=f"{r['objective']}: {r.get('reason', '')}"))
                continue
            if r.get("identical_to"):
                lines.append(self.L("a.route_same", obj=names[r["role"]], other=names[r["identical_to"]]))
                continue
            lines.append(self.L("a.route_line", obj=names[r["role"]], dest=r["destination"]["name"], dist=fmt(r["distance_km"]), dur=fmt(r["duration_h"]),
                                mx=fmt(r["max_exposure"]) if r.get("max_exposure") is not None else "n/a"))
            if r.get("max_exposure") is not None and r["max_exposure"] >= 1.0:
                lines.append(self.L("a.route_exposed", obj=names[r["role"]]))
            if r["direct_blocked_by"]:
                lines.append(self.L("a.route_detour", dest=r["destination"]["name"], ids=", ".join(r["direct_blocked_by"]), km=fmt(r["detour_km"])))
        self._remember(zones)
        return site.decision.state, " ".join(x for x in lines if x)

    def _do_avoid_areas(self) -> tuple[str, str]:
        o = self._origin()
        when = self.times[0]
        site = self.site_eval(o[0], o[1], when, "departure")
        grid = self.weather.grid(when)
        self.extra["grid"] = grid
        self.make_site_card(site, self.L("card.title_site", place=self.place_name(), when=when_label(when)))
        self._map_common(when, [], site)
        lines = [self._header(site), self.L("a.avoid_head", place=self.place_name())]
        items = []
        for f in self.cons.binding:
            from .geo import distance_to_polygon_km, bearing_deg, compass
            d = distance_to_polygon_km(o[0], o[1], f.polygon)
            cx, cy = sum(p[0] for p in f.polygon) / len(f.polygon), sum(p[1] for p in f.polygon) / len(f.polygon)
            items.append({"kind": "geofence", "id": f.fence_id, "name": f.name, "fence_kind": f.kind, "distance_km": round(d, 1), "direction": compass(bearing_deg(o[0], o[1], cx, cy))})
        for it in sorted(items, key=lambda i: i["distance_km"]):
            lines.append(self.L("a.avoid_fence", name=it["name"], kind=it["fence_kind"], dist=fmt(it["distance_km"]), dir=it["direction"]))
        if site.feed and site.feed.available:
            for a in site.feed.items:
                if datetime.fromisoformat(a.valid_from) <= when <= datetime.fromisoformat(a.valid_to):
                    items.append({"kind": "advisory", "id": a.advisory_id, "name": a.text})
                    lines.append(self.L("a.avoid_adv", text=a.text))
        th = self.cfg.thresholds
        hot = sorted((g for g in grid if g["level"] in ("MODERATE_CONDITIONS", "HIGHER_RISK") and g["wave_height"] is not None and g["wind_speed"] is not None),
                     key=lambda g: -max(g["wave_height"] / th.wave_higher_m, g["wind_speed"] / th.wind_higher_kmh))[:3]
        for g in hot:
            items.append({"kind": "hotspot", **g})
            lines.append(self.L("a.avoid_hot", lat=fmt(g["lat"]), lon=fmt(g["lon"]), wave=fmt(g["wave_height"]), wind=fmt(g["wind_speed"])))
        self.extra["avoid"] = items
        self.note(items)
        return ("RESTRICTED" if False else site.decision.state), " ".join(lines)

    def _do_compare(self) -> tuple[str, str]:
        if len(self.times) < 2:
            self.trace.add("validate", "compare_times", status="refused", detail="need two times")
            return "INSUFFICIENT_DATA", self.L("a.compare_needs")
        o = self._origin()
        rows, sites = [], []
        for w in self.times[:3]:
            s = self.site_eval(o[0], o[1], w, "departure")
            sites.append(s)
            zones = self.eval_zones(w, o)
            best = next((z for z in zones if z["recommended"]), None)
            self.make_site_card(s, self.L("card.title_site", place=self.place_name(), when=when_label(w)), best)
            f = s.fusion
            haz = []
            if s.active:
                haz += [a.text for a in s.active]
            lt = f.observation("lightning")
            if lt and lt.value >= 1:
                haz.append(self.L("a.lightning_yes"))
            rn = f.observation("precipitation")
            rows.append({"time": w.isoformat(timespec="minutes"), "label": when_label(w),
                         "wind": None if not f.observation("wind_speed") else {"value": f.observation("wind_speed").value, "unit": "km/h", "obs_id": f.observation("wind_speed").obs_id},
                         "wave": None if not f.observation("wave_height") else {"value": f.observation("wave_height").value, "unit": "m", "obs_id": f.observation("wave_height").obs_id},
                         "precipitation": None if not rn else {"value": rn.value, "unit": "mm/h"},
                         "hazards": haz, "conflicts": [c.variable for c in f.conflicts],
                         "pfz": None if not best else {"zone": best["zone"]["name"], "support": best["support"]["met"], "state": best["state"]},
                         "pfz_note": self.extra.get("pfz_note") if not zones else None,
                         "state": s.decision.state, "reasons": [r.to_dict() for r in s.decision.reasons if r.outcome != "LOWER_RISK"]})
            self.extra["zones"] = [self.clean_zone(z) for z in zones]
            self.pfz_res_last = zones
        self.extra["comparison"] = rows
        self.note(rows)
        self._map_common(self.times[0], [], sites[0])
        lines = [self.L("a.compare_head", place=self.place_name())]
        for r, s in zip(rows, sites):
            lines.append(self.L("a.timeline", when=r["label"], state=self.st(r["state"])) + ".")
            lines += [x for x in self._detail_lines(s)[:2]]
            if r["pfz"]:
                lines.append(self.L("a.pfz_best", zone=r["pfz"]["zone"], dist=fmt(next(z["relation"]["distance_km"] for z in self.extra["zones"] if z["zone"]["name"] == r["pfz"]["zone"])),
                                    dir=next(z["relation"]["direction"] for z in self.extra["zones"] if z["zone"]["name"] == r["pfz"]["zone"]), n=r["pfz"]["support"], state=self.st(r["pfz"]["state"])))
        a, b = rows[0], rows[1]
        for later in rows[1:]:
            lines.append(self.L("a.state_change", when=later["label"], a=self.st(a["state"]), b=self.st(later["state"])) if later["state"] != a["state"]
                         else self.L("a.state_same", when=later["label"], a=self.st(a["state"])))
        for key, var in (("wave", "wave_height"), ("wind", "wind_speed")):
            if a[key] and b[key]:
                if abs(a[key]["value"] - b[key]["value"]) < 1e-9:
                    lines.append(self.L("a.no_change", var=self.lab(var), a=f"{fmt(a[key]['value'])} {a[key]['unit']}"))
                else:
                    lines.append(self.L("a.change", var=self.lab(var), a=f"{fmt(a[key]['value'])} {a[key]['unit']}", b=f"{fmt(b[key]['value'])} {a[key]['unit']}"))
        return worst(s.decision.state for s in sites), " ".join(x for x in lines if x)

    def _do_why(self) -> tuple[str, str]:
        mem = self.orca.sessions.get(self.sid, {}).get("analysis")
        if not mem or not mem.get("zones"):
            self.trace.add("validate", "why_context", status="withheld", detail="no earlier PFZ analysis in this session; running it first")
            if not self.u.get("place"):
                pl = self.orca.gazetteer()["ports"][0]
                self.u["place"] = {"id": pl["id"], "name": pl["en"], "lat": pl["lat"], "lon": pl["lon"], "sea_lat": pl["sea_lat"], "sea_lon": pl["sea_lon"], "matched": "default", "source": "default"}
            if not self.u["times"]:
                self.u["times"] = [(self.cfg.local_midnight() + timedelta(days=1, hours=6)).isoformat(timespec="seconds")]
            from .tools import build_menu
            self.u["intent"] = "pfz_zones"  # the fallback really is a PFZ analysis: plan its evidence and menu properly
            self.plan = plan_evidence(self.u)
            self.menu = build_menu(self.provider.capabilities(), self.plan["tools"])
            self.trace.add("plan", "evidence_planner(fallback)", output=self.plan)
            state, ans = self._do_pfz_zones()
            return state, f"{self.L('a.why_none')} {ans}"
        z = next((x for x in mem["zones"] if x["recommended"]), mem["zones"][0])
        self.note(mem["zones"], mem.get("pfz"), self.cfg.thresholds.to_dict())
        self.extra["zones"] = mem["zones"]
        self.extra["explained_query"] = mem.get("query_id")
        pfz = z["zone"]
        lines = [self.L("a.why_head", zone=pfz["name"]) if z["recommended"] else self.L("a.pfz_excluded", zone=pfz["name"], why=self.L(z["why_key"], **z["why_kw"]) if z["why_key"] else "")]
        lines.append(self.L("a.why_pfz", source=pfz["source_label"], issued=pfz["issued_at"], valid=pfz["valid_for"]))
        items = []
        for it in z["support"]["items"]:
            if it["met"] is None:
                items.append(self.L("a.why_criterion_missing", var=self.lab(it["variable"])))
            else:
                thr = f"{it['criterion']} ({fmt(it['threshold']) if isinstance(it['threshold'], (int, float)) else it['threshold']} {it['unit']})"
                items.append(self.L("a.why_criterion_met" if it["met"] else "a.why_criterion_not", var=self.lab(it["variable"]), val=fmt(it["value"]), unit=it["unit"], thr=thr))
        lines.append(self.L("a.why_support", n=z["support"]["met"], items="; ".join(items)))
        if not z["screen"]["restricted"]:
            lines.append(self.L("a.why_geo"))
        reasons = "; ".join(f"{self.lab(r['variable'])} {fmt(r['value']) if isinstance(r['value'], (int, float)) else r['value']}{(' ' + r['unit']) if r.get('unit') and isinstance(r['value'], (int, float)) else ''}" for r in z["decision"]["reasons"]
                            if r["variable"] in ("wave_height", "wind_speed"))
        lines.append(self.L("a.why_cond", state=self.st(z["conditions_state"]), reasons=reasons or "-"))
        lines.append(self.L("a.why_rank"))
        for r in z["decision"]["reasons"]:
            if r.get("obs_id") and r["obs_id"] in mem.get("evidence", {}):
                self.evidence[r["obs_id"]] = mem["evidence"][r["obs_id"]]
        self.trace.add("explain", "why_zone", output={"zone": pfz["zone_id"], "from_query": mem.get("query_id"), "rank": z.get("rank")})
        self.decisions.append(z["decision"])
        if mem.get("map"):
            self.map_layers = mem["map"]
        return z["state"], " ".join(x for x in lines if x)


    # -- blueprint workflows: nearest PFZ, closer, route check, ocean regions, scenario ------ #

    def _ctx(self) -> dict[str, Any]:
        return self.orca.sessions.get(self.sid, {}).get("ctx") or {}

    def _do_nearest_pfz(self) -> tuple[str, str]:
        o = self._origin()
        when = self.times[0]
        zones = self.eval_zones(when, o)
        site = self.site_eval(o[0], o[1], when, "departure")
        if not zones:
            self._map_common(when, [], site)
            return "INSUFFICIENT_DATA", f"{self.L('a.pfz_none')} {self.extra.get('pfz_note', '')}".strip()
        by_dist = sorted(zones, key=lambda z: z["relation"]["distance_km"])
        legal = [z for z in by_dist if not z["screen"]["restricted"]]
        lines = []
        if not legal:
            self._map_common(when, zones, site)
            self.extra["zones"] = [self.clean_zone(z) for z in zones]
            self._remember(zones)
            return "RESTRICTED", " ".join([self.L("a.pfz_head", date=when.date().isoformat(), source=self.pfz_res.source_label)] +
                                           [self._zone_line(z) for z in by_dist])
        nz = legal[0]
        pf = nz["zone"]
        if by_dist[0] is not nz:  # a nearer zone exists but is a hard-constraint violation: say so, never silently skip
            r = by_dist[0]
            nm = (r["screen"]["status"]["inside"] or r["screen"]["overlaps"] or [{"name": "land"}])[0]["name"]
            lines.append(self.L("a.nearest_restricted", zone=r["zone"]["name"].split(" (")[0], name=nm, dist=fmt(r["relation"]["distance_km"]), dir=r["relation"]["direction"]))
        lines.append(self.L("a.nearest_head", place=self.place_name(), zone=pf["name"], dist=fmt(nz["relation"]["distance_km"]), dir=nz["relation"]["direction"]))
        lines.append(self.L("a.nearest_src", source=pf["source_label"], issued=pf["issued_at"], valid=pf["valid_for"]))
        radius = pf["radius_km"] + self.cfg.hazard_buffer_km
        hz = self.geo.hazards_near(pf["latitude"], pf["longitude"], radius, when) if "hazards_near" in self.menu["available"] else None
        self.extra["nearby_hazards"] = hz
        self.extra["buffer"] = {"lat": pf["latitude"], "lon": pf["longitude"], "radius_km": radius}
        self.note(hz, {"buffer_km": radius})
        if hz is not None:
            wl = when_label(when)
            if hz["any"]:
                lines.append(self.L("a.hz_head", buf=fmt(radius), zone=pf["name"], when=wl))
                for a in hz["official_advisories"]:
                    lines.append(self.L("a.hz_adv", text=a["text"]))
                for f in hz["geofences"]:
                    if f["hard"]:
                        lines.append(self.L("a.hz_fence", name=f["name"], dist=fmt(f["distance_km"]), zone=pf["name"]))
                for c in hz["computed_breaches"][:3]:
                    lines.append(self.L("a.hz_grid", label=self.lab(c["variable"]), val=fmt(c["value"]), unit=c["unit"], where=f"{c['point']} ({fmt(c['lat'])}, {fmt(c['lon'])})",
                                        level=self.st(c["level"]), thr=fmt(c["threshold"])))
            else:
                lines.append(self.L("a.hz_none", buf=fmt(radius), zone=pf["name"], when=wl))
        lines.append(self._zone_line(nz))
        others = [z for z in by_dist if z is not nz]
        if others:
            lines.append(self.L("a.nearest_next", others=", ".join(f"{z['zone']['name'].split(' (')[0]} {fmt(z['relation']['distance_km'])} km {z['relation']['direction']}" for z in others)))
        self.make_site_card(nz["_site"], f"{pf['name']} - {when_label(when)}", nz, kind="zone")
        self._map_common(when, zones, site)
        self.extra["zones"] = [self.clean_zone(z) for z in zones]
        self.extra["focus"] = {"id": pf["zone_id"], "name": pf["name"], "lat": pf["latitude"], "lon": pf["longitude"]}
        self._remember(zones)
        state = nz["state"]
        if hz and any(a["severity"] in ("warning", "nogo", "ban") for a in hz["official_advisories"]):
            state = worst([state, "HIGHER_RISK"])
        return state, " ".join(x for x in lines if x)

    def _do_closer(self) -> tuple[str, str]:
        ctx = self._ctx()
        zs = ctx.get("zones") or []
        if not zs:
            self.trace.add("validate", "closer_context", status="refused", detail="no PFZ list earlier in this conversation")
            return "INSUFFICIENT_DATA", self.L("a.closer_none")
        zs = sorted(zs, key=lambda z: z["distance_km"])
        self.note(zs)
        self.extra["zones_summary"] = zs
        origin = ctx.get("origin_name") or ""
        legal = [z for z in zs if not z["restricted"]]
        lines = [self.L("a.closer_head", place=origin)]
        lines += [self.L("a.closer_line_r" if z["restricted"] else "a.closer_line", zone=z["name"].split(" (")[0], dist=fmt(z["distance_km"]), dir=z["direction"]) + "." for z in zs]
        if legal:
            lines.append(self.L("a.closer_win", zone=legal[0]["name"].split(" (")[0]))
        self.trace.add("rank", "distance_comparison", output={"order": [(z["id"], z["distance_km"]) for z in zs]}, detail="deterministic haversine distances from the stored PFZ list")
        if ctx.get("map"):
            self.map_layers = ctx["map"]
        return (legal[0]["state"] if legal else "RESTRICTED"), " ".join(lines)

    def _route_candidates(self, o: tuple[float, float], when: datetime) -> tuple[list[dict[str, Any]], str]:
        """Which route(s) to check: supplied waypoints > routes shown earlier > straight line to a PFZ (stated)."""
        wp = self.u.get("route_waypoints")
        if wp:
            return [{"role": "supplied", "label": "The supplied route", "path": [tuple(p) for p in wp], "destination": {"name": "supplied destination"}}], ""
        prev = self._ctx().get("route")
        if prev:
            return [{**r, "path": [tuple(p) for p in r["path"]]} for r in prev], ""
        focus = self._ctx().get("focus")
        if focus:
            tgt = (focus["lat"], focus["lon"], focus["name"])
        else:
            zones = self.eval_zones(when, o)
            best = next((z for z in zones if z["recommended"]), None) or (zones[0] if zones else None)
            if not best:
                return [], ""
            tgt = (best["zone"]["latitude"], best["zone"]["longitude"], best["zone"]["name"])
        return [{"role": "straight_line", "label": "The straight-line route", "path": [o, (tgt[0], tgt[1])], "destination": {"name": tgt[2], "lat": tgt[0], "lon": tgt[1]}}], \
            self.L("a.rc_assumed", a=self.place_name(), b=tgt[2])

    def _do_route_check(self) -> tuple[str, str]:
        o = self._origin()
        when = self.times[0]
        site = self.site_eval(o[0], o[1], when, "departure")
        cands, assumption = self._route_candidates(o, when)
        if not cands:
            self._map_common(when, [], site)
            return "INSUFFICIENT_DATA", self.L("a.rc_none")
        lines, results, worst_state = ([assumption] if assumption else []), [], site.decision.state
        shown = []
        for c in cands:
            res = self.geo.check_route(c["path"], when)
            res = {**res, "role": c["role"], "label": c["label"], "destination": c["destination"]}
            results.append(res)
            shown.append({"objective": c["role"], "role": c["role"], "feasible": True, "path": [list(p) for p in c["path"]], "destination": c["destination"],
                          "distance_km": res.get("distance_km"), "criteria": "route as supplied/shown earlier (checked, not optimised)",
                          "direct_blocked_by": [x["id"] for x in res.get("hard_crossings", [])], "max_exposure": None, "identical_to": None})
            self.trace.add("constraint", f"route_check:{c['role']}", status="ok" if res.get("ok") else "withheld",
                           output={k: res.get(k) for k in ("n_legs", "n_fences", "hard_crossings", "land_legs", "advisories", "ok")})
            lab = c["label"]
            if res.get("hard_crossings"):
                for x in res["hard_crossings"]:
                    lines.append(self.L("a.rc_cross", label=lab, name=x["name"], kind=x["kind"]))
                worst_state = "RESTRICTED"
            if res.get("land_legs"):
                lines.append(self.L("a.rc_land", label=lab)); worst_state = "RESTRICTED"
            for a in res.get("advisories", []):
                lines.append(self.L("a.rc_adv", label=lab, text=a["text"]))
                if a["severity"] in ("warning", "nogo", "ban"):
                    worst_state = worst([worst_state, "HIGHER_RISK"])
                else:
                    worst_state = worst([worst_state, "MODERATE_CONDITIONS"])
            for x in res.get("soft_crossings", []):
                lines.append(self.L("a.rc_soft", label=lab, name=x["name"], kind=x["kind"]))
            if res.get("ok"):
                lines.append(self.L("a.rc_ok", label=lab, legs=res["n_legs"], n=res["n_fences"]))
        self.extra["route_check"] = results
        self.extra["routes"] = {"routes": shown, "note": "Routes were checked as given, not re-planned."}
        self.note(results)
        self.make_site_card(site, self.L("card.title_site", place=self.place_name(), when=when_label(when)))
        self._map_common(when, [], site)
        return worst_state, " ".join(x for x in lines if x)

    def _do_ocean_regions(self) -> tuple[str, str]:
        o = self._origin()
        when = self.times[0]
        th = self.cfg.thresholds
        site = self.site_eval(o[0], o[1], when, "departure")
        regions = []
        for p in self.provider.ocean_points():
            d = haversine_km(o[0], o[1], p["lat"], p["lon"])
            if d > self.cfg.pfz_radius_km:
                continue
            fr = fuse(self.ocean.ocean_conditions(p["lat"], p["lon"], when), self.cfg, wanted=["sst", "chlorophyll", "sst_gradient"])
            sup = self.ocean.pfz_support(fr)
            items = {i["variable"]: i for i in sup["items"]}
            chl, sst, grad = items["chlorophyll"], items["sst"], items["sst_gradient"]
            known = chl["met"] is not None and sst["met"] is not None
            meets = bool(chl["met"] and sst["met"])
            fence = self.cons.status_at(p["lat"], p["lon"])
            rel = self.geo.relation(o, p["lat"], p["lon"])
            regions.append({"id": p["id"], "name": f"Ocean region {p['id']}", "lat": p["lat"], "lon": p["lon"], **{"distance_km": rel["distance_km"], "direction": rel["direction"]},
                            "chlorophyll": chl["value"], "sst": sst["value"], "sst_gradient": grad["value"], "chl_ok": chl["met"], "sst_ok": sst["met"],
                            "front_ok": grad["met"], "meets": meets and not fence["inside"], "known": known, "restricted": bool(fence["inside"]),
                            "obs": [chl["obs_id"], sst["obs_id"], grad["obs_id"]], "sources": sorted({i["source"] for i in sup["items"] if i["source"]})})
        for r in regions:
            for k in r["obs"]:
                pass
        regions.sort(key=lambda r: (not r["meets"], -(r["chlorophyll"] or 0), r["distance_km"]))
        self.extra["regions"] = regions
        self.note(regions, {"criteria": {"chl": th.chl_min_mg_m3, "lo": th.sst_min_c, "hi": th.sst_max_c}})
        self.trace.add("rank", "ocean_region_filter", output={"criteria": {"chlorophyll_min_mg_m3": th.chl_min_mg_m3, "sst_band_c": [th.sst_min_c, th.sst_max_c], "front_min_c_per_10km": th.sst_gradient_min_c_per_10km},
                                                              "result": [(r["id"], r["meets"]) for r in regions]}, detail="explicit implemented definition of 'favourable'")
        lines = [self.L("a.reg_def", chl=fmt(th.chl_min_mg_m3), lo=fmt(th.sst_min_c), hi=fmt(th.sst_max_c), when=when_label(when))]
        good = [r for r in regions if r["meets"]]
        if good:
            lines.append(self.L("a.reg_head", place=self.place_name()))
            for r in good:
                lines.append(self.L("a.reg_line", name=r["name"], dist=fmt(r["distance_km"]), dir=r["direction"], chl=fmt(r["chlorophyll"]), sst=fmt(r["sst"]),
                                    grad=fmt(r["sst_gradient"]), front=self.L("a.reg_front") if r["front_ok"] else ""))
        else:
            lines.append(self.L("a.reg_none"))
        for r in regions:
            if r["meets"]:
                continue
            if r["restricted"]:
                lines.append(self.L("a.pfz_excluded", zone=r["name"], why=self.L("why.restricted", name="hard geofence")))
            elif not r["known"]:
                lines.append(self.L("a.reg_fail", name=r["name"], why=self.L("a.reg_why_missing")))
            else:
                why = self.L("a.reg_why_chl", chl=fmt(r["chlorophyll"]), thr=fmt(th.chl_min_mg_m3)) if not r["chl_ok"] else self.L("a.reg_why_sst", sst=fmt(r["sst"]))
                lines.append(self.L("a.reg_fail", name=r["name"], why=why))
        self.make_site_card(site, self.L("card.title_site", place=self.place_name(), when=when_label(when)))
        self._map_common(when, [], site)
        return site.decision.state, " ".join(lines)

    def _do_scenario(self) -> tuple[str, str]:
        ctx = self._ctx()
        new_times = list(self.times)
        base = ctx.get("times")
        if base:
            baseline = datetime.fromisoformat(base[0]).astimezone(IST)
        elif self.u.get("place") and self.u["place"].get("source") == "query":
            d = new_times[0].replace(hour=6, minute=0, second=0, microsecond=0)
            baseline = d
            self.warnings.append("No earlier question in this conversation: the baseline was assumed to be 06:00 on the same day.")
        else:
            self.trace.add("validate", "scenario_context", status="refused", detail="no baseline in the conversation")
            return "INSUFFICIENT_DATA", self.L("a.scn_need")
        if not self.u.get("day_given", True):  # "in the evening" means the same day as the baseline
            new_times = [baseline.replace(hour=t.hour, minute=t.minute) for t in new_times]
        new_times = [t for t in new_times if t != baseline][:2] or new_times[:1]
        self.u["times"] = [baseline.isoformat(timespec="seconds")] + [t.isoformat(timespec="seconds") for t in new_times]
        self.trace.add("scenario", "modify_time_only", output={"modified": {"time": {"from": baseline.isoformat(timespec="minutes"), "to": [t.isoformat(timespec="minutes") for t in new_times]}},
                                                              "unchanged": {"place": self.u["place"]["name"], "intent": ctx.get("intent") or "go_no_go"}},
                       detail="only the requested time is modified; every dependent input is re-retrieved and every rule re-run")
        win = self.provider.weather_window()
        if win and any(not (win[0] <= t <= win[1]) for t in [baseline, *new_times]):
            bad = next(t for t in [baseline, *new_times] if not (win[0] <= t <= win[1]))
            return "INSUFFICIENT_DATA", self.L("a.outside_window", when=when_label(bad), start=when_label(win[0]), end=when_label(win[1]))
        state, ans = self._do_compare()
        return state, f"{self.L('a.scn_head', a=when_label(baseline), b=', '.join(when_label(t) for t in new_times))} {ans}"

    # -- map, memory, finish ------------------------------------------------- #

    def _remember(self, zones: list[dict[str, Any]]) -> None:
        sess = self.orca.sessions.setdefault(self.sid, {})
        sess["place"] = self.u["place"]["name"] if self.u.get("place") else sess.get("place")
        if zones:
            sess["analysis"] = {"zones": [self.clean_zone(z) for z in zones], "pfz": self.pfz_res.to_dict() if self.pfz_res else None,
                                "query_id": self.qid, "map": None, "evidence": {}}
            self._analysis_ref = sess["analysis"]

    def _map_common(self, when: datetime, zones: list[dict[str, Any]], site: Site) -> None:
        th = self.cfg.thresholds
        grid = self.extra.get("grid")
        if grid is None:
            grid = self.weather.grid(when)
            self.extra["grid"] = grid
        adv = []
        if site.feed and site.feed.available:
            for a in site.feed.items:
                adv.append({"id": a.advisory_id, "text": a.text, "severity": a.severity, "polygon": [list(p) for p in a.polygon],
                            "active": datetime.fromisoformat(a.valid_from) <= when <= datetime.fromisoformat(a.valid_to),
                            "valid_from": a.valid_from, "valid_to": a.valid_to})
        lats = [site.lat] + [z["zone"]["latitude"] for z in zones] + [g["lat"] for g in grid]
        lons = [site.lon] + [z["zone"]["longitude"] for z in zones] + [g["lon"] for g in grid]
        self.map_layers = {
            "time": when.isoformat(timespec="minutes"), "mode_label": self.provider.label(),
            "bounds": [[min(lats) - 0.25, min(lons) - 0.25], [max(lats) + 0.25, max(lons) + 0.25]],
            "land": [list(p) for p in self.provider.coast()],
            "geofences": self.cons.to_geojson(), "geofence_note": self.provider.geofence_origin(),
            "binding": [f.fence_id for f in self.cons.binding],
            "advisories": adv,
            "departure": {"name": self.u["place"]["name"], "lat": site.lat, "lon": site.lon, "state": site.decision.state} if self.u.get("place") else None,
            "zones": [{"id": z["zone"]["zone_id"], "name": z["zone"]["name"], "lat": z["zone"]["latitude"], "lon": z["zone"]["longitude"],
                       "radius_km": z["zone"]["radius_km"], "state": z["state"], "recommended": z["recommended"], "support": z["support"]["met"],
                       "restricted": z["screen"]["restricted"], "rank": z.get("rank")} for z in zones],
            "weather_points": [{k: g[k] for k in ("id", "lat", "lon", "wave_height", "wind_speed", "level", "conflict")} for g in grid],
            "routes": [], "thresholds": {"wave_moderate_m": th.wave_moderate_m, "wave_higher_m": th.wave_higher_m,
                                         "wind_moderate_kmh": th.wind_moderate_kmh, "wind_higher_kmh": th.wind_higher_kmh},
        }
        self.map_layers["buffer"] = self.extra.get("buffer")
        self.map_layers["regions"] = [{k: r[k] for k in ("id", "lat", "lon", "meets", "chlorophyll", "sst", "restricted")} for r in self.extra.get("regions", [])]
        rs = self.extra.get("routes")
        if rs:
            self.map_layers["routes"] = [{"objective": r["objective"], "role": r["role"], "identical_to": r.get("identical_to"), "path": r["path"], "destination": r["destination"], "distance_km": r["distance_km"],
                                          "max_exposure": r.get("max_exposure"), "criteria": r["criteria"], "direct_blocked_by": r["direct_blocked_by"]}
                                         for r in rs.get("routes", []) if r.get("feasible")]

    def _why(self, state: str, answer: str, verification: dict[str, Any]) -> list[dict[str, Any]]:
        u, tr = self.u, self.trace
        calls = [{"agent": e.name.split(".")[0], "tool": e.name.split(".", 1)[1] if "." in e.name else e.name, "status": e.status,
                  "params": dict(e.params), "summary": e.output.get("n_observations", e.output.get("n_points", None)), "note": e.detail}
                 for e in tr.of_step("agent")]
        sel = []
        seen: set[str] = set()
        for d in self.decisions[:6]:
            for r in d["reasons"]:
                oid = r.get("obs_id")
                if oid and oid not in seen and oid in self.evidence:
                    seen.add(oid)
                    sel.append(self.evidence[oid])
        conflicts = [e.output for e in tr.of_step("fuse")]
        primary = max(self.decisions, key=lambda d: _RANK[d["state"]]) if self.decisions else None
        return [
            {"stage": "User Question", "content": {"text": u["text"], "language": u["language"]}},
            {"stage": "Detected Intent", "content": {"intent": u["intent"], "matched_terms": u["matched_terms"], "place": u.get("place"),
                                                     "times": u["times"], "time_label": u["time_label"]}},
            {"stage": "Required Evidence", "content": self.plan["required"]},
            {"stage": "Agents / Tools Called", "content": {"calls": calls, "withheld": self.menu["withheld"]}},
            {"stage": "Raw / Normalized Observations", "content": {"observations": sel, "source_conflicts": conflicts}},
            {"stage": "Rules and Constraints", "content": {"decision_scope": primary["scope"] if primary else None,
                                                            "reasons": primary["reasons"] if primary else [],
                                                            "thresholds": self.cfg.thresholds.to_dict(),
                                                            "hard_geofences": [f.fence_id for f in self.cons.binding],
                                                            "constraint_events": [e.to_dict() for e in tr.of_step("constraint")]}},
            {"stage": "Decision", "content": {"state": state, "flags": primary["flags"] if primary else [], "missing": primary["missing"] if primary else [],
                                              "stale": primary["stale"] if primary else [], "sensitivity": primary["sensitivity"] if primary else []}},
            {"stage": "Explanation", "content": {"answer": answer, "verification": verification}},
        ]

    ROUTE_LABELS = {"shortest": "The shortest route", "lower_exposure": "The lower-exposure route", "pfz_oriented": "The PFZ-oriented route",
                    "alternative": "The alternative-destination route", "supplied": "The supplied route", "straight_line": "The straight-line route"}

    def _update_ctx(self, status: str) -> None:
        """Conversation memory (blueprint section 21): topic, place, time, focus, last PFZ list, last routes."""
        if status != "ok":
            return
        from .agents.planner import RERUNNABLE
        u, it = self.u, self.u["intent"]
        ctx = self.orca.sessions.setdefault(self.sid, {}).setdefault("ctx", {})
        if u.get("place") and it not in ("closer", "why"):
            if u["place"].get("source") != "focus":  # "there" must not replace the home place
                ctx["place"] = {k: u["place"][k] for k in ("id", "name", "lat", "lon", "sea_lat", "sea_lon")}
                ctx["origin_name"] = self.place_name()
        if it in RERUNNABLE:
            ctx["intent"] = it
        if it == "scenario" and len(u["times"]) > 1:
            ctx["times"] = [u["times"][1]]
        elif it not in ("closer", "why") and u.get("times"):
            ctx["times"] = list(u["times"])
        zs = self.extra.get("zones")
        if zs and it not in ("why",):
            ctx["zones"] = [{"id": z["zone"]["zone_id"], "name": z["zone"]["name"], "lat": z["zone"]["latitude"], "lon": z["zone"]["longitude"],
                             "distance_km": z["relation"]["distance_km"], "direction": z["relation"]["direction"], "state": z["state"],
                             "recommended": z["recommended"], "restricted": z["screen"]["restricted"], "support": z["support"]["met"]} for z in zs]
            best = next((z for z in zs if z["recommended"]), None)
            if self.extra.get("focus"):
                ctx["focus"] = self.extra["focus"]
            elif best and it in ("pfz_zones", "go_no_go", "route"):
                ctx["focus"] = {"id": best["zone"]["zone_id"], "name": best["zone"]["name"], "lat": best["zone"]["latitude"], "lon": best["zone"]["longitude"]}
        rs = self.extra.get("routes")
        if rs and it in ("route", "route_check"):
            ctx["route"] = [{"role": r["role"], "label": self.ROUTE_LABELS.get(r["role"], r["role"]), "path": r["path"], "destination": r["destination"]}
                            for r in rs.get("routes", []) if r.get("feasible") and not r.get("identical_to")]
        if self.map_layers:
            ctx["map"] = self.map_layers

    def _finish(self, state: str, answer: str, *, status: str = "ok") -> dict[str, Any]:
        tr = self.trace
        template = answer
        ledger = build_ledger(*self.ledger_structs, {"thresholds": self.cfg.thresholds.to_dict()}, self.evidence)
        ver = verify_text(template, ledger)
        tr.add("verify", "answer_claims", status="ok" if ver.ok else "error", output=ver.to_dict(),
               detail="every number-with-unit in the answer is present in the structured evidence" if ver.ok else "UNSUPPORTED CLAIM IN TEMPLATE ANSWER")
        if not ver.ok:
            answer = f"{self.st(state)}. Details are in the evidence card; the automatic summary was withheld because it failed verification."
        narration = {"used": False, "narrator": getattr(self.orca.narrator, "name", None), "rejected_reason": None}
        if self.orca.narrator and ver.ok and status == "ok":
            try:
                draft = self.orca.narrator.narrate(self.u["text"], self.lang, {"decisions": self.decisions[:3]}, {"state": state}, template)
                ok, why, v2 = gate_narration(draft, state, ledger)
                tr.add("verify", "narration_gate", status="ok" if ok else "withheld", output=v2.to_dict(), detail=why or "accepted")
                if ok:
                    answer, narration["used"] = draft, True
                else:
                    narration["rejected_reason"] = why
            except Exception as exc:  # network/quota/anything: the template answer stands
                narration["rejected_reason"] = f"{type(exc).__name__}: {exc}"
                tr.add("verify", "narration_gate", status="error", detail=narration["rejected_reason"])
        records = build_records(self)
        linkage = link_answer(answer, records)
        tr.add("verify", "evidence_linkage", status="ok" if not linkage["unlinked"] else "withheld",
               output={"n_records": len(records), "factual_sentences": linkage["n_factual"], "linked": linkage["n_linked"], "unlinked": linkage["unlinked"]},
               detail="every factual sentence is tied to at least one evidence record" if not linkage["unlinked"] else "SOME FACTUAL SENTENCES HAVE NO EVIDENCE RECORD")
        alerts = build_alerts(self, self.lang)
        disclaimer = self.L("a.disclaimer", mode=self.provider.label())
        primary_card = max(self.cards, key=lambda c: _RANK[c["recommendation"]["state"]]) if self.cards else None
        why = self._why(state, answer, ver.to_dict())
        tr.add("compose", "final", output={"state": state, "status": status, "n_cards": len(self.cards)})
        result = {
            "query_id": self.qid, "session_id": self.sid, "query": self.u["text"], "status": status,
            "language": self.lang, "intent": self.u["intent"], "place": self.u.get("place"), "times": self.u["times"],
            "mode": self.cfg.mode, "mode_label": self.provider.label(), "state": state, "state_label": self.st(state),
            "answer": answer, "answer_template": template, "disclaimer": disclaimer, "narration": narration,
            "verification": ver.to_dict(), "warnings": self.warnings,
            "primary_card": primary_card, "cards": self.cards, "decisions": self.decisions,
            "zones": self.extra.get("zones", []), "routes": self.extra.get("routes"), "comparison": self.extra.get("comparison"),
            "avoid": self.extra.get("avoid"), "hazards": self.extra.get("hazards"), "nearby_hazards": self.extra.get("nearby_hazards"),
            "route_check": self.extra.get("route_check"), "regions": self.extra.get("regions"), "zones_summary": self.extra.get("zones_summary"),
            "interpretation": {"intent": self.u["intent"], "matched_terms": self.u["matched_terms"], "followup": self.u.get("followup", False),
                               "place": self.u["place"]["name"] if self.u.get("place") else None, "place_source": self.u["place"].get("source") if self.u.get("place") else None,
                               "time_label": self.u["time_label"]},
            "evidence_records": records, "evidence_linkage": linkage, "alerts": alerts,
            "map": self.map_layers or None, "why": why, "evidence_count": len(self.evidence),
            "trace": tr.to_dict(),
        }
        self._update_ctx(status)
        self.orca.store.put(self.qid, {**result, "evidence": self.evidence})
        ref = getattr(self, "_analysis_ref", None)
        if ref is not None:
            ref["map"] = self.map_layers or None
            ref["evidence"] = {k: v for k, v in self.evidence.items() if k in {r.get("obs_id") for z in ref["zones"] for r in z["decision"]["reasons"]}}
        return result
