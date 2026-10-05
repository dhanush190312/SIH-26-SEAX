# SEAX blueprint ↔ this prototype: compliance matrix

Source: `GEOSAR_SEAX_Complete_Project_Blueprint-1.md`. Status key: **Done** (implemented and tested) · **Partial** (works, with the stated gap) · **Not done** (not implemented — listed so nobody assumes it is).
The code package is still named `orca` (the SIH26176 problem-statement engine); SEAX is the product name in the UI and docs.

## A. Deliberate deviations from the blueprint (read these first)

| Blueprint says | What was built | Why / consequence |
|---|---|---|
| §4/§8: the LLM does intent detection, planning, tool selection | **Rules do all three** (`agents/planner.py`). The LLM is optional and only *rephrases* a verified answer (`ORCA_LLM=mistral`, untested live) | Works with no key, no network, no GPU — judge-proof. Cost: the planner is brittle on unusual phrasing (see §E: measured 3/7 on held-out paraphrases). |
| §24: Next.js, PostgreSQL + PostGIS, Pandas, Shapely | **None of them.** Python stdlib server, one HTML file, `orca/geo.py` for geometry, JSON files for data | The blueprint's own rule: "use only technologies actually implemented". Not installable offline here; at this data scale not needed. A PostGIS/Shapely swap would replace `orca/geo.py` + `decision/geofence.py` only. |
| §24: interactive map | Hand-drawn **SVG map**, layer toggles + tooltips, **no pan/zoom, no tiles** | Works with the network unplugged. A Leaflet front-end could consume the same `result.map` JSON. |
| §8: separate Weather / Ocean / PFZ / Hazard agents | Three agents: Weather&Hazard, Ocean, Geo/PFZ/Route. PFZ and hazard-proximity live in the Geo agent (`hazards_near`) | Blueprint rule 6 and the problem statement both forbid agent sprawl. |
| §25: repository layout | `orca/{agents,tools,decision,data,api,eval}` | Same responsibilities, different folder names. |

## B. Section by section

| § | Blueprint requirement | Status | Where / note |
|---|---|---|---|
| 7 | Chat + map + alerts outputs | **Done** | chat, SVG map, alert strip (critical/warning/info, each badged OFFICIAL or SEAX-COMPUTED) |
| 8 | Conversation/intent agent: intent, location, date/time, constraints, context | **Done** (rule-based) | `planner.understand`; English + Tamil |
| 8 | Mission planner: decompose, select only required tools, order | **Done** | `plan_evidence`; capability validator withholds tools a mode can't serve, with reason |
| 8 | Weather / Ocean / PFZ / Hazard retrieval returning structured data | **Done** (synthetic data; LIVE = Open-Meteo only, unverified) | `tools/registry.py`; tools never return prose |
| 8 | GIS: distance, buffers, polygon intersection, geofencing, route/restricted checks | **Done** | `geo.py`, `decision/geofence.py`, `decision/hazards.py`; property-tested |
| 8 | Fusion + quality: timestamp alignment, units, missing values, **duplicate checking**, metadata, stale rejection | **Partial** | alignment = nearest hour with `time_offset_min` shown; missing/stale/metadata done. **Duplicate-record detection is not implemented; there is no unit conversion** (units are fixed by the schema) |
| 8 | Safety engine: explicit configurable rules, official advisories not replaced | **Done** | `decision/risk.py`; thresholds in `Thresholds`; official warning/ban overrides |
| 8 | Route engine: candidates, remove restricted, check hazards, metrics | **Partial** | restricted = excluded by construction + post-verified; **advisory areas are checked by `route_check` and reported by exposure, not used to exclude candidates** |
| 9 | 9-step execution (parse → plan → retrieve → validate → compute → fuse → evidence → explain → display) | **Done** | one trace entry per hop (`trace` tab) |
| 11 | Source classes; "live" only after testing the exact source | **Done** | nothing is called live unless fetched; LIVE failure ⇒ unavailable, never DEMO fill. **No real EO/ISRO/INCOIS/IMD source is integrated** |
| 12 | Common data model (source, dataset, variable, value, unit, geometry, observation_time, valid_from/to, retrieved_at, quality_status) | **Done** | `schema.Observation` (+`dataset`, `valid_from`, `valid_to`, `quality_status`, `geometry`) |
| 13 | Nearest PFZ; geofence; hazard proximity — all deterministic | **Done** | `nearest_pfz` (haversine), `screen_zone`, `hazards_near` (buffer = PFZ radius + `ORCA_HAZARD_BUFFER_KM`) |
| 14 | Temporal: observation/forecast-valid/issue/validity; never combine an expired warning | **Done** | three clocks kept apart; expired and not-yet-valid advisories are listed as *inactive with reason* and never counted (tested) |
| 15 | PFZ workflow incl. the exact "No valid PFZ information…" sentence; never invent a PFZ | **Done** | stale (>30 h) or wrong-date PFZ product ⇒ rejected, sentence shown |
| 16 | Marine-condition workflow | **Done** | weather + ocean + advisories + hazards + geofence, no safety guarantee |
| 17 | Ocean analysis with an explicit "favourable" definition | **Done** | chlorophyll ≥ 0.5 mg/m³ and SST 26–31 °C (prototype defaults, printed in the answer, configurable). Only 4 synthetic ocean points; product is a composite, not a forecast |
| 18 | Hazard workflow; official ≠ SEAX interpretation | **Done** | evidence/alert classes `official` / `source` / `computed` |
| 19 | Route workflow, decision support not navigation | **Partial** | see §8 row above; disclaimer on every answer |
| 20 | Evidence objects; "no evidence → no factual claim" | **Done** (heuristic linker) | `evidence_pack.py`: records with claim/source/issued/valid_from/valid_to/operation/result; each factual sentence linked by value+unit or category; unlinked sentences are counted and fail the eval. The linker is heuristic string/number matching, not formal provenance |
| 21 | Conversational memory (location, topic, time, comparison) | **Done** (in-memory) | place, topic, time, "there" focus, last PFZ list, last routes. Lost on server restart; no cross-session profile |
| 22 | Scenario engine: change only the time, re-retrieve, re-run rules, compare, explain | **Done** | `scenario` intent; trace records what was modified/unchanged |
| 23 | Multilingual, tested languages only | **Done for English + Tamil** | other scripts detected and *reported as unsupported*. Tamil strings need native review. **Voice input: browser Web Speech API button — untested here, may need internet** |
| 26 | MVP 1–4 | **Done** | PFZ · conditions · hazard/geofence · evidence |
| 27 | Demos 1–4 | **Done** | "Nearest PFZ" · "Conditions there" · "Route check" · "Leave in the evening?" buttons |
| 28 | Unit / integration / data / conversation / safety tests | **Partial** | 136 tests. Present: distance, coordinates (API bounds), intersection, timestamps, rules, parsers (LIVE fixture), missing/stale, follow-ups, location/date changes, active/expired/not-yet-valid warning, route/geofence intersection, unavailable source. **Missing: duplicate-record test (feature absent), incorrect-unit test (no conversion), language-switch mid-conversation test** |
| 29 | Metrics, "don't claim until measured" | **Partial — measured, see §E** | `python -m orca eval`. Not measured: response clarity; spatial correctness is covered by unit/property tests, not the eval set |
| 31 | Risks & mitigations | **Done** | source adapters, staleness, verifier, deterministic GIS, conflict display, glossary-free Tamil (needs review) |

## C. §35 rules for the implementation team

| # | Rule | Status |
|---|---|---|
| 1 | Don't invent datasets/APIs | Followed: LIVE uses one documented public API; everything else is labelled synthetic or unavailable |
| 2 | Don't call sample data "live" | Followed (banner, labels, tests) |
| 3 | No ISRO integration claim | Followed: no ISRO source is integrated |
| 4 | No real-time claim unless real-time | Followed; LIVE not even verified |
| 5 | No foundation-model training | Followed |
| 6 | No agents for visual complexity | Followed (3 agents) |
| 7 | Numeric/GIS/safety deterministic | Followed (`decision/` imports no LLM; tested) |
| 8 | Store timestamps and source metadata | Followed |
| 9 | Provenance source→claim | Done at record level (§B-20 caveat) |
| 10 | Never fabricate missing data | Followed; `INSUFFICIENT DATA` everywhere, tested |
| 11 | Separate official from SEAX analysis | Done (badges, classes, tests) |
| 12 | Never promise guaranteed safe | Enforced by the verifier (assurance wording rejected) |
| 13 | No catch prediction | None implemented; "opportunity" means PFZ + ocean-support criteria only |
| 14 | Measure before claiming accuracy | See §E |
| 15 | Small end-to-end over large disconnected | Yes |
| 16 | Every answer explainable from evidence | Evidence panel + WHY chain + trace |

## D. Not done at all
PostgreSQL/PostGIS · Next.js/React front-end · Leaflet map with pan/zoom · real EO/raster processing (SST/chlorophyll are synthetic point values) · duplicate-record detection · unit conversion · LLM-driven planning · embeddings/vector retrieval · voice verified on a device · official INCOIS/IMD/MOSDAC integrations · more than two languages · persistent conversation memory · user accounts.

## E. Measured evaluation (`python -m orca eval`, fixed clock, synthetic data)

36 cases: **29 "core"** (written by the developer, so they act as a regression guard, *not* an accuracy estimate) and **7 "paraphrase"** (phrasings deliberately not tuned for).

| Metric | Core (29) | Paraphrase (7) |
|---|---|---|
| Intent accuracy | 26/26 scored | 3/7 |
| Tool-selection (required tools called, none outside the plan) | 22/22 | 4/4 scored |
| Location / time extraction | 21/21 · 14/14 | 6/6 · – |
| Result correctness (state/status/answer content) | 29/29 | 0/3 scored |
| Evidence completeness (factual sentences linked) | 203/203 | 32/32 |
| Groundedness (verifier passes) | 29/29 | 7/7 |
| Safety-rule cases | 5/5 | – |
| Multilingual (Tamil) | 2/2 | 0/1 |

Findings worth saying out loud:
* The evaluation **found real bugs while being built** (planner tool lists out of sync with handlers; displayed ocean values and refusals lacking evidence records; a `closer` follow-up demanding a place) — all fixed, then the core set was frozen.
* **Held-out paraphrases fail 4 of 7**: "rough seas", "is that path allowed…", a Tamil "how is the sea" phrasing, and "which fishing spot is closest" (this last one is *misread as a go/no-go question* — a wrong-but-plausible answer, not a refusal). Mitigation implemented: every answer states "Understood as: …"; nothing unsupported is ever said even when the intent is misread (groundedness 7/7). The real fix is an LLM-assisted or much larger-vocabulary planner — not done.
* Latency: median ≈ 23 ms, max ≈ 1.9 s (route search) in-process on the build machine, DEMO mode.
